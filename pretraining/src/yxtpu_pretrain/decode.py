"""Incremental (cached) decoding for the hybrid KDA/NoPE-GQA model.

The training forward recomputes the whole prefix for every position, which
is right for teacher forcing and catastrophic for generation: at window
4096 each token costs a full forward, roughly 4,096x the arithmetic the
token needs. This module keeps the state each mixer actually depends on:

* KDA is linear attention with a constant-size recurrence, so one
  ``[batch, heads, key_dim, value_dim]`` matrix per layer replaces the
  prefix entirely (the same step equation as ``recurrent_kda_reference``),
  plus a short ring of pre-convolution projections for the causal
  depthwise conv.
* NoPE-GQA keeps an ordinary key/value cache. There is no positional
  encoding to re-apply, so cached keys stay valid unchanged.
* Block-AttnRes needs no cache at all: every read attends across *depth*
  at a single token position, never across time.

Generation, sampling included, runs inside one jitted loop, so no logits
ever cross to the host mid-stream.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from flax import nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN

from yxtpu_pretrain.layers.kimi_delta_attention import l2norm
from yxtpu_pretrain.model import HybridLanguageModel


@dataclasses.dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 64
    top_p: float = 1.0
    repetition_penalty: float = 1.0


def split_cycles(model: HybridLanguageModel) -> list[nnx.Module]:
    """Materializes one concrete cycle module per scanned cycle.

    Training scans the cycles with their parameters stacked on
    ``param_scan_axis``; decoding walks 16 layers once per token, where the
    scan buys nothing and would only complicate threading per-layer caches.
    """
    graphdef, params, state = nnx.split(model.cycles, nnx.Param, ...)
    scan_axis = model.config.model.param_scan_axis
    if scan_axis != 0:
        params = jax.tree.map(lambda leaf: jnp.moveaxis(leaf, scan_axis, 0), params)
    cycles = []
    for index in range(model.config.model.num_cycles):
        cycle_params = jax.tree.map(lambda leaf, i=index: leaf[i], params)
        cycle_state = jax.tree.map(lambda leaf, i=index: leaf[i], state)
        cycles.append(nnx.merge(graphdef, cycle_params, cycle_state))
    return cycles


def init_cache(model: HybridLanguageModel, batch: int, max_length: int) -> dict:
    """Zeroed caches, one array per layer.

    Per-layer arrays rather than one stacked array per kind: a stacked
    ``cache[slot].set(...)`` makes every layer's update read and rewrite all
    layers' state, which measured as the dominant cost of a decode step
    (~300 MB of copies per token at batch 4). Separate arrays let each
    update touch only its own buffer.
    """
    model_config = model.config.model
    kda = model_config.kda
    attention = model_config.attention
    dtype = model.leaf_config.dtype
    kinds = model_config.cycle
    num_cycles = model_config.num_cycles
    kda_layers = sum(1 for kind in kinds if kind == "kda") * num_cycles
    gqa_layers = sum(1 for kind in kinds if kind == "gqa") * num_cycles
    projection = 3 * kda.num_heads * kda.key_head_dim
    return {
        "kda_state": [
            jnp.zeros(
                (batch, kda.num_heads, kda.key_head_dim, kda.value_head_dim),
                jnp.float32,
            )
            for _ in range(kda_layers)
        ],
        "kda_conv": [
            jnp.zeros((batch, kda.conv_kernel_size - 1, projection), dtype)
            for _ in range(kda_layers)
        ],
        "gqa_key": [
            jnp.zeros(
                (batch, max_length, attention.num_kv_heads, attention.head_dim),
                dtype,
            )
            for _ in range(gqa_layers)
        ],
        "gqa_value": [
            jnp.zeros(
                (batch, max_length, attention.num_kv_heads, attention.head_dim),
                dtype,
            )
            for _ in range(gqa_layers)
        ],
        "position": jnp.int32(0),
    }


def kda_step(mixer, hidden, recurrent_state, conv_cache):
    """One KDA token: conv ring, gated delta-rule update, gated output.

    Mirrors ``KimiDeltaAttention.__call__`` for a single position, with the
    chunked kernel replaced by the equivalent one-step recurrence.
    """
    config = mixer.config
    batch = hidden.shape[0]
    heads, head_dim = mixer.num_heads, mixer.head_dim
    qkv = mixer.in_proj_qkv(hidden)
    decay_hidden = mixer.decay_down(hidden)
    beta_logits = mixer.beta_proj(hidden)
    gate_hidden = mixer.output_gate_down(hidden)

    # The head-major projection already emits the convolution's
    # (head, qkv, dim) channel order.
    flat = qkv.reshape(batch, 1, -1)
    window = jnp.concatenate((conv_cache, flat), axis=1)
    kernel = jnp.asarray(mixer.conv1d.kernel[...], flat.dtype)[:, 0, :]
    convolved = jnp.sum(window * kernel[None], axis=1, keepdims=True)
    conv_cache = window[:, 1:]
    activated = jax.nn.silu(convolved.astype(jnp.float32)).astype(config.dtype)
    activated = activated.reshape(batch, 1, heads, 3, head_dim)
    query, key, value = (activated[..., i, :] for i in range(3))

    raw_decay = mixer.decay_up(decay_hidden).astype(jnp.float32) + jnp.asarray(
        mixer.dt_bias[...], jnp.float32
    )
    decay_rate = jnp.exp(jnp.asarray(mixer.A_log[...], jnp.float32))[None, None, :, None]
    if config.kda_safe_gate:
        log_decay = config.kda_gate_lower_bound * jax.nn.sigmoid(decay_rate * raw_decay)
    else:
        log_decay = -decay_rate * jax.nn.softplus(raw_decay)
    beta = jax.nn.sigmoid(beta_logits.astype(jnp.float32))

    if config.use_qk_norm_in_gdn:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query.astype(jnp.float32)[:, 0] * jax.lax.rsqrt(
        jnp.asarray(head_dim, jnp.float32)
    )
    key = key.astype(jnp.float32)[:, 0]
    value = value.astype(jnp.float32)[:, 0]
    gate = log_decay[:, 0]
    beta = beta[:, 0]

    recurrent_state = recurrent_state * jnp.exp(gate)[..., None]
    prediction = jnp.einsum(
        "bhk,bhkv->bhv", key, recurrent_state, precision=jax.lax.Precision.HIGHEST
    )
    recurrent_state = recurrent_state + jnp.einsum(
        "bhk,bhv->bhkv",
        beta[..., None] * key,
        value - prediction,
        precision=jax.lax.Precision.HIGHEST,
    )
    output = jnp.einsum(
        "bhk,bhkv->bhv", query, recurrent_state, precision=jax.lax.Precision.HIGHEST
    )

    output = output[:, None].astype(config.dtype)
    output_gate = mixer.output_gate_up(gate_hidden)
    output = mixer.output_norm(output) * jax.nn.sigmoid(
        output_gate.astype(jnp.float32)
    )
    return mixer.out_proj(output.astype(config.dtype)), recurrent_state, conv_cache


def gqa_step(mixer, hidden, key_cache, value_cache, position):
    """One NoPE-GQA token against the cache; no rotary to re-apply."""
    query, key, value = mixer._project(hidden)
    key_cache = jax.lax.dynamic_update_slice_in_dim(
        key_cache, key.astype(key_cache.dtype), position, axis=1
    )
    value_cache = jax.lax.dynamic_update_slice_in_dim(
        value_cache, value.astype(value_cache.dtype), position, axis=1
    )
    batch = hidden.shape[0]
    grouped = query.reshape(
        batch, 1, mixer.num_kv_heads, mixer.q_heads_per_kv, mixer.head_dim
    )[:, 0]
    logits = jnp.einsum(
        "bkhd,blkd->bkhl",
        grouped.astype(jnp.float32),
        key_cache.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    valid = jnp.arange(key_cache.shape[1]) <= position
    logits = jnp.where(valid, logits, jnp.float32(-1.0e30))
    probabilities = jax.nn.softmax(logits, axis=-1).astype(value_cache.dtype)
    attended = jnp.einsum(
        "bkhl,blkd->bkhd",
        probabilities,
        value_cache,
        precision=jax.lax.Precision.DEFAULT,
    )
    attended = attended.reshape(batch, 1, mixer.num_query_heads, mixer.head_dim)
    return mixer.out_proj(attended.astype(mixer.dtype)), key_cache, value_cache


def cycle_step(cycle, hidden_buffer, block_index, cache, cursors, position):
    """One cycle over a single token, threading each layer's cache.

    The buffer scores are hoisted exactly as ``HybridCycle.__call__`` does,
    so decoding reproduces the training arithmetic rather than an
    algebraically equal rearrangement of it.
    """
    layers = [getattr(cycle, f"layer_{i}") for i in range(cycle.cycle_length)]
    partial_sum = jnp.zeros_like(hidden_buffer[0])
    folded = jnp.stack(
        [
            read.folded_query()
            for layer in layers
            for read in (layer.mixer_read, layer.mlp_read)
        ],
        axis=-1,
    ).astype(hidden_buffer.dtype)
    raw_scores = jnp.einsum(
        "sbtd,dr->sbtr", hidden_buffer, folded,
        preferred_element_type=jnp.float32,
    )
    sum_squares = jnp.einsum(
        "sbtd,sbtd->sbt", hidden_buffer, hidden_buffer,
        preferred_element_type=jnp.float32,
    )
    for index, layer in enumerate(layers):
        read_input = layer.mixer_read.read_with_scores(
            hidden_buffer, block_index, partial_sum,
            raw_scores[..., 2 * index], sum_squares,
            include_partial=index != 0,
        )
        normalized = layer.input_norm(read_input)
        if layer.kind == "kda":
            slot = cursors["kda"]
            mixed, state, conv = kda_step(
                layer.mixer, normalized,
                cache["kda_state"][slot], cache["kda_conv"][slot],
            )
            cache["kda_state"][slot] = state
            cache["kda_conv"][slot] = conv
            cursors["kda"] += 1
        else:
            slot = cursors["gqa"]
            mixed, keys, values = gqa_step(
                layer.mixer, normalized,
                cache["gqa_key"][slot], cache["gqa_value"][slot], position,
            )
            cache["gqa_key"][slot] = keys
            cache["gqa_value"][slot] = values
            cursors["gqa"] += 1
        partial_sum = partial_sum + mixed
        mlp_input = layer.mlp_read.read_with_scores(
            hidden_buffer, block_index, partial_sum,
            raw_scores[..., 2 * index + 1], sum_squares,
            include_partial=True,
        )
        partial_sum = partial_sum + layer.mlp(
            layer.post_mixer_norm(mlp_input), deterministic=True
        )
    hidden_buffer = jax.lax.dynamic_update_slice_in_dim(
        hidden_buffer, partial_sum[None], block_index + 1, axis=0
    )
    return hidden_buffer, cache


def model_step(model, cycles, token, cache):
    """Logits for one token per row, with the cache advanced in place."""
    hidden = model.token_embedding(token[:, None], model_mode=MODEL_MODE_TRAIN)
    if model.final_read is None:
        raise NotImplementedError("incremental decode requires block_attnres")
    num_cycles = model.config.model.num_cycles
    buffer = jnp.concatenate(
        (hidden[None], jnp.zeros((num_cycles, *hidden.shape), hidden.dtype)), axis=0
    )
    # Copy the per-layer lists so the caller's cache is never mutated.
    cache = {
        key: list(value) if isinstance(value, list) else value
        for key, value in cache.items()
    }
    position = cache["position"]
    cursors = {"kda": 0, "gqa": 0}
    for index, cycle in enumerate(cycles):
        buffer, cache = cycle_step(cycle, buffer, index, cache, cursors, position)
    hidden = model.final_read(
        buffer, num_cycles, jnp.zeros_like(hidden), include_partial=False
    )
    hidden = model.final_norm(hidden)
    kernel = model.output_projection_kernel(hidden.dtype)
    logits = (hidden[:, 0] @ kernel).astype(jnp.float32)
    cache["position"] = position + 1
    return logits, cache


def _sample(logits, seen, params: SamplingParams, key):
    """Penalty, temperature, top-k then top-p, all on device."""
    scores = logits
    if params.repetition_penalty != 1.0:
        penalized = jnp.where(
            scores > 0, scores / params.repetition_penalty,
            scores * params.repetition_penalty,
        )
        scores = jnp.where(seen, penalized, scores)
    if params.temperature <= 0.0:
        return jnp.argmax(scores, axis=-1)
    scores = scores / params.temperature
    width = min(params.top_k or scores.shape[-1], scores.shape[-1])
    top_scores, top_indices = jax.lax.top_k(scores, width)
    if params.top_p < 1.0:
        probabilities = jax.nn.softmax(top_scores, axis=-1)
        cumulative = jnp.cumsum(probabilities, axis=-1)
        # Keep the first token that crosses the mass threshold.
        keep = cumulative - probabilities < params.top_p
        top_scores = jnp.where(keep, top_scores, jnp.float32(-jnp.inf))
    choice = jax.random.categorical(key, top_scores, axis=-1)
    return jnp.take_along_axis(top_indices, choice[:, None], axis=1)[:, 0]


@functools.partial(
    nnx.jit,
    static_argnames=("max_new_tokens", "sampling", "end_token", "max_length"),
)
def generate(
    model: HybridLanguageModel,
    prompt_tokens,
    prompt_lengths,
    key,
    *,
    max_new_tokens: int,
    sampling: SamplingParams,
    end_token: int,
    max_length: int,
):
    """Generates from right-padded prompts of differing lengths.

    Rows share one loop: a row still inside its prompt is fed the next
    prompt token, a row past it is fed what it last produced, so ragged
    prompts need no separate prefill pass. Every emitted token is recorded
    per step; the caller slices each row's continuation from its own prompt
    length.
    """
    cycles = split_cycles(model)
    batch, prompt_width = prompt_tokens.shape
    cache = init_cache(model, batch, max_length)
    total_steps = prompt_width + max_new_tokens
    samples = jnp.zeros((batch, total_steps), jnp.int32)
    seen = jnp.zeros((batch, model.config.model.vocab_size), bool)
    rows = jnp.arange(batch)

    def cond(carry):
        _, _, step, _, done, _, _ = carry
        return jnp.logical_and(step < total_steps, jnp.logical_not(jnp.all(done)))

    def body(carry):
        cache, samples, step, last, done, key, seen_mask = carry
        # Inside its prompt a row consumes the next prompt token; past it,
        # the row consumes what it last produced.
        token = jnp.where(step < prompt_lengths, prompt_tokens[:, step], last)
        seen_mask = seen_mask.at[rows, token].set(True)
        logits, cache = model_step(model, cycles, token, cache)
        key, subkey = jax.random.split(key)
        sampled = _sample(logits, seen_mask, sampling, subkey)
        samples = samples.at[:, step].set(sampled)
        # A row's first generated token is the one predicted after its last
        # prompt token; only from there does <|im_end|> end that row.
        generating = step >= prompt_lengths - 1
        done = jnp.logical_or(done, jnp.logical_and(generating, sampled == end_token))
        return cache, samples, step + 1, sampled, done, key, seen_mask

    initial = (
        cache, samples, jnp.int32(0), prompt_tokens[:, 0],
        jnp.zeros((batch,), bool), key, seen,
    )
    final = jax.lax.while_loop(cond, body, initial)
    return final[1], final[4]
