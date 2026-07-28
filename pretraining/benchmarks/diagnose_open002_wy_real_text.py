# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OPEN-002: WY-system statistics from real language tokens vs random tokens
at identical trained weights, across layer depth.

Restores a pretraining checkpoint on a single host, walks the scanned cycles
eagerly with a capture hook on every KDA mixer, and for sampled
(row, chunk, head) systems measures both the proxies (beta quantiles, key
cosine similarity at lags 1/4/16, decay retention, kappa_2(I+A), maximum
doubling-power norm) and the quantities that decide the C1 solve-precision
ladder directly: TPU-arithmetic forward and transposed solve error of the
production divide-and-conquer inverse at one-pass (DEFAULT), three-pass
(HIGH), and six-pass (HIGHEST) matmul precision against an fp64 host
reference, on the exact right-hand sides the kernel solves
([value_beta | w_input]).

EXP-027's lesson is respected: the decisive columns are computed with real
TPU matmul arithmetic at each precision, never by rounding operands once and
computing in fp32.

Run on a v4 worker under the single-host env restriction:

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 \
  .venv/bin/python benchmarks/diagnose_open002_wy_real_text.py \
      --init-destination /home/a1111/yxtpu_ckpts \
      --init-run kda_hybrid_128k-muonclip-superbpe_50b \
      --output /tmp/open002.json
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import lax
from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.utils import maxtext_utils_nnx

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.kernels.kda_fused_pallas_v4 import (
    _block_row_mask,
    _blockdiag_strictly_lower_mask,
    _half_coupling_mask,
)
from yxtpu_pretrain.layers.kimi_delta_attention import KimiDeltaAttention
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO
from yxtpu_pretrain.runtime.data import create_data_iterator
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context

CHUNK = 64


class _NoIterator:
    def set_state(self, payload):
        raise AssertionError("stream state must not restore in the diagnostic")


def _capture_kda_inputs(mixer: KimiDeltaAttention, hidden_states):
    """Mirrors the layer's projection block up to (key, beta, log_decay)."""
    batch, sequence_length, _ = hidden_states.shape
    qkv = mixer.in_proj_qkv(hidden_states)
    decay_hidden = mixer.decay_down(hidden_states)
    beta_logits = mixer.beta_proj(hidden_states)
    qkv = qkv.transpose(0, 1, 3, 2, 4).reshape(batch, sequence_length, -1)
    qkv = jnp.pad(qkv, ((0, 0), (mixer.config.gdn_conv_kernel_dim - 1, 0), (0, 0)))
    qkv = mixer.conv1d(qkv)[:, -sequence_length:]
    qkv = jax.nn.silu(qkv.astype(jnp.float32)).astype(mixer.config.dtype)
    qkv = qkv.reshape(batch, sequence_length, mixer.num_heads, 3, mixer.head_dim)
    key = qkv[..., 1, :]
    value = qkv[..., 2, :]

    decay_input = mixer.decay_up(decay_hidden)
    raw_decay = decay_input.astype(jnp.float32) + jnp.asarray(
        mixer.dt_bias[...], dtype=jnp.float32
    )
    decay_rate = jnp.exp(jnp.asarray(mixer.A_log[...], dtype=jnp.float32))[
        None, None, :, None
    ]
    log_decay = mixer.config.kda_gate_lower_bound * jax.nn.sigmoid(
        decay_rate * raw_decay
    )
    beta = jax.nn.sigmoid(beta_logits.astype(jnp.float32))
    # Kernel parity: the fused kernel L2-normalizes the key in fp32.
    key = key.astype(jnp.float32)
    key = key * lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    return (
        np.asarray(key, dtype=np.float32),
        np.asarray(value, dtype=np.float32),
        np.asarray(beta, dtype=np.float32),
        np.asarray(log_decay, dtype=np.float32),
    )


def _walk_cycles(model, config, batch_tokens):
    """Eager replica of the scanned forward that lets the capture hook fire."""
    token_ids = jnp.asarray(batch_tokens)
    segment_ids = jnp.ones_like(token_ids, dtype=jnp.int32)
    positions = jnp.broadcast_to(
        jnp.arange(token_ids.shape[1], dtype=jnp.int32), token_ids.shape
    )
    hidden = model.token_embedding(token_ids, model_mode="train")
    graphdef, params, state = nnx.split(model.cycles, nnx.Param, ...)
    scan_axis = config.model.param_scan_axis
    if scan_axis != 0:
        params = jax.tree.map(lambda value: jnp.moveaxis(value, scan_axis, 0), params)
    length = config.model.num_cycles
    params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, length)
    state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, length)

    if config.model.residual_policy == "block_attnres":
        carry = (
            jnp.concatenate(
                (
                    hidden[None],
                    jnp.zeros((length, *hidden.shape), dtype=hidden.dtype),
                ),
                axis=0,
            ),
            jnp.int32(0),
        )
    else:
        carry = hidden
    for index in range(length):
        cycle_variables = jax.tree.map(lambda value: value[index], (params, state))
        cycle_variables = maxtext_utils_nnx.nnx_remove_scan_axis(
            cycle_variables, "cycles"
        )
        current_params, current_state = cycle_variables
        cycle = nnx.merge(graphdef, current_params, current_state)
        carry = cycle(
            carry,
            decoder_segment_ids=segment_ids,
            decoder_positions=positions,
            record_max_logits=False,
        )


def _batched_inverse(system, precision):
    """The production divide-and-conquer inverse at a chosen matmul precision."""
    rows = system.shape[-1]
    lower = jnp.tril(system.astype(jnp.float32), k=-1)
    identity = jnp.broadcast_to(jnp.eye(rows, dtype=jnp.float32), system.shape)
    base_lower = lower * _blockdiag_strictly_lower_mask(rows, 2)
    inverse = identity - base_lower * _block_row_mask(rows, 2, 1)
    half = 2
    while half < rows:
        coupling = lower * _half_coupling_mask(rows, half)
        inverse = inverse - jnp.matmul(
            jnp.matmul(inverse, coupling, precision=precision),
            inverse,
            precision=precision,
        )
        half *= 2
    return inverse


def _solve_errors(systems, rhs, reference, reference_transposed, rhs_transposed):
    """Relative L2 solve errors at each TPU matmul precision vs fp64 host."""
    results = {}
    for name, precision in (
        ("default_1pass", lax.Precision.DEFAULT),
        ("high_3pass", lax.Precision.HIGH),
        ("highest_6pass", lax.Precision.HIGHEST),
    ):
        inverse = _batched_inverse(systems, precision)
        solved = jnp.matmul(inverse, rhs, precision=precision)
        solved_transposed = jnp.matmul(
            jnp.swapaxes(inverse, -1, -2), rhs_transposed, precision=precision
        )
        forward = np.asarray(solved, dtype=np.float64)
        transposed = np.asarray(solved_transposed, dtype=np.float64)
        scale = np.linalg.norm(reference, axis=(-2, -1)) + 1e-30
        scale_transposed = (
            np.linalg.norm(reference_transposed, axis=(-2, -1)) + 1e-30
        )
        results[name] = {
            "forward_rel_l2": np.linalg.norm(forward - reference, axis=(-2, -1))
            / scale,
            "transposed_rel_l2": np.linalg.norm(
                transposed - reference_transposed, axis=(-2, -1)
            )
            / scale_transposed,
        }
    return results


def _quantiles(values, points=(0.5, 0.95, 1.0)):
    array = np.asarray(values, dtype=np.float64).ravel()
    return {f"p{int(point * 100)}": float(np.quantile(array, point)) for point in points}


def _analyze_layer(key, value, beta, log_decay, rng, systems_per_layer, slice_size=64):
    """Samples systems from one layer's [B,T,H,D] capture and measures them."""
    batch, sequence, heads, dim = key.shape
    chunks = sequence // CHUNK
    key = key.reshape(batch, chunks, CHUNK, heads, dim)
    value = value.reshape(batch, chunks, CHUNK, heads, dim)
    beta = beta.reshape(batch, chunks, CHUNK, heads)
    log_decay = log_decay.reshape(batch, chunks, CHUNK, heads, dim)

    rows = rng.integers(0, batch, size=systems_per_layer)
    chunk_ids = rng.integers(0, chunks, size=systems_per_layer)
    head_ids = rng.integers(0, heads, size=systems_per_layer)
    sampled_key = key[rows, chunk_ids][np.arange(systems_per_layer), :, head_ids]
    sampled_value = value[rows, chunk_ids][np.arange(systems_per_layer), :, head_ids]
    sampled_beta = beta[rows, chunk_ids][np.arange(systems_per_layer), :, head_ids]
    sampled_decay = log_decay[rows, chunk_ids][
        np.arange(systems_per_layer), :, head_ids
    ]
    cumulative = np.cumsum(sampled_decay, axis=1)  # [N, 64, D]

    # System and RHS exactly as the kernel forms them.
    key_beta = sampled_key * sampled_beta[..., None]
    systems = np.zeros((systems_per_layer, CHUNK, CHUNK), np.float32)
    for start in range(0, systems_per_layer, slice_size):
        stop = min(start + slice_size, systems_per_layer)
        decay_weight = np.exp(
            np.clip(
                cumulative[start:stop, :, None, :] - cumulative[start:stop, None, :, :],
                -80.0,
                0.0,
            )
        )
        block = np.einsum(
            "nic,nijc,njc->nij",
            key_beta[start:stop].astype(np.float64),
            decay_weight.astype(np.float64),
            sampled_key[start:stop].astype(np.float64),
        )
        systems[start:stop] = np.tril(block, k=-1).astype(np.float32)

    w_input = (key_beta * np.exp(cumulative)).astype(np.float32)
    value_beta = (sampled_value * sampled_beta[..., None]).astype(np.float32)
    rhs = np.concatenate((value_beta, w_input), axis=-1)
    rhs_transposed = rng.standard_normal(rhs.shape).astype(np.float32)

    matrices = np.eye(CHUNK)[None] + systems.astype(np.float64)
    reference = np.linalg.solve(matrices, rhs.astype(np.float64))
    reference_transposed = np.linalg.solve(
        np.swapaxes(matrices, -1, -2), rhs_transposed.astype(np.float64)
    )
    errors = _solve_errors(
        jnp.asarray(systems),
        jnp.asarray(rhs),
        reference,
        reference_transposed,
        jnp.asarray(rhs_transposed),
    )

    # Proxies.
    kappa = np.array([np.linalg.cond(matrix) for matrix in matrices])
    power = -systems.astype(np.float64)
    growth = np.linalg.norm(power, axis=(-2, -1))
    for _ in range(5):
        power = np.matmul(power, power)
        growth = np.maximum(growth, np.linalg.norm(power, axis=(-2, -1)))
    cosines = {}
    for lag in (1, 4, 16):
        dots = np.abs(
            np.einsum(
                "ntd,ntd->nt", sampled_key[:, lag:], sampled_key[:, :-lag]
            )
        )
        cosines[f"key_cos_lag{lag}"] = _quantiles(dots)
    retention = np.exp(cumulative[:, -1, :])

    summary = {
        "beta": _quantiles(sampled_beta),
        "retention": _quantiles(retention, points=(0.05, 0.5, 0.95)),
        "kappa2": _quantiles(kappa),
        "max_power_norm": _quantiles(growth),
        **cosines,
    }
    for name, error in errors.items():
        summary[f"solve_{name}_forward"] = _quantiles(error["forward_rel_l2"])
        summary[f"solve_{name}_transposed"] = _quantiles(error["transposed_rel_l2"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-destination", default="/home/a1111/yxtpu_ckpts")
    parser.add_argument("--init-run", default="kda_hybrid_128k-muonclip-superbpe_50b")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--systems-per-layer", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="/tmp/open002.json")
    arguments = parser.parse_args()

    config = load_config(
        model="kda_hybrid_128k",
        optimizer="muonclip",
        data="climbmix_superbpe",
        hardware="v4-64",
        experiment="superbpe_50b",
        overrides=[
            "experiment.harness_eval.enabled=false",
            "experiment.diagnostics.enabled=false",
            f"experiment.checkpoint.destination={arguments.init_destination}",
            f"data.per_device_batch_size={arguments.per_device_batch}",
        ],
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))
    loader = CheckpointIO(config, run_name=arguments.init_run)
    step = loader.restore(state, _NoIterator())
    loader.close()
    if step == 0:
        raise RuntimeError("no checkpoint restored")
    print(f"restored checkpoint step {step}", flush=True)

    process_batch = arguments.per_device_batch * jax.local_device_count()
    iterator = create_data_iterator(
        config.data,
        global_batch_size=process_batch,
        vocab_size=config.model.vocab_size,
        process_index=0,
        process_count=1,
    )
    rng = np.random.default_rng(arguments.seed)
    sources = {}
    for source in ("real", "random"):
        captures = []
        original_call = KimiDeltaAttention.__call__

        def capturing_call(self, hidden_states, *args, **kwargs):
            captures.append(_capture_kda_inputs(self, hidden_states))
            return original_call(self, hidden_states, *args, **kwargs)

        KimiDeltaAttention.__call__ = capturing_call
        try:
            with logical_mesh_context(mesh, rules):
                for _ in range(arguments.batches):
                    if source == "real":
                        tokens = next(iterator)["input_ids"]
                    else:
                        tokens = rng.integers(
                            0,
                            config.model.vocab_size,
                            size=(process_batch, config.data.sequence_length),
                            dtype=np.int32,
                        )
                    _walk_cycles(model, config, tokens)
        finally:
            KimiDeltaAttention.__call__ = original_call
        layers = 3 * config.model.num_cycles  # 3 KDA layers per cycle
        print(f"{source}: captured {len(captures)} KDA layer calls", flush=True)
        per_layer = {}
        for layer_index in range(layers):
            layer_captures = captures[layer_index::layers]
            key = np.concatenate([c[0] for c in layer_captures], axis=0)
            value = np.concatenate([c[1] for c in layer_captures], axis=0)
            beta = np.concatenate([c[2] for c in layer_captures], axis=0)
            decay = np.concatenate([c[3] for c in layer_captures], axis=0)
            per_layer[f"layer_{layer_index}"] = _analyze_layer(
                key, value, beta, decay, rng, arguments.systems_per_layer
            )
            print(f"{source} layer {layer_index}: done", flush=True)
        sources[source] = per_layer

    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump(sources, handle, indent=1)
    print(f"wrote {arguments.output}", flush=True)

    for source, per_layer in sources.items():
        print(f"==== {source} ====")
        for layer_name, summary in per_layer.items():
            print(
                f"{layer_name}: kappa2 p95={summary['kappa2']['p95']:.1f} "
                f"growth p95={summary['max_power_norm']['p95']:.3e} "
                f"beta p95={summary['beta']['p95']:.3f} "
                f"cos1 p95={summary['key_cos_lag1']['p95']:.3f} | "
                f"fwd err 1pass p95={summary['solve_default_1pass_forward']['p95']:.3e} "
                f"3pass p95={summary['solve_high_3pass_forward']['p95']:.3e} "
                f"6pass p95={summary['solve_highest_6pass_forward']['p95']:.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
