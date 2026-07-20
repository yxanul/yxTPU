"""GQA adaptation of Kimi's post-update QK-Clip."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from yxtpu_pretrain.model import HybridLanguageModel


@dataclass(frozen=True)
class MuonClipTelemetry:
    max_logit: jnp.ndarray
    min_scale: jnp.ndarray
    clipped_heads: jnp.ndarray


def reduce_attention_logits(max_logits: jnp.ndarray) -> jnp.ndarray:
    """Reduces `[cycles,batch,heads]` over the globally sharded batch."""
    if max_logits.ndim == 2:
        return max_logits
    if max_logits.ndim != 3:
        raise ValueError(f"expected [cycles,batch,heads] logits, got {max_logits.shape}")
    # With a globally sharded JAX array this reduction emits the required
    # cross-data-shard collective; no pmap axis-name heuristic is involved.
    return jnp.max(max_logits.astype(jnp.float32), axis=1)


def apply_gqa_muonclip(
    model: HybridLanguageModel,
    max_logits: jnp.ndarray,
    *,
    tau: float = 100.0,
    epsilon: float = 1.0e-6,
) -> MuonClipTelemetry:
    """Scales only fused GQA Q/K slices after an optimizer update.

    This is the yxTPU GQA adaptation of MuonClip, not Kimi's original MLA
    factorization. Optimizer state is intentionally not an argument and is not
    modified.
    """
    logits = reduce_attention_logits(max_logits)
    attention = model.config.model.attention
    query_heads = attention.num_query_heads
    kv_heads = attention.num_kv_heads
    group_size = query_heads // kv_heads
    if logits.shape != (model.config.model.num_cycles, query_heads):
        raise ValueError(
            f"expected {(model.config.model.num_cycles, query_heads)} max logits, "
            f"got {logits.shape}"
        )
    coefficient = jnp.minimum(
        1.0,
        jnp.asarray(tau, dtype=jnp.float32)
        / jnp.maximum(logits, jnp.asarray(epsilon, dtype=jnp.float32)),
    )
    query_scale = jnp.sqrt(coefficient)
    key_scale = jnp.sqrt(
        jnp.min(coefficient.reshape(coefficient.shape[0], kv_heads, group_size), axis=-1)
    )
    value_scale = jnp.ones_like(key_scale)
    fused_scale = jnp.concatenate((query_scale, key_scale, value_scale), axis=-1)

    parameter = model.cycles.layer_3.mixer.qkv_proj.kernel
    metadata = parameter.get_metadata()
    scan_axis = int(metadata["param_scan_axis"])
    # Fused-head is original axis 1. Inserting the scan axis at position 1
    # moves it to position 2 for the certified layout.
    head_axis = 1 if 1 < scan_axis else 2
    broadcast_shape = [1] * parameter.get_value().ndim
    broadcast_shape[scan_axis] = fused_scale.shape[0]
    broadcast_shape[head_axis] = fused_scale.shape[1]
    scaled = parameter.get_value() * fused_scale.reshape(broadcast_shape).astype(
        parameter.get_value().dtype
    )
    parameter.set_value(scaled)
    return MuonClipTelemetry(
        max_logit=jnp.max(logits, axis=-1),
        min_scale=jnp.min(fused_scale, axis=-1),
        clipped_heads=jnp.sum(coefficient < 1.0, axis=-1),
    )

