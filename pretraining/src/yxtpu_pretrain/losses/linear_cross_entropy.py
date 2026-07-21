"""Mask-aware fused linear softmax cross entropy for replicated vocabularies.

The Mosaic implementation delegates its single-device kernel and analytical
VJP to Tokamax.  This module owns the two pieces Tokamax intentionally does
not provide:

* binary padding-mask semantics; and
* explicit data-parallel loss/count reduction.

The vocabulary and hidden projection dimensions must be replicated.  A future
vocabulary-parallel implementation needs global max, sum-exp, target-logit,
and input-gradient collectives and must not route through this module.
"""

from __future__ import annotations

import sys
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from absl import flags
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from tokamax import linear_softmax_cross_entropy_loss

Implementation = Literal["reference", "mosaic_tpu"]


def _initialize_tokamax_flags() -> None:
    """Prevents Tokamax's lazy Abseil parser from consuming our CLI flags."""
    if not flags.FLAGS.is_parsed():
        flags.FLAGS([sys.argv[0]], known_only=True)


def _linear_logits(x: jax.Array, weights: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x,
        weights,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _reference_sum(
    x: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Materializing FP32 reference used only by correctness tests."""
    mask = loss_mask.astype(jnp.float32)
    logits = _linear_logits(x, weights)
    safe_labels = jnp.where(mask > 0, labels, 0).astype(jnp.int32)
    target_logits = jnp.take_along_axis(logits, safe_labels[:, None], axis=-1)[:, 0]
    per_token = jax.nn.logsumexp(logits, axis=-1) - target_logits
    return jnp.sum(per_token * mask, dtype=jnp.float32)


def _mosaic_sum(
    x: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Tokamax sum with exact binary-mask gradients and no HBM logits."""
    _initialize_tokamax_flags()
    mask = loss_mask.astype(jnp.float32)
    masked_x = jnp.where(mask[:, None] > 0, x, jnp.zeros((), dtype=x.dtype))
    raw_sum = linear_softmax_cross_entropy_loss(
        masked_x,
        labels.astype(jnp.int32),
        weights,
        reduction="sum",
        # Be explicit: Tokamax 0.0.12 has a broken fallback error path and the
        # XLA implementation would defeat the memory objective.
        implementation="mosaic_tpu",
    )
    invalid_tokens = jnp.sum(1.0 - mask, dtype=jnp.float32)
    masked_token_loss = jnp.log(jnp.asarray(weights.shape[1], dtype=jnp.float32))
    return raw_sum - invalid_tokens * masked_token_loss


def local_linear_cross_entropy_sum(
    x: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array,
    weights: jax.Array,
    *,
    implementation: Implementation,
) -> jax.Array:
    """Computes a device-local masked loss sum.

    ``loss_mask`` is required to contain only zero or one.  All owned data
    pipelines produce binary masks; host-side fixtures assert that invariant.
    Runtime value checks are deliberately excluded from the compiled step.
    """
    if x.ndim != 2 or weights.ndim != 2:
        raise ValueError("fused linear cross entropy expects x[B,H] and weights[H,V]")
    if labels.shape != (x.shape[0],) or loss_mask.shape != (x.shape[0],):
        raise ValueError("labels and loss_mask must match the flattened token dimension")
    if x.shape[1] != weights.shape[0]:
        raise ValueError("hidden dimension does not match the output projection")
    if implementation == "reference":
        return _reference_sum(x, labels, loss_mask, weights)
    if implementation == "mosaic_tpu":
        return _mosaic_sum(x, labels, loss_mask, weights)
    raise ValueError(f"unsupported fused loss implementation: {implementation}")


def data_parallel_linear_cross_entropy(
    x: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array,
    weights: jax.Array,
    *,
    mesh: Mesh,
    implementation: Implementation = "mosaic_tpu",
) -> tuple[jax.Array, jax.Array]:
    """Returns globally normalized loss and valid-token count.

    The mapped function sees a local token shard and a complete replicated
    vocabulary.  Both local loss and valid-token count are explicitly reduced
    over the data axis.  Autodiff of the replicated weight input performs the
    corresponding data-parallel parameter-gradient reduction.
    """
    if mesh.shape.get("data", 1) < 1:
        raise ValueError("the mesh must define a data axis")

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("data", None), P("data"), P("data"), P()),
        out_specs=(P(), P()),
        axis_names={"data"},
        # Tokamax 0.0.12 constructs Pallas ShapeDtypeStruct outputs without the
        # manual_axis_type metadata required by JAX 0.10's VMA checker.  The
        # explicit psums below and the eight-device parity gate define and test
        # the distributed contract instead.
        check_vma=False,
    )
    def mapped(local_x, local_labels, local_mask, replicated_weights):
        local_sum = local_linear_cross_entropy_sum(
            local_x,
            local_labels,
            local_mask,
            replicated_weights,
            implementation=implementation,
        )
        local_tokens = jnp.sum(local_mask.astype(jnp.float32), dtype=jnp.float32)
        global_sum = jax.lax.psum(local_sum, "data")
        global_tokens = jax.lax.psum(local_tokens, "data")
        return global_sum / jnp.maximum(global_tokens, 1.0), global_tokens

    return mapped(x, labels, loss_mask, weights)
