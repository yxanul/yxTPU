"""Depth-wise attention residual reads (Block AttnRes, arXiv:2603.15031).

Each read site replaces the fixed-sum residual input with per-token softmax
attention over block representations: sources are the token embedding, every
completed block's summed output, and (except for the first sub-layer of a
block) the current intra-block partial sum. Keys are RMSNorm'd sources, the
query is a learned per-site vector decoupled from the forward computation,
and values are the raw sources, so the layer input becomes a convex
combination instead of an unbounded sum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from maxtext.layers.normalizations import RMSNorm


class DepthAttnRead(nnx.Module):
  """One pseudo-query depth-attention read over block representations."""

  def __init__(self, emb_dim: int, *, epsilon: float, dtype, weight_dtype, rngs: nnx.Rngs):
    # Zero init makes the first forward a uniform average over the valid
    # sources; there is no PreNorm-equivalent initialization by design.
    self.pseudo_query = nnx.Param(jnp.zeros((emb_dim,), dtype=weight_dtype))
    self.norm = RMSNorm(
        num_features=emb_dim,
        epsilon=epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

  def folded_query(self) -> jax.Array:
    """The pseudo-query with the RMSNorm scale folded in, in fp32.

    q . (x * rsqrt(mean(x^2)+eps) (.) scale) == ((q (.) scale) . x) *
    rsqrt(mean(x^2)+eps), so no normalized [S,B,T,D] tensor is ever
    materialized (MaxText RMSNorm has scale_offset=0 here)."""
    scale = jnp.asarray(self.norm.scale.get_value(), jnp.float32)
    return jnp.asarray(self.pseudo_query[...], dtype=jnp.float32) * scale

  def _slot_scores(self, values: jax.Array) -> jax.Array:
    """Scores one source tensor against this site's folded pseudo-query."""
    dim = values.shape[-1]
    folded = self.folded_query().astype(values.dtype)
    raw = jnp.einsum(
        "d,...d->...", folded, values,
        preferred_element_type=jnp.float32,
    )
    sum_squares = jnp.einsum(
        "...d,...d->...", values, values,
        preferred_element_type=jnp.float32,
    )
    return raw * jax.lax.rsqrt(sum_squares / dim + self.norm.epsilon)

  def _combine(
      self,
      blocks_buffer: jax.Array,
      block_index: jax.Array,
      partial_sum: jax.Array,
      buffer_scores: jax.Array,
      *,
      include_partial: bool,
  ) -> jax.Array:
    """Masks, softmaxes, and value-combines already-computed buffer scores."""
    dtype = blocks_buffer.dtype
    slots = blocks_buffer.shape[0]
    valid = jnp.arange(slots) <= block_index
    scores = jnp.where(valid[:, None, None], buffer_scores, jnp.float32(-1.0e30))
    if include_partial:
      scores = jnp.concatenate(
          (scores, self._slot_scores(partial_sum)[None]), axis=0
      )
    probabilities = jax.nn.softmax(scores, axis=0)
    combined = jnp.einsum(
        "sbt,sbtd->btd",
        probabilities[:slots].astype(dtype),
        blocks_buffer,
        preferred_element_type=jnp.float32,
    )
    if include_partial:
      combined = combined + probabilities[slots][..., None] * partial_sum.astype(
          jnp.float32
      )
    return combined.astype(dtype)

  def read_with_scores(
      self,
      blocks_buffer: jax.Array,
      block_index: jax.Array,
      partial_sum: jax.Array,
      raw_scores: jax.Array,
      sum_squares: jax.Array,
      *,
      include_partial: bool,
  ) -> jax.Array:
    """Reads with hoisted buffer scores.

    The buffer is loop-invariant within a cycle, so its raw pseudo-query
    scores for every read site of the cycle can be formed in one MXU matmul
    over a single buffer pass, and its sum of squares once rather than per
    site; only the partial-sum score (which changes between reads) stays
    local. ``raw_scores``/``sum_squares`` are this site's [slots, batch,
    length] fp32 slices of that shared computation.
    """
    dim = blocks_buffer.shape[-1]
    buffer_scores = raw_scores * jax.lax.rsqrt(
        sum_squares / dim + self.norm.epsilon
    )
    return self._combine(
        blocks_buffer,
        block_index,
        partial_sum,
        buffer_scores,
        include_partial=include_partial,
    )

  def __call__(
      self,
      blocks_buffer: jax.Array,
      block_index: jax.Array,
      partial_sum: jax.Array,
      *,
      include_partial: bool,
  ) -> jax.Array:
    """blocks_buffer is [slots, batch, length, embed]; slot 0 holds the token
    embedding and slot n holds completed block n. Slots beyond block_index
    are masked out of the softmax."""
    # Three bandwidth optimizations, verified equivalent up to rounding:
    # (1) split-scoring: score buffer and partial separately, concatenate
    # only the tiny [slots, batch, length] score tensors (RMSNorm is
    # last-axis-only, so per-slot scores are independent); (2) the RMSNorm
    # scale folds into the pseudo-query (see folded_query); (3) dots take
    # bf16 operands with fp32 accumulation instead of materializing fp32
    # copies of the buffer. Cycle-level callers hoist the buffer scores for
    # all sites into one matmul via read_with_scores; this standalone path
    # remains for single reads such as the model's final read.
    return self._combine(
        blocks_buffer,
        block_index,
        partial_sum,
        self._slot_scores(blocks_buffer),
        include_partial=include_partial,
    )
