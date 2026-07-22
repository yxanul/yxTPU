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
    if include_partial:
      sources = jnp.concatenate((blocks_buffer, partial_sum[None]), axis=0)
    else:
      sources = blocks_buffer
    keys = self.norm(sources)
    query = jnp.asarray(self.pseudo_query[...], dtype=jnp.float32)
    scores = jnp.einsum("d,sbtd->sbt", query, keys.astype(jnp.float32))
    slots = blocks_buffer.shape[0]
    valid = jnp.arange(slots) <= block_index
    if include_partial:
      valid = jnp.concatenate((valid, jnp.ones((1,), dtype=bool)))
    scores = jnp.where(valid[:, None, None], scores, jnp.float32(-1.0e30))
    probabilities = jax.nn.softmax(scores, axis=0)
    combined = jnp.einsum(
        "sbt,sbtd->btd", probabilities, sources.astype(jnp.float32)
    )
    return combined.astype(blocks_buffer.dtype)
