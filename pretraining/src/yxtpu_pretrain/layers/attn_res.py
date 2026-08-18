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

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from maxtext.layers.normalizations import RMSNorm

_MASKED_SCORE = -1.0e30


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def hoisted_depth_read(blocks_buffer, block_index, folded_queries, epsilon):
  """All of a cycle's depth reads over the (loop-invariant) buffer in ONE
  buffer pass, as unnormalized softmax numerators.

  For read site ``k`` with folded pseudo-query ``q_k`` and buffer slots
  ``B_s`` (``s <= block_index`` valid): ``score_ks = (q_k . B_s) *
  rsqrt(mean(B_s^2) + eps)``, ``m_k = max_s score_ks``, ``w_ks =
  exp(score_ks - m_k)``, and the outputs are

    numerators   N_k = sum_s w_ks B_s        (K arrays [B, T, D], buffer dtype)
    normalizers  Z_k = sum_s w_ks            ([B, T, K] fp32)
    maxima       m_k                         ([B, T, K] fp32)

  A site then merges its own partial-sum term online-softmax style
  (``DepthAttnRead.merge_hoisted``), so the buffer is read once per cycle
  instead of once per site, and the backward is one hand-written pass
  (``_hoisted_depth_read_bwd``) with fp32 accumulation into a single
  buffer cotangent - where autodiff of the per-site einsums emitted one
  buffer-sized, fp32-converted, bf16-accumulated contribution per site.
  ``epsilon`` is static; ``block_index`` is an integer (no cotangent)."""
  numerators, normalizers, maxima, _ = _hoisted_depth_read_forward(
      blocks_buffer, block_index, folded_queries, epsilon
  )
  return numerators, normalizers, maxima


def _hoisted_depth_read_forward(blocks_buffer, block_index, folded_queries, epsilon):
  slots, _, _, dim = blocks_buffer.shape
  dtype = blocks_buffer.dtype
  queries = folded_queries.astype(dtype)  # [D, K]
  raw = jnp.einsum(
      "sbtd,dk->sbtk", blocks_buffer, queries, preferred_element_type=jnp.float32
  )
  sum_squares = jnp.einsum(
      "sbtd,sbtd->sbt", blocks_buffer, blocks_buffer, preferred_element_type=jnp.float32
  )
  inverse_rms = jax.lax.rsqrt(sum_squares / dim + epsilon)  # [S, B, T]
  valid = (jnp.arange(slots) <= block_index)[:, None, None, None]
  scores = jnp.where(valid, raw * inverse_rms[..., None], jnp.float32(_MASKED_SCORE))
  maxima = jnp.max(scores, axis=0)  # [B, T, K]
  weights = jnp.where(valid, jnp.exp(scores - maxima[None]), 0.0)  # [S, B, T, K]
  normalizers = jnp.sum(weights, axis=0)  # [B, T, K]
  numerators = jnp.einsum(
      "sbtk,sbtd->kbtd",
      weights.astype(dtype),
      blocks_buffer,
      preferred_element_type=jnp.float32,
  ).astype(dtype)
  numerators = tuple(numerators[k] for k in range(numerators.shape[0]))
  residuals = (blocks_buffer, queries, raw, inverse_rms, weights)
  return numerators, normalizers, maxima, residuals


def _hoisted_depth_read_fwd(blocks_buffer, block_index, folded_queries, epsilon):
  numerators, normalizers, maxima, residuals = _hoisted_depth_read_forward(
      blocks_buffer, block_index, folded_queries, epsilon
  )
  return (numerators, normalizers, maxima), residuals


def _hoisted_depth_read_bwd(epsilon, residuals, cotangents):
  del epsilon
  blocks_buffer, queries, raw, inverse_rms, weights = residuals
  d_numerators, d_normalizers, _ = cotangents  # maxima: see below
  dtype = blocks_buffer.dtype
  dim = blocks_buffer.shape[-1]
  d_numerators = jnp.stack(d_numerators, axis=0)  # [K, B, T, D] (buffer dtype)
  # dw_ks = dN_k . B_s + dZ_k ; d score_ks = w_ks * dw_ks. The maxima's
  # cotangent is dropped on purpose: every consumer is exactly invariant to
  # the shift m_k (numerator and normalizer scale together), so the
  # derivative taken with m_k held fixed IS the derivative.
  d_weights = jnp.einsum(
      "kbtd,sbtd->sbtk", d_numerators, blocks_buffer, preferred_element_type=jnp.float32
  ) + d_normalizers[None].astype(jnp.float32)
  d_scores = weights * d_weights  # [S, B, T, K] fp32 (masked slots: w = 0)
  # score_ks = raw_ks * r_s with raw_ks = q_k . B_s and r_s = rsqrt(ss_s/D+eps):
  #   dB_s += r_s * sum_k d_score_ks q_k  -  (r_s^3 / D) (sum_k d_score_ks raw_ks) B_s
  #   dq_k += sum_{s,b,t} d_score_ks r_s B_s
  scaled_scores = d_scores * inverse_rms[..., None]  # [S, B, T, K]
  d_buffer = jnp.einsum(
      "sbtk,kbtd->sbtd", weights.astype(dtype), d_numerators, preferred_element_type=jnp.float32
  )
  d_buffer = d_buffer + jnp.einsum(
      "sbtk,dk->sbtd", scaled_scores.astype(dtype), queries, preferred_element_type=jnp.float32
  )
  radial = -jnp.sum(d_scores * raw, axis=-1) * inverse_rms**3 / dim  # [S, B, T]
  d_buffer = d_buffer + radial[..., None] * blocks_buffer.astype(jnp.float32)
  d_queries = jnp.einsum(
      "sbtk,sbtd->dk", scaled_scores.astype(dtype), blocks_buffer, preferred_element_type=jnp.float32
  )
  return d_buffer.astype(dtype), None, d_queries


hoisted_depth_read.defvjp(_hoisted_depth_read_fwd, _hoisted_depth_read_bwd)


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

  def _slot_scores(self, values: jax.Array, folded_query: jax.Array | None = None) -> jax.Array:
    """Scores one source tensor against this site's folded pseudo-query
    (``folded_query`` lets a caller that already formed it pass it in)."""
    dim = values.shape[-1]
    folded = (self.folded_query() if folded_query is None else folded_query).astype(values.dtype)
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

  def merge_hoisted(
      self,
      numerator: jax.Array,
      normalizer: jax.Array,
      maximum: jax.Array,
      partial_sum: jax.Array,
      *,
      include_partial: bool,
      folded_query: jax.Array | None = None,
  ) -> jax.Array:
    """Finishes this site's read from the cycle-hoisted buffer numerator
    (see ``hoisted_depth_read``): folds the intra-block partial sum in as
    one more softmax slot (online-softmax merge of the two maxima), then
    normalizes. Without the partial term it is ``N_k / Z_k``."""
    # Written as out = alpha_t N_k + beta_t P with the two PER-TOKEN scalars
    # formed first (tiny [B, T] work), so the only full-width work is one
    # multiply-add over [B, T, D] that XLA can keep as a single fusion with
    # bf16 in and out - the earlier form (fp32 combine, then a broadcast
    # divide) left several fp32 [B, T, D] temporaries and converts per site.
    dtype = numerator.dtype
    if not include_partial:
      alpha = 1.0 / normalizer  # [B, T]
      return (numerator.astype(jnp.float32) * alpha[..., None]).astype(dtype)
    partial_score = self._slot_scores(partial_sum, folded_query)  # [B, T] fp32
    # The result is exactly invariant to the shift, so it carries no gradient.
    merged_max = jax.lax.stop_gradient(jnp.maximum(maximum, partial_score))
    buffer_scale = jnp.exp(maximum - merged_max)
    partial_weight = jnp.exp(partial_score - merged_max)
    inverse_denominator = 1.0 / (normalizer * buffer_scale + partial_weight)
    alpha = buffer_scale * inverse_denominator  # [B, T]
    beta = partial_weight * inverse_denominator  # [B, T]
    return (
        numerator.astype(jnp.float32) * alpha[..., None]
        + partial_sum.astype(jnp.float32) * beta[..., None]
    ).astype(dtype)

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
