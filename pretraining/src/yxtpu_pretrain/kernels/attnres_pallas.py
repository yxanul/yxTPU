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

"""Fused Block-AttnRes depth read (Pallas TPU).

The XLA form of the cycle-hoisted read (``layers/attn_res.py::
hoisted_depth_read``) is exact but perf-neutral on v4: XLA materializes
each backward contraction as a full fp32 buffer-sized tensor, adds them,
applies the radial term and casts in separate passes, and the buffer
stops being updated in place. The mechanism's roofline is "read the
buffer once forward, once more for the recompute, once in the backward"
- only a fused kernel that keeps a token tile of the buffer resident in
VMEM and accumulates in fp32 there gets that.

Three kernels over token tiles of the ``[S, N = B*T, D]`` buffer, all
VPU work (elementwise FMAs over ``[tile, D]`` with ``[tile, 1]``
broadcasts and lane reductions; no MXU, no in-kernel transposes - the v4
constraints):

  scores      B_s tile -> raw_ks = q_k . B_s and ss_s = |B_s|^2 (fp32)
  numerators  w (from XLA, masked softmax weights) + B tile ->
              N_k = sum_{s <= block} w_ks B_s, one bf16 write per k
  backward    B tile, w, raw, 1/rms, dN_k, dZ_k, q -> dB_s (fp32
              accumulated in VMEM, written bf16 once) and the per-tile
              partial of dq_k = sum_{s,t} (dscore_ks r_s) B_s

Slots above ``block_index`` are skipped by ``pl.when`` (their dB is
zero, as in the XLA path). The interface and residual layout match
``hoisted_depth_read`` so ``DepthAttnRead.merge_hoisted`` is unchanged.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_MASKED_SCORE = -1.0e30
_DEFAULT_FORWARD_TILE = 64
_DEFAULT_BACKWARD_TILE = 32


def _interpret() -> bool:
  return jax.default_backend() != "tpu"


# --------------------------------------------------------------------------
# kernel 1: raw pseudo-query scores and sums of squares, per slot
# --------------------------------------------------------------------------
def _scores_kernel(index_ref, buffer_ref, query_ref, raw_ref, ss_ref, *, slots, sites):
  block_index = index_ref[0]
  for s in range(slots):

    @pl.when(s <= block_index)
    def _compute():
      values = buffer_ref[s].astype(jnp.float32)  # [tile, D]
      ss_ref[s] = jnp.sum(values * values, axis=-1, keepdims=True)
      for k in range(sites):
        query = query_ref[k : k + 1, :].astype(jnp.float32)  # [1, D]
        raw_ref[s, :, k : k + 1] = jnp.sum(values * query, axis=-1, keepdims=True)

    @pl.when(s > block_index)
    def _skip():
      ss_ref[s] = jnp.zeros(ss_ref.shape[1:], jnp.float32)
      raw_ref[s] = jnp.zeros(raw_ref.shape[1:], jnp.float32)


def _scores(buffer, block_index, queries_kd, *, tile):
  slots, tokens, dim = buffer.shape
  sites = queries_kd.shape[0]
  grid = (tokens // tile,)
  return pl.pallas_call(
      functools.partial(_scores_kernel, slots=slots, sites=sites),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=grid,
          in_specs=[
              pl.BlockSpec((slots, tile, dim), lambda i, idx: (0, i, 0)),
              pl.BlockSpec((sites, dim), lambda i, idx: (0, 0)),
          ],
          out_specs=[
              pl.BlockSpec((slots, tile, sites), lambda i, idx: (0, i, 0)),
              pl.BlockSpec((slots, tile, 1), lambda i, idx: (0, i, 0)),
          ],
      ),
      out_shape=[
          jax.ShapeDtypeStruct((slots, tokens, sites), jnp.float32),
          jax.ShapeDtypeStruct((slots, tokens, 1), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=_interpret(),
      name="attnres_scores",
  )(block_index, buffer, queries_kd)


# --------------------------------------------------------------------------
# kernel 2: numerators N_k = sum_s w_ks B_s
# --------------------------------------------------------------------------
def _numerators_kernel(index_ref, buffer_ref, weights_ref, *rest, slots, sites):
  out_refs = rest[:sites]
  (acc_ref,) = rest[sites:]
  block_index = index_ref[0]
  acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

  def body(s, carry):
    values = buffer_ref[s].astype(jnp.float32)  # [tile, D]
    for k in range(sites):
      acc_ref[k] += weights_ref[s, :, k : k + 1] * values
    return carry

  jax.lax.fori_loop(0, block_index + 1, body, 0)
  for k in range(sites):
    out_refs[k][...] = acc_ref[k].astype(out_refs[k].dtype)


def _numerators(buffer, block_index, weights, *, tile):
  slots, tokens, dim = buffer.shape
  sites = weights.shape[-1]
  grid = (tokens // tile,)
  return pl.pallas_call(
      functools.partial(_numerators_kernel, slots=slots, sites=sites),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=grid,
          in_specs=[
              pl.BlockSpec((slots, tile, dim), lambda i, idx: (0, i, 0)),
              pl.BlockSpec((slots, tile, sites), lambda i, idx: (0, i, 0)),
          ],
          out_specs=[pl.BlockSpec((tile, dim), lambda i, idx: (i, 0)) for _ in range(sites)],
          scratch_shapes=[pltpu.VMEM((sites, tile, dim), jnp.float32)],
      ),
      out_shape=[jax.ShapeDtypeStruct((tokens, dim), buffer.dtype) for _ in range(sites)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=_interpret(),
      name="attnres_numerators",
  )(block_index, buffer, weights)


# --------------------------------------------------------------------------
# kernel 3: backward - dB (single fp32-accumulated bf16 write) and dq partials
# --------------------------------------------------------------------------
def _backward_kernel(
    index_ref,
    buffer_ref,
    weights_ref,
    raw_ref,
    inv_rms_ref,
    d_norm_ref,
    query_ref,
    *rest,
    slots,
    sites,
    dim,
):
  d_num_refs = rest[:sites]
  d_buffer_ref, d_query_ref, d_num_scratch = rest[sites:]
  block_index = index_ref[0]
  # The K numerator cotangents are loop-invariant across slots: convert once.
  for k in range(sites):
    d_num_scratch[k] = d_num_refs[k][...].astype(jnp.float32)
  d_query_ref[...] = jnp.zeros(d_query_ref.shape, jnp.float32)
  for s in range(slots):

    @pl.when(s <= block_index)
    def _compute():
      values = buffer_ref[s].astype(jnp.float32)  # [tile, D]
      inv_rms = inv_rms_ref[s]  # [tile, 1]
      d_buffer = jnp.zeros(values.shape, jnp.float32)
      radial = jnp.zeros(inv_rms.shape, jnp.float32)
      for k in range(sites):
        d_num = d_num_scratch[k]  # [tile, D] fp32
        weight = weights_ref[s, :, k : k + 1]  # [tile, 1]
        # dw_ks = dN_k . B_s + dZ_k ; d score_ks = w_ks dw_ks
        d_weight = jnp.sum(d_num * values, axis=-1, keepdims=True) + d_norm_ref[:, k : k + 1]
        d_score = weight * d_weight  # [tile, 1]
        scaled = d_score * inv_rms  # [tile, 1]
        query = query_ref[k : k + 1, :].astype(jnp.float32)  # [1, D]
        d_buffer = d_buffer + weight * d_num + scaled * query
        radial = radial + d_score * raw_ref[s, :, k : k + 1]
        # dq_k += sum_t scaled_t B_t  (sublane reduction over the tile)
        d_query_ref[0, k : k + 1, :] += jnp.sum(scaled * values, axis=0, keepdims=True)
      coefficient = -radial * inv_rms * inv_rms * inv_rms / dim  # [tile, 1]
      d_buffer = d_buffer + coefficient * values
      d_buffer_ref[s] = d_buffer.astype(d_buffer_ref.dtype)

    @pl.when(s > block_index)
    def _skip():
      d_buffer_ref[s] = jnp.zeros(d_buffer_ref.shape[1:], d_buffer_ref.dtype)


def _backward(buffer, block_index, weights, raw, inv_rms, d_numerators, d_normalizers, queries_kd, *, tile):
  slots, tokens, dim = buffer.shape
  sites = queries_kd.shape[0]
  grid = (tokens // tile,)
  d_buffer, d_query_partials = pl.pallas_call(
      functools.partial(_backward_kernel, slots=slots, sites=sites, dim=dim),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=grid,
          in_specs=[
              pl.BlockSpec((slots, tile, dim), lambda i, idx: (0, i, 0)),  # buffer
              pl.BlockSpec((slots, tile, sites), lambda i, idx: (0, i, 0)),  # weights
              pl.BlockSpec((slots, tile, sites), lambda i, idx: (0, i, 0)),  # raw
              pl.BlockSpec((slots, tile, 1), lambda i, idx: (0, i, 0)),  # inv rms
              pl.BlockSpec((tile, sites), lambda i, idx: (i, 0)),  # d normalizers
              pl.BlockSpec((sites, dim), lambda i, idx: (0, 0)),  # queries [K, D]
          ]
          + [pl.BlockSpec((tile, dim), lambda i, idx: (i, 0)) for _ in range(sites)],
          out_specs=[
              pl.BlockSpec((slots, tile, dim), lambda i, idx: (0, i, 0)),
              pl.BlockSpec((1, sites, dim), lambda i, idx: (i, 0, 0)),
          ],
          scratch_shapes=[pltpu.VMEM((sites, tile, dim), jnp.float32)],
      ),
      out_shape=[
          jax.ShapeDtypeStruct((slots, tokens, dim), buffer.dtype),
          jax.ShapeDtypeStruct((tokens // tile, sites, dim), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=_interpret(),
      name="attnres_backward",
  )(block_index, buffer, weights, raw, inv_rms, d_normalizers, queries_kd, *d_numerators)
  return d_buffer, d_query_partials


# --------------------------------------------------------------------------
# custom_vjp with the hoisted_depth_read interface
# --------------------------------------------------------------------------
def _forward(blocks_buffer, block_index, folded_queries, epsilon, forward_tile):
  slots, batch, length, dim = blocks_buffer.shape
  sites = folded_queries.shape[-1]
  tokens = batch * length
  if tokens % forward_tile:
    raise ValueError(f"batch*length ({tokens}) must be a multiple of the tile ({forward_tile})")
  dtype = blocks_buffer.dtype
  buffer = blocks_buffer.reshape(slots, tokens, dim)
  queries_kd = jnp.transpose(folded_queries).astype(dtype)  # [K, D] in the buffer dtype
  index = jnp.asarray(block_index, jnp.int32).reshape(1)
  raw, sum_squares = _scores(buffer, index, queries_kd, tile=forward_tile)  # [S,N,K], [S,N,1]
  inv_rms = jax.lax.rsqrt(sum_squares / dim + epsilon)  # [S, N, 1]
  valid = (jnp.arange(slots) <= block_index)[:, None, None]
  scores = jnp.where(valid, raw * inv_rms, jnp.float32(_MASKED_SCORE))
  maxima = jnp.max(scores, axis=0)  # [N, K]
  weights = jnp.where(valid, jnp.exp(scores - maxima[None]), 0.0)  # [S, N, K]
  normalizers = jnp.sum(weights, axis=0)  # [N, K]
  numerators = _numerators(buffer, index, weights, tile=forward_tile)  # K x [N, D]
  numerators = tuple(n.reshape(batch, length, dim) for n in numerators)
  residuals = (buffer, index, queries_kd, raw, inv_rms, weights)
  return (
      numerators,
      normalizers.reshape(batch, length, sites),
      maxima.reshape(batch, length, sites),
      residuals,
  )


def pallas_hoisted_depth_read(
    blocks_buffer,
    block_index,
    folded_queries,
    epsilon,
    *,
    forward_tile: int = _DEFAULT_FORWARD_TILE,
    backward_tile: int = _DEFAULT_BACKWARD_TILE,
    mesh=None,
    buffer_spec=None,
):
  """Fused ``hoisted_depth_read`` (see layers/attn_res.py): same inputs,
  same outputs ``(numerators tuple, normalizers [B,T,K], maxima [B,T,K])``,
  same gradient (buffer and folded queries; the integer index has none).
  ``forward_tile``/``backward_tile`` are the token tiles of the kernels.

  On a sharded mesh pass ``mesh`` and the buffer's ``PartitionSpec``
  (``[S, B, T, D]``): Mosaic kernels cannot be auto-partitioned, so the
  call is wrapped in ``jax.shard_map`` with the block index and the folded
  queries replicated (their cotangent is summed over the mesh by the
  shard_map transpose, which is exactly the data-parallel gradient)."""
  if mesh is None:
    return _pallas_hoisted_depth_read(
        blocks_buffer, block_index, folded_queries, epsilon, forward_tile, backward_tile
    )
  if buffer_spec is None or len(buffer_spec) < 4:
    raise ValueError("buffer_spec must be the [S, B, T, D] PartitionSpec of the buffer")
  P = jax.sharding.PartitionSpec
  numerator_spec = P(*buffer_spec[1:4])
  stats_spec = P(buffer_spec[1], buffer_spec[2], None)
  sites = folded_queries.shape[-1]

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(P(*buffer_spec), P(), P()),
      out_specs=((numerator_spec,) * sites, stats_spec, stats_spec),
      check_vma=False,
  )
  def sharded(buffer, index, queries):
    return _pallas_hoisted_depth_read(buffer, index, queries, epsilon, forward_tile, backward_tile)

  return sharded(blocks_buffer, jnp.asarray(block_index, jnp.int32), folded_queries)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _pallas_hoisted_depth_read(
    blocks_buffer, block_index, folded_queries, epsilon, forward_tile, backward_tile
):
  del backward_tile
  numerators, normalizers, maxima, _ = _forward(
      blocks_buffer, block_index, folded_queries, epsilon, forward_tile
  )
  return numerators, normalizers, maxima


def _pallas_fwd(blocks_buffer, block_index, folded_queries, epsilon, forward_tile, backward_tile):
  del backward_tile
  numerators, normalizers, maxima, residuals = _forward(
      blocks_buffer, block_index, folded_queries, epsilon, forward_tile
  )
  return (numerators, normalizers, maxima), residuals


def _pallas_bwd(epsilon, forward_tile, backward_tile, residuals, cotangents):
  del epsilon, forward_tile
  buffer, index, queries_kd, raw, inv_rms, weights = residuals
  slots, tokens, dim = buffer.shape
  if tokens % backward_tile:
    raise ValueError(f"batch*length ({tokens}) must be a multiple of the tile ({backward_tile})")
  d_numerators, d_normalizers, _ = cotangents  # the maxima carry no gradient (shift invariance)
  batch, length, _ = d_normalizers.shape
  d_numerators = [d.reshape(tokens, dim) for d in d_numerators]
  d_normalizers = d_normalizers.reshape(tokens, -1).astype(jnp.float32)
  d_buffer, d_query_partials = _backward(
      buffer, index, weights, raw, inv_rms, d_numerators, d_normalizers, queries_kd,
      tile=backward_tile,
  )
  d_queries = jnp.transpose(jnp.sum(d_query_partials, axis=0))  # [D, K] fp32
  return d_buffer.reshape(slots, batch, length, dim), None, d_queries


_pallas_hoisted_depth_read.defvjp(_pallas_fwd, _pallas_bwd)
