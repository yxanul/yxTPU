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

"""Kimi Delta Attention reference and chunkwise JAX implementations."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from maxtext.common.common_types import (
  KV_BATCH,
  KV_HEAD,
  MODEL_MODE_TRAIN,
  Array,
  Config,
  DType,
)
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm, l2norm
from maxtext.utils.sharding import logical_to_mesh_axes

from yxtpu_pretrain.kernels.kda_fused_pallas import pallas_kda_fused

_TRIANGULAR_SOLVE_BLOCK_SIZE = 16

# Rewriting the short causal QKV mixer as shifted multiply-accumulates was
# tried and is slower; this knob is off by default and kept as a control.
#
# The motivation was an XPlane profile of the 272.9M hybrid charging 62.080
# ms/step (26.58%) to convolution fusion, nearly as much as both fused KDA
# kernels together, for a depthwise convolution with one group per channel.
# The rewrite is exactly equivalent, 2.4e-7 against the Flax causal
# convolution, but measured 537,292 tok/s against 560,919, or 4.2% slower.
#
# The profile category was misread. A convolution fusion node is not only the
# convolution: XLA fuses the surrounding elementwise work, here the SiLU and
# the reshapes, into it. The depthwise lowering itself is not pathological,
# and replacing it costs four full passes over a [batch, sequence, 3 * heads *
# head_dim] tensor plus a pad, which is more memory traffic than the
# convolution it removes.
_USE_SHIFTED_QKV_CONV = False


def _causal_depthwise_conv(inputs: Array, kernel: Array) -> Array:
  """Applies a causal depthwise 1-D convolution as shifted multiply-adds.

  ``inputs`` is ``[batch, sequence, channels]`` and ``kernel`` is the Flax
  convolution parameter, shaped ``[width, 1, channels]``. Left-padding by
  ``width - 1`` makes tap ``k`` a plain slice, so the whole convolution is a
  sum of ``width`` elementwise products and no convolution op is emitted.
  """
  width = kernel.shape[0]
  sequence_length = inputs.shape[1]
  padded = jnp.pad(inputs, ((0, 0), (width - 1, 0), (0, 0)))
  output = None
  for tap in range(width):
    term = padded[:, tap : tap + sequence_length] * kernel[tap, 0]
    output = term if output is None else output + term
  return output


def _solve_unit_lower_triangular_blocked(system: Array, rhs: Array) -> Array:
  """Solves a small unit-lower system with statically unrolled 16-row blocks.

  Adapted from SGLang-JAX's Apache-2.0 TPU KDA forward kernel:
  https://github.com/sgl-project/sglang-jax/blob/main/python/sgl_jax/srt/kernels/kda/kda.py
  """
  rows, _ = rhs.shape
  block_size = _TRIANGULAR_SOLVE_BLOCK_SIZE
  num_blocks = rows // block_size
  system = system.astype(jnp.float32)
  rhs = rhs.astype(jnp.float32)
  blocks = list(jnp.split(rhs, num_blocks, axis=0))

  for block_index in range(num_blocks):
    start = block_index * block_size
    end = start + block_size
    diagonal_block = system[start:end, start:end]
    solution_rows = [blocks[block_index][row] for row in range(block_size)]
    for row in range(block_size):
      if row > 0:
        correction = lax.dot_general(
            diagonal_block[row, :row][None, :],
            jnp.stack(solution_rows[:row]),
            (((1,), (0,)), ((), ())),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).squeeze(axis=0)
        solution_rows[row] = solution_rows[row] - correction

    solved_block = jnp.stack(solution_rows)
    blocks[block_index] = solved_block
    if block_index < num_blocks - 1:
      remaining_start = end
      remaining_rhs = jnp.concatenate(blocks[block_index + 1 :], axis=0)
      remaining_system = system[remaining_start:, start:end]
      remaining_rhs = remaining_rhs - lax.dot_general(
          remaining_system,
          solved_block,
          (((1,), (0,)), ((), ())),
          precision=lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )
      blocks[block_index + 1 :] = list(
          jnp.split(remaining_rhs, num_blocks - block_index - 1, axis=0)
      )

  return jnp.concatenate(blocks, axis=0)


def _solve_transposed_unit_lower_triangular_blocked(system: Array, rhs: Array) -> Array:
  """Solves ``(I + tril(system, -1)).T X = rhs`` by block back-substitution."""
  rows, _ = rhs.shape
  block_size = _TRIANGULAR_SOLVE_BLOCK_SIZE
  num_blocks = rows // block_size
  system = system.astype(jnp.float32)
  rhs = rhs.astype(jnp.float32)
  blocks = list(jnp.split(rhs, num_blocks, axis=0))

  for block_index in range(num_blocks - 1, -1, -1):
    start = block_index * block_size
    end = start + block_size
    diagonal_block = system[start:end, start:end]
    solution_rows = [blocks[block_index][row] for row in range(block_size)]
    for row in range(block_size - 1, -1, -1):
      if row < block_size - 1:
        correction = lax.dot_general(
            diagonal_block[row + 1 :, row][None, :],
            jnp.stack(solution_rows[row + 1 :]),
            (((1,), (0,)), ((), ())),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).squeeze(axis=0)
        solution_rows[row] = solution_rows[row] - correction

    solved_block = jnp.stack(solution_rows)
    blocks[block_index] = solved_block
    if block_index > 0:
      preceding_rhs = jnp.concatenate(blocks[:block_index], axis=0)
      # A.T[:start, start:end] == A[start:end, :start].T.
      preceding_system = system[start:end, :start].T
      preceding_rhs = preceding_rhs - lax.dot_general(
          preceding_system,
          solved_block,
          (((1,), (0,)), ((), ())),
          precision=lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )
      blocks[:block_index] = list(jnp.split(preceding_rhs, block_index, axis=0))

  return jnp.concatenate(blocks, axis=0)


def _blocked_triangular_solve_kernel(
    system_ref,
    rhs_ref,
    output_ref,
    *,
    transpose: bool,
):
  """Pallas kernel for a unit-lower solve or its transposed system."""
  system = system_ref[0].astype(jnp.float32)
  rhs = rhs_ref[0].astype(jnp.float32)
  if transpose:
    solution = _solve_transposed_unit_lower_triangular_blocked(system, rhs)
  else:
    solution = _solve_unit_lower_triangular_blocked(system, rhs)
  output_ref[0] = solution.astype(output_ref.dtype)


def _blocked_unit_triangular_solve_impl(
    system: Array,
    rhs: Array,
    *,
    transpose: bool,
) -> Array:
  """Runs the TPU Pallas blocked solve, with an XLA fallback off TPU."""
  if system.ndim < 3 or rhs.ndim != system.ndim:
    raise ValueError(
        f"expected [..., C, C] system and [..., C, R] rhs, got {system.shape}, {rhs.shape}"
    )
  chunk_size = system.shape[-1]
  if system.shape[-2] != chunk_size or rhs.shape[-2] != chunk_size:
    raise ValueError(f"incompatible triangular solve shapes: {system.shape}, {rhs.shape}")
  if chunk_size % _TRIANGULAR_SOLVE_BLOCK_SIZE:
    raise ValueError(
        f"triangular dimension {chunk_size} must be divisible by {_TRIANGULAR_SOLVE_BLOCK_SIZE}"
    )

  system = system.astype(jnp.float32)
  rhs = rhs.astype(jnp.float32)
  if jax.default_backend() != "tpu":
    identity = jnp.eye(chunk_size, dtype=jnp.float32)
    matrix = identity + jnp.tril(system, k=-1)
    if transpose:
      matrix = matrix.swapaxes(-1, -2)
    return jax.scipy.linalg.solve_triangular(
        matrix,
        rhs,
        lower=not transpose,
        unit_diagonal=True,
    )

  leading_shape = system.shape[:-2]
  program_count = 1
  for dimension in leading_shape:
    program_count *= dimension
  rhs_width = rhs.shape[-1]
  flat_system = system.reshape(program_count, chunk_size, chunk_size)
  flat_rhs = rhs.reshape(program_count, chunk_size, rhs_width)
  system_spec = pl.BlockSpec(
      block_shape=(1, chunk_size, chunk_size),
      index_map=lambda program: (program, 0, 0),
  )
  rhs_spec = pl.BlockSpec(
      block_shape=(1, chunk_size, rhs_width),
      index_map=lambda program: (program, 0, 0),
  )
  flat_output = pl.pallas_call(
      functools.partial(_blocked_triangular_solve_kernel, transpose=transpose),
      grid=(program_count,),
      in_specs=(system_spec, rhs_spec),
      out_specs=rhs_spec,
      out_shape=jax.ShapeDtypeStruct(flat_rhs.shape, jnp.float32),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          disable_bounds_checks=True,
      ),
  )(flat_system, flat_rhs)
  return flat_output.reshape(*leading_shape, chunk_size, rhs_width)


@jax.custom_vjp
def blocked_unit_lower_solve(system: Array, rhs: Array) -> Array:
  """Solves ``(I + tril(system, -1)) X = rhs`` with a TPU training VJP."""
  return _blocked_unit_triangular_solve_impl(system, rhs, transpose=False)


def _blocked_unit_lower_solve_fwd(system: Array, rhs: Array):
  solution = _blocked_unit_triangular_solve_impl(system, rhs, transpose=False)
  return solution, (system, solution)


def _blocked_unit_lower_solve_bwd(residuals, solution_cotangent: Array):
  system, solution = residuals
  rhs_cotangent = _blocked_unit_triangular_solve_impl(
      system,
      solution_cotangent,
      transpose=True,
  )
  system_cotangent = -jnp.matmul(
      rhs_cotangent,
      solution.swapaxes(-1, -2),
      precision=lax.Precision.HIGHEST,
  )
  # The primal kernel reads only the strict lower triangle and assumes an
  # implicit unit diagonal.
  system_cotangent = jnp.tril(system_cotangent, k=-1)
  return system_cotangent.astype(system.dtype), rhs_cotangent.astype(solution_cotangent.dtype)


blocked_unit_lower_solve.defvjp(
    _blocked_unit_lower_solve_fwd,
    _blocked_unit_lower_solve_bwd,
)


def recurrent_kda_reference(
    query: Array,
    key: Array,
    value: Array,
    log_decay: Array,
    beta: Array,
    initial_state: Array | None = None,
    use_qk_norm: bool = True,
) -> tuple[Array, Array]:
  """Sequential KDA reference used for numerical validation.

  Shapes are query/key/value `[B, T, H, D]`, log_decay `[B, T, H, D]`,
  beta `[B, T, H]`, and state `[B, H, D, D]`.
  """
  output_dtype = value.dtype
  if use_qk_norm:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  query = query.astype(jnp.float32)
  key = key.astype(jnp.float32)
  value = value.astype(jnp.float32)
  log_decay = log_decay.astype(jnp.float32)
  beta = beta.astype(jnp.float32)
  query = query * lax.rsqrt(jnp.asarray(query.shape[-1], dtype=jnp.float32))

  batch, _, heads, key_dim = key.shape
  value_dim = value.shape[-1]
  if initial_state is None:
    initial_state = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)
  else:
    initial_state = initial_state.astype(jnp.float32)

  xs = tuple(x.swapaxes(0, 1) for x in (query, key, value, log_decay, beta))

  def step(state, inputs):
    q_t, k_t, v_t, g_t, beta_t = inputs
    state = state * jnp.exp(g_t)[..., None]
    prediction = jnp.einsum("bhk,bhkv->bhv", k_t, state, precision=lax.Precision.HIGHEST)
    residual = v_t - prediction
    state = state + jnp.einsum(
        "bhk,bhv->bhkv",
        beta_t[..., None] * k_t,
        residual,
        precision=lax.Precision.HIGHEST,
    )
    output = jnp.einsum("bhk,bhkv->bhv", q_t, state, precision=lax.Precision.HIGHEST)
    return state, output

  final_state, output = lax.scan(step, initial_state, xs)
  return output.swapaxes(0, 1).astype(output_dtype), final_state


def _decayed_pairwise_dot(
    left: Array,
    right: Array,
    cumulative_log_decay: Array,
    *,
    include_diagonal: bool,
    row_block_size: int = 8,
) -> Array:
  """Computes causal decay-weighted pairwise dots as blockwise TPU matmuls.

  For a row ``i`` and column ``j`` the required term is
  ``sum_d left[i,d] * right[j,d] * exp(G[i,d] - G[j,d])``.  Within a
  small row block this can be factored around the block's final cumulative
  decay.  Valid right-hand positions then have non-positive exponents, while
  the positive exponent on the left spans only one small row block.  Besides
  being numerically safer than separately forming ``exp(G)`` and ``exp(-G)``,
  this maps the channel reduction to an MXU matmul and avoids a full
  ``[C, C, D]`` temporary.
  """
  chunk_size = left.shape[-2]
  row_block_size = min(row_block_size, chunk_size)
  if chunk_size % row_block_size:
    raise ValueError(
        f"chunk size {chunk_size} must be divisible by row block size {row_block_size}"
    )

  num_row_blocks = chunk_size // row_block_size
  positions = jnp.arange(chunk_size, dtype=jnp.int32)
  block_positions = jnp.arange(row_block_size, dtype=jnp.int32)

  def to_row_blocks(x):
    return x.reshape(*x.shape[:-2], num_row_blocks, row_block_size, x.shape[-1])

  scan_left = jnp.moveaxis(to_row_blocks(left), -3, 0)
  scan_decay = jnp.moveaxis(to_row_blocks(cumulative_log_decay), -3, 0)
  scan_blocks = jnp.arange(num_row_blocks, dtype=jnp.int32)

  def row_block(_, inputs):
    left_block, decay_block, block_index = inputs
    block_last_position = (block_index + 1) * row_block_size - 1
    anchor = decay_block[..., -1, :]

    # All unmasked right factors are <= 1 because cumulative log decay is
    # monotonically non-increasing and the anchor is the block's last row.
    valid_right = positions <= block_last_position
    right_exponent = anchor[..., None, :] - cumulative_log_decay
    # Mask before exp: jnp.where evaluates both branches, so masking afterwards
    # lets invalid future positions overflow and poison the backward pass.
    right_exponent = jnp.where(valid_right[:, None], right_exponent, -jnp.inf)
    right_factor = jnp.exp(right_exponent)
    weighted_right = right.astype(jnp.float32) * right_factor

    # The reciprocal factor spans at most row_block_size tokens.
    weighted_left = left_block.astype(jnp.float32) * jnp.exp(decay_block - anchor[..., None, :])
    values = jnp.matmul(
        weighted_left,
        weighted_right.swapaxes(-1, -2),
        precision=lax.Precision.HIGHEST,
    )

    row_positions = block_index * row_block_size + block_positions
    causal = positions[None, :] <= row_positions[:, None]
    if not include_diagonal:
      causal = positions[None, :] < row_positions[:, None]
    return None, jnp.where(causal, values, 0.0)

  _, row_blocks = lax.scan(row_block, None, (scan_left, scan_decay, scan_blocks))
  row_blocks = jnp.moveaxis(row_blocks, 0, -3)
  return row_blocks.reshape(*left.shape[:-2], chunk_size, chunk_size)


def _decayed_pairwise_dot_bwd(
    left: Array,
    right: Array,
    cumulative_log_decay: Array,
    output_cotangent: Array,
    *,
    include_diagonal: bool,
    row_block_size: int = 8,
) -> tuple[Array, Array, Array]:
  """Blockwise VJP for ``_decayed_pairwise_dot`` without a ``[C,C,K]`` tensor."""
  chunk_size = left.shape[-2]
  row_block_size = min(row_block_size, chunk_size)
  if chunk_size % row_block_size:
    raise ValueError(
        f"chunk size {chunk_size} must be divisible by row block size {row_block_size}"
    )

  num_row_blocks = chunk_size // row_block_size
  positions = jnp.arange(chunk_size, dtype=jnp.int32)
  block_positions = jnp.arange(row_block_size, dtype=jnp.int32)

  def to_row_blocks(x):
    return x.reshape(*x.shape[:-2], num_row_blocks, row_block_size, x.shape[-1])

  cotangent_blocks = output_cotangent.reshape(
      *output_cotangent.shape[:-2],
      num_row_blocks,
      row_block_size,
      chunk_size,
  )
  scan_left = jnp.moveaxis(to_row_blocks(left), -3, 0)
  scan_decay = jnp.moveaxis(to_row_blocks(cumulative_log_decay), -3, 0)
  scan_cotangent = jnp.moveaxis(cotangent_blocks, -3, 0)
  scan_blocks = jnp.arange(num_row_blocks, dtype=jnp.int32)

  def row_block(carry, inputs):
    right_cotangent, decay_cotangent_from_right = carry
    left_block, decay_block, cotangent_block, block_index = inputs
    block_last_position = (block_index + 1) * row_block_size - 1
    anchor = decay_block[..., -1, :]

    valid_right = positions <= block_last_position
    right_exponent = anchor[..., None, :] - cumulative_log_decay
    right_exponent = jnp.where(valid_right[:, None], right_exponent, -jnp.inf)
    right_factor = jnp.exp(right_exponent)
    weighted_right = right.astype(jnp.float32) * right_factor

    left_factor = jnp.exp(decay_block - anchor[..., None, :])
    weighted_left = left_block.astype(jnp.float32) * left_factor

    row_positions = block_index * row_block_size + block_positions
    causal = positions[None, :] <= row_positions[:, None]
    if not include_diagonal:
      causal = positions[None, :] < row_positions[:, None]
    cotangent_block = jnp.where(causal, cotangent_block.astype(jnp.float32), 0.0)

    weighted_left_cotangent = jnp.matmul(
        cotangent_block,
        weighted_right,
        precision=lax.Precision.HIGHEST,
    )
    weighted_right_cotangent = jnp.matmul(
        cotangent_block.swapaxes(-1, -2),
        weighted_left,
        precision=lax.Precision.HIGHEST,
    )

    left_cotangent = weighted_left_cotangent * left_factor
    right_cotangent = right_cotangent + weighted_right_cotangent * right_factor

    left_decay_product = weighted_left_cotangent * weighted_left
    right_decay_product = weighted_right_cotangent * weighted_right
    decay_cotangent_from_left = left_decay_product
    decay_cotangent_from_right = decay_cotangent_from_right - right_decay_product
    anchor_cotangent = -jnp.sum(left_decay_product, axis=-2) + jnp.sum(right_decay_product, axis=-2)
    decay_cotangent_from_left = decay_cotangent_from_left.at[..., -1, :].add(anchor_cotangent)

    return (
        right_cotangent,
        decay_cotangent_from_right,
    ), (
        left_cotangent,
        decay_cotangent_from_left,
    )

  initial_carry = (
      jnp.zeros_like(right, dtype=jnp.float32),
      jnp.zeros_like(cumulative_log_decay, dtype=jnp.float32),
  )
  (right_cotangent, decay_cotangent_from_right), (
      left_cotangent_blocks,
      decay_cotangent_from_left_blocks,
  ) = lax.scan(
      row_block,
      initial_carry,
      (scan_left, scan_decay, scan_cotangent, scan_blocks),
  )
  left_cotangent_blocks = jnp.moveaxis(left_cotangent_blocks, 0, -3)
  decay_cotangent_from_left_blocks = jnp.moveaxis(decay_cotangent_from_left_blocks, 0, -3)
  left_cotangent = left_cotangent_blocks.reshape(left.shape)
  decay_cotangent_from_left = decay_cotangent_from_left_blocks.reshape(cumulative_log_decay.shape)
  return (
      left_cotangent,
      right_cotangent,
      decay_cotangent_from_left + decay_cotangent_from_right,
  )


def _chunk_kda_impl(
    query: Array,
    key: Array,
    value: Array,
    log_decay: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: Array | None = None,
    use_qk_norm: bool = True,
    compute_dtype: DType = jnp.bfloat16,
    use_pallas_blocked_solve: bool = False,
) -> tuple[Array, Array]:
  """Chunkwise KDA using a WY representation and an inter-chunk scan."""
  output_dtype = value.dtype
  if use_qk_norm:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  log_decay = log_decay.astype(jnp.float32)
  beta = beta.astype(compute_dtype)
  query = query * lax.rsqrt(jnp.asarray(query.shape[-1], dtype=jnp.float32)).astype(compute_dtype)

  batch, sequence_length, heads, key_dim = key.shape
  value_dim = value.shape[-1]
  pad_length = (-sequence_length) % chunk_size
  if pad_length:

    def pad_time(x):
      return jnp.pad(x, ((0, 0), (0, pad_length)) + ((0, 0),) * (x.ndim - 2))

    query, key, value, log_decay, beta = map(pad_time, (query, key, value, log_decay, beta))

  num_chunks = query.shape[1] // chunk_size

  def to_chunks(x):
    return x.reshape(batch, num_chunks, chunk_size, heads, *x.shape[3:]).swapaxes(2, 3)

  query_chunks = to_chunks(query)
  key_chunks = to_chunks(key)
  value_chunks = to_chunks(value)
  decay_chunks = to_chunks(log_decay)
  beta_chunks = beta.reshape(batch, num_chunks, chunk_size, heads).swapaxes(2, 3)
  cumulative_decay = jnp.cumsum(decay_chunks, axis=-2)

  # A = (I + StrictLower(beta_i * <decayed k_i, k_j>))^-1.
  key_beta = key_chunks * beta_chunks[..., None]
  system = _decayed_pairwise_dot(
      key_beta,
      key_chunks,
      cumulative_decay,
      include_diagonal=False,
  )
  value_beta = value_chunks * beta_chunks[..., None]
  w_input = key_beta.astype(jnp.float32) * jnp.exp(cumulative_decay)
  if use_pallas_blocked_solve:
    # Solve both WY right-hand sides directly. This avoids materializing A^-1,
    # and the custom VJP uses a second blocked solve for A^-T in backward.
    combined_rhs = jnp.concatenate([value_beta.astype(jnp.float32), w_input], axis=-1)
    combined_solution = blocked_unit_lower_solve(system, combined_rhs)
    u_chunks, w_chunks = jnp.split(combined_solution, (value_dim,), axis=-1)
  else:
    identity = jnp.eye(chunk_size, dtype=jnp.float32)
    identity = jnp.broadcast_to(identity, system.shape)
    inverse = jax.scipy.linalg.solve_triangular(
        identity + system,
        identity,
        lower=True,
        unit_diagonal=True,
    )
    u_chunks = jnp.matmul(inverse, value_beta.astype(jnp.float32), precision=lax.Precision.HIGHEST)
    w_chunks = jnp.matmul(inverse, w_input, precision=lax.Precision.HIGHEST)

  intra_attention = _decayed_pairwise_dot(
      query_chunks,
      key_chunks,
      cumulative_decay,
      include_diagonal=True,
  )

  if initial_state is None:
    initial_state = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)
  else:
    initial_state = initial_state.astype(jnp.float32)

  scan_inputs = tuple(
      x.swapaxes(0, 1)
      for x in (
          w_chunks,
          u_chunks,
          query_chunks,
          key_chunks,
          cumulative_decay,
          intra_attention,
      )
  )

  def chunk_step(state, inputs):
    w, u, q, k, decay, intra = inputs
    q_with_decay = q.astype(jnp.float32) * jnp.exp(decay)
    inter_output = jnp.matmul(q_with_decay, state, precision=lax.Precision.HIGHEST)

    predicted_value = jnp.matmul(w, state, precision=lax.Precision.HIGHEST)
    corrected_value = u - predicted_value
    output = inter_output + jnp.matmul(intra, corrected_value, precision=lax.Precision.HIGHEST)

    last_decay = decay[..., -1, :]
    state = state * jnp.exp(last_decay)[..., None]
    key_decay = jnp.exp(last_decay[..., None, :] - decay)
    decayed_key = k.astype(jnp.float32) * key_decay
    state = state + jnp.matmul(
        decayed_key.swapaxes(-1, -2),
        corrected_value,
        precision=lax.Precision.HIGHEST,
    )
    return state, output

  final_state, output_chunks = lax.scan(chunk_step, initial_state, scan_inputs)
  output = output_chunks.swapaxes(0, 1).swapaxes(2, 3)
  output = output.reshape(batch, -1, heads, value_dim)[:, :sequence_length]
  return output.astype(output_dtype), final_state


def _l2norm_with_inverse(x: Array, eps: float = 1e-6) -> tuple[Array, Array]:
  inverse_norm = lax.rsqrt(jnp.sum(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + eps)
  return x.astype(jnp.float32) * inverse_norm, inverse_norm


def _l2norm_backward(output_cotangent: Array, normalized: Array, inverse_norm: Array) -> Array:
  projection = jnp.sum(output_cotangent * normalized, axis=-1, keepdims=True)
  return inverse_norm * (output_cotangent - normalized * projection)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def _chunk_kda_analytical(
    query: Array,
    key: Array,
    value: Array,
    log_decay: Array,
    beta: Array,
    initial_state: Array,
    chunk_size: int,
    use_qk_norm: bool,
    compute_dtype: DType,
) -> tuple[Array, Array]:
  return _chunk_kda_impl(
      query,
      key,
      value,
      log_decay,
      beta,
      chunk_size=chunk_size,
      initial_state=initial_state,
      use_qk_norm=use_qk_norm,
      compute_dtype=compute_dtype,
      use_pallas_blocked_solve=False,
  )


def _chunk_kda_analytical_fwd(
    query,
    key,
    value,
    log_decay,
    beta,
    initial_state,
    chunk_size,
    use_qk_norm,
    compute_dtype,
):
  output = _chunk_kda_impl(
      query,
      key,
      value,
      log_decay,
      beta,
      chunk_size=chunk_size,
      initial_state=initial_state,
      use_qk_norm=use_qk_norm,
      compute_dtype=compute_dtype,
      use_pallas_blocked_solve=False,
  )
  residual = (query, key, value, log_decay, beta, initial_state)
  return output, residual


def _chunk_kda_analytical_bwd(
    chunk_size,
    use_qk_norm,
    compute_dtype,
    residual,
    output_cotangents,
):
  """Analytical KDA reverse pass with blockwise channel-decay derivatives."""
  query_input, key_input, value_input, log_decay_input, beta_input, initial_state_input = residual
  output_cotangent, final_state_cotangent = output_cotangents
  batch, sequence_length, heads, key_dim = key_input.shape

  if use_qk_norm:
    query_normalized, query_inverse_norm = _l2norm_with_inverse(query_input)
    key_normalized, key_inverse_norm = _l2norm_with_inverse(key_input)
  else:
    query_normalized = query_input.astype(jnp.float32)
    key_normalized = key_input.astype(jnp.float32)
    query_inverse_norm = None
    key_inverse_norm = None

  scale = lax.rsqrt(jnp.asarray(key_dim, dtype=jnp.float32))
  query = (query_normalized * scale).astype(compute_dtype)
  key = key_normalized.astype(compute_dtype)
  value = value_input.astype(compute_dtype)
  log_decay = log_decay_input.astype(jnp.float32)
  beta = beta_input.astype(compute_dtype)
  initial_state = initial_state_input.astype(jnp.float32)

  pad_length = (-sequence_length) % chunk_size
  if pad_length:

    def pad_time(x):
      return jnp.pad(x, ((0, 0), (0, pad_length)) + ((0, 0),) * (x.ndim - 2))

    query, key, value, log_decay, beta = map(
        pad_time,
        (query, key, value, log_decay, beta),
    )
    output_cotangent = pad_time(output_cotangent)

  num_chunks = query.shape[1] // chunk_size

  def to_chunks(x):
    return x.reshape(batch, num_chunks, chunk_size, heads, *x.shape[3:]).swapaxes(2, 3)

  query_chunks = to_chunks(query)
  key_chunks = to_chunks(key)
  value_chunks = to_chunks(value)
  decay_chunks = to_chunks(log_decay)
  beta_chunks = beta.reshape(batch, num_chunks, chunk_size, heads).swapaxes(2, 3)
  output_cotangent_chunks = to_chunks(output_cotangent.astype(jnp.float32))
  cumulative_decay = jnp.cumsum(decay_chunks, axis=-2)

  key_beta = key_chunks.astype(jnp.float32) * beta_chunks.astype(jnp.float32)[..., None]
  value_beta = value_chunks.astype(jnp.float32) * beta_chunks.astype(jnp.float32)[..., None]
  system = _decayed_pairwise_dot(
      key_beta,
      key_chunks,
      cumulative_decay,
      include_diagonal=False,
  )
  identity = jnp.broadcast_to(
      jnp.eye(chunk_size, dtype=jnp.float32),
      system.shape,
  )
  inverse = jax.scipy.linalg.solve_triangular(
      identity + system,
      identity,
      lower=True,
      unit_diagonal=True,
  )
  cumulative_decay_exp = jnp.exp(cumulative_decay)
  w_input = key_beta * cumulative_decay_exp
  u_chunks = jnp.matmul(
      inverse,
      value_beta,
      precision=lax.Precision.HIGHEST,
  )
  w_chunks = jnp.matmul(
      inverse,
      w_input,
      precision=lax.Precision.HIGHEST,
  )
  intra_attention = _decayed_pairwise_dot(
      query_chunks,
      key_chunks,
      cumulative_decay,
      include_diagonal=True,
  )

  end_decay = cumulative_decay[..., -1, :]
  end_decay_exp = jnp.exp(end_decay)
  state_decay_exp = jnp.exp(end_decay[..., None, :] - cumulative_decay)
  query_with_decay = query_chunks.astype(jnp.float32) * cumulative_decay_exp
  key_for_state = key_chunks.astype(jnp.float32) * state_decay_exp

  forward_scan_inputs = tuple(
      jnp.moveaxis(x, 1, 0)
      for x in (
          w_chunks,
          u_chunks,
          key_for_state,
          end_decay_exp,
      )
  )

  def reconstruct_state(state, inputs):
    w, u, state_key, decay_end = inputs
    predicted_value = jnp.matmul(w, state, precision=lax.Precision.HIGHEST)
    corrected_value = u - predicted_value
    next_state = state * decay_end[..., None] + jnp.matmul(
        state_key.swapaxes(-1, -2),
        corrected_value,
        precision=lax.Precision.HIGHEST,
    )
    return next_state, state

  _, state_before_chunks = lax.scan(
      reconstruct_state,
      initial_state,
      forward_scan_inputs,
  )

  reverse_scan_inputs = (
      state_before_chunks,
      *(
          jnp.moveaxis(x, 1, 0)
          for x in (
              query_with_decay,
              key_for_state,
              u_chunks,
              w_chunks,
              intra_attention,
              end_decay_exp,
              output_cotangent_chunks,
          )
      ),
  )

  def reverse_chunk(state_cotangent_next, inputs):
    (
        state,
        query_decay,
        state_key,
        u,
        w,
        intra,
        decay_end,
        output_bar,
    ) = inputs
    predicted_value = jnp.matmul(w, state, precision=lax.Precision.HIGHEST)
    corrected_value = u - predicted_value

    state_cotangent = state_cotangent_next * decay_end[..., None]
    decay_end_cotangent = jnp.sum(state_cotangent_next * state, axis=-1)

    state_key_cotangent = jnp.matmul(
        corrected_value,
        state_cotangent_next.swapaxes(-1, -2),
        precision=lax.Precision.HIGHEST,
    )
    corrected_value_cotangent = jnp.matmul(
        state_key,
        state_cotangent_next,
        precision=lax.Precision.HIGHEST,
    )

    intra_cotangent = jnp.matmul(
        output_bar,
        corrected_value.swapaxes(-1, -2),
        precision=lax.Precision.HIGHEST,
    )
    corrected_value_cotangent = corrected_value_cotangent + jnp.matmul(
        intra.swapaxes(-1, -2),
        output_bar,
        precision=lax.Precision.HIGHEST,
    )

    query_decay_cotangent = jnp.matmul(
        output_bar,
        state.swapaxes(-1, -2),
        precision=lax.Precision.HIGHEST,
    )
    state_cotangent = state_cotangent + jnp.matmul(
        query_decay.swapaxes(-1, -2),
        output_bar,
        precision=lax.Precision.HIGHEST,
    )

    u_cotangent = corrected_value_cotangent
    w_cotangent = -jnp.matmul(
        corrected_value_cotangent,
        state.swapaxes(-1, -2),
        precision=lax.Precision.HIGHEST,
    )
    state_cotangent = state_cotangent - jnp.matmul(
        w.swapaxes(-1, -2),
        corrected_value_cotangent,
        precision=lax.Precision.HIGHEST,
    )

    return state_cotangent, (
        query_decay_cotangent,
        state_key_cotangent,
        u_cotangent,
        w_cotangent,
        intra_cotangent,
        decay_end_cotangent,
    )

  initial_state_cotangent, reverse_outputs = lax.scan(
      reverse_chunk,
      final_state_cotangent.astype(jnp.float32),
      reverse_scan_inputs,
      reverse=True,
  )
  (
      query_decay_cotangent,
      state_key_cotangent,
      u_cotangent,
      w_cotangent,
      intra_cotangent,
      end_decay_exp_cotangent,
  ) = (jnp.moveaxis(x, 0, 1) for x in reverse_outputs)

  query_chunks_cotangent = query_decay_cotangent * cumulative_decay_exp
  cumulative_decay_cotangent = query_decay_cotangent * query_with_decay

  key_chunks_cotangent = state_key_cotangent * state_decay_exp
  state_decay_cotangent = state_key_cotangent * key_for_state
  cumulative_decay_cotangent = cumulative_decay_cotangent - state_decay_cotangent
  end_decay_cotangent = jnp.sum(state_decay_cotangent, axis=-2)
  end_decay_cotangent = end_decay_cotangent + end_decay_exp_cotangent * end_decay_exp

  inverse_cotangent = jnp.matmul(
      u_cotangent,
      value_beta.swapaxes(-1, -2),
      precision=lax.Precision.HIGHEST,
  ) + jnp.matmul(
      w_cotangent,
      w_input.swapaxes(-1, -2),
      precision=lax.Precision.HIGHEST,
  )
  value_beta_cotangent = jnp.matmul(
      inverse.swapaxes(-1, -2),
      u_cotangent,
      precision=lax.Precision.HIGHEST,
  )
  w_input_cotangent = jnp.matmul(
      inverse.swapaxes(-1, -2),
      w_cotangent,
      precision=lax.Precision.HIGHEST,
  )
  key_beta_cotangent = w_input_cotangent * cumulative_decay_exp
  cumulative_decay_cotangent = cumulative_decay_cotangent + w_input_cotangent * w_input

  inverse_transpose = inverse.swapaxes(-1, -2)
  system_cotangent = -jnp.matmul(
      jnp.matmul(
          inverse_transpose,
          inverse_cotangent,
          precision=lax.Precision.HIGHEST,
      ),
      inverse_transpose,
      precision=lax.Precision.HIGHEST,
  )
  system_cotangent = jnp.tril(system_cotangent, k=-1)

  (
      key_beta_system_cotangent,
      key_system_cotangent,
      system_decay_cotangent,
  ) = _decayed_pairwise_dot_bwd(
      key_beta,
      key_chunks,
      cumulative_decay,
      system_cotangent,
      include_diagonal=False,
  )
  (
      query_pairwise_cotangent,
      key_intra_cotangent,
      intra_decay_cotangent,
  ) = _decayed_pairwise_dot_bwd(
      query_chunks,
      key_chunks,
      cumulative_decay,
      intra_cotangent,
      include_diagonal=True,
  )
  key_beta_cotangent = key_beta_cotangent + key_beta_system_cotangent
  key_chunks_cotangent = key_chunks_cotangent + key_system_cotangent + key_intra_cotangent
  query_chunks_cotangent = query_chunks_cotangent + query_pairwise_cotangent
  cumulative_decay_cotangent = (
      cumulative_decay_cotangent + system_decay_cotangent + intra_decay_cotangent
  )
  cumulative_decay_cotangent = cumulative_decay_cotangent.at[..., -1, :].add(end_decay_cotangent)

  beta_chunks_f32 = beta_chunks.astype(jnp.float32)
  value_chunks_cotangent = value_beta_cotangent * beta_chunks_f32[..., None]
  beta_chunks_cotangent = jnp.sum(
      value_beta_cotangent * value_chunks.astype(jnp.float32),
      axis=-1,
  )
  key_chunks_cotangent = key_chunks_cotangent + key_beta_cotangent * beta_chunks_f32[..., None]
  beta_chunks_cotangent = beta_chunks_cotangent + jnp.sum(
      key_beta_cotangent * key_chunks.astype(jnp.float32),
      axis=-1,
  )

  decay_chunks_cotangent = jnp.flip(
      jnp.cumsum(
          jnp.flip(cumulative_decay_cotangent, axis=-2),
          axis=-2,
      ),
      axis=-2,
  )

  def from_chunks(x):
    return x.swapaxes(2, 3).reshape(batch, -1, heads, *x.shape[4:])

  query_cotangent = from_chunks(query_chunks_cotangent)[:, :sequence_length]
  key_cotangent = from_chunks(key_chunks_cotangent)[:, :sequence_length]
  value_cotangent = from_chunks(value_chunks_cotangent)[:, :sequence_length]
  log_decay_cotangent = from_chunks(decay_chunks_cotangent)[:, :sequence_length]
  beta_cotangent = beta_chunks_cotangent.swapaxes(2, 3).reshape(batch, -1, heads)[
      :, :sequence_length
  ]

  query_normalized_cotangent = query_cotangent * scale
  if use_qk_norm:
    query_cotangent = _l2norm_backward(
        query_normalized_cotangent,
        query_normalized,
        query_inverse_norm,
    )
    key_cotangent = _l2norm_backward(
        key_cotangent,
        key_normalized,
        key_inverse_norm,
    )
  else:
    query_cotangent = query_normalized_cotangent

  return (
      query_cotangent.astype(query_input.dtype),
      key_cotangent.astype(key_input.dtype),
      value_cotangent.astype(value_input.dtype),
      log_decay_cotangent.astype(log_decay_input.dtype),
      beta_cotangent.astype(beta_input.dtype),
      initial_state_cotangent.astype(initial_state_input.dtype),
  )


_chunk_kda_analytical.defvjp(
    _chunk_kda_analytical_fwd,
    _chunk_kda_analytical_bwd,
)


def chunk_kda(
    query: Array,
    key: Array,
    value: Array,
    log_decay: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: Array | None = None,
    use_qk_norm: bool = True,
    compute_dtype: DType = jnp.bfloat16,
    use_pallas_blocked_solve: bool = False,
    use_analytical_custom_vjp: bool = False,
) -> tuple[Array, Array]:
  """Chunkwise KDA with selectable XLA-autodiff or analytical-VJP training."""
  if initial_state is None:
    initial_state = jnp.zeros(
        (query.shape[0], query.shape[2], key.shape[-1], value.shape[-1]),
        dtype=compute_dtype,
    )
  if use_analytical_custom_vjp:
    if use_pallas_blocked_solve:
      raise ValueError("the analytical KDA VJP currently requires the XLA inverse path")
    return _chunk_kda_analytical(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
        chunk_size,
        use_qk_norm,
        compute_dtype,
    )
  return _chunk_kda_impl(
      query,
      key,
      value,
      log_decay,
      beta,
      chunk_size=chunk_size,
      initial_state=initial_state,
      use_qk_norm=use_qk_norm,
      compute_dtype=compute_dtype,
      use_pallas_blocked_solve=use_pallas_blocked_solve,
  )


class KimiDeltaAttention(nnx.Module):
  """Dense KDA token mixer for MaxText's Qwen3-Next hybrid container."""

  def __init__(
      self,
      config: Config,
      mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs

    if config.gdn_num_key_heads != config.gdn_num_value_heads:
      raise ValueError(
          "The TPU KDA training prototype currently requires equal key and value head counts."
      )

    self.num_heads = config.gdn_num_key_heads
    self.head_dim = config.gdn_key_head_dim
    if self.head_dim != config.gdn_value_head_dim:
      raise ValueError(
          "The TPU KDA training prototype currently requires equal key and value head dimensions."
      )
    self.projection_dim = self.num_heads * self.head_dim
    self.gate_rank = config.kda_gate_rank

    self.in_proj_qkv = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(3, self.num_heads, self.head_dim),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", "qkv", "gdn_head", None),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.conv1d = nnx.Conv(
        in_features=3 * self.projection_dim,
        out_features=3 * self.projection_dim,
        kernel_size=(config.gdn_conv_kernel_dim,),
        feature_group_count=3 * self.projection_dim,
        padding="CAUSAL",
        use_bias=False,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        precision=config.matmul_precision,
        rngs=rngs,
    )

    self.decay_down = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.gate_rank,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", None),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.decay_up = DenseGeneral(
        in_features_shape=self.gate_rank,
        out_features_shape=(self.num_heads, self.head_dim),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=(None, "gdn_head", None),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.beta_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.num_heads,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", "gdn_head"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.output_gate_down = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.gate_rank,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", None),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.output_gate_up = DenseGeneral(
        in_features_shape=self.gate_rank,
        out_features_shape=(self.num_heads, self.head_dim),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=(None, "gdn_head", None),
        matmul_precision=config.matmul_precision,
        use_bias=True,
        rngs=rngs,
    )

    def a_log_init(key, shape, dtype=jnp.float32):
      values = jax.random.uniform(key, shape, dtype=dtype, minval=1e-9, maxval=16.0)
      return jnp.log(values)

    def dt_bias_init(key, shape, dtype=jnp.float32):
      log_dt = jax.random.uniform(
          key,
          shape,
          dtype=dtype,
          minval=jnp.log(jnp.asarray(0.001, dtype=dtype)),
          maxval=jnp.log(jnp.asarray(0.1, dtype=dtype)),
      )
      dt = jnp.exp(log_dt).clip(min=1e-4)
      return dt + jnp.log(-jnp.expm1(-dt))

    self.A_log = nnx.Param(a_log_init(rngs.params(), (self.num_heads,), config.weight_dtype))
    self.dt_bias = nnx.Param(
        dt_bias_init(rngs.params(), (self.num_heads, self.head_dim), config.weight_dtype)
    )
    self.output_norm = RMSNorm(
        num_features=self.head_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.out_proj = DenseGeneral(
        in_features_shape=(self.num_heads, self.head_dim),
        out_features_shape=config.emb_dim,
        axis=(-2, -1),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("gdn_head", None, "embed"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      decoder_segment_ids: Array | None = None,
      model_mode: str = MODEL_MODE_TRAIN,
      **unused_kwargs,
  ) -> tuple[Array, None]:
    if model_mode != MODEL_MODE_TRAIN:
      raise NotImplementedError("The TPU KDA prototype currently supports training only.")

    batch, sequence_length, _ = hidden_states.shape
    qkv = self.in_proj_qkv(hidden_states)
    qkv = qkv.transpose(0, 1, 3, 2, 4).reshape(batch, sequence_length, -1)
    if _USE_SHIFTED_QKV_CONV:
      qkv = _causal_depthwise_conv(
          qkv,
          self.conv1d.kernel.value.astype(qkv.dtype),
      )
    else:
      qkv = jnp.pad(qkv, ((0, 0), (self.config.gdn_conv_kernel_dim - 1, 0), (0, 0)))
      qkv = self.conv1d(qkv)[:, -sequence_length:]
    qkv = jax.nn.silu(qkv.astype(jnp.float32)).astype(self.config.dtype)
    qkv = qkv.reshape(batch, sequence_length, self.num_heads, 3, self.head_dim)
    query, key, value = (qkv[..., i, :] for i in range(3))

    decay_input = self.decay_up(self.decay_down(hidden_states))
    raw_decay = decay_input.astype(jnp.float32) + jnp.asarray(self.dt_bias[...], dtype=jnp.float32)
    decay_rate = jnp.exp(jnp.asarray(self.A_log[...], dtype=jnp.float32))[None, None, :, None]
    if self.config.kda_safe_gate:
      # This is FLA KDA's safe-gate parameterization.  Bounding each log
      # decay to [lower_bound, 0) permits factored block matmuls without
      # changing the recurrent state equation.
      log_decay = self.config.kda_gate_lower_bound * jax.nn.sigmoid(decay_rate * raw_decay)
    else:
      log_decay = -decay_rate * jax.nn.softplus(raw_decay)
    beta = jax.nn.sigmoid(self.beta_proj(hidden_states).astype(jnp.float32))

    if decoder_segment_ids is not None:
      valid = decoder_segment_ids != 0
      query = jnp.where(valid[..., None, None], query, 0)
      key = jnp.where(valid[..., None, None], key, 0)
      value = jnp.where(valid[..., None, None], value, 0)
      log_decay = jnp.where(valid[..., None, None], log_decay, 0)
      beta = jnp.where(valid[..., None], beta, 0)

    initial_state = jnp.zeros(
        (batch, self.num_heads, self.head_dim, self.head_dim),
        dtype=jnp.float32,
    )
    qkv_spec = logical_to_mesh_axes(
        (KV_BATCH, None, KV_HEAD, None),
        mesh=self.mesh,
        rules=self.config.logical_axis_rules,
    )
    beta_spec = logical_to_mesh_axes(
        (KV_BATCH, None, KV_HEAD),
        mesh=self.mesh,
        rules=self.config.logical_axis_rules,
    )
    state_spec = logical_to_mesh_axes(
        (KV_BATCH, KV_HEAD, None, None),
        mesh=self.mesh,
        rules=self.config.logical_axis_rules,
    )

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(qkv_spec, qkv_spec, qkv_spec, qkv_spec, beta_spec, state_spec),
        out_specs=(qkv_spec, state_spec),
        check_vma=False,
    )
    def sharded_kda(q, k, v, g, b, state):
      if self.config.kda_use_fused_pallas_kernel:
        return pallas_kda_fused(q, k, v, g, b, state)
      kda_impl = functools.partial(
          chunk_kda,
          chunk_size=self.config.gdn_chunk_size,
          use_qk_norm=self.config.use_qk_norm_in_gdn,
          compute_dtype=(
              jnp.float32
              if self.config.kda_precision == "full_fp32"
              else self.config.dtype
          ),
          use_pallas_blocked_solve=self.config.kda_use_pallas_blocked_solve,
          use_analytical_custom_vjp=self.config.kda_use_analytical_custom_vjp,
      )
      # KDA's channel-wise decay makes the autodiff tape of the two causal
      # pairwise scans substantially larger than GDN's scalar-decay tape.
      # Recompute the chunk recurrence in the backward pass instead of keeping
      # those scan intermediates resident for every hybrid layer.
      return jax.checkpoint(kda_impl)(q, k, v, g, b, initial_state=state)

    output, _ = sharded_kda(query, key, value, log_decay, beta, initial_state)
    output_gate = self.output_gate_up(self.output_gate_down(hidden_states))
    output = self.output_norm(output) * jax.nn.sigmoid(output_gate.astype(jnp.float32))
    output = self.out_proj(output.astype(self.config.dtype))
    return output, None
