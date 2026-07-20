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

"""Experimental fused TPU Pallas kernel for Kimi Delta Attention.

The production path assigns one ordered chunk stream to each ``(batch, head)``
pair. A ``K x V`` FP32 fast-weight state remains in VMEM while the ordered grid
walks through the sequence. Each invocation consumes one BF16 Q/K/V chunk,
recomputes compact intra-chunk quantities, emits BF16 output, and stores only
the FP32 state after that chunk for a future custom backward.
"""

from __future__ import annotations

import functools
import math

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


_SOLVE_BLOCK_SIZE = 16
_PAIRWISE_ROW_BLOCK_SIZE = 8


def _matmul(left: jax.Array, right: jax.Array) -> jax.Array:
  return lax.dot_general(
      left,
      right,
      (((1,), (0,)), ((), ())),
      precision=lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )


def _inclusive_cumsum(values: jax.Array) -> jax.Array:
  """Static power-of-two inclusive cumsum suitable for a Pallas TPU kernel."""
  length = values.shape[0]
  if length & (length - 1):
    raise ValueError(f"cumsum length must be a power of two, got {length}")
  result = values.astype(jnp.float32)
  for depth in range(int(math.log2(length))):
    stride = 1 << depth
    result = jnp.concatenate(
        (
            result[:stride],
            result[stride:] + result[:-stride],
        ),
        axis=0,
    )
  return result


def _reverse_inclusive_cumsum(values: jax.Array) -> jax.Array:
  """Static power-of-two reverse cumsum without a Mosaic ``rev`` primitive."""
  length = values.shape[0]
  if length & (length - 1):
    raise ValueError(f"cumsum length must be a power of two, got {length}")
  result = values.astype(jnp.float32)
  for depth in range(int(math.log2(length))):
    stride = 1 << depth
    result = jnp.concatenate(
        (
            result[:-stride] + result[stride:],
            result[-stride:],
        ),
        axis=0,
    )
  return result


def _l2_normalize(values: jax.Array, *, scale: float = 1.0) -> jax.Array:
  values = values.astype(jnp.float32)
  inverse_norm = lax.rsqrt(jnp.sum(values * values, axis=-1, keepdims=True) + 1e-6)
  return values * inverse_norm * scale


def _l2_normalize_with_inverse(values: jax.Array) -> tuple[jax.Array, jax.Array]:
  values = values.astype(jnp.float32)
  inverse_norm = lax.rsqrt(jnp.sum(values * values, axis=-1, keepdims=True) + 1e-6)
  return values * inverse_norm, inverse_norm


def _l2_normalize_backward(
    output_cotangent: jax.Array,
    normalized: jax.Array,
    inverse_norm: jax.Array,
) -> jax.Array:
  projection = jnp.sum(output_cotangent * normalized, axis=-1, keepdims=True)
  return (output_cotangent - normalized * projection) * inverse_norm


def _decayed_pairwise(
    left: jax.Array,
    right: jax.Array,
    cumulative_log_decay: jax.Array,
    *,
    include_diagonal: bool,
) -> jax.Array:
  """Forms a causal channel-decayed dot matrix with eight-row MXU matmuls."""
  chunk_size, _ = left.shape
  row_block_size = _PAIRWISE_ROW_BLOCK_SIZE
  if chunk_size % row_block_size:
    raise ValueError(
        f"chunk size {chunk_size} must be divisible by row block size {row_block_size}"
    )

  row_blocks = []
  for block_index in range(chunk_size // row_block_size):
    row_start = block_index * row_block_size
    row_end = row_start + row_block_size
    left_block = left[row_start:row_end].astype(jnp.float32)
    decay_block = cumulative_log_decay[row_start:row_end]
    anchor = decay_block[-1]

    right_exponent = anchor[None, :] - cumulative_log_decay[:row_end]
    if row_end < chunk_size:
      right_exponent = jnp.concatenate(
          (
              right_exponent,
              jnp.full(
                  (chunk_size - row_end, cumulative_log_decay.shape[-1]),
                  -jnp.inf,
                  dtype=jnp.float32,
              ),
          ),
          axis=0,
      )
    weighted_right = right.astype(jnp.float32) * jnp.exp(right_exponent)
    weighted_left = left_block * jnp.exp(decay_block - anchor[None, :])
    row_blocks.append(_matmul(weighted_left, weighted_right.T))

  values = jnp.concatenate(row_blocks, axis=0)
  if include_diagonal:
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  else:
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  return values * causal


def _decayed_pairwise_backward(
    left: jax.Array,
    right: jax.Array,
    cumulative_log_decay: jax.Array,
    output_cotangent: jax.Array,
    *,
    include_diagonal: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Blockwise VJP for ``_decayed_pairwise`` inside a Pallas program."""
  chunk_size, channel_dim = left.shape
  row_block_size = _PAIRWISE_ROW_BLOCK_SIZE
  if include_diagonal:
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  else:
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  output_cotangent = output_cotangent.astype(jnp.float32) * causal

  right_cotangent = jnp.zeros((chunk_size, channel_dim), dtype=jnp.float32)
  decay_cotangent_from_right = jnp.zeros(
      (chunk_size, channel_dim),
      dtype=jnp.float32,
  )
  left_cotangent_blocks = []
  decay_cotangent_from_left_blocks = []
  for block_index in range(chunk_size // row_block_size):
    row_start = block_index * row_block_size
    row_end = row_start + row_block_size
    left_block = left[row_start:row_end].astype(jnp.float32)
    decay_block = cumulative_log_decay[row_start:row_end]
    anchor = decay_block[-1]

    right_exponent = anchor[None, :] - cumulative_log_decay[:row_end]
    if row_end < chunk_size:
      right_exponent = jnp.concatenate(
          (
              right_exponent,
              jnp.full(
                  (chunk_size - row_end, channel_dim),
                  -jnp.inf,
                  dtype=jnp.float32,
              ),
          ),
          axis=0,
      )
    right_factor = jnp.exp(right_exponent)
    weighted_right = right.astype(jnp.float32) * right_factor
    left_factor = jnp.exp(decay_block - anchor[None, :])
    weighted_left = left_block * left_factor
    cotangent_block = output_cotangent[row_start:row_end]

    weighted_left_cotangent = _matmul(cotangent_block, weighted_right)
    weighted_right_cotangent = _matmul(cotangent_block.T, weighted_left)
    left_cotangent_blocks.append(weighted_left_cotangent * left_factor)
    right_cotangent = right_cotangent + weighted_right_cotangent * right_factor

    left_decay_product = weighted_left_cotangent * weighted_left
    right_decay_product = weighted_right_cotangent * weighted_right
    decay_cotangent_from_right = decay_cotangent_from_right - right_decay_product
    anchor_cotangent = -jnp.sum(left_decay_product, axis=0) + jnp.sum(
        right_decay_product,
        axis=0,
    )
    left_decay_product = jnp.concatenate(
        (
            left_decay_product[:-1],
            left_decay_product[-1:] + anchor_cotangent[None, :],
        ),
        axis=0,
    )
    decay_cotangent_from_left_blocks.append(left_decay_product)

  return (
      jnp.concatenate(left_cotangent_blocks, axis=0),
      right_cotangent,
      jnp.concatenate(decay_cotangent_from_left_blocks, axis=0) + decay_cotangent_from_right,
  )


def _solve_unit_lower_triangular(system: jax.Array, rhs: jax.Array) -> jax.Array:
  """Blocked forward substitution for a 64-row unit-lower system."""
  rows, _ = rhs.shape
  block_size = _SOLVE_BLOCK_SIZE
  if rows % block_size:
    raise ValueError(f"triangular dimension {rows} must be divisible by {block_size}")

  system = system.astype(jnp.float32)
  blocks = list(jnp.split(rhs.astype(jnp.float32), rows // block_size, axis=0))
  for block_index in range(rows // block_size):
    start = block_index * block_size
    end = start + block_size
    diagonal = system[start:end, start:end]
    solution_rows = [blocks[block_index][row] for row in range(block_size)]
    for row in range(block_size):
      if row:
        correction = _matmul(
            diagonal[row, :row][None, :],
            jnp.stack(solution_rows[:row]),
        )[0]
        solution_rows[row] = solution_rows[row] - correction

    solved = jnp.stack(solution_rows)
    blocks[block_index] = solved
    if block_index + 1 < rows // block_size:
      remaining = jnp.concatenate(blocks[block_index + 1 :], axis=0)
      remaining = remaining - _matmul(system[end:, start:end], solved)
      blocks[block_index + 1 :] = list(
          jnp.split(remaining, rows // block_size - block_index - 1, axis=0)
      )
  return jnp.concatenate(blocks, axis=0)


def _solve_unit_lower_triangular_doubling(
    system: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Exact nilpotent-series solve using logarithmic-depth MXU matmuls.

  For ``A = I + L`` with strictly lower ``L``, ``P = -L`` is nilpotent and
  ``A^-1 B = (I + P + ... + P^(C-1)) B``. Recursive doubling forms that
  finite series in ``log2(C)`` stages. It performs more FLOPs than forward
  substitution but exposes them as dense matmuls instead of serial row
  dependencies, which is a better candidate for TPU execution.
  """
  power = -jnp.tril(system.astype(jnp.float32), k=-1)
  return _nilpotent_series_solve(power, rhs)


def _nilpotent_series_solve(power: jax.Array, rhs: jax.Array) -> jax.Array:
  """Applies ``(I - power)^-1`` when ``power`` is strictly triangular."""
  rows, _ = rhs.shape
  if rows & (rows - 1):
    raise ValueError(f"triangular dimension must be a power of two, got {rows}")
  rhs = rhs.astype(jnp.float32)
  solution = rhs + _matmul(power, rhs)
  power = _matmul(power, power)
  covered_terms = 2
  while covered_terms < rows:
    solution = solution + _matmul(power, solution)
    power = _matmul(power, power)
    covered_terms *= 2
  return solution


def _solve_transposed_unit_lower_triangular_doubling(
    system: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Solves ``(I + tril(system, -1)).T X = rhs`` by recursive doubling."""
  power = -jnp.triu(system.astype(jnp.float32).T, k=1)
  return _nilpotent_series_solve(power, rhs)


def _kda_fused_forward_kernel(
    query_ref,
    key_ref,
    value_ref,
    log_decay_ref,
    beta_ref,
    initial_state_ref,
    output_ref,
    state_after_ref,
    state_scratch_ref,
    *,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
    use_qk_norm: bool,
    solve_method: str,
    profile_stage: str,
):
  """Consumes one chunk while retaining the fast-weight state in VMEM."""
  chunk_index = pl.program_id(2)
  query = query_ref[0, 0, :, :]
  key = key_ref[0, 0, :, :]
  value = value_ref[0, 0, :, :].astype(jnp.float32)
  log_decay = log_decay_ref[0, 0, :, :].astype(jnp.float32)
  beta = beta_ref[0, 0, :, 0].astype(jnp.float32)

  @pl.when(chunk_index == 0)
  def _initialize_state():
    state_scratch_ref[...] = initial_state_ref[0, 0].astype(jnp.float32)

  if use_qk_norm:
    query = _l2_normalize(query, scale=1.0 / math.sqrt(key_dim))
    key = _l2_normalize(key)
  else:
    query = query.astype(jnp.float32) * (1.0 / math.sqrt(key_dim))
    key = key.astype(jnp.float32)

  cumulative_decay = _inclusive_cumsum(log_decay)
  if profile_stage == "preprocess":
    diagnostic = query + key + 1e-3 * cumulative_decay
    output_ref[0, 0, :, :] = diagnostic.astype(output_ref.dtype)
    state_after_ref[0, 0, 0, :, :] = state_scratch_ref[...].astype(jnp.float32)
    return

  key_beta = key * beta[:, None]
  system = _decayed_pairwise(
      key_beta,
      key,
      cumulative_decay,
      include_diagonal=False,
  )
  intra = _decayed_pairwise(
      query,
      key,
      cumulative_decay,
      include_diagonal=True,
  )
  if profile_stage == "pairwise":
    output_ref[0, 0, :, :] = jnp.concatenate((system, intra), axis=-1).astype(output_ref.dtype)
    state_after_ref[0, 0, 0, :, :] = state_scratch_ref[...].astype(jnp.float32)
    return

  value_beta = value * beta[:, None]
  w_input = key_beta * jnp.exp(cumulative_decay)
  combined_rhs = jnp.concatenate((value_beta, w_input), axis=-1)
  if solve_method == "blocked":
    solved = _solve_unit_lower_triangular(system, combined_rhs)
  elif solve_method == "doubling":
    solved = _solve_unit_lower_triangular_doubling(system, combined_rhs)
  else:
    raise ValueError(f"unknown solve method: {solve_method}")
  u = solved[:, :value_dim]
  w = solved[:, value_dim : value_dim + key_dim]
  if profile_stage == "solve":
    output_ref[0, 0, :, :] = (u + w).astype(output_ref.dtype)
    state_after_ref[0, 0, 0, :, :] = state_scratch_ref[...].astype(jnp.float32)
    return

  state = state_scratch_ref[...].astype(jnp.float32)
  query_with_decay = query * jnp.exp(cumulative_decay)
  inter_output = _matmul(query_with_decay, state)
  corrected_value = u - _matmul(w, state)
  output = inter_output + _matmul(intra, corrected_value)

  final_decay = cumulative_decay[-1]
  state = state * jnp.exp(final_decay)[:, None]
  key_for_state = key * jnp.exp(final_decay[None, :] - cumulative_decay)
  state = state + _matmul(key_for_state.T, corrected_value)
  state_scratch_ref[...] = state

  output_ref[0, 0, :, :] = output.astype(output_ref.dtype)
  state_after_ref[0, 0, 0, :, :] = state


def _kda_fused_backward_kernel(
    query_ref,
    key_ref,
    value_ref,
    log_decay_ref,
    beta_ref,
    initial_state_ref,
    previous_state_after_ref,
    output_cotangent_ref,
    final_state_cotangent_ref,
    query_cotangent_ref,
    key_cotangent_ref,
    value_cotangent_ref,
    log_decay_cotangent_ref,
    beta_cotangent_ref,
    state_before_cotangent_ref,
    state_cotangent_scratch_ref,
    *,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
    num_chunks: int,
    use_qk_norm: bool,
    profile_stage: str,
):
  """Recomputes one chunk and carries the state cotangent in reverse order."""
  reverse_chunk_index = pl.program_id(2)
  chunk_index = num_chunks - 1 - reverse_chunk_index
  query_input = query_ref[0, 0, :, :]
  key_input = key_ref[0, 0, :, :]
  value = value_ref[0, 0, :, :].astype(jnp.float32)
  log_decay = log_decay_ref[0, 0, :, :].astype(jnp.float32)
  beta = beta_ref[0, 0, :, 0].astype(jnp.float32)
  output_cotangent = output_cotangent_ref[0, 0, :, :].astype(jnp.float32)

  @pl.when(reverse_chunk_index == 0)
  def _initialize_state_cotangent():
    state_cotangent_scratch_ref[...] = final_state_cotangent_ref[0, 0].astype(jnp.float32)

  state = lax.cond(
      chunk_index == 0,
      lambda: initial_state_ref[0, 0].astype(jnp.float32),
      lambda: previous_state_after_ref[0, 0, 0].astype(jnp.float32),
  )

  if use_qk_norm:
    query_normalized, query_inverse_norm = _l2_normalize_with_inverse(query_input)
    key, key_inverse_norm = _l2_normalize_with_inverse(key_input)
    query = query_normalized * (1.0 / math.sqrt(key_dim))
  else:
    query_normalized = query_input.astype(jnp.float32)
    key = key_input.astype(jnp.float32)
    query_inverse_norm = jnp.ones((chunk_size, 1), dtype=jnp.float32)
    key_inverse_norm = jnp.ones((chunk_size, 1), dtype=jnp.float32)
    query = query_normalized * (1.0 / math.sqrt(key_dim))

  cumulative_decay = _inclusive_cumsum(log_decay)
  cumulative_decay_exp = jnp.exp(cumulative_decay)
  key_beta = key * beta[:, None]
  value_beta = value * beta[:, None]
  system = _decayed_pairwise(
      key_beta,
      key,
      cumulative_decay,
      include_diagonal=False,
  )
  intra = _decayed_pairwise(
      query,
      key,
      cumulative_decay,
      include_diagonal=True,
  )
  w_input = key_beta * cumulative_decay_exp
  solved = _solve_unit_lower_triangular_doubling(
      system,
      jnp.concatenate((value_beta, w_input), axis=-1),
  )
  u = solved[:, :value_dim]
  w = solved[:, value_dim : value_dim + key_dim]

  final_decay = cumulative_decay[-1]
  final_decay_exp = jnp.exp(final_decay)
  state_decay_exp = jnp.exp(final_decay[None, :] - cumulative_decay)
  query_with_decay = query * cumulative_decay_exp
  key_for_state = key * state_decay_exp
  corrected_value = u - _matmul(w, state)

  state_cotangent_next = state_cotangent_scratch_ref[...].astype(jnp.float32)
  state_cotangent = state_cotangent_next * final_decay_exp[:, None]
  final_decay_exp_cotangent = jnp.sum(state_cotangent_next * state, axis=-1)

  key_for_state_cotangent = _matmul(corrected_value, state_cotangent_next.T)
  corrected_value_cotangent = _matmul(key_for_state, state_cotangent_next)
  intra_cotangent = _matmul(output_cotangent, corrected_value.T)
  corrected_value_cotangent = corrected_value_cotangent + _matmul(
      intra.T,
      output_cotangent,
  )
  query_with_decay_cotangent = _matmul(output_cotangent, state.T)
  state_cotangent = state_cotangent + _matmul(query_with_decay.T, output_cotangent)
  u_cotangent = corrected_value_cotangent
  w_cotangent = -_matmul(corrected_value_cotangent, state.T)
  state_cotangent = state_cotangent - _matmul(w.T, corrected_value_cotangent)
  state_cotangent_scratch_ref[...] = state_cotangent
  state_before_cotangent_ref[0, 0, 0, :, :] = state_cotangent

  query_cotangent = query_with_decay_cotangent * cumulative_decay_exp
  cumulative_decay_cotangent = query_with_decay_cotangent * query_with_decay
  key_cotangent = key_for_state_cotangent * state_decay_exp
  state_decay_cotangent = key_for_state_cotangent * key_for_state
  cumulative_decay_cotangent = cumulative_decay_cotangent - state_decay_cotangent
  final_decay_cotangent = jnp.sum(state_decay_cotangent, axis=0)
  final_decay_cotangent = final_decay_cotangent + final_decay_exp_cotangent * final_decay_exp

  def write_profile_outputs(query_bar, key_bar, value_bar, decay_bar, beta_bar):
    query_cotangent_ref[0, 0, :, :] = query_bar.astype(query_cotangent_ref.dtype)
    key_cotangent_ref[0, 0, :, :] = key_bar.astype(key_cotangent_ref.dtype)
    value_cotangent_ref[0, 0, :, :] = value_bar.astype(value_cotangent_ref.dtype)
    log_decay_cotangent_ref[0, 0, :, :] = decay_bar.astype(log_decay_cotangent_ref.dtype)
    beta_cotangent_ref[0, 0, :, 0] = beta_bar.astype(beta_cotangent_ref.dtype)

  if profile_stage == "reverse_state":
    write_profile_outputs(
        query_cotangent,
        key_cotangent,
        u_cotangent,
        cumulative_decay_cotangent,
        jnp.zeros((chunk_size,), dtype=jnp.float32),
    )
    return

  solved_cotangent = jnp.concatenate((u_cotangent, w_cotangent), axis=-1)
  combined_rhs_cotangent = _solve_transposed_unit_lower_triangular_doubling(
      system,
      solved_cotangent,
  )
  system_cotangent = -_matmul(combined_rhs_cotangent, solved.T)
  system_cotangent = system_cotangent * jnp.tril(
      jnp.ones((chunk_size, chunk_size), dtype=jnp.float32),
      k=-1,
  )
  value_beta_cotangent = combined_rhs_cotangent[:, :value_dim]
  w_input_cotangent = combined_rhs_cotangent[:, value_dim : value_dim + key_dim]
  key_beta_cotangent = w_input_cotangent * cumulative_decay_exp
  cumulative_decay_cotangent = cumulative_decay_cotangent + w_input_cotangent * w_input
  if profile_stage == "solve_vjp":
    write_profile_outputs(
        query_cotangent,
        key_beta_cotangent,
        value_beta_cotangent,
        cumulative_decay_cotangent,
        jnp.zeros((chunk_size,), dtype=jnp.float32),
    )
    return

  (
      key_beta_system_cotangent,
      key_system_cotangent,
      system_decay_cotangent,
  ) = _decayed_pairwise_backward(
      key_beta,
      key,
      cumulative_decay,
      system_cotangent,
      include_diagonal=False,
  )
  (
      query_pairwise_cotangent,
      key_intra_cotangent,
      intra_decay_cotangent,
  ) = _decayed_pairwise_backward(
      query,
      key,
      cumulative_decay,
      intra_cotangent,
      include_diagonal=True,
  )
  key_beta_cotangent = key_beta_cotangent + key_beta_system_cotangent
  key_cotangent = key_cotangent + key_system_cotangent + key_intra_cotangent
  query_cotangent = query_cotangent + query_pairwise_cotangent
  cumulative_decay_cotangent = (
      cumulative_decay_cotangent + system_decay_cotangent + intra_decay_cotangent
  )
  cumulative_decay_cotangent = jnp.concatenate(
      (
          cumulative_decay_cotangent[:-1],
          cumulative_decay_cotangent[-1:] + final_decay_cotangent[None, :],
      ),
      axis=0,
  )
  if profile_stage == "pairwise_vjp":
    write_profile_outputs(
        query_cotangent,
        key_cotangent,
        value_beta_cotangent,
        cumulative_decay_cotangent,
        jnp.zeros((chunk_size,), dtype=jnp.float32),
    )
    return

  value_cotangent = value_beta_cotangent * beta[:, None]
  beta_cotangent = jnp.sum(value_beta_cotangent * value, axis=-1)
  key_cotangent = key_cotangent + key_beta_cotangent * beta[:, None]
  beta_cotangent = beta_cotangent + jnp.sum(key_beta_cotangent * key, axis=-1)
  log_decay_cotangent = _reverse_inclusive_cumsum(cumulative_decay_cotangent)

  query_normalized_cotangent = query_cotangent * (1.0 / math.sqrt(key_dim))
  if use_qk_norm:
    query_cotangent = _l2_normalize_backward(
        query_normalized_cotangent,
        query_normalized,
        query_inverse_norm,
    )
    key_cotangent = _l2_normalize_backward(
        key_cotangent,
        key,
        key_inverse_norm,
    )
  else:
    query_cotangent = query_normalized_cotangent

  query_cotangent_ref[0, 0, :, :] = query_cotangent.astype(query_cotangent_ref.dtype)
  key_cotangent_ref[0, 0, :, :] = key_cotangent.astype(key_cotangent_ref.dtype)
  value_cotangent_ref[0, 0, :, :] = value_cotangent.astype(value_cotangent_ref.dtype)
  log_decay_cotangent_ref[0, 0, :, :] = log_decay_cotangent
  beta_cotangent_ref[0, 0, :, 0] = beta_cotangent


@functools.partial(
    jax.jit,
    static_argnames=("chunk_size", "use_qk_norm", "solve_method", "profile_stage"),
)
def pallas_kda_fused_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    log_decay: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
    *,
    chunk_size: int = 64,
    use_qk_norm: bool = True,
    solve_method: str = "doubling",
    profile_stage: str = "full",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs the fixed-layout fused KDA forward on TPU.

  Inputs use ``[B,T,H,D]`` layout. The returned state history contains the
  state *after* each chunk as ``[B,NC,H,K,V]``. The public final state is the
  final history entry.
  """
  if jax.default_backend() != "tpu":
    raise RuntimeError("pallas_kda_fused_forward requires a TPU backend")
  if query.shape != key.shape or query.ndim != 4:
    raise ValueError(f"expected matching [B,T,H,K] Q/K, got {query.shape}, {key.shape}")
  batch, sequence_length, heads, key_dim = query.shape
  value_dim = value.shape[-1]
  if value.shape[:3] != (batch, sequence_length, heads):
    raise ValueError(f"incompatible value shape: {value.shape}")
  if log_decay.shape != query.shape:
    raise ValueError(f"incompatible log-decay shape: {log_decay.shape}")
  if beta.shape != (batch, sequence_length, heads):
    raise ValueError(f"incompatible beta shape: {beta.shape}")
  if initial_state.shape != (batch, heads, key_dim, value_dim):
    raise ValueError(f"incompatible initial state shape: {initial_state.shape}")
  if sequence_length % chunk_size:
    raise ValueError(
        f"sequence length {sequence_length} must be divisible by chunk size {chunk_size}"
    )
  if chunk_size != 64 or key_dim != 128 or value_dim != 128:
    raise ValueError(
        "the first production kernel is specialized to chunk=64 and K=V=128, "
        f"got chunk={chunk_size}, K={key_dim}, V={value_dim}"
    )
  if solve_method not in ("blocked", "doubling"):
    raise ValueError(f"solve_method must be blocked or doubling, got {solve_method}")
  if profile_stage not in ("preprocess", "pairwise", "solve", "full"):
    raise ValueError(f"unknown forward profile stage: {profile_stage}")

  num_chunks = sequence_length // chunk_size
  qkv_spec = pl.BlockSpec(
      block_shape=(1, 1, chunk_size, key_dim),
      index_map=lambda batch_index, head_index, chunk_index: (
          batch_index,
          head_index,
          chunk_index,
          0,
      ),
  )
  value_spec = pl.BlockSpec(
      block_shape=(1, 1, chunk_size, value_dim),
      index_map=lambda batch_index, head_index, chunk_index: (
          batch_index,
          head_index,
          chunk_index,
          0,
      ),
  )
  beta_spec = pl.BlockSpec(
      block_shape=(1, 1, chunk_size, 1),
      index_map=lambda batch_index, head_index, chunk_index: (
          batch_index,
          head_index,
          chunk_index,
          0,
      ),
  )
  initial_state_spec = pl.BlockSpec(
      block_shape=(1, 1, key_dim, value_dim),
      index_map=lambda batch_index, head_index, chunk_index: (
          batch_index,
          head_index,
          0,
          0,
      ),
  )
  state_history_spec = pl.BlockSpec(
      block_shape=(1, 1, 1, key_dim, value_dim),
      index_map=lambda batch_index, head_index, chunk_index: (
          batch_index,
          head_index,
          chunk_index,
          0,
          0,
      ),
  )
  output_shape = jax.ShapeDtypeStruct(
      (batch, heads, sequence_length, value_dim),
      value.dtype,
  )
  state_history_shape = jax.ShapeDtypeStruct(
      (batch, heads, num_chunks, key_dim, value_dim),
      jnp.float32,
  )
  output, state_history = pl.pallas_call(
      functools.partial(
          _kda_fused_forward_kernel,
          chunk_size=chunk_size,
          key_dim=key_dim,
          value_dim=value_dim,
          use_qk_norm=use_qk_norm,
          solve_method=solve_method,
          profile_stage=profile_stage,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(batch, heads, num_chunks),
          in_specs=(
              qkv_spec,
              qkv_spec,
              value_spec,
              qkv_spec,
              beta_spec,
              initial_state_spec,
          ),
          out_specs=(
              value_spec,
              state_history_spec,
          ),
          scratch_shapes=(pltpu.VMEM((key_dim, value_dim), jnp.float32),),
      ),
      out_shape=(output_shape, state_history_shape),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      name=f"kda_fused_forward_{solve_method}_{profile_stage}",
  )(
      query.transpose(0, 2, 1, 3),
      key.transpose(0, 2, 1, 3),
      value.transpose(0, 2, 1, 3),
      log_decay.astype(jnp.float32).transpose(0, 2, 1, 3),
      beta.astype(jnp.float32).transpose(0, 2, 1)[..., None],
      initial_state.astype(jnp.float32),
  )
  output = output.transpose(0, 2, 1, 3)
  final_state = state_history[:, :, -1]
  return output, final_state, state_history.transpose(0, 2, 1, 3, 4)


@functools.partial(
    jax.jit,
    static_argnames=("chunk_size", "use_qk_norm", "profile_stage"),
)
def pallas_kda_fused_backward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    log_decay: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
    state_history: jax.Array,
    output_cotangent: jax.Array,
    final_state_cotangent: jax.Array,
    *,
    chunk_size: int = 64,
    use_qk_norm: bool = True,
    profile_stage: str = "full",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs the fused reverse chunk stream and returns all six input gradients."""
  batch, sequence_length, heads, key_dim = query.shape
  value_dim = value.shape[-1]
  num_chunks = sequence_length // chunk_size
  expected_history_shape = (batch, num_chunks, heads, key_dim, value_dim)
  if state_history.shape != expected_history_shape:
    raise ValueError(f"expected state history {expected_history_shape}, got {state_history.shape}")
  if profile_stage not in ("reverse_state", "solve_vjp", "pairwise_vjp", "full"):
    raise ValueError(f"unknown backward profile stage: {profile_stage}")

  reverse_qkv_spec = pl.BlockSpec(
      block_shape=(1, 1, chunk_size, key_dim),
      index_map=lambda batch_index, head_index, reverse_chunk_index: (
          batch_index,
          head_index,
          num_chunks - 1 - reverse_chunk_index,
          0,
      ),
  )
  reverse_value_spec = pl.BlockSpec(
      block_shape=(1, 1, chunk_size, value_dim),
      index_map=lambda batch_index, head_index, reverse_chunk_index: (
          batch_index,
          head_index,
          num_chunks - 1 - reverse_chunk_index,
          0,
      ),
  )
  reverse_beta_spec = pl.BlockSpec(
      block_shape=(1, 1, chunk_size, 1),
      index_map=lambda batch_index, head_index, reverse_chunk_index: (
          batch_index,
          head_index,
          num_chunks - 1 - reverse_chunk_index,
          0,
      ),
  )
  state_spec = pl.BlockSpec(
      block_shape=(1, 1, key_dim, value_dim),
      index_map=lambda batch_index, head_index, reverse_chunk_index: (
          batch_index,
          head_index,
          0,
          0,
      ),
  )
  previous_state_spec = pl.BlockSpec(
      block_shape=(1, 1, 1, key_dim, value_dim),
      index_map=lambda batch_index, head_index, reverse_chunk_index: (
          batch_index,
          head_index,
          jnp.maximum(num_chunks - 2 - reverse_chunk_index, 0),
          0,
          0,
      ),
  )
  state_before_cotangent_spec = pl.BlockSpec(
      block_shape=(1, 1, 1, key_dim, value_dim),
      index_map=lambda batch_index, head_index, reverse_chunk_index: (
          batch_index,
          head_index,
          num_chunks - 1 - reverse_chunk_index,
          0,
          0,
      ),
  )

  query_t = query.transpose(0, 2, 1, 3)
  key_t = key.transpose(0, 2, 1, 3)
  value_t = value.transpose(0, 2, 1, 3)
  log_decay_t = log_decay.astype(jnp.float32).transpose(0, 2, 1, 3)
  beta_t = beta.astype(jnp.float32).transpose(0, 2, 1)[..., None]
  state_history_t = state_history.transpose(0, 2, 1, 3, 4)
  output_cotangent_t = output_cotangent.transpose(0, 2, 1, 3)

  query_cotangent_shape = jax.ShapeDtypeStruct(query_t.shape, query.dtype)
  key_cotangent_shape = jax.ShapeDtypeStruct(key_t.shape, key.dtype)
  value_cotangent_shape = jax.ShapeDtypeStruct(value_t.shape, value.dtype)
  log_decay_cotangent_shape = jax.ShapeDtypeStruct(log_decay_t.shape, log_decay.dtype)
  beta_cotangent_shape = jax.ShapeDtypeStruct(beta_t.shape, beta.dtype)
  state_before_cotangent_shape = jax.ShapeDtypeStruct(
      state_history_t.shape,
      jnp.float32,
  )
  (
      query_cotangent_t,
      key_cotangent_t,
      value_cotangent_t,
      log_decay_cotangent_t,
      beta_cotangent_t,
      state_before_cotangent_t,
  ) = pl.pallas_call(
      functools.partial(
          _kda_fused_backward_kernel,
          chunk_size=chunk_size,
          key_dim=key_dim,
          value_dim=value_dim,
          num_chunks=num_chunks,
          use_qk_norm=use_qk_norm,
          profile_stage=profile_stage,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(batch, heads, num_chunks),
          in_specs=(
              reverse_qkv_spec,
              reverse_qkv_spec,
              reverse_value_spec,
              reverse_qkv_spec,
              reverse_beta_spec,
              state_spec,
              previous_state_spec,
              reverse_value_spec,
              state_spec,
          ),
          out_specs=(
              reverse_qkv_spec,
              reverse_qkv_spec,
              reverse_value_spec,
              reverse_qkv_spec,
              reverse_beta_spec,
              state_before_cotangent_spec,
          ),
          scratch_shapes=(pltpu.VMEM((key_dim, value_dim), jnp.float32),),
      ),
      out_shape=(
          query_cotangent_shape,
          key_cotangent_shape,
          value_cotangent_shape,
          log_decay_cotangent_shape,
          beta_cotangent_shape,
          state_before_cotangent_shape,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      name=f"kda_fused_backward_{profile_stage}",
  )(
      query_t,
      key_t,
      value_t,
      log_decay_t,
      beta_t,
      initial_state.astype(jnp.float32),
      state_history_t,
      output_cotangent_t,
      final_state_cotangent.astype(jnp.float32),
  )
  return (
      query_cotangent_t.transpose(0, 2, 1, 3),
      key_cotangent_t.transpose(0, 2, 1, 3),
      value_cotangent_t.transpose(0, 2, 1, 3),
      log_decay_cotangent_t.transpose(0, 2, 1, 3),
      beta_cotangent_t[..., 0].transpose(0, 2, 1),
      state_before_cotangent_t[:, :, 0],
  )


@jax.custom_vjp
def pallas_kda_fused(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    log_decay: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Differentiable fixed-shape fused KDA operation for TPU training."""
  output, final_state, _ = pallas_kda_fused_forward(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      chunk_size=64,
      use_qk_norm=True,
      solve_method="doubling",
  )
  return output, final_state


def _pallas_kda_fused_fwd(query, key, value, log_decay, beta, initial_state):
  output, final_state, state_history = pallas_kda_fused_forward(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      chunk_size=64,
      use_qk_norm=True,
      solve_method="doubling",
  )
  return (output, final_state), (
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      state_history,
  )


def _pallas_kda_fused_bwd(residual, output_cotangents):
  query, key, value, log_decay, beta, initial_state, state_history = residual
  output_cotangent, final_state_cotangent = output_cotangents
  return pallas_kda_fused_backward(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      state_history,
      output_cotangent,
      final_state_cotangent,
      chunk_size=64,
      use_qk_norm=True,
  )


pallas_kda_fused.defvjp(
    _pallas_kda_fused_fwd,
    _pallas_kda_fused_bwd,
)
