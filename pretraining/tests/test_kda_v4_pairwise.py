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

"""Bit-identity and correctness tests for the v4 kernel's merged pairwise pass.

The merged helpers claim bit-identity with the single-matrix versions they
replace: row-stacked MXU matmuls keep every output row an independent dot
product, trimmed contractions drop only exactly-zero terms, and reductions
keep their original operand shapes. These tests pin that claim on CPU; the
on-device gate reruns the full kernel comparison on v4 hardware where the MXU
accumulates.
"""

import jax
import jax.numpy as jnp
import numpy as np

from yxtpu_pretrain.kernels import kda_fused_pallas_v4 as kernel
from yxtpu_pretrain.kernels.kda_fused_pallas_v4 import (
  _decayed_pairwise,
  _decayed_pairwise_backward,
  _decayed_pairwise_backward_pair,
  _decayed_pairwise_pair,
)


def _pairwise_inputs(streams=3, chunk=64, channels=128, gate_lower_bound=-5.0):
  keys = jax.random.split(jax.random.key(23), 5)
  left_system = jax.random.normal(keys[0], (streams, chunk, channels), jnp.float32)
  left_intra = jax.random.normal(keys[1], (streams, chunk, channels), jnp.float32)
  right = jax.random.normal(keys[2], (streams, chunk, channels), jnp.float32)
  # Worst-case-shaped decay: per-token log decays drawn across the full safe
  # gate range, accumulated exactly as the kernel accumulates them.
  log_decay = gate_lower_bound * jax.random.uniform(
      keys[3], (streams, chunk, channels), jnp.float32
  )
  cumulative = jnp.cumsum(log_decay, axis=-2)
  return left_system, left_intra, right, cumulative, keys[4]


def test_pairwise_pair_is_bitwise_identical_to_singles():
  left_system, left_intra, right, cumulative, _ = _pairwise_inputs()
  expected_system = _decayed_pairwise(
      left_system, right, cumulative, include_diagonal=False
  )
  expected_intra = _decayed_pairwise(
      left_intra, right, cumulative, include_diagonal=True
  )
  system, intra = _decayed_pairwise_pair(left_system, left_intra, right, cumulative)
  np.testing.assert_array_equal(np.asarray(system), np.asarray(expected_system))
  np.testing.assert_array_equal(np.asarray(intra), np.asarray(expected_intra))
  assert bool(jnp.all(jnp.isfinite(system))) and bool(jnp.all(jnp.isfinite(intra)))


def test_pairwise_backward_pair_is_bitwise_identical_to_singles():
  left_system, left_intra, right, cumulative, cotangent_key = _pairwise_inputs()
  cot_keys = jax.random.split(cotangent_key, 2)
  shape = left_system.shape[:-1] + (left_system.shape[-2],)
  system_cotangent = jax.random.normal(cot_keys[0], shape, jnp.float32)
  intra_cotangent = jax.random.normal(cot_keys[1], shape, jnp.float32)

  expected_system = _decayed_pairwise_backward(
      left_system, right, cumulative, system_cotangent, include_diagonal=False
  )
  expected_intra = _decayed_pairwise_backward(
      left_intra, right, cumulative, intra_cotangent, include_diagonal=True
  )
  actual = _decayed_pairwise_backward_pair(
      left_system, left_intra, right, cumulative, system_cotangent, intra_cotangent
  )
  expected = (
      expected_system[0],
      expected_intra[0],
      expected_system[1],
      expected_intra[1],
      expected_system[2],
      expected_intra[2],
  )
  for actual_part, expected_part in zip(actual, expected, strict=True):
    np.testing.assert_array_equal(np.asarray(actual_part), np.asarray(expected_part))


def test_pairwise_backward_pair_matches_autodiff():
  """The merged VJP must also agree with autodiff of a direct reference, so a
  consistency bug shared with the single-matrix code cannot hide."""
  left_system, left_intra, right, cumulative, cotangent_key = _pairwise_inputs(
      streams=2, chunk=32
  )
  chunk = left_system.shape[-2]
  cot_keys = jax.random.split(cotangent_key, 2)
  shape = left_system.shape[:-1] + (chunk,)
  system_cotangent = jax.random.normal(cot_keys[0], shape, jnp.float32)
  intra_cotangent = jax.random.normal(cot_keys[1], shape, jnp.float32)

  def reference(left_s, left_i, right_operand, cumulative_decay):
    decay = jnp.exp(
        cumulative_decay[..., :, None, :] - cumulative_decay[..., None, :, :]
    )
    strict = jnp.tril(jnp.ones((chunk, chunk), jnp.float32), k=-1)
    causal = jnp.tril(jnp.ones((chunk, chunk), jnp.float32))
    system = (
        jnp.einsum("...ic,...ijc,...jc->...ij", left_s, decay, right_operand)
        * strict
    )
    intra = (
        jnp.einsum("...ic,...ijc,...jc->...ij", left_i, decay, right_operand)
        * causal
    )
    return jnp.sum(system * system_cotangent) + jnp.sum(intra * intra_cotangent)

  # Mask the strictly-upper decay exponents through where() so the reference
  # never exponentiates the +large upper triangle the kernel structure avoids.
  def masked_reference(left_s, left_i, right_operand, cumulative_decay):
    difference = (
        cumulative_decay[..., :, None, :] - cumulative_decay[..., None, :, :]
    )
    keep = jnp.tril(jnp.ones((chunk, chunk), jnp.float32))[..., :, :, None] > 0
    decay = jnp.where(keep, jnp.exp(jnp.where(keep, difference, 0.0)), 0.0)
    strict = jnp.tril(jnp.ones((chunk, chunk), jnp.float32), k=-1)
    causal = jnp.tril(jnp.ones((chunk, chunk), jnp.float32))
    system = (
        jnp.einsum("...ic,...ijc,...jc->...ij", left_s, decay, right_operand)
        * strict
    )
    intra = (
        jnp.einsum("...ic,...ijc,...jc->...ij", left_i, decay, right_operand)
        * causal
    )
    return jnp.sum(system * system_cotangent) + jnp.sum(intra * intra_cotangent)

  grads = jax.grad(masked_reference, argnums=(0, 1, 2, 3))(
      left_system, left_intra, right, cumulative
  )
  (
      left_system_cotangent,
      left_intra_cotangent,
      right_system_cotangent,
      right_intra_cotangent,
      system_decay_cotangent,
      intra_decay_cotangent,
  ) = _decayed_pairwise_backward_pair(
      left_system, left_intra, right, cumulative, system_cotangent, intra_cotangent
  )
  np.testing.assert_allclose(
      left_system_cotangent, grads[0], rtol=2e-4, atol=2e-4
  )
  np.testing.assert_allclose(left_intra_cotangent, grads[1], rtol=2e-4, atol=2e-4)
  np.testing.assert_allclose(
      right_system_cotangent + right_intra_cotangent, grads[2], rtol=2e-4, atol=2e-4
  )
  np.testing.assert_allclose(
      system_decay_cotangent + intra_decay_cotangent,
      grads[3],
      rtol=2e-4,
      atol=2e-4,
  )


def test_pairwise_row_block_stays_inside_the_gate_exponent_budget():
  """K3 §2.1.1: with the safe-gate bound |g| <= 5, one-sided reciprocal
  rescaling over a 16-token tile stays below e^80, inside the shared
  fp32/bf16 exponent range. The kernel must never configure a tiling whose
  worst-case factored exponent leaves that envelope."""
  bound = 5.0
  rows = kernel._PAIRWISE_ROW_BLOCK_SIZE
  one_sided = rows / 2 if kernel._PAIRWISE_ANCHOR_MIDPOINT else rows
  worst_exponent = one_sided * bound
  assert worst_exponent <= 85.0, (
      f"pairwise tiling exponent budget exceeded: {worst_exponent}"
  )
  assert bool(jnp.isfinite(jnp.exp(jnp.float32(worst_exponent))))
