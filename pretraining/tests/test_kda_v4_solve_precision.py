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

"""The three-pass (bf16x3) production solve must stay within its qualified
error envelope on the correlated-key fixture, and the six-pass rollback path
must remain intact. The bf16 splits are explicit, so CPU arithmetic emulates
the TPU pass structure faithfully (exact products of rounded operands with
fp32 accumulation)."""

import jax
import jax.numpy as jnp
import numpy as np

from yxtpu_pretrain.kernels import kda_fused_pallas_v4 as kernel
from yxtpu_pretrain.kernels.kda_fused_pallas_v4 import (
  _solve_transposed_unit_lower_triangular_inverse,
  _solve_unit_lower_triangular_inverse,
)


def _correlated_system(rows=64, channels=128, correlation=0.95):
  """The real-text failure class fixture from the solver test suite."""
  base_key, noise_key, rhs_key = jax.random.split(jax.random.key(121), 3)
  shared = jax.random.normal(base_key, (1, channels), dtype=jnp.float32)
  noise = jax.random.normal(noise_key, (rows, channels), dtype=jnp.float32)
  keys = jnp.sqrt(correlation) * shared + jnp.sqrt(1.0 - correlation) * noise
  keys = keys / jnp.linalg.norm(keys, axis=-1, keepdims=True)
  beta = jnp.full((rows,), 0.95, dtype=jnp.float32)
  cumulative_decay = jnp.arange(rows, dtype=jnp.float32)[:, None] * -1.0e-3
  decay_weight = jnp.exp(cumulative_decay - cumulative_decay.T)
  system = jnp.tril(beta[:, None] * (keys @ keys.T) * decay_weight, k=-1)
  rhs = jax.random.normal(rhs_key, (rows, 256), dtype=jnp.float32)
  return system, rhs


def test_production_policy_is_three_pass_with_six_pass_rollback():
  assert kernel._SOLVE_INVERSE_PASSES == 3
  # The control paths (doubling/substitution) stay at the full decomposition.
  assert kernel._SOLVE_MATMUL_PRECISION == jax.lax.Precision.HIGHEST
  assert kernel._SOLVE_APPLY_MATMUL_PRECISION == jax.lax.Precision.HIGHEST


def test_three_pass_solve_stays_inside_the_qualified_envelope(monkeypatch):
  system, rhs = _correlated_system()
  matrix = np.eye(64, dtype=np.float64) + np.asarray(system, dtype=np.float64)
  expected = np.linalg.solve(matrix, np.asarray(rhs, dtype=np.float64))
  expected_transposed = np.linalg.solve(matrix.T, np.asarray(rhs, dtype=np.float64))

  def relative(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    return np.linalg.norm(actual - reference) / np.linalg.norm(reference)

  monkeypatch.setattr(kernel, "_SOLVE_INVERSE_PASSES", 3)
  three_forward = relative(
      _solve_unit_lower_triangular_inverse(system, rhs), expected
  )
  three_transposed = relative(
      _solve_transposed_unit_lower_triangular_inverse(system, rhs),
      expected_transposed,
  )
  monkeypatch.setattr(kernel, "_SOLVE_INVERSE_PASSES", 6)
  six_forward = relative(_solve_unit_lower_triangular_inverse(system, rhs), expected)

  # EXP-039's qualified envelope: few-times-1e-4 whole-gradient criterion;
  # the direct three-pass solve error measured 2.3e-5 worst-case on real
  # systems. The synthetic correlated fixture must stay in the same class.
  assert three_forward < 1.0e-4, three_forward
  assert three_transposed < 1.0e-4, three_transposed
  # Full decomposition stays near the fp32 floor, and three passes must not
  # silently degrade to one-pass error levels (~1e-3).
  assert six_forward < 5.0e-6, six_forward
  assert three_forward < 5.0e-4


def test_six_pass_rollback_reproduces_the_highest_precision_path(monkeypatch):
  system, rhs = _correlated_system()
  monkeypatch.setattr(kernel, "_SOLVE_INVERSE_PASSES", 6)
  rolled_back = _solve_unit_lower_triangular_inverse(system, rhs)
  direct = kernel._solve_apply_matmul(
      kernel._unit_lower_inverse(system), rhs.astype(jnp.float32)
  )
  np.testing.assert_array_equal(np.asarray(rolled_back), np.asarray(direct))
