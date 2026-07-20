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

"""Numerical tests for the JAX Kimi Delta Attention implementation."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.kda_fused_pallas import _solve_transposed_unit_lower_triangular_doubling
from maxtext.kernels.kda_fused_pallas import _solve_unit_lower_triangular_doubling
from maxtext.layers.kimi_delta_attention import blocked_unit_lower_solve
from maxtext.layers.kimi_delta_attention import chunk_kda
from maxtext.layers.kimi_delta_attention import _decayed_pairwise_dot
from maxtext.layers.kimi_delta_attention import _decayed_pairwise_dot_bwd
from maxtext.layers.kimi_delta_attention import recurrent_kda_reference


def _inputs():
  keys = jax.random.split(jax.random.key(17), 6)
  shape = (2, 8, 2, 4)
  query = jax.random.normal(keys[0], shape, dtype=jnp.float32)
  key = jax.random.normal(keys[1], shape, dtype=jnp.float32)
  value = jax.random.normal(keys[2], shape, dtype=jnp.float32)
  raw_decay = jax.random.normal(keys[3], shape, dtype=jnp.float32)
  log_decay = -0.01 - 0.05 * jax.nn.sigmoid(raw_decay)
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], shape[:-1], dtype=jnp.float32))
  initial_state = 0.05 * jax.random.normal(keys[5], (2, 2, 4, 4), dtype=jnp.float32)
  return query, key, value, log_decay, beta, initial_state


def test_blocked_unit_lower_solve_and_gradients_match_xla():
  keys = jax.random.split(jax.random.key(11), 3)
  system = 0.03 * jnp.tril(
      jax.random.normal(keys[0], (2, 2, 1, 16, 16), dtype=jnp.float32),
      k=-1,
  )
  rhs = jax.random.normal(keys[1], (2, 2, 1, 16, 32), dtype=jnp.float32)
  cotangent = jax.random.normal(keys[2], rhs.shape, dtype=jnp.float32)

  def reference(a, b):
    identity = jnp.eye(a.shape[-1], dtype=jnp.float32)
    return jax.scipy.linalg.solve_triangular(
        identity + a,
        b,
        lower=True,
        unit_diagonal=True,
    )

  expected, expected_vjp = jax.vjp(reference, system, rhs)
  actual, actual_vjp = jax.vjp(blocked_unit_lower_solve, system, rhs)
  np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
  for actual_grad, expected_grad in zip(
      actual_vjp(cotangent),
      expected_vjp(cotangent),
      strict=True,
  ):
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=3e-5, atol=3e-5)


def test_doubling_triangular_solves_match_xla():
  keys = jax.random.split(jax.random.key(12), 2)
  system = 0.03 * jnp.tril(
      jax.random.normal(keys[0], (16, 16), dtype=jnp.float32),
      k=-1,
  )
  rhs = jax.random.normal(keys[1], (16, 32), dtype=jnp.float32)
  matrix = jnp.eye(16, dtype=jnp.float32) + system
  expected = jax.scipy.linalg.solve_triangular(
      matrix,
      rhs,
      lower=True,
      unit_diagonal=True,
  )
  actual = _solve_unit_lower_triangular_doubling(system, rhs)
  expected_transposed = jax.scipy.linalg.solve_triangular(
      matrix.T,
      rhs,
      lower=False,
      unit_diagonal=True,
  )
  actual_transposed = _solve_transposed_unit_lower_triangular_doubling(system, rhs)
  np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
  np.testing.assert_allclose(actual_transposed, expected_transposed, rtol=2e-5, atol=2e-5)


def test_decayed_pairwise_dot_blockwise_backward_matches_autodiff():
  keys = jax.random.split(jax.random.key(13), 4)
  shape = (1, 2, 1, 16, 8)
  left = jax.random.normal(keys[0], shape, dtype=jnp.float32)
  right = jax.random.normal(keys[1], shape, dtype=jnp.float32)
  log_decay = -0.01 - 0.03 * jax.nn.sigmoid(jax.random.normal(keys[2], shape, dtype=jnp.float32))
  cumulative_decay = jnp.cumsum(log_decay, axis=-2)
  cotangent = jax.random.normal(keys[3], shape[:-1] + (shape[-2],), dtype=jnp.float32)

  for include_diagonal in (False, True):

    def pairwise(a, b, g):
      return _decayed_pairwise_dot(
          a,
          b,
          g,
          include_diagonal=include_diagonal,
      )

    _, reference_vjp = jax.vjp(pairwise, left, right, cumulative_decay)
    expected = reference_vjp(cotangent)
    actual = _decayed_pairwise_dot_bwd(
        left,
        right,
        cumulative_decay,
        cotangent,
        include_diagonal=include_diagonal,
    )
    for actual_grad, expected_grad in zip(actual, expected, strict=True):
      np.testing.assert_allclose(actual_grad, expected_grad, rtol=3e-5, atol=3e-5)


def test_chunk_kda_matches_recurrent_reference():
  query, key, value, log_decay, beta, initial_state = _inputs()
  expected_output, expected_state = recurrent_kda_reference(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state=initial_state,
  )
  actual_output, actual_state = chunk_kda(
      query,
      key,
      value,
      log_decay,
      beta,
      chunk_size=4,
      initial_state=initial_state,
      compute_dtype=jnp.float32,
      use_pallas_blocked_solve=False,
  )

  np.testing.assert_allclose(actual_output, expected_output, rtol=2e-4, atol=2e-4)
  np.testing.assert_allclose(actual_state, expected_state, rtol=2e-4, atol=2e-4)


def test_chunk_kda_gradients_match_recurrent_reference():
  inputs = _inputs()

  def reference_loss(*args):
    output, state = recurrent_kda_reference(*args[:-1], initial_state=args[-1])
    return jnp.mean(output**2) + 0.1 * jnp.mean(state**2)

  def chunk_loss(*args):
    output, state = chunk_kda(
        *args[:-1],
        chunk_size=4,
        initial_state=args[-1],
        compute_dtype=jnp.float32,
        use_pallas_blocked_solve=False,
    )
    return jnp.mean(output**2) + 0.1 * jnp.mean(state**2)

  reference_grads = jax.grad(reference_loss, argnums=tuple(range(len(inputs))))(*inputs)
  chunk_grads = jax.grad(chunk_loss, argnums=tuple(range(len(inputs))))(*inputs)
  for actual, expected in zip(chunk_grads, reference_grads, strict=True):
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_chunk_kda_blocked_solve_gradients_match_recurrent_reference():
  keys = jax.random.split(jax.random.key(23), 6)
  shape = (1, 16, 1, 16)
  inputs = (
      jax.random.normal(keys[0], shape, dtype=jnp.float32),
      jax.random.normal(keys[1], shape, dtype=jnp.float32),
      jax.random.normal(keys[2], shape, dtype=jnp.float32),
      -0.01 - 0.03 * jax.nn.sigmoid(jax.random.normal(keys[3], shape, dtype=jnp.float32)),
      jax.nn.sigmoid(jax.random.normal(keys[4], shape[:-1], dtype=jnp.float32)),
      0.02 * jax.random.normal(keys[5], (1, 1, 16, 16), dtype=jnp.float32),
  )

  def reference_loss(*args):
    output, state = recurrent_kda_reference(*args[:-1], initial_state=args[-1])
    return jnp.mean(output**2) + 0.1 * jnp.mean(state**2)

  def blocked_loss(*args):
    output, state = chunk_kda(
        *args[:-1],
        chunk_size=16,
        initial_state=args[-1],
        compute_dtype=jnp.float32,
        use_pallas_blocked_solve=True,
    )
    return jnp.mean(output**2) + 0.1 * jnp.mean(state**2)

  reference_grads = jax.grad(reference_loss, argnums=tuple(range(len(inputs))))(*inputs)
  blocked_grads = jax.grad(blocked_loss, argnums=tuple(range(len(inputs))))(*inputs)
  for actual, expected in zip(blocked_grads, reference_grads, strict=True):
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_chunk_kda_analytical_vjp_matches_generic_autodiff():
  keys = jax.random.split(jax.random.key(27), 6)
  shape = (1, 16, 1, 16)
  inputs = (
      jax.random.normal(keys[0], shape, dtype=jnp.float32),
      jax.random.normal(keys[1], shape, dtype=jnp.float32),
      jax.random.normal(keys[2], shape, dtype=jnp.float32),
      -0.01 - 0.03 * jax.nn.sigmoid(jax.random.normal(keys[3], shape, dtype=jnp.float32)),
      jax.nn.sigmoid(jax.random.normal(keys[4], shape[:-1], dtype=jnp.float32)),
      0.02 * jax.random.normal(keys[5], (1, 1, 16, 16), dtype=jnp.float32),
  )

  def loss(use_analytical_custom_vjp, *args):
    output, state = chunk_kda(
        *args[:-1],
        chunk_size=16,
        initial_state=args[-1],
        compute_dtype=jnp.float32,
        use_pallas_blocked_solve=False,
        use_analytical_custom_vjp=use_analytical_custom_vjp,
    )
    return jnp.mean(output**2) + 0.1 * jnp.mean(state**2)

  generic_value, generic_grads = jax.value_and_grad(
      functools.partial(loss, False),
      argnums=tuple(range(len(inputs))),
  )(*inputs)
  analytical_value, analytical_grads = jax.value_and_grad(
      functools.partial(loss, True),
      argnums=tuple(range(len(inputs))),
  )(*inputs)
  np.testing.assert_allclose(analytical_value, generic_value, rtol=2e-5, atol=2e-5)
  for actual, expected in zip(analytical_grads, generic_grads, strict=True):
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)


def test_chunk_kda_masks_future_decay_before_exponentiation():
  """Large cumulative decays must not overflow at masked future positions."""
  keys = jax.random.split(jax.random.key(29), 4)
  shape = (1, 64, 1, 4)
  query = jax.random.normal(keys[0], shape)
  key = jax.random.normal(keys[1], shape)
  value = jax.random.normal(keys[2], shape)
  log_decay = -jnp.full(shape, 2.0)
  beta = jax.nn.sigmoid(jax.random.normal(keys[3], shape[:-1]))

  def loss(q, k, v, g, b):
    output, state = chunk_kda(
        q,
        k,
        v,
        g,
        b,
        chunk_size=64,
        compute_dtype=jnp.float32,
        use_pallas_blocked_solve=True,
    )
    return jnp.mean(output**2) + jnp.mean(state**2)

  value_and_grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(
      query, key, value, log_decay, beta
  )
  for array in jax.tree.leaves(value_and_grads):
    assert np.all(np.isfinite(array))
