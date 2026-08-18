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

"""CPU tests for the v4 conv + SiLU fold's chunk math (the Pallas kernels
themselves are gated on device by benchmarks/verify_kda_v4_conv_fold.py).

The kernels call ``_causal_conv_silu`` on VMEM values chunk by chunk with a
16-row halo of the previous chunk; here the same function runs on plain
arrays and must reproduce the layer's XLA path (Flax causal depthwise conv
with bf16 weights/output, SiLU in fp32, bf16 activation) exactly, and the
backward formulas the VJP kernel implements must match autodiff."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from yxtpu_pretrain.kernels.kda_fused_pallas_v4 import (
    _CONV_HALO_ROWS,
    _causal_conv_silu,
    _silu_grad,
)


def _reference(x, w):
  """x [S, T, D] bf16, w [W, S, D] f32 -> (conv bf16-rounded f32, act bf16)."""
  width = w.shape[0]
  xp = jnp.pad(x.astype(jnp.float32), ((0, 0), (width - 1, 0), (0, 0)))
  wb = w.astype(jnp.bfloat16).astype(jnp.float32)
  conv = sum(xp[:, k : k + x.shape[1]] * wb[k][:, None, :] for k in range(width))
  conv = conv.astype(jnp.bfloat16).astype(jnp.float32)
  act = jax.nn.silu(conv).astype(jnp.bfloat16)
  return conv, act


def test_chunked_conv_silu_matches_the_layer_path_bitwise():
  S, C, D, W, NC = 3, 64, 128, 4, 5
  T = NC * C
  k1, k2 = jax.random.split(jax.random.key(0))
  x = jax.random.normal(k1, (S, T, D), jnp.float32).astype(jnp.bfloat16)
  w = jax.random.normal(k2, (W, S, D), jnp.float32) * 0.3
  conv_ref, act_ref = _reference(x, w)
  for c in range(NC):
    raw = x[:, c * C : (c + 1) * C]
    prev = x[:, max(c * C - _CONV_HALO_ROWS, 0) : c * C] if c > 0 else jnp.zeros((S, _CONV_HALO_ROWS, D), x.dtype)
    conv, act = _causal_conv_silu(raw, prev, jnp.transpose(w, (1, 0, 2)), c > 0, conv_width=W)
    np.testing.assert_array_equal(np.asarray(conv), np.asarray(conv_ref[:, c * C : (c + 1) * C]))
    np.testing.assert_array_equal(
        np.asarray(act.astype(jnp.bfloat16)), np.asarray(act_ref[:, c * C : (c + 1) * C])
    )


def test_vjp_kernel_formulas_match_autodiff():
  S, C, D, W, NC = 2, 64, 8, 4, 3
  T = NC * C
  HALO = _CONV_HALO_ROWS
  k1, k2, k3 = jax.random.split(jax.random.key(1), 3)
  x = jax.random.normal(k1, (S, T, D), jnp.float32)
  w = jax.random.normal(k2, (W, S, D), jnp.float32) * 0.3

  def conv_silu(x, w):
    xp = jnp.pad(x, ((0, 0), (W - 1, 0), (0, 0)))
    y = sum(xp[:, k : k + T] * w[k][:, None, :] for k in range(W))
    return jax.nn.silu(y)

  act, vjp = jax.vjp(conv_silu, x, w)
  d_act = jax.random.normal(k3, act.shape, jnp.float32)
  dx_ref, dw_ref = vjp(d_act)

  # stage B emission: cotangent of the conv output
  xp = jnp.pad(x, ((0, 0), (W - 1, 0), (0, 0)))
  conv = sum(xp[:, k : k + T] * w[k][:, None, :] for k in range(W))
  g = d_act * _silu_grad(conv)
  dx = jnp.zeros_like(x)
  dw = jnp.zeros_like(w)
  for c in range(NC):
    raw = x[:, c * C : (c + 1) * C]
    prev = x[:, c * C - HALO : c * C] if c > 0 else jnp.zeros((S, HALO, D))
    padded = jnp.concatenate((prev, raw), axis=1)
    gc = g[:, c * C : (c + 1) * C]
    gnext = g[:, (c + 1) * C : (c + 1) * C + HALO] if c < NC - 1 else jnp.zeros((S, HALO, D))
    g_ext = jnp.concatenate((gc, gnext), axis=1)
    dxc = sum(g_ext[:, (W - 1) - k : (W - 1) - k + C] * w[k][:, None, :] for k in range(W))
    dx = dx.at[:, c * C : (c + 1) * C].set(dxc)
    for k in range(W):
      start = HALO - (W - 1) + k
      dw = dw.at[k].add(jnp.sum(gc * padded[:, start : start + C], axis=1))
  np.testing.assert_allclose(np.asarray(dx), np.asarray(dx_ref), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(np.asarray(dw), np.asarray(dw_ref), rtol=1e-5, atol=1e-5)


def _xla_conv_reference(raw_q, raw_k, raw_v, conv_weight, log_decay, beta, initial_state):
  """The layer's XLA path: bf16 causal depthwise conv on the head-major
  projection, SiLU in fp32 -> bf16, then the base v4 kernel."""
  from yxtpu_pretrain.kernels.kda_fused_pallas_v4 import pallas_kda_fused_v4

  batch, sequence_length, heads, dim = raw_q.shape
  width = conv_weight.shape[0]
  qkv = jnp.stack((raw_q, raw_k, raw_v), axis=3).reshape(batch, sequence_length, -1)
  kernel = conv_weight.reshape(width, 1, heads * 3 * dim).astype(jnp.bfloat16)
  padded = jnp.pad(qkv, ((0, 0), (width - 1, 0), (0, 0)))
  conv = lax.conv_general_dilated(
      padded, kernel, window_strides=(1,), padding="VALID",
      dimension_numbers=("NWC", "WIO", "NWC"), feature_group_count=heads * 3 * dim,
      preferred_element_type=jnp.bfloat16,
  )
  act = jax.nn.silu(conv.astype(jnp.float32)).astype(jnp.bfloat16)
  act = act.reshape(batch, sequence_length, heads, 3, dim)
  q, k, v = (act[..., i, :] for i in range(3))
  return pallas_kda_fused_v4(q, k, v, log_decay, beta, initial_state)


def _rel_l2(a, b):
  a = np.asarray(a, np.float64).ravel()
  b = np.asarray(b, np.float64).ravel()
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def test_conv_fold_custom_vjp_matches_the_xla_path_in_interpret_mode(monkeypatch):
  """Runs the WHOLE folded kernel (forward, stage A, stage B, conv VJP) on
  CPU through Pallas interpret mode against the XLA-conv + base-kernel
  reference, with a nonzero initial state and a stream count that is not
  a multiple of 8. Forward and state must agree to interpret-mode
  reduction order; the seven cotangents to bf16 output rounding. This is
  the CPU counterpart of benchmarks/verify_kda_v4_conv_fold.py."""
  from jax.experimental import pallas as pl

  from yxtpu_pretrain.kernels import kda_fused_pallas_v4 as module

  monkeypatch.setenv("YXTPU_KDA_INTERPRET", "1")
  real_pallas_call = pl.pallas_call
  monkeypatch.setattr(
      module.pl,
      "pallas_call",
      lambda *args, **kwargs: real_pallas_call(*args, **{**kwargs, "interpret": True}),
  )

  batch, sequence_length, heads, dim, width = 1, 192, 2, 128, 4
  keys = jax.random.split(jax.random.key(3), 8)
  shape = (batch, sequence_length, heads, dim)
  raw = [jax.random.normal(keys[i], shape, jnp.bfloat16) for i in range(3)]
  conv_weight = jax.random.normal(keys[3], (width, heads, 3, dim), jnp.float32) * 0.3
  log_decay = -5.0 * jax.nn.sigmoid(jax.random.normal(keys[4], shape, jnp.float32) * 2.0)
  beta = jax.nn.sigmoid(jax.random.normal(keys[5], (batch, sequence_length, heads), jnp.float32))
  initial_state = jax.random.normal(keys[6], (batch, heads, dim, dim), jnp.float32) * 0.1
  primals = (*raw, conv_weight, log_decay, beta, initial_state)
  cotangents = (
      jax.random.normal(keys[7], shape, jnp.bfloat16),
      jnp.full((batch, heads, dim, dim), 0.01, jnp.float32),
  )

  def run(fn):
    outputs, vjp = jax.vjp(fn, *primals)
    return outputs, vjp(cotangents)

  (ref_out, ref_state), ref_grads = run(_xla_conv_reference)
  (out, state), grads = run(module.pallas_kda_fused_v4_conv)

  assert _rel_l2(out, ref_out) < 1e-5
  assert _rel_l2(state, ref_state) < 1e-5
  assert len(grads) == 7
  names = ("d_raw_q", "d_raw_k", "d_raw_v", "d_conv_weight", "d_log_decay", "d_beta", "d_state")
  for name, got, want in zip(names, grads, ref_grads):
    assert got.shape == want.shape, name
    assert np.all(np.isfinite(np.asarray(got, np.float64))), name
    # bf16 output rounding for the raw/weight cotangents; the decay/beta/
    # state cotangents flow through the base kernel's math (interpret-mode
    # reduction order differs from Mosaic's, hence not bitwise here).
    assert _rel_l2(got, want) < 1e-2, (name, _rel_l2(got, want))
