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

"""On-device gate for the v4 conv + SiLU fold (pallas_kda_fused_v4_conv).

Reference: the layer's XLA path - Flax-style causal depthwise conv on the
bf16 [B, T, H*3*D] projection (bf16 weights, bf16 output), SiLU in fp32
cast back to bf16, per-head split, then the production ``pallas_kda_fused_v4``.
Candidate: ``pallas_kda_fused_v4_conv`` on the raw projection. Compares the
outputs and all seven cotangents (raw q/k/v, conv weight, log decay, beta,
initial state) and times fwd+bwd of both.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \\
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \\
  benchmarks/verify_kda_v4_conv_fold.py --batch 2 --seq 8192 --heads 12
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from yxtpu_pretrain.kernels.kda_fused_pallas_v4 import (
    pallas_kda_fused_v4,
    pallas_kda_fused_v4_conv,
)


def _inputs(batch, sequence_length, heads, width, seed):
  keys = jax.random.split(jax.random.key(seed), 8)
  shape = (batch, sequence_length, heads, 128)
  raw_q = jax.random.normal(keys[0], shape, jnp.bfloat16)
  raw_k = jax.random.normal(keys[1], shape, jnp.bfloat16)
  raw_v = jax.random.normal(keys[2], shape, jnp.bfloat16)
  conv_weight = (
      jax.random.normal(keys[3], (width, heads, 3, 128), jnp.float32) * 0.3
  )
  log_decay = -5.0 * jax.nn.sigmoid(jax.random.normal(keys[4], shape, jnp.float32) * 2.0)
  beta = jax.nn.sigmoid(jax.random.normal(keys[5], (batch, sequence_length, heads), jnp.float32))
  initial_state = jnp.zeros((batch, heads, 128, 128), jnp.float32)
  return raw_q, raw_k, raw_v, conv_weight, log_decay, beta, initial_state


def reference(raw_q, raw_k, raw_v, conv_weight, log_decay, beta, initial_state):
  batch, sequence_length, heads, dim = raw_q.shape
  width = conv_weight.shape[0]
  # [B, T, H, 3, D] -> [B, T, H*3*D] (the layer's head-major channel order)
  qkv = jnp.stack((raw_q, raw_k, raw_v), axis=3).reshape(batch, sequence_length, -1)
  kernel = conv_weight.reshape(width, 1, heads * 3 * dim).astype(jnp.bfloat16)
  padded = jnp.pad(qkv, ((0, 0), (width - 1, 0), (0, 0)))
  conv = lax.conv_general_dilated(
      padded,
      kernel,
      window_strides=(1,),
      padding="VALID",
      dimension_numbers=("NWC", "WIO", "NWC"),
      feature_group_count=heads * 3 * dim,
      preferred_element_type=jnp.bfloat16,
  )
  act = jax.nn.silu(conv.astype(jnp.float32)).astype(jnp.bfloat16)
  act = act.reshape(batch, sequence_length, heads, 3, dim)
  q, k, v = (act[..., i, :] for i in range(3))
  return pallas_kda_fused_v4(q, k, v, log_decay, beta, initial_state)


def candidate(raw_q, raw_k, raw_v, conv_weight, log_decay, beta, initial_state):
  return pallas_kda_fused_v4_conv(raw_q, raw_k, raw_v, conv_weight, log_decay, beta, initial_state)


def _run(fn, primals, cotangents):
  outputs, vjp = jax.vjp(fn, *primals)
  grads = vjp(cotangents)
  return outputs, grads


def _rel_l2(a, b):
  a = np.asarray(a, np.float64).ravel()
  b = np.asarray(b, np.float64).ravel()
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--seq", type=int, default=8192)
  parser.add_argument("--heads", type=int, default=12)
  parser.add_argument("--width", type=int, default=4)
  parser.add_argument("--seeds", type=int, default=2)
  parser.add_argument("--timing-iters", type=int, default=5)
  parser.add_argument(
      "--exact-tolerance", type=float, default=1e-6,
      help="rel L2 bound for the outputs the fold reproduces bitwise",
  )
  parser.add_argument(
      "--bf16-tolerance", type=float, default=1e-2,
      help="rel L2 bound for the raw-input / conv-weight cotangents (bf16 rounding)",
  )
  args = parser.parse_args()

  names = ["output", "final_state", "d_raw_q", "d_raw_k", "d_raw_v", "d_conv_weight",
           "d_log_decay", "d_beta", "d_initial_state"]
  ref_fn = jax.jit(lambda p, c: _run(reference, p, c))
  cand_fn = jax.jit(lambda p, c: _run(candidate, p, c))
  worst = {name: 0.0 for name in names}
  all_finite = True
  for seed in range(args.seeds):
    primals = _inputs(args.batch, args.seq, args.heads, args.width, seed)
    keys = jax.random.split(jax.random.key(1000 + seed), 2)
    cotangents = (
        jax.random.normal(keys[0], primals[0].shape, jnp.bfloat16),
        jax.random.normal(keys[1], primals[6].shape, jnp.float32) * 0.01,
    )
    (ref_out, ref_state), ref_grads = ref_fn(primals, cotangents)
    (cand_out, cand_state), cand_grads = cand_fn(primals, cotangents)
    values = [(ref_out, cand_out), (ref_state, cand_state)] + list(zip(ref_grads, cand_grads))
    print(f"seed {seed}:")
    for name, (r, c) in zip(names, values):
      rel = _rel_l2(c, r)
      worst[name] = max(worst[name], rel)
      max_abs = float(np.max(np.abs(np.asarray(c, np.float64) - np.asarray(r, np.float64))))
      finite = bool(np.all(np.isfinite(np.asarray(c, np.float64))))
      all_finite = all_finite and finite
      print(f"  {name:16s} rel_l2 {rel:.3e}  max_abs {max_abs:.3e}  finite {finite}")

  # timing at the given shape (fwd + bwd)
  primals = _inputs(args.batch, args.seq, args.heads, args.width, 0)
  keys = jax.random.split(jax.random.key(7), 2)
  cotangents = (
      jax.random.normal(keys[0], primals[0].shape, jnp.bfloat16),
      jax.random.normal(keys[1], primals[6].shape, jnp.float32) * 0.01,
  )
  for label, fn in (("reference (XLA conv + kernel)", ref_fn), ("candidate (fold)", cand_fn)):
    jax.block_until_ready(fn(primals, cotangents))
    started = time.perf_counter()
    for _ in range(args.timing_iters):
      out = fn(primals, cotangents)
    jax.block_until_ready(out)
    per = (time.perf_counter() - started) / args.timing_iters * 1e3
    print(f"{label:32s} fwd+bwd {per:8.2f} ms")
  print("worst rel_l2:", {k: f"{v:.2e}" for k, v in worst.items()})
  # Gate. Forward, final state, d_log_decay, d_beta and d_initial_state are
  # bitwise on device (the fold's chunk math is the base kernel's; measured
  # 2026-08-18 at the production shape); the raw-input and conv-weight
  # cotangents differ by the XLA path's bf16 rounding of its conv-transpose
  # and dW (one bf16 ulp, rel L2 ~3e-3), so they get a bf16-ulp threshold.
  failures = []
  for name in ("output", "final_state", "d_log_decay", "d_beta", "d_initial_state"):
    if worst[name] > args.exact_tolerance:
      failures.append(f"{name} rel_l2 {worst[name]:.2e} > {args.exact_tolerance:.1e}")
  for name in ("d_raw_q", "d_raw_k", "d_raw_v", "d_conv_weight"):
    if worst[name] > args.bf16_tolerance:
      failures.append(f"{name} rel_l2 {worst[name]:.2e} > {args.bf16_tolerance:.1e}")
  if not all_finite:
    failures.append("non-finite candidate output")
  if failures:
    print("GATE FAILED: " + "; ".join(failures))
    return 1
  print("GATE PASSED")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
