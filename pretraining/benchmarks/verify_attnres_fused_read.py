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

"""On-device gate for the fused Block-AttnRes depth read
(kernels/attnres_pallas.py) against the XLA hoisted read
(layers/attn_res.py::hoisted_depth_read) at the production shape.

Compares the K numerators, normalizers, maxima and the buffer / folded-query
cotangents, and times forward+backward of both at every block_index. Exits 1
if any tolerance is exceeded.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \\
  TPU_VISIBLE_DEVICES=0 .venv/bin/python \\
  benchmarks/verify_attnres_fused_read.py --slots 9 --batch 2 --seq 8192 --dim 1536
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from yxtpu_pretrain.kernels.attnres_pallas import pallas_hoisted_depth_read
from yxtpu_pretrain.layers.attn_res import hoisted_depth_read

EPS = 1.0e-5


def _rel_l2(a, b):
  a = np.asarray(a, np.float64).ravel()
  b = np.asarray(b, np.float64).ravel()
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--slots", type=int, default=9)
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--seq", type=int, default=8192)
  parser.add_argument("--dim", type=int, default=1536)
  parser.add_argument("--sites", type=int, default=8)
  parser.add_argument("--forward-tile", type=int, default=64)
  parser.add_argument("--backward-tile", type=int, default=32)
  parser.add_argument("--block-indices", type=str, default="0,4,8")
  parser.add_argument("--timing-iters", type=int, default=5)
  parser.add_argument("--forward-tolerance", type=float, default=1e-2,
                      help="rel L2 on the bf16 numerators (both paths round once)")
  parser.add_argument("--exact-tolerance", type=float, default=1e-5,
                      help="rel L2 on normalizers/maxima (fp32, same formula)")
  parser.add_argument("--grad-tolerance", type=float, default=2e-2,
                      help="rel L2 on the bf16 buffer cotangent and fp32 query cotangent")
  args = parser.parse_args()

  keys = jax.random.split(jax.random.key(0), 4)
  shape = (args.slots, args.batch, args.seq, args.dim)
  buffer = jax.random.normal(keys[0], shape, jnp.float32).astype(jnp.bfloat16)
  queries = 0.5 * jax.random.normal(keys[1], (args.dim, args.sites), jnp.float32)
  cot_n = [jax.random.normal(jax.random.fold_in(keys[2], k), shape[1:], jnp.bfloat16)
           for k in range(args.sites)]
  cot_z = jax.random.normal(keys[3], (args.batch, args.seq, args.sites), jnp.float32) * 0.01

  def make(read):
    def fwd_bwd(buffer, queries, block_index):
      def f(buffer, queries):
        n, z, m = read(buffer, block_index, queries, EPS)
        return n, z, m
      (n, z, m), vjp = jax.vjp(f, buffer, queries)
      d_buffer, d_queries = vjp((tuple(cot_n), cot_z, jnp.zeros_like(m)))
      return n, z, m, d_buffer, d_queries
    return jax.jit(fwd_bwd)

  xla = make(hoisted_depth_read)
  fused = make(lambda *a: pallas_hoisted_depth_read(
      *a, forward_tile=args.forward_tile, backward_tile=args.backward_tile))

  failures = []
  for block_index in (int(x) for x in args.block_indices.split(",")):
    idx = jnp.int32(block_index)
    rn, rz, rm, rdb, rdq = xla(buffer, queries, idx)
    fn, fz, fm, fdb, fdq = fused(buffer, queries, idx)
    jax.block_until_ready((rn, fn))
    worst_n = max(_rel_l2(f, r) for f, r in zip(fn, rn))
    rel = {
        "numerators": worst_n,
        "normalizers": _rel_l2(fz, rz),
        "maxima": _rel_l2(fm, rm),
        "d_buffer": _rel_l2(fdb, rdb),
        "d_queries": _rel_l2(fdq, rdq),
    }
    finite = all(bool(jnp.all(jnp.isfinite(x.astype(jnp.float32)))) for x in (*fn, fz, fm, fdb, fdq))
    print(f"block_index {block_index}: " + "  ".join(f"{k} {v:.2e}" for k, v in rel.items())
          + f"  finite {finite}")
    for k, tol in (("numerators", args.forward_tolerance), ("normalizers", args.exact_tolerance),
                   ("maxima", args.exact_tolerance), ("d_buffer", args.grad_tolerance),
                   ("d_queries", args.grad_tolerance)):
      if rel[k] > tol:
        failures.append(f"block {block_index} {k} rel_l2 {rel[k]:.2e} > {tol:.0e}")
    if not finite:
      failures.append(f"block {block_index}: non-finite output")
    # timing (fwd + bwd)
    for label, fn_ in (("xla hoisted", xla), ("fused pallas", fused)):
      jax.block_until_ready(fn_(buffer, queries, idx))
      started = time.perf_counter()
      for _ in range(args.timing_iters):
        out = fn_(buffer, queries, idx)
      jax.block_until_ready(out)
      per = (time.perf_counter() - started) / args.timing_iters * 1e3
      print(f"  {label:14s} fwd+bwd {per:8.2f} ms")
  if failures:
    print("GATE FAILED: " + "; ".join(failures))
    return 1
  print("GATE PASSED")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
