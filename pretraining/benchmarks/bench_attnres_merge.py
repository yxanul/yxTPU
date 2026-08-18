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

"""Microbenchmark + numerics check of the Block-AttnRes site merge:
XLA (``out = alpha_t N + beta_t P``, fp32 inside, bf16 in/out) versus the
fused Pallas kernel (``pallas_site_merge``), forward+backward, at the
per-device shape. Prints ms per call and rel L2 of outputs/gradients.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0 \\
  .venv/bin/python benchmarks/bench_attnres_merge.py --batch 4 --seq 4096 --dim 1536
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from yxtpu_pretrain.kernels.attnres_pallas import pallas_site_merge


def _rel(a, b):
  a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--batch", type=int, default=4)
  parser.add_argument("--seq", type=int, default=4096)
  parser.add_argument("--dim", type=int, default=1536)
  parser.add_argument("--tile", type=int, default=256)
  parser.add_argument("--iters", type=int, default=20)
  args = parser.parse_args()
  keys = jax.random.split(jax.random.key(0), 5)
  shape = (args.batch, args.seq, args.dim)
  N = jax.random.normal(keys[0], shape, jnp.float32).astype(jnp.bfloat16)
  P = jax.random.normal(keys[1], shape, jnp.float32).astype(jnp.bfloat16)
  alpha = jax.nn.sigmoid(jax.random.normal(keys[2], shape[:2], jnp.float32))
  beta = 1.0 - alpha
  cot = jax.random.normal(keys[3], shape, jnp.float32).astype(jnp.bfloat16)

  def xla_merge(N, alpha, P, beta):
    return (N.astype(jnp.float32) * alpha[..., None] + P.astype(jnp.float32) * beta[..., None]).astype(jnp.bfloat16)

  def fused_merge(N, alpha, P, beta):
    return pallas_site_merge(N, alpha, P, beta, tile=args.tile)

  def make(fn):
    def fwd_bwd(N, alpha, P, beta):
      out, vjp = jax.vjp(fn, N, alpha, P, beta)
      return out, vjp(cot)
    return jax.jit(fwd_bwd)

  results = {}
  for label, fn in (("xla", make(xla_merge)), ("fused", make(fused_merge))):
    out = fn(N, alpha, P, beta); jax.block_until_ready(out)
    started = time.perf_counter()
    for _ in range(args.iters):
      out = fn(N, alpha, P, beta)
    jax.block_until_ready(out)
    ms = (time.perf_counter() - started) / args.iters * 1e3
    results[label] = (out, ms)
    print(f"{label:6s} fwd+bwd {ms:7.3f} ms  ({shape})")
  (xo, (xdn, xda, xdp, xdb)), _ = results["xla"]
  (fo, (fdn, fda, fdp, fdb)), _ = results["fused"]
  print("rel L2  out %.2e  dN %.2e  dalpha %.2e  dP %.2e  dbeta %.2e" % (
      _rel(fo, xo), _rel(fdn, xdn), _rel(fda, xda), _rel(fdp, xdp), _rel(fdb, xdb)))
  bytes_moved = (3 * 2 + 5 * 2) * np.prod(shape)  # fwd: read 2 write 1; bwd: read 3 write 2 (bf16)
  print(f"roofline at 1.2 TB/s: {bytes_moved / 1.2e12 * 1e3:.3f} ms")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
