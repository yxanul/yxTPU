"""Cost of the Gated DeltaNet-2 gate widening on v4.

Times fwd+bwd of the untouched KDA kernel (scalar beta, [B,T,H]) against the
GDN-2 port (channel-wise [b | w], [B,T,H,K+V]) at production shape. The gates
are the only difference, so the delta is the price of carrying two full-width
gate tensors instead of one scalar per head-token.
"""
from __future__ import annotations

import importlib.util
import pathlib
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def main() -> int:
  from yxtpu_pretrain.kernels import kda_fused_pallas_v4 as kda
  gdn2 = _load(HERE / "gdn2_fused_pallas_v4.py", "gdn2_fused_pallas_v4")

  batch, seq, heads, dim = 1, 4096, 16, 128
  rng = np.random.default_rng(0)
  def r(*shape):
    return jnp.asarray(rng.standard_normal(shape), dtype=jnp.bfloat16)

  q, k, v = r(batch, seq, heads, dim), r(batch, seq, heads, dim), r(batch, seq, heads, dim)
  log_decay = jnp.asarray(
      -np.abs(rng.standard_normal((batch, seq, heads, dim))) * 0.05, jnp.float32
  )
  beta = jnp.asarray(rng.uniform(0.1, 0.9, (batch, seq, heads)), jnp.float32)
  gates = jnp.asarray(rng.uniform(0.1, 0.9, (batch, seq, heads, 2 * dim)), jnp.float32)
  state0 = jnp.zeros((batch, heads, dim, dim), jnp.float32)

  def make(mod):
    def loss(q, k, v, g, b, s):
      out, final = mod.pallas_kda_fused_v4(q, k, v, g, b, s)
      return (out.astype(jnp.float32) ** 2).sum() + (final ** 2).sum()
    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5)))

  def bench(fn, gate, label, iters=30):
    args = (q, k, v, log_decay, gate, state0)
    out = jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
      t0 = time.perf_counter()
      jax.block_until_ready(fn(*args))
      times.append((time.perf_counter() - t0) * 1e3)
    p10 = sorted(times)[len(times) // 10]
    print(f"  {label:22s} p10 {p10:7.3f} ms  median {statistics.median(times):7.3f} ms",
          flush=True)
    return p10

  print(f"shape B={batch} T={seq} H={heads} K=V={dim} (fwd+bwd, per call)", flush=True)
  a = bench(make(kda), beta, "KDA (scalar beta)")
  b = bench(make(gdn2), gates, "GDN-2 (b | w)")
  print(f"GDN-2 cost: {(b / a - 1) * 100:+.1f}%  ({b - a:+.3f} ms)", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
