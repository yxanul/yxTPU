"""Compile probe + tied-gate equivalence for the Gated DeltaNet-2 port.

Retires the one risk that could invalidate the whole plan: widening the gate
ref from [chunk, 1] to [chunk, K+V] changes Mosaic layout assignment, and the
v4 ISA already refuses the integrated backward for a relayout reason.

Two checks:
  1. COMPILE  - forward and split backward lower + compile on v4.
  2. TIED     - with b = beta*1_K and w = beta*1_V, Gated Delta Rule-2 reduces
                to KDA exactly (paper Eq 47), so every output must match the
                untouched production kernel BITWISE. The scalar beta gradient
                is recovered as <dL/db, 1> + <dL/dw, 1> (paper Eq 48).

Run on one worker, its own 4 chips:
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python MFU/gdn2/probe_gdn2_compile.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

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
  from yxtpu_pretrain.kernels import kda_fused_pallas_v4 as kda  # untouched
  gdn2 = _load(HERE / "gdn2_fused_pallas_v4.py", "gdn2_fused_pallas_v4")

  batch, seq, heads, dim = 1, 2048, 16, 128
  print(f"shape B={batch} T={seq} H={heads} K=V={dim}, chunk=64", flush=True)

  rng = np.random.default_rng(0)
  def r(*shape, scale=1.0):
    return jnp.asarray(rng.standard_normal(shape) * scale, dtype=jnp.bfloat16)

  q = r(batch, seq, heads, dim)
  k = r(batch, seq, heads, dim)
  v = r(batch, seq, heads, dim)
  # log-decay must be negative and bounded (the kernel's gate budget is |g|<=5)
  log_decay = jnp.asarray(
      -np.abs(rng.standard_normal((batch, seq, heads, dim))) * 0.05, dtype=jnp.float32
  )
  beta = jnp.asarray(rng.uniform(0.1, 0.9, (batch, seq, heads)), dtype=jnp.float32)
  state0 = jnp.zeros((batch, heads, dim, dim), jnp.float32)

  # tied gates: b = beta * 1_K, w = beta * 1_V, concatenated as [b | w]
  gates = jnp.concatenate(
      [jnp.broadcast_to(beta[..., None], (batch, seq, heads, dim))] * 2, axis=-1
  ).astype(jnp.float32)

  def loss_kda(q, k, v, g, b, s):
    out, final = kda.pallas_kda_fused_v4(q, k, v, g, b, s)
    return (out.astype(jnp.float32) ** 2).sum() + (final ** 2).sum()

  def loss_gdn2(q, k, v, g, b, s):
    out, final = gdn2.pallas_kda_fused_v4(q, k, v, g, b, s)
    return (out.astype(jnp.float32) ** 2).sum() + (final ** 2).sum()

  # --- 1. COMPILE ------------------------------------------------------
  grad_gdn2 = jax.jit(jax.grad(loss_gdn2, argnums=(0, 1, 2, 3, 4, 5)))
  try:
    lowered = grad_gdn2.lower(q, k, v, log_decay, gates, state0)
    compiled = lowered.compile()
    print("COMPILE: OK (forward + split backward lowered and compiled on v4)", flush=True)
    mem = compiled.memory_analysis()
    if mem is not None:
      print(f"  temp {mem.temp_size_in_bytes/2**30:.3f} GB, "
            f"args {mem.argument_size_in_bytes/2**30:.3f} GB", flush=True)
  except Exception as exc:  # noqa: BLE001
    print(f"COMPILE: FAIL\n{type(exc).__name__}: {str(exc)[:2000]}", flush=True)
    return 1

  # --- 2. TIED-GATE BITWISE EQUIVALENCE --------------------------------
  grad_kda = jax.jit(jax.grad(loss_kda, argnums=(0, 1, 2, 3, 4, 5)))
  gk = grad_kda(q, k, v, log_decay, beta, state0)
  gg = compiled(q, k, v, log_decay, gates, state0)

  names = ["dQ", "dK", "dV", "dlog_decay", "dgate", "dstate0"]
  ok = True
  for name, a, b in zip(names, gk, gg):
    if name == "dgate":
      # Eq 48: the scalar gradient is the sum of the two channel gradients.
      b = b[..., :dim].sum(-1) + b[..., dim:].sum(-1)
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if a.shape != b.shape:
      print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
      ok = False
      continue
    bitwise = np.array_equal(a, b)
    denom = max(float(np.abs(a).max()), 1e-30)
    rel = float(np.abs(a - b).max()) / denom
    print(f"  {name:11s} bitwise={bitwise}  max_rel={rel:.3e}", flush=True)
    ok = ok and (bitwise or rel < 1e-6)

  # forward outputs too
  o_kda, s_kda = kda.pallas_kda_fused_v4(q, k, v, log_decay, beta, state0)
  o_g, s_g = gdn2.pallas_kda_fused_v4(q, k, v, log_decay, gates, state0)
  for name, a, b in (("output", o_kda, o_g), ("final_state", s_kda, s_g)):
    a = np.asarray(a.astype(jnp.float32))
    b = np.asarray(b.astype(jnp.float32))
    print(f"  {name:11s} bitwise={np.array_equal(a, b)}", flush=True)
    ok = ok and np.array_equal(a, b)

  print("TIED: " + ("PASS - Gated Delta Rule-2 reduces to KDA exactly" if ok else "FAIL"))
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
