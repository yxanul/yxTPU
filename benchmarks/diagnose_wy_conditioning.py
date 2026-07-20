#!/usr/bin/env python3
"""Separate WY system conditioning from recursive-doubling instability.

The BF16 divergence was originally attributed to the WY system being poorly
conditioned. That conflates two different things, and at the exact 64x64
extreme they come apart:

  * ``A`` positive strictly-lower all-ones, which parallel keys drive toward:
    the system is benign, ``max |(I + A)^-1| == 1``, yet ``||A^32||`` is
    astronomically large. Recursive doubling builds huge intermediates that
    must cancel, on a problem that is not hard.
  * ``A`` negative strictly-lower all-ones: the system itself is
    catastrophically ill-conditioned.

Since ``A`` is strictly triangular its spectral radius is exactly zero and
carries no information, so every measure here is a norm or a growth factor.

For each regime this reports problem conditioning and algorithmic growth side
by side, then the residual actually achieved by each solver at each precision,
so the two can be checked against which one predicts failure.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
  return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _keys_independent(rng, chunk, dim):
  return _normalize(rng.standard_normal((chunk, dim)))


def _keys_correlated(rng, chunk, dim, correlation, mixed_signs=False):
  """Keys with a target pairwise correlation.

  For ``v_i = sqrt(c) * base + sqrt(1 - c) * noise_i`` the shared component
  contributes ``c * dim`` to ``E[v_i . v_j]`` and the total norm is ``dim``, so
  the expected correlation after normalizing is ``c`` itself. Using ``c`` and
  ``sqrt(1 - c**2)`` as the coefficients instead would target ``c**2``.
  """
  base = rng.standard_normal(dim)
  noise = rng.standard_normal((chunk, dim))
  signs = rng.choice([-1.0, 1.0], size=(chunk, 1)) if mixed_signs else 1.0
  raw = signs * np.sqrt(correlation) * base + np.sqrt(1.0 - correlation) * noise
  return _normalize(raw)


def _keys_ar1(rng, chunk, dim, phi):
  """AR(1) keys, so correlation decays with distance as it does in a sequence."""
  raw = np.empty((chunk, dim))
  raw[0] = rng.standard_normal(dim)
  for step in range(1, chunk):
    raw[step] = phi * raw[step - 1] + np.sqrt(1.0 - phi**2) * rng.standard_normal(dim)
  return _normalize(raw)


def _build_system(keys, beta, log_decay):
  """Forms the strictly lower ``A`` of the WY system in float64."""
  cumulative = np.cumsum(log_decay, axis=0)
  chunk = keys.shape[0]
  decayed = np.einsum(
      "ic,jc,ijc->ij",
      keys,
      keys,
      np.exp(cumulative[:, None, :] - cumulative[None, :, :]),
  )
  return np.tril(beta[:, None] * decayed, k=-1)


def _doubling_growth(power, rows):
  """Norms of the powers recursive doubling actually forms."""
  norms = [float(np.linalg.norm(power, 2))]
  current = power.copy()
  covered = 1
  while covered * 2 < rows:
    current = current @ current
    covered *= 2
    norms.append(float(np.linalg.norm(current, 2)))
  return norms


def _solve_doubling(system, rhs, dtype):
  """Recursive doubling at a given working precision, tracking growth."""
  power = (-system).astype(dtype)
  rhs = rhs.astype(dtype)
  solution = rhs + (power @ rhs).astype(dtype)
  peak = float(np.max(np.abs(solution.astype(np.float64))))
  power = (power @ power).astype(dtype)
  covered = 2
  while covered < rhs.shape[0]:
    solution = (solution + (power @ solution).astype(dtype)).astype(dtype)
    peak = max(peak, float(np.max(np.abs(solution.astype(np.float64)))))
    power = (power @ power).astype(dtype)
    covered *= 2
  return solution.astype(np.float64), peak


def _solve_substitution(system, rhs, dtype, block=16):
  """Blocked forward substitution with a row-serial base case."""
  lower = system.astype(dtype)
  rhs = rhs.astype(dtype)
  rows = rhs.shape[0]
  solved = np.zeros_like(rhs)
  peak = 0.0
  for start in range(0, rows, block):
    end = start + block
    block_rhs = rhs[start:end].copy()
    if start:
      block_rhs = (block_rhs - (lower[start:end, :start] @ solved[:start]).astype(dtype)).astype(dtype)
    for row in range(start, end):
      value = block_rhs[row - start]
      if row > start:
        value = (value - lower[row, start:row] @ solved[start:row]).astype(dtype)
      solved[row] = value
      peak = max(peak, float(np.max(np.abs(value.astype(np.float64)))))
  return solved.astype(np.float64), peak


def _relative_residual(system, solution, rhs):
  matrix = np.eye(system.shape[0]) + system
  residual = matrix @ solution - rhs
  denominator = np.linalg.norm(matrix, 2) * np.linalg.norm(solution, 2) + 1e-30
  return float(np.linalg.norm(residual, 2) / denominator)


def _analyze(system, rhs, label):
  rows = system.shape[0]
  matrix = np.eye(rows) + system
  inverse = np.linalg.inv(matrix)
  growth = _doubling_growth(-system, rows)

  record = {
      "regime": label,
      "problem_conditioning": {
          "norm_A": float(np.linalg.norm(system, 2)),
          "max_abs_A": float(np.max(np.abs(system))),
          "kappa_2_I_plus_A": float(np.linalg.cond(matrix, 2)),
          "max_abs_inverse": float(np.max(np.abs(inverse))),
      },
      "algorithmic_growth": {
          "power_norms": growth,
          "max_power_norm": max(growth),
      },
  }
  reference = np.linalg.solve(matrix, rhs)
  for name, solver in (("doubling", _solve_doubling), ("substitution", _solve_substitution)):
    for precision, dtype in (("float32", np.float32), ("bfloat16", "bf16")):
      if dtype == "bf16":
        # numpy has no bfloat16; emulate by truncating the mantissa to 8 bits
        # after every product, which is what a single TPU MXU pass delivers.
        def truncate(values):
          as32 = np.asarray(values, dtype=np.float32)
          bits = as32.view(np.uint32) & np.uint32(0xFFFF0000)
          return bits.view(np.float32)

        solution, peak = solver(truncate(system), truncate(rhs), np.float32)
      else:
        solution, peak = solver(system, rhs, dtype)
      record.setdefault("solvers", {})[f"{name}_{precision}"] = {
          "relative_residual": _relative_residual(system, solution, rhs),
          "max_error_vs_exact": float(np.max(np.abs(solution - reference))),
          "peak_intermediate": peak,
          "finite": bool(np.all(np.isfinite(solution))),
      }
  return record


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--chunk", type=int, default=64)
  parser.add_argument("--dim", type=int, default=128)
  parser.add_argument("--width", type=int, default=256)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  rng = np.random.default_rng(args.seed)
  chunk, dim = args.chunk, args.dim
  rhs = rng.standard_normal((chunk, args.width))
  records = []

  # The two exact extremes, which is where conditioning and growth separate.
  ones = np.tril(np.ones((chunk, chunk)), k=-1)
  records.append(_analyze(ones, rhs, "exact: strictly-lower all-ones, positive"))
  records.append(_analyze(-ones, rhs, "exact: strictly-lower all-ones, negative"))

  slow_decay = np.full((chunk, dim), -0.005)
  fast_decay = np.full((chunk, dim), -0.5)
  for label, keys, beta, decay in [
      (
          "harness: independent keys, centered beta, mild decay",
          _keys_independent(rng, chunk, dim),
          np.full(chunk, 0.5),
          np.full((chunk, dim), -0.025),
      ),
      (
          "correlated keys c=0.9, beta 0.95, slow decay",
          _keys_correlated(rng, chunk, dim, 0.9),
          np.full(chunk, 0.95),
          slow_decay,
      ),
      (
          "correlated keys c=0.99, beta 0.99, slow decay",
          _keys_correlated(rng, chunk, dim, 0.99),
          np.full(chunk, 0.99),
          slow_decay,
      ),
      (
          "mixed-sign correlated keys c=0.9, beta 0.95, slow decay",
          _keys_correlated(rng, chunk, dim, 0.9, mixed_signs=True),
          np.full(chunk, 0.95),
          slow_decay,
      ),
      (
          "AR(1) keys phi=0.95, beta 0.95, slow decay",
          _keys_ar1(rng, chunk, dim, 0.95),
          np.full(chunk, 0.95),
          slow_decay,
      ),
      (
          "correlated keys c=0.9, beta 0.95, fast decay",
          _keys_correlated(rng, chunk, dim, 0.9),
          np.full(chunk, 0.95),
          fast_decay,
      ),
  ]:
    records.append(_analyze(_build_system(keys, beta, decay), rhs, label))

  print(json.dumps({"chunk": chunk, "key_dim": dim, "regimes": records}, indent=2))

  print("\n" + "=" * 118)
  header = f"{'regime':<52} {'||A||':>9} {'k2(I+A)':>10} {'max|inv|':>9} {'max||P^k||':>11} {'dbl bf16':>10} {'sub bf16':>10}"
  print(header)
  print("-" * 118)
  for record in records:
    cond = record["problem_conditioning"]
    grow = record["algorithmic_growth"]
    solvers = record["solvers"]
    print(
        f"{record['regime'][:52]:<52} "
        f"{cond['norm_A']:9.3g} {cond['kappa_2_I_plus_A']:10.3g} "
        f"{cond['max_abs_inverse']:9.3g} {grow['max_power_norm']:11.3g} "
        f"{solvers['doubling_bfloat16']['relative_residual']:10.3g} "
        f"{solvers['substitution_bfloat16']['relative_residual']:10.3g}"
    )
  print("=" * 118)
  print("k2 is problem conditioning; max||P^k|| is what recursive doubling forms.")
  print("The last two columns are relative residuals at one BF16 pass.")


if __name__ == "__main__":
  main()
