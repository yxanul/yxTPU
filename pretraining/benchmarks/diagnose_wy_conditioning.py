#!/usr/bin/env python3
"""Separate WY problem conditioning from recursive-doubling growth, on TPU.

Two failure modes are easy to conflate. A system can be genuinely
ill-conditioned, or an algorithm can be unstable on a well-conditioned system.
They are complementary rather than alternatives:

  * recursive-doubling growth predicts the *backward* error of the doubling
    solve, that is, how far the computed solution is from solving a nearby
    system;
  * ``kappa_2(I + A)`` predicts how much of that backward error is amplified
    into *forward* error in the solution itself.

Both are reported here. In the positive-correlated regimes that motivated this
work, growth is the dominant mechanism, but a small backward error on an
ill-conditioned system can still carry an enormous forward error, so neither
number alone is sufficient for a harness assertion.

Because ``A`` is strictly lower triangular its spectral radius is identically
zero and carries no information, so every measure here is a norm or a growth
factor.

Arithmetic is the real thing rather than an emulation. A TPU matmul at
``Precision.DEFAULT`` rounds *both operands* to BF16 for *every* matmul and
accumulates in FP32, so each power and each solution update is re-rounded as
it is formed. Truncating the inputs once and then computing in FP32 models
something strictly more accurate and would understate the failure.

The key regimes below are plausible stress regimes chosen to bracket the
behaviour. They are not measured trained-model distributions; see the
real-token instrumentation entry in EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
from jax import lax


def _dot(left, right, *, bf16: bool):
  """One matmul under TPU semantics at the requested precision."""
  if bf16:
    return lax.dot_general(
        left.astype(jnp.bfloat16),
        right.astype(jnp.bfloat16),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
  return lax.dot_general(
      left.astype(jnp.float32),
      right.astype(jnp.float32),
      (((1,), (0,)), ((), ())),
      precision=lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )


def _normalize(vectors):
  return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _keys_independent(rng, chunk, dim):
  return _normalize(rng.standard_normal((chunk, dim)))


def _keys_correlated(rng, chunk, dim, correlation, mixed_signs=False):
  """Keys with a target pairwise correlation of ``correlation``.

  For ``v_i = sqrt(c) * base + sqrt(1 - c) * noise_i`` the shared component
  contributes ``c * dim`` to ``E[v_i . v_j]`` while the total norm is ``dim``,
  so the expected correlation after normalizing is ``c``. Using ``c`` and
  ``sqrt(1 - c**2)`` as coefficients would target ``c**2`` instead.
  """
  base = rng.standard_normal(dim)
  noise = rng.standard_normal((chunk, dim))
  signs = rng.choice([-1.0, 1.0], size=(chunk, 1)) if mixed_signs else 1.0
  return _normalize(
      signs * np.sqrt(correlation) * base + np.sqrt(1.0 - correlation) * noise
  )


def _keys_ar1(rng, chunk, dim, phi):
  """AR(1) keys, so correlation decays with lag as it does along a sequence."""
  raw = np.empty((chunk, dim))
  raw[0] = rng.standard_normal(dim)
  for step in range(1, chunk):
    raw[step] = phi * raw[step - 1] + np.sqrt(1.0 - phi**2) * rng.standard_normal(dim)
  return _normalize(raw)


def _build_system(keys, beta, log_decay):
  """Strictly lower ``A`` of the WY system, in float64."""
  cumulative = np.cumsum(log_decay, axis=0)
  decayed = np.einsum(
      "ic,jc,ijc->ij",
      keys,
      keys,
      np.exp(cumulative[:, None, :] - cumulative[None, :, :]),
  )
  return np.tril(beta[:, None] * decayed, k=-1)


def _power_norms(system, rows):
  """Norms of the powers recursive doubling actually forms, in float64."""
  norms = [float(np.linalg.norm(system, 2))]
  current = system.copy()
  covered = 1
  while covered * 2 < rows:
    current = current @ current
    covered *= 2
    norms.append(float(np.linalg.norm(current, 2)))
  return norms


def _solve_doubling(system, rhs, *, bf16):
  """Recursive doubling, returning the solution and its whole path.

  The path is what distinguishes regimes that grow harmlessly from those that
  grow and then cancel: ``increments`` is the magnitude added at each stage and
  ``solution_norms`` is where the running solution sat, so a large sum of
  increments against a small final norm is exactly the cancellation that BF16
  cannot survive.
  """
  power = jnp.asarray(-system, dtype=jnp.float32)
  solution = jnp.asarray(rhs, dtype=jnp.float32)
  increments = []
  solution_norms = []
  snapshots = []
  covered = 1
  while covered < rhs.shape[0]:
    delta = _dot(power, solution, bf16=bf16)
    solution = solution + delta
    increments.append(float(jnp.linalg.norm(delta.astype(jnp.float32))))
    solution_norms.append(float(jnp.linalg.norm(solution.astype(jnp.float32))))
    snapshots.append(np.asarray(solution, dtype=np.float64))
    power = _dot(power, power, bf16=bf16)
    covered *= 2
  final = np.asarray(solution, dtype=np.float64)
  final_norm = float(np.linalg.norm(final, 2)) + 1e-300
  path = {
      "stage_solution_norms": solution_norms,
      "stage_increment_norms": increments,
      "max_intermediate_ratio": max(solution_norms) / final_norm,
      "cancellation_factor": float(sum(increments)) / final_norm,
  }
  return final, path, snapshots


def _solve_substitution(system, rhs, *, bf16, block=16):
  lower = jnp.asarray(system, dtype=jnp.float32)
  rhs = jnp.asarray(rhs, dtype=jnp.float32)
  rows = rhs.shape[0]
  solved = []
  for start in range(0, rows, block):
    end = start + block
    block_rhs = rhs[start:end]
    if start:
      block_rhs = block_rhs - _dot(
          lower[start:end, :start], jnp.concatenate(solved, axis=0), bf16=bf16
      )
    # Row-serial base case: forms no power of the diagonal block.
    rows_out = []
    for row in range(block):
      value = block_rhs[row : row + 1]
      if row:
        value = value - _dot(
            lower[start + row : start + row + 1, start : start + row],
            jnp.concatenate(rows_out, axis=0),
            bf16=bf16,
        )
      rows_out.append(value)
    solved.append(jnp.concatenate(rows_out, axis=0))
  return np.asarray(jnp.concatenate(solved, axis=0), dtype=np.float64), None, None


def _reference_solution(system, rhs, kappa):
  """Ground truth for the forward error, chosen so it is actually trustworthy.

  float64 forward substitution is only meaningful while ``kappa * eps_64`` stays
  far below one. The negative all-ones extreme has ``kappa`` near 2.8e17, so
  ``kappa * eps_64`` is about 62 and a float64 reference carries no correct
  digits at all. Where the system entries are exactly representable, which is
  the case for both all-ones extremes, exact rational forward substitution is
  used instead: the entries are 0 or +/-1 and the right-hand side is float64,
  so every intermediate stays a dyadic rational and nothing is rounded.
  """
  rows, width = rhs.shape
  entries = np.unique(np.abs(system))
  exactly_representable = bool(np.all(np.isin(entries, (0.0, 1.0))))
  if kappa * np.finfo(np.float64).eps < 1e-8:
    return np.linalg.solve(np.eye(rows) + system, rhs), "float64"
  if exactly_representable:
    lower = [[Fraction(system[i][j]) for j in range(i)] for i in range(rows)]
    solved: list[list[Fraction]] = []
    for i in range(rows):
      row = []
      for k in range(width):
        accumulator = Fraction(rhs[i][k])
        for j in range(i):
          if lower[i][j]:
            accumulator -= lower[i][j] * solved[j][k]
        row.append(accumulator)
      solved.append(row)
    return (
        np.array([[float(value) for value in row] for row in solved]),
        "exact rational",
    )
  mpmath.mp.dps = 120
  lower_mp = [[mpmath.mpf(float(system[i][j])) for j in range(i)] for i in range(rows)]
  solved_mp: list[list] = []
  for i in range(rows):
    row = []
    for k in range(width):
      accumulator = mpmath.mpf(float(rhs[i][k]))
      for j in range(i):
        if lower_mp[i][j] != 0:
          accumulator -= lower_mp[i][j] * solved_mp[j][k]
      row.append(accumulator)
    solved_mp.append(row)
  return (
      np.array([[float(value) for value in row] for row in solved_mp]),
      "mpmath dps=120",
  )


def _errors(system, rhs, computed, exact):
  """Backward error, which is how nearly the solution solves the system, and
  forward error, which is how far it is from the true solution."""
  matrix = np.eye(system.shape[0]) + system
  residual = matrix @ computed - rhs
  backward = float(
      np.linalg.norm(residual, 2)
      / (np.linalg.norm(matrix, 2) * np.linalg.norm(computed, 2) + np.linalg.norm(rhs, 2))
  )
  forward = float(np.linalg.norm(computed - exact, 2) / np.linalg.norm(exact, 2))
  return backward, forward, bool(np.all(np.isfinite(computed)))


def _analyze(system, rhs, label):
  rows = system.shape[0]
  matrix = np.eye(rows) + system
  kappa = float(np.linalg.cond(matrix, 2))
  exact, reference_method = _reference_solution(system, rhs, kappa)
  growth = _power_norms(-system, rows)

  record = {
      "regime": label,
      "reference_method": reference_method,
      "problem_conditioning": {
          "norm_A": float(np.linalg.norm(system, 2)),
          "kappa_2_I_plus_A": kappa,
          "max_abs_inverse": float(np.max(np.abs(np.linalg.inv(matrix)))),
      },
      "algorithmic_growth": {
          "power_norms": growth,
          "max_power_norm": max(growth),
      },
      "solvers": {},
  }

  # Per-stage divergence of the BF16 path from the full-precision path, after
  # powers 1, 2, 4, ... This localizes where the doubling solve actually loses
  # the answer, which max power norm alone cannot show.
  _, path_hi, snapshots_hi = _solve_doubling(system, rhs, bf16=False)
  _, path_lo, snapshots_lo = _solve_doubling(system, rhs, bf16=True)
  stage_errors = []
  for hi, lo in zip(snapshots_hi, snapshots_lo, strict=True):
    denominator = np.linalg.norm(hi, 2) + 1e-300
    stage_errors.append(float(np.linalg.norm(lo - hi, 2) / denominator))
  record["doubling_path"] = {
      "covered_terms": [2**i for i in range(1, len(stage_errors) + 1)],
      "stage_bf16_vs_highest": stage_errors,
      "max_intermediate_ratio_bf16": path_lo["max_intermediate_ratio"],
      "cancellation_factor_bf16": path_lo["cancellation_factor"],
      "max_intermediate_ratio_fp32": path_hi["max_intermediate_ratio"],
      "cancellation_factor_fp32": path_hi["cancellation_factor"],
  }

  for name, solver in (("doubling", _solve_doubling), ("substitution", _solve_substitution)):
    for precision, bf16 in (("fp32", False), ("bf16", True)):
      computed, _, _ = solver(system, rhs, bf16=bf16)
      backward, forward, finite = _errors(system, rhs, computed, exact)
      record["solvers"][f"{name}_{precision}"] = {
          "backward_error": backward,
          "forward_error": forward,
          "finite": finite,
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

  ones = np.tril(np.ones((chunk, chunk)), k=-1)
  records.append(_analyze(ones, rhs, "exact extreme: all-ones, positive"))
  records.append(_analyze(-ones, rhs, "exact extreme: all-ones, negative"))

  slow = np.full((chunk, dim), -0.005)
  fast = np.full((chunk, dim), -0.5)
  for label, keys, beta, decay in [
      (
          "harness today: independent keys, centered beta",
          _keys_independent(rng, chunk, dim),
          np.full(chunk, 0.5),
          np.full((chunk, dim), -0.025),
      ),
      (
          "stress: correlated c=0.9, beta 0.95, slow decay",
          _keys_correlated(rng, chunk, dim, 0.9),
          np.full(chunk, 0.95),
          slow,
      ),
      (
          "stress: correlated c=0.99, beta 0.99, slow decay",
          _keys_correlated(rng, chunk, dim, 0.99),
          np.full(chunk, 0.99),
          slow,
      ),
      (
          "stress: mixed-sign correlated c=0.9, beta 0.95",
          _keys_correlated(rng, chunk, dim, 0.9, mixed_signs=True),
          np.full(chunk, 0.95),
          slow,
      ),
      (
          "stress: AR(1) phi=0.95, beta 0.95, slow decay",
          _keys_ar1(rng, chunk, dim, 0.95),
          np.full(chunk, 0.95),
          slow,
      ),
      (
          "stress: correlated c=0.9, beta 0.95, fast decay",
          _keys_correlated(rng, chunk, dim, 0.9),
          np.full(chunk, 0.95),
          fast,
      ),
  ]:
    records.append(_analyze(_build_system(keys, beta, decay), rhs, label))

  print(json.dumps({"backend": jax.default_backend(), "regimes": records}, indent=2))

  print("\n" + "=" * 124)
  print(
      f"{'regime':<40} {'k2(I+A)':>9} {'max|P^k|':>9} {'cancel':>9} "
      f"{'dbl bwd':>9} {'dbl fwd':>9} {'sub bwd':>9} {'ref':>15}"
  )
  print("-" * 124)
  for record in records:
    cond = record["problem_conditioning"]
    grow = record["algorithmic_growth"]
    solvers = record["solvers"]
    print(
        f"{record['regime'][:40]:<40} {cond['kappa_2_I_plus_A']:9.3g} "
        f"{grow['max_power_norm']:9.3g} "
        f"{record['doubling_path']['cancellation_factor_bf16']:9.3g} "
        f"{solvers['doubling_bf16']['backward_error']:9.2g} "
        f"{solvers['doubling_bf16']['forward_error']:9.2g} "
        f"{solvers['substitution_bf16']['backward_error']:9.2g} "
        f"{record['reference_method']:>15}"
    )
  print("=" * 124)
  print("cancel is sum of stage increment norms over final solution norm.")
  print("Growth alone does not predict backward error; the cancellation factor")
  print("separates regimes that grow harmlessly from those that grow and cancel.")
  print("kappa bounds how far a backward error may be amplified into forward")
  print("error; it does not predict the realized forward error for a given RHS.")


if __name__ == "__main__":
  main()
