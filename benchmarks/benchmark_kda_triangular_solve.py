#!/usr/bin/env python3
"""Benchmark KDA's triangular solve alternatives on one TPU device."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
from jax import lax
import jax.numpy as jnp

from maxtext.layers.kimi_delta_attention import blocked_unit_lower_solve


def _xla_direct_impl(system, rhs, *, transpose: bool):
  width = system.shape[-1]
  matrix = jnp.eye(width, dtype=jnp.float32) + jnp.tril(system, k=-1)
  if transpose:
    matrix = matrix.swapaxes(-1, -2)
  return jax.scipy.linalg.solve_triangular(
      matrix,
      rhs,
      lower=not transpose,
      unit_diagonal=True,
  )


@jax.custom_vjp
def xla_direct_solve(system, rhs):
  return _xla_direct_impl(system, rhs, transpose=False)


def _xla_direct_solve_fwd(system, rhs):
  solution = _xla_direct_impl(system, rhs, transpose=False)
  return solution, (system, solution)


def _xla_direct_solve_bwd(residuals, solution_cotangent):
  system, solution = residuals
  rhs_cotangent = _xla_direct_impl(system, solution_cotangent, transpose=True)
  system_cotangent = jnp.tril(
      -jnp.matmul(
          rhs_cotangent,
          solution.swapaxes(-1, -2),
          precision=lax.Precision.HIGHEST,
      ),
      k=-1,
  )
  return system_cotangent, rhs_cotangent


xla_direct_solve.defvjp(_xla_direct_solve_fwd, _xla_direct_solve_bwd)


def inverse_then_two_matmuls(system, rhs):
  """Matches the selected KDA path: solve for A^-1, then form U and W."""
  width = system.shape[-1]
  split = rhs.shape[-1] // 2
  identity = jnp.broadcast_to(jnp.eye(width, dtype=jnp.float32), system.shape)
  inverse = jax.scipy.linalg.solve_triangular(
      identity + system,
      identity,
      lower=True,
      unit_diagonal=True,
  )
  left = jnp.matmul(inverse, rhs[..., :split], precision=lax.Precision.HIGHEST)
  right = jnp.matmul(inverse, rhs[..., split:], precision=lax.Precision.HIGHEST)
  return jnp.concatenate((left, right), axis=-1)


def _block_until_ready(tree):
  jax.tree.map(lambda value: value.block_until_ready(), tree)


def _measure(name, function, arguments, repetitions):
  compiled = jax.jit(function)
  start = time.perf_counter()
  _block_until_ready(compiled(*arguments))
  compile_seconds = time.perf_counter() - start

  samples = []
  for _ in range(repetitions):
    start = time.perf_counter()
    _block_until_ready(compiled(*arguments))
    samples.append(1_000 * (time.perf_counter() - start))
  return {
      "name": name,
      "compile_seconds": compile_seconds,
      "mean_milliseconds": statistics.fmean(samples),
      "median_milliseconds": statistics.median(samples),
      "min_milliseconds": min(samples),
      "max_milliseconds": max(samples),
      "repetitions": repetitions,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--programs", type=int, default=2048)
  parser.add_argument("--chunk-size", type=int, default=64)
  parser.add_argument("--rhs-width", type=int, default=256)
  parser.add_argument("--repetitions", type=int, default=20)
  args = parser.parse_args()

  key = jax.random.key(31)
  system = jnp.tril(
      0.003
      * jax.random.normal(
          key,
          (args.programs, args.chunk_size, args.chunk_size),
          dtype=jnp.float32,
      ),
      k=-1,
  )
  rhs = jax.random.normal(
      jax.random.fold_in(key, 1),
      (args.programs, args.chunk_size, args.rhs_width),
      dtype=jnp.float32,
  )
  cotangent = jax.random.normal(jax.random.fold_in(key, 2), rhs.shape, dtype=jnp.float32)

  paths = (
      ("XLA inverse + two matmuls", inverse_then_two_matmuls),
      ("XLA direct + custom VJP", xla_direct_solve),
      ("Pallas blocked + custom VJP", blocked_unit_lower_solve),
  )
  results = []
  for label, function in paths:
    results.append(
        _measure(
            f"{label}, forward",
            function,
            (system, rhs),
            args.repetitions,
        )
    )
    results.append(
        _measure(
            f"{label}, forward+VJP",
            lambda a, b, g, solve=function: jax.vjp(solve, a, b)[1](g),
            (system, rhs, cotangent),
            max(5, args.repetitions // 2),
        )
    )

  print(
      json.dumps(
          {
              "backend": jax.default_backend(),
              "device": str(jax.devices()[0]),
              "shape": {
                  "systems": args.programs,
                  "chunk_size": args.chunk_size,
                  "rhs_width": args.rhs_width,
              },
              "results": results,
          },
          indent=2,
      )
  )


if __name__ == "__main__":
  main()
