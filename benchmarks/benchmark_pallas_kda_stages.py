#!/usr/bin/env python3
"""Measure cumulative stages inside the fused Pallas KDA forward and backward."""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import time

import jax
import jax.numpy as jnp

from maxtext.kernels.kda_fused_pallas import pallas_kda_fused_backward
from maxtext.kernels.kda_fused_pallas import pallas_kda_fused_forward


def _block_until_ready(tree):
  jax.tree.map(lambda value: value.block_until_ready(), tree)


def _make_inputs(batch, sequence_length, heads, seed):
  keys = jax.random.split(jax.random.key(seed), 8)
  shape = (batch, sequence_length, heads, 128)
  query = jax.random.normal(keys[0], shape, dtype=jnp.bfloat16)
  key = jax.random.normal(keys[1], shape, dtype=jnp.bfloat16)
  value = jax.random.normal(keys[2], shape, dtype=jnp.bfloat16)
  log_decay = -0.01 - 0.03 * jax.nn.sigmoid(jax.random.normal(keys[3], shape, dtype=jnp.float32))
  beta = jax.nn.sigmoid(
      jax.random.normal(
          keys[4],
          (batch, sequence_length, heads),
          dtype=jnp.float32,
      )
  )
  initial_state = (
      0.01
      * jax.random.normal(
          keys[5],
          (batch, heads, 128, 128),
          dtype=jnp.float32,
      )
  ).astype(jnp.float32)
  output_cotangent = jax.random.normal(keys[6], shape, dtype=jnp.bfloat16)
  final_state_cotangent = jax.random.normal(
      keys[7],
      (batch, heads, 128, 128),
      dtype=jnp.float32,
  )
  return (
      (query, key, value, log_decay, beta, initial_state),
      output_cotangent,
      final_state_cotangent,
  )


def _measure(name, function, arguments, repetitions):
  start = time.perf_counter()
  compiled = jax.jit(function).lower(*arguments).compile()
  compile_seconds = time.perf_counter() - start
  _block_until_ready(compiled(*arguments))

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
  parser.add_argument("--batch", type=int, default=8)
  parser.add_argument("--sequence-length", type=int, default=2048)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--repetitions", type=int, default=10)
  parser.add_argument("--seed", type=int, default=83)
  args = parser.parse_args()

  inputs, output_cotangent, final_state_cotangent = _make_inputs(
      args.batch,
      args.sequence_length,
      args.heads,
      args.seed,
  )
  _, _, state_history = pallas_kda_fused_forward(
      *inputs,
      chunk_size=64,
      use_qk_norm=True,
      solve_method="doubling",
      profile_stage="full",
  )
  state_history.block_until_ready()

  forward_results = []
  for stage in ("preprocess", "pairwise", "solve", "full"):
    forward_results.append(
        _measure(
            f"forward through {stage}",
            functools.partial(
                pallas_kda_fused_forward,
                chunk_size=64,
                use_qk_norm=True,
                solve_method="doubling",
                profile_stage=stage,
            ),
            inputs,
            args.repetitions,
        )
    )

  backward_arguments = (
      *inputs,
      state_history,
      output_cotangent,
      final_state_cotangent,
  )
  backward_results = []
  for stage in ("reverse_state", "solve_vjp", "pairwise_vjp", "full"):
    backward_results.append(
        _measure(
            f"backward through {stage}",
            functools.partial(
                pallas_kda_fused_backward,
                chunk_size=64,
                use_qk_norm=True,
                profile_stage=stage,
            ),
            backward_arguments,
            args.repetitions,
        )
    )

  def add_incremental(results):
    previous = 0.0
    for result in results:
      result["incremental_milliseconds_from_previous_stage"] = (
          result["mean_milliseconds"] - previous
      )
      previous = result["mean_milliseconds"]

  add_incremental(forward_results)
  add_incremental(backward_results)
  print(
      json.dumps(
          {
              "backend": jax.default_backend(),
              "device": str(jax.devices()[0]),
              "shape": {
                  "batch": args.batch,
                  "sequence_length": args.sequence_length,
                  "heads": args.heads,
                  "key_dim": 128,
                  "value_dim": 128,
                  "chunk_size": 64,
                  "chunks": args.sequence_length // 64,
                  "tokens": args.batch * args.sequence_length,
              },
              "interpretation": (
                  "Each row is cumulative within one kernel variant. "
                  "Differences estimate incremental stage cost; all variants "
                  "use identical input/output block traffic."
              ),
              "forward": forward_results,
              "backward": backward_results,
          },
          indent=2,
      )
  )


if __name__ == "__main__":
  main()
