#!/usr/bin/env python3
"""Benchmark MaxText KDA's generic and analytical backward paths on one TPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp

from maxtext.layers.kimi_delta_attention import chunk_kda


def _block_until_ready(tree):
  jax.tree.map(lambda value: value.block_until_ready(), tree)


def _measure(name, function, arguments, repetitions):
  start = time.perf_counter()
  compiled = jax.jit(function).lower(*arguments).compile()
  compile_seconds = time.perf_counter() - start
  _block_until_ready(compiled(*arguments))

  memory = compiled.memory_analysis()
  memory_bytes = None
  if memory is not None:
    memory_bytes = {
        "argument": memory.argument_size_in_bytes,
        "output": memory.output_size_in_bytes,
        "temporary": memory.temp_size_in_bytes,
        "alias": memory.alias_size_in_bytes,
        "total": (
            memory.argument_size_in_bytes
            + memory.output_size_in_bytes
            + memory.temp_size_in_bytes
            - memory.alias_size_in_bytes
        ),
    }

  samples = []
  for _ in range(repetitions):
    start = time.perf_counter()
    output = compiled(*arguments)
    _block_until_ready(output)
    samples.append(1_000 * (time.perf_counter() - start))
  return {
      "name": name,
      "compile_seconds": compile_seconds,
      "mean_milliseconds": statistics.fmean(samples),
      "median_milliseconds": statistics.median(samples),
      "min_milliseconds": min(samples),
      "max_milliseconds": max(samples),
      "repetitions": repetitions,
      "compiled_memory_bytes": memory_bytes,
  }


def _make_inputs(args):
  keys = jax.random.split(jax.random.key(args.seed), 6)
  shape = (args.batch, args.sequence_length, args.heads, args.key_dim)
  query = jax.random.normal(keys[0], shape, dtype=jnp.bfloat16)
  key = jax.random.normal(keys[1], shape, dtype=jnp.bfloat16)
  value = jax.random.normal(
      keys[2],
      (args.batch, args.sequence_length, args.heads, args.value_dim),
      dtype=jnp.bfloat16,
  )
  log_decay = -0.01 - 0.03 * jax.nn.sigmoid(jax.random.normal(keys[3], shape, dtype=jnp.float32))
  beta = jax.nn.sigmoid(
      jax.random.normal(
          keys[4],
          (args.batch, args.sequence_length, args.heads),
          dtype=jnp.float32,
      )
  )
  initial_state = (
      0.01
      * jax.random.normal(
          keys[5],
          (args.batch, args.heads, args.key_dim, args.value_dim),
          dtype=jnp.float32,
      )
  ).astype(jnp.bfloat16)
  return query, key, value, log_decay, beta, initial_state


def _bind(chunk_size, use_analytical_custom_vjp):
  def forward(query, key, value, log_decay, beta, initial_state):
    return chunk_kda(
        query,
        key,
        value,
        log_decay,
        beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        use_qk_norm=True,
        compute_dtype=jnp.bfloat16,
        use_pallas_blocked_solve=False,
        use_analytical_custom_vjp=use_analytical_custom_vjp,
    )

  def loss_and_grad(query, key, value, log_decay, beta, initial_state):
    def loss(q, k, v, g, b, state):
      output, final_state = forward(q, k, v, g, b, state)
      return jnp.mean(output.astype(jnp.float32) ** 2) + 1e-3 * jnp.mean(
          final_state.astype(jnp.float32) ** 2
      )

    return jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
    )

  return forward, loss_and_grad


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--paths", default="generic,analytical")
  parser.add_argument("--batch", type=int, default=8)
  parser.add_argument("--sequence-length", type=int, default=2048)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--key-dim", type=int, default=128)
  parser.add_argument("--value-dim", type=int, default=128)
  parser.add_argument("--chunk-size", type=int, default=64)
  parser.add_argument("--forward-repetitions", type=int, default=20)
  parser.add_argument("--backward-repetitions", type=int, default=10)
  parser.add_argument("--seed", type=int, default=61)
  args = parser.parse_args()

  implementations = {
      "generic": False,
      "analytical": True,
  }
  selected_paths = [path.strip() for path in args.paths.split(",") if path.strip()]
  unknown_paths = sorted(set(selected_paths) - implementations.keys())
  if unknown_paths:
    raise ValueError(f"unknown paths: {unknown_paths}")

  inputs = _make_inputs(args)
  results = []
  for path in selected_paths:
    forward, loss_and_grad = _bind(
        args.chunk_size,
        implementations[path],
    )
    results.append(
        _measure(
            f"{path}, forward",
            forward,
            inputs,
            args.forward_repetitions,
        )
    )
    results.append(
        _measure(
            f"{path}, forward+backward",
            loss_and_grad,
            inputs,
            args.backward_repetitions,
        )
    )

  tokens = args.batch * args.sequence_length
  for result in results:
    result["tokens_per_second"] = tokens / (result["mean_milliseconds"] / 1_000)

  print(
      json.dumps(
          {
              "backend": jax.default_backend(),
              "device": str(jax.devices()[0]),
              "shape": {
                  "batch": args.batch,
                  "sequence_length": args.sequence_length,
                  "heads": args.heads,
                  "key_dim": args.key_dim,
                  "value_dim": args.value_dim,
                  "chunk_size": args.chunk_size,
                  "tokens": tokens,
                  "dtype": "bfloat16",
                  "qk_l2norm": True,
              },
              "results": results,
          },
          indent=2,
      )
  )


if __name__ == "__main__":
  main()
