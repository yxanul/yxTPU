#!/usr/bin/env python3
"""Benchmark ejkernel's unmodified XLA gated-delta implementations on TPU.

The source files are loaded directly from an ejkernel checkout so importing
the full multi-backend package is unnecessary. The benchmark intentionally
exercises both:

* ``_chunk_gdr_fwd``: the active exact triangular-solve path with JAX autodiff.
* ``_chunk_gdr_fwd_neumann``: the private custom-VJP path backed by
  ``_xla_impl_bwd.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import types

import jax
import jax.numpy as jnp


def _load_ejkernel_gdr(source_root: Path):
  source_dir = source_root / "ejkernel/kernels/_xla/gated_delta_rule"
  package_name = "_ejkernel_gdr_snapshot"
  package = types.ModuleType(package_name)
  package.__path__ = [str(source_dir)]
  sys.modules[package_name] = package

  modules = {}
  for short_name in ("_xla_impl_fwd", "_xla_impl_bwd"):
    module_name = f"{package_name}.{short_name}"
    spec = importlib.util.spec_from_file_location(module_name, source_dir / f"{short_name}.py")
    if spec is None or spec.loader is None:
      raise RuntimeError(f"could not load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    modules[short_name] = module
  return modules["_xla_impl_fwd"]


def _block_until_ready(tree):
  jax.tree.map(lambda value: value.block_until_ready(), tree)


def _measure(name, function, arguments, repetitions):
  jitted = jax.jit(function)
  start = time.perf_counter()
  compiled = jitted.lower(*arguments).compile()
  compile_seconds = time.perf_counter() - start
  compiled_output = compiled(*arguments)
  _block_until_ready(compiled_output)

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
  keys = jax.random.split(jax.random.key(args.seed), 5)
  shape = (args.batch, args.heads, args.sequence_length, args.key_dim)
  value_shape = (args.batch, args.heads, args.sequence_length, args.value_dim)
  scalar_shape = shape[:-1]
  query = jax.random.normal(keys[0], shape, dtype=jnp.bfloat16)
  key = jax.random.normal(keys[1], shape, dtype=jnp.bfloat16)
  value = jax.random.normal(keys[2], value_shape, dtype=jnp.bfloat16)
  beta = jax.nn.sigmoid(jax.random.normal(keys[3], scalar_shape, dtype=jnp.float32)).astype(
      jnp.bfloat16
  )
  decay = (
      -0.01 - 0.03 * jax.nn.sigmoid(jax.random.normal(keys[4], scalar_shape, dtype=jnp.float32))
  ).astype(jnp.bfloat16)
  initial_state = jnp.zeros(
      (args.batch, args.heads, args.key_dim, args.value_dim),
      dtype=jnp.bfloat16,
  )
  return query, key, value, beta, decay, initial_state


def _bind(function, chunk_size, use_qk_l2norm):
  def forward(query, key, value, beta, decay, initial_state):
    return function(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
    )

  def loss_and_grad(query, key, value, beta, decay, initial_state):
    def loss(q, k, v, b, g, state):
      output, final_state = forward(q, k, v, b, g, state)
      return jnp.mean(output.astype(jnp.float32) ** 2) + 1e-3 * jnp.mean(
          final_state.astype(jnp.float32) ** 2
      )

    return jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))(
        query,
        key,
        value,
        beta,
        decay,
        initial_state,
    )

  return forward, loss_and_grad


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-root", type=Path, required=True)
  parser.add_argument("--source-commit", default="unknown")
  parser.add_argument("--paths", default="exact,custom_vjp")
  parser.add_argument("--batch", type=int, default=8)
  parser.add_argument("--sequence-length", type=int, default=2048)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--key-dim", type=int, default=128)
  parser.add_argument("--value-dim", type=int, default=128)
  parser.add_argument("--chunk-size", type=int, default=64)
  parser.add_argument("--forward-repetitions", type=int, default=20)
  parser.add_argument("--backward-repetitions", type=int, default=10)
  parser.add_argument("--seed", type=int, default=41)
  parser.add_argument("--disable-qk-l2norm", action="store_true")
  args = parser.parse_args()

  module = _load_ejkernel_gdr(args.source_root)
  implementations = {
      "exact": module._chunk_gdr_fwd,
      "custom_vjp": module._chunk_gdr_fwd_neumann,
  }
  selected_paths = [path.strip() for path in args.paths.split(",") if path.strip()]
  unknown_paths = sorted(set(selected_paths) - implementations.keys())
  if unknown_paths:
    raise ValueError(f"unknown paths: {unknown_paths}")

  inputs = _make_inputs(args)
  results = []
  for path in selected_paths:
    forward, loss_and_grad = _bind(
        implementations[path],
        args.chunk_size,
        not args.disable_qk_l2norm,
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
              "source_commit": args.source_commit,
              "shape": {
                  "batch": args.batch,
                  "sequence_length": args.sequence_length,
                  "heads": args.heads,
                  "key_dim": args.key_dim,
                  "value_dim": args.value_dim,
                  "chunk_size": args.chunk_size,
                  "tokens": tokens,
                  "dtype": "bfloat16",
                  "qk_l2norm": not args.disable_qk_l2norm,
              },
              "results": results,
          },
          indent=2,
      )
  )


if __name__ == "__main__":
  main()
