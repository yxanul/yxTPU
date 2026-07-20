#!/usr/bin/env python3
"""Validate and benchmark the fused production-shape Pallas KDA kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp

from maxtext.kernels.kda_fused_pallas import pallas_kda_fused_forward
from maxtext.kernels.kda_fused_pallas import pallas_kda_fused
from maxtext.layers.kimi_delta_attention import _chunk_kda_impl
from maxtext.layers.kimi_delta_attention import chunk_kda
from maxtext.layers.kimi_delta_attention import recurrent_kda_reference


def _block_until_ready(tree):
  jax.tree.map(lambda value: value.block_until_ready(), tree)


def _make_inputs(batch, sequence_length, heads, seed):
  keys = jax.random.split(jax.random.key(seed), 6)
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
  return query, key, value, log_decay, beta, initial_state


def _reference_forward(query, key, value, log_decay, beta, initial_state):
  return _chunk_kda_impl(
      query,
      key,
      value,
      log_decay,
      beta,
      chunk_size=64,
      initial_state=initial_state,
      use_qk_norm=True,
      compute_dtype=jnp.bfloat16,
      use_pallas_blocked_solve=False,
  )


def _recurrent_forward(query, key, value, log_decay, beta, initial_state):
  return recurrent_kda_reference(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state=initial_state,
      use_qk_norm=True,
  )


def _fused_forward_with_solver(
    query,
    key,
    value,
    log_decay,
    beta,
    initial_state,
    *,
    solve_method,
):
  return pallas_kda_fused_forward(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      chunk_size=64,
      use_qk_norm=True,
      solve_method=solve_method,
  )


def _fused_forward(query, key, value, log_decay, beta, initial_state):
  return _fused_forward_with_solver(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      solve_method="doubling",
  )


def _fused_blocked_forward(query, key, value, log_decay, beta, initial_state):
  return _fused_forward_with_solver(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      solve_method="blocked",
  )


def _loss(forward, *inputs):
  output, final_state = forward(*inputs)
  return jnp.mean(output.astype(jnp.float32) ** 2) + 1e-3 * jnp.mean(
      final_state.astype(jnp.float32) ** 2
  )


def _reference_loss_and_grad(query, key, value, log_decay, beta, initial_state):
  def forward(q, k, v, g, b, state):
    return chunk_kda(
        q,
        k,
        v,
        g,
        b,
        chunk_size=64,
        initial_state=state,
        use_qk_norm=True,
        compute_dtype=jnp.bfloat16,
        use_pallas_blocked_solve=False,
        use_analytical_custom_vjp=True,
    )

  return jax.value_and_grad(lambda *xs: _loss(forward, *xs), argnums=(0, 1, 2, 3, 4, 5))(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
  )


def _fused_loss_and_grad(query, key, value, log_decay, beta, initial_state):
  return jax.value_and_grad(
      lambda *xs: _loss(pallas_kda_fused, *xs),
      argnums=(0, 1, 2, 3, 4, 5),
  )(query, key, value, log_decay, beta, initial_state)


def _error(actual, expected):
  actual_f32 = actual.astype(jnp.float32)
  expected_f32 = expected.astype(jnp.float32)
  difference = actual_f32 - expected_f32
  return {
      "max_abs": float(jnp.max(jnp.abs(difference))),
      "mean_abs": float(jnp.mean(jnp.abs(difference))),
      "rmse": float(jnp.sqrt(jnp.mean(difference * difference))),
      "reference_rms": float(jnp.sqrt(jnp.mean(expected_f32 * expected_f32))),
      "all_finite": bool(jnp.all(jnp.isfinite(actual_f32))),
  }


def _correctness(args):
  inputs = _make_inputs(
      args.correctness_batch,
      args.correctness_sequence_length,
      args.correctness_heads,
      args.seed,
  )
  fused_output, fused_final_state, state_history = _fused_forward(*inputs)
  xla_output, xla_final_state = _reference_forward(*inputs)
  recurrent_output, recurrent_final_state = _recurrent_forward(*inputs)
  fused_loss, fused_gradients = _fused_loss_and_grad(*inputs)
  reference_loss, reference_gradients = _reference_loss_and_grad(*inputs)
  _block_until_ready(
      (
          fused_output,
          fused_final_state,
          state_history,
          xla_output,
          xla_final_state,
          recurrent_output,
          recurrent_final_state,
          fused_loss,
          fused_gradients,
          reference_loss,
          reference_gradients,
      )
  )

  result = {
      "shape": {
          "batch": args.correctness_batch,
          "sequence_length": args.correctness_sequence_length,
          "heads": args.correctness_heads,
          "key_dim": 128,
          "value_dim": 128,
          "chunk_size": 64,
      },
      "fused_vs_recurrent": {
          "output": _error(fused_output, recurrent_output),
          "final_state": _error(fused_final_state, recurrent_final_state),
      },
      "fused_vs_xla_chunked": {
          "output": _error(fused_output, xla_output),
          "final_state": _error(fused_final_state, xla_final_state),
      },
      "state_history_shape": list(state_history.shape),
      "loss": {
          "fused": float(fused_loss),
          "analytical_xla": float(reference_loss),
          "absolute_difference": float(jnp.abs(fused_loss - reference_loss)),
      },
      "gradient_errors_vs_analytical_xla": {
          name: _error(actual, expected)
          for name, actual, expected in zip(
              ("query", "key", "value", "log_decay", "beta", "initial_state"),
              fused_gradients,
              reference_gradients,
              strict=True,
          )
      },
  }
  for comparison in ("fused_vs_recurrent", "fused_vs_xla_chunked"):
    for tensor in ("output", "final_state"):
      metrics = result[comparison][tensor]
      if not metrics["all_finite"]:
        raise AssertionError(f"{comparison} {tensor} contains non-finite values")
      threshold = args.output_atol if tensor == "output" else args.state_atol
      if metrics["max_abs"] > threshold:
        raise AssertionError(
            f"{comparison} {tensor} max_abs={metrics['max_abs']:.6g} exceeds {threshold}"
        )
  for name, metrics in result["gradient_errors_vs_analytical_xla"].items():
    if not metrics["all_finite"]:
      raise AssertionError(f"{name} gradient contains non-finite values")
    if metrics["max_abs"] > args.gradient_atol:
      raise AssertionError(
          f"{name} gradient max_abs={metrics['max_abs']:.6g} exceeds {args.gradient_atol}"
      )
  return result


def _measure(name, function, arguments, repetitions, tokens):
  start = time.perf_counter()
  compiled = jax.jit(function).lower(*arguments).compile()
  compile_seconds = time.perf_counter() - start
  _block_until_ready(compiled(*arguments))

  samples = []
  for _ in range(repetitions):
    start = time.perf_counter()
    output = compiled(*arguments)
    _block_until_ready(output)
    samples.append(1_000 * (time.perf_counter() - start))

  memory = compiled.memory_analysis()
  memory_result = None
  if memory is not None:
    memory_result = {
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
  mean_milliseconds = statistics.fmean(samples)
  return {
      "name": name,
      "compile_seconds": compile_seconds,
      "mean_milliseconds": mean_milliseconds,
      "median_milliseconds": statistics.median(samples),
      "min_milliseconds": min(samples),
      "max_milliseconds": max(samples),
      "repetitions": repetitions,
      "tokens_per_second": tokens / (mean_milliseconds / 1_000),
      "compiled_memory_bytes": memory_result,
  }


def _production_benchmark(args):
  inputs = _make_inputs(
      args.batch,
      args.sequence_length,
      args.heads,
      args.seed + 1,
  )
  tokens = args.batch * args.sequence_length
  results = [
      _measure(
          "XLA chunked forward",
          _reference_forward,
          inputs,
          args.repetitions,
          tokens,
      ),
      _measure(
          "Analytical XLA forward+backward",
          _reference_loss_and_grad,
          inputs,
          max(5, args.repetitions // 2),
          tokens,
      ),
      _measure(
          "Pallas fused forward, doubling solve + boundary states",
          _fused_forward,
          inputs,
          args.repetitions,
          tokens,
      ),
      _measure(
          "Pallas fused forward+backward",
          _fused_loss_and_grad,
          inputs,
          max(5, args.repetitions // 2),
          tokens,
      ),
  ]
  if args.include_blocked:
    results.append(
        _measure(
            "Pallas fused forward, blocked solve + boundary states",
            _fused_blocked_forward,
            inputs,
            args.repetitions,
            tokens,
        )
    )
  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch", type=int, default=8)
  parser.add_argument("--sequence-length", type=int, default=2048)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--correctness-batch", type=int, default=1)
  parser.add_argument("--correctness-sequence-length", type=int, default=128)
  parser.add_argument("--correctness-heads", type=int, default=1)
  parser.add_argument("--output-atol", type=float, default=0.04)
  parser.add_argument("--state-atol", type=float, default=0.04)
  parser.add_argument("--gradient-atol", type=float, default=0.002)
  parser.add_argument("--repetitions", type=int, default=20)
  parser.add_argument("--seed", type=int, default=71)
  parser.add_argument("--correctness-only", action="store_true")
  parser.add_argument("--include-blocked", action="store_true")
  args = parser.parse_args()

  result = {
      "backend": jax.default_backend(),
      "device": str(jax.devices()[0]),
      "correctness": _correctness(args),
  }
  if not args.correctness_only:
    result["production_shape"] = {
        "batch": args.batch,
        "sequence_length": args.sequence_length,
        "heads": args.heads,
        "key_dim": 128,
        "value_dim": 128,
        "chunk_size": 64,
        "tokens": args.batch * args.sequence_length,
    }
    result["benchmarks"] = _production_benchmark(args)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
