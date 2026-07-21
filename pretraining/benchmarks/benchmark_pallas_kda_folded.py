#!/usr/bin/env python3
"""Validate and benchmark the conv-folded production Pallas KDA kernel.

The production kernel consumes the raw fused-projection output
``[B, T, 3, H, D]`` and owns the causal depthwise convolution, SiLU, Q/K
normalization, and chunked recurrence. The reference applies the same
convolution and activation in XLA and then runs the analytical-VJP chunked
KDA, which is the equation-level reference the earlier unfolded kernel was
qualified against.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp

from yxtpu_pretrain.kernels.kda_fused_pallas import pallas_kda_fused
from yxtpu_pretrain.layers.kimi_delta_attention import chunk_kda


def _block_until_ready(tree):
  jax.tree.map(lambda value: value.block_until_ready(), tree)


def _make_inputs(batch, sequence_length, heads, width, seed, decay_lower_bound=-5.0):
  keys = jax.random.split(jax.random.key(seed), 6)
  raw_qkv = jax.random.normal(
      keys[0],
      (batch, sequence_length, 3, heads, 128),
      dtype=jnp.bfloat16,
  )
  conv_weight = 0.5 * jax.random.normal(
      keys[1],
      (width, 3, heads, 128),
      dtype=jnp.float32,
  )
  log_decay = decay_lower_bound * jax.nn.sigmoid(
      jax.random.normal(keys[2], (batch, sequence_length, heads, 128), dtype=jnp.float32)
  )
  beta = jax.nn.sigmoid(
      jax.random.normal(keys[3], (batch, sequence_length, heads), dtype=jnp.float32)
  )
  initial_state = 0.01 * jax.random.normal(
      keys[4],
      (batch, heads, 128, 128),
      dtype=jnp.float32,
  )
  return raw_qkv, conv_weight, log_decay, beta, initial_state


def _reference_mixer(raw_qkv, conv_weight):
  """Causal depthwise conv + SiLU on ``[B, T, 3, H, D]``, matching the layer."""
  width = conv_weight.shape[0]
  padded = jnp.pad(
      raw_qkv.astype(jnp.float32),
      ((0, 0), (width - 1, 0), (0, 0), (0, 0), (0, 0)),
  )
  sequence_length = raw_qkv.shape[1]
  convolved = None
  for tap in range(width):
    term = padded[:, tap : tap + sequence_length] * conv_weight[tap]
    convolved = term if convolved is None else convolved + term
  activated = jax.nn.silu(convolved).astype(jnp.bfloat16)
  return activated[:, :, 0], activated[:, :, 1], activated[:, :, 2]


@jax.jit
def _reference_forward(raw_qkv, conv_weight, log_decay, beta, initial_state):
  query, key, value = _reference_mixer(raw_qkv, conv_weight)
  return chunk_kda(
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
      use_analytical_custom_vjp=True,
  )


@jax.jit
def _fused_forward(raw_qkv, conv_weight, log_decay, beta, initial_state):
  return pallas_kda_fused(raw_qkv, conv_weight, log_decay, beta, initial_state)


def _loss_from(forward):
  def loss(*inputs):
    output, final_state = forward(*inputs)
    return jnp.mean(output.astype(jnp.float32) ** 2) + 1e-3 * jnp.mean(
        final_state.astype(jnp.float32) ** 2
    )

  return loss


_reference_loss_and_grad = jax.jit(
    jax.value_and_grad(_loss_from(_reference_forward), argnums=(0, 1, 2, 3, 4))
)
_fused_loss_and_grad = jax.jit(
    jax.value_and_grad(_loss_from(_fused_forward), argnums=(0, 1, 2, 3, 4))
)


def _error_stats(actual, expected):
  actual = jnp.asarray(actual, dtype=jnp.float32)
  expected = jnp.asarray(expected, dtype=jnp.float32)
  difference = jnp.abs(actual - expected)
  return {
      "max_abs": float(jnp.max(difference)),
      "rmse": float(jnp.sqrt(jnp.mean(difference**2))),
      "reference_rms": float(jnp.sqrt(jnp.mean(expected**2))),
      "all_finite": bool(jnp.all(jnp.isfinite(actual))),
  }


def _measure(name, function, inputs, repetitions, tokens):
  compile_start = time.perf_counter()
  _block_until_ready(function(*inputs))
  compile_seconds = time.perf_counter() - compile_start
  timings = []
  for _ in range(repetitions):
    start = time.perf_counter()
    _block_until_ready(function(*inputs))
    timings.append((time.perf_counter() - start) * 1_000)
  mean_ms = statistics.fmean(timings)
  return {
      "name": name,
      "compile_seconds": compile_seconds,
      "mean_milliseconds": mean_ms,
      "median_milliseconds": statistics.median(timings),
      "min_milliseconds": min(timings),
      "repetitions": repetitions,
      "tokens_per_second": tokens / (mean_ms / 1_000),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch", type=int, default=8)
  parser.add_argument("--sequence-length", type=int, default=2048)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--conv-width", type=int, default=4)
  parser.add_argument("--repetitions", type=int, default=20)
  parser.add_argument("--seed", type=int, default=71)
  parser.add_argument("--output-atol", type=float, default=0.05)
  parser.add_argument("--state-atol", type=float, default=0.05)
  parser.add_argument("--gradient-atol", type=float, default=0.01)
  parser.add_argument("--correctness-only", action="store_true")
  args = parser.parse_args()

  correctness_inputs = _make_inputs(2, 256, args.heads, args.conv_width, args.seed)
  expected_output, expected_state = _reference_forward(*correctness_inputs)
  actual_output, actual_state = _fused_forward(*correctness_inputs)
  expected_loss, expected_grads = _reference_loss_and_grad(*correctness_inputs)
  actual_loss, actual_grads = _fused_loss_and_grad(*correctness_inputs)

  gradient_names = ("raw_qkv", "conv_weight", "log_decay", "beta", "initial_state")
  correctness = {
      "output": _error_stats(actual_output, expected_output),
      "final_state": _error_stats(actual_state, expected_state),
      "loss_abs_difference": float(abs(actual_loss - expected_loss)),
      "gradients": {
          name: _error_stats(actual_grad, expected_grad)
          for name, actual_grad, expected_grad in zip(
              gradient_names, actual_grads, expected_grads, strict=True
          )
      },
  }
  failures = []
  if correctness["output"]["max_abs"] > args.output_atol:
    failures.append("output")
  if correctness["final_state"]["max_abs"] > args.state_atol:
    failures.append("final_state")
  for name in gradient_names:
    stats = correctness["gradients"][name]
    if not stats["all_finite"] or stats["max_abs"] > args.gradient_atol:
      failures.append(f"grad_{name}")
  correctness["failures"] = failures

  result = {
      "backend": jax.default_backend(),
      "device": str(jax.devices()[0]),
      "correctness": correctness,
  }
  if not args.correctness_only:
    inputs = _make_inputs(
        args.batch, args.sequence_length, args.heads, args.conv_width, args.seed
    )
    tokens = args.batch * args.sequence_length
    result["production_shape"] = {
        "batch": args.batch,
        "sequence_length": args.sequence_length,
        "heads": args.heads,
        "tokens": tokens,
    }
    result["benchmarks"] = [
        _measure("XLA mixer + analytical chunked forward", _reference_forward, inputs, args.repetitions, tokens),
        _measure(
            "XLA mixer + analytical forward+backward",
            _reference_loss_and_grad,
            inputs,
            max(5, args.repetitions // 2),
            tokens,
        ),
        _measure("Folded Pallas forward", _fused_forward, inputs, args.repetitions, tokens),
        _measure(
            "Folded Pallas forward+backward",
            _fused_loss_and_grad,
            inputs,
            max(5, args.repetitions // 2),
            tokens,
        ),
    ]
  print(json.dumps(result, indent=2))
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
