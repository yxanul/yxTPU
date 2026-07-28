# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""On-device equivalence gate between two builds of the v4 fused KDA kernel.

Runs the full custom-VJP path (fused forward + split backward) of a candidate
module and a reference module on identical inputs and compares all eight
outputs: output, final state, and the six input cotangents.

Usage on a v4 worker (single-host env restriction applies on pod slices):

  git show main:pretraining/src/yxtpu_pretrain/kernels/kda_fused_pallas_v4.py \
      > /tmp/kda_main.py
  python benchmarks/verify_kda_v4_kernel_equivalence.py \
      --reference /tmp/kda_main.py --bitwise        # A1 gate: bit-identical
  python benchmarks/verify_kda_v4_kernel_equivalence.py \
      --reference /tmp/kda_main.py                  # A4 gate: closeness report

Without --candidate the installed production module is the candidate.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _inputs(batch: int, sequence_length: int, heads: int, seed: int):
    keys = jax.random.split(jax.random.key(seed), 7)
    qkv_shape = (batch, sequence_length, heads, 128)
    query = jax.random.normal(keys[0], qkv_shape, jnp.bfloat16)
    key = jax.random.normal(keys[1], qkv_shape, jnp.bfloat16)
    value = jax.random.normal(keys[2], qkv_shape, jnp.bfloat16)
    # Safe-gate-shaped decay across the full production range [-5, 0).
    log_decay = -5.0 * jax.nn.sigmoid(
        jax.random.normal(keys[3], qkv_shape, jnp.float32)
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(keys[4], (batch, sequence_length, heads), jnp.float32)
    )
    initial_state = jnp.zeros((batch, heads, 128, 128), jnp.float32)
    output_cotangent = jax.random.normal(keys[5], qkv_shape, jnp.float32).astype(
        jnp.bfloat16
    )
    state_cotangent = 0.05 * jax.random.normal(
        keys[6], (batch, heads, 128, 128), jnp.float32
    )
    return (query, key, value, log_decay, beta, initial_state), (
        output_cotangent,
        state_cotangent,
    )


_OUTPUT_NAMES = (
    "output",
    "final_state",
    "d_query",
    "d_key",
    "d_value",
    "d_log_decay",
    "d_beta",
    "d_initial_state",
)


def _run(module, primals, cotangents):
    outputs, vjp = jax.vjp(module.pallas_kda_fused_v4, *primals)
    gradients = vjp(cotangents)
    results = [*outputs, *gradients]
    jax.block_until_ready(results)
    return [np.asarray(value) for value in results]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="reference module path")
    parser.add_argument(
        "--candidate",
        default=None,
        help="candidate module path (default: the installed production module)",
    )
    parser.add_argument("--bitwise", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()

    if arguments.candidate:
        candidate = _load_module(arguments.candidate, "kda_v4_candidate")
    else:
        from yxtpu_pretrain.kernels import kda_fused_pallas_v4 as candidate
    reference = _load_module(arguments.reference, "kda_v4_reference")

    failed = False
    for seed in range(arguments.seeds):
        primals, cotangents = _inputs(
            arguments.batch, arguments.seq, arguments.heads, seed
        )
        candidate_results = _run(candidate, primals, cotangents)
        reference_results = _run(reference, primals, cotangents)
        for name, actual, expected in zip(
            _OUTPUT_NAMES, candidate_results, reference_results, strict=True
        ):
            if arguments.bitwise:
                equal = np.array_equal(actual, expected)
                verdict = "BITWISE-EQUAL" if equal else "MISMATCH"
                detail = ""
                if not equal:
                    failed = True
                    difference = np.abs(
                        actual.astype(np.float64) - expected.astype(np.float64)
                    )
                    detail = (
                        f" max_abs_diff={difference.max():.3e}"
                        f" mismatches={int((difference > 0).sum())}/{difference.size}"
                    )
                print(f"seed={seed} {name}: {verdict}{detail}")
            else:
                actual64 = actual.astype(np.float64)
                expected64 = expected.astype(np.float64)
                absolute = np.abs(actual64 - expected64)
                scale = np.abs(expected64)
                denominator = math.sqrt(float(np.sum(expected64 * expected64))) or 1.0
                relative_l2 = math.sqrt(float(np.sum(absolute * absolute))) / denominator
                max_relative = float(
                    (absolute / np.maximum(scale, 1e-6)).max()
                )
                finite = bool(np.isfinite(actual64).all())
                if not finite:
                    failed = True
                print(
                    f"seed={seed} {name}: rel_l2={relative_l2:.3e}"
                    f" max_rel={max_relative:.3e} finite={finite}"
                )
    print("RESULT: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
