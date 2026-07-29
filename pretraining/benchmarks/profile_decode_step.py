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

"""Where a decoded token's time goes, and what caching would remove.

The chat and sweep scripts have no incremental state: every generated
token re-runs a forward over the whole window for the whole batch. This
measures the per-token cost against window and batch, splits device time
from the host round trip, and prints the arithmetic each token actually
needs, so the cost of the missing KV/KDA cache is a number rather than an
assertion.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/profile_decode_step.py
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel, count_parameters
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", default="/tmp/decode_profile.json")
    arguments = parser.parse_args()

    windows = (512, 1024, 2048, 4096)
    batches = (4, 16)
    results = []
    devices = jax.device_count()

    for window in windows:
        for batch in batches:
            config = load_config(
                model="kda_hybrid_128k", optimizer="muonclip",
                data="climbmix_superbpe", hardware="v4-64",
                experiment="superbpe_50b",
                overrides=[
                    f"data.sequence_length={window}",
                    "experiment.wandb.enabled=false",
                    "experiment.token_budget=null",
                    "experiment.checkpoint.enabled=false",
                    "experiment.acknowledge_no_checkpoint=true",
                    "experiment.harness_eval.enabled=false",
                    "experiment.diagnostics.enabled=false",
                ],
            )
            mesh = create_mesh(config.hardware, allow_device_mismatch=True)
            rules = make_leaf_config(config).logical_axis_rules
            with logical_mesh_context(mesh, rules):
                model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
                transform, _ = build_optimizer(model, config.optimizer)
                TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))
            parameters = count_parameters(model)

            @nnx.jit
            def next_logits(current_model, tokens, segments, positions, cursors):
                hidden = current_model.hidden_states(
                    tokens, decoder_segment_ids=segments,
                    decoder_positions=positions)
                picked = jnp.take_along_axis(
                    hidden, cursors[:, None, None], axis=1)[:, 0, :]
                kernel = current_model.output_projection_kernel(picked.dtype)
                return (picked @ kernel).astype(jnp.float32)

            tokens = jnp.asarray(
                np.random.default_rng(0).integers(
                    1, 128000, (batch, window), dtype=np.int32))
            segments = jnp.ones((batch, window), jnp.int32)
            positions = jnp.asarray(
                np.tile(np.arange(window, dtype=np.int32), (batch, 1)))
            cursors = jnp.full((batch,), window - 1, jnp.int32)

            with logical_mesh_context(mesh, rules):
                out = next_logits(model, tokens, segments, positions, cursors)
                jax.block_until_ready(out)
                # device-only: keep the result on device
                began = time.perf_counter()
                for _ in range(arguments.repeats):
                    out = next_logits(model, tokens, segments, positions, cursors)
                jax.block_until_ready(out)
                device_ms = (time.perf_counter() - began) / arguments.repeats * 1000
                # with the host round trip the decode loop actually performs
                began = time.perf_counter()
                for _ in range(arguments.repeats):
                    out = next_logits(model, tokens, segments, positions, cursors)
                    host = np.asarray(out).astype(np.float64)
                total_ms = (time.perf_counter() - began) / arguments.repeats * 1000

            logits_mb = batch * config.model.vocab_size * 4 / 1e6
            forward_tflop = 2 * parameters * batch * window / 1e12
            entry = {
                "window": window, "batch": batch,
                "device_ms": round(device_ms, 1),
                "host_roundtrip_ms": round(total_ms - device_ms, 1),
                "total_ms_per_token": round(total_ms, 1),
                "logits_transfer_mb": round(logits_mb, 1),
                "forward_tflop_per_token": round(forward_tflop, 2),
                "achieved_tflops": round(forward_tflop / (device_ms / 1000), 1),
            }
            results.append(entry)
            print(json.dumps(entry), flush=True)
            del model

    parameters_only_tflop = 2 * 337_228_384 * 16 / 1e12
    print("\nper-token arithmetic actually required for batch 16 with cached "
          f"state: {parameters_only_tflop * 1e6:.1f} MFLOP "
          f"({parameters_only_tflop:.6f} TFLOP)", flush=True)
    print(f"slice peak (4 chips x 275 TFLOP/s bf16): {4 * 275} TFLOP/s", flush=True)
    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"written {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
