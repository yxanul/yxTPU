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

"""Length-extrapolation pre-check on a pretraining checkpoint.

NoPE-GQA has only ever attended within the trained 2048-token window; KDA
is recurrent and length-free. Before SFT at a longer pack length this
measures holdout loss on validation-reserved documents packed at --seq,
bucketed by position, so degradation beyond the trained window is visible
directly (healthy: deeper positions match or beat early ones; broken:
loss climbs past position 2048).

Runs on one worker against its local checkpoint replica:
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/eval_length_extrapolation.py --seq 8192 --batches 32
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO
from yxtpu_pretrain.runtime.data import create_data_iterator
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context
from yxtpu_pretrain.train import _device_batch


class _NoIterator:
    def set_state(self, payload):
        raise AssertionError("stream state must not restore in this probe")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--init-destination", default="/home/a1111/yxtpu_ckpts")
    parser.add_argument("--init-run", default="kda_hybrid_128k-muonclip-superbpe_50b")
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    config = load_config(
        model="kda_hybrid_128k",
        optimizer="muonclip",
        data="climbmix_superbpe",
        hardware="v4-64",
        experiment="superbpe_50b",
        overrides=[
            "experiment.harness_eval.enabled=false",
            "experiment.diagnostics.enabled=false",
            "experiment.token_budget=null",
            f"experiment.checkpoint.destination={arguments.init_destination}",
            f"data.sequence_length={arguments.seq}",
            f"data.per_device_batch_size={arguments.per_device_batch}",
        ],
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))
    loader = CheckpointIO(config, run_name=arguments.init_run)
    step = loader.restore(state, _NoIterator())
    loader.close()
    if step == 0:
        raise RuntimeError("no checkpoint restored")
    print(f"restored checkpoint step {step}", flush=True)

    process_batch = arguments.per_device_batch * jax.local_device_count()
    iterator = create_data_iterator(
        config.data,
        global_batch_size=process_batch,
        vocab_size=config.model.vocab_size,
        validation=True,
        process_index=0,
        process_count=1,
    )

    @nnx.jit
    def positional_nll(current_model, batch):
        hidden = current_model.hidden_states(
            batch["input_ids"],
            decoder_segment_ids=batch["segment_ids"],
            decoder_positions=batch["positions"],
            record_max_logits=False,
        )
        kernel = current_model.output_projection_kernel(hidden.dtype)
        logits = jnp.einsum("bth,hv->btv", hidden, kernel).astype(jnp.float32)
        log_normalizer = jax.nn.logsumexp(logits, axis=-1)
        picked = jnp.take_along_axis(
            logits, batch["labels"][..., None], axis=-1
        )[..., 0]
        nll = (log_normalizer - picked) * batch["loss_mask"]
        return nll.sum(axis=0), batch["loss_mask"].sum(axis=0)

    position_sum = np.zeros(arguments.seq, np.float64)
    position_count = np.zeros(arguments.seq, np.float64)
    with logical_mesh_context(mesh, rules):
        for index in range(arguments.batches):
            batch = _device_batch(next(iterator), mesh)
            batch_sum, batch_count = positional_nll(model, batch)
            position_sum += np.asarray(batch_sum, np.float64)
            position_count += np.asarray(batch_count, np.float64)
            if (index + 1) % 8 == 0:
                done = position_sum.sum() / position_count.sum()
                print(f"batch {index + 1}/{arguments.batches} loss {done:.4f}",
                      flush=True)

    overall = position_sum.sum() / position_count.sum()
    quarter = arguments.seq // 4
    buckets = {
        f"{start}-{start + quarter - 1}": float(
            position_sum[start : start + quarter].sum()
            / position_count[start : start + quarter].sum()
        )
        for start in range(0, arguments.seq, quarter)
    }
    profile = {
        str(start): float(
            position_sum[start : start + 1024].sum()
            / position_count[start : start + 1024].sum()
        )
        for start in range(0, arguments.seq, 1024)
    }
    result = {
        "checkpoint_step": step,
        "sequence_length": arguments.seq,
        "tokens": float(position_count.sum()),
        "overall_loss": float(overall),
        "quarter_buckets": buckets,
        "per_1024_profile": profile,
    }
    print(json.dumps(result, indent=2), flush=True)
    output = arguments.output or f"/tmp/length_extrapolation_{arguments.seq}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"written {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
