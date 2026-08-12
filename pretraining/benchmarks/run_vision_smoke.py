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

"""Joint vision+text training smoke: fresh tower on the pretrained base.

Grafts a from-scratch vision tower onto the pretrained backbone (restored
into a text-only twin, then copied over), streams FineVisionMax, and runs
the ordinary train step. What a healthy smoke shows: total loss falling,
finite gradients with no spikes, and a step time within budget.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/run_vision_smoke.py --steps 60
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
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO
from yxtpu_pretrain.runtime.data import load_fast_tokenizer
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context
from yxtpu_pretrain.runtime.vision_data import (
    MixedVisionTextIterator,
    PooledMixedIterator,
    VisionBatchSpec,
)
from yxtpu_pretrain.train import _device_batch, _make_train_step, _vision_metrics


class _NoIterator:
    def set_state(self, payload):
        raise AssertionError("stream state must not restore during the smoke")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="kda_hybrid_yx49k_l20")
    parser.add_argument("--data", default="climbmix_yx49k")
    parser.add_argument("--init-destination", default="/home/a1111/yxtpu_ckpts")
    parser.add_argument(
        "--init-run", default="kda_hybrid_yx49k_l20-muonclip-superbpe_50b"
    )
    parser.add_argument("--no-init", action="store_true",
                        help="skip the base restore (fresh backbone smoke)")
    parser.add_argument("--dataset", default="HuggingFaceM4/FineVisionMax")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-per-device", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--min-visual-dependency", type=int, default=0)
    parser.add_argument("--p-text", type=float, default=0.3,
                        help="probability a packed row is a plain-text "
                             "document from the pretraining corpus")
    parser.add_argument("--text-row-tokens", type=int, default=1024)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--producer-threads", type=int, default=4)
    parser.add_argument("--no-output-gate", action="store_true",
                        help="disable the head-specific sigmoid output gate "
                             "(G1) on the attention layers")
    parser.add_argument("--allow-device-mismatch", action="store_true")
    parser.add_argument("--profile-dir", default=None,
                        help="capture a jax profiler trace on the primary "
                             "process over a few steady-state steps")
    parser.add_argument("--profile-start", type=int, default=8)
    parser.add_argument("--profile-count", type=int, default=3)
    parser.add_argument("--metrics-out", default="/tmp/vision_smoke_steps.jsonl")
    parser.add_argument("--set", action="append", dest="overrides", default=[])
    arguments = parser.parse_args()

    overrides = [
        f"experiment.steps={arguments.steps}",
        "experiment.token_budget=null",
        "experiment.benchmark=false",
        "experiment.acknowledge_no_checkpoint=true",
        "experiment.checkpoint.enabled=false",
        "experiment.harness_eval.enabled=false",
        "experiment.diagnostics.enabled=false",
        "experiment.wandb.enabled=false",
        f"data.sequence_length={arguments.sequence_length}",
        f"data.per_device_batch_size={arguments.batch_per_device}",
        f"optimizer.learning_rate={arguments.learning_rate}",
        f"optimizer.schedule_steps={max(arguments.steps, 4)}",
        "optimizer.warmup_steps=2",
        "model.vision.enabled=true",
        f"model.attention.output_gate={str(not arguments.no_output_gate).lower()}",
    ] + list(arguments.overrides or [])
    config = load_config(
        model=arguments.model, optimizer="muonclip", data=arguments.data,
        hardware="v4-64", experiment="superbpe_50b", overrides=overrides,
    )
    vision = config.model.vision
    mesh = create_mesh(config.hardware,
                       allow_device_mismatch=arguments.allow_device_mismatch)
    rules = make_leaf_config(config).logical_axis_rules

    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(config.experiment.seed))
    print(f"vision model parameters: {count_parameters(model):,} "
          f"({vision.visual_tokens_per_image} visual tokens/image)", flush=True)

    if not arguments.no_init:
        # The restore twin must match the checkpoint's parameter tree
        # exactly: no vision tower, no attention output gate. Both graft
        # onto the vision model afterwards and train from fresh init.
        text_config = config.model_copy(deep=True)
        text_config.model.vision.enabled = False
        text_config.model.attention.output_gate = False
        text_config.experiment.checkpoint.enabled = True
        text_config.experiment.checkpoint.destination = arguments.init_destination
        text_config.experiment.acknowledge_no_checkpoint = False
        with logical_mesh_context(mesh, rules):
            text_model = HybridLanguageModel(
                text_config, mesh, rngs=nnx.Rngs(config.experiment.seed)
            )
            text_transform, _ = build_optimizer(text_model, text_config.optimizer)
            text_state = TrainStateNNX(
                text_model, nnx.Optimizer(text_model, text_transform, wrt=nnx.Param)
            )
        loader = CheckpointIO(text_config, run_name=arguments.init_run)
        step = loader.restore(text_state, _NoIterator())
        loader.close()
        if step == 0:
            raise RuntimeError("no base checkpoint found to graft onto")
        nnx.update(model, nnx.state(text_model, nnx.Param))
        del text_model, text_state
        print(f"grafted base checkpoint step {step} onto the vision model", flush=True)

    with logical_mesh_context(mesh, rules):
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))

    tokenizer = load_fast_tokenizer(
        config.data.tokenizer, padded_vocab_size=config.model.vocab_size
    )
    spec = VisionBatchSpec(
        sequence_length=config.data.sequence_length,
        visual_tokens=vision.visual_tokens_per_image,
        image_size=vision.image_size,
        placeholder_id=vision.placeholder_token_id,
        pad_id=49119,
        eos_id=49119,
        max_images=arguments.max_images,
    )
    process_batch = config.data.per_device_batch_size * jax.local_device_count()
    threads = max(1, arguments.producer_threads)

    text_sources = (
        [
            {
                "name": "text",
                "dataset": config.data.dataset_name,
                "field": config.data.text_field,
                "weight": 1.0,
                "format": "plain",
                "row_tokens": arguments.text_row_tokens,
            }
        ]
        if arguments.p_text > 0
        else []
    )

    def make_source(thread_index: int) -> MixedVisionTextIterator:
        return MixedVisionTextIterator(
            tokenizer=tokenizer,
            spec=spec,
            batch_size=process_batch,
            vision_dataset=arguments.dataset,
            text_sources=text_sources,
            p_text=arguments.p_text,
            text_row_tokens=arguments.text_row_tokens,
            min_visual_dependency=arguments.min_visual_dependency,
            shuffle_seed=config.data.shuffle_seed,
            shard_index=jax.process_index() * threads + thread_index,
            shard_count=jax.process_count() * threads,
        )

    pool = PooledMixedIterator(
        make_source, threads=threads, batch_size=process_batch
    )
    iterator = pool
    if arguments.prefetch > 1:
        from yxtpu_pretrain.runtime.data import PrefetchIterator

        iterator = PrefetchIterator(pool, depth=arguments.prefetch)
        iterator.rows_stats = pool.rows_stats

    train_step = _make_train_step(config)
    records = []
    global_batch = process_batch * jax.process_count()
    profiling = arguments.profile_dir and jax.process_index() == 0
    for step_index in range(arguments.steps):
        if profiling and step_index == arguments.profile_start:
            jax.profiler.start_trace(arguments.profile_dir)
        began = time.perf_counter()
        host_batch = next(iterator)
        data_wait_ms = (time.perf_counter() - began) * 1000.0
        # Formed OUTSIDE logical_mesh_context, exactly like the proven
        # multi-host SFT loop: an ambient mesh context must not influence
        # the host-local -> global conversion.
        batch = _device_batch(host_batch, mesh)
        if step_index == 0 and batch["input_ids"].shape[0] != global_batch:
            raise RuntimeError(
                f"global batch is {batch['input_ids'].shape[0]}, expected "
                f"{global_batch}: host-local to global conversion failed"
            )
        with logical_mesh_context(mesh, rules):
            metrics = train_step(state, batch)
            metrics = {
                key: float(value)
                for key, value in metrics.items()
                if jnp.ndim(value) == 0
            }
            consumed, skipped, vision_rows, text_rows = iterator.rows_stats()
            record = {
                "step": step_index,
                "step_ms": round((time.perf_counter() - began) * 1000.0, 1),
                "data_wait_ms": round(data_wait_ms, 1),
                "rows_consumed": consumed,
                "rows_skipped": skipped,
                "vision_rows": vision_rows,
                "text_rows": text_rows,
                **{key: round(value, 6) for key, value in metrics.items()},
            }
            vision_derived = _vision_metrics(metrics)
            if vision_derived is not None:
                record.update(
                    {key: round(value, 6) for key, value in vision_derived.items()}
                )
            record.update(
                {
                    f"data_{key}": round(value, 6)
                    for key, value in pool.stats.items()
                    if key
                    in (
                        "pad_fraction",
                        "visual_token_fraction",
                        "vision_loss_token_share",
                        "images_per_sequence",
                        "image_slot_utilization",
                        "rows_per_sequence",
                        "row_skip_rate",
                    )
                }
            )
            records.append(record)
            if jax.process_index() == 0:
                print(json.dumps(record), flush=True)
        if (
            profiling
            and step_index == arguments.profile_start + arguments.profile_count
        ):
            jax.profiler.stop_trace()
            print(f"profile written to {arguments.profile_dir}", flush=True)

    if jax.process_index() == 0:
        with open(arguments.metrics_out, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        losses = [record["loss"] for record in records if "loss" in record]
        if losses:
            print(f"loss first->last: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
