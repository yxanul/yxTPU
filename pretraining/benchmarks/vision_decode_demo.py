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

"""Qualitative decode smoke for the joint vision+text checkpoint.

Eager full-forward greedy decode (recompiled once for a fixed [1, 512]
shape): each step reruns the whole prefix and reads the last real
position's logits on device. Slow per token but dependency-free - the
incremental KDA decode path predates the vision splice.

Prompts follow the training contract exactly: an image enters as 196
placeholder tokens followed by ``Q: ...\nA:`` (vision_data.render_texts),
images resized to 448 and scaled to [-1, 1] (process_image).

Single-chip carve-out invocation on a worker that holds the checkpoint:

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_VISIBLE_DEVICES=0 HF_HOME=/mnt/ram/hf .venv/bin/python \
  benchmarks/vision_decode_demo.py --out /mnt/ram/vt_transcripts.json
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
from yxtpu_pretrain.model import HybridLanguageModel, count_parameters
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO
from yxtpu_pretrain.runtime.data import load_fast_tokenizer
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context
from yxtpu_pretrain.runtime.vision_data import process_image

SEQ = 512
PAD_ID = 49119
EOS_ID = 49119
PLACEHOLDER_ID = 49150


class _NoIterator:
    def set_state(self, payload):
        raise AssertionError("stream state must not restore during decode")


def _dataset_prompts(count: int, image_size: int):
    """Image+question rows from the vision corpus head (train distribution)."""
    from datasets import load_dataset

    stream = load_dataset("HuggingFaceM4/FineVisionMax", split="train", streaming=True)
    prompts = []
    for row in stream:
        images = row.get("images") or []
        turns = row.get("texts") or []
        if len(images) != 1 or not turns:
            continue
        user = (turns[0].get("user") or "").strip()
        reference = (turns[0].get("assistant") or "").strip()
        if not user or not reference:
            continue
        prompts.append(
            {
                "name": f"finevision_{len(prompts)}",
                "image": process_image(images[0], image_size),
                "prompt": f"Q: {user}\nA:",
                "reference": reference,
            }
        )
        if len(prompts) >= count:
            break
    return prompts


def _synthetic_prompts(image_size: int):
    red = np.full((image_size, image_size, 3), -1.0, dtype=np.float32)
    red[..., 0] = 1.0
    blue = np.full((image_size, image_size, 3), -1.0, dtype=np.float32)
    blue[..., 2] = 1.0
    question = "Q: What color is the image?\nA:"
    return [
        {"name": "synthetic_red", "image": red, "prompt": question, "reference": "red"},
        {"name": "synthetic_blue", "image": blue, "prompt": question, "reference": "blue"},
    ]


def _text_prompts():
    return [
        {"name": "capital", "image": None, "prompt": "The capital of France is"},
        {"name": "gold", "image": None, "prompt": "Q: What is the chemical symbol for gold?\nA:"},
        {"name": "boil", "image": None, "prompt": "Water boils at a temperature of"},
        {"name": "arithmetic", "image": None, "prompt": "Q: What is 17 + 25?\nA:"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dest", default="/home/a1111/yxtpu_ckpts")
    parser.add_argument(
        "--run-name", default="kda_hybrid_1b_yx49k-muonclip-vision_1b_trial"
    )
    parser.add_argument("--dataset-samples", type=int, default=6)
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--out", default="/mnt/ram/vt_transcripts.json")
    arguments = parser.parse_args()

    config = load_config(
        model="kda_hybrid_1b_yx49k",
        optimizer="muonclip",
        data="climbmix_yx49k",
        hardware="v4-64",
        experiment="vision_1b_trial",
        overrides=[
            f"experiment.checkpoint.destination={arguments.checkpoint_dest}",
            "experiment.wandb.enabled=false",
            "experiment.harness_eval.enabled=false",
            "experiment.diagnostics.enabled=false",
            "data.eval_interval=0",
        ],
    )
    vision = config.model.vision
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules

    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))
    print(f"parameters: {count_parameters(model):,}", flush=True)

    loader = CheckpointIO(config, run_name=arguments.run_name)
    step = loader.restore(state, _NoIterator())
    loader.close()
    if step == 0:
        raise RuntimeError("no checkpoint restored")
    print(f"restored checkpoint step {step}", flush=True)

    tokenizer = load_fast_tokenizer(
        config.data.tokenizer, padded_vocab_size=config.model.vocab_size
    )

    @nnx.jit
    def next_logits(current_model, ids, images, position):
        logits = current_model(ids, images=images)
        return jax.lax.dynamic_index_in_dim(logits[0], position, axis=0, keepdims=False)

    blank = np.zeros((1, 1, vision.image_size, vision.image_size, 3), np.float32)
    prompts = (
        _dataset_prompts(arguments.dataset_samples, vision.image_size)
        + _synthetic_prompts(vision.image_size)
        + _text_prompts()
    )
    transcripts = []
    for prompt in prompts:
        text_ids = tokenizer.encode(prompt["prompt"], add_special_tokens=False)
        prefix = (
            [PLACEHOLDER_ID] * vision.visual_tokens_per_image
            if prompt["image"] is not None
            else []
        )
        ids = prefix + text_ids
        if len(ids) + arguments.max_new >= SEQ:
            ids = ids[: SEQ - arguments.max_new - 1]
        images = (
            prompt["image"][None, None].astype(np.float32)
            if prompt["image"] is not None
            else blank
        )
        images = jnp.asarray(images)
        generated = []
        with logical_mesh_context(mesh, rules):
            for _ in range(arguments.max_new):
                padded = np.full((1, SEQ), PAD_ID, np.int32)
                padded[0, : len(ids)] = ids
                logits = next_logits(
                    model, jnp.asarray(padded), images, jnp.int32(len(ids) - 1)
                )
                token = int(jnp.argmax(logits))
                if token in (EOS_ID, PLACEHOLDER_ID):
                    break
                ids.append(token)
                generated.append(token)
        completion = tokenizer.decode(generated)
        record = {
            "name": prompt["name"],
            "prompt": prompt["prompt"],
            "completion": completion,
            "reference": prompt.get("reference"),
        }
        transcripts.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    with open(arguments.out, "w", encoding="utf-8") as handle:
        json.dump(transcripts, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(transcripts)} transcripts to {arguments.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
