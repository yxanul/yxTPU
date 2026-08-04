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

"""Generation-scored benchmarks (IFEval, TriviaQA, BBH) via the cached decoder.

These are the rows the loglikelihood harness could never fill. Prompts are
chat-templated for an instruction-tuned checkpoint and left raw for a base
one, and the think block is stripped before scoring so tasks judge the
answer rather than the reasoning trace.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/eval_generative_tasks.py --sft-checkpoint <dir>/state.pkl \
    --tasks ifeval
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.decode import SamplingParams
from yxtpu_pretrain.evaluation.lm_harness import (
    GenerationSettings,
    JaxHarnessLM,
    run_harness_evaluation,
)
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO, _persistent_state
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context

# Gemma 3 270M's published generative rows, for the same-row comparison.
GEMMA = {"ifeval": ("IT", 51.2), "triviaqa": ("PT", 15.4), "bbh": ("IT", 26.7)}
FEWSHOT = {"ifeval": 0, "triviaqa": 5, "bbh": 3, "gsm8k": 5}


class _NoIterator:
    def set_state(self, payload):
        raise AssertionError("stream state must not restore during evaluation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="ifeval")
    parser.add_argument("--model", default="kda_hybrid_128k")
    parser.add_argument("--data", default="climbmix_superbpe")
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--init-destination", default="/home/a1111/yxtpu_ckpts")
    parser.add_argument("--init-run",
                        default="kda_hybrid_128k-muonclip-superbpe_50b")
    parser.add_argument("--max-gen-toks", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--batch-per-device", type=int, default=4)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--output", default="/tmp/generative_eval.json")
    arguments = parser.parse_args()

    tasks = [task.strip() for task in arguments.tasks.split(",") if task.strip()]
    config = load_config(
        model=arguments.model, optimizer="muonclip", data=arguments.data,
        hardware="v4-64", experiment="superbpe_50b",
        overrides=[
            "experiment.token_budget=null",
            "experiment.wandb.enabled=false",
            "experiment.diagnostics.enabled=false",
            f"data.sequence_length={arguments.sequence_length}",
            f"experiment.checkpoint.destination={arguments.init_destination}",
            f"experiment.harness_eval.batch_size_per_device={arguments.batch_per_device}",
        ],
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))

    if arguments.sft_checkpoint:
        target = _persistent_state(state)
        with open(arguments.sft_checkpoint, "rb") as handle:
            nnx.replace_by_pure_dict(target, pickle.load(handle))
        nnx.update(state, target)
        step, label = 0, arguments.sft_checkpoint
    else:
        loader = CheckpointIO(config, run_name=arguments.init_run)
        step = loader.restore(state, _NoIterator())
        loader.close()
        if step == 0:
            raise RuntimeError("no checkpoint restored")
        label = f"step {step}"
    print(f"loaded {label}", flush=True)

    generation = GenerationSettings(
        sampling=SamplingParams(
            temperature=arguments.temperature, top_k=64,
            top_p=arguments.top_p,
            repetition_penalty=arguments.repetition_penalty),
        max_gen_toks=arguments.max_gen_toks,
        apply_chat_template=not arguments.no_chat_template,
        strip_reasoning=not arguments.no_chat_template,
    )
    run_dir = Path("/tmp/generative_eval")
    run_dir.mkdir(parents=True, exist_ok=True)
    collected: dict[str, dict] = {}
    for task in tasks:
        task_config = config.model_copy(deep=True)
        task_config.experiment.harness_eval.tasks = (task,)
        task_config.experiment.harness_eval.num_fewshot = FEWSHOT.get(task, 0)
        task_config.experiment.harness_eval.limit = arguments.limit
        print(f"\n=== {task} ({FEWSHOT.get(task, 0)}-shot)", flush=True)
        began = time.perf_counter()
        with logical_mesh_context(mesh, rules):
            adapter = JaxHarnessLM(
                task_config, model, mesh, rules, generation=generation)
            metrics, path = run_harness_evaluation(
                adapter, task_config, run_dir=run_dir, step=0)
        elapsed = time.perf_counter() - began
        scores = {
            name: round(value * 100, 1)
            for name, value in sorted(metrics.items())
            if name.startswith(task) and isinstance(value, float)
        }
        collected[task] = {"seconds": round(elapsed), "scores": scores,
                           "artifact": str(path)}
        print(f"[{elapsed:.0f}s] {json.dumps(scores)}", flush=True)
        if task in GEMMA:
            variant, published = GEMMA[task]
            print(f"    Gemma 3 {variant} 270M reports {published}", flush=True)

    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": label, "generation": {
            "temperature": arguments.temperature, "top_p": arguments.top_p,
            "repetition_penalty": arguments.repetition_penalty,
            "max_gen_toks": arguments.max_gen_toks,
            "chat_template": generation.apply_chat_template,
        }, "tasks": collected}, handle, indent=2)
    print(f"\nwritten {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
