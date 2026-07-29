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

"""Gemma 3 270M's reported panel, at Gemma's own shot counts.

Gemma reports each benchmark at a different number of shots, and shot
count moves these scores by several points, so a comparison run at a
single global n-shot is not a comparison. This runs one harness pass per
shot group and reports the union.

The pretrained panel goes against Gemma 3 PT 270M, the SFT panel against
Gemma 3 IT 270M. Loads either an orbax pretraining checkpoint (--step) or
an SFT pickle (--sft-checkpoint).

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/eval_gemma_comparison.py --panel pt
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.evaluation.lm_harness import JaxHarnessLM, run_harness_evaluation
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO, _persistent_state
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context

# Gemma 3 270M's published numbers and the shot count each was taken at.
GEMMA_PT = {
    "hellaswag": (10, 40.9), "boolq": (0, 61.4), "piqa": (0, 67.7),
    "arc_challenge": (25, 29.0), "arc_easy": (0, 57.7), "winogrande": (5, 52.0),
}
GEMMA_IT = {
    "hellaswag": (0, 37.7), "piqa": (0, 66.2), "arc_challenge": (0, 28.2),
    "winogrande": (0, 52.3),
}
# TriviaQA (5-shot), IF Eval and BIG-Bench Hard are generation tasks; the
# loglikelihood harness cannot serve them, so they are reported separately.


class _NoIterator:
    def set_state(self, payload):
        raise AssertionError("stream state must not restore during evaluation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("pt", "it"), default="pt")
    parser.add_argument("--init-destination", default="/home/a1111/yxtpu_ckpts")
    parser.add_argument("--init-run",
                        default="kda_hybrid_128k-muonclip-superbpe_50b")
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--batch-per-device", type=int, default=1)
    parser.add_argument("--output", default="/tmp/gemma_comparison.json")
    arguments = parser.parse_args()

    panel = GEMMA_PT if arguments.panel == "pt" else GEMMA_IT
    groups: dict[int, list[str]] = {}
    for task, (shots, _) in panel.items():
        groups.setdefault(shots, []).append(task)

    config = load_config(
        model="kda_hybrid_128k", optimizer="muonclip", data="climbmix_superbpe",
        hardware="v4-64", experiment="superbpe_50b",
        overrides=[
            "experiment.token_budget=null",
            "experiment.wandb.enabled=false",
            "experiment.diagnostics.enabled=false",
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
        step = 0
        print(f"loaded SFT pickle {arguments.sft_checkpoint}", flush=True)
    else:
        loader = CheckpointIO(config, run_name=arguments.init_run)
        step = loader.restore(state, _NoIterator())
        loader.close()
        if step == 0:
            raise RuntimeError("no checkpoint restored")
        print(f"restored pretraining checkpoint step {step}", flush=True)

    run_dir = Path("/tmp") / f"gemma_panel_{arguments.panel}"
    run_dir.mkdir(parents=True, exist_ok=True)
    measured: dict[str, dict] = {}
    for shots in sorted(groups):
        tasks = sorted(groups[shots])
        shot_config = config.model_copy(deep=True)
        shot_config.experiment.harness_eval.tasks = tuple(tasks)
        shot_config.experiment.harness_eval.num_fewshot = shots
        shot_config.experiment.harness_eval.limit = arguments.limit
        print(f"\n=== {shots}-shot: {', '.join(tasks)}", flush=True)
        began = time.perf_counter()
        with logical_mesh_context(mesh, rules):
            adapter = JaxHarnessLM(shot_config, model, mesh, rules)
            metrics, _ = run_harness_evaluation(
                adapter, shot_config, run_dir=run_dir, step=shots)
        elapsed = time.perf_counter() - began
        for task in tasks:
            # Tasks outside the training panel have no registered primary
            # metric; fall back to the conventional one for the task.
            primary = metrics.get(f"{task}/primary")
            if primary is None:
                primary = metrics.get(
                    f"{task}/acc_norm", metrics.get(f"{task}/acc"))
            measured[task] = {
                "shots": shots,
                "score": None if primary is None else round(primary * 100, 1),
                "acc": metrics.get(f"{task}/acc"),
                "acc_norm": metrics.get(f"{task}/acc_norm"),
            }
        print(f"[{elapsed:.0f}s] " + json.dumps(
            {t: measured[t]["score"] for t in tasks}), flush=True)

    reference = "Gemma 3 PT 270M" if arguments.panel == "pt" else "Gemma 3 IT 270M"
    print(f"\n{'benchmark':<16}{'shots':>6}{'ours':>8}{'gemma':>8}{'delta':>8}",
          flush=True)
    rows = []
    for task in sorted(panel):
        shots, gemma = panel[task]
        ours = measured.get(task, {}).get("score")
        delta = None if ours is None else round(ours - gemma, 1)
        rows.append({"task": task, "shots": shots, "ours": ours,
                     "gemma": gemma, "delta": delta})
        print(f"{task:<16}{shots:>6}{ours if ours is not None else '-':>8}"
              f"{gemma:>8}{delta if delta is not None else '-':>8}", flush=True)
    wins = sum(1 for row in rows if row["delta"] is not None and row["delta"] > 0)
    print(f"\nahead on {wins}/{len(rows)} against {reference}", flush=True)

    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump({
            "panel": arguments.panel, "reference": reference,
            "checkpoint": arguments.sft_checkpoint or f"step {step}",
            "rows": rows, "raw": measured,
        }, handle, indent=2)
    print(f"written {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
