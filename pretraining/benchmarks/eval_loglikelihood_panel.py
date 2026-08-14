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

"""Arbitrary loglikelihood task panel against a pretraining checkpoint.

Generalizes the Gemma-panel runner: any task at any shot count, grouped
into one harness pass per shot value, against any model/data config pair.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/eval_loglikelihood_panel.py \
    --model kda_hybrid_yx49k_l20 --data climbmix_yx49k \
    --spec winogrande:0,winogrande:5,mmlu_continuation:0,mmlu:5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.evaluation.lm_harness import JaxHarnessLM, run_harness_evaluation
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context


class _NoIterator:
    def set_state(self, payload):
        raise AssertionError("stream state must not restore during evaluation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="kda_hybrid_yx49k_l20")
    parser.add_argument("--data", default="climbmix_yx49k")
    parser.add_argument("--spec", required=True,
                        help="comma list of task:shots, e.g. winogrande:5,mmlu:5")
    parser.add_argument("--init-destination", default="/home/a1111/yxtpu_ckpts")
    parser.add_argument("--init-run", default=None,
                        help="defaults to <model>-muonclip-superbpe_50b")
    parser.add_argument("--sft-checkpoint", default=None,
                        help="load an SFT-stage state.pkl instead of orbax")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--batch-per-device", type=int, default=1)
    parser.add_argument("--output", default="/tmp/loglik_panel.json")
    parser.add_argument("--experiment", default="superbpe_50b")
    parser.add_argument("--set", action="append", dest="overrides", default=[])
    arguments = parser.parse_args()

    groups: dict[int, list[str]] = {}
    for entry in arguments.spec.split(","):
        task, _, shots = entry.strip().partition(":")
        groups.setdefault(int(shots or 0), []).append(task)

    run_name = arguments.init_run or f"{arguments.model}-muonclip-superbpe_50b"
    config = load_config(
        model=arguments.model, optimizer="muonclip", data=arguments.data,
        hardware="v4-64", experiment=arguments.experiment,
        overrides=[
            "experiment.token_budget=null",
            "experiment.wandb.enabled=false",
            "experiment.diagnostics.enabled=false",
            f"experiment.checkpoint.destination={arguments.init_destination}",
            f"experiment.harness_eval.batch_size_per_device={arguments.batch_per_device}",
        ] + list(arguments.overrides or []),
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))
    if arguments.sft_checkpoint:
        import pickle

        from yxtpu_pretrain.runtime.checkpoints import _persistent_state

        target = _persistent_state(state)
        with open(arguments.sft_checkpoint, "rb") as handle:
            nnx.replace_by_pure_dict(target, pickle.load(handle))
        nnx.update(state, target)
        step = arguments.sft_checkpoint
        print(f"restored {arguments.sft_checkpoint}", flush=True)
    else:
        loader = CheckpointIO(config, run_name=run_name)
        step = loader.restore(state, _NoIterator())
        loader.close()
        if step == 0:
            raise RuntimeError("no checkpoint restored")
        print(f"restored {run_name} step {step}", flush=True)

    run_dir = Path("/tmp/loglik_panel")
    run_dir.mkdir(parents=True, exist_ok=True)
    collected: dict[str, dict] = {}
    for shots in sorted(groups):
        tasks = sorted(set(groups[shots]))
        shot_config = config.model_copy(deep=True)
        shot_config.experiment.harness_eval.tasks = tuple(tasks)
        shot_config.experiment.harness_eval.num_fewshot = shots
        shot_config.experiment.harness_eval.limit = arguments.limit
        print(f"\n=== {shots}-shot: {', '.join(tasks)}", flush=True)
        began = time.perf_counter()
        with logical_mesh_context(mesh, rules):
            adapter = JaxHarnessLM(shot_config, model, mesh, rules)
            metrics, path = run_harness_evaluation(
                adapter, shot_config, run_dir=run_dir, step=shots)
        elapsed = time.perf_counter() - began
        for task in tasks:
            primary = metrics.get(f"{task}/primary")
            if primary is None:
                primary = metrics.get(f"{task}/acc_norm", metrics.get(f"{task}/acc"))
            collected[f"{task}@{shots}"] = {
                "score": None if primary is None else round(primary * 100, 2),
                "acc": metrics.get(f"{task}/acc"),
                "acc_norm": metrics.get(f"{task}/acc_norm"),
                "artifact": str(path),
            }
        print(f"[{elapsed:.0f}s] " + json.dumps(
            {t: collected[f'{t}@{shots}']['score'] for t in tasks}), flush=True)

    print()
    for name, row in sorted(collected.items()):
        print(f"  {name:<26} {row['score']}")
    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump({"checkpoint_step": step, "model": arguments.model,
                   "results": collected}, handle, indent=2)
    print(f"written {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
