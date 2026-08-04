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

"""GOLD at lambda = 0: the teacher scores fixed data, not student rollouts.

The rung below on-policy, and the one that isolates the parts worth
isolating. Sequences come from the Mephisto sets rather than from the
student, so nothing here depends on generation - which means a bad result
implicates the projection, the alignment or the objective, and not the
rollout loop that does not exist yet.

The data path is the SFT stage's, unchanged: ``MephistoIterator`` renders
with the yx49k chat template (Qwen3.5's, verbatim) and packs whole
examples. The teacher then reads ``student_to_teacher[input_ids]`` - the
student's own segmentation, mapped - so supervision is 1:1 at every
position with no alignment pass.

``--dry-run`` skips the teacher and reports the student's CE alone, which
is the number GOLD has to beat.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/run_gold_offpolicy.py \
    --teacher-checkpoint /home/a1111/yxtpu_ckpts/qwen35-4b-teacher \
    --student-checkpoint ckpt/sft-mephisto-v2/1200/state.pkl --steps 20
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

SYSTEM = ("You are a helpful assistant. Answer the user's question "
          "accurately, clearly, and concisely.")


def load_student(model_config, data_config, sequence, checkpoint):
    from maxtext.common.train_state_nnx import TrainStateNNX

    from yxtpu_pretrain.config import load_config
    from yxtpu_pretrain.model import HybridLanguageModel
    from yxtpu_pretrain.optimizers import build_optimizer
    from yxtpu_pretrain.runtime.checkpoints import _persistent_state
    from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
    from yxtpu_pretrain.runtime.mesh import create_mesh
    from yxtpu_pretrain.runtime.sharding import logical_mesh_context

    config = load_config(
        model=model_config, optimizer="muonclip", data=data_config,
        hardware="v4-64", experiment="superbpe_50b",
        overrides=["experiment.token_budget=null",
                   "experiment.wandb.enabled=false",
                   "experiment.diagnostics.enabled=false",
                   f"data.sequence_length={sequence}"],
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform,
                                                   wrt=nnx.Param))
    if checkpoint:
        target = _persistent_state(state)
        with open(checkpoint, "rb") as handle:
            nnx.replace_by_pure_dict(target, pickle.load(handle))
        nnx.update(state, target)
        print(f"student restored from {checkpoint}", flush=True)
    return config, state, mesh, rules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--model", default="kda_hybrid_yx49k_l20")
    parser.add_argument("--data", default="climbmix_yx49k")
    parser.add_argument("--tokenizer", default="tokenizers/yx49k")
    parser.add_argument("--datasets", default="Yxanul/Mephisto-IF_172k")
    parser.add_argument("--mapping",
                        default="tokenizers/yx49k/student_to_teacher.npy")
    parser.add_argument("--covered",
                        default="tokenizers/yx49k/teacher_covered.npy")
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--tensor-parallelism", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="student CE only; the baseline GOLD must beat")
    parser.add_argument("--output", default="/tmp/gold_offpolicy.json")
    arguments = parser.parse_args()

    from transformers import AutoTokenizer

    from yxtpu_pretrain.distillation.objective import gold_objective
    from yxtpu_pretrain.runtime.sharding import logical_mesh_context
    from yxtpu_pretrain.sft.mephisto import MephistoIterator

    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer, use_fast=True)
    config, state, mesh, rules = load_student(
        arguments.model, arguments.data, arguments.sequence,
        arguments.student_checkpoint)

    teacher = None
    if not arguments.dry_run:
        from yxtpu_pretrain.distillation.teacher import Qwen35Teacher

        covered = np.load(arguments.covered)
        teacher = Qwen35Teacher(
            arguments.teacher_checkpoint,
            student_to_teacher=np.load(arguments.mapping),
            sequence=arguments.sequence, batch=arguments.batch,
            tensor_parallelism=arguments.tensor_parallelism,
            valid_vocab=int(covered.shape[0]),
        )

    stream = MephistoIterator(
        tokenizer,
        datasets=[s for s in arguments.datasets.split(",") if s],
        sequence_length=arguments.sequence, process_batch=arguments.batch,
        process_index=0, process_count=1, epochs=1, system=SYSTEM,
    )

    history = []
    began = time.perf_counter()
    for step, batch in enumerate(stream):
        if step >= arguments.steps:
            break
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        loss_mask = jnp.asarray(batch["loss_mask"])
        positions = jnp.asarray(batch["positions"])[:, :input_ids.shape[1]]
        segments = jnp.asarray(batch["segment_ids"])[:, :input_ids.shape[1]]

        targets = {}
        if teacher is not None:
            matched, residual = teacher.score(input_ids, positions, segments)
            targets = {"teacher_matched_logprobs": matched,
                       "teacher_residual_mass": residual}

        with logical_mesh_context(mesh, rules):
            student_logits = state.model(
                input_ids, decoder_positions=positions,
                decoder_segment_ids=segments)
        if isinstance(student_logits, tuple):
            student_logits = student_logits[0]

        _, metrics = gold_objective(
            student_logits, labels, loss_mask, beta=arguments.beta,
            distill_weight=arguments.distill_weight,
            ce_weight=arguments.ce_weight, **targets)
        row = {k: float(v) for k, v in metrics.items()}
        row["step"] = step
        history.append(row)
        print(f"step {step:3d}  loss {row['loss']:8.4f}  ce {row['ce']:7.4f}"
              + (f"  distill {row['distill']:7.4f}"
                 f"  residual {row['teacher_residual_mass']:.4f}"
                 f"  teacher@1 {row['teacher_top1_is_label']:.2%}"
                 if "distill" in row else "")
              + f"  student@1 {row['student_top1_is_label']:.2%}"
                f"  tok {int(row['tokens'])}", flush=True)

    elapsed = time.perf_counter() - began
    summary = {
        "steps": len(history), "seconds": round(elapsed, 1),
        "mean_ce": float(np.mean([r["ce"] for r in history])),
        "student_ppl": float(np.exp(np.mean([r["ce"] for r in history]))),
    }
    if history and "distill" in history[0]:
        summary["mean_distill"] = float(np.mean([r["distill"] for r in history]))
        summary["mean_residual"] = float(
            np.mean([r["teacher_residual_mass"] for r in history]))
        summary["teacher_top1_is_label"] = float(
            np.mean([r["teacher_top1_is_label"] for r in history]))
    print(f"\n{json.dumps(summary, indent=2)}", flush=True)
    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "history": history,
                   "settings": vars(arguments)}, handle, indent=2)
    print(f"written {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
