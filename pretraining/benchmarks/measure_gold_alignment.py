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

"""Measures GOLD's effective supervision rate on real text.

The tokenizer meta reports 98.8% of positions aligning 1:1 on its own
held-out mix; this measures the same quantity with the v1 alignment code
on the text distillation will actually see - our model's chat transcripts
(think traces and answers) and ClimbMix pretraining text - against a real
Qwen3.5 teacher tokenization. The 1:1 rate is the fraction of student
positions that receive teacher supervision under masking, i.e. the direct
cost of shipping without product-rule merges.

  .venv/bin/python benchmarks/measure_gold_alignment.py \
      --transcripts ../results/sft-ultradata-10000/transcripts_8k.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yxtpu_pretrain.distillation.alignment import align_by_byte_offsets


def measure(texts, student, teacher, label):
    positions = aligned = 0
    grouped = 0
    for text in texts:
        if not text or not text.strip():
            continue
        student_enc = student(text, add_special_tokens=False,
                              return_offsets_mapping=True)
        teacher_enc = teacher(text, add_special_tokens=False,
                              return_offsets_mapping=True)
        result = align_by_byte_offsets(
            text, student_enc["offset_mapping"], teacher_enc["offset_mapping"]
        )
        positions += len(result.one_to_one)
        aligned += int(result.one_to_one.sum())
        grouped += result.grouped_positions
    rate = aligned / max(positions, 1)
    print(f"{label:<26} positions {positions:>9,}  1:1 {rate:8.4f}  "
          f"masked {1 - rate:7.4f}")
    return {"label": label, "positions": positions, "one_to_one_rate": rate}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="tokenizers/yx49k")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--transcripts", default=None)
    parser.add_argument("--climbmix-chars", type=int, default=1_000_000)
    parser.add_argument("--output", default="/tmp/gold_alignment.json")
    arguments = parser.parse_args()

    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(arguments.student, use_fast=True)
    teacher = AutoTokenizer.from_pretrained(arguments.teacher, use_fast=True)
    reports = []

    if arguments.transcripts:
        data = json.loads(Path(arguments.transcripts).read_text())
        rows = data["transcripts"] if isinstance(data, dict) else data
        thinks = [row.get("think", "") for row in rows]
        answers = [row.get("answer", "") for row in rows]
        reports.append(measure(thinks, student, teacher, "chat think traces"))
        reports.append(measure(answers, student, teacher, "chat answers"))

    if arguments.climbmix_chars:
        from datasets import load_dataset

        stream = load_dataset(
            "karpathy/climbmix-400b-shuffle", split="train", streaming=True
        )
        texts, chars = [], 0
        for record in stream:
            texts.append(record["text"])
            chars += len(record["text"])
            if chars >= arguments.climbmix_chars:
                break
        reports.append(measure(texts, student, teacher, "climbmix pretrain"))

    Path(arguments.output).write_text(json.dumps(reports, indent=2))
    print(f"written {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
