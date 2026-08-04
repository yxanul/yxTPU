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

"""Compares the two ways of getting teacher supervision onto our positions.

For each corpus this reports both strategies on the same text:

* direct map - the teacher scores the student's own segmentation. Valid
  only if the mapped ids decode back to the source byte-for-byte, which is
  the number to watch; when it holds, supervision is 1:1 everywhere. The
  canonical rate is the quality caveat rather than a failure: it is how
  often the student's boundaries are ones the teacher's own BPE would have
  placed, and the rest are positions where the teacher scores a
  segmentation it never saw in training.
* byte walk - the teacher re-tokenizes natively and non-1:1 groups are
  masked. Its 1:1 rate is the coverage the direct map buys back.

  .venv/bin/python benchmarks/measure_gold_alignment.py \
      --mephisto Yxanul/Mephisto-IF_172k --climbmix-chars 400000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yxtpu_pretrain.distillation import (
    align_by_byte_offsets,
    validate_student_to_teacher,
    verify_direct_map,
)

SYSTEM = ("You are a helpful assistant. Answer the user's question "
          "accurately, clearly, and concisely.")


def measure_walk(texts, student, teacher):
    """1:1 rate under native teacher tokenization plus the byte walker."""
    positions = aligned = grouped = 0
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
    return {"positions": positions,
            "one_to_one_rate": aligned / max(positions, 1),
            "grouped_positions": grouped}


def measure(label, texts, student, teacher, mapping):
    direct = verify_direct_map(student, teacher, mapping, texts)
    walk = measure_walk(texts, student, teacher)
    # Supervision is total exactly when every text reads back intact.
    direct_coverage = 1.0 if direct.roundtrip_rate == 1.0 else None
    print(f"\n{label}")
    print(f"  texts {direct.texts:,}   student tokens "
          f"{direct.student_tokens:,}   fertility {direct.fertility:.4f}")
    print(f"  direct map : round-trip {direct.roundtrip_rate:9.4%}   "
          f"canonical {direct.canonical_rate:9.4%}   coverage "
          + (f"{direct_coverage:9.4%}" if direct_coverage else "  INVALID"))
    print(f"  byte walk  : coverage   {walk['one_to_one_rate']:9.4%}   "
          f"masked {1 - walk['one_to_one_rate']:9.4%}")
    for want, got in direct.mismatches:
        print(f"    ROUND-TRIP MISMATCH\n      want {want!r}\n      got  {got!r}")
    return {
        "label": label,
        "texts": direct.texts,
        "student_tokens": direct.student_tokens,
        "fertility": direct.fertility,
        "direct_map": {
            "roundtrip_exact_rate": direct.roundtrip_rate,
            "canonical_boundary_rate": direct.canonical_rate,
            "supervised_position_rate": direct_coverage,
        },
        "byte_walk": {
            "supervised_position_rate": walk["one_to_one_rate"],
            "grouped_positions": walk["grouped_positions"],
        },
    }


def load_mephisto(repo, rows, teacher):
    """Rows rendered exactly as the SFT stage renders them."""
    from datasets import load_dataset

    stream = load_dataset(repo, split="train", streaming=True)
    chats = []
    for index, record in enumerate(stream):
        if index >= rows:
            break
        messages = list(record["messages"])
        if not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system",
                         "content": record.get("system") or SYSTEM}, *messages]
        chats.append(teacher.apply_chat_template(messages, tokenize=False))
    return chats


def load_climbmix(chars):
    from datasets import load_dataset

    stream = load_dataset("karpathy/climbmix-400b-shuffle", split="train",
                          streaming=True)
    texts, seen = [], 0
    for record in stream:
        texts.append(record["text"])
        seen += len(record["text"])
        if seen >= chars:
            break
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="tokenizers/yx49k")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mapping",
                        default="tokenizers/yx49k/student_to_teacher.npy")
    parser.add_argument("--covered",
                        default="tokenizers/yx49k/teacher_covered.npy")
    parser.add_argument("--mephisto", default="Yxanul/Mephisto-IF_172k")
    parser.add_argument("--mephisto-rows", type=int, default=300)
    parser.add_argument("--transcripts", default=None)
    parser.add_argument("--climbmix-chars", type=int, default=400_000)
    parser.add_argument("--output", default="/tmp/gold_alignment.json")
    arguments = parser.parse_args()

    import numpy as np
    from transformers import AutoTokenizer

    mapping = np.load(arguments.mapping)
    covered = np.load(arguments.covered)
    # A non-injective or out-of-range map silently biases the projection,
    # so fail here rather than three days into a run.
    validate_student_to_teacher(mapping, teacher_vocab=covered.shape[0])
    print(f"map ok: {mapping.size:,} student ids into a {covered.shape[0]:,} "
          f"teacher vocab, injective")

    student = AutoTokenizer.from_pretrained(arguments.student, use_fast=True)
    teacher = AutoTokenizer.from_pretrained(arguments.teacher, use_fast=True)
    reports = []

    if arguments.mephisto:
        reports.append(measure(
            f"{arguments.mephisto} rendered chat",
            load_mephisto(arguments.mephisto, arguments.mephisto_rows, teacher),
            student, teacher, mapping))

    if arguments.transcripts:
        data = json.loads(Path(arguments.transcripts).read_text())
        rows = data["transcripts"] if isinstance(data, dict) else data
        for field, label in (("think", "chat think traces"),
                             ("answer", "chat answers")):
            reports.append(measure(
                label, [row.get(field, "") for row in rows],
                student, teacher, mapping))

    if arguments.climbmix_chars:
        reports.append(measure(
            "climbmix pretrain", load_climbmix(arguments.climbmix_chars),
            student, teacher, mapping))

    payload = {"student": arguments.student, "teacher": arguments.teacher,
               "corpora": reports}
    Path(arguments.output).write_text(json.dumps(payload, indent=2))
    print(f"\nwritten {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
