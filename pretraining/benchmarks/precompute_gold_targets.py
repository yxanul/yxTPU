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

"""Precompute per-example GOLD teacher targets for the lambda=0 stage.

Scores every rendered Mephisto example with the teacher ONCE and stores
its top-K target triple, so training sweeps never pay the teacher's
forward again. Two properties are deliberate:

* Each example is scored IN ISOLATION - positions from zero, one segment,
  no packing. That is the exact conditioning the teacher generated the
  answer under, so the stored targets are free of both packing pollution
  and the recurrent-state segment leak, on the teacher side, permanently.
* The store is keyed by a hash of the rendered token ids, so the training
  iterator may shuffle, interleave and pack however it likes; a render
  drift (tokenizer, system prompt, template) surfaces as missing keys,
  not as wrong supervision.

Examples are bucketed by length so the teacher compiles once per bucket
shape, and only the top-K columns cross the device boundary.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/precompute_gold_targets.py \
    --teacher-checkpoint /home/a1111/yxtpu_ckpts/qwen35-4b-teacher \
    --limit 2048 --output-dir ckpt/gold-targets-smoke

Fan-out across hosts: run one process per host with --shard-index i
--shard-count N and a distinct --prefix, then merge the directories; the
store reads every manifest it finds.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

SYSTEM = ("You are a helpful assistant. Answer the user's question "
          "accurately, clearly, and concisely.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizers/yx49k")
    parser.add_argument("--datasets", default="Yxanul/Mephisto-IF_172k")
    parser.add_argument("--mapping",
                        default="tokenizers/yx49k/student_to_teacher.npy")
    parser.add_argument("--covered",
                        default="tokenizers/yx49k/teacher_covered.npy")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--buckets", default="256,512,1024")
    parser.add_argument("--limit", type=int, default=None,
                        help="rows taken per dataset (streaming order)")
    parser.add_argument("--system", default=SYSTEM)
    parser.add_argument("--tensor-parallelism", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    arguments = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from yxtpu_pretrain.distillation.store import GoldTargetWriter
    from yxtpu_pretrain.distillation.teacher import Qwen35Teacher
    from yxtpu_pretrain.sft.mephisto import ENDOFTEXT, render_example

    buckets = sorted(int(b) for b in arguments.buckets.split(","))
    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer,
                                              use_fast=True)
    covered = np.load(arguments.covered)
    teacher = Qwen35Teacher(
        arguments.teacher_checkpoint,
        student_to_teacher=np.load(arguments.mapping),
        sequence=buckets[-1], batch=arguments.batch,
        tensor_parallelism=arguments.tensor_parallelism,
        valid_vocab=int(covered.shape[0]),
    )
    writer = GoldTargetWriter(
        arguments.output_dir, k=arguments.k, prefix=arguments.prefix,
        metadata={
            "teacher_checkpoint": arguments.teacher_checkpoint,
            "datasets": arguments.datasets, "system": arguments.system,
            "tokenizer": arguments.tokenizer, "buckets": buckets,
            "shard_index": arguments.shard_index,
            "shard_count": arguments.shard_count,
        })

    queues: dict[int, list] = {bucket: [] for bucket in buckets}
    dropped_oversize = 0
    render_failures = 0
    batches = 0
    began = time.perf_counter()

    def flush(bucket: int) -> None:
        nonlocal batches
        rows = queues[bucket]
        if not rows:
            return
        batch = len(rows)
        ids = np.full((batch, bucket), ENDOFTEXT, np.int32)
        segments = np.zeros((batch, bucket), np.int32)
        for row, example_ids in enumerate(rows):
            ids[row, :len(example_ids)] = example_ids
            segments[row, :len(example_ids)] = 1
        positions = np.tile(np.arange(bucket, dtype=np.int32), (batch, 1))
        top_ids, top_logprobs, rest = teacher.score_topk(
            ids, positions, segments, arguments.k)
        top_ids = np.asarray(top_ids)
        top_logprobs = np.asarray(top_logprobs)
        rest = np.asarray(rest)
        for row, example_ids in enumerate(rows):
            length = len(example_ids)
            writer.add(example_ids, top_ids[row, :length],
                       top_logprobs[row, :length], rest[row, :length])
        queues[bucket] = []
        batches += 1
        if batches % arguments.progress_every == 0:
            elapsed = time.perf_counter() - began
            print(f"{writer.examples} examples / {writer.positions} positions "
                  f"in {elapsed:.0f}s ({writer.positions / elapsed:.0f} pos/s), "
                  f"dropped {dropped_oversize} oversize", flush=True)

    ordinal = 0
    for spec in (s for s in arguments.datasets.split(",") if s):
        stream = load_dataset(spec, split="train", streaming=True)
        if arguments.limit:
            stream = stream.take(arguments.limit)
        for record in stream:
            ordinal += 1
            if (ordinal - 1) % arguments.shard_count != arguments.shard_index:
                continue
            try:
                ids, _ = render_example(tokenizer, record,
                                        system=arguments.system)
            except (KeyError, ValueError):
                render_failures += 1
                continue
            if len(ids) > buckets[-1]:
                dropped_oversize += 1
                continue
            bucket = next(b for b in buckets if b >= len(ids))
            queues[bucket].append(ids)
            if len(queues[bucket]) >= arguments.batch:
                flush(bucket)
    for bucket in buckets:
        flush(bucket)

    summary = writer.close()
    summary.update({
        "seconds": round(time.perf_counter() - began, 1),
        "rows_seen": ordinal, "dropped_oversize": dropped_oversize,
        "render_failures": render_failures, "k": arguments.k,
    })
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
