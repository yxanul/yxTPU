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

"""Streaming interleaved image-text batches from FineVision-class datasets.

Row schema (HuggingFaceM4/FineVisionMax): ``images`` (list of PIL images),
``texts`` (list of ``{user, assistant}`` turns), ``source``, and per-turn
quality ratings with ``*_min`` row minima — ``visual_dependency_min`` is
the one that selects text a model cannot predict without the image.

Each emitted sequence is one row rendered as

  [placeholder x visual_tokens] Q: {user}\\nA: {assistant}\\n ... <eos>

with the loss masked on the placeholder run, on any position whose label
is a placeholder, and on padding. v1 keeps one image and one row per
sequence (no cross-row packing); rows whose text contains the
placeholder id are cleaned by stripping it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VisionBatchSpec:
    sequence_length: int
    visual_tokens: int
    image_size: int
    placeholder_id: int
    pad_id: int
    eos_id: int


def process_image(image, image_size: int) -> np.ndarray:
    """PIL image -> [size, size, 3] float32 in [-1, 1]."""
    resized = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array * 2.0 - 1.0


def render_row(text_token_ids, image_array, spec: VisionBatchSpec):
    """Assembles one training example from tokenized text + one image.

    Returns None when the visual prefix leaves no room for supervised
    text. ``text_token_ids`` must already exclude the placeholder id."""
    budget = spec.sequence_length + 1 - spec.visual_tokens
    if budget < 8:
        return None
    text = list(text_token_ids[: budget - 1]) + [spec.eos_id]
    tokens = np.full(spec.sequence_length + 1, spec.pad_id, dtype=np.int32)
    tokens[: spec.visual_tokens] = spec.placeholder_id
    tokens[spec.visual_tokens : spec.visual_tokens + len(text)] = text
    valid_length = spec.visual_tokens + len(text)

    input_ids = tokens[:-1]
    labels = tokens[1:]
    # Supervise only real text predictions: not the visual prefix, not any
    # position whose label is a placeholder, not padding.
    mask = np.ones(spec.sequence_length, dtype=np.float32)
    mask[labels == spec.placeholder_id] = 0.0
    mask[valid_length - 1 :] = 0.0
    segments = (np.arange(spec.sequence_length) < valid_length - 1).astype(np.int32)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": mask,
        "segment_ids": segments,
        "positions": np.arange(spec.sequence_length, dtype=np.int32),
        "images": image_array[None],  # [1, H, W, 3]
    }


def render_texts(turns) -> str:
    parts = []
    for turn in turns:
        user = (turn.get("user") or "").strip()
        assistant = (turn.get("assistant") or "").strip()
        if user:
            parts.append(f"Q: {user}\n")
        if assistant:
            parts.append(f"A: {assistant}\n")
    return "".join(parts)


class FineVisionIterator:
    """Streams FineVision rows into fixed-shape host batches."""

    def __init__(
        self,
        *,
        tokenizer,
        spec: VisionBatchSpec,
        batch_size: int,
        dataset_name: str = "HuggingFaceM4/FineVisionMax",
        split: str = "train",
        min_visual_dependency: int = 0,
        shuffle_seed: int = 42,
        shuffle_buffer: int = 1000,
        shard_index: int = 0,
        shard_count: int = 1,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.spec = spec
        self.batch_size = batch_size
        self.min_visual_dependency = min_visual_dependency
        stream = load_dataset(dataset_name, split=split, streaming=True)
        if shuffle_buffer:
            stream = stream.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
        self._stream = iter(stream)
        self._shard_index = shard_index
        self._shard_count = shard_count
        self._ordinal = 0
        self.rows_consumed = 0
        self.rows_skipped = 0

    def _next_example(self):
        while True:
            row = next(self._stream)
            self._ordinal += 1
            if (self._ordinal - 1) % self._shard_count != self._shard_index:
                continue
            images = row.get("images") or []
            turns = row.get("texts") or []
            if len(images) != 1 or not turns:
                self.rows_skipped += 1
                continue
            rating = row.get("visual_dependency_min")
            if (
                self.min_visual_dependency
                and rating is not None
                and rating < self.min_visual_dependency
            ):
                self.rows_skipped += 1
                continue
            text = render_texts(turns)
            if not text:
                self.rows_skipped += 1
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = [i for i in ids if i != self.spec.placeholder_id]
            try:
                image = process_image(images[0], self.spec.image_size)
            except Exception:
                self.rows_skipped += 1
                continue
            example = render_row(ids, image, self.spec)
            if example is None:
                self.rows_skipped += 1
                continue
            self.rows_consumed += 1
            return example

    def __iter__(self):
        return self

    def __next__(self):
        examples = [self._next_example() for _ in range(self.batch_size)]
        return {
            key: np.stack([example[key] for example in examples])
            for key in examples[0]
        }
