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

"""Packed mixed vision+text streaming batches.

One iterator serves the joint diet: vision rows (FineVision-class
``images`` + ``texts`` records rendered as a visual-token prefix plus
Q/A text) and plain-text rows (a ClimbMix-class corpus, one truncated
document per row) are interleaved by a per-shard RNG at ``p_text`` and
greedily packed into fixed-length sequences with per-row segments and
restarting positions — the same whole-row packing contract the SFT stage
uses, so the KDA state resets and cross-row attention is masked at every
boundary by the existing model machinery.

Loss falls only on real text predicted from within its own row: row
boundaries, placeholder labels, and padding are masked. Images ride in a
fixed ``[max_images, H, W, 3]`` slot per sequence, blank-padded; a
sequence of pure text simply never gathers the tower's output.
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
    max_images: int = 4


def process_image(image, image_size: int) -> np.ndarray:
    """PIL image -> [size, size, 3] float32 in [-1, 1]."""
    resized = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array * 2.0 - 1.0


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


def pack_rows(rows, spec: VisionBatchSpec):
    """Packs complete rows into one example with per-row segments.

    ``rows`` is a list of ``(token_list, image_or_None)`` whose total
    token count must not exceed ``sequence_length + 1`` and whose image
    count must not exceed ``max_images``."""
    flat, segments, positions, images = [], [], [], []
    for index, (tokens, image) in enumerate(rows, start=1):
        flat.extend(tokens)
        segments.extend([index] * len(tokens))
        positions.extend(range(len(tokens)))
        if image is not None:
            images.append(image)
    capacity = spec.sequence_length + 1
    if len(flat) > capacity:
        raise ValueError("packed rows exceed the sequence budget")
    pad = capacity - len(flat)
    flat.extend([spec.pad_id] * pad)
    segments.extend([0] * pad)
    positions.extend([0] * pad)

    flat = np.asarray(flat, dtype=np.int32)
    segments = np.asarray(segments, dtype=np.int32)
    positions = np.asarray(positions, dtype=np.int32)
    input_ids, labels = flat[:-1], flat[1:]
    segment_in, segment_label = segments[:-1], segments[1:]
    mask = (
        (segment_label == segment_in)
        & (segment_label > 0)
        & (labels != spec.placeholder_id)
    ).astype(np.float32)

    image_block = np.zeros(
        (spec.max_images, spec.image_size, spec.image_size, 3), dtype=np.float32
    )
    for slot, image in enumerate(images):
        image_block[slot] = image
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": mask,
        "segment_ids": segment_in,
        "positions": positions[:-1],
        "images": image_block,
    }


class MixedVisionTextIterator:
    """Streams packed mixed batches from a vision and a text corpus."""

    def __init__(
        self,
        *,
        tokenizer,
        spec: VisionBatchSpec,
        batch_size: int,
        vision_dataset: str = "HuggingFaceM4/FineVisionMax",
        text_dataset: str | None = None,
        text_field: str = "text",
        p_text: float = 0.0,
        text_row_tokens: int = 1024,
        min_visual_dependency: int = 0,
        split: str = "train",
        shuffle_seed: int = 42,
        shuffle_buffer: int = 1000,
        shard_index: int = 0,
        shard_count: int = 1,
    ):
        from datasets import load_dataset

        if p_text > 0.0 and not text_dataset:
            raise ValueError("p_text > 0 requires a text_dataset")
        self.tokenizer = tokenizer
        self.spec = spec
        self.batch_size = batch_size
        self.p_text = p_text
        self.text_row_tokens = text_row_tokens
        self.text_field = text_field
        self.min_visual_dependency = min_visual_dependency
        self._rng = np.random.default_rng(shuffle_seed * 1009 + shard_index)

        def sharded(stream):
            for ordinal, row in enumerate(stream):
                if ordinal % shard_count == shard_index:
                    yield row

        vision_stream = load_dataset(vision_dataset, split=split, streaming=True)
        if shuffle_buffer:
            vision_stream = vision_stream.shuffle(
                seed=shuffle_seed, buffer_size=shuffle_buffer
            )
        self._vision = sharded(iter(vision_stream))
        if text_dataset:
            text_stream = load_dataset(text_dataset, split=split, streaming=True)
            if shuffle_buffer:
                text_stream = text_stream.shuffle(
                    seed=shuffle_seed + 1, buffer_size=shuffle_buffer
                )
            self._text = sharded(iter(text_stream))
        else:
            self._text = None
        self._pending = None
        self.rows_consumed = 0
        self.rows_skipped = 0
        self.text_rows = 0
        self.vision_rows = 0

    def _next_vision_row(self):
        spec = self.spec
        text_cap = spec.sequence_length - spec.visual_tokens - 1
        while True:
            row = next(self._vision)
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
            ids = [i for i in ids if i != spec.placeholder_id]
            if not ids:
                self.rows_skipped += 1
                continue
            try:
                image = process_image(images[0], spec.image_size)
            except Exception:
                self.rows_skipped += 1
                continue
            tokens = (
                [spec.placeholder_id] * spec.visual_tokens
                + ids[:text_cap]
                + [spec.eos_id]
            )
            self.vision_rows += 1
            return tokens, image

    def _next_text_row(self):
        spec = self.spec
        while True:
            document = next(self._text)
            text = (document.get(self.text_field) or "").strip()
            if not text:
                self.rows_skipped += 1
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = [i for i in ids if i != spec.placeholder_id]
            if len(ids) < 8:
                self.rows_skipped += 1
                continue
            self.text_rows += 1
            return ids[: self.text_row_tokens] + [spec.eos_id], None

    def _next_row(self):
        if self._text is not None and self._rng.random() < self.p_text:
            return self._next_text_row()
        return self._next_vision_row()

    def _next_example(self):
        spec = self.spec
        capacity = spec.sequence_length + 1
        rows, used, image_count = [], 0, 0
        while True:
            row = self._pending if self._pending is not None else self._next_row()
            self._pending = None
            tokens, image = row
            needs_image = 1 if image is not None else 0
            if used + len(tokens) > capacity or image_count + needs_image > spec.max_images:
                self._pending = row
                break
            rows.append(row)
            used += len(tokens)
            image_count += needs_image
            self.rows_consumed += 1
            if used >= capacity - 8:
                break
        return pack_rows(rows, spec)

    def __iter__(self):
        return self

    def __next__(self):
        examples = [self._next_example() for _ in range(self.batch_size)]
        return {
            key: np.stack([example[key] for example in examples])
            for key in examples[0]
        }
