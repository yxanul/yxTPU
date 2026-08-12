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
uses. Segment ids mask cross-row GQA attention and zero padded
positions, but the KDA recurrent state is NOT reset at row boundaries:
it only decays, with per-token log decay bounded by the safe gate, so a
bounded channel-dependent carryover crosses each boundary (fast channels
forget within tokens; slow channels can carry for hundreds). Loss
masking still confines every label to its own row.

Loss falls only on real text predicted from within its own row: row
boundaries, placeholder labels, and padding are masked. Images ride in a
fixed ``[max_images, H, W, 3]`` slot per sequence, blank-padded; a
sequence of pure text simply never gathers the tower's output.
``vision_mask`` marks (on label positions, like ``loss_mask``) the
tokens belonging to image-carrying rows, so the loss can be split by
modality on device.
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
    """PIL image -> [size, size, 3] uint8 raw pixels.

    Pixels stay uint8 through packing and the host->device transfer (4x
    less traffic than fp32 - the fp32 image block was 154 MB/host/step
    and its assembly chronically exceeded the 1.5 s device step); the
    vision tower normalizes to [-1, 1] on device."""
    resized = image.convert("RGB").resize((image_size, image_size))
    return np.asarray(resized, dtype=np.uint8)


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
    has_image = np.zeros(len(rows) + 1, dtype=bool)
    for index, (tokens, image) in enumerate(rows, start=1):
        flat.extend(tokens)
        segments.extend([index] * len(tokens))
        positions.extend(range(len(tokens)))
        if image is not None:
            images.append(image)
            has_image[index] = True
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
    vision_mask = has_image[segment_label].astype(np.float32)

    # The block dtype follows the rows' images (uint8 in the production
    # pipeline; float in unit tests exercising the tower's float path).
    block_dtype = images[0].dtype if images else np.uint8
    image_block = np.zeros(
        (spec.max_images, spec.image_size, spec.image_size, 3), dtype=block_dtype
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
        "vision_mask": vision_mask,
    }


_COUNTER_KEYS = (
    "rows_consumed",
    "rows_skipped",
    "vision_rows",
    "text_rows",
    "sequences_packed",
    "tokens_total",
    "tokens_padding",
    "tokens_placeholder",
    "images_packed",
    "loss_tokens_vision",
    "loss_tokens_text",
)


def _derived_stats(
    counters: dict[str, int],
    max_images: int,
    source_loss_tokens: dict[str, int] | None = None,
) -> dict[str, float]:
    """Raw packing counters plus the ratios that describe batch composition.

    The ratios are the knobs' instruments: ``vision_loss_token_share`` is the
    realized modality mix the loss actually sees (``p_text`` is a row-level
    Bernoulli, so the token-level mix must be measured, not assumed);
    ``pad_fraction`` exposes packing waste (padding still costs compute);
    ``image_slot_utilization`` says whether ``max_images`` binds."""
    stats: dict[str, float] = {key: float(counters[key]) for key in _COUNTER_KEYS}
    sequences = counters["sequences_packed"]
    tokens = counters["tokens_total"]
    loss_total = counters["loss_tokens_vision"] + counters["loss_tokens_text"]
    if tokens:
        stats["pad_fraction"] = counters["tokens_padding"] / tokens
        stats["visual_token_fraction"] = counters["tokens_placeholder"] / tokens
    if sequences:
        stats["images_per_sequence"] = counters["images_packed"] / sequences
        stats["image_slot_utilization"] = counters["images_packed"] / (
            sequences * max_images
        )
        stats["rows_per_sequence"] = counters["rows_consumed"] / sequences
    if loss_total:
        stats["vision_loss_token_share"] = counters["loss_tokens_vision"] / loss_total
        for name, tokens_of_source in (source_loss_tokens or {}).items():
            stats[f"{name}_loss_token_share"] = tokens_of_source / loss_total
    rows_seen = counters["rows_consumed"] + counters["rows_skipped"]
    if rows_seen:
        stats["row_skip_rate"] = counters["rows_skipped"] / rows_seen
    return stats


class MixedVisionTextIterator:
    """Streams packed mixed batches from a vision and a text corpus."""

    def __init__(
        self,
        *,
        tokenizer,
        spec: VisionBatchSpec,
        batch_size: int,
        vision_dataset: str = "HuggingFaceM4/FineVisionMax",
        # Weighted text sources: each entry is a dict with ``name``,
        # ``dataset``, optional ``subset``, ``weight``, ``field``,
        # ``format`` ("plain" | "repo") and optional ``row_tokens``.
        # A text draw picks a source by normalized weight.
        text_sources: tuple | list = (),
        p_text: float = 0.0,
        text_row_tokens: int = 1024,
        min_visual_dependency: int = 0,
        split: str = "train",
        shuffle_seed: int = 42,
        # Image rows are heavy; the buffer must fill before the first
        # example emerges, so keep it modest and let file-shard order plus
        # a small window provide the mixing.
        shuffle_buffer: int = 256,
        shard_index: int = 0,
        shard_count: int = 1,
    ):
        from datasets import load_dataset

        if p_text > 0.0 and not text_sources:
            raise ValueError("p_text > 0 requires at least one text source")
        self.tokenizer = tokenizer
        self.spec = spec
        self.batch_size = batch_size
        self.p_text = p_text
        self.text_row_tokens = text_row_tokens
        self.min_visual_dependency = min_visual_dependency
        self._rng = np.random.default_rng(shuffle_seed * 1009 + shard_index)

        def open_stream(name, seed, subset=None):
            stream = load_dataset(name, subset, split=split, streaming=True)
            # Shard by FILES, never by row-modulo: a modulo shard still
            # downloads every row and discards shard_count-1 of every
            # shard_count - at 8 hosts x 4 producer threads that is 32x
            # redundant download of image-heavy rows, and the first
            # example only appears after shuffle_buffer * shard_count
            # raw rows. File shards are disjoint from the first byte.
            if shard_count > 1:
                try:
                    stream = stream.shard(num_shards=shard_count, index=shard_index)
                except Exception:
                    base = stream

                    def modulo(source):
                        for ordinal, row in enumerate(source):
                            if ordinal % shard_count == shard_index:
                                yield row

                    return iter(modulo(iter(base)))
            if shuffle_buffer:
                stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
            return iter(stream)

        self._vision = open_stream(vision_dataset, shuffle_seed)
        self._text_sources = []
        for offset, source in enumerate(text_sources):
            entry = dict(source)
            entry.setdefault("field", "text")
            entry.setdefault("format", "plain")
            entry.setdefault("weight", 1.0)
            if not entry.get("row_tokens"):
                entry["row_tokens"] = text_row_tokens
            # Slice text to this many chars BEFORE tokenizing: ~8 chars per
            # eventual token of budget, a generous margin over the ~3-5
            # chars/token of prose and code, so multi-megabyte documents
            # and source files never reach the tokenizer whole (data_wait
            # protection).
            entry["char_cap"] = 8 * entry["row_tokens"]
            entry["stream"] = open_stream(
                entry["dataset"], shuffle_seed + 1 + offset, entry.get("subset")
            )
            entry["queue"] = []
            entry["loss_tokens"] = 0
            self._text_sources.append(entry)
        self._weight_total = sum(entry["weight"] for entry in self._text_sources)
        self._pending = None
        self._pending_fill = None
        self.rows_consumed = 0
        self.rows_skipped = 0
        self.text_rows = 0
        self.vision_rows = 0
        self.sequences_packed = 0
        self.tokens_total = 0
        self.tokens_padding = 0
        self.tokens_placeholder = 0
        self.images_packed = 0
        self.loss_tokens_vision = 0
        self.loss_tokens_text = 0

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
            return tokens, image, "vision"

    def _pick_source(self):
        threshold = self._rng.random() * self._weight_total
        cumulative = 0.0
        for entry in self._text_sources:
            cumulative += entry["weight"]
            if threshold < cumulative:
                return entry
        return self._text_sources[-1]

    def _next_source_text(self, source) -> str:
        """One document's text from a source stream.

        ``plain`` reads the text field. ``repo`` (Stack-v3-style rows: one
        repository per row with a ``files[]`` array) emits each usable
        file's ``content`` as its own document - content ONLY, no paths or
        metadata in the token stream; the packer's per-row segments are
        the file boundaries. Vendor files are skipped, files are capped
        per repository, and content is char-sliced at collection time so
        giant repositories cannot stall the producer."""
        if source["format"] == "plain":
            while True:
                row = next(source["stream"])
                text = (row.get(source["field"]) or "").strip()
                if text:
                    return text
                self.rows_skipped += 1
        while True:
            if source["queue"]:
                return source["queue"].pop()
            row = next(source["stream"])
            contents = []
            for file in row.get("files") or []:
                if file.get("is_vendor"):
                    continue
                content = file.get("content") or ""
                if len(content) < 32:
                    continue
                contents.append(content[: source["char_cap"]])
                if len(contents) >= 64:
                    break
            if not contents:
                self.rows_skipped += 1
                continue
            contents.reverse()  # pop() then consumes in original file order
            source["queue"] = contents

    def _next_text_row(self):
        spec = self.spec
        source = self._pick_source()
        while True:
            text = self._next_source_text(source)[: source["char_cap"]]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = [i for i in ids if i != spec.placeholder_id]
            if len(ids) < 8:
                self.rows_skipped += 1
                continue
            self.text_rows += 1
            return ids[: source["row_tokens"]] + [spec.eos_id], None, source["name"]

    def _next_row(self):
        if self._text_sources and self._rng.random() < self.p_text:
            return self._next_text_row()
        return self._next_vision_row()

    def _next_example(self):
        spec = self.spec
        capacity = spec.sequence_length + 1
        rows, used, image_count = [], 0, 0
        while True:
            if self._pending is not None:
                row, self._pending = self._pending, None
            elif self._pending_fill is not None:
                row, self._pending_fill = self._pending_fill, None
            else:
                row = self._next_row()
            tokens, image, _ = row
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
        # Budget-aware fill: when a stashed row ended the pack early (the
        # image budget bound, or an oversized draw against a part-full
        # sequence), the tail would otherwise be padding at full compute
        # cost - measured at pad_fraction 0.28 under p_text 0.3. Text rows
        # need no image slot, so fill the tail from the text streams; a
        # draw that does not fit is stashed like the main loop's pending
        # row and opens a later sequence.
        if self._text_sources and self._pending is not None:
            while used < capacity - 8:
                if self._pending_fill is not None:
                    row, self._pending_fill = self._pending_fill, None
                else:
                    row = self._next_text_row()
                tokens, _, _ = row
                if used + len(tokens) > capacity:
                    self._pending_fill = row
                    break
                rows.append(row)
                used += len(tokens)
                self.rows_consumed += 1
        # Per-source supervision accounting: a fully packed text row of L
        # tokens contributes exactly L-1 in-segment labels (the boundary
        # label is masked), so consume-time counting is exact.
        for tokens, image, source_name in rows:
            if image is None:
                self._source_loss_counter(source_name, len(tokens) - 1)
        example = pack_rows([(tokens, image) for tokens, image, _ in rows], spec)
        loss_mask = example["loss_mask"]
        vision_mask = example["vision_mask"]
        self.sequences_packed += 1
        self.tokens_total += int(example["input_ids"].size)
        self.tokens_padding += int((example["segment_ids"] == 0).sum())
        self.tokens_placeholder += int((example["input_ids"] == spec.placeholder_id).sum())
        self.images_packed += image_count
        self.loss_tokens_vision += int((loss_mask * vision_mask).sum())
        self.loss_tokens_text += int((loss_mask * (1.0 - vision_mask)).sum())
        return example

    def _source_loss_counter(self, name: str, count: int) -> None:
        for entry in self._text_sources:
            if entry["name"] == name:
                entry["loss_tokens"] += count
                return

    def raw_counters(self) -> dict[str, int]:
        return {key: int(getattr(self, key)) for key in _COUNTER_KEYS}

    def raw_source_loss_tokens(self) -> dict[str, int]:
        return {entry["name"]: int(entry["loss_tokens"]) for entry in self._text_sources}

    @property
    def stats(self) -> dict[str, float]:
        return _derived_stats(
            self.raw_counters(), self.spec.max_images, self.raw_source_loss_tokens()
        )

    def get_state(self):
        """Checkpoint contract: the mixed stream position is not resumable.

        ``CheckpointIO.save`` catches this and stores the unresumable
        sentinel, so restores keep weights+optimizer and restart the
        stream - the same policy as text streaming."""
        raise RuntimeError("mixed vision+text stream position is not resumable")

    def __iter__(self):
        return self

    def __next__(self):
        examples = [self._next_example() for _ in range(self.batch_size)]
        return {
            key: np.stack([example[key] for example in examples])
            for key in examples[0]
        }


class PooledMixedIterator:
    """N producer threads over disjoint stream shards, one example queue.

    A single producer thread decodes images, tokenizes, and packs at
    roughly one 4k sequence per 25 ms - slower than a heavy training
    step consumes them across a local batch. The pool splits the shard
    N ways (each thread owns its own HF streams; they are not
    thread-safe) and the consumer assembles batches in arrival order,
    so batch composition is nondeterministic across runs but each row
    is still seen exactly once per epoch per shard."""

    def __init__(self, factory, *, threads: int, batch_size: int):
        import queue
        import threading

        self.batch_size = batch_size
        self._sources = [factory(thread) for thread in range(threads)]
        self._queue: queue.Queue = queue.Queue(maxsize=max(2 * batch_size, 8))
        self._threads = []
        for source in self._sources:
            worker = threading.Thread(
                target=self._produce, args=(source,), daemon=True
            )
            worker.start()
            self._threads.append(worker)

    def _produce(self, source) -> None:
        while True:
            try:
                example = source._next_example()
            except StopIteration:
                return
            self._queue.put(example)

    def rows_stats(self):
        return (
            sum(source.rows_consumed for source in self._sources),
            sum(source.rows_skipped for source in self._sources),
            sum(source.vision_rows for source in self._sources),
            sum(source.text_rows for source in self._sources),
        )

    @property
    def stats(self) -> dict[str, float]:
        totals = {key: 0 for key in _COUNTER_KEYS}
        source_totals: dict[str, int] = {}
        for source in self._sources:
            for key, value in source.raw_counters().items():
                totals[key] += value
            for name, value in source.raw_source_loss_tokens().items():
                source_totals[name] = source_totals.get(name, 0) + value
        return _derived_stats(totals, self._sources[0].spec.max_images, source_totals)

    def get_state(self):
        raise RuntimeError("mixed vision+text stream position is not resumable")

    def __iter__(self):
        return self

    def __next__(self):
        examples = [self._queue.get() for _ in range(self.batch_size)]
        return {
            key: np.stack([example[key] for example in examples])
            for key in examples[0]
        }
