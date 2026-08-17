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
fixed ``[max_images, H, W, 3]`` slot per sequence (or
``[max_images, grid^2, patch^2*3]`` when ``host_patchify`` is set),
blank-padded; a sequence of pure text simply never gathers the tower's
output. ``vision_mask`` marks (on label positions, like ``loss_mask``)
the tokens belonging to image-carrying rows, so the loss can be split by
modality on device.

Producer topology (2026-08): each stream (the vision corpus and every
text source) is drawn by its own fetch thread that prepares rows
(filter, decode, resize, tokenize) into a bounded per-source buffer; the
packer thread only pops. Any number of such packers run as producer
PROCESSES per host (``ProcessPooledMixedIterator``), shipping whole
process batches to the trainer - no GIL coupling with the training loop
and image decode across the host's cores.
"""

from __future__ import annotations

import queue as queue_module
import threading
import time
import traceback
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
    # Host-side patchify: images leave the packer as [grid^2, patch^2 * 3]
    # uint8 patch rows (lane-dense on TPU) instead of [H, W, 3].
    patch_size: int = 16
    host_patchify: bool = False

    @property
    def patch_grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def image_shape(self) -> tuple[int, ...]:
        if self.host_patchify:
            return (self.patch_grid**2, self.patch_size * self.patch_size * 3)
        return (self.image_size, self.image_size, 3)


def patchify_pixels(image: np.ndarray, patch_size: int) -> np.ndarray:
    """[H, W, 3] -> [(H/p)*(W/p), p*p*3], row-major patches, any dtype.

    Exactly the tower's ``_patchify`` (reshape/transpose/reshape), so the
    on-device patch embedding sees identical rows either way."""
    height, width, channels = image.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = image.reshape(grid_h, patch_size, grid_w, patch_size, channels)
    patches = patches.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(
        patches.reshape(grid_h * grid_w, patch_size * patch_size * channels)
    )


def _row_images(image) -> list:
    """Normalizes a row's image field to a list (None -> [], array -> [array])."""
    if image is None:
        return []
    if isinstance(image, (list, tuple)):
        return list(image)
    return [image]


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


def pack_rows(rows, spec: VisionBatchSpec, *, with_images: bool = True):
    """Packs complete rows into one example with per-row segments.

    ``rows`` is a list of ``(token_list, images)`` where ``images`` is
    None, one array, or a list of arrays (multi-image rows: the k-th
    placeholder run of a row takes the row's k-th image, in order). The
    total token count must not exceed ``sequence_length + 1`` and the
    image count must not exceed ``max_images``. ``with_images=False``
    builds the text-only contract (no ``images``/``vision_mask`` keys) the
    "rows" holdout packing uses."""
    flat, segments, positions, images = [], [], [], []
    has_image = np.zeros(len(rows) + 1, dtype=bool)
    for index, (tokens, image) in enumerate(rows, start=1):
        flat.extend(tokens)
        segments.extend([index] * len(tokens))
        positions.extend(range(len(tokens)))
        row_images = _row_images(image)
        if row_images:
            images.extend(row_images)
            has_image[index] = True
    if len(images) > spec.max_images:
        raise ValueError("packed rows exceed the image budget")
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
    example = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": mask,
        "segment_ids": segment_in,
        "positions": positions[:-1],
    }
    if not with_images:
        return example
    vision_mask = has_image[segment_label].astype(np.float32)

    # The block dtype follows the rows' images (uint8 in the production
    # pipeline; float in unit tests exercising the tower's float path).
    block_dtype = images[0].dtype if images else np.uint8
    image_block = np.zeros((spec.max_images, *spec.image_shape), dtype=block_dtype)
    for slot, image in enumerate(images):
        if spec.host_patchify and image.ndim == 3:
            image = patchify_pixels(image, spec.patch_size)
        image_block[slot] = image
    example["images"] = image_block
    example["vision_mask"] = vision_mask
    return example


def pack_text_rows(token_rows, sequence_length: int, pad_id: int):
    """Text-only rows -> the training packer's per-row-segment example.

    Used by the "rows" holdout packing so held-out documents are scored
    under exactly the contract the mixed training batches use (segment
    isolation, restarting positions, masked row boundaries)."""
    spec = VisionBatchSpec(
        sequence_length=sequence_length,
        visual_tokens=0,
        image_size=0,
        placeholder_id=-1,
        pad_id=pad_id,
        eos_id=pad_id,
        max_images=0,
    )
    return pack_rows([(tokens, None) for tokens in token_rows], spec, with_images=False)


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


class _SourceFetcher:
    """One stream's draw -> prepare thread with a bounded prepared-row buffer.

    ``prepare(row)`` returns ``(rows, skipped)``: zero or more prepared rows
    for the packer and the number of rows rejected on the way. Any draw
    failure is handled by ``draw`` (reopen with backoff); an exception that
    escapes ``prepare`` is re-raised on the packer thread at the next
    ``get``. Counters are owned by this object and read approximately by
    the packer (monotonic ints)."""

    _SENTINEL = object()

    def __init__(self, name: str, draw, prepare, buffer: int):
        self.name = name
        self._draw = draw
        self._prepare = prepare
        self._queue: queue_module.Queue = queue_module.Queue(maxsize=max(1, buffer))
        self.rows_prepared = 0
        self.rows_skipped = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name=f"fetch-{name}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                rows, skipped = self._prepare(self._draw())
                self.rows_skipped += skipped
                for row in rows:
                    self._queue.put(row)
                    self.rows_prepared += 1
        except BaseException as error:  # noqa: BLE001 - re-raised on the packer thread
            self._error = error
            self._queue.put(self._SENTINEL)

    def get(self):
        item = self._queue.get()
        if item is self._SENTINEL:
            raise RuntimeError(f"stream {self.name}: fetch thread died") from self._error
        return item

    @property
    def depth(self) -> int:
        return self._queue.qsize()


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
        # Rows may carry 1..max_images_per_row images (bounded by
        # spec.max_images); rows outside that range are skipped.
        max_images_per_row: int = 1,
        # Per-source prepared-row buffer served by a fetch thread per
        # stream; 0 draws inline on the packer thread.
        row_buffer: int = 512,
    ):
        from yxtpu_pretrain.runtime.data import open_streaming_dataset

        if p_text > 0.0 and not text_sources:
            raise ValueError("p_text > 0 requires at least one text source")
        if not 1 <= max_images_per_row <= max(1, spec.max_images):
            raise ValueError("max_images_per_row must lie in [1, spec.max_images]")
        self.tokenizer = tokenizer
        self.spec = spec
        self.batch_size = batch_size
        self.p_text = p_text
        self.text_row_tokens = text_row_tokens
        self.min_visual_dependency = min_visual_dependency
        self.max_images_per_row = max_images_per_row
        self._rng = np.random.default_rng(shuffle_seed * 1009 + shard_index)

        def open_stream(name, seed, subset=None):
            stream = open_streaming_dataset(
                name, subset, split=split, label=f"{name}[shard {shard_index}]"
            )
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

        self._vision = {
            "name": "vision",
            "open": lambda: open_stream(vision_dataset, shuffle_seed),
        }
        self._vision["stream"] = self._vision["open"]()
        self._vision_ready: list = []
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
            entry["open"] = (
                lambda dataset=entry["dataset"],
                seed=shuffle_seed + 1 + offset,
                subset=entry.get("subset"): open_stream(dataset, seed, subset)
            )
            entry["stream"] = entry["open"]()
            entry["queue"] = []
            entry["ready"] = []
            entry["loss_tokens"] = 0
            entry["rows"] = 0
            entry["row_tokens_total"] = 0
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
        self.vision_row_tokens_total = 0
        self._fetchers: dict[str, _SourceFetcher] = {}
        if row_buffer > 0:
            self._fetchers["vision"] = _SourceFetcher(
                "vision",
                lambda: self._draw(self._vision),
                self._prepare_vision_rows,
                row_buffer,
            )
            for entry in self._text_sources:
                self._fetchers[entry["name"]] = _SourceFetcher(
                    entry["name"],
                    lambda entry=entry: self._draw(entry),
                    lambda row, entry=entry: self._prepare_text_rows(entry, row),
                    row_buffer,
                )

    def _draw(self, holder):
        """One row from a stream, reopening it on ANY failure.

        Multi-day streaming dies in varied ways - hung reads (fixed with
        HTTP timeouts), 'client has been closed' after a retry aborts the
        shared HTTP client, transient parquet errors, shard exhaustion.
        Every one becomes a reopen-and-continue with backoff: the stream
        restarts from its shard head with a fresh client, the same
        semantics a weights-only resume already has. Only six consecutive
        failures on one stream raise."""
        failures = 0
        while True:
            try:
                return next(holder["stream"])
            except StopIteration:
                print(f"stream {holder['name']}: exhausted, reopening", flush=True)
                holder["stream"] = holder["open"]()
            except Exception as error:
                failures += 1
                print(
                    f"stream {holder['name']}: draw failed "
                    f"({type(error).__name__}: {str(error)[:120]}), "
                    f"reopen {failures}/6",
                    flush=True,
                )
                if failures >= 6:
                    raise
                time.sleep(min(2.0**failures, 30.0))
                try:
                    holder["stream"] = holder["open"]()
                except Exception as reopen_error:
                    print(
                        f"stream {holder['name']}: reopen failed "
                        f"({type(reopen_error).__name__})",
                        flush=True,
                    )

    # ----- row preparation (fetch threads, or inline when row_buffer == 0)

    def _prepare_vision_rows(self, row):
        """FineVision row -> ([(tokens, [images], "vision")], skipped)."""
        spec = self.spec
        images = row.get("images") or []
        turns = row.get("texts") or []
        if not 1 <= len(images) <= self.max_images_per_row or not turns:
            return [], 1
        rating = row.get("visual_dependency_min")
        if (
            self.min_visual_dependency
            and rating is not None
            and rating < self.min_visual_dependency
        ):
            return [], 1
        text = render_texts(turns)
        if not text:
            return [], 1
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = [i for i in ids if i != spec.placeholder_id]
        if not ids:
            return [], 1
        text_cap = spec.sequence_length - len(images) * spec.visual_tokens - 1
        if text_cap < 1:
            return [], 1
        try:
            pixels = [process_image(image, spec.image_size) for image in images]
        except Exception:
            return [], 1
        tokens = (
            [spec.placeholder_id] * (spec.visual_tokens * len(images))
            + ids[:text_cap]
            + [spec.eos_id]
        )
        return [(tokens, pixels, "vision")], 0

    def _row_texts(self, source, row) -> list[str]:
        """The documents one source row yields (plain: 0/1; repo: per file).

        ``repo`` (Stack-v3-style rows: one repository per row with a
        ``files[]`` array) emits each usable file's ``content`` as its own
        document - content ONLY, no paths or metadata in the token stream;
        the packer's per-row segments are the file boundaries. Vendor files
        are skipped, files are capped per repository, and content is
        char-sliced at collection time so giant repositories cannot stall
        the producer."""
        if source["format"] == "plain":
            text = (row.get(source["field"]) or "").strip()
            return [text] if text else []
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
        return contents

    def _tokenize_text_row(self, source, text):
        spec = self.spec
        ids = self.tokenizer.encode(text[: source["char_cap"]], add_special_tokens=False)
        ids = [i for i in ids if i != spec.placeholder_id]
        if len(ids) < 8:
            return None
        return ids[: source["row_tokens"]] + [spec.eos_id], None, source["name"]

    def _prepare_text_rows(self, source, row):
        """Text source row -> ([(tokens, None, name), ...], skipped)."""
        texts = self._row_texts(source, row)
        if not texts:
            return [], 1
        rows = []
        skipped = 0
        for text in texts:
            prepared = self._tokenize_text_row(source, text)
            if prepared is None:
                skipped += 1
            else:
                rows.append(prepared)
        return rows, skipped

    # ----- row supply to the packer

    def _next_vision_row(self):
        fetcher = self._fetchers.get("vision")
        if fetcher is not None:
            row = fetcher.get()
        else:
            while not self._vision_ready:
                rows, skipped = self._prepare_vision_rows(self._draw(self._vision))
                self.rows_skipped += skipped
                self._vision_ready.extend(rows)
            row = self._vision_ready.pop(0)
        self.vision_rows += 1
        return row

    def _pick_source(self):
        threshold = self._rng.random() * self._weight_total
        cumulative = 0.0
        for entry in self._text_sources:
            cumulative += entry["weight"]
            if threshold < cumulative:
                return entry
        return self._text_sources[-1]

    def _next_source_text(self, source) -> str:
        """One document's text from a source stream (inline path)."""
        while True:
            if source["queue"]:
                return source["queue"].pop()
            texts = self._row_texts(source, self._draw(source))
            if not texts:
                self.rows_skipped += 1
                continue
            texts.reverse()  # pop() then consumes in original file order
            source["queue"] = texts

    def _next_text_row(self):
        source = self._pick_source()
        fetcher = self._fetchers.get(source["name"])
        if fetcher is not None:
            row = fetcher.get()
        else:
            while True:
                prepared = self._tokenize_text_row(source, self._next_source_text(source))
                if prepared is not None:
                    row = prepared
                    break
                self.rows_skipped += 1
        self.text_rows += 1
        return row

    def _next_row(self):
        if self._text_sources and self._rng.random() < self.p_text:
            return self._next_text_row()
        return self._next_vision_row()

    @staticmethod
    def _image_count(image) -> int:
        return len(_row_images(image))

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
            needs_image = self._image_count(image)
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
            if self._image_count(image) == 0:
                self._source_loss_counter(source_name, len(tokens) - 1)
            else:
                self.vision_row_tokens_total += len(tokens)
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
                entry["rows"] = entry.get("rows", 0) + 1
                entry["row_tokens_total"] = entry.get("row_tokens_total", 0) + count + 1
                return

    def raw_counters(self) -> dict[str, int]:
        counters = {key: int(getattr(self, key)) for key in _COUNTER_KEYS}
        counters["rows_skipped"] += sum(
            fetcher.rows_skipped for fetcher in self._fetchers.values()
        )
        return counters

    def raw_source_loss_tokens(self) -> dict[str, int]:
        return {entry["name"]: int(entry["loss_tokens"]) for entry in self._text_sources}

    def raw_source_rows(self) -> dict[str, tuple[int, int]]:
        """Per-source ``(rows_packed, row_tokens_total)`` - the calibration
        record (mean row tokens = tokens / rows), including the vision rows."""
        result = {
            entry["name"]: (int(entry["rows"]), int(entry["row_tokens_total"]))
            for entry in self._text_sources
        }
        result["vision"] = (int(self.vision_rows), int(self.vision_row_tokens_total))
        return result

    @property
    def fetch_depths(self) -> dict[str, int]:
        return {name: fetcher.depth for name, fetcher in self._fetchers.items()}

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


def _sum_counters(sources_counters, sources_source_loss, max_images):
    totals = {key: 0 for key in _COUNTER_KEYS}
    source_totals: dict[str, int] = {}
    for counters in sources_counters:
        for key, value in counters.items():
            totals[key] += value
    for source_loss in sources_source_loss:
        for name, value in source_loss.items():
            source_totals[name] = source_totals.get(name, 0) + value
    return _derived_stats(totals, max_images, source_totals)


class PooledMixedIterator:
    """N producer threads over disjoint stream shards, one example queue.

    Each thread owns its own HF streams (they are not thread-safe) and its
    own per-source fetch threads; the consumer assembles batches in
    arrival order, so batch composition is nondeterministic across runs
    but each row is still seen exactly once per epoch per shard."""

    def __init__(self, factory, *, threads: int, batch_size: int):
        self.batch_size = batch_size
        self._sources = [factory(thread) for thread in range(threads)]
        self._queue: queue_module.Queue = queue_module.Queue(
            maxsize=max(2 * batch_size, 8)
        )
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
            sum(source.raw_counters()["rows_skipped"] for source in self._sources),
            sum(source.vision_rows for source in self._sources),
            sum(source.text_rows for source in self._sources),
        )

    @property
    def stats(self) -> dict[str, float]:
        return _sum_counters(
            [source.raw_counters() for source in self._sources],
            [source.raw_source_loss_tokens() for source in self._sources],
            self._sources[0].spec.max_images,
        )

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


@dataclass
class ProducerSpec:
    """Picklable description of one MixedVisionTextIterator (a producer
    process rebuilds tokenizer and streams from it; no live objects cross
    the process boundary)."""

    tokenizer_name: str
    padded_vocab_size: int
    spec: VisionBatchSpec
    batch_size: int
    vision_dataset: str
    text_sources: list
    p_text: float
    text_row_tokens: int
    min_visual_dependency: int
    shuffle_seed: int
    max_images_per_row: int
    row_buffer: int
    # Producer processes open their streams ``stagger_seconds * shard_index``
    # after start. Every open costs ~10-15 Hub ``api`` requests (paginated
    # repo-tree listing; ~41 per producer over the four sources) against an
    # account quota of 1000 per 5 minutes: 8 hosts x 4 producers exhausted
    # it on 2026-08-17 (the second launch of the day died at its first
    # open). Keep producer_processes <= 2 on 8 hosts and spread the opens.
    stagger_seconds: float = 3.0


def build_mixed_iterator(spec: ProducerSpec, *, shard_index: int, shard_count: int):
    from yxtpu_pretrain.runtime.data import load_fast_tokenizer

    tokenizer = load_fast_tokenizer(
        spec.tokenizer_name, padded_vocab_size=spec.padded_vocab_size
    )
    return MixedVisionTextIterator(
        tokenizer=tokenizer,
        spec=spec.spec,
        batch_size=spec.batch_size,
        vision_dataset=spec.vision_dataset,
        text_sources=spec.text_sources,
        p_text=spec.p_text,
        text_row_tokens=spec.text_row_tokens,
        min_visual_dependency=spec.min_visual_dependency,
        shuffle_seed=spec.shuffle_seed,
        shard_index=shard_index,
        shard_count=shard_count,
        max_images_per_row=spec.max_images_per_row,
        row_buffer=spec.row_buffer,
    )


def _die_with_parent(parent_pid: int) -> None:
    """Ties this producer's life to the trainer's.

    ``daemon=True`` only helps on a graceful interpreter exit; a trainer
    that dies by SIGABRT (the JAX distributed service's fatal-error path
    on a multi-host slice) leaves the spawned producers streaming forever
    - observed 2026-08-17: every worker kept its producers alive for an
    hour after two crashed launches, holding memory and Hub quota. Linux
    PR_SET_PDEATHSIG delivers SIGTERM when the parent goes; the getppid
    watchdog covers the cases it does not (thread-based parenting)."""
    import os
    import signal
    import threading

    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, int(signal.SIGTERM))  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - best effort, the watchdog remains
        pass

    def watch() -> None:
        while True:
            time.sleep(5.0)
            if os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=watch, name="parent-watchdog", daemon=True).start()


def _producer_main(
    worker: int, spec: ProducerSpec, shard_index: int, shard_count: int, out, parent_pid: int
):
    """Entry point of one producer process: batches + counters to ``out``."""
    import os

    _die_with_parent(parent_pid)
    # The children never touch the accelerator: keep any accidental JAX
    # import on CPU and the Rust tokenizer single-threaded per process.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        if spec.stagger_seconds > 0:
            time.sleep(spec.stagger_seconds * shard_index)
        iterator = build_mixed_iterator(spec, shard_index=shard_index, shard_count=shard_count)
        while True:
            batch = next(iterator)
            out.put(
                {
                    "worker": worker,
                    "batch": batch,
                    "counters": iterator.raw_counters(),
                    "source_loss": iterator.raw_source_loss_tokens(),
                    "source_rows": iterator.raw_source_rows(),
                    "fetch_depths": iterator.fetch_depths,
                }
            )
    except BaseException:  # noqa: BLE001 - reported to the trainer, which raises
        out.put({"worker": worker, "error": traceback.format_exc()})
        raise


class ProcessPooledMixedIterator:
    """N producer PROCESSES over disjoint stream shards, one batch queue.

    Spawn context (the trainer has libtpu initialized; forking it is
    unsafe). Each child builds its own MixedVisionTextIterator from a
    ProducerSpec, packs whole process batches and ships them - with its
    counters - through one multiprocessing queue; the trainer's prefetch
    thread unpickles (~40 MB, tens of ms per batch) off the main thread.
    Stats are the sum of each child's latest counters."""

    def __init__(
        self,
        spec: ProducerSpec,
        *,
        workers: int,
        shard_base: int,
        shard_count: int,
        max_pending_batches: int | None = None,
    ):
        import multiprocessing

        if workers < 1:
            raise ValueError("producer_processes must be positive")
        self.batch_size = spec.batch_size
        self._max_images = spec.spec.max_images
        import os

        context = multiprocessing.get_context("spawn")
        self._queue = context.Queue(maxsize=max_pending_batches or 2 * workers)
        self._latest: dict[int, dict] = {}
        self._processes = []
        parent_pid = os.getpid()
        for worker in range(workers):
            process = context.Process(
                target=_producer_main,
                args=(worker, spec, shard_base + worker, shard_count, self._queue, parent_pid),
                name=f"vision-producer-{worker}",
                daemon=True,
            )
            process.start()
            self._processes.append(process)

    def __iter__(self):
        return self

    def __next__(self):
        payload = self._queue.get()
        if "error" in payload:
            raise RuntimeError(
                f"producer process {payload['worker']} failed:\n{payload['error']}"
            )
        self._latest[payload["worker"]] = payload
        return payload["batch"]

    @property
    def stats(self) -> dict[str, float]:
        if not self._latest:
            return {}
        payloads = list(self._latest.values())
        stats = _sum_counters(
            [payload["counters"] for payload in payloads],
            [payload["source_loss"] for payload in payloads],
            self._max_images,
        )
        depths = [payload.get("fetch_depths", {}) for payload in payloads]
        for name in sorted({name for depth in depths for name in depth}):
            stats[f"fetch_depth_min_{name}"] = float(
                min(depth.get(name, 0) for depth in depths)
            )
        return stats

    def raw_source_rows(self) -> dict[str, tuple[int, int]]:
        totals: dict[str, list[int]] = {}
        for payload in self._latest.values():
            for name, (rows, tokens) in payload.get("source_rows", {}).items():
                entry = totals.setdefault(name, [0, 0])
                entry[0] += rows
                entry[1] += tokens
        return {name: (rows, tokens) for name, (rows, tokens) in totals.items()}

    def get_state(self):
        raise RuntimeError("mixed vision+text stream position is not resumable")

    def close(self) -> None:
        for process in self._processes:
            if process.is_alive():
                process.terminate()
