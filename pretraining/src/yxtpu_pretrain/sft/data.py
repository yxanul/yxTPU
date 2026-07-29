"""SFT data: render conversations, pack with shifted loss masks, iterate epochs."""

from __future__ import annotations

import numpy as np

from yxtpu_pretrain.sft.tokens import (
    DOCUMENT_SEPARATOR,
    IM_END,
    IM_MIDDLE,
    ROLE_TOKENS,
    THINK_CLOSE,
    THINK_OPEN,
    normalize_messages,
    render_conversation,
)


def conversations_to_messages(record) -> list[dict]:
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    return [
        {"role": roles[turn["from"]], "content": turn["value"]}
        for turn in record["conversations"]
    ]


def build_packed_dataset(tokenizer, *, dataset, subset, rows, sequence_length):
    """Tokenizes the first `rows` conversations and packs them densely.

    labels are next-token targets, so the loss mask is the RENDER mask
    shifted by one: position t trains iff stream token t+1 is assistant
    content (or its <|im_end|>)."""
    from datasets import load_dataset

    stream = load_dataset(dataset, subset, split="train", streaming=True)
    conversations = []
    for record in stream:
        conversations.append(conversations_to_messages(record))
        if len(conversations) >= rows:
            break
    # One batched encode of every message body: the fast tokenizer
    # parallelizes internally, and per-string results are identical to
    # element-wise encoding (render_conversation stays the reference).
    contents = [m["content"] for msgs in conversations for m in msgs]
    encoded = iter(tokenizer(contents, add_special_tokens=False)["input_ids"])
    role_ids = {
        role: tokenizer.encode(role, add_special_tokens=False)
        for role in ROLE_TOKENS
    }
    all_ids, all_mask = [], []
    for msgs in conversations:
        ids, mask = [DOCUMENT_SEPARATOR], [0]
        for m in msgs:
            role = m["role"]
            header = [ROLE_TOKENS[role], *role_ids[role], IM_MIDDLE]
            body = next(encoded)
            trainable = 1 if role == "assistant" else 0
            ids.extend(header + body)
            mask.extend([0] * len(header) + [trainable] * len(body))
            ids.append(IM_END)
            mask.append(trainable)
        all_ids.append(np.asarray(ids, np.int32))
        all_mask.append(np.asarray(mask, np.float32))
    ids = np.concatenate(all_ids)
    mask = np.concatenate(all_mask)
    count = (len(ids) - 1) // sequence_length
    inputs = ids[: count * sequence_length].reshape(count, sequence_length)
    labels = ids[1 : count * sequence_length + 1].reshape(count, sequence_length)
    loss_mask = mask[1 : count * sequence_length + 1].reshape(count, sequence_length)
    return inputs, labels, loss_mask


class SFTIterator:
    """Deterministic multi-epoch iterator over pre-packed rows.

    Every process builds the identical pack, shuffles identically per
    epoch, then takes its rank-strided rows, so global batches are
    disjoint. Raises StopIteration when the epochs are exhausted."""

    def __init__(self, inputs, labels, loss_mask, *, process_batch, epochs,
                 seed, process_index, process_count):
        order = np.concatenate([
            np.random.default_rng(seed + epoch).permutation(len(inputs))
            for epoch in range(epochs)
        ])
        mine = order[process_index::process_count]
        usable = (len(mine) // process_batch) * process_batch
        self._rows = mine[:usable]
        self._inputs, self._labels, self._loss_mask = inputs, labels, loss_mask
        self._batch = process_batch
        self._cursor = 0
        length = inputs.shape[1]
        self._positions = np.tile(np.arange(length, dtype=np.int32), (process_batch, 1))
        self._segments = np.ones((process_batch, length), np.int32)
        self.metadata = {"epochs": epochs, "packed_rows": int(len(inputs))}
        self.stats = {}

    def get_state(self) -> dict[str, int]:
        return {"cursor": int(self._cursor)}

    def set_state(self, payload) -> None:
        self._cursor = int(payload["cursor"])

    def __iter__(self):
        return self

    def __next__(self):
        if self._cursor >= len(self._rows):
            raise StopIteration
        take = self._rows[self._cursor : self._cursor + self._batch]
        self._cursor += self._batch
        return {
            "input_ids": self._inputs[take],
            "labels": self._labels[take],
            "loss_mask": self._loss_mask[take],
            "segment_ids": self._segments,
            "positions": self._positions,
        }


class StreamingSFTIterator:
    """Rank-strided on-the-fly render+pack for single-epoch full-dataset SFT.

    Streams rows, drops conversations whose assistant turns open <think>
    without closing it (K2.5 dumps: ~0.05%), batch-encodes per buffer, and
    packs densely like pretraining. Not resumable; get_state reports
    progress counters only."""

    def __init__(self, tokenizer, *, dataset, sequence_length, process_batch,
                 process_index, process_count, buffer_rows=512,
                 sources=None, shuffle_seed=None, mixture=None, split="train",
                 max_render_tokens=None, pack_whole=False):
        from datasets import load_dataset

        self._tok = tokenizer
        if mixture:
            from datasets import interleave_datasets

            seed = shuffle_seed if shuffle_seed is not None else 0
            streams = [
                load_dataset(dataset, name, split=split, streaming=True).shuffle(
                    seed=seed, buffer_size=10_000
                )
                for name, _ in mixture
            ]
            stream = interleave_datasets(
                streams,
                probabilities=[probability for _, probability in mixture],
                seed=seed,
                stopping_strategy="all_exhausted",
            )
        else:
            stream = load_dataset(dataset, split=split, streaming=True)
            if shuffle_seed is not None:
                stream = stream.shuffle(seed=shuffle_seed, buffer_size=10_000)
        self._sources = set(sources) if sources else None
        self._iter = iter(stream)
        self._pi, self._pc = process_index, process_count
        self._row_idx = 0
        self._buffer_rows = buffer_rows
        self._B, self._T = process_batch, sequence_length
        self._need = self._B * self._T + 1
        self._max_render = max_render_tokens
        self._pack_whole = pack_whole
        self._pool: list[tuple[np.ndarray, np.ndarray]] = []
        self._exhausted = False
        self._ids = np.empty(0, np.int32)
        self._mask = np.empty(0, np.float32)
        self._role_ids = {
            role: tokenizer.encode(role, add_special_tokens=False)
            for role in ROLE_TOKENS
        }
        self._positions = np.tile(
            np.arange(sequence_length, dtype=np.int32), (process_batch, 1))
        self._segments = np.ones((process_batch, sequence_length), np.int32)
        self.rows_consumed = 0
        self.rows_dropped = 0
        self.rows_dropped_oversize = 0
        self.pad_tokens = 0
        self.metadata = {
            "streaming": True,
            "dataset": dataset,
            "mixture": [list(entry) for entry in mixture] if mixture else None,
            "split": split,
            "max_render_tokens": max_render_tokens,
            "pack_whole": pack_whole,
        }
        self.stats = {}

    def _refill(self) -> bool:
        batch = []
        while len(batch) < self._buffer_rows:
            try:
                record = next(self._iter)
            except StopIteration:
                break
            index = self._row_idx
            self._row_idx += 1
            if index % self._pc != self._pi:
                continue
            if self._sources and record.get("source") not in self._sources:
                self.rows_dropped += 1
                continue
            try:
                msgs = normalize_messages(record)
            except KeyError:
                self.rows_dropped += 1
                continue
            if any(m["role"] == "assistant" and "<think>" in m["content"]
                   and "</think>" not in m["content"] for m in msgs):
                self.rows_dropped += 1
                continue
            batch.append(msgs)
        if not batch:
            return False
        texts = []
        for msgs in batch:
            for m in msgs:
                if m.get("reasoning"):
                    texts.append(m["reasoning"])
                texts.append(m["content"])
        encoded = iter(self._tok(texts, add_special_tokens=False)["input_ids"])
        rendered = []
        for msgs in batch:
            ids, mask = [DOCUMENT_SEPARATOR], [0]
            for m in msgs:
                role = m["role"]
                header = [ROLE_TOKENS[role], *self._role_ids[role], IM_MIDDLE]
                body = []
                if m.get("reasoning"):
                    body.extend([THINK_OPEN, *next(encoded), THINK_CLOSE])
                body.extend(next(encoded))
                train = 1 if role == "assistant" else 0
                ids.extend(header + body)
                mask.extend([0] * len(header) + [train] * len(body))
                ids.append(IM_END)
                mask.append(train)
            if self._max_render is not None and len(ids) > self._max_render:
                self.rows_dropped_oversize += 1
                continue
            rendered.append(
                (np.asarray(ids, np.int32), np.asarray(mask, np.float32))
            )
            self.rows_consumed += 1
        if self._pack_whole:
            self._pool.extend(rendered)
        elif rendered:
            self._ids = np.concatenate([self._ids] + [r[0] for r in rendered])
            self._mask = np.concatenate([self._mask] + [r[1] for r in rendered])
        return True

    def _assemble_row(self):
        """First-fit packs whole conversations into one T+1 row; the padded
        tail is separator tokens with zero loss and segment 0 (excluded
        from attention). The stream refills only an empty pool — when the
        pool holds records but none fits the remaining gap, the row closes
        padded, keeping pool memory bounded at one refill buffer. Returns
        None once the stream and pool are dry."""
        capacity = self._T + 1
        row_ids: list[np.ndarray] = []
        row_mask: list[np.ndarray] = []
        filled = 0
        while filled < capacity:
            picked = None
            for position, (ids, _) in enumerate(self._pool):
                if len(ids) <= capacity - filled:
                    picked = position
                    break
            if picked is None:
                if not self._pool and not self._exhausted and self._refill():
                    continue
                if not self._pool:
                    self._exhausted = True
                break
            ids, mask = self._pool.pop(picked)
            row_ids.append(ids)
            row_mask.append(mask)
            filled += len(ids)
        if not filled:
            return None
        pad = capacity - filled
        segments = np.ones(self._T, np.int32)
        if pad:
            row_ids.append(np.full(pad, DOCUMENT_SEPARATOR, np.int32))
            row_mask.append(np.zeros(pad, np.float32))
            self.pad_tokens += pad
            segments[filled:] = 0
        return np.concatenate(row_ids), np.concatenate(row_mask), segments

    def get_state(self) -> dict[str, int]:
        return {"rows_consumed": int(self.rows_consumed),
                "rows_dropped": int(self.rows_dropped),
                "rows_dropped_oversize": int(self.rows_dropped_oversize),
                "pad_tokens": int(self.pad_tokens)}

    def set_state(self, payload) -> None:
        raise RuntimeError("streaming SFT data is not resumable")

    def __iter__(self):
        return self

    def __next__(self):
        if self._pack_whole:
            rows = []
            while len(rows) < self._B:
                row = self._assemble_row()
                if row is None:
                    raise StopIteration
                rows.append(row)
            ids = np.stack([row[0] for row in rows])
            mask = np.stack([row[1] for row in rows])
            segments = np.stack([row[2] for row in rows])
            return {
                "input_ids": ids[:, :-1],
                "labels": ids[:, 1:],
                "loss_mask": mask[:, 1:],
                "segment_ids": segments,
                "positions": self._positions,
            }
        while len(self._ids) < self._need:
            if not self._refill():
                raise StopIteration
        count = self._B * self._T
        batch = {
            "input_ids": self._ids[:count].reshape(self._B, self._T),
            "labels": self._ids[1:count + 1].reshape(self._B, self._T),
            "loss_mask": self._mask[1:count + 1].reshape(self._B, self._T),
            "segment_ids": self._segments,
            "positions": self._positions,
        }
        self._ids = self._ids[count:]
        self._mask = self._mask[count:]
        return batch
