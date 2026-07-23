"""SFT data: render conversations, pack with shifted loss masks, iterate epochs."""

from __future__ import annotations

import numpy as np

from yxtpu_pretrain.sft.tokens import (
    DOCUMENT_SEPARATOR,
    IM_END,
    IM_MIDDLE,
    ROLE_TOKENS,
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
                 sources=None, shuffle_seed=None):
        from datasets import load_dataset

        self._tok = tokenizer
        stream = load_dataset(dataset, split="train", streaming=True)
        if shuffle_seed is not None:
            stream = stream.shuffle(seed=shuffle_seed, buffer_size=10_000)
        self._sources = set(sources) if sources else None
        self._iter = iter(stream)
        self._pi, self._pc = process_index, process_count
        self._row_idx = 0
        self._buffer_rows = buffer_rows
        self._B, self._T = process_batch, sequence_length
        self._need = self._B * self._T + 1
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
        self.metadata = {"streaming": True, "dataset": dataset}
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
                msgs = conversations_to_messages(record)
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
        contents = [m["content"] for msgs in batch for m in msgs]
        encoded = iter(self._tok(contents, add_special_tokens=False)["input_ids"])
        chunks_i, chunks_m = [self._ids], [self._mask]
        for msgs in batch:
            ids, mask = [DOCUMENT_SEPARATOR], [0]
            for m in msgs:
                role = m["role"]
                header = [ROLE_TOKENS[role], *self._role_ids[role], IM_MIDDLE]
                body = next(encoded)
                train = 1 if role == "assistant" else 0
                ids.extend(header + body)
                mask.extend([0] * len(header) + [train] * len(body))
                ids.append(IM_END)
                mask.append(train)
            chunks_i.append(np.asarray(ids, np.int32))
            chunks_m.append(np.asarray(mask, np.float32))
            self.rows_consumed += 1
        self._ids = np.concatenate(chunks_i)
        self._mask = np.concatenate(chunks_m)
        return True

    def get_state(self) -> dict[str, int]:
        return {"rows_consumed": int(self.rows_consumed),
                "rows_dropped": int(self.rows_dropped)}

    def set_state(self, payload) -> None:
        raise RuntimeError("streaming SFT data is not resumable")

    def __iter__(self):
        return self

    def __next__(self):
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
