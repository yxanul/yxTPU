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
