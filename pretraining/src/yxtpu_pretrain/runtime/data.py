"""Synthetic, Hugging Face, and Grain pretraining iterators."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from yxtpu_pretrain.config import DataConfig

Batch = dict[str, np.ndarray]


def _example(tokens: np.ndarray, sequence_length: int) -> dict[str, np.ndarray]:
    tokens = np.asarray(tokens, dtype=np.int32)
    required = sequence_length + 1
    if tokens.size < required:
        padded = np.zeros(required, dtype=np.int32)
        padded[: tokens.size] = tokens
        tokens = padded
    else:
        tokens = tokens[:required]
    inputs = tokens[:-1]
    labels = tokens[1:]
    valid = (inputs != 0) & (labels != 0)
    segments = valid.astype(np.int32)
    return {
        "input_ids": inputs,
        "labels": labels,
        "loss_mask": valid.astype(np.float32),
        "segment_ids": segments,
        "positions": np.arange(sequence_length, dtype=np.int32),
    }


def _stack(examples: list[dict[str, np.ndarray]]) -> Batch:
    return {
        key: np.stack([example[key] for example in examples], axis=0)
        for key in examples[0]
    }


class SyntheticIterator(Iterator[Batch]):
    """Stateless-per-index random batches, exactly resumable by one integer."""

    def __init__(self, config: DataConfig, global_batch_size: int, vocab_size: int):
        self.config = config
        self.global_batch_size = global_batch_size
        self.vocab_size = vocab_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        rng = np.random.default_rng(self.config.shuffle_seed + self.index)
        tokens = rng.integers(
            1,
            self.vocab_size,
            size=(self.global_batch_size, self.config.sequence_length + 1),
            dtype=np.int32,
        )
        self.index += 1
        return {
            "input_ids": tokens[:, :-1],
            "labels": tokens[:, 1:],
            "loss_mask": np.ones(tokens[:, :-1].shape, dtype=np.float32),
            "segment_ids": np.ones(tokens[:, :-1].shape, dtype=np.int32),
            "positions": np.broadcast_to(
                np.arange(self.config.sequence_length, dtype=np.int32),
                tokens[:, :-1].shape,
            ).copy(),
        }

    def get_state(self) -> dict[str, int]:
        return {"index": self.index}

    def set_state(self, state: dict[str, Any]) -> None:
        self.index = int(state["index"])


def _read_json_records(path: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"dataset fixture contains no records: {path}")
    return records


def _prepare_records(config: DataConfig) -> list[dict[str, np.ndarray]]:
    if config.dataset_path is None and config.dataset_name is None:
        raise ValueError(f"{config.type} data requires dataset_path or dataset_name")
    if config.dataset_path is not None:
        raw_records = _read_json_records(config.dataset_path)
    else:
        from datasets import load_dataset

        dataset = load_dataset(config.dataset_name, split=config.split)
        raw_records = [dict(record) for record in dataset]

    tokenizer = None
    prepared = []
    for record in raw_records:
        tokens = record.get("input_ids")
        if tokens is None:
            text = record.get("text")
            if text is None:
                raise ValueError("records must contain input_ids or text")
            if config.tokenizer is None:
                raise ValueError("text records require data.tokenizer")
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
            tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        prepared.append(_example(np.asarray(tokens, dtype=np.int32), config.sequence_length))
    return prepared


class HuggingFaceIterator(Iterator[Batch]):
    """Finite records with deterministic epoch shuffling and resumable index."""

    def __init__(self, config: DataConfig, global_batch_size: int):
        self.config = config
        self.global_batch_size = global_batch_size
        self.records = _prepare_records(config)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        rng = np.random.default_rng(self.config.shuffle_seed + self.index // len(self.records))
        order = rng.permutation(len(self.records))
        examples = [
            self.records[order[(self.index + offset) % len(order)]]
            for offset in range(self.global_batch_size)
        ]
        self.index += self.global_batch_size
        return _stack(examples)

    def get_state(self) -> dict[str, int]:
        return {"index": self.index}

    def set_state(self, state: dict[str, Any]) -> None:
        self.index = int(state["index"])


class GrainIterator(Iterator[Batch]):
    """Grain-backed offline iterator with native iterator state."""

    def __init__(self, config: DataConfig, global_batch_size: int):
        import grain.python as grain

        records = _prepare_records(config)
        dataset = (
            grain.MapDataset.source(records)
            .shuffle(seed=config.shuffle_seed)
            .repeat()
            .batch(
                global_batch_size,
                drop_remainder=True,
                batch_fn=lambda values: _stack(list(values)),
            )
            .to_iter_dataset()
        )
        self.iterator = iter(dataset)

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        return next(self.iterator)

    def get_state(self):
        return self.iterator.get_state()

    def set_state(self, state) -> None:
        self.iterator.set_state(state)


def create_data_iterator(
    config: DataConfig,
    *,
    global_batch_size: int,
    vocab_size: int,
):
    if config.type == "synthetic":
        return SyntheticIterator(config, global_batch_size, vocab_size)
    if config.type == "huggingface":
        return HuggingFaceIterator(config, global_batch_size)
    if config.type == "grain":
        return GrainIterator(config, global_batch_size)
    raise ValueError(f"unsupported data type: {config.type}")
