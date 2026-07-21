"""Synthetic, offline, and streaming pretraining iterators."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from yxtpu_pretrain.config import DataConfig

Batch = dict[str, np.ndarray]


def _example(
    tokens: np.ndarray,
    sequence_length: int,
    *,
    pad_token_id: int = 0,
) -> dict[str, np.ndarray]:
    """Builds one causal example without assuming token id zero is padding."""
    tokens = np.asarray(tokens, dtype=np.int32)
    required = sequence_length + 1
    valid_targets = min(max(tokens.size - 1, 0), sequence_length)
    if tokens.size < required:
        padded = np.full(required, pad_token_id, dtype=np.int32)
        padded[: tokens.size] = tokens
        tokens = padded
    else:
        tokens = tokens[:required]
    valid = np.arange(sequence_length) < valid_targets
    return {
        "input_ids": tokens[:-1],
        "labels": tokens[1:],
        "loss_mask": valid.astype(np.float32),
        "segment_ids": valid.astype(np.int32),
        "positions": np.arange(sequence_length, dtype=np.int32),
    }


def _packed_batch(tokens: np.ndarray, sequence_length: int) -> Batch:
    rows = np.asarray(tokens, dtype=np.int32).reshape((-1, sequence_length + 1))
    inputs = rows[:, :-1]
    return {
        "input_ids": inputs,
        "labels": rows[:, 1:],
        "loss_mask": np.ones(inputs.shape, dtype=np.float32),
        "segment_ids": np.ones(inputs.shape, dtype=np.int32),
        "positions": np.broadcast_to(
            np.arange(sequence_length, dtype=np.int32), inputs.shape
        ).copy(),
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
        seed_offset = 0 if self.config.reuse_example_batch else self.index
        rng = np.random.default_rng(self.config.shuffle_seed + seed_offset)
        tokens = rng.integers(
            1,
            self.vocab_size,
            size=(self.global_batch_size, self.config.sequence_length + 1),
            dtype=np.int32,
        )
        self.index += 1
        return _packed_batch(tokens, self.config.sequence_length)

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


def load_fast_tokenizer(name: str, *, padded_vocab_size: int):
    """Loads the Rust-backed tokenizer and validates the padded model vocabulary."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(f"tokenizer {name!r} did not resolve to a Rust fast tokenizer")
    if tokenizer.eos_token_id is None:
        raise ValueError(f"tokenizer {name!r} has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) > padded_vocab_size:
        raise ValueError(
            f"tokenizer vocabulary {len(tokenizer)} exceeds model vocabulary {padded_vocab_size}"
        )
    return tokenizer


def _prepare_records(config: DataConfig, *, vocab_size: int) -> list[dict[str, np.ndarray]]:
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
            text = record.get(config.text_field)
            if text is None:
                raise ValueError(
                    f"records must contain input_ids or text field {config.text_field!r}"
                )
            if config.tokenizer is None:
                raise ValueError("text records require data.tokenizer")
            if tokenizer is None:
                tokenizer = load_fast_tokenizer(
                    config.tokenizer,
                    padded_vocab_size=vocab_size,
                )
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if config.append_eos:
                tokens = [*tokens, tokenizer.eos_token_id]
        pad_token_id = tokenizer.pad_token_id if tokenizer is not None else 0
        prepared.append(
            _example(
                np.asarray(tokens, dtype=np.int32),
                config.sequence_length,
                pad_token_id=pad_token_id,
            )
        )
    return prepared


class HuggingFaceIterator(Iterator[Batch]):
    """Finite records with deterministic epoch shuffling and resumable index."""

    def __init__(self, config: DataConfig, global_batch_size: int, vocab_size: int):
        self.config = config
        self.global_batch_size = global_batch_size
        self.records = _prepare_records(config, vocab_size=vocab_size)
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


def _is_validation_record(text: str, *, fraction: float, seed: int) -> bool:
    """Stable content-hash assignment shared by train and validation iterators."""
    if fraction <= 0.0:
        return False
    person = int(seed).to_bytes(8, "little", signed=False)
    digest = hashlib.blake2b(
        text.encode("utf-8", errors="replace"),
        digest_size=8,
        person=person,
    ).digest()
    bucket = int.from_bytes(digest, "big") / float(1 << 64)
    return bucket < fraction


class PackedTokenBatcher(Iterator[Batch]):
    """Batches a text stream through one batched Rust-tokenizer call at a time."""

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        tokenizer,
        config: DataConfig,
        *,
        global_batch_size: int,
        vocab_size: int,
        validation: bool,
    ):
        self.records = iter(records)
        self.tokenizer = tokenizer
        self.config = config
        self.global_batch_size = global_batch_size
        self.vocab_size = vocab_size
        self.validation = validation
        self._chunks: list[np.ndarray] = []
        self._available = 0
        self.documents_seen = 0
        self.documents_selected = 0

    def __iter__(self):
        return self

    def _fill(self, required: int) -> None:
        while self._available < required:
            texts: list[str] = []
            while len(texts) < self.config.tokenize_batch_size:
                record = next(self.records)
                self.documents_seen += 1
                text = record.get(self.config.text_field)
                if not isinstance(text, str):
                    raise ValueError(
                        f"streaming record lacks string field {self.config.text_field!r}"
                    )
                assigned_to_validation = _is_validation_record(
                    text,
                    fraction=self.config.validation_fraction,
                    seed=self.config.validation_seed,
                )
                if assigned_to_validation != self.validation:
                    continue
                texts.append(text)
                self.documents_selected += 1
            encoded = self.tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            for token_ids in encoded:
                if self.config.append_eos:
                    token_ids = [*token_ids, self.tokenizer.eos_token_id]
                if not token_ids:
                    continue
                chunk = np.asarray(token_ids, dtype=np.int32)
                if int(chunk.max(initial=0)) >= self.vocab_size:
                    raise ValueError("token id exceeds the padded model vocabulary")
                self._chunks.append(chunk)
                self._available += chunk.size

    def __next__(self) -> Batch:
        required = self.global_batch_size * (self.config.sequence_length + 1)
        self._fill(required)
        joined = np.concatenate(self._chunks)
        selected = joined[:required]
        remainder = joined[required:]
        self._chunks = [remainder] if remainder.size else []
        self._available = int(remainder.size)
        return _packed_batch(selected, self.config.sequence_length)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.config.tokenizer,
            "tokenizer_backend": "rust_fast",
            "tokenizer_vocab_size": len(self.tokenizer),
            "padded_vocab_size": self.vocab_size,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streaming": True,
            "validation_partition": self.validation,
            "validation_fraction": self.config.validation_fraction,
        }

    def get_state(self):
        raise RuntimeError("streaming packed data is not checkpointable in this profile")

    def set_state(self, state) -> None:
        del state
        raise RuntimeError("streaming packed data is not checkpointable in this profile")


class StreamingHuggingFaceIterator(PackedTokenBatcher):
    """Streaming Hugging Face source with deterministic train/validation reservation."""

    def __init__(
        self,
        config: DataConfig,
        global_batch_size: int,
        vocab_size: int,
        *,
        validation: bool,
    ):
        from datasets import load_dataset

        if config.tokenizer is None:
            raise ValueError("streaming text data requires data.tokenizer")
        dataset = load_dataset(
            config.dataset_name,
            split=config.split,
            streaming=True,
        )
        dataset = dataset.shuffle(
            seed=config.shuffle_seed,
            buffer_size=config.shuffle_buffer_size,
        )
        tokenizer = load_fast_tokenizer(config.tokenizer, padded_vocab_size=vocab_size)
        super().__init__(
            dataset,
            tokenizer,
            config,
            global_batch_size=global_batch_size,
            vocab_size=vocab_size,
            validation=validation,
        )


@dataclass(frozen=True)
class _WorkerFailure:
    error: BaseException


_END = object()


class PrefetchIterator(Iterator[Batch]):
    """Runs source reads and Rust tokenization in one bounded background thread."""

    def __init__(self, source: Iterator[Batch], depth: int):
        self.source = source
        self.depth = depth
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            while True:
                self._queue.put(next(self.source))
        except StopIteration:
            self._queue.put(_END)
        except BaseException as error:
            self._queue.put(_WorkerFailure(error))

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        value = self._queue.get()
        if value is _END:
            raise StopIteration
        if isinstance(value, _WorkerFailure):
            raise value.error
        return value

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(getattr(self.source, "metadata", {}))
        metadata["prefetch_batches"] = self.depth
        return metadata

    def get_state(self):
        return self.source.get_state()

    def set_state(self, state) -> None:
        self.source.set_state(state)


class GrainIterator(Iterator[Batch]):
    """Grain-backed offline iterator with native iterator state."""

    def __init__(self, config: DataConfig, global_batch_size: int, vocab_size: int):
        import grain.python as grain

        records = _prepare_records(config, vocab_size=vocab_size)
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
    validation: bool = False,
):
    if config.type == "synthetic":
        source = SyntheticIterator(config, global_batch_size, vocab_size)
    elif config.type == "huggingface" and config.streaming:
        source = StreamingHuggingFaceIterator(
            config,
            global_batch_size,
            vocab_size,
            validation=validation,
        )
    elif config.type == "huggingface":
        source = HuggingFaceIterator(config, global_batch_size, vocab_size)
    elif config.type == "grain":
        source = GrainIterator(config, global_batch_size, vocab_size)
    else:
        raise ValueError(f"unsupported data type: {config.type}")
    if config.prefetch_batches:
        return PrefetchIterator(source, config.prefetch_batches)
    return source
