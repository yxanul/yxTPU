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

"""Transient streaming-source failures must rebuild the record stream in
place — preserving the consumer's packing state — while sustained outages
and programming errors keep failing loudly."""

import ssl

import pytest

from yxtpu_pretrain.config import DataConfig
from yxtpu_pretrain.runtime import data as data_module
from yxtpu_pretrain.runtime.data import PackedTokenBatcher, _ResilientRecordStream


class _FlakyDataset:
    """Iterable raising a chosen error at one record index, a limited number
    of times, across fresh iterations — the shape of a streaming shard fetch
    that fails and then recovers."""

    def __init__(self, records, *, fail_at, error, failures=1):
        self.records = records
        self.fail_at = fail_at
        self.error = error
        self.failures_left = failures
        self.epochs: list[int] = []
        self.iterations = 0

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)

    def __iter__(self):
        self.iterations += 1
        for index, record in enumerate(self.records):
            if index == self.fail_at and self.failures_left:
                self.failures_left -= 1
                raise self.error
            yield record


@pytest.fixture(autouse=True)
def _fast_recovery(monkeypatch):
    """No real sleeps or Hugging Face session teardown inside unit tests."""
    calls = {"sleeps": [], "resets": 0}
    monkeypatch.setattr(
        data_module.time, "sleep", lambda seconds: calls["sleeps"].append(seconds)
    )

    def _record_reset():
        calls["resets"] += 1

    monkeypatch.setattr(data_module, "_reset_hub_sessions", _record_reset)
    return calls


def test_transient_failure_rebuilds_and_continues(_fast_recovery):
    dataset = _FlakyDataset(
        list(range(6)), fail_at=3, error=ssl.SSLError("tlsv1 alert decode error")
    )
    stream = _ResilientRecordStream(dataset)
    consumed = [next(stream) for _ in range(7)]
    # Records 0-2 arrive, then the rebuilt stream replays from its head.
    assert consumed == [0, 1, 2, 0, 1, 2, 3]
    assert stream.rebuilds == 1
    assert dataset.iterations == 2
    assert dataset.epochs == [1]
    assert _fast_recovery["resets"] == 1
    # The first recovery attempt is immediate; no sleep was needed.
    assert _fast_recovery["sleeps"] == []


def test_retry_budget_exhausts_to_the_last_error(_fast_recovery):
    dataset = _FlakyDataset(
        list(range(4)),
        fail_at=0,
        error=ConnectionError("[Errno 9] Bad file descriptor"),
        failures=10_000,
    )
    stream = _ResilientRecordStream(dataset)
    with pytest.raises(ConnectionError):
        next(stream)
    assert dataset.iterations == 1 + data_module._STREAM_REBUILD_ATTEMPTS
    # Escalating waits after the immediate first attempt.
    assert _fast_recovery["sleeps"] == [
        data_module._STREAM_REBUILD_BACKOFF_SECONDS * n
        for n in range(1, data_module._STREAM_REBUILD_ATTEMPTS)
    ]


def test_non_transient_errors_propagate_immediately(_fast_recovery):
    dataset = _FlakyDataset(
        list(range(4)), fail_at=1, error=ValueError("bad record")
    )
    stream = _ResilientRecordStream(dataset)
    assert next(stream) == 0
    with pytest.raises(ValueError):
        next(stream)
    assert stream.rebuilds == 0
    assert dataset.iterations == 1
    assert _fast_recovery["resets"] == 0


def test_clean_exhaustion_passes_through(_fast_recovery):
    dataset = _FlakyDataset(list(range(3)), fail_at=99, error=OSError())
    stream = _ResilientRecordStream(dataset)
    assert list(stream) == [0, 1, 2]
    assert stream.rebuilds == 0


class _FakeTokenizer:
    eos_token_id = 5
    pad_token_id = 0

    def __len__(self):
        return 8

    def __call__(self, texts, **_):
        return {"input_ids": [[1, 2, 3] for _ in texts]}


def test_batcher_survives_a_mid_batch_stream_failure(_fast_recovery):
    """The rebuild happens underneath the packing buffers: a failure between
    records must not lose the partially accumulated tokenizer batch or kill
    the batch stream."""
    config = DataConfig(
        name="test",
        type="huggingface",
        sequence_length=7,
        tokenize_batch_size=4,
        append_eos=True,
    )
    records = [{"text": f"document {index}"} for index in range(64)]
    dataset = _FlakyDataset(
        records, fail_at=2, error=ssl.SSLError("tlsv1 alert decode error")
    )
    stream = _ResilientRecordStream(dataset)
    batcher = PackedTokenBatcher(
        stream,
        _FakeTokenizer(),
        config,
        global_batch_size=2,
        vocab_size=8,
        validation=False,
    )
    batch = next(batcher)
    assert batch["input_ids"].shape == (2, 7)
    assert stream.rebuilds == 1
    assert batcher.batches_emitted == 1
