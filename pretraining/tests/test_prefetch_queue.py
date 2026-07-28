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

"""The background host-batch queue must preserve order, propagate errors,
and forward iterator attributes."""

import time

import pytest

from yxtpu_pretrain.train import _PrefetchedIterator


class _FakeIterator:
    def __init__(self, count, fail_at=None):
        self.count = count
        self.fail_at = fail_at
        self.position = 0
        self.metadata = {"source": "fake"}
        self.stats = {"documents_seen": 0.0}

    def __next__(self):
        if self.fail_at is not None and self.position == self.fail_at:
            raise RuntimeError("stream broke")
        if self.position >= self.count:
            raise StopIteration
        value = self.position
        self.position += 1
        self.stats["documents_seen"] = float(self.position)
        return value


def test_order_is_preserved_and_exhaustion_propagates():
    iterator = _PrefetchedIterator(_FakeIterator(100), depth=3)
    values = [next(iterator) for _ in range(100)]
    assert values == list(range(100))
    with pytest.raises(StopIteration):
        next(iterator)


def test_errors_surface_on_the_consuming_thread():
    iterator = _PrefetchedIterator(_FakeIterator(100, fail_at=5), depth=2)
    values = [next(iterator) for _ in range(5)]
    assert values == list(range(5))
    with pytest.raises(RuntimeError, match="stream broke"):
        next(iterator)


def test_attributes_forward_and_queue_depth_reports_readiness():
    inner = _FakeIterator(10)
    iterator = _PrefetchedIterator(inner, depth=3)
    deadline = time.monotonic() + 5.0
    while iterator.queue_depth < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert iterator.queue_depth == 3
    assert iterator.metadata == {"source": "fake"}
    assert iterator.stats["documents_seen"] >= 3.0


def _experiment(**overrides):
    from yxtpu_pretrain.config import CheckpointConfig, ExperimentConfig

    checkpoint = CheckpointConfig(
        enabled=True,
        destination="/tmp/ckpt",
        save_interval=10,
        async_save=False,
        keep=1,
        resume=True,
        allow_weights_only_resume=overrides.pop("allow_weights_only_resume"),
    )
    return ExperimentConfig(
        name="unit",
        benchmark=False,
        checkpoint=checkpoint,
        **overrides,
    )


def test_deep_prefetch_is_rejected_with_exact_iterator_persistence():
    with pytest.raises(Exception, match="prefetch_batches"):
        _experiment(allow_weights_only_resume=False, prefetch_batches=3)


def test_deep_prefetch_is_allowed_for_weights_only_streaming():
    experiment = _experiment(allow_weights_only_resume=True, prefetch_batches=3)
    assert experiment.prefetch_batches == 3
    # Depth 1 stays valid regardless of checkpoint mode.
    assert _experiment(allow_weights_only_resume=False).prefetch_batches == 1
