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

"""Mephisto SFT rendering must reproduce the teacher's own chat format
exactly, and train only the assistant span."""

import numpy as np
import pytest
from transformers import AutoTokenizer

from yxtpu_pretrain.sft.mephisto import (
    IM_END,
    IM_START,
    MephistoIterator,
    render_example,
)

SYSTEM = ("You are a helpful assistant. Answer the user's question "
          "accurately, clearly, and concisely.")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("tokenizers/yx49k", use_fast=True)


def _record(question="What is 2+2?", answer="4 is the sum."):
    return {
        "messages": [{"role": "user", "content": question},
                     {"role": "assistant", "content": answer}],
        "system": SYSTEM,
        "uid": "test_0",
    }


def test_render_matches_the_tokenizers_own_chat_template(tokenizer):
    record = _record()
    ids, mask = render_example(tokenizer, record)
    full = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, *record["messages"]],
        tokenize=False,
    )
    # Byte-identical to what the teacher emits for the same turn.
    assert tokenizer.decode(ids) == full
    assert tokenizer.encode(full, add_special_tokens=False) == ids


def test_only_the_answer_and_its_im_end_are_trained(tokenizer):
    record = _record()
    ids, mask = render_example(tokenizer, record)
    mask = np.asarray(mask)
    trained = [i for i, m in zip(ids, mask) if m]
    assert tokenizer.decode(trained) == "4 is the sum.<|im_end|>\n"
    # Prompt scaffolding, system text and the empty think block stay masked.
    untrained = tokenizer.decode([i for i, m in zip(ids, mask) if not m])
    assert SYSTEM in untrained
    assert "<think>" in untrained and "</think>" in untrained
    assert untrained.endswith("\n\n")
    assert IM_START in ids and IM_END in ids


def test_empty_think_block_is_prompt_not_completion(tokenizer):
    """Qwen3.5 non-thinking puts <think></think> in the generation prompt;
    the student must continue after it, not produce it."""
    ids, mask = render_example(tokenizer, _record())
    first_trained = int(np.argmax(np.asarray(mask) > 0))
    prefix = tokenizer.decode(ids[:first_trained])
    assert prefix.endswith("<think>\n\n</think>\n\n")


def test_explicit_system_overrides_the_row(tokenizer):
    ids, _ = render_example(tokenizer, _record(), system="OVERRIDE PROMPT")
    text = tokenizer.decode(ids)
    assert "OVERRIDE PROMPT" in text and SYSTEM not in text


def test_multi_assistant_rows_are_rejected(tokenizer):
    bad = _record()
    bad["messages"].append({"role": "assistant", "content": "again"})
    with pytest.raises(ValueError):
        render_example(tokenizer, bad)


class _FakeStream:
    """Minimal stand-in for an IterableDataset: take/shuffle/iteration."""

    def __init__(self, rows):
        self._rows = list(rows)

    def take(self, n):
        return _FakeStream(self._rows[:n])

    def shuffle(self, seed=0, buffer_size=None):
        import random

        rows = list(self._rows)
        random.Random(seed).shuffle(rows)
        return _FakeStream(rows)

    def __iter__(self):
        return iter(self._rows)


def _fake_interleave(streams, probabilities, seed, stopping_strategy):
    """Weighted draw that restarts exhausted sources, like all_exhausted."""
    import random

    rng = random.Random(seed)
    pools = [list(s) for s in streams]
    cursors = [0] * len(pools)
    done = [False] * len(pools)
    merged = []
    while not all(done):
        i = rng.choices(range(len(pools)), weights=probabilities)[0]
        if not pools[i]:
            done[i] = True
            continue
        if cursors[i] >= len(pools[i]):
            done[i] = True
            cursors[i] = 0
            if all(done):
                break
        merged.append(pools[i][cursors[i]])
        cursors[i] += 1
    return _FakeStream(merged)


def test_iterator_packs_whole_examples_and_cycles_epochs(tokenizer, monkeypatch):
    import datasets

    rows_a = [_record(f"Q{i}?", f"A{i}") for i in range(6)]
    rows_b = [_record(f"R{i}?", f"B{i}") for i in range(6)]
    calls = {"n": 0}

    def fake_load(spec, split, streaming):
        calls["n"] += 1
        return _FakeStream(rows_a if "IF" in spec else rows_b)

    monkeypatch.setattr(datasets, "load_dataset", fake_load, raising=False)
    monkeypatch.setattr(datasets, "interleave_datasets", _fake_interleave, raising=False)
    iterator = MephistoIterator(
        tokenizer,
        datasets=["Yxanul/Mephisto-IF_172k", "Yxanul/Mephisto-Knowledge_538k"],
        sequence_length=256, process_batch=1,
        process_index=0, process_count=1, epochs=2, system=SYSTEM,
    )
    batches = list(iterator)
    assert batches, "expected at least one packed batch"
    # Two epochs over two sources. Weighted interleaving with
    # all_exhausted oversamples the smaller source rather than draining
    # it, so the count is bounded, not an identity.
    assert iterator.rows_consumed >= 2 * max(len(rows_a), len(rows_b))
    assert iterator.epochs_started == 2  # one open per epoch, all sources
    batch = batches[0]
    assert batch["input_ids"].shape == (1, 256)
    # Pad tail masked out of both loss and attention.
    segments = batch["segment_ids"][0]
    filled = int(segments.sum())
    if filled < 256:
        assert not batch["loss_mask"][0, filled:].any()


def test_sources_are_mixed_throughout_not_drained_in_sequence(tokenizer, monkeypatch):
    """The gen-1 failure: round-robin left the back half of every epoch
    single-source, and the model collapsed onto that format. Weighted
    interleaving must keep both sources present across the whole epoch."""
    import datasets

    small = [dict(_record(f"IF{i}?", f"a{i}"), uid=f"IF_{i}") for i in range(50)]
    large = [dict(_record(f"KN{i}?", f"b{i}"), uid=f"KN_{i}") for i in range(400)]

    def fake_load(spec, split, streaming):
        rows = small if "IF" in spec else large
        return _FakeStream(list(rows))

    monkeypatch.setattr(datasets, "load_dataset", fake_load, raising=False)
    monkeypatch.setattr(datasets, "interleave_datasets", _fake_interleave, raising=False)
    seen = []
    real_render = None

    it = MephistoIterator(
        tokenizer,
        datasets=["Yxanul/Mephisto-IF_172k", "Yxanul/Mephisto-Knowledge_538k:400"],
        sequence_length=128, process_batch=1, process_index=0, process_count=1,
        epochs=1, system=SYSTEM, shuffle_buffer=32, seed=3,
    )
    # Capture the source of every drawn row in order.
    original_draw = it._draw

    def tracking_draw():
        row = original_draw()
        if row is not None:
            seen.append(row["uid"].split("_")[0])
        return row

    it._draw = tracking_draw
    list(it)
    assert len(seen) > 200
    half = len(seen) // 2
    first_if = seen[:half].count("IF")
    second_if = seen[half:].count("IF")
    # Both halves must contain the small source; round-robin scored 0 here.
    assert first_if > 0 and second_if > 0, (first_if, second_if)


def test_row_limits_are_parsed_and_applied(tokenizer):
    it = MephistoIterator(
        tokenizer, datasets=["repo/a", "repo/b:1234"],
        sequence_length=64, process_batch=1, process_index=0, process_count=1,
    )
    assert it._specs == [("repo/a", None), ("repo/b", 1234)]
