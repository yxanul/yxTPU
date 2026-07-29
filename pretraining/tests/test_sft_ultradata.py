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

"""UltraData-shape SFT data: role/content records with separate reasoning
render between the single-token think markers, oversize conversations
drop, and whole-conversation packing pads rows instead of splitting a
conversation across row boundaries."""

import numpy as np
import pytest

from yxtpu_pretrain.sft.data import StreamingSFTIterator
from yxtpu_pretrain.sft.tokens import (
    DOCUMENT_SEPARATOR,
    THINK_CLOSE,
    THINK_OPEN,
    load_sft_tokenizer,
    normalize_messages,
    render_conversation,
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_sft_tokenizer(
        "alisawuffles/superbpe-tokenizer-128k", padded_vocab_size=128256
    )


def _ultradata_record(index, *, reasoning="thinking it through", answer=None):
    return {
        "uid": f"u{index}",
        "source": "test",
        "domain": "Math",
        "think_type": "think",
        "messages": [
            {"role": "user", "content": f"question {index}?"},
            {"role": "assistant", "content": answer or f"answer {index}",
             "reasoning_content": reasoning},
        ],
    }


def test_normalize_messages_handles_both_schemas():
    ultra = normalize_messages(_ultradata_record(0))
    assert [m["role"] for m in ultra] == ["user", "assistant"]
    assert ultra[1]["reasoning"] == "thinking it through"
    assert "reasoning" not in ultra[0]

    sharegpt = normalize_messages({"conversations": [
        {"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]})
    assert [m["role"] for m in sharegpt] == ["user", "assistant"]
    assert "reasoning" not in sharegpt[1]

    with pytest.raises(KeyError):
        normalize_messages({"messages": [
            {"role": "user", "content": "q", "reasoning_content": "bad"}]})
    with pytest.raises(KeyError):
        normalize_messages({"other": []})


def test_render_places_reasoning_between_think_markers(tokenizer):
    messages = normalize_messages(_ultradata_record(1))
    ids, mask = render_conversation(tokenizer, messages)
    reasoning_tokens = tokenizer.encode(
        "thinking it through", add_special_tokens=False)
    answer_tokens = tokenizer.encode("answer 1", add_special_tokens=False)
    open_at = ids.index(THINK_OPEN)
    close_at = ids.index(THINK_CLOSE)
    assert ids[open_at + 1 : close_at] == reasoning_tokens
    assert ids[close_at + 1 : close_at + 1 + len(answer_tokens)] == answer_tokens
    # The full think block trains; the user turn does not.
    assert all(mask[open_at : close_at + 1 + len(answer_tokens)])
    user_tokens = tokenizer.encode("question 1?", add_special_tokens=False)
    user_at = _find_subsequence(ids, user_tokens)
    assert not any(mask[user_at : user_at + len(user_tokens)])


def _find_subsequence(haystack, needle):
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    raise AssertionError("subsequence not found")


class _FakeStream:
    def __init__(self, records):
        self._records = records

    def shuffle(self, **_):
        return self

    def __iter__(self):
        return iter(self._records)


def _make_iterator(tokenizer, records, monkeypatch, **kwargs):
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: _FakeStream(records),
        raising=False,
    )
    return StreamingSFTIterator(
        tokenizer, dataset="fake", sequence_length=kwargs.pop("sequence_length", 64),
        process_batch=kwargs.pop("process_batch", 2),
        process_index=0, process_count=1, **kwargs,
    )


def test_pack_whole_rows_hold_only_complete_conversations(tokenizer, monkeypatch):
    records = [_ultradata_record(i) for i in range(12)]
    iterator = _make_iterator(
        tokenizer, records, monkeypatch, pack_whole=True, max_render_tokens=65
    )
    renders = {
        tuple(render_conversation(tokenizer, normalize_messages(r))[0])
        for r in records
    }
    batches = list(iterator)
    assert batches, "expected at least one batch"
    seen_records = 0
    for batch in batches:
        full = np.concatenate(
            [batch["input_ids"], batch["labels"][:, -1:]], axis=1)
        for row_index in range(full.shape[0]):
            row = full[row_index]
            segments = batch["segment_ids"][row_index]
            filled = int(segments.sum())
            # Pad tail: separators, no loss, segment 0.
            assert (row[filled:] == DOCUMENT_SEPARATOR).all()
            assert not batch["loss_mask"][row_index, filled:].any()
            # The filled region is a concatenation of complete renders.
            cursor = 0
            while cursor < filled:
                match = None
                for render in renders:
                    span = len(render)
                    if cursor + span <= len(row) and tuple(
                            row[cursor : cursor + span]) == render:
                        match = span
                        break
                assert match, f"row fragment at {cursor} is not a whole render"
                seen_records += 1
                cursor += match
    assert seen_records == len(records) - iterator.rows_dropped_oversize
    assert iterator.pad_tokens > 0


def test_oversize_conversations_drop_and_short_ones_survive(tokenizer, monkeypatch):
    """Both drop paths fire: the byte pre-filter on records far too long,
    and the exact token check on one that slips under the byte cap."""
    borderline = "reasoning about the problem in some detail, " * 8
    assert len(borderline) < 65 / 0.15  # passes the pre-filter
    assert len(tokenizer.encode(borderline, add_special_tokens=False)) > 65
    records = [
        _ultradata_record(0),
        _ultradata_record(1, reasoning="word " * 300),
        _ultradata_record(2, reasoning=borderline),
        _ultradata_record(3),
    ]
    iterator = _make_iterator(
        tokenizer, records, monkeypatch, pack_whole=True,
        max_render_tokens=65, process_batch=1,
    )
    list(iterator)
    assert iterator.rows_dropped_prefilter == 1
    assert iterator.rows_dropped_oversize == 1
    assert iterator.rows_consumed == 2


def test_byte_prefilter_drops_giants_before_tokenizing(tokenizer, monkeypatch):
    """Records that cannot possibly fit a row are dropped on their byte
    length, never reaching the tokenizer; borderline ones still go through
    the exact token check."""
    from yxtpu_pretrain.sft import data as data_module

    giant = "x" * (8193 * 10)
    records = [
        _ultradata_record(0),
        _ultradata_record(1, reasoning=giant),
        _ultradata_record(2),
    ]
    seen_texts = []
    original = tokenizer.__call__

    def counting_call(texts, **kwargs):
        seen_texts.extend(texts)
        return original(texts, **kwargs)

    monkeypatch.setattr(tokenizer, "__call__", counting_call, raising=False)
    iterator = _make_iterator(
        tokenizer, records, monkeypatch, pack_whole=True,
        max_render_tokens=8193, process_batch=1,
    )
    list(iterator)
    assert iterator.rows_dropped_prefilter == 1
    assert iterator.rows_dropped_oversize == 0
    assert iterator.rows_consumed == 2
    assert not any(giant in text for text in seen_texts)
    # The threshold is a conservative floor: it must never drop a record
    # that would actually have fit.
    assert data_module._MIN_TOKENS_PER_BYTE < 0.238


def test_mixture_shards_files_per_process_and_interleaves(tokenizer, monkeypatch):
    import datasets
    import huggingface_hub

    repo_files = [
        f"data/think/Math/Math_think_part-{i:02d}.jsonl" for i in range(4)
    ] + [
        f"data/think/IF/IF_think_part-{i:02d}.jsonl" for i in range(3)
    ] + ["README.md", "data/no_think/Math/x.jsonl"]
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files",
        lambda dataset, repo_type: list(repo_files), raising=False)

    loaded = []

    def fake_load(builder, data_files, split, streaming):
        assert builder == "json" and split == "train" and streaming
        loaded.append(data_files)
        tag = len(loaded)
        return _FakeStream(
            [_ultradata_record(f"s{tag} r{i}") for i in range(4)])

    def fake_interleave(streams, probabilities, seed, stopping_strategy):
        assert stopping_strategy == "all_exhausted"
        assert probabilities == pytest.approx([0.75, 0.25])
        merged = []
        iterators = [iter(s) for s in streams]
        for _ in range(4):
            for it in iterators:
                merged.append(next(it))
        return _FakeStream(merged)

    monkeypatch.setattr(datasets, "load_dataset", fake_load, raising=False)
    monkeypatch.setattr(
        datasets, "interleave_datasets", fake_interleave, raising=False)
    iterator = StreamingSFTIterator(
        tokenizer, dataset="repo", sequence_length=64, process_batch=1,
        process_index=1, process_count=2,
        mixture=[("Math", 0.75), ("IF", 0.25)], split="think",
        pack_whole=True, max_render_tokens=65, shuffle_seed=11,
    )
    batches = list(iterator)
    # Process 1 of 2 takes the odd-indexed shard files of each config, and
    # no records are stride-filtered on top of the file split.
    assert loaded == [
        ["hf://datasets/repo/data/think/Math/Math_think_part-01.jsonl",
         "hf://datasets/repo/data/think/Math/Math_think_part-03.jsonl"],
        ["hf://datasets/repo/data/think/IF/IF_think_part-01.jsonl"],
    ]
    assert iterator.rows_consumed == 8
    assert iterator.records_drawn == 8
    assert iterator.metadata["sharded_files"] == {"Math": 2, "IF": 1}
    assert batches

    with pytest.raises(ValueError, match="fewer than"):
        StreamingSFTIterator(
            tokenizer, dataset="repo", sequence_length=64, process_batch=1,
            process_index=0, process_count=8,
            mixture=[("IF", 1.0)], split="think",
        )
