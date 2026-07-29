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

"""The generative harness path: chat templating, stop strings, reasoning
strip, and the refusal to sample when the adapter was built for scoring."""

import contextlib
import types

import numpy as np
import pytest

from yxtpu_pretrain.decode import SamplingParams
from yxtpu_pretrain.evaluation.lm_harness import GenerationSettings, JaxHarnessLM
from yxtpu_pretrain.sft.tokens import (
    DOCUMENT_SEPARATOR,
    IM_END,
    IM_MIDDLE,
    ROLE_TOKENS,
    load_sft_tokenizer,
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_sft_tokenizer(
        "alisawuffles/superbpe-tokenizer-128k", padded_vocab_size=128256
    )


def _adapter(tokenizer, generation, replies=None):
    """A JaxHarnessLM with the model and mesh stubbed out.

    Only the request plumbing is under test here; the decoder itself is
    covered against the training forward in test_decode_cache.py.
    """
    adapter = JaxHarnessLM.__new__(JaxHarnessLM)
    adapter.tokenizer = tokenizer
    adapter.generation = generation
    adapter.batch_size = 2
    adapter.max_length = 512
    adapter.mesh = None
    adapter.logical_axis_rules = ()
    adapter.model = None
    adapter._replies = replies or []
    return adapter


def test_scoring_only_adapter_refuses_to_generate(tokenizer):
    adapter = _adapter(tokenizer, None)
    with pytest.raises(NotImplementedError, match="loglikelihood"):
        adapter.generate_until([])


def test_chat_template_wraps_the_prompt(tokenizer):
    settings = GenerationSettings(sampling=SamplingParams(), apply_chat_template=True)
    adapter = _adapter(tokenizer, settings)
    rendered = adapter._render_generation_prompt("hello")
    assert rendered[0] == DOCUMENT_SEPARATOR
    assert rendered[1] == ROLE_TOKENS["user"]
    assert IM_MIDDLE in rendered and IM_END in rendered
    assert rendered[-1] == IM_MIDDLE
    # The assistant header opens after the user turn closes, so the model is
    # asked to continue as the assistant rather than to keep being the user.
    assert rendered.index(ROLE_TOKENS["assistant"]) > rendered.index(IM_END)
    body = tokenizer.encode("hello", add_special_tokens=False)
    assert all(token in rendered for token in body)

    base = GenerationSettings(sampling=SamplingParams(), apply_chat_template=False)
    plain = _adapter(tokenizer, base)._render_generation_prompt("hello")
    assert plain == [DOCUMENT_SEPARATOR, *body]
    assert ROLE_TOKENS["user"] not in plain


def test_reasoning_is_stripped_and_unclosed_traces_score_empty(tokenizer):
    settings = GenerationSettings(sampling=SamplingParams(), strip_reasoning=True)
    adapter = _adapter(tokenizer, settings)
    assert adapter._strip_reasoning("<think>plan</think>\n\nAnswer: 42") == "Answer: 42"
    assert adapter._strip_reasoning("no reasoning here") == "no reasoning here"
    # A runaway trace has no answer; returning it could let a substring
    # check pass on the reasoning text.
    assert adapter._strip_reasoning("<think>looping forever and ever") == ""

    kept = GenerationSettings(sampling=SamplingParams(), strip_reasoning=False)
    assert _adapter(tokenizer, kept)._strip_reasoning("<think>a</think>b") == (
        "<think>a</think>b"
    )


def test_generate_until_cuts_at_stop_strings_and_keeps_request_order(
    tokenizer, monkeypatch
):
    settings = GenerationSettings(
        sampling=SamplingParams(), max_gen_toks=16, strip_reasoning=False)
    adapter = _adapter(tokenizer, settings)

    answers = ["alpha\nBETA", "gamma###tail"]

    def fake_generate(model, prompts, lengths, key, **kwargs):
        del model, key
        width = prompts.shape[0]
        budget = kwargs["max_new_tokens"]
        rows = np.zeros((width, prompts.shape[1] + budget), np.int32)
        for index in range(width):
            body = tokenizer.encode(
                answers[index % len(answers)], add_special_tokens=False)
            start = int(lengths[index]) - 1
            rows[index, start : start + len(body)] = body
            rows[index, start + len(body)] = kwargs["end_token"]
        return rows, np.ones((width,), bool)

    monkeypatch.setattr(
        "yxtpu_pretrain.decode.generate", fake_generate, raising=True)
    monkeypatch.setattr(
        "yxtpu_pretrain.evaluation.lm_harness.logical_mesh_context",
        lambda *a, **k: contextlib.nullcontext(),
        raising=False,
    )

    requests = [
        types.SimpleNamespace(args=("first question", {"until": ["\n"]})),
        types.SimpleNamespace(args=("second question", {"until": ["###"]})),
    ]
    outputs = adapter.generate_until(requests)
    assert outputs == ["alpha", "gamma"]
