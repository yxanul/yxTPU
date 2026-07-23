import pytest

from yxtpu_pretrain.sft.tokens import (
    SPECIAL_TOKENS,
    load_sft_tokenizer,
    render_conversation,
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_sft_tokenizer(
        "alisawuffles/superbpe-tokenizer-128k", padded_vocab_size=128256
    )


def test_special_tokens_land_on_reserved_ids(tokenizer):
    for text, expected in SPECIAL_TOKENS:
        assert tokenizer.convert_tokens_to_ids(text) == expected
        assert tokenizer.encode(text, add_special_tokens=False) == [expected]


def test_think_markers_collapse_inside_running_text(tokenizer):
    ids = tokenizer.encode(
        "<think>\nplan things\n</think>\n\nAnswer.", add_special_tokens=False
    )
    assert ids[0] == 128006
    assert 128007 in ids
    assert ids.count(128006) == 1 and ids.count(128007) == 1


def test_render_masks_exactly_the_assistant_span(tokenizer):
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\n2+2=4\n</think>\n\n4"},
    ]
    ids, mask = render_conversation(tokenizer, messages)
    assert len(ids) == len(mask)
    assert ids[0] == 128000 and mask[0] == 0
    first_end = ids.index(128005)
    assert all(m == 0 for m in mask[: first_end + 1])
    assistant_middle = ids.index(128004, ids.index(128003))
    assert all(m == 1 for m in mask[assistant_middle + 1 :])
    assert ids[-1] == 128005 and mask[-1] == 1
    assert 128006 in ids and 128007 in ids
    decoded = tokenizer.decode(ids[assistant_middle + 1 : -1])
    assert "4" in decoded and "<think>" in decoded
