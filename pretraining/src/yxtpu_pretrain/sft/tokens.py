"""SFT chat special tokens, template, and conversation rendering.

Mirrors Kimi K2.5's conventions (role headers, <|im_middle|>, <|im_end|>,
single-token <think>/</think>, and the tool-call marker set) mapped onto
the SuperBPE tokenizer's free padded ids. The base vocabulary ends at
<|endoftext|> = 128000; adding specials in this exact order lands them on
the asserted ids, well inside the padded 128256 the model already has.
"""

from __future__ import annotations

from yxtpu_pretrain.runtime.data import load_fast_tokenizer

SPECIAL_TOKENS = (
    ("<|im_system|>", 128001),
    ("<|im_user|>", 128002),
    ("<|im_assistant|>", 128003),
    ("<|im_middle|>", 128004),
    ("<|im_end|>", 128005),
    ("<think>", 128006),
    ("</think>", 128007),
    ("<|tool_calls_section_begin|>", 128008),
    ("<|tool_calls_section_end|>", 128009),
    ("<|tool_call_begin|>", 128010),
    ("<|tool_call_argument_begin|>", 128011),
    ("<|tool_call_end|>", 128012),
)
ROLE_TOKENS = {"system": 128001, "user": 128002, "assistant": 128003}
IM_MIDDLE, IM_END = 128004, 128005
DOCUMENT_SEPARATOR = 128000  # <|endoftext|>, matches pretraining packing

CHAT_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'system' -%}<|im_system|>system<|im_middle|>"
    "{%- elif m['role'] == 'user' -%}<|im_user|>user<|im_middle|>"
    "{%- else -%}<|im_assistant|>assistant<|im_middle|>{%- endif -%}"
    "{{ m['content'] }}<|im_end|>"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|im_assistant|>assistant<|im_middle|>{%- endif -%}"
)


def load_sft_tokenizer(name: str, *, padded_vocab_size: int):
    """Loads the pretraining tokenizer and appends the chat specials at
    their exact reserved ids."""
    tokenizer = load_fast_tokenizer(name, padded_vocab_size=padded_vocab_size)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [text for text, _ in SPECIAL_TOKENS]}
    )
    for text, expected in SPECIAL_TOKENS:
        actual = tokenizer.convert_tokens_to_ids(text)
        if actual != expected:
            raise ValueError(f"{text} landed on id {actual}, expected {expected}")
    if len(tokenizer) > padded_vocab_size:
        raise ValueError("special tokens exceed the padded vocabulary")
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def render_conversation(tokenizer, messages) -> tuple[list[int], list[int]]:
    """Token ids plus a per-token loss mask (assistant content and its
    <|im_end|> train; headers, system, and user turns do not). Starts with
    the document separator so packing matches the pretraining convention."""
    ids = [DOCUMENT_SEPARATOR]
    mask = [0]
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    for message in messages:
        role = message["role"]
        header = [ROLE_TOKENS[role], *encode(role), IM_MIDDLE]
        body = encode(message["content"])
        trainable = 1 if role == "assistant" else 0
        ids.extend(header)
        mask.extend([0] * len(header))
        ids.extend(body)
        mask.extend([trainable] * len(body))
        ids.append(IM_END)
        mask.append(trainable)
    return ids, mask
