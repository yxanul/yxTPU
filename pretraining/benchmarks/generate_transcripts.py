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

"""Transcript set for an SFT checkpoint at a chosen decoding config.

Runs a fixed 32-prompt panel through the cached decoder, keeps the think
trace and the answer for every prompt, scores the ones with a
programmatic checker, and writes both a JSON record and a readable
markdown transcript for analysis.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/generate_transcripts.py --checkpoint <dir>/state.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_decode_settings import (  # noqa: E402
    contains,
    four_numbered_lines,
    numeric,
    runs_code,
    single_word,
    wrapped_in_quotes,
)

from yxtpu_pretrain.config import load_config  # noqa: E402
from yxtpu_pretrain.decode import SamplingParams, generate  # noqa: E402
from yxtpu_pretrain.model import HybridLanguageModel  # noqa: E402
from yxtpu_pretrain.optimizers import build_optimizer  # noqa: E402
from yxtpu_pretrain.runtime.checkpoints import _persistent_state  # noqa: E402
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config  # noqa: E402
from yxtpu_pretrain.runtime.mesh import create_mesh  # noqa: E402
from yxtpu_pretrain.runtime.sharding import logical_mesh_context  # noqa: E402
from yxtpu_pretrain.sft.tokens import (  # noqa: E402
    DOCUMENT_SEPARATOR,
    IM_END,
    IM_MIDDLE,
    ROLE_TOKENS,
    THINK_CLOSE,
    THINK_OPEN,
    load_sft_tokenizer,
)


def exactly_two_sentences(answer):
    text = answer.strip().strip('"')
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return len(sentences) == 2


def three_comma_items(answer):
    parts = [p.strip() for p in answer.strip().rstrip(".").split(",")]
    return len(parts) == 3 and all(p and " " not in p for p in parts)


def all_uppercase(answer):
    letters = [c for c in answer if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def avoids_letter_e(answer):
    body = answer.strip().strip('"')
    return bool(body) and "e" not in body.lower()


PROMPTS = [
    # ---- knowledge ----
    ("knowledge", "What is the capital of Australia? Answer with just the city name.",
     contains("canberra")),
    ("knowledge", "Who wrote the play Romeo and Juliet? Answer with just the name.",
     contains("shakespeare")),
    ("knowledge", "What is the chemical symbol for gold? Answer with just the symbol.",
     contains("au")),
    ("knowledge", "What is the largest planet in our solar system?",
     contains("jupiter")),
    ("knowledge", "In what year did the Second World War end?", numeric(1945)),
    ("knowledge", "What is the boiling point of water in Celsius at sea level?",
     numeric(100)),
    ("knowledge", "Who painted the Mona Lisa?", contains("vinci", "leonardo")),
    ("knowledge", "What is the capital of Japan? Answer with just the city name.",
     contains("tokyo")),
    # ---- math ----
    ("math", "What is 17 + 25?", numeric(42)),
    ("math", "A train travels 60 km in 1.5 hours. What is its average speed in km/h?",
     numeric(40)),
    ("math", "A shirt costs $40 and is discounted by 25%. What is the final price "
             "in dollars?", numeric(30)),
    ("math", "If 3x + 7 = 22, what is x?", numeric(5)),
    ("math", "What is 12 multiplied by 12?", numeric(144)),
    ("math", "What is half of 96?", numeric(48)),
    ("math", "A rectangle is 7 cm by 5 cm. What is its area in square centimetres?",
     numeric(35)),
    ("math", "What is 100 minus 37?", numeric(63)),
    # ---- instruction following ----
    ("instruction", "Respond with a single word only: what colour is the sky on a "
                    "clear day?", single_word("blue")),
    ("instruction", "List exactly four fruits, one per line, numbered 1 to 4. "
                    "Write nothing else.", four_numbered_lines),
    ("instruction", "Name one country in Europe. Wrap your entire answer in double "
                    "quotes.", wrapped_in_quotes),
    ("instruction", "Write exactly two sentences about rain. Write nothing else.",
     exactly_two_sentences),
    ("instruction", "Answer with only the word yes or the word no: is the Earth "
                    "round?", single_word("yes")),
    ("instruction", "List three colours separated by commas. Write nothing else.",
     three_comma_items),
    ("instruction", "Reply in all uppercase letters: say hello to the reader.",
     all_uppercase),
    ("instruction", "Write one short sentence that does not contain the letter e.",
     avoids_letter_e),
    # ---- code ----
    ("code", "Write a Python function named reverse_string(s) that returns the "
             "reversed string. Reply with only a Python code block.",
     runs_code("assert reverse_string('hello') == 'olleh'\n"
               "assert reverse_string('') == ''")),
    ("code", "Write a Python function named fib(n) that returns the nth Fibonacci "
             "number, where fib(1)=1 and fib(2)=1. Reply with only a Python code "
             "block.",
     runs_code("assert fib(1) == 1 and fib(2) == 1 and fib(7) == 13")),
    ("code", "Write a Python function named is_palindrome(s) that returns True if "
             "s is a palindrome ignoring case and spaces. Reply with only a Python "
             "code block.",
     runs_code("assert is_palindrome('A man a plan a canal Panama')\n"
               "assert not is_palindrome('hello world')")),
    ("code", "Write a Python function named sum_even(numbers) that returns the sum "
             "of the even numbers in a list. Reply with only a Python code block.",
     runs_code("assert sum_even([1,2,3,4]) == 6\nassert sum_even([]) == 0")),
    ("code", "Write a Python function named count_vowels(s) that returns how many "
             "vowels are in the string. Reply with only a Python code block.",
     runs_code("assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0")),
    ("code", "Write a Python function named max_of_list(numbers) that returns the "
             "largest number in a list. Reply with only a Python code block.",
     runs_code("assert max_of_list([3,9,2]) == 9\nassert max_of_list([-1,-5]) == -1")),
    ("code", "Write a Python function named celsius_to_fahrenheit(c) that converts "
             "Celsius to Fahrenheit. Reply with only a Python code block.",
     runs_code("assert celsius_to_fahrenheit(0) == 32\n"
               "assert abs(celsius_to_fahrenheit(100) - 212) < 1e-6")),
    ("code", "Write a Python function named is_prime(n) that returns True if n is "
             "prime. Reply with only a Python code block.",
     runs_code("assert is_prime(7) and is_prime(2)\n"
               "assert not is_prime(1) and not is_prime(9)")),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="kda_hybrid_128k")
    parser.add_argument("--data", default="climbmix_superbpe")
    parser.add_argument("--chat-scheme", choices=("k25", "qwen"), default="k25")
    parser.add_argument("--system", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--json-output", default="/tmp/transcripts.json")
    parser.add_argument("--markdown-output", default="/tmp/transcripts.md")
    parser.add_argument("--experiment", default="superbpe_50b")
    parser.add_argument("--set", action="append", dest="overrides", default=[])
    arguments = parser.parse_args()

    window = 8192
    config = load_config(
        model=arguments.model, optimizer="muonclip", data=arguments.data,
        hardware="v4-64", experiment=arguments.experiment,
        overrides=[
            f"data.sequence_length={window}",
            "experiment.wandb.enabled=false",
            "experiment.token_budget=null",
            "experiment.checkpoint.enabled=false",
            "experiment.acknowledge_no_checkpoint=true",
            "experiment.harness_eval.enabled=false",
            "experiment.diagnostics.enabled=false",
        ] + list(arguments.overrides or []),
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))
    target = _persistent_state(state)
    with open(arguments.checkpoint, "rb") as handle:
        nnx.replace_by_pure_dict(target, pickle.load(handle))
    nnx.update(state, target)
    if arguments.chat_scheme == "qwen":
        # yx49k carries the Qwen3.5 chat template and its specials natively.
        from yxtpu_pretrain.runtime.data import load_fast_tokenizer

        tokenizer = load_fast_tokenizer(
            config.data.tokenizer, padded_vocab_size=config.model.vocab_size)
        stop_token, pad_token = 49121, 49119
    else:
        tokenizer = load_sft_tokenizer(
            config.data.tokenizer, padded_vocab_size=config.model.vocab_size)
        stop_token, pad_token = IM_END, DOCUMENT_SEPARATOR
    print(f"loaded {arguments.checkpoint}", flush=True)

    if arguments.chat_scheme == "qwen":
        def _render(text):
            messages = []
            if arguments.system:
                messages.append({"role": "system", "content": arguments.system})
            messages.append({"role": "user", "content": text})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            return tokenizer.encode(prompt, add_special_tokens=False)
    else:
        def _render(text):
            return [DOCUMENT_SEPARATOR, ROLE_TOKENS["user"],
                    *tokenizer.encode("user", add_special_tokens=False), IM_MIDDLE,
                    *tokenizer.encode(text, add_special_tokens=False), IM_END,
                    ROLE_TOKENS["assistant"],
                    *tokenizer.encode("assistant", add_special_tokens=False),
                    IM_MIDDLE]
    rendered = [_render(text) for _, text, _ in PROMPTS]
    lengths = np.asarray([len(row) for row in rendered], np.int32)
    width = int(lengths.max())
    padded = np.full((len(rendered), width), pad_token, np.int32)
    for index, row in enumerate(rendered):
        padded[index, : len(row)] = row

    sampling = SamplingParams(
        temperature=arguments.temperature, top_k=arguments.top_k,
        top_p=arguments.top_p, repetition_penalty=arguments.repetition_penalty)
    began = time.perf_counter()
    with logical_mesh_context(mesh, rules):
        samples, _ = generate(
            model, jnp.asarray(padded), jnp.asarray(lengths),
            jax.random.key(arguments.seed), max_new_tokens=arguments.max_new,
            sampling=sampling, end_token=stop_token,
            max_length=width + arguments.max_new + 8)
        samples = np.asarray(samples)
    elapsed = time.perf_counter() - began
    print(f"generated {len(PROMPTS)} transcripts in {elapsed:.1f}s", flush=True)

    setting = {
        "temperature": arguments.temperature, "top_p": arguments.top_p,
        "top_k": arguments.top_k, "repetition_penalty": arguments.repetition_penalty,
        "seed": arguments.seed, "max_new_tokens": arguments.max_new,
    }
    records = []
    for index, (domain, text, checker) in enumerate(PROMPTS):
        row = list(samples[index, int(lengths[index]) - 1:])
        stop = [i for i, token in enumerate(row) if token == stop_token]
        generated = row[: stop[0]] if stop else row
        opens = generated.count(THINK_OPEN)
        closes = generated.count(THINK_CLOSE)
        well_formed = opens == 1 and closes == 1 and (
            generated.index(THINK_OPEN) < generated.index(THINK_CLOSE))
        if well_formed:
            close_at = generated.index(THINK_CLOSE)
            think = tokenizer.decode(generated[generated.index(THINK_OPEN) + 1:close_at])
            answer_ids = generated[close_at + 1:]
        else:
            think, answer_ids = "", generated
        answer = tokenizer.decode(answer_ids)
        try:
            correct = bool(checker(answer))
        except Exception:
            correct = False
        words = answer.split()
        grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
        records.append({
            "index": index, "domain": domain, "prompt": text,
            "think": think, "answer": answer, "correct": correct,
            "well_formed_think": well_formed,
            "think_tokens": len(tokenizer.encode(think, add_special_tokens=False))
            if think else 0,
            "answer_tokens": len(answer_ids),
            "stopped_on_im_end": bool(stop),
            "repeated_4grams": len(grams) - len(set(grams)),
        })

    domains = sorted({record["domain"] for record in records})
    summary = {
        "setting": setting,
        "checkpoint": arguments.checkpoint,
        "generation_seconds": round(elapsed, 1),
        "total": len(records),
        "correct": sum(r["correct"] for r in records),
        "well_formed_think": sum(r["well_formed_think"] for r in records),
        "stopped_on_im_end": sum(r["stopped_on_im_end"] for r in records),
        "by_domain": {
            domain: {
                "correct": sum(
                    r["correct"] for r in records if r["domain"] == domain),
                "total": sum(1 for r in records if r["domain"] == domain),
            }
            for domain in domains
        },
    }
    print(json.dumps(summary, indent=2), flush=True)

    with open(arguments.json_output, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "transcripts": records}, handle, indent=2)

    lines = [
        f"# SFT transcripts - {Path(arguments.checkpoint).parent.name}",
        "",
        f"Decoding: temperature {setting['temperature']}, top-p {setting['top_p']}, "
        f"top-k {setting['top_k']}, repetition penalty "
        f"{setting['repetition_penalty']}, seed {setting['seed']}.",
        "",
        f"Scored {summary['correct']}/{summary['total']} correct; "
        f"{summary['well_formed_think']}/{summary['total']} well-formed think "
        f"blocks; {summary['stopped_on_im_end']}/{summary['total']} stopped on "
        f"`<|im_end|>`. Generated in {summary['generation_seconds']}s.",
        "",
    ]
    for domain in domains:
        scored = summary["by_domain"][domain]
        lines += [f"## {domain} ({scored['correct']}/{scored['total']})", ""]
        for record in [r for r in records if r["domain"] == domain]:
            mark = "PASS" if record["correct"] else "FAIL"
            lines += [
                f"### [{mark}] {record['prompt']}",
                "",
                f"*think ({record['think_tokens']} tokens):*",
                "",
                "```", record["think"].strip() or "(none)", "```",
                "",
                "*answer:*",
                "",
                "```", record["answer"].strip() or "(empty)", "```",
                "",
            ]
    Path(arguments.markdown_output).write_text("\n".join(lines), encoding="utf-8")
    print(f"written {arguments.json_output} and {arguments.markdown_output}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
