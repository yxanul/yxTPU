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

"""Scored sweep over decoding settings for an SFT checkpoint.

Every prompt carries a programmatic checker - exact-match facts, numeric
results, verifiable format constraints, and code executed against tests -
so a setting is judged by what it answers correctly, not by how fluent it
reads. Structural health (one closed think block, stopping on <|im_end|>,
repetition) is reported alongside, since a setting that scores by never
terminating is not usable.

The model loads and compiles once and every setting reuses it; the whole
prompt set decodes in one batch per token.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/sweep_decode_settings.py --checkpoint <dir>/state.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.checkpoints import _persistent_state
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context
from yxtpu_pretrain.sft.tokens import (
    DOCUMENT_SEPARATOR,
    IM_END,
    IM_MIDDLE,
    ROLE_TOKENS,
    THINK_CLOSE,
    THINK_OPEN,
    load_sft_tokenizer,
)

# ---------------------------------------------------------------- checkers


def contains(*needles):
    def check(answer):
        lowered = answer.lower()
        return any(needle.lower() in lowered for needle in needles)
    return check


def numeric(expected, tolerance=1e-6):
    """True when the last number the answer states equals the expected one.

    Reasoning replies restate many intermediate values, so the final number
    is the one being claimed as the answer."""
    def check(answer):
        found = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
        if not found:
            return False
        return abs(float(found[-1]) - expected) <= tolerance
    return check


def single_word(expected):
    def check(answer):
        words = answer.strip().split()
        return len(words) == 1 and words[0].strip(".!\"'").lower() == expected
    return check


def four_numbered_lines(answer):
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if len(lines) != 4:
        return False
    return all(
        re.match(rf"^{index}[.)]?\s+\S+", line)
        for index, line in enumerate(lines, start=1)
    )


def wrapped_in_quotes(answer):
    stripped = answer.strip()
    return len(stripped) > 2 and stripped[0] == '"' and stripped[-1] == '"'


def exactly_two_sentences(answer):
    text = answer.strip().strip('"')
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return len(sentences) == 2


def runs_code(call_and_assert):
    """Executes the reply's code block plus assertions in a subprocess."""
    def check(answer):
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.S)
        source = blocks[0] if blocks else answer
        program = source + "\n" + call_and_assert + "\nprint('PASS')\n"
        try:
            done = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            return False
        return done.returncode == 0 and "PASS" in done.stdout
    return check


PROMPTS = [
    ("knowledge", "What is the capital of Australia? Answer with just the city name.",
     contains("canberra")),
    ("knowledge", "Who wrote the play Romeo and Juliet? Answer with just the name.",
     contains("shakespeare")),
    ("knowledge", "What is the chemical symbol for gold? Answer with just the symbol.",
     contains("au")),
    ("knowledge", "What is the largest planet in our solar system? Answer with just "
                  "the name.", contains("jupiter")),
    ("math", "What is 17 + 25?", numeric(42)),
    ("math", "A train travels 60 km in 1.5 hours. What is its average speed in km/h?",
     numeric(40)),
    ("math", "A shirt costs $40 and is discounted by 25%. What is the final price "
             "in dollars?", numeric(30)),
    ("math", "If 3x + 7 = 22, what is x?", numeric(5)),
    ("instruction", "Respond with a single word only: what colour is the sky on a "
                    "clear day?", single_word("blue")),
    ("instruction", "List exactly four fruits, one per line, numbered 1 to 4. "
                    "Write nothing else.", four_numbered_lines),
    ("instruction", "Name one country in Europe. Wrap your entire answer in double "
                    "quotes.", wrapped_in_quotes),
    ("instruction", "Write exactly two sentences about rain. Write nothing else.",
     exactly_two_sentences),
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
]

SETTINGS = [
    {"name": "greedy", "temperature": 0.0, "top_k": 0, "top_p": 1.0, "penalty": 1.0},
    {"name": "greedy+pen1.15", "temperature": 0.0, "top_k": 0, "top_p": 1.0,
     "penalty": 1.15},
    {"name": "T0.3/p0.9/pen1.1", "temperature": 0.3, "top_k": 0, "top_p": 0.9,
     "penalty": 1.1},
    {"name": "T0.6/k50/pen1.15", "temperature": 0.6, "top_k": 50, "top_p": 1.0,
     "penalty": 1.15},
    {"name": "T0.6/p0.9/pen1.0", "temperature": 0.6, "top_k": 0, "top_p": 0.9,
     "penalty": 1.0},
    {"name": "T0.7/p0.95/pen1.05", "temperature": 0.7, "top_k": 0, "top_p": 0.95,
     "penalty": 1.05},
    {"name": "T0.8/p0.95/pen1.1", "temperature": 0.8, "top_k": 0, "top_p": 0.95,
     "penalty": 1.1},
    {"name": "T1.0/p0.95/pen1.05", "temperature": 1.0, "top_k": 0, "top_p": 0.95,
     "penalty": 1.05},
]


def pick_token(logits, setting, emitted, rng):
    scores = logits.astype(np.float64)
    if setting["penalty"] != 1.0 and emitted:
        seen = np.unique(np.asarray(emitted))
        scores = scores.copy()
        scores[seen] = np.where(
            scores[seen] > 0, scores[seen] / setting["penalty"],
            scores[seen] * setting["penalty"])
    if setting["temperature"] <= 0.0:
        return int(scores.argmax())
    scores = scores / setting["temperature"]
    order = np.argsort(scores)[::-1]
    if setting["top_k"]:
        order = order[: setting["top_k"]]
    probabilities = np.exp(scores[order] - scores[order].max())
    probabilities /= probabilities.sum()
    if setting["top_p"] < 1.0:
        cumulative = np.cumsum(probabilities)
        keep = int(np.searchsorted(cumulative, setting["top_p"]) + 1)
        order, probabilities = order[:keep], probabilities[:keep]
        probabilities /= probabilities.sum()
    return int(rng.choice(order, p=probabilities))


def structure(generated):
    opens, closes = generated.count(THINK_OPEN), generated.count(THINK_CLOSE)
    well_formed = (
        opens == 1 and closes == 1
        and generated.index(THINK_OPEN) < generated.index(THINK_CLOSE)
    )
    if well_formed:
        close_at = generated.index(THINK_CLOSE)
        return True, close_at - generated.index(THINK_OPEN) - 1, generated[close_at + 1:]
    return False, 0, generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--max-new", type=int, default=2400)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--output", default="/tmp/decode_sweep.json")
    arguments = parser.parse_args()

    config = load_config(
        model="kda_hybrid_128k", optimizer="muonclip", data="climbmix_superbpe",
        hardware="v4-64", experiment="superbpe_50b",
        overrides=[
            f"data.sequence_length={arguments.window}",
            "experiment.wandb.enabled=false",
            "experiment.token_budget=null",
            "experiment.checkpoint.enabled=false",
            "experiment.acknowledge_no_checkpoint=true",
            "experiment.harness_eval.enabled=false",
            "experiment.diagnostics.enabled=false",
        ],
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
    tokenizer = load_sft_tokenizer(
        config.data.tokenizer, padded_vocab_size=config.model.vocab_size)
    print(f"loaded {arguments.checkpoint}", flush=True)

    devices = jax.device_count()
    batch = len(PROMPTS)
    if batch % devices:
        raise ValueError(f"{batch} prompts must divide across {devices} devices")
    window = arguments.window
    positions = np.tile(np.arange(window, dtype=np.int32), (batch, 1))

    @nnx.jit
    def next_logits(current_model, tokens, segments, position_ids, cursors):
        hidden = current_model.hidden_states(
            tokens, decoder_segment_ids=segments, decoder_positions=position_ids)
        picked = jnp.take_along_axis(hidden, cursors[:, None, None], axis=1)[:, 0, :]
        kernel = current_model.output_projection_kernel(picked.dtype)
        return (picked @ kernel).astype(jnp.float32)

    prompt_rows = [
        [DOCUMENT_SEPARATOR, ROLE_TOKENS["user"],
         *tokenizer.encode("user", add_special_tokens=False), IM_MIDDLE,
         *tokenizer.encode(text, add_special_tokens=False), IM_END,
         ROLE_TOKENS["assistant"],
         *tokenizer.encode("assistant", add_special_tokens=False), IM_MIDDLE]
        for _, text, _ in PROMPTS
    ]

    records = []
    for setting in SETTINGS:
        for seed in range(arguments.seeds):
            if setting["temperature"] <= 0.0 and seed:
                continue  # greedy is deterministic
            rng = np.random.default_rng(1000 + seed)
            rows = [list(row) for row in prompt_rows]
            lengths = [len(row) for row in rows]
            finished = [False] * batch
            began = time.perf_counter()
            with logical_mesh_context(mesh, rules):
                for _ in range(arguments.max_new):
                    if all(finished):
                        break
                    tokens = np.full((batch, window), DOCUMENT_SEPARATOR, np.int32)
                    segments = np.zeros((batch, window), np.int32)
                    cursors = np.zeros((batch,), np.int32)
                    for index, row in enumerate(rows):
                        length = min(len(row), window)
                        tokens[index, :length] = row[:length]
                        segments[index, :length] = 1
                        cursors[index] = length - 1
                    logits = np.asarray(next_logits(
                        model, jnp.asarray(tokens), jnp.asarray(segments),
                        jnp.asarray(positions), jnp.asarray(cursors)))
                    for index in range(batch):
                        if finished[index]:
                            continue
                        token = pick_token(
                            logits[index], setting, rows[index][lengths[index]:], rng)
                        rows[index].append(token)
                        if token == IM_END or len(rows[index]) >= window:
                            finished[index] = True
            elapsed = time.perf_counter() - began
            for index, (domain, text, checker) in enumerate(PROMPTS):
                generated = rows[index][lengths[index]:]
                well_formed, think_tokens, answer_ids = structure(generated)
                answer = tokenizer.decode([i for i in answer_ids if i != IM_END])
                try:
                    correct = bool(checker(answer))
                except Exception:
                    correct = False
                words = answer.split()
                grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
                records.append({
                    "setting": setting["name"], "seed": seed, "domain": domain,
                    "prompt": text, "answer": answer, "correct": correct,
                    "well_formed": well_formed, "think_tokens": think_tokens,
                    "answer_tokens": len(answer_ids),
                    "stopped": IM_END in generated,
                    "repeated_4grams": len(grams) - len(set(grams)),
                })
            done = [r for r in records if r["setting"] == setting["name"]
                    and r["seed"] == seed]
            print("%-20s seed %d  correct %2d/%d  well-formed %2d/%d  "
                  "stopped %2d/%d  [%.0fs]" % (
                      setting["name"], seed, sum(r["correct"] for r in done), batch,
                      sum(r["well_formed"] for r in done), batch,
                      sum(r["stopped"] for r in done), batch, elapsed), flush=True)

    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    print("\n%-20s %6s %7s %7s %7s %6s %6s %6s %6s" % (
        "setting", "score", "wellfrm", "stopped", "repeat", "know", "math", "instr",
        "code"), flush=True)
    summary = []
    for setting in SETTINGS:
        rows = [r for r in records if r["setting"] == setting["name"]]
        if not rows:
            continue
        by_domain = {}
        for domain in ("knowledge", "math", "instruction", "code"):
            subset = [r for r in rows if r["domain"] == domain]
            by_domain[domain] = sum(r["correct"] for r in subset) / max(len(subset), 1)
        entry = {
            "setting": setting["name"],
            "score": sum(r["correct"] for r in rows) / len(rows),
            "well_formed": sum(r["well_formed"] for r in rows) / len(rows),
            "stopped": sum(r["stopped"] for r in rows) / len(rows),
            "repeats": float(np.mean([r["repeated_4grams"] for r in rows])),
            **by_domain,
        }
        summary.append(entry)
        print("%-20s %6.3f %7.2f %7.2f %7.1f %6.2f %6.2f %6.2f %6.2f" % (
            entry["setting"], entry["score"], entry["well_formed"], entry["stopped"],
            entry["repeats"], entry["knowledge"], entry["math"], entry["instruction"],
            entry["code"]), flush=True)
    best = max(summary, key=lambda e: (e["score"], e["stopped"]))
    print(f"\nbest by score: {best['setting']} ({best['score']:.3f})", flush=True)
    print(f"written {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
