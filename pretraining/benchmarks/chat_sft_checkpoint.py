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

"""Chat with an SFT checkpoint: chat template in, sampled decode out.

Decoding is the recompute loop (no KDA incremental state yet), so cost per
token scales with the window; prompts are decoded in groups the size of the
mesh's data axis so one forward serves every row of the group instead of
replicating a single prompt. Alongside the text this reports the structural
metrics that matter more than content at 337M: whether the reply opens and
closes exactly one think block, whether it stops on <|im_end|> instead of
running to the budget, and how the think length compares to the answer.

Single worker:
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/chat_sft_checkpoint.py --checkpoint <dir>/state.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
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

PROMPTS = [
    ("knowledge", "What is the capital of Australia?"),
    ("knowledge", "Who wrote the play Romeo and Juliet?"),
    ("knowledge", "What causes the seasons on Earth? Answer briefly."),
    ("instruction", "List exactly four fruits, one per line, numbered 1 to 4. "
                    "Do not write anything else."),
    ("instruction", "Describe a rainstorm in exactly two paragraphs, separated "
                    "by two line breaks. Wrap your entire answer in double quotes."),
    ("instruction", "Respond with a single word only: what colour is the sky on "
                    "a clear day?"),
    ("math", "What is 17 + 25?"),
    ("math", "A train travels 60 km in 1.5 hours. What is its average speed in km/h?"),
    ("math", "A shirt costs $40 and is discounted by 25%. What is the final price?"),
    ("code", "Write a Python function that reverses a string."),
    ("code", "Write a Python function that returns the nth Fibonacci number."),
    ("code", "Write a Python function that checks whether a string is a "
             "palindrome, ignoring case and spaces."),
]


def render_prompt(tokenizer, text):
    return [
        DOCUMENT_SEPARATOR,
        ROLE_TOKENS["user"], *tokenizer.encode("user", add_special_tokens=False),
        IM_MIDDLE, *tokenizer.encode(text, add_special_tokens=False), IM_END,
        ROLE_TOKENS["assistant"],
        *tokenizer.encode("assistant", add_special_tokens=False), IM_MIDDLE,
    ]


def structure_report(tokenizer, generated_ids, *, hit_budget):
    opens = generated_ids.count(THINK_OPEN)
    closes = generated_ids.count(THINK_CLOSE)
    well_formed = (
        opens == 1 and closes == 1
        and generated_ids.index(THINK_OPEN) < generated_ids.index(THINK_CLOSE)
    ) if opens and closes else False
    if well_formed:
        close_at = generated_ids.index(THINK_CLOSE)
        think_tokens = close_at - generated_ids.index(THINK_OPEN) - 1
        answer_ids = generated_ids[close_at + 1:]
    else:
        think_tokens = 0
        answer_ids = generated_ids
    answer_ids = [i for i in answer_ids if i != IM_END]
    answer = tokenizer.decode(answer_ids)
    words = answer.split()
    repeated = 0
    if len(words) > 8:
        grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
        repeated = len(grams) - len(set(grams))
    return {
        "think_open": opens,
        "think_close": closes,
        "well_formed_think": well_formed,
        "think_tokens": think_tokens,
        "answer_tokens": len(answer_ids),
        "stopped_on_im_end": not hit_budget,
        "repeated_4grams": repeated,
    }, answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--max-new", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="/tmp/chat_sft.json")
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
    print(f"loaded {arguments.checkpoint}", flush=True)

    tokenizer = load_sft_tokenizer(
        config.data.tokenizer, padded_vocab_size=config.model.vocab_size)
    group = jax.device_count()
    window = arguments.window

    @nnx.jit
    def next_logits(current_model, tokens, segments, positions, cursors):
        hidden = current_model.hidden_states(
            tokens, decoder_segment_ids=segments, decoder_positions=positions)
        picked = jnp.take_along_axis(
            hidden, cursors[:, None, None], axis=1)[:, 0, :]
        kernel = current_model.output_projection_kernel(picked.dtype)
        return (picked @ kernel).astype(jnp.float32)

    rng = np.random.default_rng(arguments.seed)
    positions = np.tile(np.arange(window, dtype=np.int32), (group, 1))
    results = []
    for start in range(0, len(PROMPTS), group):
        chunk = PROMPTS[start : start + group]
        while len(chunk) < group:  # pad the group with a repeat, dropped later
            chunk = chunk + [chunk[-1]]
        rows = [render_prompt(tokenizer, text) for _, text in chunk]
        prompt_lengths = [len(row) for row in rows]
        finished = [False] * group
        began = time.perf_counter()
        with logical_mesh_context(mesh, rules):
            for _ in range(arguments.max_new):
                if all(finished):
                    break
                tokens = np.full((group, window), DOCUMENT_SEPARATOR, np.int32)
                segments = np.zeros((group, window), np.int32)
                cursors = np.zeros((group,), np.int32)
                for index, row in enumerate(rows):
                    length = min(len(row), window)
                    tokens[index, :length] = row[:length]
                    segments[index, :length] = 1
                    cursors[index] = length - 1
                logits = np.asarray(next_logits(
                    model, jnp.asarray(tokens), jnp.asarray(segments),
                    jnp.asarray(positions),
                    jnp.asarray(cursors),
                )).astype(np.float64)
                for index in range(group):
                    if finished[index]:
                        continue
                    row_logits = logits[index]
                    if arguments.repetition_penalty != 1.0:
                        emitted = rows[index][prompt_lengths[index]:]
                        if emitted:
                            seen = np.unique(np.asarray(emitted))
                            row_logits = row_logits.copy()
                            row_logits[seen] = np.where(
                                row_logits[seen] > 0,
                                row_logits[seen] / arguments.repetition_penalty,
                                row_logits[seen] * arguments.repetition_penalty,
                            )
                    if arguments.temperature > 0:
                        row_logits = row_logits / arguments.temperature
                        top = np.argpartition(row_logits, -arguments.top_k)[
                            -arguments.top_k:]
                        probabilities = np.exp(row_logits[top] - row_logits[top].max())
                        probabilities /= probabilities.sum()
                        token = int(rng.choice(top, p=probabilities))
                    else:
                        token = int(row_logits.argmax())
                    rows[index].append(token)
                    if token == IM_END or len(rows[index]) >= window:
                        finished[index] = True
        elapsed = time.perf_counter() - began
        for index, (domain, text) in enumerate(chunk):
            if start + index >= len(PROMPTS):
                break
            generated = rows[index][prompt_lengths[index]:]
            hit_budget = IM_END not in generated
            report, answer = structure_report(
                tokenizer, generated, hit_budget=hit_budget)
            think_text = ""
            if report["well_formed_think"]:
                open_at = generated.index(THINK_OPEN)
                close_at = generated.index(THINK_CLOSE)
                think_text = tokenizer.decode(generated[open_at + 1 : close_at])
            results.append({
                "domain": domain, "prompt": text, "answer": answer,
                "think": think_text, **report,
                "generated_tokens": len(generated),
            })
            print(f"\n=== [{domain}] {text}", flush=True)
            print(f"--- think ({report['think_tokens']} tok): "
                  f"{think_text[:600]}", flush=True)
            print(f"--- answer: {answer[:1200]}", flush=True)
            print(f"--- {json.dumps(report)}", flush=True)
        print(f"[group {start // group}: {elapsed:.0f}s]", flush=True)

    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    formed = sum(r["well_formed_think"] for r in results)
    stopped = sum(r["stopped_on_im_end"] for r in results)
    print(f"\nwell-formed think {formed}/{len(results)}; "
          f"stopped on <|im_end|> {stopped}/{len(results)}; "
          f"written {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
