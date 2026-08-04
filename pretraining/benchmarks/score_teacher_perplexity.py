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

"""Checks a converted Qwen3.5 teacher by asking it to predict its own output.

Conversion bugs are quiet: a scrambled qkvz interleave, a transposed
kernel or a wrong RoPE all yield finite, plausibly-scaled logits and
nothing crashes. This is the cheap structural check that catches them
without a torch reference.

The Mephisto sets were generated *by Qwen3.5-4B*, so a correct conversion
should assign their assistant turns low perplexity - the model is being
asked to predict text it produced. Any structural error destroys
next-token prediction and sends perplexity toward the vocabulary size.

Read it against the two controls, not against an absolute threshold:

  mephisto completions  <  climbmix prose  <<  random ids (~ vocab size)

If that ordering holds with a large gap to random, the layout is right.
If all three are close, the model is degenerate however good any single
number looks.

Sampling note: the sets were drawn at temperature 0.7 / top_p 0.8 /
top_k 20, which restricts to the high-probability head and pushes
perplexity down, but with presence_penalty 1.5, which pushes it back up by
steering off already-used tokens. So expect low-but-not-tiny, and judge by
the ordering.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/score_teacher_perplexity.py \
    --checkpoint /home/a1111/yxtpu_ckpts/qwen35-4b-teacher
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

SYSTEM = ("You are a helpful assistant. Answer the user's question "
          "accurately, clearly, and concisely.")


def _find_weights(node, depth=0):
    """Finds the parameter root: the dict holding decoder and token_embedder."""
    if not isinstance(node, dict):
        raise ValueError("no parameter root found in the restored checkpoint")
    if "decoder" in node and "token_embedder" in node:
        return node
    if depth > 4:
        raise ValueError(f"no parameter root within 4 levels; keys {list(node)}")
    for value in node.values():
        if isinstance(value, dict):
            try:
                return _find_weights(value, depth + 1)
            except ValueError:
                continue
    raise ValueError(f"no parameter root under keys {list(node)}")


def load_model(checkpoint, base, model_name, sequence):
    from jax.sharding import Mesh
    import orbax.checkpoint as ocp

    from maxtext import pyconfig
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.models.models import Transformer

    config = pyconfig.initialize([
        "score_teacher", base, f"model_name={model_name}",
        "run_name=score_teacher", "per_device_batch_size=1",
        f"max_target_length={sequence}", "skip_jax_distributed_system=true",
        "enable_checkpointing=false", "scan_layers=true",
    ])
    devices = np.array(jax.devices())
    mesh = Mesh(devices.reshape(1, -1), ("data", "model"))
    print(f"{len(devices)} devices", flush=True)

    model = Transformer(config=config, mesh=mesh, quant=None,
                        rngs=nnx.Rngs(0), model_mode=MODEL_MODE_TRAIN)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    manager = ocp.CheckpointManager(checkpoint, options=ocp.CheckpointManagerOptions())
    step = manager.latest_step()
    print(f"restoring step {step} from {checkpoint}", flush=True)
    # MaxText saves a TrainState under a Composite item named "items", so a
    # bare restore() returns the wrapper rather than the tree. Ask for a
    # PyTree explicitly, then locate the weights by their own shape instead
    # of hard-coding how many wrappers deep they sit.
    restored = manager.restore(
        step, args=ocp.args.Composite(items=ocp.args.PyTreeRestore()))
    weights = _find_weights(restored["items"])

    nnx.replace_by_pure_dict(params, weights)
    model = nnx.merge(graphdef, params, rest)
    return config, model, mesh


def render_chat(tokenizer, record):
    """Prompt (masked) and the assistant turn the teacher actually emitted."""
    messages = list(record["messages"])
    prompt = [m for m in messages if m["role"] != "assistant"]
    if not any(m["role"] == "system" for m in prompt):
        prompt = [{"role": "system", "content": record.get("system") or SYSTEM},
                  *prompt]
    answer = next(m for m in messages if m["role"] == "assistant")["content"]
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(answer + "<|im_end|>\n",
                                      add_special_tokens=False)
    return prompt_ids, completion_ids


def batches(rows, sequence, batch_size, pad_id):
    """Fixed-shape batches of (ids, score mask); one example per row."""
    packed = []
    for prompt_ids, completion_ids in rows:
        ids = (prompt_ids + completion_ids)[:sequence]
        mask = ([0] * len(prompt_ids) + [1] * len(completion_ids))[:sequence]
        if sum(mask) < 8:            # too little signal to be informative
            continue
        pad = sequence - len(ids)
        packed.append((np.asarray(ids + [pad_id] * pad, np.int32),
                       np.asarray(mask + [0] * pad, np.float32)))
    for start in range(0, len(packed) - batch_size + 1, batch_size):
        chunk = packed[start:start + batch_size]
        yield (np.stack([c[0] for c in chunk]),
               np.stack([c[1] for c in chunk]))


def score(model, config, ids, mask):
    """Mean CE and top-1 agreement over the masked (generated) positions."""
    from maxtext.common.common_types import MODEL_MODE_TRAIN

    batch, sequence = ids.shape
    positions = jnp.broadcast_to(jnp.arange(sequence, dtype=jnp.int32),
                                 (batch, sequence))
    # One segment per row: padding is excluded by the score mask, not by
    # attention, which only costs a little wasted attention on pad tails.
    segments = jnp.ones((batch, sequence), jnp.int32)
    logits = model(
        jnp.asarray(ids), positions, segments,
        enable_dropout=False, model_mode=MODEL_MODE_TRAIN,
    )
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits[:, :-1, :].astype(jnp.float32)
    targets = jnp.asarray(ids)[:, 1:]
    weights = jnp.asarray(mask)[:, 1:]
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0]
    total = jnp.maximum(weights.sum(), 1.0)
    nll = -(picked * weights).sum() / total
    agree = ((logits.argmax(-1) == targets) * weights).sum() / total
    return float(nll), float(agree), float(total)


def run(label, model, config, rows, sequence, batch_size, pad_id, limit):
    nll_sum = agree_sum = token_sum = 0.0
    seen = 0
    for ids, mask in batches(rows, sequence, batch_size, pad_id):
        nll, agree, tokens = score(model, config, ids, mask)
        nll_sum += nll * tokens
        agree_sum += agree * tokens
        token_sum += tokens
        seen += 1
        if seen >= limit:
            break
    if not token_sum:
        print(f"{label}: no scorable tokens")
        return None
    mean_nll = nll_sum / token_sum
    report = {"label": label, "tokens": int(token_sum),
              "nll": mean_nll, "perplexity": float(np.exp(mean_nll)),
              "top1_agreement": agree_sum / token_sum}
    print(f"{label:<26} ppl {report['perplexity']:10.3f}   "
          f"nll {mean_nll:7.4f}   top-1 {report['top1_agreement']:6.2%}   "
          f"tokens {int(token_sum):,}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base", default="../maxtext/src/maxtext/configs/base.yml")
    parser.add_argument("--model-name", default="qwen3.5-4b")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--mephisto", default="Yxanul/Mephisto-IF_172k")
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--output", default="/tmp/teacher_perplexity.json")
    arguments = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arguments.teacher, use_fast=True)
    pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    config, model, mesh = load_model(
        arguments.checkpoint, arguments.base, arguments.model_name,
        arguments.sequence)

    need = arguments.batch_size * arguments.batches * 2
    chat_rows = []
    stream = load_dataset(arguments.mephisto, split="train", streaming=True)
    for index, record in enumerate(stream):
        if len(chat_rows) >= need:
            break
        try:
            chat_rows.append(render_chat(tokenizer, record))
        except (KeyError, StopIteration):
            continue

    prose_rows = []
    climb = load_dataset("karpathy/climbmix-400b-shuffle", split="train",
                         streaming=True)
    for record in climb:
        if len(prose_rows) >= need:
            break
        ids = tokenizer.encode(record["text"], add_special_tokens=False)
        if len(ids) > 64:
            prose_rows.append(([ids[0]], ids[1:arguments.sequence]))

    rng = np.random.default_rng(0)
    random_rows = [([1], rng.integers(0, 248_077, arguments.sequence - 1).tolist())
                   for _ in range(need)]

    print(flush=True)
    reports = [
        run("mephisto completions", model, config, chat_rows,
            arguments.sequence, arguments.batch_size, pad_id, arguments.batches),
        run("climbmix prose", model, config, prose_rows,
            arguments.sequence, arguments.batch_size, pad_id, arguments.batches),
        run("random ids (control)", model, config, random_rows,
            arguments.sequence, arguments.batch_size, pad_id, 2),
    ]
    reports = [r for r in reports if r]
    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": arguments.checkpoint, "corpora": reports},
                  handle, indent=2)
    print(f"\nwritten {arguments.output}", flush=True)

    by_label = {r["label"]: r["perplexity"] for r in reports}
    ordered = (by_label.get("mephisto completions", 1e9)
               < by_label.get("climbmix prose", 1e9)
               < by_label.get("random ids (control)", 0))
    print("ordering mephisto < climbmix << random: "
          f"{'OK' if ordered else 'VIOLATED - inspect the layout'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
