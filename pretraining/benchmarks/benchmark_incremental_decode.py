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

"""Incremental decode against the recompute loop, on real weights.

Checks first that cached decoding agrees with the recompute path it
replaces - same greedy tokens, matching next-token logits - then measures
what the cache is worth in tokens per second.

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  benchmarks/benchmark_incremental_decode.py --checkpoint <dir>/state.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.decode import (
    SamplingParams,
    generate,
    init_cache,
    model_step,
    split_cycles,
)
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
    load_sft_tokenizer,
)

QUESTIONS = [
    "What is 17 + 25?",
    "Respond with a single word only: what colour is the sky on a clear day?",
    "Write a Python function named reverse_string(s) that returns the reversed "
    "string. Reply with only a Python code block.",
    "Name one country in Europe. Wrap your entire answer in double quotes.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window", type=int, default=4096)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--compare-tokens", type=int, default=24)
    parser.add_argument("--output", default="/tmp/incremental_decode.json")
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

    prompts = [
        [DOCUMENT_SEPARATOR, ROLE_TOKENS["user"],
         *tokenizer.encode("user", add_special_tokens=False), IM_MIDDLE,
         *tokenizer.encode(text, add_special_tokens=False), IM_END,
         ROLE_TOKENS["assistant"],
         *tokenizer.encode("assistant", add_special_tokens=False), IM_MIDDLE]
        for text in [
            QUESTIONS[i % len(QUESTIONS)] for i in range(arguments.batch)
        ]
    ]
    lengths = np.asarray([len(p) for p in prompts], np.int32)
    width = int(lengths.max())
    padded = np.full((len(prompts), width), DOCUMENT_SEPARATOR, np.int32)
    for index, prompt in enumerate(prompts):
        padded[index, : len(prompt)] = prompt
    batch = len(prompts)

    # ---- agreement with the recompute path this replaces -------------------
    window = arguments.window

    @nnx.jit
    def recompute_logits(current_model, tokens, segments, position_ids, cursors):
        hidden = current_model.hidden_states(
            tokens, decoder_segment_ids=segments, decoder_positions=position_ids)
        picked = jnp.take_along_axis(hidden, cursors[:, None, None], axis=1)[:, 0, :]
        kernel = current_model.output_projection_kernel(picked.dtype)
        return (picked @ kernel).astype(jnp.float32)

    rows = [list(prompt) for prompt in prompts]
    recompute_tokens = [[] for _ in prompts]
    positions_full = np.tile(np.arange(window, dtype=np.int32), (batch, 1))
    warm_tokens = np.zeros((batch, window), np.int32)
    warm_cursors = np.zeros((batch,), np.int32)
    with logical_mesh_context(mesh, rules):
        # Compile before timing; otherwise the first call's compile is
        # amortized into the per-token number and inflates it several-fold.
        jax.block_until_ready(recompute_logits(
            model, jnp.asarray(warm_tokens), jnp.asarray(warm_tokens),
            jnp.asarray(positions_full), jnp.asarray(warm_cursors)))
    began = time.perf_counter()
    with logical_mesh_context(mesh, rules):
        for _ in range(arguments.compare_tokens):
            tokens = np.full((batch, window), DOCUMENT_SEPARATOR, np.int32)
            segments = np.zeros((batch, window), np.int32)
            cursors = np.zeros((batch,), np.int32)
            for index, row in enumerate(rows):
                tokens[index, : len(row)] = row
                segments[index, : len(row)] = 1
                cursors[index] = len(row) - 1
            logits = np.asarray(recompute_logits(
                model, jnp.asarray(tokens), jnp.asarray(segments),
                jnp.asarray(positions_full), jnp.asarray(cursors)))
            for index in range(batch):
                token = int(logits[index].argmax())
                rows[index].append(token)
                recompute_tokens[index].append(token)
    recompute_ms = (time.perf_counter() - began) / arguments.compare_tokens * 1000

    with logical_mesh_context(mesh, rules):
        samples, _ = generate(
            model, jnp.asarray(padded), jnp.asarray(lengths), jax.random.key(0),
            max_new_tokens=arguments.compare_tokens,
            sampling=SamplingParams(temperature=0.0), end_token=-1,
            max_length=width + arguments.compare_tokens + 8,
        )
        samples = np.asarray(samples)
    cached_tokens = [
        list(samples[index, lengths[index] - 1 : lengths[index] - 1
                     + arguments.compare_tokens])
        for index in range(batch)
    ]
    # How far apart the two paths' next-token distributions actually are on
    # real weights, at the first generated position (same inputs both ways).
    cycles = split_cycles(model)
    with logical_mesh_context(mesh, rules):
        cache = init_cache(model, batch, max_length=width + 8)
        for step in range(width):
            cached_logits, cache = model_step(
                model, cycles, jnp.asarray(padded[:, step]), cache)
        tokens = np.full((batch, window), DOCUMENT_SEPARATOR, np.int32)
        segments = np.zeros((batch, window), np.int32)
        cursors = np.zeros((batch,), np.int32)
        for index, prompt in enumerate(prompts):
            tokens[index, : len(prompt)] = prompt
            segments[index, : len(prompt)] = 1
            cursors[index] = len(prompt) - 1
        reference_logits = np.asarray(recompute_logits(
            model, jnp.asarray(tokens), jnp.asarray(segments),
            jnp.asarray(positions_full), jnp.asarray(cursors)))
    # Only the rows whose prompt fills the padded width are comparable here.
    full_rows = [i for i in range(batch) if int(lengths[i]) == width]
    cached_logits = np.asarray(cached_logits)
    if full_rows:
        difference = np.abs(cached_logits[full_rows] - reference_logits[full_rows])
        relative = float(difference.max() / np.abs(reference_logits).max())
        top_gap = float(np.mean([
            np.sort(reference_logits[i])[-1] - np.sort(reference_logits[i])[-2]
            for i in full_rows
        ]))
        print(f"logit agreement: max rel {relative:.2e}; mean top-1/top-2 gap "
              f"{top_gap:.3f}", flush=True)
    else:
        relative, top_gap = float("nan"), float("nan")

    agreements = []
    for index in range(batch):
        same = [int(a) == int(b) for a, b in
                zip(recompute_tokens[index], cached_tokens[index])]
        prefix = next((i for i, ok in enumerate(same) if not ok), len(same))
        agreements.append(prefix)
        print(f"row {index}: identical greedy prefix {prefix}/"
              f"{arguments.compare_tokens}", flush=True)

    # ---- throughput --------------------------------------------------------
    key = jax.random.key(0)
    sampling = SamplingParams(temperature=0.7, top_k=64, top_p=0.95,
                              repetition_penalty=1.1)
    with logical_mesh_context(mesh, rules):
        # Warm with the SAME static arguments as the timed call: max_new_tokens
        # is static, so a different value recompiles the whole loop and the
        # compile lands inside the measurement.
        warm, _ = generate(
            model, jnp.asarray(padded), jnp.asarray(lengths), key,
            max_new_tokens=arguments.max_new, sampling=sampling,
            end_token=IM_END, max_length=width + arguments.max_new + 8)
        jax.block_until_ready(warm)
        began = time.perf_counter()
        samples, done = generate(
            model, jnp.asarray(padded), jnp.asarray(lengths), key,
            max_new_tokens=arguments.max_new, sampling=sampling,
            end_token=IM_END, max_length=width + arguments.max_new + 8)
        jax.block_until_ready(samples)
        elapsed = time.perf_counter() - began
    samples = np.asarray(samples)
    steps = 0
    for index in range(batch):
        row = samples[index, lengths[index] - 1:]
        stop = np.where(row == IM_END)[0]
        steps = max(steps, int(stop[0]) + 1 if len(stop) else len(row))
    cached_ms = elapsed / max(steps, 1) * 1000

    print(f"\nrecompute  {recompute_ms:7.1f} ms/token (batch {batch}, "
          f"window {window})", flush=True)
    print(f"cached     {cached_ms:7.1f} ms/token  over {steps} loop steps "
          f"in {elapsed:.1f}s", flush=True)
    print(f"speedup    {recompute_ms / max(cached_ms, 1e-9):7.1f}x", flush=True)
    print(f"throughput {batch / cached_ms * 1000:7.0f} tokens/s "
          f"({1000 / cached_ms:.0f} tokens/s/row)", flush=True)

    for index in range(batch):
        row = samples[index, lengths[index] - 1:]
        stop = np.where(row == IM_END)[0]
        text = tokenizer.decode(list(row[: stop[0]] if len(stop) else row))
        print(f"\n=== {QUESTIONS[index % len(QUESTIONS)]}\n{text[:400]}", flush=True)

    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump({
            "recompute_ms_per_token": recompute_ms,
            "cached_ms_per_token": cached_ms,
            "speedup": recompute_ms / max(cached_ms, 1e-9),
            "greedy_agreement_prefix": agreements,
            "logit_max_relative_difference": relative,
            "mean_top1_top2_gap": top_gap,
            "compare_tokens": arguments.compare_tokens,
            "batch": batch, "window": window,
        }, handle, indent=2)
    print(f"\nwritten {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
