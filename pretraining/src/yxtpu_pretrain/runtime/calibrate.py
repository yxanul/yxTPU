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

"""Mix calibration for the packed vision+text pipeline.

The composition knobs (``vision.p_text`` and the per-source ``weight``)
are row-level draw probabilities, but what matters is the LOSS-TOKEN
share per source. Those differ by the mean supervised tokens per row of
each source, so the weights are solved from a measured calibration:
run the real producer for N sequences, record per-source rows and row
tokens, and solve

    w_i  ~  target_share_i / mean_row_tokens_i          (text side)
    p_text = v(1-s_v) / (s_v * m_t + v(1-s_v))          (vision share s_v)

with ``v`` the mean supervised tokens per vision row and ``m_t`` the
weight-averaged mean supervised tokens per text row under the solved
weights. This is the hand calibration of the 30B continuation (96
sequences, 2026-08-12) as a repeatable command; rerun it whenever
row_tokens, sequence_length, max_images_per_row, or the source list
changes - each moves the per-row token means.
"""

from __future__ import annotations

import json
import time
from typing import Any

from yxtpu_pretrain.config import ResolvedConfig


def solve_weights(
    targets: dict[str, float],
    mean_row_tokens: dict[str, float],
    *,
    vision_key: str = "vision",
) -> dict[str, Any]:
    """Solves per-draw text weights and p_text from target loss-token shares."""
    text_targets = {name: share for name, share in targets.items() if name != vision_key}
    total_text = sum(text_targets.values())
    if total_text <= 0:
        raise ValueError("targets must give the text sources a positive share")
    raw = {}
    for name, share in text_targets.items():
        mean = mean_row_tokens.get(name)
        if not mean:
            raise ValueError(f"no measured row tokens for source {name!r}")
        raw[name] = share / mean
    norm = sum(raw.values())
    weights = {name: value / norm for name, value in raw.items()}
    # Expected supervised tokens per text draw under the solved weights.
    m_t = sum(weights[name] * mean_row_tokens[name] for name in weights)
    result: dict[str, Any] = {"weights": weights, "text_row_tokens_per_draw": m_t}
    s_v = targets.get(vision_key, 0.0)
    v = mean_row_tokens.get(vision_key)
    if s_v > 0 and v:
        p_text = v * (1.0 - s_v) / (s_v * m_t + v * (1.0 - s_v))
        result["p_text"] = p_text
        # Predicted loss-token shares under the solution (self-consistency).
        text_mass = p_text * m_t
        vision_mass = (1.0 - p_text) * v
        total = text_mass + vision_mass
        predicted = {vision_key: vision_mass / total}
        for name in weights:
            predicted[name] = p_text * weights[name] * mean_row_tokens[name] / total
        result["predicted_shares"] = predicted
    return result


def calibrate_mix(
    config: ResolvedConfig,
    *,
    sequences: int,
    targets: dict[str, float],
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Runs the packer for ``sequences`` examples and solves the mix."""
    from yxtpu_pretrain.runtime.data import load_fast_tokenizer
    from yxtpu_pretrain.runtime.vision_data import MixedVisionTextIterator, VisionBatchSpec

    vision = config.model.vision
    if not (vision.enabled and vision.dataset_name):
        raise ValueError("calibrate-mix needs model.vision.enabled and dataset_name")
    tokenizer = load_fast_tokenizer(
        config.data.tokenizer, padded_vocab_size=config.model.vocab_size
    )
    spec = VisionBatchSpec(
        sequence_length=config.data.sequence_length,
        visual_tokens=vision.visual_tokens_per_image,
        image_size=vision.image_size,
        placeholder_id=vision.placeholder_token_id,
        pad_id=vision.pad_token_id,
        eos_id=vision.eos_token_id,
        max_images=vision.max_images_per_sequence,
        patch_size=vision.patch_size,
        host_patchify=vision.host_patchify,
    )
    text_sources = [
        {
            "name": source.name,
            "dataset": source.dataset,
            "subset": source.subset,
            "weight": source.weight,
            "field": source.field,
            "format": source.format,
            "row_tokens": source.row_tokens or vision.text_row_tokens,
        }
        for source in vision.text_datasets
    ]
    iterator = MixedVisionTextIterator(
        tokenizer=tokenizer,
        spec=spec,
        batch_size=1,
        vision_dataset=vision.dataset_name,
        text_sources=text_sources,
        p_text=vision.p_text,
        text_row_tokens=vision.text_row_tokens,
        min_visual_dependency=vision.min_visual_dependency,
        shuffle_seed=config.data.shuffle_seed,
        shard_index=shard_index,
        shard_count=shard_count,
        max_images_per_row=vision.max_images_per_row,
        row_buffer=vision.row_buffer,
    )
    started = time.perf_counter()
    for _ in range(sequences):
        iterator._next_example()
    elapsed = time.perf_counter() - started
    rows = iterator.raw_source_rows()
    # Mean SUPERVISED tokens per row: text rows contribute L-1 labels; a
    # vision row's supervised tokens are its dialogue labels, i.e. its
    # loss tokens divided by its rows.
    mean_row_tokens = {}
    for name, (count, tokens) in rows.items():
        if name == "vision":
            mean_row_tokens[name] = (
                iterator.loss_tokens_vision / max(iterator.vision_rows, 1)
            )
        else:
            mean_row_tokens[name] = (tokens - count) / max(count, 1)
    stats = iterator.stats
    realized = {
        key.removesuffix("_loss_token_share"): value
        for key, value in stats.items()
        if key.endswith("_loss_token_share")
    }
    result = {
        "sequences": sequences,
        "seconds": elapsed,
        "sequences_per_second": sequences / max(elapsed, 1e-9),
        "rows": {name: {"rows": count, "row_tokens_total": tokens} for name, (count, tokens) in rows.items()},
        "mean_supervised_tokens_per_row": mean_row_tokens,
        "realized_loss_token_shares": realized,
        "pad_fraction": stats.get("pad_fraction"),
        "images_per_sequence": stats.get("images_per_sequence"),
        "image_slot_utilization": stats.get("image_slot_utilization"),
        "row_skip_rate": stats.get("row_skip_rate"),
        "current": {
            "p_text": vision.p_text,
            "weights": {source["name"]: source["weight"] for source in text_sources},
        },
    }
    if targets:
        result["targets"] = targets
        result["solution"] = solve_weights(targets, mean_row_tokens)
    return result


def parse_targets(raw: str | None) -> dict[str, float]:
    """``vision=0.35,climbmix=0.40,stack=0.17,math=0.08`` -> dict."""
    if not raw:
        return {}
    targets = {}
    for part in raw.split(","):
        name, _, value = part.partition("=")
        if not name or not value:
            raise ValueError(f"bad target {part!r}; expected name=share")
        targets[name.strip()] = float(value)
    total = sum(targets.values())
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"target shares must sum to 1 (got {total:.4f})")
    return targets


def format_report(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True)
