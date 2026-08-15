"""Part II figures: the 1B multimodal campaigns, built from the local
per-step records (results/vision-1b-{trial,cont30b}/leg*/metrics.jsonl).

Leg splices: the trial resumed from its step-36,000 checkpoint after a
stream stall (leg1 kept through 36,000); the continuation resumed from
step 8,000 after a dead-HTTP-client stall (leg1 kept through 8,000).
The continuation's steps are offset by 48,000 onto one cumulative axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TOKENS_PER_STEP = 524_288

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#8a8a85"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e2",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def _load(path, cutoff=None, offset=0):
    rows = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            step = row.get("step")
            vision = row.get("vision")
            if step is None or not vision:
                continue
            if cutoff is not None and step > cutoff:
                continue
            rows.append(
                (
                    step + offset,
                    vision.get("vision_loss"),
                    vision.get("text_loss"),
                    vision.get("embed_rms_ratio"),
                )
            )
    return rows


def _ema(values, alpha=0.02):
    out, state = [], None
    for value in values:
        state = value if state is None else (1 - alpha) * state + alpha * value
        out.append(state)
    return out


def campaign_rows():
    trial = _load(
        ROOT / "results/vision-1b-trial/leg1/metrics.jsonl", cutoff=36_000
    ) + _load(ROOT / "results/vision-1b-trial/leg2/metrics.jsonl")
    cont = _load(
        ROOT / "results/vision-1b-cont30b/leg1/metrics.jsonl",
        cutoff=8_000,
        offset=48_000,
    ) + _load(
        ROOT / "results/vision-1b-cont30b/leg2/metrics.jsonl", offset=48_000
    )
    rows = sorted(trial + cont)
    return rows


def vision_loss_curves(rows):
    steps = [r[0] * TOKENS_PER_STEP / 1e9 for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(steps, _ema([r[2] for r in rows]), color=BLUE, lw=1.3, label="text loss")
    ax.plot(steps, _ema([r[1] for r in rows]), color=ORANGE, lw=1.3, label="vision loss")
    boundary = 48_000 * TOKENS_PER_STEP / 1e9
    ax.axvline(boundary, color=GRAY, lw=0.8, ls="--")
    ax.text(boundary, ax.get_ylim()[1], " continuation", color=GRAY, fontsize=7, va="top")
    for start, stop in ((43_200, 48_000), (48_000 + 51_498, 48_000 + 57_220)):
        ax.axvspan(start * TOKENS_PER_STEP / 1e9, stop * TOKENS_PER_STEP / 1e9,
                   color=GRAY, alpha=0.12, lw=0)
    ax.set_xlabel("training tokens (billions)")
    ax.set_ylabel("loss (nats)")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "vision_loss_curves.pdf")
    plt.close(fig)


def embed_ratio(rows):
    steps = [r[0] * TOKENS_PER_STEP / 1e9 for r in rows]
    values = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.plot(steps, _ema(values, alpha=0.05), color=AQUA, lw=1.3)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    boundary = 48_000 * TOKENS_PER_STEP / 1e9
    ax.axvline(boundary, color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("training tokens (billions)")
    ax.set_ylabel("visual/text embed RMS")
    fig.savefig(OUT / "embed_ratio.pdf")
    plt.close(fig)


def vision_eval_bars():
    # Values from results/vision-1b-cont30b/gsm8k, the standalone final
    # evals (this transcript's pinned harness), and
    # post_training/results/pretrain-700k/SUMMARY.md.
    tasks = ["HellaSwag", "PIQA", "ARC-e", "ARC-c", "Lambada", "SciQ",
             "BoolQ", "OBQA", "COPA", "CSQA"]
    cont = [54.6, 74.3, 63.0, 34.6, 43.3, 92.9, 62.6, 37.6, 73.0, 31.9]
    trial = [55.4, 74.5, 65.2, 36.5, 42.3, 92.4, 63.7, 37.6, 70.0, 31.4]
    m308 = [56.9, 75.2, 67.5, 40.1, 47.4, 92.0, 58.0, 38.8, 72.0, 26.2]
    x = range(len(tasks))
    width = 0.27
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar([i - width for i in x], m308, width, color=GRAY, label="308M @ 367B (text)")
    ax.bar(list(x), trial, width, color=BLUE, label="1B @ 25B (multimodal)")
    ax.bar([i + width for i in x], cont, width, color=ORANGE, label="1B @ 55B (+code/math)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("primary metric")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(OUT / "vision_eval_bars.pdf")
    plt.close(fig)


if __name__ == "__main__":
    rows = campaign_rows()
    print(f"{len(rows)} step rows")
    vision_loss_curves(rows)
    embed_ratio(rows)
    vision_eval_bars()
    print("wrote vision_loss_curves.pdf embed_ratio.pdf vision_eval_bars.pdf")
