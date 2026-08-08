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

"""Build every figure in the technical report from the exported data.

Inputs: data/{60sndjih,oiggd5fp}*.csv (W&B exports; see export_wandb.py),
../../results/gold/gold_mix300k_steps.jsonl, and the v6e8 sequence-sweep
summary.json files. Run:

  uv run --with matplotlib --with pandas python make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"

TOKENS_PER_STEP = 524_288
MAIN, RESUME = "60sndjih", "oiggd5fp"
RESUME_OFFSET = 700_000 - 50_032  # continuation counters restarted at zero
ANNEAL_START = 630_000
CRASH_STEP = 678_420

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


def ema(series, alpha=0.02):
    return series.ewm(alpha=alpha).mean()


def load_leg(leg: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / f"{leg}.csv")
    offset = RESUME_OFFSET if leg == RESUME else 0
    frame["global_step"] = frame["_step"] + offset
    frame["tokens_b"] = frame["global_step"] * TOKENS_PER_STEP / 1e9
    return frame


def shade_anneal(ax, x_tokens=False):
    lo = ANNEAL_START * TOKENS_PER_STEP / 1e9 if x_tokens else ANNEAL_START / 1e3
    hi = 700_000 * TOKENS_PER_STEP / 1e9 if x_tokens else 700.0
    ax.axvspan(lo, hi, color="#f2f0ea", zorder=0)


def pretrain_loss() -> None:
    main, resume = load_leg(MAIN), load_leg(RESUME)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    shade_anneal(ax, x_tokens=True)
    ax.plot(main["tokens_b"], ema(main["train/loss"]), color=BLUE, lw=1.2,
            label="train loss (main leg)")
    ax.plot(resume["tokens_b"], ema(resume["train/loss"]), color=ORANGE,
            lw=1.2, label="train loss (resume leg)")
    for leg, color in ((MAIN, BLUE), (RESUME, ORANGE)):
        holdout = pd.read_csv(DATA / f"{leg}_holdout.csv")
        offset = RESUME_OFFSET if leg == RESUME else 0
        tokens = (holdout["_step"] + offset) * TOKENS_PER_STEP / 1e9
        ax.plot(tokens, holdout["eval/train_holdout_loss"], color=color,
                lw=0, marker="o", ms=2.2, alpha=0.75,
                label="holdout eval" if leg == MAIN else None)
    crash_tokens = CRASH_STEP * TOKENS_PER_STEP / 1e9
    ax.axvline(crash_tokens, color=GRAY, lw=0.8, ls="--")
    ax.annotate("main leg crash\n(step 678,420)", (crash_tokens, 3.28),
                textcoords="offset points", xytext=(-6, 0), ha="right",
                va="top", fontsize=7.5, color="#55554f")
    ax.annotate("anneal\n(final 70k steps)", (328, 2.85), fontsize=7.5,
                ha="right", color="#55554f")
    ax.annotate("holdout 2.3494", (367, 2.36), textcoords="offset points",
                xytext=(-8, 8), ha="right", fontsize=7.5, color="#55554f")
    ax.set_xlim(0, 372)
    ax.set_ylim(2.25, 3.4)
    ax.set_xlabel("training tokens (billions)")
    ax.set_ylabel("cross-entropy (nats)")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.savefig(HERE / "pretrain_loss.pdf")
    plt.close(fig)


def lr_gradnorm() -> None:
    main, resume = load_leg(MAIN), load_leg(RESUME)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.2, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.12})
    for ax in (top, bottom):
        shade_anneal(ax)
    for frame, color in ((main, BLUE), (resume, ORANGE)):
        steps_k = frame["global_step"] / 1e3
        top.plot(steps_k, frame["optimizer/learning_rate"] * 1e3,
                 color=color, lw=1.2)
        bottom.plot(steps_k, ema(frame["optimizer/grad_norm"], 0.05),
                    color=color, lw=1.0)
    top.set_ylabel("LR ($\\times 10^{-3}$)")
    top.set_ylim(0, 2.3)
    top.annotate("constant $2\\times10^{-3}$", (300, 2.05), fontsize=7.5,
                 ha="center", color="#55554f")
    top.annotate("cosine to 0", (622, 0.9), fontsize=7.5, ha="right",
                 color="#55554f")
    bottom.set_yscale("log")
    bottom.set_ylabel("gradient norm")
    bottom.set_xlabel("step (thousands)")
    bottom.set_xlim(0, 710)
    fig.savefig(HERE / "lr_gradnorm.pdf")
    plt.close(fig)


def throughput() -> None:
    main, resume = load_leg(MAIN), load_leg(RESUME)
    both = pd.concat([main, resume])
    median = both["performance/tokens_per_second"].median()
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    shade_anneal(ax)
    for frame, color, label in ((main, BLUE, "main leg"),
                                (resume, ORANGE, "resume leg")):
        steps_k = frame["global_step"] / 1e3
        tok_s = frame["performance/tokens_per_second"] / 1e3
        ax.plot(steps_k, tok_s, color=color, lw=0.4, alpha=0.22)
        ax.plot(steps_k, tok_s.rolling(101, center=True, min_periods=10)
                .median(), color=color, lw=1.3, label=label)
    ax.axhline(median / 1e3, color=GRAY, lw=0.8, ls=":")
    ax.annotate(f"median {median/1e3:.0f}k tok/s", (20, median / 1e3),
                textcoords="offset points", xytext=(0, 5), fontsize=7.5,
                color="#55554f")
    ax.set_xlim(0, 710)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("step (thousands)")
    ax.set_ylabel("tokens/s (thousands)")
    ax.legend(loc="lower right", fontsize=7.5, ncols=2)
    fig.savefig(HERE / "throughput.pdf")
    plt.close(fig)


def eval_trajectory() -> None:
    tasks = (("lm_eval/hellaswag/primary", "HellaSwag", BLUE),
             ("lm_eval/arc_easy/primary", "ARC-easy", ORANGE),
             ("lm_eval/lambada_openai/primary", "Lambada", AQUA),
             ("lm_eval/sciq/primary", "SciQ", YELLOW))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    shade_anneal(ax)
    for leg in (MAIN, RESUME):
        rounds = pd.read_csv(DATA / f"{leg}_lmeval.csv")
        offset = RESUME_OFFSET if leg == RESUME else 0
        steps_k = (rounds["_step"] + offset) / 1e3
        for key, label, color in tasks:
            ax.plot(steps_k, rounds[key] * 100, color=color, lw=1.2,
                    marker="o", ms=2.4,
                    label=label if leg == MAIN else None)
    for key, label, color in tasks:
        final = pd.read_csv(DATA / f"{RESUME}_lmeval.csv")[key].iloc[-1] * 100
        ax.annotate(f"{final:.1f}", (703, final), fontsize=7.5, color=color,
                    va="center")
    ax.set_xlim(0, 745)
    ax.set_xlabel("step (thousands)")
    ax.set_ylabel("0-shot accuracy (%)")
    ax.legend(loc="center left", fontsize=7.5, bbox_to_anchor=(0.02, 0.62))
    fig.savefig(HERE / "eval_trajectory.pdf")
    plt.close(fig)


def eval_bars() -> None:
    # Values from post_training/results/pretrain-700k/SUMMARY.md
    # (ours: 0-shot in-training lm-eval at step 700k; SmolLM2-360M base at
    # its published settings).
    tasks = ("HellaSwag", "ARC avg", "PIQA", "OpenBookQA", "WinoGrande")
    ours = (56.9, 53.8, 75.2, 38.8, 59.4)
    smollm2 = (54.5, 53.0, 71.7, 37.4, 52.5)
    x = range(len(tasks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    bars_ours = ax.bar([i - width / 2 for i in x], ours, width,
                       color=BLUE, label="ours (308M, 0.37T tokens)")
    bars_ref = ax.bar([i + width / 2 for i in x], smollm2, width,
                      color=ORANGE, label="SmolLM2-360M (4T tokens)")
    for bars in (bars_ours, bars_ref):
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1.5,
                     color="#55554f")
    ax.set_xticks(list(x), tasks)
    ax.set_ylim(0, 84)
    ax.set_ylabel("accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.savefig(HERE / "eval_bars.pdf")
    plt.close(fig)


def gold_curves() -> None:
    rows = [json.loads(line) for line in
            (REPO / "results/gold/gold_mix300k_steps.jsonl").open()]
    frame = pd.DataFrame(rows)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.2, 3.4), sharex=True,
        gridspec_kw={"hspace": 0.14})
    steps = frame["step"]
    top.plot(steps, ema(frame["distill"], 0.05), color=BLUE, lw=1.2,
             label="distillation loss (fwd KL + coverage)")
    top.plot(steps, ema(frame["ce"], 0.05), color=GRAY, lw=1.0,
             label="CE vs teacher tokens (monitor)")
    top.set_ylabel("loss (nats)")
    top.legend(loc="upper right", fontsize=7.5)
    bottom.plot(steps, ema(frame["teacher_top1_is_label"], 0.05) * 100,
                color=ORANGE, lw=1.2, label="teacher top-1 = label")
    bottom.plot(steps, ema(frame["student_top1_is_label"], 0.05) * 100,
                color=AQUA, lw=1.2, label="student top-1 = label")
    bottom.plot(steps, ema(frame["teacher_rest_mass"], 0.05) * 100,
                color=YELLOW, lw=1.2, label="teacher rest mass")
    bottom.set_ylabel("%")
    bottom.set_xlabel("step")
    bottom.legend(loc="center right", fontsize=7.5)
    fig.savefig(HERE / "gold_curves.pdf")
    plt.close(fig)


def seq_sweep() -> None:
    lengths = (2048, 4096, 8192, 16384, 32768)
    series = {}
    for kind in ("kda", "attn"):
        points = []
        for length in lengths:
            path = REPO / f"results/v6e8-seq-{kind}-t{length}/summary.json"
            summary = json.loads(path.read_text())
            points.append(summary["tokens_per_second_global"]["median"] / 1e3)
        series[kind] = points
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(lengths, series["kda"], color=BLUE, lw=1.4, marker="o", ms=4,
            label="hybrid (KDA 3:1)")
    ax.plot(lengths, series["attn"], color=ORANGE, lw=1.4, marker="s", ms=4,
            label="pure attention")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, [f"{n//1024}k" for n in lengths])
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("throughput (k tok/s)")
    ax.legend(fontsize=7.5)
    fig.savefig(HERE / "seq_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for figure in (pretrain_loss, lr_gradnorm, throughput, eval_trajectory,
                   eval_bars, gold_curves, seq_sweep):
        figure()
        print("built", figure.__name__, flush=True)
