# yxTPU

A complete small-language-model program on TPU Research Cloud hardware:
custom JAX/Pallas kernels, a 307.7M-parameter hybrid gated-delta-attention
model (`kda_hybrid_yx49k_l20`) pretrained on **367B tokens** of ClimbMix on
a TPU v4-64, a distillation-aligned 49k tokenizer, and a post-training
stack (SFT + offline logit distillation) that leaves the model competitive
with same-size models trained on 11-16x more data.

The full write-up — architecture math, kernel derivations, optimizer,
campaign narrative, and evaluations — is the technical report at
[`report/main.pdf`](report/main.pdf).

## Headline results

| | |
| --- | --- |
| Base model (step 700k) | train 2.3617 / holdout 2.3494 (ppl 10.48) |
| HellaSwag / PIQA / ARC-e (0-shot) | **56.9** / **75.2** / **67.5** |
| WinoGrande / SciQ (0-shot) | **59.4** / 92.0 |
| vs SmolLM2-360M (4T tokens) | ahead on every comparable row at ~1/11 the tokens |
| Post-training (GOLD mix300k) | IFEval mean **51.0** — above Gemma 3 270M IT and SmolLM2-360M-IT measured under the same harness |

Details: [`post_training/results/pretrain-700k/SUMMARY.md`](post_training/results/pretrain-700k/SUMMARY.md),
[`post_training/GOLD.md`](post_training/GOLD.md),
[`results/gemma-vs-gold/COMPARISON.md`](results/gemma-vs-gold/COMPARISON.md).

## The model

`kda_hybrid_yx49k_l20`: 307,698,680 parameters. Embedding 1024, 20 layers
as 5 cycles of `[KDA, KDA, KDA, GQA]` — three gated-delta-attention layers
(per-channel decay, chunkwise WY training form, chunk 64) per NoPE
global-attention layer (8 query / 2 KV heads). Fused SwiGLU MLP (2816),
RMSNorm, tied embeddings, vocab 49,152. Architecture notes:
[`docs/KDA_HYBRID.md`](docs/KDA_HYBRID.md).

## Kernels

The KDA training path is a custom fused Pallas kernel (forward + backward)
that owns the whole mixer — causal depthwise conv, SiLU, normalization,
cumulative decay, the WY solve, and the chunk recurrence — with the
fast-weight state resident in VMEM:

- The WY triangular solve is a **divide-and-conquer explicit inverse**
  (`inv = M - M C M`, exact by nilpotency) — +30% end-to-end over forward
  substitution and numerically qualified against fp64 on real text.
- Decay-anchored pairwise dot products keep every exponential bounded by
  `e^85 < fp32 max`, which is what makes the safe-gate bound `g in [-5, 0)`
  a *kernel compute parameter*, not just a modeling choice.
- Precision is `guarded_fp32`: one-pass bf16 everywhere except the solve
  (6-pass, or 3-pass emulated bf16x3 on v4). Blanket bf16 is 4.9x faster
  in the core and diverges by step two.
- TPU v4 gets its own kernel variant with a split backward (the folded
  kernel's sublane gathers don't compile on v4's Mosaic backend).

Numerics dossier: [`pretraining/src/yxtpu_pretrain/kernels/PRECISION.md`](pretraining/src/yxtpu_pretrain/kernels/PRECISION.md).
Kernel-era ledger: [`EXPERIMENTS.md`](EXPERIMENTS.md) (EXP-001..042) and
[`results/RESULTS.md`](results/RESULTS.md).

## Optimizer

MuonClip: Muon (5-step Newton-Schulz, consistent-RMS update scaling) for
all matrices, AdamW for embeddings/norms/scalars, plus a GQA-adapted
QK-clip (tau = 100; per-head scale on Q, min-over-group on shared K). In
the 700k-step campaign the max attention logit peaked at 27.5 — the clip
never fired, at zero cost.
Notes: [`pretraining/src/yxtpu_pretrain/optimizers/MUONCLIP.md`](pretraining/src/yxtpu_pretrain/optimizers/MUONCLIP.md).

## Tokenizer: yx49k

A 49,152-token BPE built so that **every student token maps to a distinct
Qwen3.5 teacher token** (injective subset projection). Heldout fertility
1.0113, 4.637 chars/token on ClimbMix — within 3.2% of a 128k reference
tokenizer at 2.6x smaller vocabulary — and 98.75% of teacher probability
mass lands in-image, which is what makes the distillation stage alignment-
free. Artifacts: [`pretraining/tokenizers/yx49k/`](pretraining/tokenizers/yx49k/).

## Pretraining campaign

367B step-tokens (~341B unique) of ClimbMix, TPU v4-64 (32 chips, 8 hosts,
pure data parallel), 524,288 tokens/step at a median 1.09M tokens/s.
Constant LR 2e-3 with cosine anneal to zero over the final 70k of 700k
steps; the anneal alone took holdout 2.5446 → 2.3494. The main W&B leg
crashed at step 678,420 and the campaign finished on a resume leg — the
overlap region reproduces the original trajectory. W&B group
`climbmix-yx49k-367b`.
Result card: [`post_training/results/pretrain-700k/SUMMARY.md`](post_training/results/pretrain-700k/SUMMARY.md).

## Post-training

SFT on the Mephisto datasets (`Yxanul/Mephisto-IF_172k` and
`Yxanul/Mephisto-Knowledge_538k` were generated on a TPU v6e slice), then
GOLD offline logit distillation from a frozen Qwen3.5-4B teacher using
per-example precomputed top-K targets. Because the tokenizer is an exact
subset of the teacher's, position i supervises position i by construction
— no sequence alignment machinery exists in the training path. Best
checkpoints: GOLD-mix300k (IFEval 51.0) and GOLD-over-SFT (best transcript
panel). External calibration against Gemma 3 270M IT and SmolLM 360M v1/v2,
all measured under this repo's own harness: capability ours, decode
discipline theirs, next stage on-policy distillation.
Full narrative: [`post_training/GOLD.md`](post_training/GOLD.md).

## Repository layout

- `pretraining/`: the standalone trainer (model, kernels, optimizers,
  losses, distillation, decode, configs, tests) — the current stack. See
  [`pretraining/README.md`](pretraining/README.md).
- `post_training/`: the SFT/GOLD experiment ledger and base-model result
  cards.
- `report/`: the technical report (LaTeX source, figures, data exports,
  built PDF).
- `docs/`: algorithm notes (KDA hybrid, block attention residuals, kernel
  audits).
- `results/`: compact metrics, benchmark summaries, eval artifacts, and
  transcript panels for every experiment. Raw XPlane traces are ignored.
- `benchmarks/`: the original MaxText-shell benchmark harness (v6e era).
- `maxtext/`: vendored [`AI-Hypercomputer/maxtext`](https://github.com/AI-Hypercomputer/maxtext)
  at commit `dfd8d293d266fe224b90f7cb0b49f3e8084e9892`, plus the
  experimental implementations developed here. MaxText retains its
  upstream Apache 2.0 license in [`maxtext/LICENSE`](maxtext/LICENSE).
- `EXPERIMENTS.md` / `AGENTS.md`: the chronological experiment ledger and
  TPU operations lore.

## Reproduce

```bash
git clone https://github.com/yxanul/yxTPU.git
cd yxTPU/pretraining
bash scripts/setup.sh            # uv sync + doctor
# single-host smoke on a v6e-8:
.venv/bin/python -m yxtpu_pretrain.run \
  --model kda_hybrid_yx49k_l20 --optimizer muonclip \
  --data climbmix_yx49k --hardware v6e-8 --experiment superbpe_smoke
```

Rebuild the technical report:

```bash
cd report && bash build.sh   # figures (if data present) + tectonic main.tex
```

The TPU quota is research-cloud capacity. Checkpoint durable training
runs and delete idle TPU resources. Never commit cloud credentials,
private keys, API tokens, or signed URLs.
