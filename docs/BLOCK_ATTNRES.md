# BlockAttnRes: a block-level attention residual for the KDA hybrid

Status: design + hypothesis. `model.residual_policy=block_attnres` is reserved
in config and rejected at validation until the experiment described here runs.
The `ATTNRES_PSEUDOQUERY` optimizer role (AdamW-routed) is already declared.

## The problem it targets

The hybrid's cycle gives each block exactly one exact-retrieval channel: three
KDA layers hold a fixed-size fast-weight state (fading, associative), and one
NoPE-GQA layer does unbounded-range softmax retrieval. Information the global
layer retrieves reaches other layers only through the token-wise residual
stream — compressed into one hidden vector per token, then immediately mixed
by MLPs and further KDA updates.

The SuperBPE-1B campaign (W&B group `superbpe-1b`, 2026-07-22) measured the
cost of that bottleneck directly: the KDA hybrid beat the pure-GQA transformer
on loss and on 5/10 zero-shot tasks, but lost exactly one clearly — lambada
(0.112 vs 0.127 acc), the most retrieval-bound metric in the suite. The
hybrid's weakness is not knowledge or throughput; it is precise long-range
recall density.

## Mechanism

Treat the four-layer cycle as one unit with a single shared full-attention
resource ("3 KDA + 1 global attention combined into one block"):

**Variant A — output re-injection (cheap ablation).** Carry the block's GQA
mixer output forward and add it, through per-layer learned gates initialized
at zero, to the input of each KDA mixer of the next block. No new attention
math; three gate vectors per block. Crosses the scan boundary, so the scan
carry becomes `(hidden, attn_out)`.

**Variant B — shared-KV pseudo-query reads (the pseudoquery design;
recommended).** Reorder the cycle to `[gqa, kda, kda, kda]` so the block's
exact retrieval happens first. The GQA layer exposes its (post-RoPE/NoPE)
K and V; each of the three KDA layers adds a thin causal softmax read of that
same K/V, using a small learned pseudo-query projection:

    kda_out_i  +=  gate_i * softmax(PQ_i(x_norm) K^T / sqrt(d)) V

- `PQ_i`: pseudo-query projection, `read_heads` (default 2) heads against the
  block's 2 KV heads — role `ATTNRES_PSEUDOQUERY`, deliberately AdamW-routed
  (no Newton-Schulz on these thin matrices).
- `gate_i`: per-layer scalar or per-head vector, initialized 0 — the block is
  bit-identical to a plain `[gqa, kda, kda, kda]` stack at init.
- K/V are computed once per block by the GQA layer and reused three times:
  zero additional KV memory, no extra KV projections.

Everything stays within one block, so the scan carry is untouched — this is
the decisive engineering advantage of B over A.

## Hypothesis and falsifiable predictions

Hypothesis: giving every KDA layer a thin exact-retrieval channel over the
block's shared KV recovers the hybrid's retrieval deficit at negligible cost,
because retrieval capacity in the hybrid is limited by access density to
softmax attention, not by attention FLOPs.

Predictions at 1B tokens under the campaign protocol (constant LR 3e-4,
muonclip, tied SuperBPE-128k):

1. lambada acc >= 0.127 (the pure-GQA level), ppl < 6,500 — the primary gate.
2. Train/holdout loss <= run 3's 3.872/3.929; a regression > 0.02 nats
   rejects the mechanism regardless of eval wins.
3. Gradient-norm profile unchanged (zero-init gates make the start exactly
   the control model; no new instability surface).
4. Step-time overhead <= 3%. EXP-011 profiled Splash attention at 0.84% of
   the hybrid step, so even ~1.75x more attention math per block is noise
   against the KDA-dominated step.

Secondary, mechanistic: the GQA layer's attention-logit maxima (already
recorded for QK-Clip) should drift toward retrieval specialization once KDA
layers can reuse its work — a free observable in existing telemetry.

## Experiment design: three arms, not two

The reorder to `[gqa, kda, kda, kda]` is itself an architecture change, so a
two-arm A/B confounds ordering with the residual. Run three:

1. Certified order `[kda, kda, kda, gqa]` — exists (campaign run 3).
2. Reordered control `[gqa, kda, kda, kda]`, gates absent — isolates the
   ordering effect.
3. Reordered + BlockAttnRes — the mechanism.

Each arm is ~35 minutes on the current v4-64; total cost under two hours.

## Implementation map

- `config.py`: accept `block_attnres`; add `attnres: {read_heads: 2,
  gate_init: 0.0}` under ModelConfig; require the reordered cycle when the
  policy is active; flip `test_block_attnres_is_reserved_but_rejected` to an
  acceptance test.
- `config.py` allowed cycles: add `("gqa", "kda", "kda", "kda")`.
- `layers/nope_gqa.py`: optional `return_kv` — `_project` already computes
  K/V; return them post-RoPE.
- `layers/attn_res.py` (new, small): `PseudoQueryRead` module — pseudo-query
  DenseGeneral (role `ATTNRES_PSEUDOQUERY`), zero-init gate, plain causal
  einsum attention (2 read heads do not need Splash; revisit only if
  profiling disagrees).
- `model.py`: policy-aware `HybridCycle.__call__` — GQA layer first, thread
  (k, v) to the three KDA layers, add gated reads after each KDA mixer. Fix
  the hardcoded `for index in range(4)` while there.
- Roles/routing: pseudo-queries and gates route to AdamW via the existing
  `ATTNRES_PSEUDOQUERY` role; no Muon dimension numbers needed.
- Tests: gate-zero bit-equivalence against the reordered control; parameter
  count delta (3 blocks' worth of PQ + gates); role-coverage assertion.

## Risks

- Ordering regression: if arm 2 loses to arm 1, the mechanism must beat that
  loss too — the three-arm design surfaces this instead of hiding it.
- Muon interaction: pseudo-queries under AdamW while surrounding matrices use
  Muon is a deliberate routing decision (already encoded in the role tables);
  if gates stay near zero at 1B, check them before concluding the mechanism
  is useless — short runs may under-train thin residual paths.
- KDA precision: reads add FP32 softmax outputs into a bf16 residual path;
  cast at the gate, as the GQA layer already does for its own output.
