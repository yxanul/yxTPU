# GOLD distillation: experiments and results

State as of 2026-08-05. GOLD (General On-policy Logit Distillation,
HuggingFaceH4/on-policy-distillation) adapted to this project's setting:
a 308M `kda_hybrid_yx49k_l20` student distilling from a frozen
Qwen3.5-4B teacher. Everything below ran on four v4 chips of the
`yxtpu-v4-64-train` slice (single-host env carve-out) unless noted.

## The design, and why it is simpler than TRL's

General GOLD spends most of its machinery compensating for tokenizer
mismatch: ULD sort-and-pad tails, product-rule token merges, byte-offset
sequence alignment. We removed the mismatch at the source instead — the
yx49k tokenizer was authored as an exact-subset projection of the
Qwen3.5 vocabulary (`student_to_teacher` injective, ~98.8% of teacher
mass in-image). Consequences:

- **Direct-map supervision.** The teacher scores the student's own
  segmentation (`student_to_teacher[input_ids]`), so position i
  supervises position i by construction. No alignment pass exists in the
  training path; TRL issue #4393's BOS-asymmetry failure class is
  structurally impossible here. The byte-offset walker (TRL #5885's
  design) is kept only as an offline verification tool.
- **Exact residual.** The unmatched teacher tail collapses to a single
  measured bucket (reported, renormalized out, never trained against)
  instead of ULD's sorted approximation.
- **Verified against TRL source**: β endpoints, `ce_weight=0` default,
  temperature=1 all agree. One latent bug found in our interior JSD
  (teacher/student mirrored against GKD Eq. 1 — β=0.1 behaved as their
  β=0.9) — fixed and pinned by a small-β limit test before any β≠0 run.

**The λ=0 architecture: the teacher never enters the train step.**
Targets are precomputed per example — scored in isolation, positions
from zero, which also makes them permanently free of packing pollution
and recurrent-state leakage on the teacher side — compressed to top-64
ids + log-probs + one tail scalar (~172 B/position), and stored in a
hash-keyed sharded store. Training is the ordinary SFT loop with a
richer loss; a render drift between precompute and training surfaces as
`rows_missing_targets` (counted, loud), never as silently wrong
supervision. Mesh unification and reshard costs simply do not arise
until on-policy.

## Infrastructure numbers

| stage | before | after | change |
| --- | --- | --- | --- |
| teacher scorer, steady (b8×1024) | 20,346 ms | 478 ms | 42× |
| bare 4B forward (same shape) | 425 ms | 425 ms | the floor |
| precompute rate | 256 pos/s | ~8,100 pos/s | ~32× |
| 50k-example store build | (proj.) ~18 h | 34 min | — |

The whale was one op: `take_along_axis` with broadcast indices lowers to
a general gather on TPU — 11.9 s per operand vs 11.8 ms for `jnp.take`
with the constant index vector (values identical). Everything else —
`approx_max_k` over exact `top_k` (514→6 ms at 98% recall, misses land
in the tail bucket by construction), per-block fp32 casting in the
blockwise logsumexp, batch-over-tensor resharding before the projection
— was real but minor beside it. One negative result worth keeping: a
split-once `jax.jit` teacher scorer made MaxText's layer scan carry the
parameters unsharded (batch-independent 21.5G program); `nnx.jit`'s
per-call walk measures 4.4 ms and is paid knowingly.

## Experiments and results

All IFEval rows: 541 prompts, 0-shot, T0.3/p0.9/pen1.1 chat-template
decode unless noted. Student initialized from `sft-mephisto-v2/1200`
(gen-2 SFT) in every GOLD run; optimizer always fresh muonclip at LR
3e-5 annealed over the true data-limited horizon.

### Off-policy probe (pre-training sanity, 20 steps forward-only)

Teacher@1 = 78.8% on Mephisto-IF (an empirical alignment gate — a
one-position shift would send it to ~0), residual 3.3%, distill 1.13
nats of signal CE cannot see. Machinery validated end to end.

### GOLD-10k and GOLD-50k (IF prompts only)

| IFEval | gen-2 base | +GOLD-10k | +GOLD-50k |
| --- | --- | --- | --- |
| prompt strict | 30.9 | 32.2 | **40.1** |
| prompt loose | 35.1 | 37.2 | **43.4** |
| instruction strict | 45.4 | 47.2 | **53.4** |
| instruction loose | 50.2 | 51.7 | **56.7** |
| mean | 40.4 | 42.1 | **48.4** |

50k = 759 steps / 29.8M supervised tokens / ~7 minutes of training.
Scaling is super-linear in this range (+1.7 mean at 10k → +8.0 at 50k,
same baseline, same recipe). Instruction-loose clears the 51.2 Gemma 3
IT 270M reports. Training internals for the 50k run: distill 1.17→0.85,
CE 1.55→1.36 with ce_weight=0 (never trained, improved anyway — the
GOLD mechanism in one line), student@1 62→66%, rest mass flat at 4.05%
with zero drift, teacher@1 pinned at 83.8%, `rows_missing_targets` 0
across all 29,955 draws. For scale: the previous-generation ultradata
SFT measured 23.1 mean on this benchmark.

### Transcript panel and decode sweep (attribution)

42-prompt scored panel, gen-2 vs GOLD-50k at identical settings:
instruction 0/8 → 2/8, clean `<|im_end|>` stops 2/42 → 16/42, knowledge
7/8 both — and math/code 0/8 **both**: the arithmetic hole is inherited
from gen-2, not caused by IF-only distillation. Decode sweep on
GOLD-50k: pen1.0 wins the short-form panel (17/42, knowledge 8/8);
penalties 1.3/1.5 loop worse; at IFEval scale the decode effect washes
out (GOLD-50k 47.6@pen1.0 vs 48.4@pen1.1) and the GOLD gain is robust
across settings. Verdict: pen1.0 for interactive short-form; data
domains, not decode knobs, for math.

### Third-party math/code data (the regeneration question)

Mephisto-MathCode_2M's completions came from Qwen3/DeepSeek V3, not the
teacher. 2k-row smoke, measured on exactly the positions training
supervises: **teacher@1 91.9%** (Math 92.9 / Code 89.2) — above the 84%
the teacher scores on its own IF generations, because math/code text is
low-entropy and argmaxes coincide across strong models — rest mass
0.86%. Training smoke reproduced both numbers in-loop and optimized
cleanly. Conclusions: no teacher-regeneration needed, and no separate
SFT stage either — λ=0 GOLD trains on the same rendered rows SFT would,
with a denser signal (`--gold-ce-weight` can blend hard-label CE back
at will). Operational: rows without a `system` column make the
`--system` flag load-bearing (the hash guard caught the mismatch by
dropping 100% of rows rather than training wrong — its job), and 14% of
MathCode rows exceed the 2048-token window.

### GOLD-mix300k: 100k IF + 100k Knowledge + 100k MathCode

Precomputed via the first 8-way host fan-out (each worker converted its
own teacher from HF and built 1/8 of the store: 281,329 examples /
~146M positions / 25 GB in ~45 min at ~54k pos/s aggregate), trained as
one stationary probability-weighted interleave: 4,449 steps / 182.9M
supervised tokens / 2 epochs / ~40 min. Internals: distill 0.98→0.62,
CE 1.21→0.92 untrained, teacher@1 90.0% (the mix is easier for the
teacher than pure IF), rest mass 1.8%, 26 missing rows in 562,826
draws.

| IFEval | gen-2 | +50k IF | **+mix300k** |
| --- | --- | --- | --- |
| prompt strict | 30.9 | 40.1 | **42.9** |
| prompt loose | 35.1 | 43.4 | **45.5** |
| instruction strict | 45.4 | 53.4 | **56.4** |
| instruction loose | 50.2 | 56.7 | **59.1** |
| mean | 40.4 | 48.4 | **51.0** |

The compositionality result matters most: mix300k beats the pure-IF 50k
run on IFEval even though two-thirds of its training was non-IF data —
the mixed diet helped the IF metric more than more IF did. Mean 51.0
now sits at Gemma 3 IT 270M's reported 51.2.

42-prompt panel (gen-2 → 50k → mix300k): total 10 → 12 → **16**/42;
code 0 → 0 → **2**/8 (the first code passes in the lineage); math 0 →
0 → 1/8; if 3 → 3 → 5/10; knowledge 7/8 held; clean `<|im_end|>` stops
2 → 16 → **33**/42.

GSM8K (500-question slice, 5-shot): gen-2 2.0, mix300k 1.8 — both at
the noise floor; the mix did NOT move formal chain arithmetic. Honest
read: ~86k math/code examples × 2 epochs teaches surface competence
(panel movement, code passes) but not GSM8K at 308M — SmolLM2-360M
reports 3.2 after 4T tokens. The levers are scale (the full 2M MathCode
set) and sequence length: the 2048-token window drops 14% of MathCode
rows, and those are precisely the longest derivations.

## Verdict and next campaign

Distillation scales cleanly through 300k examples with zero stability
incidents across every run. The recommended next rung: full-set GOLD
(172k IF + 538k Knowledge + a MathCode slice sized to taste) — the
fan-out prices the full 2.7M-example store at ~7 h on this slice
(hardware the TRC grant covers; ~180 GB of store means disk planning
first — checkpoint save_interval sized against free space, the ENOSPC
lesson from this run's first attempt); a 4096 window for the MathCode
share; then on-policy as the stage after.

## Ledger

- Store cost: ~172 B/position at K=64 (fp16 logprobs, uint16 ids,
  compressed npz). 300k examples ≈ ~150M positions ≈ ~25 GB.
- Precompute: ~8,100 pos/s per 4-chip host; scales linearly with hosts.
- Training: ~520 ms/step at 65,536 tokens/step on 4 chips.
- Full eval loop (IFEval): ~2.5 min.
