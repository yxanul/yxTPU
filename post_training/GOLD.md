# GOLD distillation: experiments and results

State as of 2026-08-07. GOLD (General On-policy Logit Distillation,
HuggingFaceH4/on-policy-distillation) adapted to this project's setting:
a 308M `kda_hybrid_yx49k_l20` student distilling from a frozen
Qwen3.5-4B teacher. Early experiments ran on four v4 chips of the
`yxtpu-v4-64-train` slice (single-host env carve-out); from SFT-6M
onward, training runs use the full 8-host slice.

The lineage at a glance (every checkpoint in the private HF repo
`Yxanul/yx49k-l20-checkpoints`, 19.9 GiB verified):

| model | recipe | IFEval mean | panel /42 |
| --- | --- | --- | --- |
| gen-2 | base@700k + SFT 690k IF+Knowledge rows @2048 | 40.4 | 10 |
| GOLD-10k | gen-2 + distill 10k IF | 42.1 | — |
| GOLD-50k | gen-2 + distill 50k IF | 48.4 | 12 |
| GOLD-mix300k | gen-2 + distill 100k×{IF,Knowledge,MathCode} | **51.0** | 16 |
| SFT-6M | base@700k + SFT 6.27M rows @4096 | 35.3 | 15 |
| GOLD-over-SFT | SFT-6M + distill 1/3 of the same mix, K=32 | 41.3 | **22** |
| (2-epoch control) | same, 2 epochs | 40.3 | 22 |

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

### Base-capability check (the LR question)

0-shot loglikelihood panel vs the base model at pretrain step 700k:
mix300k drifts −1.2 mean with seven of nine tasks within ±1.5 (boolq
and copa UP); SFT-6M likewise −1.5. A hot LR degrades broadly — nothing
here does — so LR 3e-5 is confirmed safe from the 30M-token GOLD dose
through the 2.55B-token SFT dose. The one systematic cost is
**lambada** (47.4 base → 40.1 mix300k / 42.4 SFT-6M / 38.7 GOLD-over-
SFT): the classic chat-tuning tax on raw-prose continuation,
distribution shift rather than optimization damage, compounding with
each chat-format stage.

### SFT-6M: 6.27M rows, one epoch, full slice

The plain-SFT baseline the GOLD numbers needed: AMD-SFT-Mix_3.5M +
Mephisto IF/Knowledge/MathCode (6,268,050 rows) exported to tmpfs,
globally shuffled with `shuf`, split round-robin into 8 per-host
shards, trained @4096 on all 8 hosts from the 700k base — 4,734 steps /
1.62B supervised tokens / ~41 min, loss 1.76→0.79.

Verdict: IFEval 35.3 (the 88%-code/math diet diluted IF to ~3% of
rows), but the lineage's best raw code/math surface skill (panel code
4/8, math 2/8) and clean stops 35/42. **300k distilled rows beat 6.27M
one-hot rows by +15.7 IFEval mean** — a 21× data disadvantage overcome
by the denser signal (pipeline comparison, not a controlled ablation:
inits differ).

### GOLD-over-SFT: the synthesis

K=32 targets for the first third of the same shuffled mix (per-host
sharded stores, ~9.1 GB each, built in ~3.5 h wall by all 8 hosts;
`rows_missing_targets` = 1 in 243,504 training draws — the
store/render identity holds at full scale), distilled onto the SFT-6M
checkpoint: 1,575 steps / 539M tokens / ~15 min.

**Best model of the lineage: panel 22/42** — math 4/8 and code 4/8
kept from SFT, first perfect knowledge 8/8, IFEval +6.0 over its base.
The telling internal: CE stayed flat at the SFT optimum while distill
fell 31% — the teacher's distributions carried signal beyond what
another one-hot epoch could give, which is the GOLD thesis in one
number. Its IFEval (41.3) trails mix300k's 51.0 for a data-mix reason
(3% IF share here vs 33% there), not a method reason.

Qualitatively (12-prompt chat panel, both models 9/12): the mechanics
arrived — `is_even` textbook, planets correct with no Vulcan, 15% of
240 computed correctly as 36 (then spoiled by adding it back: the
arithmetic engine works, the task model misfires — a different failure
class than mix300k's "15% of 24000 is 3200"). Still missing:
multi-step word problems, translation (absent from every diet), and
open-ended generation stability.

### Epoch scaling: a clean null

Controlled comparison — identical init, store, and single-anneal
recipe; 1 epoch vs 2: distill floor 0.504→0.480, panel 22/42→22/42,
IFEval 41.3→40.3. The second epoch memorized the store without moving
anything downstream. **Standing rule: one epoch per store; spend the
next unit of compute on fresh rows.**

### Decode settings are checkpoint-dependent

Every generation got its own 10-config sweep on the 42/12-prompt
panels. GOLD-50k wanted pen1.0 (penalties looped it worse); GOLD-over-
SFT wants pen1.15 for chat (kills the repetition attractor, ties the
best score, cleanest stops) while low temperature — not high — is its
loop-maker (T0.2: worst). Both models: penalties tax IFEval by ~1
point. The profile that generalizes: **interactive chat at
T0.3/p0.9/pen1.1–1.15; benchmarks and long-form at pen1.0–1.1; re-sweep
after every training generation** (~10 min).

### The Gemma 3 270M IT head-to-head (external calibration)

Same 54 prompts + a 12-prompt freeform set + IFEval + the loglik panel
through both models, checkers identical, Gemma at its better decode per
panel (`results/gemma-vs-gold/COMPARISON.md` has the full transcripts).
Result: **capability ours, behavior Gemma's, strict-format neither's.**
Measured loglikelihood is a 4/4 sweep for GOLD-over-SFT (hellaswag
+15.8, arc-c +9.3, piqa +7.2, winogrande +5.7) despite Gemma's ~120×
pretraining-token advantage — the base is not the problem. Transcripts
invert it: panel 31/42 Gemma vs 22/42 ours, driven entirely by math
(8/8 vs 4/8, their drilled `Final Answer` ritual vs our
right-number-then-self-destruct pattern) and code (7/8 vs 4/8); chat
9/12 ours vs 8/12. On the freeform set Gemma's answers are usable
as-is (~8/12) because they *terminate*; ours open better (best prose of
either model) and then loop — only 5/12 emitted `<|im_end|>` in 400
tokens. Instruction-format is the shared floor: Gemma 2/8, ours 3/8 —
6T tokens and Google's whole IT pipeline don't buy all-lowercase or
no-letter-e at ~300M. IFEval under our harness: Gemma measures 31.4
mean (its published 51.2 does not survive harness transfer) vs our
41.3 (GOLD-over-SFT) and 51.0 (mix300k) — both checkpoints beat it
under identical conditions. Attribution: the entire deficit is termination
discipline and answer ritual — post-training properties — and the
right-answer-then-drift signature is textbook exposure bias, the
failure class on-policy distillation exists to fix.

## Verdict and next campaign

Distillation scales cleanly through 300k examples and stacks on top of
large-scale SFT, with zero numerics incidents across every run. The
epoch null plus the compositionality result fix the campaign shape:
one epoch, mixed domains, fresh rows over repeats. The Gemma
head-to-head adds the external calibration: the base wins on capability
at 120× less data, so pretraining, size, and the pretraining diet stay
fixed; every next unit of compute belongs to post-training. Recommended
next rungs, in expected-value order: (1) an IF-weighted distillation
pass on the GOLD-over-SFT checkpoint (its one trailing metric is a mix
artifact); (2) the untouched two thirds of the K=32 mix store's data;
(3) the full campaign — 172k IF + 538k Knowledge + MathCode sized to
taste at K=32 with 4096 buckets (~15–20 h fan-out precompute, ~250–450
GB across `/mnt/ram`, per-host sharded stores, no gather), over-sampling
short-form completions with hard stops and boxed-answer math rows; then
on-policy — promoted from "the stage after" to the indicated treatment
for the observed disease (loops and post-answer drift are exactly what
student-sampled rollouts + teacher scoring penalize). Re-run the Gemma
comparison after on-policy; revisit model size only if the spiral
survives it.

## Operations ledger

- Store cost: ~172 B/position at K=64, ~90 B at K=32 (fp16 logprobs,
  uint16 ids, compressed npz). K=32 validated end-to-end; rest mass
  1.5% on the mixed diet.
- Precompute: ~8,100 pos/s per 4-chip host, linear across hosts (54k
  pos/s on the slice). Buckets past 2048 need the halved flush batch
  (the [8,4096] scorer program does not fit alongside the smaller
  buckets' resident programs).
- Training: ~520 ms/step at 65,536 tokens/step (4 chips, @2048); ~520
  ms/step at ~340k tokens/step (8 hosts, @4096). Host RAM never
  exceeded 53 GB of 400 (W&B system metrics), so tmpfs is the data
  plane: **use `/mnt/ram` (plain tmpfs), not `/dev/shm`** — systemd
  logind's RemoveIPC purges /dev/shm when the owning session ends,
  which silently deleted a distributed dataset once.
- Multi-host data-limited runs need COLLECTIVE termination (allgather a
  has-data flag; c3a1d6d): per-host shards exhaust at different step
  counts, and one host breaking alone deadlocks the rest in the next
  collective — cost SFT-6M its final save before the fix.
- Per-host sharded stores align with per-host data shards by
  construction; `rows_missing_targets` is the guard and measured ≤2
  per run at every scale.
- Eval battery per checkpoint: IFEval ~2.5 min, 42-prompt panel ~30 s,
  9-task loglikelihood panel ~12 min, decode sweep ~10 min.
- Checkpoint custody: every lineage checkpoint + metadata in
  `Yxanul/yx49k-l20-checkpoints` (HF, private) — pickles are
  irreplaceable and belong on durable storage; stores are recomputable
  and belong in RAM.
