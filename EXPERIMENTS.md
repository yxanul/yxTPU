# TPU experiment ledger

This is the durable index for experiments performed in this workspace. Every
benchmark should record its exact workload, comparison intent, result directory,
and status here. Detailed measurements and profile interpretation live in
[`results/RESULTS.md`](results/RESULTS.md).

## Environment

- TPU VM: `yxtpu-v6e8-dev`
- Accelerator: one Spot `v6e-8` slice (8 chips)
- Zone: `europe-west4-a`
- MaxText commit: `dfd8d293d266fe224b90f7cb0b49f3e8084e9892`
- JAX / jaxlib: 0.10.2 / 0.10.2
- libtpu: 0.0.42.1
- Flax: 0.12.7
- Optax: 0.2.8
- Unless noted otherwise: synthetic reused tokens, sequence length 2048,
  eight-way data parallelism, BF16 compute, FP32 master weights, no
  checkpointing, and five warmup steps excluded.

## Completed experiments

| ID | Experiment | Principal result | Artifacts |
| --- | --- | --- | --- |
| EXP-001 | Original 271.6M model, AdamW, batch 8/chip | 1,014,920 global tokens/s; 14.8 GB/chip | `results/v6e8-llama-272m-s2048-b8-20260720T104542Z/` |
| EXP-002 | Original 271.6M model, AdamW, batch 16/chip | 1,100,290 global tokens/s; 24.6 GB/chip | `results/v6e8-llama-272m-s2048-b16-20260720T104714Z/` |
| EXP-003 | Qwix INT8 smoke test on EXP-002 shape | Functional, but 846,059 tokens/s and 24.0 GB/chip | `results/v6e8-llama-272m-qwix-int8-s2048-b16-20260720T105941Z/` |
| EXP-004 | XPlane profile of EXP-002 | Splash Attention 56.915 ms/step (23.95%) | `results/v6e8-llama-272m-bf16-profile-s2048-b16-20260720T110058Z/` |
| EXP-005 | Modern 270.0M, 18-layer fused-GQA/SwiGLU model, AdamW | 1,082,149 tokens/s; 23.0 GB/chip | `results/v6e8-modern-270m-adamw-s2048-b16-20260720T111528Z/` |
| EXP-006 | Tokamax fused-backward flag A/B on EXP-005 | +0.008%; no effect because Tokamax forces fused backward | `results/v6e8-modern-270m-adamw-fusedattnbwd-s2048-b16-20260720T111652Z/` |
| EXP-007 | Modern 270.0M with Muon matrices + AdamW remainder | 955,500 tokens/s; 21.4 GB/chip | `results/v6e8-modern-270m-muon-s2048-b16-20260720T111817Z/` |
| EXP-008 | XPlane profile of EXP-007 | Muon 32.483 ms/step; Tokamax Splash 64.636 ms/step | `results/v6e8-modern-270m-muon-profile-s2048-b16-20260720T111937Z/` |
| EXP-009 | Implement KDA from MaxText GDN and validate against a recurrent reference | Three CPU tests cover forward, final state, all gradients, and overflow stress | `docs/KDA_HYBRID.md`; `maxtext/tests/unit/kimi_delta_attention_test.py` |
| EXP-010 | 272.9M 3:1 KDA/NoPE-GQA hybrid, matched at batch 8/chip | 156,290 tokens/s vs 1,003,900 control; 22.9 vs 14.0 GB/chip | `results/v6e8-kda-hybrid-273m-block8-safe-adamw-s2048-b8-20260720/`; `results/v6e8-modern-270m-adamw-matched-s2048-b8-20260720/` |
| EXP-011 | XPlane profile of EXP-010 | KDA 88.75%; WY solve 22.53%; decay block math 37.62%; Splash only 0.84% | `results/v6e8-kda-hybrid-273m-block8-safe-profile-s2048-b8-20260720/` |
| EXP-012 | Audit SGLang-JAX's TPU Pallas KDA kernel | Useful forward-solve design, but forward-only: 425,307 tok/s in the orientation microbenchmark and no reverse-mode rule | `results/v6e8-sglang-jax-kda-forward-audit-20260720/`; `docs/KDA_HYBRID.md` |
| EXP-013 | Add a 16-row Pallas blocked solve and exact custom VJP, validate it, and retest the 272.9M hybrid | Correct on native TPU, but 89,649 tok/s: 42.64% below the selected XLA path; custom-call HLOs use 55.58% of the step | `results/v6e8-kda-triangular-solve-ab-20260720/`; `results/v6e8-kda-hybrid-273m-pallas-vjp-adamw-s2048-b8-20260720/`; `results/v6e8-kda-hybrid-273m-pallas-vjp-profile-s2048-b8-20260720/` |
| EXP-014 | Audit and benchmark ejkernel's XLA GDR forward and handwritten backward as a KDA starting point | At chunk 64 the active scalar GDR core reaches 6.06M tok/s forward and 2.25M tok/s training; its custom VJP is 36.57% slower in training. Current KDA is 4.48x slower than scalar GDR through backward | `results/v6e8-ejkernel-gdr-xla-audit-20260720/`; `docs/EJKERNEL_GDR_TO_KDA.md` |
| EXP-015 | Implement a whole-KDA analytical XLA VJP with a blockwise channel-decay derivative, validate all six gradients, and retest the 272.9M hybrid | Core training is 2.57% faster and uses 70.82% less compiled memory; the full model reaches 186,815 tok/s (+19.53%) at 17.9 GB/chip. The profile places the 137.31 ms step reduction entirely in backward | `results/v6e8-kda-analytical-vjp-core-20260720/`; `results/v6e8-kda-hybrid-273m-analytical-vjp-adamw-s2048-b8-20260720/`; `results/v6e8-kda-hybrid-273m-analytical-vjp-profile-20260720/` |
| EXP-016 | Build a production-shape fused TPU Pallas KDA forward and matched backward, with FP32 state/accumulation and BF16 Q/K/V traffic | Native TPU correctness passes against the recurrent and analytical-XLA references. Core training reaches 895,927 tok/s (+74.29%) at 0.883 GB compiled memory (-54.46%); the 272.9M hybrid reaches 304,300 global tok/s (+62.89%) at 15.4 GB/chip | `results/v6e8-kda-fused-pallas-core-20260720/`; `results/v6e8-kda-hybrid-273m-fused-pallas-adamw-s2048-b8-20260720/`; `results/v6e8-kda-hybrid-273m-fused-pallas-profile-20260720/` |
| EXP-017 | Selectively retain the fused custom-VJP residuals across MaxText's outer decoder remat | Rejected: 297,925 tok/s (-2.10% versus EXP-016) and 20.4 GB/chip (+5.0 GB). The original `minimal_with_context` policy remains selected | `results/v6e8-kda-hybrid-273m-fused-pallas-remat-adamw-s2048-b8-20260720/` |
| EXP-018 | Collapse the fused Pallas grid so one program advances eight streams instead of one | The grid falls from 2,048 to 256 programs and the core reaches 12.577 ms forward+backward; the model reaches 398,061 tok/s (+30.8%) with an unchanged loss curve. `streams_per_program` above 8 fails with `CompileTimeScopedVmemOom` | `results/v6e8-kda-hybrid-273m-fused-collapsed-h8-s2048-b8-20260720/`; `results/v6e8-kda-diag-chighest-shighest-20260720/` |
| EXP-019 | Reduce in-kernel MXU precision from six-pass FP32 to one-pass BF16 | A blanket reduction is 4.93x faster in the core but diverges to NaN at model step 2. Bisecting by matmul role isolates the repeated-squaring triangular solve as the sole cause: guarding only the solve trains normally at 560,923 tok/s (+84.3%), while guarding only the pairwise still diverges. Q/K/V arrive in BF16, so elsewhere the extra passes refined mantissa the operands never carried | `results/v6e8-kda-solveguard-20260720/`; `results/v6e8-kda-hybrid-273m-selected-s2048-b8-20260720/` |
| EXP-020 | Split the solve series by matmul role to recover the guarded cost | Rejected. Applying a power looks additive but the update is `solution <- (I + P^(2^k)) solution`, so the running solution compounds exactly as the power does; BF16 applications still reach NaN at step 2. The whole solve needs six passes, and the FP32 solve is 2.73 ms of the 6.437 ms core | `results/v6e8-kda-solveapply-bf16-20260720/` |
| EXP-021 | Replace the doubling solve with blocked forward substitution to make it BF16-safe | Rejected on throughput, not on numerics. The first attempt was invalid: both custom-VJP call sites hardcoded `solve_method="doubling"`, so the forward never used the selected method and the base case was itself a nilpotent series. Rerun correctly, substitution is 100-350x more accurate under one BF16 pass, but a row-serial base case costs 8.617 ms against 6.437 for guarded doubling, and a 16x16 full-precision base case with BF16 coupling is 6.111 ms in the core yet 548,450 tok/s against 560,923 in the model | `results/v6e8-kda-substitution-bf16-20260720/`; `results/v6e8-kda-subst-hybrid-20260720/` |
| EXP-022 | XPlane profile of the selected 560,923 tok/s configuration | The 233.564 ms step is 86.597 ms fused KDA kernels (37.08%), 62.080 ms convolution fusion (26.58%), 41.591 ms loop fusion (17.81%), 19.768 ms data formatting (8.46%), and 6.958 ms Splash (2.98%) | `results/v6e8-kda-selected-profile-20260720/` |
| EXP-023 | Replace the depthwise QKV convolution with shifted multiply-accumulates | Rejected: 537,292 tok/s against 560,919, 4.2% slower, though exactly equivalent at 2.4e-7 and causality-checked. The convolution-fusion profile category also covers the SiLU and reshapes XLA fused into it, so it overstates the convolution; the rewrite adds four full passes over the QKV tensor plus a pad | `results/v6e8-kda-shiftedconv-1-20260720/`; `results/v6e8-kda-shiftedconv-0-20260720/` |
| EXP-024 | Re-test remat policies now that the kernel is 2.8x cheaper and the step uses 15.2 of 31.25 GB | `minimal_with_context` stays selected at 560,919 tok/s. `minimal` is 533,080 and `save_dot_except_mlp` is 550,776, but the latter uses 9.2 GB rather than 15.2, trading 1.8% throughput for 6 GB of headroom | `results/v6e8-kda-remat-minimal-20260720/`; `results/v6e8-kda-remat-save_dot_except_mlp-20260720/` |
| EXP-025 | Spend the freed memory on batch, since the profile's loop-fusion and data-formatting time is largely batch-independent | At batch 16/chip `save_dot_except_mlp` reaches 582,117 tok/s at 13.3 GB, 3.8% above the batch-8 selected point; batch 24 regresses to 578,758 at 17.4 GB. This is a separate operating point, not a matched-batch comparison | `results/v6e8-kda-sdem-b16-20260720/`; `results/v6e8-kda-sdem-b24-20260720/` |
| EXP-026 | Separate gradient accumulation from batch scaling | At a matched effective batch of 16, accumulation costs 1.2% (575,364 against 582,117), as expected for an overhead that does not change the update. It wins only by reaching effective batches that do not fit directly: microbatch 16 with `ga=8` reaches 602,362 tok/s at 15.4 GB, against 578,758 for a direct batch of 24 at 17.4 GB. The gain saturates (+2.5%, +0.7%, +0.3% for ga 2, 4, 8) and memory is flat in `ga`, consistent with amortizing the 13.3 ms optimizer and all-reduce tail rather than anything that scales. Microbatch 16 is optimal at every `ga` | `results/v6e8-kda-sdem-b16ga*-20260720/`; `results/v6e8-kda-sdem-b8ga2-20260720/` |
| EXP-027 | Separate WY problem conditioning from recursive-doubling growth, using real TPU arithmetic | The two are complementary: growth predicts the solve's backward error, `kappa_2(I+A)` predicts its amplification into forward error, and growth dominates in the regimes of interest. The positive all-ones extreme has `kappa_2` 82.1 and `max abs inverse` exactly 1, a benign problem, yet forms powers of norm 6.17e17 and returns 1.7e15 forward error at one BF16 pass. Stress regimes sit at `kappa_2` 25 to 70 with growth 1e15 to 1e17. Conditioning still matters: on the ill-conditioned extreme substitution reaches 5.9e-19 backward error and still carries 3.2e-3 forward error. An earlier version of the script rounded operands once and then computed in FP32, understating the failure by three orders of magnitude in one regime | `benchmarks/diagnose_wy_conditioning.py` |
| EXP-028 | Matched sequence-length sweep from 2k to 32k, attention control against the 3:1 KDA hybrid | Throughput crossover is near 16,384: attention leads 1.87x at 2,048 and 1.31x at 8,192, the hybrid leads 1.03x at 16,384 and 1.57x at 32,768. Attention sheds 14/22/32/41% per doubling as the quadratic term takes over while KDA sheds 2/4/8/10%, tracking the shrinking batch rather than sequence cost. Memory is not a differentiator in training, within 1 GB at every length, because Splash is flash-style and linear in sequence length. Batch is confounded with length, since holding tokens per step fixed forces batch from 8 to 1 | `results/v6e8-seq-*/` |
| EXP-029 | Restore MaxText's logical activation-sharding contract in the standalone scanned model | The scan primitive was not the gap: both paths use merge-inside `lax.scan`. The old standalone trace instead has 256 exact all-gathers across two steps, including MLP activations expanding from batch 4 to 32; the corrected trace has zero. At matched batch 4 throughput rises 69.1%, from 266,482 to 450,508 tok/s. Batch 8 reaches 545,495 tok/s, 97.25% of the historical 560,923 MaxText selected point. A second bug divided the configured microbatch by GA; after correcting it, microbatch 16 with GA=8 processes the intended 2,097,152 tokens/update and reaches 598,517 tok/s, 99.36% of the historical 602,373 result | `results/v6e8-standalone-sharding-parity-20260721/` |
| EXP-030 | Add an explicitly data-parallel Tokamax fused linear cross-entropy and sweep batch capacity | Eight-device loss, `dx`, `dw`, and full 272.9M AdamW one-step parity pass, including uneven padding with one wholly masked device and edge vocabulary labels. The fused path is not selected for throughput: at batch 8 it is 1.53% slower and its compiled estimate is 1.04 GB larger; at microbatch 16/GA=8 it is 2.60% slower but saves 5.11 GB (13.8%). At batch 32/GA=4 it remains 1.67% slower than standard and saves only 0.775 GB. With full rematerialization it narrowly unlocks batch 64/GA=2 at 550,039 tok/s where standard loss exceeds available HBM by 84.6 MB. Standard batch 16/GA=8 remains selected at 598,543 tok/s. The tested all-reduce XLA-flag bundle is rejected: 0.72% slower and 5.63 GB larger | `results/v6e8-fused-linear-ce-20260721/` |
| EXP-031 | Build and validate the real-data ClimbMix 10B training stack | The 309,111,392-parameter GPT-2-vocabulary model streams and tokenizes ClimbMix fast enough to stay ahead of the TPU, with a deterministic 1% held-out partition and three-batch prefetch. Standard loss at microbatch 16/GA=8 fails compilation at 36.29 GB of temporaries; Tokamax fused loss fits with a 31,989,071,680-byte compiled estimate and sustains 566,328 tok/s after warmup across real streamed updates. The full eight-device smoke passes one optimizer step, held-out loss, a separate finite stability/attention diagnostic, all ten requested lm-eval tasks, result serialization, and W&B artifact upload. The 10B profile is explicitly non-resumable and reaches its budget in 4,769 updates | `results/v6e8-climbmix-10b-smoke-20260721/`; W&B run `raxd2gkf` |
| EXP-032 | Gate the fused KDA backward on exact real-text trigger batches | The 566k guarded-Pallas result from EXP-031 is rejected for training. With effectively frozen weights, real ClimbMix updates produced repeatable gradient spikes (446.8 at update 7); at the exact offending microbatch the guarded backward gives norm 3,933.7 and max element 171 against 2.407 and 0.0545 under the full-FP32 reference. A normal run becomes non-finite at step 12. Individual and all-role six-pass promotion, row blocks 2/4/8, and midpoint anchoring all fail, while forward losses stay close, isolating the fused backward. Full-FP32 analytical VJP stays finite through the trigger and 15 training steps at 172,961 tok/s, 9.0% faster than generic XLA autodiff. Real training now rejects guarded Pallas and fails fast on non-finite metrics | `results/v6e8-climbmix-realtext-precision-20260721/` |
| EXP-033 | Keep attention-logit diagnostics batch-independent across donated NNX train state | The step-250 failure was a telemetry shape leak, not a model failure: both GQA intermediates changed from `[cycles,1,heads]` to `[cycles,batch,heads]`. Reducing over batch in both attention paths and keeping the accumulator at `[cycles,1,heads]` passes 47 CPU tests and a native v6e-8 step-2 validation/diagnostics -> step-3 transition with finite diagnostics | `results/v6e8-climbmix-diagnostics-shape-20260721/` |
| EXP-034 | Replace recursive doubling with correlated-key-qualified blocked substitution | Doubling explicitly formed `L^2...L^32`; correlated real-text keys drove intermediate norms to ~1e12 and catastrophic FP32 cancellation. Sixteen-row serial substitution with HIGHEST inter-block coupling matches the exact bad ClimbMix microbatch at gradient norm 2.406645/max 0.054548 versus 2.406825/0.054542 from the analytical reference. A 15-step native v6e-8 run covers all known triggers, stays finite, matches the reference loss within 4.2e-5 at step 15, and reaches 472,668 tok/s at a 31,989,071,680-byte compiled estimate—2.73x the safe analytical path | `results/v6e8-kda-substitution-realtext-20260721/` |

## Open experiments

These are recorded so the reasoning is not lost. None has been run.

| ID | Experiment | Why it matters | Prerequisite |
| --- | --- | --- | --- |
| OPEN-002 | Measure the WY system from real language tokens against random tokens at identical weights, across layer depth and training time | Determines whether another 40% of the core is genuinely available for real pretraining, and is the only entry that can. Every stress regime so far is synthetic, and the training workload is random tokens, which may not reproduce the key-correlation distribution of real text. Holding weights fixed and varying only the token source separates data-induced structure from structure the architecture imposes or has learned. Four outcomes, each actionable: every layer benign, which justifies testing an explicitly qualified BF16 configuration; only some layers problematic, which admits a static per-layer precision assignment; rare problematic tails, which points at beta and decay behaviour before any adaptive scheme; or widespread failure, which settles the FP32 guard conclusively. The compact record per sampled chunk is beta quantiles, key cosine similarity at lags 1, 4 and 16, decay retention, `kappa_2(I + A)`, maximum doubling-power norm, and direct BF16-versus-full-precision solve forward and backward error and gradient error. The proxies all mis-rank regimes, so the direct errors are the part that decides anything | A tokenized batch and a hook on the KDA layer |
| OPEN-003 | Conditional beta-cap sweep at 0.9, 0.75, and 0.5 | Bounding beta bounds `norm(A)` and might make the solve BF16-safe, which is worth about 40% of the core. It is not free: `0.5 * sigmoid(z)` moves mean beta from roughly 0.5 to 0.25 with centered logits, a substantial model change, and a cap of 0.5 does not guarantee `norm(A) < 1` at chunk 64, where the c=0.9 regime measures 30.5. Prefer `beta_max * sigmoid(z)` over hard clipping so saturated logits retain gradients | OPEN-002, to establish that beta is actually responsible for the observed growth |
| OPEN-004 | Fixed-batch sequence sweep, to separate length from batch | The fixed-token sweep in EXP-028 forces batch from 8 down to 1 as length grows, and a control at fixed length 2048 shows attention losing 45% from batch 8 to batch 1 against a smaller KDA decline. Part of attention's fall is therefore batch effect, and the 16,384 crossover margin is only 3%. Holding batch constant and letting tokens per step grow isolates the length term. If the crossover survives both, it is real; if it appears only at fixed tokens, the true crossover is longer than 16,384 | None |
| OPEN-005 | Validation loss on real text at matched compute, at and beyond the crossover | EXP-028 establishes the hybrid is faster past 16,384 but says nothing about whether it learns as well, and the synthetic reused-token workload cannot answer that. Quality per unit compute at 16,384 and 32,768 decides whether the throughput crossover is worth acting on | A tokenized corpus and an evaluation split |
| OPEN-006 | Recurrent decode throughput and state memory against a KV cache | The architectural case for KDA is strongest at inference, where state is constant-size rather than growing with context, and this is the one advantage no training measurement here captures. Decode throughput and resident state at 16k, 32k and 128k context against the attention control's KV cache | Decode path support in the KDA layer, which currently raises on non-training model modes |

## Precision policy

Guarded doubling is the default and stays the default. The existing
NaN-at-step-2 run proves that BF16 doubling is reachable and unsafe for the
random-token workload, which settles the kernel's general correctness status
independently of anything OPEN-002 finds.

If OPEN-002 shows real-language training is consistently benign, BF16 doubling
becomes a workload-qualified fast path rather than a generally safe
implementation, and it should be described that way wherever it is offered.
Qualifying it requires evidence that is strong across all of: multiple layers
and chunks; initialization and later checkpoints; multiple batches and seeds;
direct BF16-versus-full solve and gradient errors rather than correlation or
growth proxies; and a sufficiently long unguarded full-model run. Anything
short of that leaves the guard in place.

## Selection rule

Isolated core measurements are hypothesis filters, never selection criteria.
Full-model throughput is authoritative. Two changes in this ledger improved the
core and regressed the model: the shifted depthwise convolution and the hybrid
substitution solve, the latter 5.1% faster in the core and 2.2% slower in the
model. A core result is grounds for running the model, nothing more.

## Recording checklist

For each new run:

1. Preserve the immutable config and launch command.
2. Record model parameter count, sequence length, global tokens/step, precision,
   optimizer, parallelism, and exact software versions.
3. Discard compile and dispatch warmup before calculating throughput.
4. Preserve `metrics.jsonl`, `summary.json`, and `train.log`.
5. For kernel changes, add a numerical reference test before the performance
   run and retain an XPlane trace for the selected implementation.
6. Add the result to this ledger and the detailed comparison to
   `results/RESULTS.md`.
