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
| EXP-017 | Collapse the fused Pallas grid so one program advances eight streams instead of one | The grid falls from 2,048 to 256 programs and the core reaches 12.577 ms forward+backward; the model reaches 398,061 tok/s (+30.8%) with an unchanged loss curve. `streams_per_program` above 8 fails with `CompileTimeScopedVmemOom` | `results/v6e8-kda-hybrid-273m-fused-collapsed-h8-s2048-b8-20260720/`; `results/v6e8-kda-diag-chighest-shighest-20260720/` |
| EXP-018 | Reduce in-kernel MXU precision from six-pass FP32 to one-pass BF16 | A blanket reduction is 4.93x faster in the core but diverges to NaN at model step 2. Bisecting by matmul role isolates the repeated-squaring triangular solve as the sole cause: guarding only the solve trains normally at 560,923 tok/s (+84.3%), while guarding only the pairwise still diverges. Q/K/V arrive in BF16, so elsewhere the extra passes refined mantissa the operands never carried | `results/v6e8-kda-solveguard-20260720/`; `results/v6e8-kda-hybrid-273m-selected-s2048-b8-20260720/` |
| EXP-019 | Split the solve series by matmul role to recover the guarded cost | Rejected. Applying a power looks additive but the update is `solution <- (I + P^(2^k)) solution`, so the running solution compounds exactly as the power does; BF16 applications still reach NaN at step 2. The whole solve needs six passes, and the FP32 solve is 2.73 ms of the 6.437 ms core | `results/v6e8-kda-solveapply-bf16-20260720/` |
| EXP-016 | Build a production-shape fused TPU Pallas KDA forward and matched backward, with FP32 state/accumulation and BF16 Q/K/V traffic | Native TPU correctness passes against the recurrent and analytical-XLA references. Core training reaches 895,927 tok/s (+74.29%) at 0.883 GB compiled memory (-54.46%); the 272.9M hybrid reaches 304,300 global tok/s (+62.89%) at 15.4 GB/chip | `results/v6e8-kda-fused-pallas-core-20260720/`; `results/v6e8-kda-hybrid-273m-fused-pallas-adamw-s2048-b8-20260720/`; `results/v6e8-kda-hybrid-273m-fused-pallas-profile-20260720/` |
| EXP-017 | Selectively retain the fused custom-VJP residuals across MaxText's outer decoder remat | Rejected: 297,925 tok/s (-2.10% versus EXP-016) and 20.4 GB/chip (+5.0 GB). The original `minimal_with_context` policy remains selected | `results/v6e8-kda-hybrid-273m-fused-pallas-remat-adamw-s2048-b8-20260720/` |

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
