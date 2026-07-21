# One-billion-token qualification of fused KDA substitution

## Workload

- TPU: one v6e-8 Spot slice, eight-way data parallelism
- Model: 309,111,392 parameters, four `[KDA, KDA, KDA, NoPE-GQA]` cycles
- Data: streamed `karpathy/climbmix-400b-shuffle`, Rust GPT-2 tokenizer
- Shape: sequence 2,048, microbatch 16/device, gradient accumulation 8
- Optimizer: AdamW, using the original 10B learning-rate schedule
- KDA: fused Pallas, 16-row substitution, HIGHEST solve coupling
- Loss: Tokamax fused linear cross-entropy
- Checkpoints: disabled

The run completed 477 updates and 1,000,341,504 packed tokens. It was a
stability and throughput qualification, not a converged 1B-token schedule: the
learning-rate schedule still targeted 4,769 updates.

## Full-model result

- Final loss: `4.2389517`
- Final raw gradient norm: `0.3008611`
- Mean throughput after warmup: **472,463 global tok/s**
- Maximum throughput: `472,948 global tok/s`
- Compiled peak estimate: `31,989,071,680` bytes
- Every recorded loss and gradient norm was finite.
- Step-250 held-out loss: `4.9815603`
- Step-250 diagnostic gradient norm: `0.7533451`; max element `0.0174660`
- W&B: `https://wandb.ai/davidfranco2300-other/yxtpu-pretrain/runs/g4gaekvg`

This closes the long-run stability and throughput part of the qualification.

## Exact-trigger aggregate comparison

The deterministic update-7/microbatch-4 gate was rerun after the 1B job:

| Path | Loss | Gradient norm | Max gradient element |
| --- | ---: | ---: | ---: |
| fused substitution | 11.3825674 | 2.4066451 | 0.05454788 |
| full-FP32 analytical | 11.3826771 | 2.4068251 | 0.05454240 |

The gradient-norm relative difference is `7.48e-5`, and the maxima differ by
`1.00e-4` relative to the reference maximum. These aggregate values reproduce
the earlier qualification but do not prove elementwise gradient agreement.

## Exhaustive gradient-vector comparison

The harness now computes the gradient trees for both paths inside one TPU
program and reduces their elementwise difference on device. The input
parameters are bit-identical.

| Fused precision | Relative L2 error | Cosine | Max abs difference | Max difference / reference max |
| --- | ---: | ---: | ---: | ---: |
| production policy | **0.0186555** | 0.9998266 | 0.00015831 | 0.0029025 |
| all fused matmul roles HIGHEST | **0.0178933** | 0.9998407 | 0.00016392 | 0.0030053 |

Promoting chunk, state, pairwise, and coupling matmuls therefore does not
collapse the discrepancy. Under a proposed few-times-`1e-4` whole-gradient
relative tolerance, the analytical-equivalence gate fails. The 1B run strongly
qualifies stability for this workload, but it does not establish that the
fused gradient is numerically interchangeable with the analytical reference.

No 10B run should be described as unconditionally reference-qualified until
the remaining 1.8% vector difference is either explained against an accepted
BF16 baseline or reduced.

The diagnostic's `--direct-compare` mode now defaults to a `3e-4` relative-L2
acceptance threshold, requires bit-identical input parameters, and exits
nonzero when either condition fails. The measured production path therefore
fails the executable qualification gate rather than merely carrying a prose
caveat.
