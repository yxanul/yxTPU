# Divide-and-conquer inverse WY solve qualification

## Why the solver changed again

Substitution (EXP-034) closed the doubling numerical gate but is
latency-bound: its 16-row serial base case chains roughly sixty dependent
six-pass matmuls per solve and the backward runs that chain twice. EXP-022's
profile had already established the solve is bound by matmul latency on small
serial steps, not FLOPs.

## Method

For `A = I + L` with strictly lower `L`, the kernel forms `inv(A)` bottom-up:

1. Base 2×2 diagonal blocks: `inv = I - L_blockdiag`, zero matmuls.
2. Five dyadic merge levels (2→4→…→64): `inv = M - M @ C_w @ M`, where `C_w`
   couples the halves of each `2w` block. Premultiplied by the block-diagonal
   inverse, `C_w` is nilpotent of index two, so the two-term expansion is
   exact.

Ten uniform `[64, 64]` matmuls form the inverse; one dense `[64, 64] @
[64, 256]` matmul solves the combined U/W right-hand side. Backward reuses a
single formation for its forward recompute and its transposed solve. No
matrix power and no series sum is formed anywhere — the failure mode that
invalidated doubling on correlated real-text keys cannot occur — and every
solve matmul keeps the six-pass FP32 decomposition.

## Core result (one v6e chip, `B=8`, `T=2048`, `H=8`, chunk 64)

| Fused core | Substitution | Inverse |
| --- | ---: | ---: |
| Forward | 3.269 ms | 2.192 ms |
| Forward+backward | 9.240 ms | **5.033 ms** |

Rejected doubling measured 6.44 ms forward+backward. Correctness against the
recurrent and analytical references passes with gradient errors at or below
`1.4e-7` (`core-inverse.json`). The "doubling solve" text inside the two
archived core JSON names is a stale hardcoded label that predates the
benchmark-script fix; timings follow the `KDA_SOLVE_METHOD` selection.

## Exact trigger gate

Deterministic ClimbMix update 7, gradient-accumulation microbatch 4,
identical initialized 309.1M model:

| Path | Loss | Gradient norm | Max absolute gradient |
| --- | ---: | ---: | ---: |
| fused inverse | 11.382546 | 2.406743 | 0.054526 |
| full-FP32 analytical | 11.382677 | 2.406825 | 0.054542 |
| fused substitution (recorded) | 11.382567 | 2.406645 | 0.054548 |

Exhaustive on-device gradient-vector comparison, bit-identical parameters:
relative L2 `0.018622`, cosine `0.999827`, max absolute difference `1.75e-4`.
Substitution measured `0.018656` at cosine `0.999827`, so the solver change
leaves the known BF16-policy discrepancy unchanged, and the `3e-4`
analytical-equivalence gate fails for the same pre-existing reason.

## Full-model gate

Microbatch 16/device, GA=8, 2,097,152 tokens/update, real streamed ClimbMix,
15 steps covering the known spike positions 7, 13, and 15:

- Every loss and gradient finite.
- Mean after warmup: **616,303 global tok/s** (+30.4% over substitution's
  472,668; the numerically invalid doubling path measured 566,328).
- Compiled peak estimate unchanged: **31,989,071,680 bytes**.
- Step-15 loss `9.228989`; substitution `9.228992`; analytical `9.228950`.
- Step-15 gradient norm `1.990424`; analytical `1.990441`.

The gap to the matched global-attention control narrows from about 2.3x to
about 1.76x at sequence length 2048.

## Files

- `core-inverse.json`, `core-substitution.json` — matched core A/B
- `trigger-gradient-comparison.jsonl` — aggregate trigger gate, both paths
- `trigger-gradient-vector-comparison.jsonl` — exhaustive vector gate
- `run/` — 15-step model run config, metrics, and summary
