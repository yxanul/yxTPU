# Real-text qualification of fused KDA substitution

## Root cause

The rejected fused kernel used recursive doubling for the unit-lower WY solve.
It explicitly formed `L^2 ... L^32`. Correlated real-text keys made those
intermediates reach roughly `1e12`; the final FP32 nilpotent-series sum then
catastrophically cancelled although the true solution stayed bounded. This
explains why promoting matmul roles and changing the pairwise factorization did
not repair the gradient.

The replacement is 16-row serial substitution in both forward and transposed
backward solves, with `HIGHEST` inter-block coupling. It never forms global
matrix powers.

## Exact trigger gate

Deterministic ClimbMix update 7, gradient-accumulation microbatch 4, identical
initialized 309.1M model:

| Path | Loss | Gradient norm | Max absolute gradient |
| --- | ---: | ---: | ---: |
| fused substitution | 11.382567 | 2.406645 | 0.054548 |
| full-FP32 analytical | 11.382677 | 2.406825 | 0.054542 |
| rejected doubling | 11.382854 | 3,933.710 | 171.0 |

The correlated-key unit fixture also asserts that the generated system reaches
`max ||L^(2^k)|| > 1e8`; substitution must match both the forward and
transposed `solve_triangular` references.

## Full-model gate

Microbatch 16/device, GA=8, 2,097,152 tokens/update, real streamed ClimbMix:

- 15 steps complete with finite loss and gradients, covering the known spike
  positions 7, 13, and 15.
- Mean after warmup: **472,668 global tok/s**.
- Compiled peak estimate: **31,989,071,680 bytes**.
- Step-15 loss: `9.2289915`; analytical reference run: `9.2289495`.
- Step-15 gradient norm: `1.9903924`; analytical reference: `1.9904410`.

This is 2.73x the 172,961 tok/s analytical fallback. The old doubling path was
566,328 tok/s but is invalid and remains benchmark-only.

## Diagnostics transition

The same fused substitution path ran held-out evaluation and diagnostics at
step 2, reported finite gradient/activation/attention telemetry, then executed
step 3 normally at 472,679 tok/s. This jointly validates the solver repair and
the batch-independent telemetry fix before another long run.
