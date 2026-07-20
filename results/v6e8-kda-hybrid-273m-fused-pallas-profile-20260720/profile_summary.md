# TPU XPlane profile summary

- Device: `/device:TPU:0`
- Profiled optimizer steps: 3
- Mean device step: 430.747 ms
- Leaf device time accounted: 99.95%

## Training phases

| Phase | ms/step | % of step |
| --- | ---: | ---: |
| embedding/input prep | 1.606 | 0.37% |
| forward transformer scan | 99.332 | 23.06% |
| output head + loss | 7.315 | 1.70% |
| backward transformer scan | 314.689 | 73.06% |
| post-backward + optimizer/metrics | 7.606 | 1.77% |

The post-backward phase contains fused output/embedding gradients, gradient
clipping and norm metrics, and the optimizer update. XLA fusion prevents a reliable
finer split of that phase from source attribution alone.

## Splash Attention kernels

| Kernel | ms/step | % of step |
| --- | ---: | ---: |
| forward | 2.510 | 0.58% |
| backward fused dQ/dK/dV | 4.450 | 1.03% |
| total | 6.961 | 1.62% |

Tokamax Splash computes dQ, dK, and dV in this single backward Pallas
kernel; its underlying trace name retains the historical `dkv` suffix.

## Kimi Delta Attention source attribution

These are leaf TPU operations whose direct source or source stack points
into the KDA implementation. Direct-source groups are disjoint; operations
that retain KDA only in their fused source stack are included in the overall
total but not assigned to a group.

- Source-or-stack KDA time: 345.871 ms/step (80.30%)
- Direct-source grouped time: 177.646 ms/step (41.24%)

| Direct-source KDA group | ms/step | % of whole step |
| --- | ---: | ---: |
| QKV convolution and gates | 36.263 | 8.42% |
| shard/rematerialization wrapper | 141.382 | 32.82% |

The source groups distinguish the generic autodiff path from the analytical
whole-KDA VJP. Source attribution may still assign a fused operation to only
one contributing line, so use the phase totals for the authoritative split.

## Fused Pallas KDA kernels

| Kernel | ms/step | % of step |
| --- | ---: | ---: |
| forward | 136.215 | 31.62% |
| backward | 139.752 | 32.44% |
| total | 275.967 | 64.07% |

## XLA HLO categories

| XLA category | ms/step | % of step |
| --- | ---: | ---: |
| custom-call | 282.928 | 65.68% |
| convolution fusion | 60.597 | 14.07% |
| loop fusion | 41.090 | 9.54% |
| data formatting | 31.377 | 7.28% |
| dynamic-update-slice | 5.678 | 1.32% |
| all-reduce | 5.654 | 1.31% |
| reduce | 1.031 | 0.24% |
| broadcast | 0.751 | 0.17% |
| non-fusion elementwise | 0.673 | 0.16% |
| custom fusion | 0.652 | 0.15% |
| copy-done | 0.091 | 0.02% |
| sort | 0.008 | 0.00% |
| iota | 0.007 | 0.00% |
| copy-start | 0.006 | 0.00% |
| async-done | 0.004 | 0.00% |
| async-start | 0.001 | 0.00% |
