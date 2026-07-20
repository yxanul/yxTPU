# TPU XPlane profile summary

- Device: `/device:TPU:0`
- Profiled optimizer steps: 3
- Mean device step: 700.450 ms
- Leaf device time accounted: 99.72%

## Training phases

| Phase | ms/step | % of step |
| --- | ---: | ---: |
| embedding/input prep | 1.418 | 0.20% |
| forward transformer scan | 150.354 | 21.47% |
| output head + loss | 5.901 | 0.84% |
| backward transformer scan | 531.523 | 75.88% |
| post-backward + optimizer/metrics | 9.306 | 1.33% |

The post-backward phase contains fused output/embedding gradients, gradient
clipping and norm metrics, and the optimizer update. XLA fusion prevents a reliable
finer split of that phase from source attribution alone.

## Splash Attention kernels

| Kernel | ms/step | % of step |
| --- | ---: | ---: |
| forward | 2.510 | 0.36% |
| backward fused dQ/dK/dV | 4.462 | 0.64% |
| total | 6.972 | 1.00% |

Tokamax Splash computes dQ, dK, and dV in this single backward Pallas
kernel; its underlying trace name retains the historical `dkv` suffix.

## Kimi Delta Attention source attribution

These are leaf TPU operations whose direct source or source stack points
into the KDA implementation. Direct-source groups are disjoint; operations
that retain KDA only in their fused source stack are included in the overall
total but not assigned to a group.

- Source-or-stack KDA time: 599.729 ms/step (85.62%)
- Direct-source grouped time: 562.738 ms/step (80.34%)

| Direct-source KDA group | ms/step | % of whole step |
| --- | ---: | ---: |
| public KDA dispatch | 516.477 | 73.74% |
| QKV convolution and gates | 37.368 | 5.33% |
| shard/rematerialization wrapper | 8.892 | 1.27% |

The source groups distinguish the generic autodiff path from the analytical
whole-KDA VJP. Source attribution may still assign a fused operation to only
one contributing line, so use the phase totals for the authoritative split.

## XLA HLO categories

| XLA category | ms/step | % of step |
| --- | ---: | ---: |
| convolution fusion | 307.784 | 43.94% |
| custom-call | 171.584 | 24.50% |
| loop fusion | 92.084 | 13.15% |
| data formatting | 62.879 | 8.98% |
| output fusion | 24.987 | 3.57% |
| all-reduce | 12.309 | 1.76% |
| dynamic-update-slice | 10.472 | 1.50% |
| broadcast | 9.670 | 1.38% |
| non-fusion elementwise | 2.259 | 0.32% |
| reverse | 1.586 | 0.23% |
| slice | 1.110 | 0.16% |
| custom fusion | 0.670 | 0.10% |
| reduce | 0.499 | 0.07% |
| dynamic-slice | 0.465 | 0.07% |
| copy-done | 0.109 | 0.02% |
| iota | 0.010 | 0.00% |
| copy-start | 0.010 | 0.00% |
| sort | 0.008 | 0.00% |
| async-done | 0.005 | 0.00% |
| async-start | 0.000 | 0.00% |
