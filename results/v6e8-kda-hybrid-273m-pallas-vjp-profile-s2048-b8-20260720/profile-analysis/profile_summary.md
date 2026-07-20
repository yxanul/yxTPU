# TPU XPlane profile summary

- Device: `/device:TPU:0`
- Profiled optimizer steps: 3
- Mean device step: 1462.402 ms
- Leaf device time accounted: 99.86%

## Training phases

| Phase | ms/step | % of step |
| --- | ---: | ---: |
| embedding/input prep | 1.417 | 0.10% |
| forward transformer scan | 295.771 | 20.22% |
| output head + loss | 5.922 | 0.40% |
| backward transformer scan | 1147.997 | 78.50% |
| post-backward + optimizer/metrics | 9.293 | 0.64% |

The post-backward phase contains fused output/embedding gradients, gradient
clipping and norm metrics, and the optimizer update. XLA fusion prevents a reliable
finer split of that phase from source attribution alone.

## Splash Attention kernels

| Kernel | ms/step | % of step |
| --- | ---: | ---: |
| forward | 2.536 | 0.17% |
| backward fused dQ/dK/dV | 4.504 | 0.31% |
| total | 7.041 | 0.48% |

Tokamax Splash computes dQ, dK, and dV in this single backward Pallas
kernel; its underlying trace name retains the historical `dkv` suffix.

## Kimi Delta Attention source attribution

These are leaf TPU operations whose direct source or source stack points
into the KDA implementation. Direct-source groups are disjoint; operations
that retain KDA only in their fused source stack are included in the overall
total but not assigned to a group.

- Source-or-stack KDA time: 1358.300 ms/step (92.88%)
- Direct-source grouped time: 1315.252 ms/step (89.94%)

| Direct-source KDA group | ms/step | % of whole step |
| --- | ---: | ---: |
| decay-weighted block matmuls | 277.004 | 18.94% |
| chunk transforms and cumulative decay | 18.225 | 1.25% |
| WY inputs and triangular solve | 843.661 | 57.69% |
| intra/inter-chunk recurrence | 90.298 | 6.17% |
| QKV convolution and gates | 38.293 | 2.62% |
| shard/rematerialization wrapper | 47.772 | 3.27% |

The WY triangular solve and decay-weighted pairwise construction dominate
this prototype. A fused Pallas KDA forward/backward kernel should target
those operations before tuning the four global-attention layers.

## XLA HLO categories

| XLA category | ms/step | % of step |
| --- | ---: | ---: |
| custom-call | 812.868 | 55.58% |
| convolution fusion | 268.651 | 18.37% |
| loop fusion | 240.096 | 16.42% |
| data formatting | 60.046 | 4.11% |
| broadcast | 43.534 | 2.98% |
| output fusion | 12.502 | 0.85% |
| all-reduce | 10.990 | 0.75% |
| dynamic-update-slice | 5.618 | 0.38% |
| slice | 2.807 | 0.19% |
| non-fusion elementwise | 1.441 | 0.10% |
| custom fusion | 0.654 | 0.04% |
| reduce | 0.504 | 0.03% |
| dynamic-slice | 0.396 | 0.03% |
| copy-done | 0.259 | 0.02% |
| copy-start | 0.011 | 0.00% |
| iota | 0.011 | 0.00% |
| sort | 0.008 | 0.00% |
| async-done | 0.001 | 0.00% |
| async-start | 0.000 | 0.00% |
