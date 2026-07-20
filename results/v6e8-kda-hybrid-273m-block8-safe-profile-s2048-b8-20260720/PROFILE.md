# TPU XPlane profile summary

- Device: `/device:TPU:0`
- Profiled optimizer steps: 3
- Mean device step: 837.757 ms
- Leaf device time accounted: 99.76%

## Training phases

| Phase | ms/step | % of step |
| --- | ---: | ---: |
| embedding/input prep | 1.412 | 0.17% |
| forward transformer scan | 150.866 | 18.01% |
| output head + loss | 5.949 | 0.71% |
| backward transformer scan | 668.242 | 79.77% |
| post-backward + optimizer/metrics | 9.308 | 1.11% |

The post-backward phase contains fused output/embedding gradients, gradient
clipping and norm metrics, and the optimizer update. XLA fusion prevents a reliable
finer split of that phase from source attribution alone.

## Splash Attention kernels

| Kernel | ms/step | % of step |
| --- | ---: | ---: |
| forward | 2.537 | 0.30% |
| backward fused dQ/dK/dV | 4.505 | 0.54% |
| total | 7.042 | 0.84% |

Tokamax Splash computes dQ, dK, and dV in this single backward Pallas
kernel; its underlying trace name retains the historical `dkv` suffix.

## Kimi Delta Attention source attribution

These are leaf TPU operations whose direct source or source stack points
into the KDA implementation. Direct-source groups are disjoint; operations
that retain KDA only in their fused source stack are included in the overall
total but not assigned to a group.

- Source-or-stack KDA time: 743.492 ms/step (88.75%)
- Direct-source grouped time: 703.096 ms/step (83.93%)

| Direct-source KDA group | ms/step | % of whole step |
| --- | ---: | ---: |
| decay-weighted block matmuls | 315.187 | 37.62% |
| chunk transforms and cumulative decay | 13.525 | 1.61% |
| WY triangular solve | 188.753 | 22.53% |
| WY U/W construction | 20.833 | 2.49% |
| intra/inter-chunk recurrence | 84.944 | 10.14% |
| QKV convolution and gates | 39.028 | 4.66% |
| shard/rematerialization wrapper | 40.826 | 4.87% |

The WY triangular solve and decay-weighted pairwise construction dominate
this prototype. A fused Pallas KDA forward/backward kernel should target
those operations before tuning the four global-attention layers.

## XLA HLO categories

| XLA category | ms/step | % of step |
| --- | ---: | ---: |
| convolution fusion | 301.791 | 36.02% |
| loop fusion | 243.404 | 29.05% |
| custom-call | 171.657 | 20.49% |
| data formatting | 56.811 | 6.78% |
| broadcast | 36.747 | 4.39% |
| all-reduce | 8.580 | 1.02% |
| output fusion | 7.206 | 0.86% |
| dynamic-update-slice | 5.605 | 0.67% |
| non-fusion elementwise | 1.443 | 0.17% |
| slice | 0.840 | 0.10% |
| custom fusion | 0.655 | 0.08% |
| reduce | 0.506 | 0.06% |
| dynamic-slice | 0.406 | 0.05% |
| copy-done | 0.092 | 0.01% |
| copy-start | 0.013 | 0.00% |
| iota | 0.010 | 0.00% |
| sort | 0.008 | 0.00% |
| async-done | 0.001 | 0.00% |
| async-start | 0.000 | 0.00% |
