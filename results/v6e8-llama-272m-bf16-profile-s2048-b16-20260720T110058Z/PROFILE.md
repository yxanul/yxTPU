# TPU XPlane profile summary

- Device: `/device:TPU:0`
- Profiled optimizer steps: 3
- Mean device step: 237.619 ms
- Leaf device time accounted: 99.82%

## Training phases

| Phase | ms/step | % of step |
| --- | ---: | ---: |
| embedding/input prep | 1.466 | 0.62% |
| forward transformer scan | 73.548 | 30.95% |
| output head + loss | 13.872 | 5.84% |
| backward transformer scan | 137.048 | 57.68% |
| post-backward + optimizer/metrics | 11.255 | 4.74% |

The post-backward phase contains fused output/embedding gradients, gradient
clipping and norm metrics, and the optimizer update. XLA fusion prevents a reliable
finer split of that phase from source attribution alone.

## Splash Attention kernels

| Kernel | ms/step | % of step |
| --- | ---: | ---: |
| forward | 18.187 | 7.65% |
| backward dKV | 21.467 | 9.03% |
| backward dQ | 17.261 | 7.26% |
| total | 56.915 | 23.95% |

## XLA HLO categories

| XLA category | ms/step | % of step |
| --- | ---: | ---: |
| loop fusion | 91.109 | 38.34% |
| convolution fusion | 68.686 | 28.91% |
| custom-call | 56.915 | 23.95% |
| data formatting | 10.703 | 4.50% |
| all-reduce | 7.076 | 2.98% |
| custom fusion | 1.244 | 0.52% |
| broadcast | 0.795 | 0.33% |
| non-fusion elementwise | 0.425 | 0.18% |
| copy-done | 0.191 | 0.08% |
| dynamic-update-slice | 0.027 | 0.01% |
| sort | 0.011 | 0.00% |
| copy-start | 0.003 | 0.00% |
| iota | 0.002 | 0.00% |
| async-start | 0.001 | 0.00% |
| async-done | 0.000 | 0.00% |
