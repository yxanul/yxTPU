# TPU XPlane profile summary

- Device: `/device:TPU:0`
- Profiled optimizer steps: 3
- Mean device step: 274.290 ms
- Leaf device time accounted: 99.85%

## Training phases

| Phase | ms/step | % of step |
| --- | ---: | ---: |
| embedding/input prep | 1.834 | 0.67% |
| forward transformer scan | 64.945 | 23.68% |
| output head + loss | 12.803 | 4.67% |
| backward transformer scan | 150.695 | 54.94% |
| post-backward + optimizer/metrics | 43.603 | 15.90% |

The post-backward phase contains fused output/embedding gradients, gradient
clipping and norm metrics, and the optimizer update. XLA fusion prevents a reliable
finer split of that phase from source attribution alone.

## Splash Attention kernels

| Kernel | ms/step | % of step |
| --- | ---: | ---: |
| forward | 24.493 | 8.93% |
| backward fused dQ/dK/dV | 40.143 | 14.64% |
| total | 64.636 | 23.56% |

Tokamax Splash computes dQ, dK, and dV in this single backward Pallas
kernel; its underlying trace name retains the historical `dkv` suffix.

## Muon optimizer lowering

These are leaf TPU operations whose source stack points into Optax Muon.
They are a subset of the post-backward phase.

- Source-attributed Muon time: 32.483 ms/step (11.84%)
- Source-attributed leaf operations: 54.0/step

| Muon XLA category | ms/step | % of whole step |
| --- | ---: | ---: |
| convolution fusion | 29.162 | 10.63% |
| loop fusion | 2.165 | 0.79% |
| data formatting | 1.156 | 0.42% |

The Newton–Schulz matrix multiplications lower as XLA `convolution fusion`
operations on the TPU MXU. No Pallas/custom-call Muon kernel appears in the trace.

## XLA HLO categories

| XLA category | ms/step | % of step |
| --- | ---: | ---: |
| convolution fusion | 103.239 | 37.64% |
| loop fusion | 73.477 | 26.79% |
| custom-call | 64.636 | 23.56% |
| data formatting | 18.868 | 6.88% |
| all-reduce | 6.153 | 2.24% |
| reduce | 4.204 | 1.53% |
| broadcast | 1.493 | 0.54% |
| custom fusion | 1.243 | 0.45% |
| non-fusion elementwise | 0.323 | 0.12% |
| copy-done | 0.184 | 0.07% |
| dynamic-update-slice | 0.040 | 0.01% |
| sort | 0.011 | 0.00% |
| copy-start | 0.005 | 0.00% |
| iota | 0.002 | 0.00% |
| async-start | 0.001 | 0.00% |
| async-done | 0.000 | 0.00% |
