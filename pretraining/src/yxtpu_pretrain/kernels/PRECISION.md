# KDA precision policy

The production kernel exports only `pallas_kda_fused`. It is specialized to a
64-token chunk and a 128×128 FP32 recurrent state.

`guarded_fp32` means:

- BF16 Q/K/V input and output traffic;
- FP32 chunk-boundary state;
- one-pass TPU matmuls for ordinary chunk, pairwise, and state work;
- 16-row serial substitution for the WY solve and its transpose;
- full-pass FP32 solve application and inter-block coupling;
- backward recomputation of intra-chunk values with a reverse state scan.

Real ClimbMix batches invalidated recursive doubling even when every solve
matmul used the full TPU FP32 decomposition. The algorithm explicitly formed
`L^2 ... L^32`; correlated keys drove those intermediates above `1e12`, then
the finite nilpotent-series sum catastrophically cancelled despite a benign
unit-lower system and a bounded true solution. The exact offending microbatch
measured gradient norm `3,933.7` under doubling against `2.407` in the
full-FP32 reference.

Blocked substitution removes global matrix powers. With HIGHEST inter-block
coupling, the same native-TPU trigger measures `2.406645` and max gradient
`0.0545479`, against `2.406825` and `0.0545424` from the analytical reference.
A 15-step real-text run covers all known spike positions, matches the reference
loss curve, remains finite, and sustains 472,668 global tok/s. A longer run
completes 1,000,341,504 real tokens with every loss and gradient finite, final
loss `4.23895`, and mean throughput 472,463 global tok/s. This is the selected
workload-qualified path in the current ClimbMix profile.

The aggregate norm and maximum are not an elementwise gradient comparison. An
exhaustive on-device comparison of every gradient element on the same trigger
measures relative L2 error `0.0186555`, cosine similarity `0.9998266`, and max
absolute difference `0.00015831`. Promoting chunk, state, pairwise, and coupling
matmuls to HIGHEST still measures `0.0178933` relative L2. Parameters are
bit-identical. Substitution therefore passes the long-run stability gate but
fails a proposed few-times-`1e-4` analytical-equivalence threshold.

The full model routes `full_fp32` to the owned XLA/recurrent reference with the
validated analytical VJP. It remains the correctness fallback. It recomputes
compact chunk values during backward, remains finite on the real-text trigger,
and is about 9% faster than generic XLA autodiff in the measured full model.

Recursive doubling, unsafe BF16 solve precision, alternate block solvers,
shifted convolution, and stage exits are intentionally confined to
`pretraining/benchmarks/`. They are not workload-independent safe
implementations. The production solver itself is fixed in the kernel and is
not a training-config option. Any future solver or precision-policy change must
cover the correlated-key growth fixture, the exact real-text trigger with an
exhaustive per-microbatch gradient-vector comparison, and a full-model
loss/throughput run. Solve-only residuals and aggregate gradient norms are not
sufficient.

The current `guarded_fp32` path may be described as stable on the measured
ClimbMix workload. It must not be described as reference-equivalent or as an
unconditional 10B default until the 1.8% vector discrepancy is accepted against
an explicit BF16 baseline or reduced.
