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
loss curve, remains finite, and sustains 472,668 global tok/s. This is the
selected real-training path.

The full model routes `full_fp32` to the owned XLA/recurrent reference with the
validated analytical VJP. It remains the correctness fallback. It recomputes
compact chunk values during backward, remains finite on the real-text trigger,
and is about 9% faster than generic XLA autodiff in the measured full model.

Recursive doubling, unsafe BF16 solve precision, alternate block solvers,
shifted convolution, and stage exits are intentionally confined to
`pretraining/benchmarks/`. They are not workload-independent safe
implementations. Any future solver change must cover the correlated-key growth
fixture, the exact real-text trigger with direct per-microbatch gradient
comparison, and a full-model loss/throughput run. Solve-only residuals are not
sufficient.
