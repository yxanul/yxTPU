# KDA precision policy

The production kernel exports only `pallas_kda_fused`. It is specialized to a
64-token chunk and a 128×128 FP32 recurrent state.

`guarded_fp32` means:

- BF16 Q/K/V input and output traffic;
- FP32 chunk-boundary state;
- one-pass TPU matmuls for ordinary chunk, pairwise, and state work;
- full-pass FP32 recursive doubling for both the WY solve and its application;
- backward recomputation of intra-chunk values with a reverse state scan.

Real ClimbMix batches invalidated `guarded_fp32` as a production mode. With
effectively frozen weights, one exact update produced an accumulated gradient
norm of 446.8. Its offending microbatch measured 3,933.7 in the guarded fused
backward against 2.407 in the full-FP32 reference. A normal-learning-rate run
became non-finite at step 12. Promoting individual or all Pallas matmul roles,
shrinking the pairwise row block, and moving the decay anchor did not restore
the reference gradient. The forward loss remained close, locating the failure
in the fused backward rather than the optimizer or output loss.

Consequently, `guarded_fp32` is a synthetic-benchmark implementation only.
Typed configuration rejects it whenever `experiment.benchmark=false`.

The full model routes `full_fp32` to the owned XLA/recurrent reference with the
validated analytical VJP. This is the only real-training mode. It recomputes
compact chunk values during backward, remains finite on the real-text trigger,
and is about 9% faster than generic XLA autodiff in the measured full model.

Unsafe BF16 solve precision, alternate substitution/block solvers, shifted
convolution, and stage exits are intentionally confined to
`pretraining/benchmarks/`. They are not workload-independent safe
implementations. A future workload-qualified fast path must cover exact
real-text trigger batches with direct per-microbatch gradient comparison, then
pass a long full-model loss and throughput validation. Solve-only tests are no
longer sufficient because EXP-032 exposed a fused-backward failure outside the
previous WY gate.
