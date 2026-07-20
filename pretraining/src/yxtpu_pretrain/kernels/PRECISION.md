# KDA production precision policy

The production kernel exports only `pallas_kda_fused`. It is specialized to a
64-token chunk and a 128×128 FP32 recurrent state.

`guarded_fp32` means:

- BF16 Q/K/V input and output traffic;
- FP32 chunk-boundary state;
- one-pass TPU matmuls for ordinary chunk, pairwise, and state work;
- full-pass FP32 recursive doubling for both the WY solve and its application;
- backward recomputation of intra-chunk values with a reverse state scan.

The full model routes `full_fp32` to the owned XLA/recurrent correctness
reference rather than changing the Pallas kernel's fixed production contract.

Unsafe BF16 solve precision, alternate substitution/block solvers, shifted
convolution, and stage exits are intentionally confined to
`pretraining/benchmarks/`. They are not workload-independent safe
implementations. A future workload-qualified fast path must satisfy OPEN-001
with direct solve forward/backward and gradient errors, then pass full-model
loss and throughput validation.
