# KDA precision policy

The production kernel exports only `pallas_kda_fused`. It is specialized to a
64-token chunk and a 128×128 FP32 recurrent state, and it owns the whole QKV
mixer: it consumes the raw fused-projection output `[B, T, 3, H, D]` and the
conv weight, and runs the causal depthwise convolution, SiLU, Q/K
normalization, and chunked recurrence in one program per batch element
(EXP-037).

`guarded_fp32` means:

- BF16 raw-QKV input and output traffic;
- FP32 in-kernel convolution accumulation and SiLU (the XLA reference path
  convolves in BF16);
- FP32 chunk-boundary state;
- one-pass TPU matmuls for ordinary chunk, pairwise, and state work;
- a full-pass FP32 divide-and-conquer explicit inverse for the WY solve and
  its transpose;
- backward recomputation of intra-chunk values with a reverse state scan.

Real ClimbMix batches invalidated recursive doubling even when every solve
matmul used the full TPU FP32 decomposition. The algorithm explicitly formed
`L^2 ... L^32`; correlated keys drove those intermediates above `1e12`, then
the finite nilpotent-series sum catastrophically cancelled despite a benign
unit-lower system and a bounded true solution. The exact offending microbatch
measured gradient norm `3,933.7` under doubling against `2.407` in the
full-FP32 reference.

The fail-closed replacement was 16-row serial substitution with HIGHEST
inter-block coupling. It removed global matrix powers and passed every
stability gate, but it is latency-bound: its base case chains roughly sixty
serially dependent six-pass matmuls per solve and the backward runs that
chain twice. It sustained 472,668 global tok/s.

Production now selects the "inverse" solver. It computes the exact unit-lower
inverse bottom-up: 2×2 diagonal blocks invert as `I - L` with no matmul, and
each dyadic merge level applies `inv = M - M @ C @ M`, which is exact because
the coupling between two inverted halves is nilpotent of index two once
premultiplied. No matrix power and no series sum is formed anywhere, so the
method sits in substitution's stability class, not doubling's; every
formation and application matmul stays at the full six-pass FP32
decomposition. The full 64×64 inverse forms in ten uniform matmuls, one dense
MXU matmul applies it to the 256-wide right-hand side, and the backward
reuses a single formation for both its forward recompute and its transposed
solve.

Qualification on native v6e-8, 2026-07-21:

- Fused core forward+backward: 5.02 ms against 9.24 ms for substitution and
  6.44 ms for the rejected doubling.
- Exact update-7/microbatch-4 trigger: loss `11.382546`, gradient norm
  `2.406743`, max gradient `0.054526`, against `11.382677` / `2.406825` /
  `0.054542` from the full-FP32 analytical reference. This is at least as
  close as substitution measured.
- Exhaustive on-device gradient-vector comparison on the same trigger:
  relative L2 `0.018622`, cosine `0.999827`, max absolute difference
  `1.75e-4`, with bit-identical parameters. Substitution measured `0.018656`
  at cosine `0.999827`, so the solver change leaves the known discrepancy
  unchanged: it is carried by the one-pass BF16 chunk/state/pairwise policy,
  not by the solve.
- Fifteen real-text steps covering the known spike positions 7, 13, and 15
  finish finite with step-15 loss `9.228989` (substitution `9.228992`,
  analytical reference `9.228950`) and mean throughput **616,303 global
  tok/s**, +30.4% over substitution and above every previously recorded
  operating point of this model, including the numerically invalid doubling
  path.

The conv-folded kernel (EXP-037) reran every gate on the same day: core
gradients including the conv weight match the XLA-mixer + analytical
reference at or below `1.8e-9`; the trigger measures gradient norm
`2.4068875` against the reference `2.4068251`; the exhaustive vector
comparison measures `0.019792` relative L2 at cosine `0.999805` with
bit-identical parameters (part of the movement from `0.018622` is the FP32
in-kernel convolution against the reference's BF16 convolution); and 15
real-text steps stay finite at **735,658 global tok/s** with a
29,960,940,032-byte compiled peak, 2.03 GB below the unfolded kernel.

The aggregate norm and maximum are not an elementwise gradient comparison,
and the `3e-4` relative-L2 analytical-equivalence gate still fails for the
same pre-existing reason it failed under substitution. The path may be
described as stable on the measured ClimbMix workload. It must not be
described as reference-equivalent or as an unconditional 10B default until
the roughly 2% vector discrepancy is accepted against an explicit BF16
baseline or reduced.

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
