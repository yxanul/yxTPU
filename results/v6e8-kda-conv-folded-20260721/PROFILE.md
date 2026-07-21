# Conv-folded fused KDA kernel qualification

## Motivating profile

An XPlane profile of the 616k inverse-solve configuration
(`prefold_profile_categories.json`, self-time shares across all cores)
corrected two long-standing readings:

- The Chrome-trace export truncates at about one million events, and host
  python threads plus counters consume nearly the entire budget, which had
  silently dropped most device ops — including the whole backward — from
  exported traces. The shares below come from the raw `.xplane.pb` through
  xprof's `hlo_stats` instead.
- The "convolution fusion" HLO category (26.7%) is dominated by ordinary
  MLP/projection GEMMs, which XLA canonicalizes into convolutions on TPU. The
  actual depthwise QKV convolution is only about 2.5% — EXP-023's suspicion
  that the category overstated the convolution was itself understated.

The genuinely reducible block was the QKV-mixer *region* around the kernel:
activation-layout copies (`bf16/f32[16,2048,3072]`, `[16,2048,8,3,128]`), the
Q/K/V split copies, the FP32 SiLU cast round-trip, the conv and its weight
gradient, and adjacent elementwise fusions — together roughly 19% of device
time, almost all of it memory traffic the XLA graph materializes between the
projection and the fused kernel.

## Change

`pallas_kda_fused` now consumes the raw fused-projection output
`[B, T, 3, H, D]` plus the conv weight `[width, 3, H, D]` and owns the whole
mixer: causal depthwise convolution, SiLU, Q/K normalization, and the chunked
recurrence. One program advances all heads of one batch element, so blocks
arrive in the tensor's natural layout and head-major transposes happen on
VMEM-resident values. The convolution's raw history rides in a VMEM scratch
across the ordered chunk grid exactly like the fast-weight state; the
backward carries the `width - 1` future conv-output cotangent rows in a
second scratch (the reverse grid has already processed the chronologically
later chunk) and accumulates conv weight gradients into one revisited
per-batch output block. The XLA graph no longer materializes a convolved,
activated, split, or head-transposed QKV copy, and the kernel-boundary
stream transposes are gone on both sides.

The convolution parameter keeps its `(head, qkv, dim)` channel order; only
the tiny weight is transposed at trace time, so the `full_fp32` XLA reference
consumes bit-identical parameters through its unchanged path. On the fused
path the segment mask applies to the raw projection output before the
convolution; under this trainer's dense packing the mask is all ones, where
both orderings are exactly the identity. The in-kernel conv accumulates in
FP32, where the XLA reference convolves in BF16.

## Core result (one v6e chip, `B=8`, `T=2048`, `H=8`, chunk 64)

| Path | Forward | Forward+backward |
| --- | ---: | ---: |
| XLA mixer + analytical chunked KDA | 10.579 ms | 35.464 ms |
| Folded Pallas kernel | **1.847 ms** | **4.931 ms** |

The unfolded inverse kernel measured 2.192 / 5.033 ms — and those numbers
excluded the XLA conv, SiLU, splits, and transposes the folded kernel now
absorbs. Native correctness against the XLA mixer + analytical reference:
output max `7.3e-4`, final state `6.5e-3`, loss `1.28e-8`, and all five
gradients (`raw_qkv`, `conv_weight`, `log_decay`, `beta`, `initial_state`)
at or below `1.8e-9` (`core-correctness.json`).

## Exact trigger gate

Deterministic ClimbMix update 7, microbatch 4, identical parameters:

| Path | Loss | Gradient norm | Max absolute gradient |
| --- | ---: | ---: | ---: |
| fused folded (guarded) | 11.382810 | 2.4068875 | 0.0545042 |
| full-FP32 analytical | 11.382677 | 2.4068251 | 0.0545424 |

Exhaustive on-device gradient-vector comparison: relative L2 `0.019792`,
cosine `0.999805`, max absolute difference `1.48e-4`, parameters
bit-identical. The slight movement from the unfolded `0.018622` reflects the
FP32 in-kernel conv against the reference's BF16 conv; the `3e-4`
analytical-equivalence gate keeps failing for the same pre-existing one-pass
BF16 policy reason.

## Full-model gate

Microbatch 16/device, GA=8, real streamed ClimbMix, 15 steps covering the
known spike positions 7, 13, and 15:

- All 15 losses and gradients finite; per-step losses match the unfolded
  inverse run within about `1e-4`.
- Mean after warmup: **735,658 global tok/s** (+19.4% over the unfolded
  inverse's 616,303; +55.7% over the substitution baseline's 472,668).
- Compiled peak estimate: **29,960,940,032 bytes**, down 2.03 GB.
- Step-15 loss `9.229117` (unfolded `9.228989`, analytical `9.228950`);
  step-15 gradient norm `1.990650` (analytical `1.990441`).

The gap to the matched global-attention control at sequence 2048 narrows to
about 1.47x.

## Files

- `prefold_profile_categories.json` — motivating device-time shares
- `core-correctness.json`, `core-benchmark.json` — folded core gates
- `trigger-gradient-comparison.jsonl` — aggregate trigger gate, both paths
- `trigger-gradient-vector-comparison.jsonl` — exhaustive vector gate
- `run/` — 15-step model run config, metrics, and summary
