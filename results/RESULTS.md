# v6e-8 MaxText synthetic pretraining results

Measured on `yxtpu-v6e8-dev`, a single eight-chip TPU v6e slice, on
2026-07-20. The model is a 271.6M-parameter Llama-style decoder using BF16
compute, FP32 weights, Flash Attention, and AdamW. Each run used 2048-token
sequences and pure eight-way data parallelism.

| Batch/chip | Global tokens/step | Mean step time | Mean tokens/s/chip | Mean global tokens/s | Mean TFLOP/s/chip |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 131,072 | 0.129145 s | 126,865 | 1,014,920 | 219.5 |
| 16 | 262,144 | 0.238250 s | 137,536 | 1,100,290 | 238.0 |

Each summary contains 25 measured steps (indices 5–29); indices 0–4 were
excluded as compile/dispatch warmup. The batch-16 result is the selected
default. Its median global throughput was 1,100,465 tokens/s.

Raw logs and metrics:

- `v6e8-llama-272m-s2048-b8-20260720T104542Z/`
- `v6e8-llama-272m-s2048-b16-20260720T104714Z/`

## Qwix INT8 smoke test

The same batch-16 workload was run for 15 optimizer steps with
`use_qwix_quantization=true quantization=int8`. The run completed normally,
and loss decreased from 10.811 at measured step 5 to 10.710 at step 14.

| Precision recipe | Mean step | Mean global tokens/s | XLA memory/chip |
| --- | ---: | ---: | ---: |
| BF16 baseline | 0.238250 s | 1,100,290 | 24.6 GB |
| Qwix INT8 | 0.309842 s | 846,059 | 24.0 GB |

For this 271.6M-parameter shape, Qwix INT8 was 23.1% slower in throughput and
increased step time by 30.0%. The reported MaxText TFLOP/s is a
model-equivalent FLOP estimate, not a measurement of INT8 TOPS. Qwix
dynamically quantizes decoder-layer `dot_general` weights, activations, and
backward operands; FP32 master weights, optimizer state, normalization,
softmax/loss, and the Splash Attention kernel remain in their appropriate
higher-precision paths. The small model does not amortize calibration and
quantize/dequantize overhead well.

Raw log and metrics:

- `v6e8-llama-272m-qwix-int8-s2048-b16-20260720T105941Z/`

## BF16 XPlane profile

The profiler captured three steady-state BF16 optimizer steps after five
warmup steps. Their mean TPU device time was 237.619 ms. Removing only the
parent decoder-scan `while` records leaves 99.82% of the step accounted for
without double-counting nested device operations.

| Training phase | ms/step | % of step |
| --- | ---: | ---: |
| Embedding/input preparation | 1.466 | 0.62% |
| Forward transformer scan | 73.548 | 30.95% |
| Output head and cross-entropy | 13.872 | 5.84% |
| Backward transformer scan | 137.048 | 57.68% |
| Post-backward, AdamW, and metrics | 11.255 | 4.74% |

This run uses the JAX Pallas TPU Splash Attention implementation selected by
MaxText's `attention=flash` path. Its three main kernels account for:

| Splash Attention kernel | ms/step | % of step |
| --- | ---: | ---: |
| Forward | 18.187 | 7.65% |
| Backward dKV | 21.467 | 9.03% |
| Backward dQ | 17.261 | 7.26% |
| **Total** | **56.915** | **23.95%** |

At the lower XLA category level, dense `dot_general` work lowered as
`convolution fusion` uses 68.686 ms (28.91%), elementwise/layout/scan work
lowered as `loop fusion` uses 91.109 ms (38.34%), explicit data formatting
uses 10.703 ms (4.50%), and all-reduce communication uses 7.076 ms (2.98%).
The phase split is more useful than trying to assign every fused HLO to one
source line: the final update, gradient norms, and some gradient work are
fused together.

Profile artifacts and the generated machine-readable analysis:

- `v6e8-llama-272m-bf16-profile-s2048-b16-20260720T110058Z/`

## Modern 270M architecture

The follow-up workload keeps the parameter count within 0.6% of the original
model while changing the block shape and training kernels:

| Item | Modern workload |
| --- | --- |
| Parameters | 270,046,208 |
| Decoder | 18 pre-norm Llama blocks |
| Width | 1024 |
| Attention | 8 query heads, 2 KV heads, 128 head dimension (4:1 GQA) |
| QKV projection | One fused 1024 x 1536 GEMM, split as 8Q + 2K + 2V |
| MLP | Fused 1024 x (2 x 2816) input projection, SiLU gate x linear value |
| Normalization / positions | RMSNorm / RoPE, 500,000 maximum timescale |
| Precision | BF16 compute, FP32 master weights |
| Attention kernel | Tokamax Pallas Splash Attention, exact causal mask |

MaxText's existing fused-QKV implementation assumed equal Q, K, and V head
counts. The local patch adds an unequal-head fused representation, avoiding
padding K and V back to eight heads. Fused SwiGLU uses one input
`DenseGeneral`; XLA fuses the SiLU and elementwise gate work around it.
Tokamax Splash does not materialize the complete sequence-by-sequence score
matrix.

The analytical parameter count is:

- Embedding and untied output: 67,108,864
- 18 transformer layers: 202,936,320
- Final RMSNorm: 1,024
- Total: 270,046,208

### AdamW throughput

The run used the same 262,144 global tokens/step as the selected original
benchmark. Each number below is the mean of 25 steps after discarding step
indices 0–4.

| Model | Mean step | Mean tokens/s/chip | Mean global tokens/s | XLA memory/chip |
| --- | ---: | ---: | ---: | ---: |
| Original 271.6M, 12-layer MHA | 0.238250 s | 137,536 | 1,100,290 | 24.6 GB |
| Modern 270.0M, 18-layer GQA | 0.242244 s | 135,269 | 1,082,149 | 23.0 GB |

The modern architecture is 1.65% slower in tokens/s despite slightly fewer
estimated FLOPs and parameters. Eighteen sequential blocks and their smaller
per-layer GEMMs trade some utilization for depth. It uses 1.6 GB (6.5%) less
compiled memory. The measured median was 1,082,180 global tokens/s, and loss
fell from 10.837 to 10.556 over the measured interval.

Tokamax always selects its fused dQ/dK/dV backward kernel. An additional run
with `sa_use_fused_bwd_kernel=true` measured 1,082,239 tokens/s, just 0.008%
above the main run and therefore measurement noise; MaxText's Tokamax adapter
sets the fused flag internally regardless of that top-level override.

Raw results:

- `v6e8-modern-270m-adamw-s2048-b16-20260720T111528Z/`
- `v6e8-modern-270m-adamw-fusedattnbwd-s2048-b16-20260720T111652Z/`

## Muon + AdamW

The second run uses Optax Muon for transformer matrix weights and AdamW for
parameters that MaxText marks unsuitable for Muon: embeddings, the untied
logit projection, RMSNorm scales, and biases. The fused QKV projection was
added to MaxText's Muon dimension mapping so that it follows the matrix path
rather than the excluded path.

This is a throughput and execution smoke test, not an optimizer-quality
comparison: both runs deliberately use the same 3e-4 learning rate and only
30 synthetic-data steps.

| Optimizer | Mean step | Mean global tokens/s | XLA memory/chip | Measured loss |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 0.242244 s | 1,082,149 | 23.0 GB | 10.837 to 10.556 |
| Muon matrices + AdamW remainder | 0.274353 s | 955,500 | 21.4 GB | 10.832 to 10.556 |

Muon is 11.70% lower in throughput and 13.25% higher in step time here, while
using 1.6 GB (7.0%) less compiled memory. The lower state footprint comes from
Muon keeping one momentum accumulator for its matrix parameters rather than
AdamW's first- and second-moment accumulators. MaxText's model TFLOP/s metric
does not count Muon's optimizer matrix multiplications, so its reported 196.5
TFLOP/s/chip is not a hardware-utilization comparison.

### Muon kernel and profile

Optax 0.2.8 applies five Newton–Schulz iterations to orthogonalize the
momentum update. In the installed implementation each iteration is expressed
with ordinary JAX matrix multiplication, including `x @ x.T`, `a @ a`, and
`b @ x`, and is vectorized across the scanned layer matrices. XLA lowers those
operations to TPU MXU `convolution fusion` HLOs. There is no dedicated Pallas
Muon kernel or Muon custom call in this software path.

Three profiled steady-state Muon steps averaged 274.290 ms:

| Component | ms/step | % of step |
| --- | ---: | ---: |
| Embedding/input preparation | 1.834 | 0.67% |
| Forward transformer scan | 64.945 | 23.68% |
| Output head and loss | 12.803 | 4.67% |
| Backward transformer scan | 150.695 | 54.94% |
| Post-backward optimizer and metrics | 43.603 | 15.90% |
| of which source-attributed Muon work | 32.483 | 11.84% |

Muon's 32.483 ms source-attributed cost closely matches the 32.109 ms
unprofiled step-time difference between AdamW and Muon. Of that Muon time,
29.162 ms is matrix work lowered as `convolution fusion`, 2.165 ms is loop
fusion (primarily the Frobenius norm path), and 1.156 ms is data formatting.

The same profile verifies the fast attention path:

| Tokamax Splash kernel | ms/step | % of step |
| --- | ---: | ---: |
| Forward | 24.493 | 8.93% |
| Fused backward dQ/dK/dV | 40.143 | 14.64% |
| **Total** | **64.636** | **23.56%** |

Raw Muon results, XPlane trace, and generated profile analysis:

- `v6e8-modern-270m-muon-s2048-b16-20260720T111817Z/`
- `v6e8-modern-270m-muon-profile-s2048-b16-20260720T111937Z/`

## Kimi Delta Attention 3:1 hybrid

The KDA experiment repeats three KDA layers followed by one NoPE global-GQA
layer four times. Its 16 decoder layers use width 1024, eight 128-wide KDA
heads, dense fused SwiGLU, RMSNorm, BF16 compute, FP32 weights, and AdamW.
The model has 272,935,520 parameters. It is a controlled dense KDA/GQA model,
not an exact reproduction of Kimi Linear's 48B NoPE-MLA/MoE architecture.

The implementation generalizes MaxText's chunked Gated DeltaNet WY recurrence
to KDA's per-key-channel decay. The selected version uses 64-token chunks,
eight-row decay-weighted MXU matmuls, FLA's bounded `[-5, 0)` safe gate, and
explicit KDA rematerialization in backward. A recurrent reference verifies
forward output, final state, and all five input gradients; a stress case
verifies finite forward and backward values.

### Matched throughput

The requested batch of 16 sequences/chip did not fit. Before blockwise
pairwise matmuls, explicit rematerialization still produced a 55.5 GB/chip
temporary estimate. The selected block-eight version fits eight
sequences/chip at 22.9 GB/chip. The matched global-attention control was
therefore rerun at the same eight sequences/chip and 131,072 global
tokens/step.

Both results average 25 optimizer steps after excluding steps 0–4:

| Model | Parameters | Mean step | Global tokens/s | Tokens/s/chip | XLA memory/chip |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3:1 KDA/NoPE-GQA hybrid | 272,935,520 | 0.838646 s | 156,290 | 19,536 | 22.9 GB |
| 18-layer global-GQA control | 270,046,208 | 0.130563 s | 1,003,900 | 125,488 | 14.0 GB |

The current KDA lowering delivers 15.57% of the control's throughput: it is
6.42 times slower per step and uses 63.6% more compiled memory. Converting
the initial row-wise pairwise scan to eight-row matrix multiplications made
KDA 2.83 times faster than that prototype, but generic XLA remains far from a
competitive training kernel.

KDA loss falls from 7.750 to 0.445. This only establishes a finite optimizer
path, because the throughput benchmark deliberately repeats one synthetic
random batch.

Raw throughput results:

- `v6e8-kda-hybrid-273m-block8-safe-adamw-s2048-b8-20260720/`
- `v6e8-modern-270m-adamw-matched-s2048-b8-20260720/`

### KDA XPlane profile

Three profiled KDA steps average 837.757 ms, with 99.76% of device time
accounted for by leaf operations:

| Training phase | ms/step | % of step |
| --- | ---: | ---: |
| Embedding/input preparation | 1.412 | 0.17% |
| Forward transformer scan | 150.866 | 18.01% |
| Output head and loss | 5.949 | 0.71% |
| Backward transformer scan | 668.242 | 79.77% |
| Post-backward AdamW and metrics | 9.308 | 1.11% |

Source-or-stack attribution assigns 743.492 ms/step (88.75%) to
`kimi_delta_attention.py`. Direct-source hotspot groups are:

| KDA operation group | ms/step | % of whole step |
| --- | ---: | ---: |
| Decay-weighted block matmuls | 315.187 | 37.62% |
| WY triangular solve | 188.753 | 22.53% |
| Intra/inter-chunk recurrence | 84.944 | 10.14% |
| Shard/rematerialization wrapper | 40.826 | 4.87% |
| QKV convolution and gates | 39.028 | 4.66% |
| WY U/W construction | 20.833 | 2.49% |
| Chunk transforms and cumulative decay | 13.525 | 1.61% |

At the HLO level, convolution fusion consumes 301.791 ms (36.02%), loop
fusion 243.404 ms (29.05%), generic custom calls 171.657 ms (20.49%), and
data formatting 56.811 ms (6.78%). The four Tokamax Splash global-attention
layers consume just 7.042 ms (0.84%). The optimization target is therefore a
fused TPU Pallas forward/backward KDA kernel that avoids the generic
triangular solve and keeps the `128 x 128` fast-weight state tiled in VMEM;
sparsifying the four global layers cannot materially change this result.

Profile trace and generated analysis:

- `v6e8-kda-hybrid-273m-block8-safe-profile-s2048-b8-20260720/`

The complete algorithm, kernel survey, implementation decisions, and next
Pallas experiment are in `docs/KDA_HYBRID.md`.

### SGLang-JAX TPU Pallas forward audit

SGLang-JAX now contains a genuine Mosaic/Pallas KDA implementation for
packed variable-length prefill. It uses a Pallas gate cumsum, an intra-chunk
kernel with 16-row blocked forward substitution, an FP32 VMEM state
recurrence, and a Pallas output kernel. This corrects the earlier kernel
inventory: an open TPU KDA *forward* exists, but an open TPU KDA *training*
kernel still does not. The exported function is `chunk_kda_fwd`, its serving
backend calls it only for prefill/extend, and there is no backward or custom
VJP.

Under the installed JAX 0.10.2 stack, attempting reverse-mode differentiation
failed during linearization because the Pallas call did not produce known
output primals. A one-device orientation benchmark used 16,384 tokens, eight
heads, 128-wide K/V, chunk size 64, BF16 inputs, and FP32 state:

| Forward KDA core | Mean | Tokens/s |
| --- | ---: | ---: |
| SGLang-JAX packed Pallas forward | 38.523 ms | 425,307 |
| Current MaxText JAX chunk core | 9.664 ms | 1,695,337 |

The comparison is not perfectly component matched: SGLang-JAX includes
variable-length repacking and gate activation, while the current core includes
Q/K normalization and receives activated decay. It is sufficient to reject a
wholesale transplant for the 2048-token training workload.

The high-value part is its triangular-solve structure. The next A/B should
replace `solve(A, I) @ [B_value, B_key]` with a blocked direct
`solve(A, [B_value, B_key])` and add a custom training VJP:
`dB = A^-T dX`, `dA = -(dB X^T)` on the strict lower triangle. The existing
profile splits the WY solve into 60.133 ms forward and 128.619 ms backward,
so using only SGLang-JAX's forward would leave most of the cost untouched.

Audit summary:

- `v6e8-sglang-jax-kda-forward-audit-20260720/`

### Blocked Pallas solve and custom-VJP A/B

The SGLang-inspired experiment adds an exact 16-row blocked forward
substitution in Pallas. Its custom VJP solves the transposed unit-upper
system and constructs
`dA = -(A^-T dX) X^T`, masked to the strict lower triangle. U and W share one
combined 256-column right-hand side, so this path never materializes
`A^-1`.

Native TPU checks used the production `C=64`, `R=256` solve shape. Relative to
XLA, the maximum absolute forward error was `9.54e-7`; maximum gradient errors
were `2.29e-5` for the triangular system and `7.15e-7` for the right-hand
side. An end-to-end KDA check at `K=V=128` matched the recurrent reference for
the loss and all six input/state gradients, with maximum absolute errors from
`1.21e-13` to `1.46e-11`. All values were finite. The complete CPU suite has
five passing tests.

The isolated solve benchmark represents one chip's local workload: 2,048
independent `64 x 64` systems and a `64 x 256` combined right-hand side.

| Solve path | Forward | Forward + VJP |
| --- | ---: | ---: |
| XLA `A^-1` + two matmuls | 1.883 ms | 2.836 ms |
| XLA direct solve + custom VJP | 1.576 ms | 3.496 ms |
| Pallas blocked solve + custom VJP | 16.932 ms | 34.437 ms |

The direct XLA solve wins forward-only, but its explicit transpose solve and
`dA` construction lose in training. The standalone Pallas kernel is an order
of magnitude slower because each program performs serial row substitution;
it does not fuse KDA's pairwise construction or recurrence.

The end-to-end 272,935,520-parameter run confirms the microbenchmark:

| KDA solve backend | Mean step | Global tok/s | Tok/s/chip | XLA memory/chip |
| --- | ---: | ---: | ---: | ---: |
| Generic-autodiff XLA inverse control | 0.838646 s | 156,290 | 19,536 | 22.9 GB |
| Pallas blocked + custom VJP | 1.462050 s | 89,649 | 11,206 | 22.9 GB |

The Pallas path is 42.64% lower in throughput and 74.34% higher in step time.
Its loss remains finite and falls from 7.750 to 0.444.

The three-step device profile averages 1,462.402 ms. KDA source-or-stack work
is 1,358.300 ms (92.88%), the WY inputs/solve group is 843.661 ms (57.69%),
and the custom-call HLO category consumes 812.868 ms (55.58%); the Pallas
solve is its dominant new operation. Backward remains dominant at
1,147.997 ms (78.50%). The four global Tokamax Splash layers use only 7.041 ms
(0.48%).

The implementation is retained behind
`kda_use_pallas_blocked_solve=true`, with a dedicated experimental config.
The XLA inverse is retained in both generic and analytical paths.
Reproduction artifacts:

- `v6e8-kda-triangular-solve-ab-20260720/`
- `v6e8-kda-hybrid-273m-pallas-vjp-adamw-s2048-b8-20260720/`
- `v6e8-kda-hybrid-273m-pallas-vjp-profile-s2048-b8-20260720/`

### ejkernel XLA Gated Delta Rule audit

ejkernel 0.0.82 at commit
`2a426edbd4f88368d9d31d80cf2e4219aa69d7cc` contains an active exact
triangular-solve GDR forward using ordinary autodiff and a private
handwritten-custom-VJP variant. The latter is the only path that calls the
linked `_xla_impl_bwd.py`; the public multi-token API does not.

The private path matches the active path on a small FP32 TPU validation:
outputs and final state are exact at the measured precision, and maximum
absolute gradient differences for all six differentiable inputs range from
`9.09e-12` to `5.24e-10`.

The standalone one-layer workload uses `B=8`, `T=2048`, eight heads,
`K=V=128`, BF16, Q/K L2 normalization, and one v6e chip:

| Path | Chunk | Forward | Forward + backward | Training tok/s | Training compiled memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Active exact + autodiff | 64 | 2.705 ms | 7.290 ms | 2,247,489 | 1.223 GB |
| Private custom VJP | 64 | 2.556 ms | 9.956 ms | 1,645,682 | 1.353 GB |
| Active exact + autodiff | 256 | 12.730 ms | 17.277 ms | 948,318 | 1.953 GB |
| Private custom VJP | 256 | 12.685 ms | 18.817 ms | 870,716 | 1.390 GB |

The custom VJP is 5.84% faster forward at chunk 64 but 36.57% slower through
backward. The public default chunk 256 is also a poor schedule for this
shape: active training is 2.37 times slower than chunk 64.

A matched current-KDA core run at chunk 64 measures 9.657 ms forward and
32.653 ms forward+backward, or 1,696,629 and 501,764 tok/s. Per-channel KDA is
therefore 3.57 times slower forward and 4.48 times slower through backward
than scalar GDR.

The ejkernel inverse derivative and reverse state scan are reusable for a KDA
custom VJP. Its scalar pairwise-decay derivative is not: a direct shape
conversion would materialize `[B,H,NC,C,C,K]`. The KDA version must recompute
decay factors row-block by row-block and accumulate Q/K/cumulative-decay
gradients before the key-channel reduction. The complete mapping and proposed
implementation sequence are in `docs/EJKERNEL_GDR_TO_KDA.md`.

Artifacts:

- `v6e8-ejkernel-gdr-xla-audit-20260720/`

### Whole-KDA analytical XLA backward

The ejkernel-derived follow-up wraps the complete KDA core in a custom VJP.
The forward remains the selected XLA implementation. Backward explicitly
differentiates the WY inverse and reverse state recurrence, then recomputes
eight-row channel-decay blocks to accumulate pairwise Q/K/gate derivatives
without a `[chunk,chunk,key_dim]` tensor.

Seven CPU tests pass. A native-v6e comparison at `B=1`, `T=64`, `H=1`,
`K=V=128` gives an absolute loss difference of `5.17e-8`; all six gradients
are finite, and the largest absolute gradient difference is `1.76e-7`.

The one-chip production core A/B uses `B=8`, `T=2048`, `H=8`, `K=V=128`,
chunk 64, BF16 inputs, and gradients with respect to Q, K, V, log-decay,
beta, and initial state:

| KDA core | Forward | Forward + backward | Training tok/s | Training compiled memory |
| --- | ---: | ---: | ---: | ---: |
| Generic autodiff | 9.660 ms | 32.696 ms | 501,100 | 6.614 GB |
| Whole analytical VJP | 9.666 ms | 31.878 ms | 513,962 | 1.930 GB |

Forward is unchanged within measurement noise. Training throughput improves
by 2.57%, while compiled memory falls by 70.82%.

The full 272,935,520-parameter, 12-KDA/4-global-layer model amplifies the
memory and scheduling benefit:

| KDA derivative | Mean step | Global tok/s | Tok/s/chip | Compiled memory/chip |
| --- | ---: | ---: | ---: | ---: |
| Generic autodiff | 0.838646 s | 156,290 | 19,536 | 22.9 GB |
| Whole analytical VJP | 0.701620 s | 186,815 | 23,352 | 17.9 GB |

The analytical path is 19.53% faster in throughput, 16.34% lower in step
time, and about 21.8% lower in compiled memory. Loss falls from 7.750 to
0.445 over the measured synthetic-data interval.

The three-step XPlane comparison shows where the model gain comes from:

| Phase | Generic autodiff | Analytical VJP | Difference |
| --- | ---: | ---: | ---: |
| Forward transformer | 150.866 ms | 150.354 ms | -0.512 ms |
| Backward transformer | 668.242 ms | 531.523 ms | -136.719 ms |
| Whole device step | 837.757 ms | 700.450 ms | -137.307 ms |

KDA source-or-stack time falls from 743.492 to 599.729 ms/step, and
loop-fusion HLO time falls from 243.404 to 92.084 ms/step. The analytical
behavior is preserved by
`benchmarks/maxtext_v6e_kda_hybrid_273m_analytical_vjp.yml`; generic autodiff
is preserved by
`benchmarks/maxtext_v6e_kda_hybrid_273m_generic_autodiff.yml`. The main
benchmark now selects the faster fused Pallas path below.

Artifacts:

- `v6e8-kda-analytical-vjp-core-20260720/`
- `v6e8-kda-hybrid-273m-analytical-vjp-adamw-s2048-b8-20260720/`
- `v6e8-kda-hybrid-273m-analytical-vjp-profile-20260720/`

### Fused production-shape Pallas KDA

The new TPU specialization processes one ordered chunk stream per
`(batch, head)` pair. It is fixed to chunk 64 and `K=V=128`, which matches
the production model used here. Forward fuses Q/K normalization, FP32 gate
accumulation, both channel-decayed pairwise matrices, the WY solve, state
read/update, and output. The `128 x 128` state remains in FP32 VMEM across
the ordered chunk grid. Q/K/V enter and output leaves in BF16; decay,
pairwise factors, solves, state, and gradients accumulate in FP32.

The matched custom VJP walks the chunk grid in reverse, carries the FP32
state cotangent in VMEM, loads only the previous chunk-boundary state, and
recomputes the compact intra-chunk values. It fuses the transposed WY solve,
reverse state scan, two pairwise VJPs, reverse gate cumsum, normalization
gradients, and all six input gradients.

The first 16-row serial blocked solve took about 20.45 ms in the otherwise
fused forward. The selected exact solver exploits nilpotence. For
`A = I + L`, it evaluates the finite series
`A^-1 B = (I + P + ... + P^63) B`, `P=-L`, with recursive doubling and dense
MXU matmuls. The transpose VJP uses the corresponding strictly upper
nilpotent series.

Strict native-v6e correctness used `B=1`, `T=128`, one head, two chunks, and
a nonzero initial state:

| Comparison | Output max abs | Final-state max abs |
| --- | ---: | ---: |
| Fused vs token-recurrent reference | 4.88e-4 | 2.33e-3 |
| Fused vs analytical chunked XLA | 4.88e-4 | 3.10e-3 |

The loss differs from analytical XLA by `2.39e-7`. Maximum absolute gradient
differences for Q, K, V, log decay, beta, and initial state range from
`1.61e-9` to `1.04e-7`; every checked value is finite. Eight CPU unit tests
also pass, including exact lower and transposed recursive-doubling solve
checks.

The production one-chip core uses `B=8`, `T=2048`, eight heads, chunk 64,
and 16,384 tokens:

| KDA core | Mean | Tokens/s | Compiled memory |
| --- | ---: | ---: | ---: |
| Analytical XLA forward | 9.665 ms | 1,695,111 | 0.985 GB |
| Fused Pallas forward + boundary states | 6.354 ms | 2,578,567 | 0.579 GB |
| Analytical XLA forward + backward | 31.872 ms | 514,053 | 1.939 GB |
| Fused Pallas forward + backward | 18.287 ms | 895,927 | 0.883 GB |

The fused core improves forward throughput by 52.12%, training throughput by
74.29%, and training compiled memory by 54.46%.

Each stage variant uses the same block traffic and output shapes, so
successive cumulative differences estimate the newly enabled work:

| Forward cumulative stage | Total | Increment |
| --- | ---: | ---: |
| Normalize + gate accumulation | 1.679 ms | 1.679 ms |
| Pairwise construction | 3.169 ms | 1.491 ms |
| WY solve | 4.691 ms | 1.521 ms |
| State update + output | 6.358 ms | 1.667 ms |

| Backward cumulative stage | Total | Increment |
| --- | ---: | ---: |
| Reverse state scan | 7.354 ms | 7.354 ms |
| Transposed solve VJP | 9.995 ms | 2.642 ms |
| Fused pairwise VJPs | 12.631 ms | 2.635 ms |
| Gate/normalization finalization | 12.799 ms | 0.168 ms |

The integrated 272,935,520-parameter 3:1 hybrid uses the same BF16/FP32
AdamW workload and batch eight/chip as the earlier KDA runs:

| KDA backend | Mean step | Global tok/s | Tok/s/chip | Compiled memory/chip |
| --- | ---: | ---: | ---: | ---: |
| Generic XLA autodiff | 0.838646 s | 156,290 | 19,536 | 22.9 GB |
| Analytical XLA VJP | 0.701620 s | 186,815 | 23,352 | 17.9 GB |
| Fused Pallas forward/backward | 0.430733 s | 304,300 | 38,038 | 15.4 GB |

The fused model is 62.89% faster than the analytical-XLA model and 94.70%
faster than generic autodiff. Loss falls from 7.748 to 0.443. It remains
3.30 times behind the matched 1,003,900 tok/s global-attention control, so
KDA still dominates the model.

The selected three-step profile averages 430.747 ms:

| Component | ms/step | % of step |
| --- | ---: | ---: |
| Forward transformer | 99.332 | 23.06% |
| Backward transformer | 314.689 | 73.06% |
| Named fused KDA forward calls | 136.215 | 31.62% |
| Named fused KDA backward calls | 139.752 | 32.44% |
| Four Tokamax Splash layers | 6.961 | 1.62% |

KDA source-or-stack attribution is 345.871 ms (80.30%), and fused custom
calls alone are 275.967 ms (64.07%). The outer decoder-layer checkpoint
places some fused forward recomputation in the backward phase. A targeted
A/B that explicitly retained Q/K/V, decay, beta, and chunk states was
rejected: memory rose from 15.4 to 20.4 GB/chip and throughput fell 2.10% to
297,925 tok/s. The original `minimal_with_context` policy remains selected.

Artifacts:

- `v6e8-kda-fused-pallas-core-20260720/`
- `v6e8-kda-hybrid-273m-fused-pallas-adamw-s2048-b8-20260720/`
- `v6e8-kda-hybrid-273m-fused-pallas-profile-20260720/`
- `v6e8-kda-hybrid-273m-fused-pallas-remat-adamw-s2048-b8-20260720/`

Software:

- MaxText source commit: `dfd8d293d266fe224b90f7cb0b49f3e8084e9892`
- Python: 3.12.13
- JAX / jaxlib: 0.10.2 / 0.10.2
- libtpu: 0.0.42.1
- Flax: 0.12.7
- Optax: 0.2.8

### Stream-collapsed grid and selective BF16 MXU precision

Two changes to the fused kernel, measured against the 895,538 tok/s fused
core and the 304,300 tok/s model of the previous section.

**Grid.** The fused kernel ran `grid=(batch, heads, num_chunks)` with a
`[1,1,C,K]` block, so a v6e core executed 2,048 sequential programs of about
13 MFLOP each: 6.354 ms over 2,048 programs is 3.1 us per program, far more
than the arithmetic justifies. Only the chunk axis carries a real dependency;
batch and head are independent. Folding them into the block leaves 256
programs. Mosaic's `tpu.matmul` accepts one batch dimension, so batch and head
are merged into a single stream axis rather than kept as two leading block
axes. `streams_per_program` above 8 fails with `CompileTimeScopedVmemOom`.

**Precision.** Every in-kernel matmul used `Precision.HIGHEST`, which TPU
evaluates as six BF16 passes. Q/K/V enter in BF16, so outside the solve those
passes refine mantissa bits the operands never carried. `Precision.HIGH`
(three passes) fails to compile in this kernel.

A blanket reduction to one pass is 4.93x faster in the core and diverges to
NaN at model step two. Bisecting by matmul role isolates the cause:

| Guarded at six passes | Model | Loss |
| --- | ---: | --- |
| Everything | 398,061 | 7.749 -> 0.442 |
| All but the state matmuls | 424,458 | 7.750 -> 0.445 |
| The triangular solve only | 560,923 | 7.750 -> 0.444 |
| The pairwise only | NaN at step 2 | diverges |
| The solve's squarings only | NaN at step 2 | diverges |

The repeated-squaring solve is the sole fragile operation. It reaches `P^63`
by squaring `P` six times, so relative error compounds multiplicatively.
Splitting the series by role does not help: the update is
`solution <- (I + P^(2^k)) solution`, so the running solution compounds
exactly as the power does. The mechanism is algorithmic growth rather than
problem conditioning; the two are separated quantitatively below.

The decay-rescaled pairwise operands were the first suspect and were wrong.
Their factors reach `exp(row_block * |gate_lower_bound|)`, but that is well
inside BF16's exponent range and each product is formed once rather than fed
back.

Selected core, `B=8`, `T=2048`, eight heads, chunk 64:

| Core | Forward | Forward + backward | Training tok/s |
| --- | ---: | ---: | ---: |
| Previous fused | 6.361 ms | 18.295 ms | 895,538 |
| Stream-collapsed, guarded solve | 2.360 ms | 6.437 ms | 2,545,320 |
| Stream-collapsed, unguarded (diverges) | 1.462 ms | 3.709 ms | 4,416,880 |

Model step time falls from 0.4307 to 0.2337 s. The FP32 solve is 2.73 ms of
the 6.437 ms core and no precision setting reaches it.

The correctness harness passes on the configuration that NaNs the model, so
it is not sufficient evidence for a precision change. `A` is strictly lower
triangular, so its spectral radius is exactly zero and carries no information;
what matters is norm and power growth. With independent L2-normalized keys in
128 dimensions `|k_i . k_j|` is about `1/sqrt(128)`, so the harness's `||A||`
is well under one, the Neumann terms decay geometrically, and every squaring
past the second operates on a numerically negligible matrix.
`--decay-mode production` was added while investigating and widens log decay to
the configured gate range, but it does not reproduce the failure either.

Two failure mechanisms have to be separated here, and the evidence so far
establishes only the second. A system can be genuinely ill-conditioned, or the
algorithm can be unstable on a well-conditioned system. At the exact 64x64
extreme they come apart sharply. For `A` the positive strictly-lower all-ones
matrix, which is what parallel keys drive toward, `kappa_2(I + A)` is only
82.1 and `max |(I + A)^-1|` is exactly 1, yet `||A^32||_2` is 6.17e17: the
problem is benign and recursive doubling still builds enormous intermediates
that have to cancel. For the negative all-ones matrix `kappa_2(I + A)` is
2.79e17 and the system itself is catastrophically ill-conditioned.

So `||A|| > 1` does not imply large entries in `(I + A)^-1`; that bound is
loose precisely because the Neumann terms cancel. What the model runs prove is
that BF16 recursive doubling is unstable on model-generated systems, not that
those systems are inherently ill-conditioned. Telling the two apart needs
problem conditioning, `kappa_2(I + A)`, measured separately from algorithmic
growth: the sequence `||P||, ||P^2||, ..., ||P^32||`, the largest intermediate
solution norm, and the final relative residual. Precision changes should be
gated on a training run until the harness covers both.

Artifacts:

- `v6e8-kda-hybrid-273m-fused-collapsed-h8-s2048-b8-20260720/`
- `v6e8-kda-diag-chighest-shighest-20260720/`
- `v6e8-kda-diag-chighest-sdefault-20260720/`
- `v6e8-kda-solveguard-20260720/`
- `v6e8-kda-solveapply-bf16-20260720/`
- `v6e8-kda-hybrid-273m-selected-s2048-b8-20260720/`

Next: the solve is the remaining barrier. Recursive doubling was chosen for
dense MXU matmuls but is not backward stable, which is why it alone cannot
take BF16. Blocked forward substitution bounds error growth to `||L||^15`
within a 16-row block instead of `||L||^63` across the chunk while keeping
off-diagonal updates as plain matmuls. The earlier 16-row solve was slow for
an unrelated reason, a row-by-row scalar loop, so a properly blocked version
has not been tried.

### Profile of the selected configuration, and two rejected follow-ups

The 233.564 ms step of the selected 560,923 tok/s configuration divides as:

| Component | ms/step | % of step |
| --- | ---: | ---: |
| Fused Pallas KDA kernels | 86.597 | 37.08% |
| Convolution fusion | 62.080 | 26.58% |
| Loop fusion | 41.591 | 17.81% |
| Data formatting | 19.768 | 8.46% |
| Splash Attention, four layers | 6.958 | 2.98% |

Forward and backward of the fused kernel are 40.519 and 46.078 ms.

**Blocked forward substitution, rejected.** Written to test two claims about
the doubling solve and refuting both. It should cost about a fifth of the
arithmetic, since doubling applies six full-width powers to the `K + V`-wide
right-hand side; measured, it is 1.6% faster at 6.315 against 6.418 ms. The
solve is therefore bound by matmul latency on 16-row blocks and four serial
block steps, not by FLOPs. It should also be better conditioned, capping
growth at `||L||^15` per block against `||L||^63` across the chunk; measured,
one BF16 pass still reaches NaN, at step zero rather than step two.

**Shifted QKV convolution, rejected.** The convolution-fusion line above
suggested the depthwise mixer was mislowered, so it was rewritten as one pad
and four slice-multiply-accumulates. The rewrite is exactly equivalent, 2.4e-7
against the Flax causal convolution, and a perturbation test confirms nothing
leaks backwards in time. It measured 537,292 tok/s against 560,919, 4.2%
slower. The profile category was misread: a convolution fusion node also
carries the SiLU and reshapes XLA fused into it, so it overstates the
convolution, and four full passes over the QKV tensor plus a pad cost more
traffic than the convolution they remove.

Both are retained behind default-off switches with the measurements recorded
at the switch, so neither hypothesis has to be re-derived.

Artifacts:

- `v6e8-kda-selected-profile-20260720/`
- `v6e8-kda-substitution-bf16-20260720/`
- `v6e8-kda-shiftedconv-1-20260720/`
- `v6e8-kda-shiftedconv-0-20260720/`

### Remat policy and batch scaling

The remat policy was re-tested because the two inputs to the earlier rejection
had both moved: the fused kernel is 2.8x cheaper and the step now uses 15.2 of
31.25 GB.

| Policy | Global tok/s | Compiled memory |
| --- | ---: | ---: |
| `minimal_with_context` | 560,919 | 15.2 GB |
| `minimal` | 533,080 | 15.1 GB |
| `save_dot_except_mlp` | 550,776 | 9.2 GB |

`minimal_with_context` remains selected on throughput. The interesting result
is `save_dot_except_mlp`, which gives up 1.8% for 6 GB, because everything
optimized so far reduces per-step cost while batch increases the work each
step amortizes it over. The profile's 41.591 ms of loop fusion and 19.768 ms
of data formatting are largely batch-independent.

| Batch/chip, `save_dot_except_mlp` | Global tok/s | Compiled memory |
| --- | ---: | ---: |
| 8 | 550,776 | 9.2 GB |
| 16 | 582,117 | 13.3 GB |
| 24 | 578,758 | 17.4 GB |

Batch 16 is 3.8% above the batch-8 selected configuration and batch 24
regresses. This is a different operating point rather than a matched
comparison: batch size changes the gradient noise scale and would need the
learning-rate schedule revisited, so the selected batch-8 number remains the
like-for-like result against the 1,003,900 tok/s batch-8 control.

Artifacts:

- `v6e8-kda-remat-minimal-20260720/`
- `v6e8-kda-remat-save_dot_except_mlp-20260720/`
- `v6e8-kda-sdem-b16-20260720/`
- `v6e8-kda-sdem-b24-20260720/`

### Batch and gradient accumulation

All rows use the selected kernel with `save_dot_except_mlp` unless noted.

| Microbatch/chip | `ga` | Effective | Global tok/s | Compiled memory |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 8 | 550,776 | 9.2 GB |
| 12 | 1 | 12 | 571,302 | 11.3 GB |
| 16 | 1 | 16 | 582,117 | 13.3 GB |
| 20 | 1 | 20 | 580,183 | 15.3 GB |
| 24 | 1 | 24 | 578,758 | 17.4 GB |
| 8 | 2 | 16 | 575,364 | 11.3 GB |
| 16 | 2 | 32 | 596,562 | 15.4 GB |
| 16 | 4 | 64 | 600,548 | 15.4 GB |
| 16 | 8 | 128 | 602,362 | 15.4 GB |
| 20 | 4 | 80 | 596,522 | 17.4 GB |
| 16, `minimal_with_context` | 1 | 16 | 568,993 | 25.2 GB |

Two separate effects, and only one of them is free.

Microbatch size has an interior optimum at 16: 12 gains, 20 and 24 lose. Past
16 the larger working set costs more than the amortization returns, and this
is not a memory ceiling.

Gradient accumulation is not itself a throughput win. At a matched effective
batch of 16 it costs 1.2%, 575,364 against 582,117, which is what an overhead
that does not change the update should cost. It wins by reaching effective
batches that do not fit directly while holding the microbatch at its optimum:
16 with `ga=8` reaches 602,362 at 15.4 GB, against 578,758 for a direct batch
of 24 at 17.4 GB.

The gain saturates, at +2.5%, +0.7%, and +0.3% for `ga` of 2, 4, and 8, and
compiled memory is flat in `ga`. Both follow from the mechanism: accumulation
spreads the per-optimizer-step tail, 7.602 ms post-backward plus 5.696 ms
all-reduce, or 5.7% of the step, across more tokens, while per-microbatch
forward and backward are unchanged. A fixed cost can only be spread so thin.

`minimal_with_context` needs 25.2 GB of 31.25 at batch 16, so the selected
policy has no headroom for this direction; batch scaling requires the swap to
`save_dot_except_mlp`.

Every row here is a different training configuration. Batch size changes the
gradient noise scale and the learning-rate schedule would need revisiting, so
the batch-8 `minimal_with_context` result remains the like-for-like number
against the 1,003,900 tok/s batch-8 control.

Artifacts:

- `v6e8-kda-sdem-b12-20260720/`, `v6e8-kda-sdem-b16-20260720/`,
  `v6e8-kda-sdem-b20-20260720/`, `v6e8-kda-sdem-b24-20260720/`
- `v6e8-kda-sdem-b8ga2-20260720/`, `v6e8-kda-sdem-b16ga2-20260720/`,
  `v6e8-kda-sdem-b16ga4-20260720/`, `v6e8-kda-sdem-b16ga8-20260720/`,
  `v6e8-kda-sdem-b20ga4-20260720/`
- `v6e8-kda-mwc-b16-20260720/`

### Separating WY conditioning from recursive-doubling growth

The earlier attribution of the BF16 divergence to an ill-conditioned WY system
conflated two mechanisms that are complementary rather than alternatives:

- recursive-doubling growth predicts the *backward* error of the solve, how
  nearly the computed solution solves the system;
- `kappa_2(I + A)` predicts how much of that backward error is amplified into
  *forward* error in the solution itself.

`benchmarks/diagnose_wy_conditioning.py` measures both. Because `A` is strictly
lower triangular its spectral radius is identically zero and carries no
information, so every measure is a norm or a growth factor. Arithmetic is real
rather than emulated: a TPU matmul at `Precision.DEFAULT` rounds *both*
operands to BF16 for *every* matmul with FP32 accumulation, so each power and
each solution update is re-rounded as it is formed. An earlier version of this
script rounded the inputs once and then computed in FP32, which models
something strictly more accurate and understated the failure, in one regime by
three orders of magnitude.

Chunk 64, `K=V=128`, one BF16 pass:

| Regime | `k2(I+A)` | `max norm(P^k)` | dbl backward | dbl forward | sub backward | sub forward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact extreme: all-ones, positive | 82.1 | 6.17e17 | 4.8e-2 | 1.7e15 | 8.4e-4 | 1.4e-3 |
| exact extreme: all-ones, negative | 2.79e17 | 6.17e17 | 6.0e-6 | 2.4e-3 | 5.9e-19 | 3.2e-3 |
| harness today: independent keys | 1.52 | 0.367 | 2.6e-4 | 5.4e-4 | 2.4e-4 | 5.1e-4 |
| stress: correlated c=0.9, beta 0.95 | 56.2 | 3.04e15 | 6.8e-2 | 1.4e13 | 9.1e-4 | 1.4e-2 |
| stress: correlated c=0.99, beta 0.99 | 69.7 | 2.39e17 | 5.9e-2 | 7.1e14 | 8.4e-4 | 1.3e-2 |
| stress: mixed-sign correlated c=0.9 | 55.7 | 2.63e15 | 6.4e-2 | 1.3e13 | 9.5e-4 | 1.4e-2 |
| stress: AR(1) phi=0.95, beta 0.95 | 24.6 | 3.12e15 | 1.5e-1 | 9.6e12 | 8.9e-4 | 8.8e-3 |
| stress: correlated c=0.9, fast decay | 3.39 | 137 | 7.0e-1 | 1.5 | 6.4e-4 | 2.3e-3 |

Growth is the dominant mechanism in the regimes that motivated this work. The
positive all-ones extreme is a benign problem, `max abs inverse` exactly 1 and
`k2` only 82.1, yet doubling forms powers of norm 6.17e17 and returns a
forward error of 1.7e15. The stress regimes sit at `k2` between 25 and 70,
also benign, with growth of 1e15 to 1e17 and forward errors of 1e12 or worse.
So `norm(A) > 1` does not imply large entries in `(I + A)^-1`; that bound is
loose because the Neumann terms cancel.

Conditioning still has to be reported alongside it. On the negative extreme,
substitution reaches a backward error of 5.9e-19, essentially exact, and still
carries 3.2e-3 forward error, because `k2` of 2.79e17 amplifies it. Backward
error alone would have scored that solve as perfect.

The harness row explains why it passes on a configuration that NaNs the model.
Independent L2-normalized keys in 128 dimensions give `abs(k_i . k_j)` near
`1/sqrt(128)`, so `norm(A)` stays under one, terms decay geometrically, and
every squaring past the second operates on a numerically negligible matrix.

For `k_i = normalize(sqrt(c) * base + sqrt(1 - c) * noise_i)` the expected
pairwise correlation is `c`; using `c` and `sqrt(1 - c**2)` would target
`c**2`.

These correlated, mixed-sign, and AR(1) cases are plausible stress regimes
chosen to bracket the behaviour. They are not measured trained-model
distributions, and should not be described as such until the real-token
instrumentation below exists.

### Substitution solve, re-measured

The first substitution comparison was invalid. Both custom-VJP call sites
hardcoded `solve_method="doubling"`, so the forward never used the selected
method and only the backward switched, and the substitution base case was
itself a nilpotent series. Those runs compared doubling-forward against
doubling-forward, at 16x16 instead of 64x64 in the backward.

With the plumbing fixed and a base case that forms no powers:

| Solve | Core forward+backward | Model |
| --- | ---: | ---: |
| Doubling, FP32 solve (selected) | 6.437 ms | 560,923 |
| Substitution, row-serial base, BF16 | 8.617 ms | not run |
| Substitution, FP32 16x16 base, BF16 coupling | 6.111 ms | 548,450 |

Substitution is the better algorithm numerically and still loses. The stable
row-serial base case costs more than the FP32 doubling it replaces. Splitting
the roles, keeping full passes only for the small diagonal block while the
wide inter-block coupling takes one BF16 pass, is 5.1% faster in the core and
2.2% slower in the model, a core gain that inverts at model level exactly as
the shifted convolution did. Both variants train, at 7.749 to 0.443.

Artifacts:

- `benchmarks/diagnose_wy_conditioning.py`
- `v6e8-kda-subst-hybrid-20260720/`
