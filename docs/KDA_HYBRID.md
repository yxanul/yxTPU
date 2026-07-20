# Kimi Delta Attention hybrid on TPU

## Status

The 3:1 KDA hybrid, solve-only Pallas experiment, ejkernel GDR audit,
whole-KDA analytical XLA backward, and production-shape fused Pallas training
kernel are complete. The model compiles, trains without NaNs, matches the
references for all six differentiable inputs, and has retained XPlane
profiles for the principal paths.

The selected fused path reaches 304,300 global tokens/s at sequence length
2048 and 15.4 GB/chip. That is 62.89% faster than the analytical-XLA path,
94.70% faster than generic autodiff, and 3.30 times slower than the matched
1,003,900-token/s global-attention control. A standalone SGLang-inspired
Pallas solve was correct but slower at 89,649 tokens/s; fusing the entire
forward and matched reverse scan is what makes the TPU custom call win.

## Objective

Implement and benchmark a dense, approximately 270M-parameter hybrid decoder
with the repeating layer pattern:

```text
KDA → KDA → KDA → global full attention
```

KDA replaces 12 of 16 attention layers, while the remaining four layers use
the already validated Tokamax Splash GQA path. The MLP, precision, optimizer,
vocabulary, sequence length, batch, and data-parallel layout stay as close as
possible to the modern global-attention control.

This is a KDA/GQA research model, not an exact reproduction of the released
48B Kimi Linear model. The paper's global layers are NoPE MLA and its channel
mixers are MoE. This controlled comparison uses NoPE GQA and dense SwiGLU so
that the attention change can be measured on one v6e-8 slice.

## Algorithm

Gated DeltaNet (GDN) updates a fast-weight state with one scalar decay per
token and head:

```text
S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
```

Kimi Delta Attention (KDA) replaces the scalar decay with a diagonal,
per-key-channel decay:

```text
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1}
      + beta_t k_t v_t^T
o_t = S_t^T q_t
```

The state remains `[batch, heads, key_dim, value_dim]`; only the decay changes
from `[batch, time, heads]` to `[batch, time, heads, key_dim]`. The original
gate is:

```text
log(alpha) = -exp(A_log) * softplus(f_proj(x) + dt_bias)
```

where `f_proj` is a low-rank projection with rank equal to the 128-wide head
dimension. Q and K use L2 normalization. Q, K, and V each pass through a
four-token depthwise causal convolution and SiLU.

The benchmark uses FLA's bounded safe-gate form:

```text
log(alpha) = lower_bound * sigmoid(exp(A_log) *
                                   (f_proj(x) + dt_bias))
lower_bound = -5
```

This keeps each per-token log decay in `[-5, 0)`. The bound is required by
the factored block-matmul formulation: without it, a positive reciprocal
exponent can overflow even though the mathematically combined decay is
non-positive. Masking future positions before `exp`, plus the bounded gate,
eliminated both forward and backward NaNs in the stress test and TPU run.

## Kernel inventory

| Source | Hardware | Training backward | Reusable here |
| --- | --- | --- | --- |
| MaxText Qwen3-Next GDN | JAX/XLA, TPU | Yes | Base chunked WY/scan structure |
| `vllm-tpu` ragged GDN | Pallas/Mosaic, TPU | Inference path | Recurrent/decode reference only |
| SGLang-JAX KDA | Pallas/Mosaic, TPU | No; chunked prefill forward only | Blocked triangular solve and VMEM state recurrence |
| FLA KDA | Triton and TileLang, GPU | Yes | Equations, tiling, numerical reference |
| Moonshot FlashKDA | CUTLASS, NVIDIA GPU | No; forward inference only | Algorithm/shape reference only |

SGLang-JAX does provide an open TPU Pallas KDA forward kernel. It is an
inference/prefill path rather than a training kernel: the exported entry point
is `chunk_kda_fwd`, the serving backend invokes it only for `EXTEND`, decode
uses a naive recurrence, and the implementation defines no backward or custom
VJP. The first training implementation here therefore generalized MaxText's
differentiable JAX chunked GDN function. The focused blocked-solve experiment
below supplies a custom VJP for the solve; the later fused training kernel
completes the TPU forward/backward path.

## Implemented TPU path

1. A recurrent KDA reference defines the exact state update.
2. The chunked implementation uses a WY representation and a 64-token
   inter-chunk scan.
3. Decay-weighted pairwise products use eight-row blocks. Each channel
   reduction becomes a TPU-friendly matrix multiplication without
   materializing a `[chunk, chunk, key_dim]` tensor.
4. The KDA module includes fused QKV projection, one fused depthwise QKV
   convolution, SiLU, low-rank decay and output gates, L2-normalized Q/K,
   RMSNorm, and an output projection.
5. The Qwen3-Next heterogeneous container selects KDA on three layers followed
   by one NoPE global GQA layer.
6. The generic control uses `jax.checkpoint` to rematerialize KDA in backward.
   The whole-KDA custom VJP explicitly recomputes compact chunk
   quantities and reconstructs only chunk-boundary states.
7. An optional Pallas backend solves the combined U/W right-hand side with
   16-row blocked forward substitution. Its custom VJP runs explicit
   transposed block back-substitution and masks `dA` to the strict lower
   triangle. It remains an experimental, non-selected backend.
8. The analytical XLA backward differentiates the WY inverse,
   inter-chunk state scan, normalization, and decay cumsum explicitly. Its
   eight-row pairwise VJP recomputes channel-decay blocks without ever
   materializing `[chunk, chunk, key_dim]`.
9. The selected Pallas path fuses the complete production-shape forward and
   matched reverse scan, keeps the state/cotangent in FP32 VMEM, stores only
   boundary states, and recomputes compact intra-chunk values in backward.

The layer implementation lives in
`maxtext/src/maxtext/layers/kimi_delta_attention.py`; the fused kernels live
in `maxtext/src/maxtext/kernels/kda_fused_pallas.py`. Reference and gradient
tests live in `maxtext/tests/unit/kimi_delta_attention_test.py`. All eight
tests pass on the TPU VM's CPU backend.

## Benchmark controls

- 16 decoder layers: 12 KDA and 4 NoPE global GQA
- Width 1024; KDA head dimension 128
- Fused SwiGLU, RMSNorm, untied 32,768-token output
- BF16 compute and FP32 master weights
- AdamW
- Sequence length 2048, 8 sequences/chip, 131,072 global tokens/step
- Synthetic reused batch and eight-way data parallelism
- Five warmup steps, then 25 measured steps

The requested 16 sequences/chip did not fit. Before blockwise matmuls, the
exact workload's compiled temporary estimate was 55.5 GB/chip even with
rematerialization. The selected block-eight implementation compiles at
22.9 GB/chip for eight sequences/chip.

## Throughput result

Both rows below use sequence length 2048, eight sequences/chip, eight-way data
parallelism, BF16 compute, FP32 weights, AdamW, and 25 measured steps after
five warmup steps.

| Model | Parameters | Mean step | Global tok/s | Tok/s/chip | Compiled memory/chip |
| --- | ---: | ---: | ---: | ---: | ---: |
| KDA hybrid, generic autodiff | 272,935,520 | 0.838646 s | 156,290 | 19,536 | 22.9 GB |
| KDA hybrid, whole analytical VJP | 272,935,520 | 0.701620 s | 186,815 | 23,352 | 17.9 GB |
| KDA hybrid, fused Pallas forward/backward | 272,935,520 | 0.430733 s | 304,300 | 38,038 | 15.4 GB |
| 18-layer global-GQA control | 270,046,208 | 0.130563 s | 1,003,900 | 125,488 | 14.0 GB |

The analytical backward first raised KDA throughput by 19.53% and reduced
compiled memory by about 21.8%. The fused kernel then raises throughput by
another 62.89% and reduces compiled memory by another 14.0%. The selected
KDA path remains 69.69% below the global-attention control at this sequence
length.

The loss drop from 7.750 to 0.445 only verifies that the end-to-end update is
finite. It is not a quality result because the synthetic benchmark repeatedly
uses one random batch.

## XPlane profile

Three steady-state KDA steps average 837.757 ms on device:

| Phase | ms/step | % of step |
| --- | ---: | ---: |
| Input and embedding | 1.412 | 0.17% |
| Forward transformer | 150.866 | 18.01% |
| Output head and loss | 5.949 | 0.71% |
| Backward transformer | 668.242 | 79.77% |
| AdamW, metrics, and other post-backward work | 9.308 | 1.11% |

KDA source-or-stack attribution accounts for 743.492 ms, or 88.75% of the
step. The disjoint direct-source groups are:

| KDA group | ms/step | % of whole step |
| --- | ---: | ---: |
| Decay-weighted block matmuls | 315.187 | 37.62% |
| WY triangular solve | 188.753 | 22.53% |
| Intra/inter-chunk recurrence | 84.944 | 10.14% |
| Shard/rematerialization wrapper | 40.826 | 4.87% |
| QKV convolution and gates | 39.028 | 4.66% |
| WY U/W construction | 20.833 | 2.49% |
| Chunk transforms and cumulative decay | 13.525 | 1.61% |

The four Tokamax Splash global-attention layers use only 7.042 ms/step
(0.84%). Optimizing or sparsifying those layers would not materially improve
this run.

The analytical-VJP profile averages 700.450 ms over the same three
steady-state device steps:

| Phase | Generic autodiff | Analytical VJP | Change |
| --- | ---: | ---: | ---: |
| Input and embedding | 1.412 ms | 1.418 ms | +0.006 ms |
| Forward transformer | 150.866 ms | 150.354 ms | -0.512 ms |
| Output head and loss | 5.949 ms | 5.901 ms | -0.048 ms |
| Backward transformer | 668.242 ms | 531.523 ms | -136.719 ms |
| Post-backward AdamW and metrics | 9.308 ms | 9.306 ms | -0.002 ms |
| **Whole step** | **837.757 ms** | **700.450 ms** | **-137.307 ms** |

The profile confirms that the entire meaningful saving is in backward.
KDA source-or-stack attribution falls from 743.492 to 599.729 ms/step.
At the HLO level, loop-fusion time falls from 243.404 to 92.084 ms/step;
the forward and Tokamax Splash paths remain effectively unchanged.

## SGLang-JAX Pallas kernel audit

SGLang-JAX's `kda.py` is a substantial TPU Mosaic/Pallas implementation, not
just a wrapper. The audited repository head is
`91399b45da2c7af6509546c60534de780e0b1d1f`; the KDA file was most recently
changed by `13fc48de4e00635262765b296a7a53c76616b88d`, which added bounded
Stage-1 residency and 32K-token TPU regression tests on 2026-07-20.

Its packed-variable-length forward has four stages:

1. A Pallas tree cumsum activates and accumulates the per-channel gate.
2. An intra-chunk Pallas kernel constructs pairwise decays and performs an
   exact 16-row-block forward substitution for the unit-lower-triangular WY
   system. It solves a combined value/key/identity right-hand side.
3. A sequential-grid Pallas kernel keeps the `K x V` state in FP32 VMEM while
   propagating it across chunks.
4. A final Pallas kernel combines inter-chunk state output with intra-chunk
   output.

This directly addresses the shape of our generic triangular-solve hotspot,
but the complete kernel cannot replace the training path:

- There is no `custom_vjp`, backward kernel, or other KDA gradient code.
- On our JAX 0.10.2 TPU stack, differentiating `chunk_kda_fwd` fails during
  linearization because not all Pallas output primals have reverse-mode
  support.
- It requires packed `B=1` variable-length layout, chunk-aligned repacking,
  `K <= 256`, external Q/K L2 normalization, and several unsupported options
  are asserted off.
- Its `safe_gate` argument is unused by the intra kernel. Bounded gating only
  happens when the caller supplies `lower_bound`; the SGLang serving backend
  currently does not.

An orientation microbenchmark used one v6e device, 16,384 tokens, eight
heads, `K=V=128`, chunk size 64, BF16 inputs, and FP32 state:

| Forward KDA core | Layout | Mean | Tokens/s |
| --- | --- | ---: | ---: |
| SGLang-JAX complete packed forward | `B=1`, eight packed 2048-token segments | 38.523 ms | 425,307 |
| Current MaxText chunk core | `B=8`, `T=2048` | 9.664 ms | 1,695,337 |

Both outputs and final states were finite. This is not a full-model or
perfectly component-matched comparison: SGLang includes variable-length
alignment/gather/scatter and gate activation, while the current core includes
Q/K normalization but receives activated log decay. It does establish that
copying the full inference pipeline would make this shape slower, not faster.

The reusable idea is narrower: replace
`solve(A, I) @ [B_value, B_key]` with a direct blocked solve
`solve(A, [B_value, B_key])`, then give that operation a training VJP. For
`X = A^-1 B`, the backward is:

```text
dB = A^-T dX
dA = -(dB X^T), restricted to the strict lower triangle
```

That permits a second upper-triangular Pallas solve in backward and avoids
materializing the inverse. Our profile attributes 60.133 ms/step of the WY
solve to forward and 128.619 ms/step to backward, so a forward-only transplant
would leave most of that hotspot intact.

## Blocked solve and custom VJP result

The focused training implementation ports the 16-row forward substitution and
adds an explicit block back-substitution for `A^T`. It solves U and W together
with a `64 x 256` right-hand side. The custom derivative is exactly:

```text
dB = solve(A^T, dX)
dA = tril(-(dB @ X^T), -1)
```

The TPU implementation cannot use a simple reversed forward solve because
Mosaic does not lower the required `rev` primitive here, so the transposed
kernel uses explicit descending block and row loops.

Correctness was checked twice on native v6e:

- At the production batched-solve shape, forward differs from XLA by at most
  `9.54e-7`; `dA` by `2.29e-5`; and `dB` by `7.15e-7`.
- A complete `K=V=128` KDA loss and all six gradients match the token-wise
  recurrent reference. The largest absolute gradient error is `1.46e-11`.

For one chip's 2,048 local chunk/head systems:

| Solve path | Forward | Forward + VJP |
| --- | ---: | ---: |
| XLA inverse + two matmuls | 1.883 ms | 2.836 ms |
| XLA direct + custom VJP | 1.576 ms | 3.496 ms |
| Pallas blocked + custom VJP | 16.932 ms | 34.437 ms |

The full 272.9M model gives the same answer:

| Solve backend | Mean step | Global tok/s | Tok/s/chip |
| --- | ---: | ---: | ---: |
| XLA inverse path | 0.838646 s | 156,290 | 19,536 |
| Pallas blocked + custom VJP | 1.462050 s | 89,649 | 11,206 |

The Pallas path is 42.64% lower in throughput. Its profile averages
1,462.402 ms/step: the custom-call HLO category consumes 812.868 ms (55.58%),
with the Pallas solve as its dominant new operation; the complete WY
input/solve region consumes 843.661 ms (57.69%); and backward consumes
1,147.997 ms (78.50%). Compiled memory is unchanged at 22.9 GB/chip.

The implementation is retained as an experimental backend because it is a
correct training reference for future fusion work:

```text
CONFIG=~/yxTPU/benchmarks/maxtext_v6e_kda_hybrid_273m_pallas_vjp.yml \
  bash ~/yxTPU/benchmarks/run_kda_hybrid_adamw_v6e_273m.sh
```

## Fused Pallas training kernel

The requested production-shape kernel is now implemented in
`maxtext/src/maxtext/kernels/kda_fused_pallas.py`. One Pallas program owns an
ordered chunk stream for each `(batch, head)` pair. Forward keeps its
`128 x 128` FP32 fast-weight state in VMEM across 64-token chunks and fuses
Q/K normalization, FP32 gate cumsum, decay-weighted pairwise construction,
the WY solve, state update, and BF16 output.

Backward uses the opposite ordered grid. It carries the FP32 state
cotangent in VMEM, reads only the previous boundary state, recomputes compact
intra-chunk quantities, and fuses reverse state propagation, the transposed
WY derivative, both pairwise derivatives, reverse gate cumsum, Q/K
normalization derivatives, and all input-gradient stores.

Serial 16-row forward substitution remained a poor TPU schedule even inside
the fused kernel. The selected solve uses the exact finite inverse series for
the strictly triangular nilpotent part:

```text
A = I + L
P = -L
A^-1 B = (I + P + ... + P^63) B
```

Recursive doubling evaluates that series in logarithmic depth with dense MXU
matmuls. The transposed backward uses the strictly upper counterpart.

Native TPU correctness against both `recurrent_kda_reference` and the
whole-KDA analytical XLA VJP covers two chunks, nonzero initial state, output,
final state, loss, and all six gradients. Output max error is `4.88e-4`,
final-state max error is at most `3.10e-3`, loss differs by `2.39e-7`, and
gradient max errors range from `1.61e-9` to `1.04e-7`. Eight CPU tests pass.

At the one-chip production core shape (`B=8`, `T=2048`, `H=8`,
`K=V=128`, chunk 64), fused forward takes 6.354 ms versus 9.665 ms for XLA.
Fused forward+backward takes 18.287 ms and reaches 895,927 tok/s versus
31.872 ms and 514,053 tok/s for analytical XLA. Training compiled memory
falls from 1.939 GB to 0.883 GB.

The complete 272.9M 3:1 hybrid reaches 304,300 global tok/s at 15.4 GB/chip,
up 62.89% from the analytical-XLA model's 186,815 tok/s. A profile still
attributes 345.871 ms of the 430.747 ms step to KDA; named fused forward and
backward calls total 275.967 ms. Explicitly saving the custom-VJP residuals
across the outer decoder remat was tested and rejected because it increased
memory to 20.4 GB/chip and reduced throughput to 297,925 tok/s.

Reproduce the selected model run with:

```text
CONFIG=~/yxTPU/benchmarks/maxtext_v6e_kda_hybrid_273m_fused_pallas.yml \
  bash ~/yxTPU/benchmarks/run_kda_hybrid_adamw_v6e_273m.sh
```

The analytical backward remains the equation-level reference and portable
fallback. The main KDA config now selects fused Pallas; the analytical,
generic-autodiff, and solve-only paths each have explicit control configs.

At sequence length 2048 this benchmark primarily measures kernel overhead.
KDA's intended asymptotic advantage should be evaluated with a
sequence-length sweep only after the dedicated kernel is competitive; a
long-context sweep of the current lowering would mostly measure the known
implementation bottleneck.

## Stream-collapsed grid and selective BF16 precision

Two follow-ups to the fused kernel take the 272.9M hybrid from 304,300 to
560,923 global tok/s with the loss curve unchanged.

### Grid occupancy

The fused kernel launched `grid=(batch, heads, num_chunks)` against a
`[1, 1, chunk, key_dim]` block, so one v6e core walked 2,048 sequential
programs that each did roughly 13 MFLOP. At 6.354 ms that is 3.1 us per
program, far above what the arithmetic accounts for, and the cost is grid and
DMA bookkeeping rather than math.

Only the chunk axis carries a dependency: the state recurrence is ordered
within a stream, and batch and head are independent. Folding them into the
block leaves 256 programs. Mosaic's `tpu.matmul` accepts a single batch
dimension, so the two are merged into one stream axis instead of being kept as
two leading block axes; `streams_per_program` above 8 fails with
`CompileTimeScopedVmemOom` because the per-chunk intermediates scale with it.

### Precision

Every in-kernel matmul used `Precision.HIGHEST`, which TPU evaluates as six
BF16 passes. Q and K and V arrive in BF16, so outside the solve those passes
refine mantissa bits the operands never carried. `Precision.HIGH`, the
three-pass mode, fails to compile in this kernel.

Reducing every matmul to one pass is 4.93x faster in the core and reaches NaN
at model step two. Bisecting by matmul role isolates a single cause:

| Guarded at six passes | Model | Outcome |
| --- | ---: | --- |
| Everything | 398,061 | trains |
| All but the state matmuls | 424,458 | trains |
| The triangular solve only | 560,923 | trains |
| The pairwise only | - | NaN at step 2 |
| The solve's squarings only | - | NaN at step 2 |

The repeated-squaring solve is the only fragile operation. It forms `P^63` by
squaring `P` six times, so relative error compounds multiplicatively. Splitting
the series by role does not rescue it, because the update is
`solution <- (I + P^(2^k)) solution` and the running solution compounds exactly
as the power does.

The mechanism is algorithmic growth rather than problem conditioning, and the
two are complementary rather than alternatives. `A` is strictly lower
triangular, so its spectral radius is identically zero and says nothing; what
matters is norm and power growth. Growth predicts the *backward* error of the
solve, how nearly the computed solution solves the system, while
`kappa_2(I + A)` predicts how much of that backward error is amplified into
*forward* error in the solution. In the stress regimes that motivated this
work, `kappa_2` is only about 25 to 70, a benign problem, while the powers
reach 1e15 to 1e17 and one BF16 pass gives 5 to 15 percent backward error and
forward error of 1e12 or worse. Both must be reported: on the genuinely
ill-conditioned extreme, substitution reaches a backward error of 5.9e-19 and
still carries 3.2e-3 forward error. See
`benchmarks/diagnose_wy_conditioning.py`.

The decay-rescaled pairwise operands were the first hypothesis and were wrong.
Their factors reach `exp(row_block * |gate_lower_bound|)`, but that sits well
inside BF16's exponent range, and each product is formed once rather than fed
back into itself.

### What the correctness harness does not cover

The harness passes on the configuration that NaNs the model, so it cannot by
itself justify a precision change. Its synthetic system has a small spectral
radius: L2-normalized keys and `beta = sigmoid(normal)` give an `A` whose
series converges within two terms, so `P^2` is already negligible and the
squarings are never stressed. A `--decay-mode production` option was added
while investigating and widens log decay to the configured gate range, but it
does not reproduce the failure either. The missing axis is the conditioning of
the WY system rather than the decay range. Until the harness covers that,
precision changes should be gated on a training run.

### Remaining barrier

The FP32 solve is 2.73 ms of the 6.437 ms core and no precision setting
reaches it. Recursive doubling was chosen because it is entirely dense MXU
matmuls, but it is not backward stable, which is exactly why it alone cannot
take BF16. Blocked forward substitution bounds error growth to `||L||^15`
inside a 16-row block instead of `||L||^63` across the chunk while leaving the
off-diagonal updates as plain matmuls. The earlier 16-row solve was slow for
an unrelated reason, a row-by-row scalar dependency chain, so a properly
blocked version has not been measured.

## Artifacts

- Throughput:
  `results/v6e8-kda-hybrid-273m-block8-safe-adamw-s2048-b8-20260720/`
- Matched global-attention control:
  `results/v6e8-modern-270m-adamw-matched-s2048-b8-20260720/`
- XPlane trace and generated analysis:
  `results/v6e8-kda-hybrid-273m-block8-safe-profile-s2048-b8-20260720/`
- SGLang-JAX forward-kernel audit:
  `results/v6e8-sglang-jax-kda-forward-audit-20260720/`
- Triangular solve microbenchmark:
  `results/v6e8-kda-triangular-solve-ab-20260720/`
- Pallas/custom-VJP throughput:
  `results/v6e8-kda-hybrid-273m-pallas-vjp-adamw-s2048-b8-20260720/`
- Pallas/custom-VJP profile:
  `results/v6e8-kda-hybrid-273m-pallas-vjp-profile-s2048-b8-20260720/`
- ejkernel XLA GDR audit and KDA transformation plan:
  `results/v6e8-ejkernel-gdr-xla-audit-20260720/` and
  `docs/EJKERNEL_GDR_TO_KDA.md`
- Whole-KDA analytical VJP core benchmark:
  `results/v6e8-kda-analytical-vjp-core-20260720/`
- Analytical-VJP model throughput:
  `results/v6e8-kda-hybrid-273m-analytical-vjp-adamw-s2048-b8-20260720/`
- Analytical-VJP XPlane profile:
  `results/v6e8-kda-hybrid-273m-analytical-vjp-profile-20260720/`
- Fused Pallas core correctness, production A/B, and staged timings:
  `results/v6e8-kda-fused-pallas-core-20260720/`
- Fused Pallas model throughput:
  `results/v6e8-kda-hybrid-273m-fused-pallas-adamw-s2048-b8-20260720/`
- Fused Pallas XPlane profile:
  `results/v6e8-kda-hybrid-273m-fused-pallas-profile-20260720/`
- Rejected selective-residual remat A/B:
  `results/v6e8-kda-hybrid-273m-fused-pallas-remat-adamw-s2048-b8-20260720/`

## Primary references

- Kimi Linear paper: <https://arxiv.org/abs/2510.26692>
- Official Kimi Linear repository:
  <https://github.com/MoonshotAI/Kimi-Linear>
- Official released KDA implementation:
  <https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda>
- MaxText Qwen3-Next GDN implementation:
  <https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/models/qwen3.py>
- SGLang-JAX TPU Pallas KDA forward:
  <https://github.com/sgl-project/sglang-jax/blob/main/python/sgl_jax/srt/kernels/kda/kda.py>
