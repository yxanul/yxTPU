# ejkernel XLA Gated Delta Rule to KDA audit

## Conclusion

The ejkernel XLA forward is a useful scalar-decay control and its analytical
backward contains reusable structure, but neither file can be converted to KDA
by only changing the shape of `decay`.

The active ejkernel forward is already the scalar counterpart of our current
KDA algorithm: it constructs a causal unit-lower WY system, solves for its
inverse, forms U/W, and scans a recurrent state across chunks. KDA requires
the scalar decay factors to move *inside every key-channel contraction*. That
is exactly the additional work responsible for most of our present slowdown.

The most promising reusable component is not ejkernel's complete custom VJP.
It is the analytical inverse derivative and the forward/reverse state-scan
skeleton. A KDA backward must combine those with a new blockwise derivative
for channel-weighted pairwise products that never materializes
`[chunk, chunk, key_dim]`.

## Source audited

- Repository: <https://github.com/erfanzar/ejkernel>
- Commit: `2a426edbd4f88368d9d31d80cf2e4219aa69d7cc`
- Version: `0.0.82`
- Forward:
  <https://github.com/erfanzar/ejkernel/blob/main/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py>
- Backward:
  <https://github.com/erfanzar/ejkernel/blob/main/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py>

There are two materially different paths in these files:

1. `_chunk_gdr_fwd` is the active public multi-token path. Despite its name,
   it delegates to `_recurrent_gdr_fwd`, which is an exact chunked
   triangular-solve implementation. It uses ordinary JAX autodiff and does
   not call `_xla_impl_bwd.py`.
2. `_chunk_gdr_fwd_neumann` is a private custom-VJP path connected to
   `_chunk_gdr_bwd`. Its name and several comments still say “Neumann,” but
   the current code also computes the inverse with an exact unit-lower
   triangular solve. The repository keeps it private because its backward is
   segment-blind and the former approximation was problematic for padded SFT
   batches.

Benchmarking both is therefore necessary to evaluate both linked files.

## Forward mapping

Let `G` denote cumulative log decay inside a chunk. GDR uses scalar
`G[b,h,t]`; KDA uses one decay per key channel, `G[b,h,t,k]`.

| Stage | Scalar GDR | Required KDA change |
| --- | --- | --- |
| Input decay | `[B,H,T]` | `[B,H,T,K]` |
| Cumulative decay | scalar cumsum over tokens | vector cumsum over tokens |
| WY system | `(βᵢ kᵢ·kⱼ) exp(Gᵢ-Gⱼ)` | `Σₖ βᵢ kᵢₖ kⱼₖ exp(Gᵢₖ-Gⱼₖ)` |
| Unit-lower inverse | `A=(I+S)⁻¹` | unchanged |
| U | `A @ (βV)` | unchanged |
| W input | `βK * exp(G)[...,None]` | `βK * exp(G)` elementwise |
| Intra readout | `(qᵢ·kⱼ) exp(Gᵢ-Gⱼ)` | `Σₖ qᵢₖ kⱼₖ exp(Gᵢₖ-Gⱼₖ)` |
| State decay | `state * exp(G_end)` | `state * exp(G_end)[...,K,None]` |
| State-update key | `kᵢ exp(G_end-Gᵢ)` | same expression elementwise in K |
| Scanned state | `[B,H,K,V]` | unchanged |

The triangular solve, U construction, state shape, and chunk scan are
structurally reusable. The system and intra-attention constructions are not:
the scalar implementation first computes a QK/KK dot and then multiplies by
one pairwise decay. KDA must apply channel-specific decay before reducing K.

Our existing `_decayed_pairwise_dot` and `chunk_kda` already implement this
forward transformation with eight-row block matmuls. Copying ejkernel's
forward would reproduce the same algorithm with a less memory-safe
`[C,C,K]` formulation unless its pairwise code were replaced.

## Backward mapping

### Directly reusable

The derivative of the inverse is independent of the decay parameterization.
For `A=(I+S)⁻¹`, ejkernel computes the equivalent of:

```text
dS = -Aᵀ dA Aᵀ
```

restricted to the strict lower triangle. This is preferable to asking generic
autodiff to differentiate through every downstream use of the inverse.

The following structures are also reusable:

- the forward scan that reconstructs each chunk's incoming state;
- the reverse chunk scan carrying `d_state`;
- the U/W and value/beta gradient contractions;
- the Q/K L2-normalization backward;
- the reverse cumsum that maps cumulative-decay gradients back to per-token
  log-decay gradients.

### Required KDA changes

Every scalar gate adjoint becomes a key-channel vector:

```text
d_state_in[k,v] = d_state_out[k,v] * exp(G_end[k])
d_exp_G_end[k]  = Σ_v d_state_out[k,v] * state_in[k,v]
d_exp_G_q[t,k]  = d_q_scaled[t,k] * q[t,k]
d_exp_G_k[t,k]  = d_k_scaled[t,k] * k[t,k]
```

The scalar backward stores and differentiates a pairwise
`decay_mask[B,H,NC,C,C]`. A literal KDA conversion would create
`[B,H,NC,C,C,K]`. At the benchmark shape this is over one billion elements
per layer, so that approach is not viable.

Instead, KDA needs a blockwise pairwise backward. For each row block it should
recompute the stable decay factors and accumulate:

```text
dq_i[k] += Σ_j dP_ij * k_j[k] * E_ij[k]
dk_j[k] += Σ_i dP_ij * q_i[k] * E_ij[k]
dE_ij[k] = dP_ij * q_i[k] * k_j[k]
```

plus the analogous system contribution:

```text
d(β_i k_i[k]) += Σ_j dS_ij * k_j[k] * E_ij[k]
dk_j[k]       += Σ_i dS_ij * β_i k_i[k] * E_ij[k]
dE_ij[k]      += dS_ij * β_i k_i[k] * k_j[k]
```

Since `E_ij[k]=exp(G_i[k]-G_j[k])`, each block accumulates `dG_i += dE*E`
and `dG_j -= dE*E`. No full channel-resolved pairwise tensor needs to survive
the block.

The custom backward must also become segment-aware before it can support
packed training.

## TPU benchmark

The standalone workload is one token-mixer layer on one v6e chip:

- `B=8`, `T=2048`, `H=8`, `K=V=128`
- 16,384 tokens per invocation
- BF16 inputs, FP32-sensitive math, Q/K L2 normalization
- Forward numbers average 20 repetitions at chunk 64; training numbers
  average 10. The chunk-256 schedule uses 10 and 5 repetitions.

These are token-mixer kernel rates, not whole-model pretraining throughput.

### Chunk size 64

| ejkernel path | Forward | Forward tok/s | Forward + backward | Training tok/s | Compiled memory, training |
| --- | ---: | ---: | ---: | ---: | ---: |
| Active exact + autodiff | 2.705 ms | 6,056,429 | 7.290 ms | 2,247,489 | 1.223 GB |
| Private custom VJP | 2.556 ms | 6,410,001 | 9.956 ms | 1,645,682 | 1.353 GB |

The private path is 5.84% faster in forward-only execution, but its complete
training call is 36.57% slower and uses 10.58% more compiled memory. Its
analytical backward is numerically correct for Q, K, V, beta, decay, and the
initial state on the small validation case; it simply does not improve this
TPU shape.

### Public default chunk size 256

| ejkernel path | Forward | Forward tok/s | Forward + backward | Training tok/s | Compiled memory, training |
| --- | ---: | ---: | ---: | ---: | ---: |
| Active exact + autodiff | 12.730 ms | 1,287,065 | 17.277 ms | 948,318 | 1.953 GB |
| Private custom VJP | 12.685 ms | 1,291,610 | 18.817 ms | 870,716 | 1.390 GB |

Chunk 256 is 4.71 times slower in forward and 2.37 times slower in training
than active chunk 64 for this workload. The custom VJP reduces memory at
chunk 256 but remains slower. We should retain chunk 64 for the KDA shape.

### Matched channel-wise KDA control

| Token mixer, chunk 64 | Forward | Forward tok/s | Forward + backward | Training tok/s |
| --- | ---: | ---: | ---: | ---: |
| ejkernel scalar GDR exact/autodiff | 2.705 ms | 6,056,429 | 7.290 ms | 2,247,489 |
| Current MaxText KDA exact/autodiff | 9.657 ms | 1,696,629 | 32.653 ms | 501,764 |

Moving scalar decay inside the key-channel contraction makes the current KDA
core 3.57 times slower forward and 4.48 times slower through backward. This
is the clearest evidence that the next optimization should target the
channel-weighted pairwise forward/backward rather than another standalone
triangular solver.

## Recommended implementation sequence

1. Add a custom VJP around the complete `chunk_kda`, not only around its
   triangular solve.
2. Keep the current numerically safe eight-row forward construction and exact
   XLA inverse path.
3. Recompute the inverse and compact decay factors in backward under
   `jax.checkpoint`; use ejkernel's analytical `dS=-AᵀdAAᵀ`.
4. Port the forward-state reconstruction and reverse-state scan, promoting
   scalar gate adjoints to `[K]` vectors as described above.
5. Implement `_decayed_pairwise_dot_bwd` as blockwise recomputation and
   accumulation. Do not save or construct `[C,C,K]`.
6. Validate loss, final state, and all six gradients against
   `recurrent_kda_reference`, including packed-segment boundaries.
7. Benchmark the one-layer core first. Continue to a 12-KDA-layer model only
   if forward+backward beats the current 32.653 ms and compiled memory falls.
8. If the analytical XLA backward still loses, fuse the same equations in a
   larger Pallas KDA stage; the prior solve-only experiment showed that a
   standalone serial solve is insufficient.

## Implemented analytical KDA result

The recommended whole-operation VJP is now implemented. It retains the
selected pure-XLA forward and adds:

- an explicit `dS=-AᵀdAAᵀ` derivative for the WY inverse;
- a forward scan that reconstructs chunk-entry fast-weight states;
- a reverse chunk scan carrying the state cotangent;
- vector-valued state and cumulative-decay adjoints;
- `_decayed_pairwise_dot_bwd`, which recomputes eight-row decay blocks and
  accumulates Q/K/gate derivatives without constructing `[C,C,K]`;
- explicit Q/K normalization and reverse-cumsum derivatives.

The analytical and generic paths were compared at `B=1`, `T=64`, `H=1`,
`K=V=128` on native v6e. Loss differed by `5.17e-8`; all gradients were
finite; and the largest absolute gradient difference across Q, K, V,
log-decay, beta, and initial state was `1.76e-7`. Seven CPU unit tests cover
the pairwise block derivative and the complete six-input KDA VJP. Packed
segment-boundary resets remain future work; the throughput run uses synthetic
unpacked sequences.

At the production one-chip core shape, the analytical path leaves forward
unchanged and improves the complete training call:

| KDA core, chunk 64 | Forward | Forward + backward | Training tok/s | Compiled training memory |
| --- | ---: | ---: | ---: | ---: |
| Generic autodiff | 9.660 ms | 32.696 ms | 501,100 | 6.614 GB |
| Whole analytical VJP | 9.666 ms | 31.878 ms | 513,962 | 1.930 GB |

That is a 2.57% training-throughput gain and a 70.82% compiled-memory
reduction. The much larger end-to-end effect comes from applying the compact
backward tape across 12 KDA layers:

| 272.9M hybrid | Mean step | Global tok/s | Compiled memory/chip |
| --- | ---: | ---: | ---: |
| Generic autodiff | 0.838646 s | 156,290 | 22.9 GB |
| Whole analytical VJP | 0.701620 s | 186,815 | 17.9 GB |

Throughput improves by 19.53%, step time falls by 16.34%, and compiled memory
falls by about 21.8%. An XPlane comparison attributes 136.72 ms of the
137.31 ms device-step saving to the backward transformer scan; forward time
is effectively unchanged. The analytical path is therefore selected in the
273M benchmark configuration, while the generic path remains available as a
control.

## Artifacts

- Reproducible benchmark:
  `benchmarks/benchmark_ejkernel_gdr_xla.py`
- Whole-KDA generic/analytical benchmark:
  `benchmarks/benchmark_maxtext_kda_core.py`
- Machine-readable results:
  `results/v6e8-ejkernel-gdr-xla-audit-20260720/summary.json`
- Analytical KDA results:
  `results/v6e8-kda-analytical-vjp-core-20260720/summary.json`,
  `results/v6e8-kda-hybrid-273m-analytical-vjp-adamw-s2048-b8-20260720/`,
  and `results/v6e8-kda-hybrid-273m-analytical-vjp-profile-20260720/`

The matched chunk-64 command is:

```bash
python benchmarks/benchmark_ejkernel_gdr_xla.py \
  --source-root vendor/ejkernel \
  --source-commit 2a426edbd4f88368d9d31d80cf2e4219aa69d7cc \
  --paths exact,custom_vjp \
  --chunk-size 64
```
