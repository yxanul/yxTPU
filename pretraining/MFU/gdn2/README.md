# Gated DeltaNet-2 on the v4 KDA kernel — sandbox port

Prototype for **Gated DeltaNet-2** (arXiv:2605.22791, NVIDIA, 2026-05-21) on
`pallas_kda_fused_v4`. Measured 2026-08-20 on `yxtpu-v4-64-train`, worker 0,
its own 4 chips.

**This is a sandbox. The production kernel
`src/yxtpu_pretrain/kernels/kda_fused_pallas_v4.py` is NOT modified.**
`gdn2_fused_pallas_v4.py` is a copy plus `gdn2.patch`; the probe/bench
reconstruct it on the worker by copying the production file and applying the
patch, so the sandbox can never drift from the kernel it claims to derive from.

## The change

KDA (Eq 7) carries one scalar `beta_t` per head-token that decides *both* how
much old content to erase on the key side and how much new value to write:

```
S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t k_t v_t^T
```

Gated Delta Rule-2 (Eq 10) splits it into a channel-wise **erase** gate
`b_t in [0,1]^dk` (key axis) and a channel-wise **write** gate
`w_t in [0,1]^dv` (value axis):

```
S_t = (I - k_t (b_t (.) k_t)^T) D_t S_{t-1} + k_t (w_t (.) v_t)^T
```

Set `b_t = beta_t 1_dk` and `w_t = beta_t 1_dv` and it reduces to KDA exactly
(Eq 47). Decay, WY solve, state recurrence and output are unchanged.

## Why this kernel was already most of the way there

The paper's §B.5 warns that the scalar shortcut — factoring `beta_s` outside
the dot products that accumulate `dA` — breaks for Gated Delta Rule-2, and
calls the gate-aware accumulation its main kernel change.

**Our kernel never used that shortcut.** It forms the gated tensors *before*
the pairwise and the solve, and the system cotangent is

```python
system_cotangent = -_matmul(combined_rhs_cotangent, _transpose(solved))
```

whose operands are the already-gated rhs cotangent and `[U | Y]`. No `beta`
appears at the accumulation site, so Eq 64/65 are structurally already
implemented. The same holds for `_decayed_pairwise_backward_pair`, which takes
the gated `key_beta` as an operand.

The forward likewise already matches the paper's chunk equations:
`_decayed_pairwise_pair(key_beta, query, key, cumulative_decay)` is Eq 87
(the decay factors are applied inside as `gamma_r` on the left and
`gamma_s^-1` on the right), and `combined_rhs = concat(value_beta, erase_rhs)`
solved in one triangular solve is Eq 88's `U = A(W (.) V)`, `Y = A E_bar`.

## What the port actually does

**Phase 1 — rename (semantics-preserving).** Paper §C.2 notes the historical
buffer named `w` is the erase-side auxiliary **Y**, *not* the write gate.
Keeping both would be a live footgun, so:

| old | new | role |
| --- | --- | --- |
| `w` | `y_aux` | Y, erase-side WY auxiliary |
| `w_cotangent` | `y_aux_cotangent` | dY |
| `w_input` | `erase_rhs` | E_bar = gamma (.) (B (.) K) |
| `w_input_cotangent` | `erase_rhs_cotangent` | dE_bar |

**Phase 2 — gate widening.** The gate buffer carries `[b | w]` concatenated,
width `K+V`, mirroring how the kernel already concatenates `combined_rhs`.
One buffer means one fused projection in the layer rather than two.

Forward, at all 4 bodies — two broadcasts become two elementwise products:

```python
gates = beta_ref[0].astype(jnp.float32)
erase_gate = gates[..., :key_dim]     # b_t
write_gate = gates[..., key_dim:]     # w_t
key_beta   = key * erase_gate         # was key * beta[..., None]
value_beta = value * write_gate       # was value * beta[..., None]
```

Backward, at 2 sites — **the entire mathematical change** is deleting two
channel-sums, because the scalar gradient *is* the sum of the vector one:

```python
value_cotangent      = value_beta_cotangent * write_gate   # dV = dZ (.) W   Eq 73
write_gate_cotangent = value_beta_cotangent * value        # dW = dZ (.) V   Eq 73
key_cotangent       += key_beta_cotangent * erase_gate     # dK += dE'(.)B   Eq 75
erase_gate_cotangent = key_beta_cotangent * key            # dB = dE' (.) K  Eq 74
```

Plumbing: gate block specs widen `1 -> key_dim + value_dim` in four places —
`beta_spec`, `reverse_beta_spec`, and **three `chunk_spec(1, ...)` calls in the
split-backward wrapper**. That last group is easy to miss; missing it fails
late with `mul got incompatible shapes (8,64,128),(8,64,0)`, i.e. the write
gate slicing empty.

**Unchanged, deliberately:** the decay gradient path. `key_beta_cotangent =
erase_rhs_cotangent * cumulative_decay_exp` and `cumulative_decay_cotangent +=
erase_rhs_cotangent * erase_rhs` already implement Eq 74-76's gamma accounting
generically, because gamma multiplies the *product* `B (.) K`.

## Results

### Compile — the risk this probe existed to retire

The v4 ISA already refuses the *integrated* backward (Mosaic wants a
sublane-gather relayout v4 lacks), so widening a ref's minor dimension was not
safe to assume.

```
COMPILE: OK (forward + split backward lowered and compiled on v4)
  temp 0.078 GB, args 0.071 GB
```

### Tied-gate equivalence (Eq 47/48) — bitwise on all 8 outputs

With `b = beta*1_K`, `w = beta*1_V`, against the untouched production kernel.
The scalar gradient is recovered as `<dL/db, 1> + <dL/dw, 1>` (Eq 48).

| output | bitwise |
| --- | --- |
| dQ, dK, dV, dlog_decay, dgate, dstate0 | **True** (max_rel 0.000e+00) |
| output, final_state | **True** |

### Cost of the widening (B=1, T=4096, H=16, K=V=128, fwd+bwd)

| kernel | p10 | median |
| --- | ---: | ---: |
| KDA (scalar beta) | 2.744 ms | 2.755 ms |
| GDN-2 (`[b | w]`) | 2.853 ms | 2.862 ms |
| **delta** | **+4.0%** | +0.109 ms |

Cheaper than predicted. HBM traffic for two full-width gate tensors was the
top-ranked risk; at +4% on a kernel that is ~17% of device time (AGENTS.md 8k
profile), that is **~+0.7% end-to-end** — not the blocker it looked like.

## What is NOT validated

- **Untied numerics.** The tied test only exercises the reduction. The general
  path needs the fp64 tokenwise recurrent reference (paper §D.6); the
  references in `layers/kimi_delta_attention.py` (tokenwise `:338`, chunkwise
  XLA `:576,583`) still assume scalar beta and must be ported in lockstep or
  the gate is meaningless.
- **Conv-fold variants.** `pallas_kda_fused_v4_conv` was patched consistently
  but never compiled or run.
- **Layer and model.** `beta_proj` (`emb_dim -> num_heads`, `:1286`) must
  become `emb_dim -> H*(dk+dv)` with sigmoid. Our implementation already
  requires equal key/value head counts and dims (`:1200-1208`), so the paper's
  GQA repeat rule (§C.1) collapses to identical `[B,T,H,128]` shapes.
- **Parameter cost.** +2 qkv-sized projections per KDA layer: ~8.4 M/layer at
  emb 2048, H=16, d=128, i.e. ~200 M over 24 layers. GDN-2 is not free; the
  paper matches parameter counts by compensating elsewhere.
- **Negative-eigenvalue variant.** Scaling the erase gate to `[0,2]` (§3.1)
  doubles `T` entries; `_PAIRWISE_MAX_SAFE_EXPONENT = 85.0` was derived
  assuming beta in `[0,1]` and should be re-checked (a 1-bit exponent change,
  almost certainly fine).
- **End-to-end training.** No loss curve, no convergence claim.

## Reproduce

```bash
# on one worker, its own 4 chips
FLEET_WORKERS=0 ./scripts/fleet.sh launch gdn2probe "$(cat MFU/gdn2/remote_probe.sh)"
FLEET_WORKERS=0 ./scripts/fleet.sh launch gdn2bench "$(cat MFU/gdn2/remote_bench.sh)"
```

`remote_probe.sh` / `remote_bench.sh` embed `gdn2.patch` and the python
gzip+base64, copy the production kernel, apply the patch, and run. Regenerate
the patch after editing the sandbox copy:

```bash
diff -u src/yxtpu_pretrain/kernels/kda_fused_pallas_v4.py \
        MFU/gdn2/gdn2_fused_pallas_v4.py > MFU/gdn2/gdn2.patch
```
