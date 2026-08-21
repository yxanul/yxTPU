# MFU on TPU v4 — measured envelope for dense and MoE

Measured 2026-08-19 on the `yxtpu-v4-64-train` slice. Everything here is a
real on-device measurement, not an estimate.

## Contents

- This file: dense and MoE MFU measurements on v4 (sections 1-5).
- [`gdn2/`](gdn2/README.md): sandbox port of **Gated DeltaNet-2**
  (arXiv:2605.22791) onto the v4 KDA Pallas kernel — compiles on v4, reduces
  to KDA bitwise under tied gates, +4.0% kernel cost. Does not modify the
  production kernel.

## Setup

| | |
| --- | --- |
| Hardware | `yxtpu-v4-64-train`, `us-central2-b`, `v4-64` = 32 chips, topology `2x4x4` |
| Chips used | **worker 0 only, its own 4 chips** (a v4-8 equivalent); the other 7 workers were never touched |
| Isolation | `TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_DEVICES=0,1,2,3` |
| Trainer | vendored MaxText (`maxtext/`, upstream pin `dfd8d293`), `maxtext.trainers.pre_train.train` |
| Repo / JAX | worker `main` @ `88a30d7`, jax 0.10.2 |
| Data | `dataset_type=synthetic reuse_example_batch=1` — no HF streaming, no quota exposure |
| Common | bf16 compute, adamw, FSDP over 4 chips (`ici_fsdp_parallelism=4`), seq 2048 unless noted, 22-25 steps, steady-state step read after warmup |

**MFU denominator: 275 TFLOP/s/chip** (v4 bf16 peak). This is the convention
AGENTS.md already uses (105 TFLOP/s = 38%). MaxText's `TFLOP/s/device` is
per JAX device = per chip (megacore presents 1 device/chip).

Convention note: MaxText counts **model FLOPs** — for MoE that means *active*
FLOPs (`num_experts_per_tok` x expert FFN + gate), the standard convention.
Rematerialised recompute is **not** counted, which is why remat policy moves
the number.

---

## 1. Best dense <1B: **165.5 TFLOP/s/device = 60.2% MFU**

All dims are multiples of 128 (v4 MXU is 128x128; VPU is (8,128)).

| Shape | Params | PDB | s/step | TFLOP/s | MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| emb 2048, mlp 2816, 24L — FFN-starved baseline | 0.798 B | 8 | 0.692 | 118.4 | 43.1% |
| emb 2048, mlp 5632, 16L | 0.853 B | 8 | 0.585 | 143.6 | 52.2% |
| emb 2048, mlp 5632, 16L | 0.853 B | 16 | 1.136 | 147.8 | 53.8% |
| emb 2560, mlp 6912, 12L | 0.990 B | 16 | 1.191 | 160.2 | 58.3% |
| **emb 3072, mlp 8192, 9L, tied embeddings** | **0.976 B** | 16 | 1.227 | **165.5** | **60.2%** |
| emb 3072, mlp 8192, 9L, untied | 1.074 B | 16 | 1.212 | 167.6 | 60.9% |

Winning recipe:

```
remat_policy=minimal
attention=flash                 # -> jax Pallas TPU splash attention
weight_dtype=bfloat16
logits_via_embedding=true       # keeps the 3072 shape under 1B
num_vocab_tiling=1              # OFF - see section 4
per_device_batch_size=16
max_target_length=2048
LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
```

### Dense levers, ranked

| Lever | Effect | Note |
| --- | ---: | --- |
| **FFN aspect ratio / width** | **+40% rel** | 43.1% -> 60.2% going emb 2048/mlp 2816 (0.798 B) -> emb 3072/mlp 8192 tied (0.976 B), both under 1 B. The single biggest lever. |
| **splash vs `dot_product`** | **+23% rel** | 50.5% vs 41.2% on the identical model. `attention=flash` is load-bearing even at seq 2048. |
| `remat_policy=minimal` | +5% | Default `full` recomputes everything; recompute FLOPs are uncounted, so it depresses MFU twice over. Plenty of HBM at <1B on 4 chips. |
| XLA flags | +3% | `--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE`, from `configs/tpu/v4/22b.sh`. |
| PDB 8 -> 16 | +1.5% | Small but free if memory allows. |
| `weight_dtype=bfloat16` | +0.2% | Inside noise. FSDP all-gather is not the bottleneck at this scale. |
| seq 2048 -> 4096 | **-8%** | 50.5% -> 46.4%, even with splash. |
| `num_vocab_tiling=8` | **-2.5% / -8%** | Costs at 32k vocab and costs *more* at 128k. See section 4. |

**Width beats depth.** Fewer, fatter GEMMs feed the 128x128 MXU far better
than more, thinner ones. The curve was still climbing at emb 3072, so the
plateau is above 1B.

---

## 1b. The <=500M / 18-24 layer regime

Constraining to ~500M params **and** 18-24 layers forces a narrow model, which
runs against the width finding above. Measured cost: ~8 points of MFU versus
the 1B optimum — but **1.8x the tokens/s**, because a smaller model spends
fewer FLOPs per token.

Settings for all rows: PDB 16, tied embeddings, `weight_dtype=bfloat16`,
`remat_policy=minimal`, XLA flags, splash, seq 2048, 4 chips.
`rel tok/s` is relative to the 0.976 B / 60.2% MFU best-MFU point
(26,714 tok/s/device), re-run in the same session as the anchor.

### GPT-2 vocab (50,304 = 393x128)

| Config | emb | L | mlp | mlp/emb | Params | TFLOP/s | MFU | tok/s/dev | rel tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **e1280 L18** | 1280 | 18 | 4608 | 3.60 | 0.454 B | 144.0 | **52.3%** | **47,901** | **1.79x** |
| e1280 L20 | 1280 | 20 | 4608 | 3.60 | 0.497 B | 143.8 | 52.3% | 43,623 | 1.63x |
| **e1024 L24** (full 24 layers) | 1024 | 24 | 4608 | 4.50 | 0.454 B | 138.7 | **50.4%** | 45,828 | 1.72x |
| e1536 L18 | 1536 | 18 | 3584 | 2.33 | 0.474 B | 135.1 | 49.1% | 42,466 | 1.59x |
| e1280 L24 | 1280 | 24 | 3584 | 2.80 | 0.489 B | 132.7 | 48.3% | 40,073 | 1.50x |

### vocab 32,000

| Config | emb | L | mlp | mlp/emb | Params | TFLOP/s | MFU | tok/s/dev | rel tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e1280 L18 | 1280 | 18 | 4608 | 3.60 | 0.430 B | 143.0 | 52.0% | **49,918** | **1.87x** |
| e1536 L18 | 1536 | 18 | 4096 | 2.67 | 0.488 B | 137.1 | 49.9% | 41,972 | 1.57x |
| e1536 L20 | 1536 | 20 | 3584 | 2.33 | 0.490 B | 134.0 | 48.7% | 40,419 | 1.51x |
| e1024 L24 | 1024 | 24 | 4096 | 4.00 | 0.398 B | 133.3 | 48.5% | 49,580 | 1.86x |
| e1280 L24 | 1280 | 24 | 3456 | 2.70 | 0.454 B | 131.8 | 47.9% | 42,514 | 1.59x |
| e2048 L18 | 2048 | 18 | 2048 | 1.00 | 0.481 B | 123.3 | 44.8% | 36,950 | 1.38x |

### vocab 128,256 (SuperBPE) — the embedding eats the budget

At 128k the tied embedding is 26% of a 500M budget, so the body must shrink.

| Config | emb | L | mlp | mlp/emb | Params | TFLOP/s | MFU | tok/s/dev | rel tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e1280 L18 | 1280 | 18 | 3840 | 3.00 | 0.500 B | 143.0 | 52.0% | 43,517 | 1.63x |
| e1536 L18 | 1536 | 18 | 2432 | 1.58 | 0.498 B | 131.6 | 47.8% | 39,552 | 1.48x |
| e1024 L24 | 1024 | 24 | 4096 | 4.00 | — | **OOM** | — | — | 34.05 G > 30.75 G at PDB 16 |

### What drives it

1. **`mlp/emb` ratio is the dominant lever at fixed budget, not emb width.**
   At 18 layers, vocab 32k: ratio 1.00 -> 44.8%, 2.33 -> 49.1%, 2.67 -> 49.9%,
   3.60 -> 52.0%. Spend the parameter budget on a **fat FFN**, not a wide
   residual stream. This is the same MXU effect as section 1, expressed under
   a constraint: a fat FFN keeps the per-layer GEMMs large even when `emb`
   is small.
2. **Depth is cheap if the FFN stays fat.** Going 18 -> 24 layers costs only
   ~2 points (52.3% -> 50.4%) and 4% of throughput *if* you buy the layers by
   narrowing `emb` (1280 -> 1024) while holding mlp at 4608. Buying them by
   thinning the FFN instead (mlp 4608 -> 3584 at emb 1280) costs 4 points.
3. **GPT-2 vocab padding 50257 -> 50304 is worth ~+0.3%** (143.35 -> 143.79),
   i.e. inside noise. The classic padding trick does not pay here — XLA
   handles the ragged vocab dim fine. Pad for tidiness, not for speed.
4. **PDB 16 is the ceiling** at `remat_policy=minimal` on 32 GB/chip: PDB 24
   (32.46 G) and PDB 32 (43.03 G) both OOM. Activations do not shard under
   FSDP, so this ceiling does not move on a larger slice — only a heavier
   remat policy would raise it, at an MFU cost.

**Recommendation.** For ~500M with >=18 layers: **emb 1280 / mlp 4608 / 18
layers / 10 q heads / 2 kv heads / head_dim 128, tied embeddings** —
0.454 B at 52.3% MFU and 47,901 tok/s/device. If a full 24 layers is
required, **emb 1024 / mlp 4608 / 24 layers** gives 0.454 B at 50.4% and
45,828 tok/s — only 4% less throughput.

Extrapolated to the full 32-chip slice that is ~1.53 M tok/s, though
cross-host collectives will take a bite (see Caveats).

---

## 2. MoE envelope: **~26% MFU best case**, and only with few, large experts

Equal-active-FLOPs A/B against dense: dense `mlp_dim=2816` vs MoE 8 experts x
1408 top-2. Both reported the same 81.9 TFLOP/step/device, so step time *is*
the tax.

| Config | Params | PDB | s/step | TFLOP/s | MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense reference (same counted FLOPs) | 0.798 B | 8 | 0.692 | 118.4 | **43.1%** |
| **MoE megablox GMM, dropless** | 2.044 B | 8 | 1.131 | 72.4 | **26.3%** |
| MoE + `use_ragged_sort=true` | 2.044 B | 8 | 1.131 | 72.4 | 26.3% |
| MoE megablox GMM | 2.044 B | 16 | 2.326 | 70.5 | 25.6% |
| MoE, `remat_policy=full` | 2.044 B | 8 | 1.209 | 67.7 | 24.6% |
| MoE, `dense_matmul` + `capacity_factor=1.25` | 2.044 B | 8 | 1.234 | 66.4 | 24.1% |
| MoE, `megablox=false` (jax `lax.ragged_dot`) | 2.044 B | 8 | 1.537 | 53.3 | 19.4% |

**MoE tax = 1.63x step time at identical counted FLOPs.** The cost is
permute/unpermute, ragged group padding and the router — none of which are
counted as FLOPs.

### MoE lever findings

- **`megablox=true` is load-bearing**: the Pallas GMM beats jax `ragged_dot`
  by 36% (72.4 vs 53.3). Never disable it on v4.
- **Token dropping is not a speedup**: the `dense_matmul` + capacity-factor
  path is *slower* than dropless GMM. No reason to use it here.
- **`use_ragged_sort` is exactly neutral** (72.428 vs 72.436). Skip it.
- **Batch size does not amortise the overhead**: 26.3% at PDB 8 *drops* to
  25.6% at PDB 16. The MoE cost is per-token bandwidth work, not a fixed
  cost you can dilute. This constrains everything else.

### Fine-grained routing collapses on v4

Modern MoEs are fine-grained (Qwen3-30B-A3B: 128 experts; DeepSeek: 256;
Qwen3.5-35B-A3B: 256). On v4 that shape falls apart:

| Config | Params | PDB | s/step | TFLOP/s | MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 experts, top-2, moe_mlp 1408 | 2.044 B | 4 | 0.626 | 65.5 | **23.8%** |
| 64 experts, top-8, moe_mlp 352 (2.75x128, misaligned) | 3.708 B | 4 | 1.578 | 26.0 | **9.5%** |
| 64 experts, top-8, moe_mlp 384 (3x128, aligned) | 4.010 B | 4 | 1.586 | 27.1 | **9.9%** |
| 64 experts, top-8, moe_mlp 352, PDB 8 | — | 8 | OOM | — | 36.31 G temporaries > 30.75 G HBM |

**MXU alignment is not the cause.** 352 -> 384 moved MFU only 9.5% -> 9.9%,
and the step time was *identical* (1.578 vs 1.586 s) despite 384 doing ~9%
more FLOPs. That is the tell: the fine-grained path is not GEMM-bound, so
tiling barely registers. The cost scales with `num_experts x top_k` through
the permute/unpermute, and v4's 1,200 GB/s HBM cannot feed it.

**Consequence: the v4 MoE envelope is few, large experts (8-16, top-2).**
A modern fine-grained MoE is a v5e/v6e workload — the TRC grant has 64
unused v6e Spot chips in `europe-west4-a` and `us-east1-d`.

---

## 3. Kernel availability on v4 — all present

- **megablox GMM trains**: `kernels/megablox/ops.py:103-105` wires a real
  `jax.custom_vjp` (`_gmm_fwd` / `_gmm_bwd`). Forward *and* backward ran on
  v4 across seven configurations with zero kernel-level failures.
- `kernels/megablox/common.py:46` gates bf16xbf16 GMM on
  `tpu_generation() >= 4` — v4 qualifies. Probe confirmed:
  `tpu_gen 4, bf16_gmm True`.
- The `base.yml:247` comment "megablox/jax ragged dot - supports forward pass
  only" refers to the **tile-size tuning knobs** (megablox exposes 6 forward
  tiles, tokamax 18), *not* to pass support. It is not a training limitation.
- `attention=flash` dispatches to
  `jax.experimental.pallas.ops.tpu.splash_attention` (`attention_op.py:32`).

## 4. Cross-entropy: MaxText has no Pallas CE kernel

MaxText's answer to the unfused loss head is `num_vocab_tiling` — a
`jax.custom_vjp` chunked cross-entropy (`utils/vocabulary_tiling.py:134`)
that scans over vocab tiles and never materialises the full `[B*T, vocab]`
logits. Pure XLA, so unlike tokamax's linear-CE (hard-blocked on v4, see
AGENTS.md) **it runs on v4**. But it is a memory feature, not a speed one:

| Vocab | `num_vocab_tiling` | TFLOP/s | MFU |
| ---: | ---: | ---: | ---: |
| 32,000 | 1 | 124.7 | 45.4% |
| 32,000 | 8 | 121.6 | 44.2% |
| 128,256 | 1 | 148.8 | 54.1% |
| 128,256 | 8 | 137.0 | 49.8% |

It costs 2.5% at 32k and **8% at our real 128,256 SuperBPE vocab** — the
opposite of the expected direction. Leave it at 1 unless HBM-bound.

Note the 128k-vocab model scores *higher* MFU than the 32k one on the same
body (54.1% vs 52.2%): the large logits GEMM is efficient work.

---

## 5. Traps and bugs found

1. **`padded_base_moe_mlp_dim` is broken at pin `dfd8d293`.** The knob
   MaxText documents as padding the expert dim "for efficient GMM_v2 kernel
   execution" (`layers/moe.py:527`) crashes:
   `ValueError: Custom VJP bwd rule must produce an output with the same type
   as the args tuple of the primal function`. The GMM backward does not
   handle the padded dim. **Choose 128-aligned expert dims directly instead.**

2. **mixtral `mlp_dim` / `moe_mlp_dim` mismatch.** For
   `decoder_block=mixtral`, the layer uses `mlp_dim` (`models/mixtral.py:115`)
   but the FLOP counter uses `moe_mlp_dim` (`utils/maxtext_utils.py:1166`).
   Set both to the same value or your MFU is silently wrong. The shipped
   `mixtral-8x7b.yml` keeps them equal for exactly this reason.

3. **There is no ready-made v4 MoE config.** `configs/tpu/v4/{22b,52b}.sh` are
   dense; the MoE benchmark presets in
   `benchmarks/maxtext_{trillium,v5e,v5p}_model_configs.py` have **no v4
   entries**; `deepseek3-tiny` / `deepseek4-tiny` are unit-test debug shapes
   (emb 64). `configs/gpu/models/mixtral_8x1b.yml` is the only small-MoE test
   config shipped, and it is GPU-flavoured (needs `hardware=tpu
   attention=flash`). Compose from `base.yml` + CLI overrides.

4. **Sizing a dense baseline to match a MoE's active FLOPs makes it
   FFN-starved.** `mlp_dim=2816` at emb 2048 is 1.375x, versus the normal
   ~2.75x. It cost ~5 points of MFU (45.4% vs 50.5% at matched
   settings) and made the first dense reference look worse than the
   hardware can do.

---

## Caveats

- **4 chips, no cross-host collectives.** A real v4-64 run adds gradient
  all-reduce (and all-to-all if expert-parallel) and will land below these
  numbers. Treat them as device-efficiency ceilings.
- These are MaxText's dense/MoE paths, **not** our KDA + AttnRes production
  loop, which sits at 24-26% model-FLOPs MFU for reasons that are
  kernel-bound (bandwidth) rather than shape-bound.
- Model-FLOPs convention; recompute uncounted. Step times were extremely
  stable (+-0.002 s over 20+ steps), so single-run numbers are reliable here.

## Reproduction

The exact scripts that produced every number above are archived in
`MFU/scripts/` (01-10, in the order they were run). Each is a
self-contained bash script meant to be shipped by
`scripts/fleet.sh launch <name> "$(cat MFU/scripts/0N_*.sh)"`.
The single best dense point reproduces as:

```bash
# on one worker, its own 4 chips
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_DEVICES=0,1,2,3
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
.venv/bin/python -m maxtext.trainers.pre_train.train \
  ~/yxTPU/maxtext/src/maxtext/configs/base.yml run_name=dense_best \
  base_output_directory=/tmp/mfu enable_checkpointing=false \
  dataset_type=synthetic reuse_example_batch=1 enable_dropout=false steps=22 \
  attention=flash remat_policy=minimal weight_dtype=bfloat16 \
  logits_via_embedding=true num_experts=1 \
  base_emb_dim=3072 base_num_query_heads=24 base_num_kv_heads=4 head_dim=128 \
  base_mlp_dim=8192 base_num_decoder_layers=9 \
  max_target_length=2048 per_device_batch_size=16 ici_fsdp_parallelism=4
```

Launch via `scripts/fleet.sh` with `FLEET_WORKERS=0`; verify the slice is idle
with `fleet.sh procs` first and after.
