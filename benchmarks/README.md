# MaxText v6e-8 synthetic pretraining benchmark

This benchmark measures a full forward, backward, and AdamW update for a
271,606,784-parameter Llama-style decoder on random synthetic tokens.

## Default workload

| Item | Value |
| --- | ---: |
| Decoder layers | 12 |
| Model width | 1024 |
| Attention | 8 query heads, 8 KV heads, head dimension 256 |
| MLP width | 2816 |
| Vocabulary | 32768 |
| Logit projection | Untied |
| Precision | BF16 compute, FP32 weights |
| Optimizer | AdamW |
| Sequence length | 2048 |
| Batch per chip | 16 sequences / 32768 tokens |
| Global batch | 128 sequences / 262144 tokens |
| Parallelism | 8-way data parallel |
| Input | Synthetic random tokens; one batch reused |

The first five step indices are treated as warmup. The summary reports both
per-device and global tokens per second from the remaining steps. Compilation
is not included in steady-state throughput.

## Run on the TPU VM

```bash
cd ~/yxTPU
bash benchmarks/setup_maxtext_tpu.sh
bash benchmarks/run_maxtext_v6e_272m.sh
```

Optional overrides:

```bash
PDB=16 SEQ_LEN=2048 STEPS=40 WARMUP_STEPS=8 \
  bash benchmarks/run_maxtext_v6e_272m.sh
```

Results are written below `~/yxTPU/results/`. No checkpoint or object-storage
bucket is created.

Extra MaxText overrides may be supplied as positional arguments:

```bash
bash benchmarks/run_maxtext_v6e_272m.sh remat_policy=minimal
```

Run the Qwix INT8 mini benchmark:

```bash
bash benchmarks/run_qwix_int8_v6e_272m.sh
```

Run the modern 270M BF16 architecture with ordinary AdamW:

```bash
bash benchmarks/run_modern_adamw_v6e_270m.sh
```

Run the identical model with Muon for matrix weights and AdamW for excluded
parameters such as embeddings, logits, norms, and biases:

```bash
bash benchmarks/run_modern_muon_v6e_270m.sh
```

Capture three steady-state Muon+AdamW steps:

```bash
bash benchmarks/run_modern_muon_profile_v6e_270m.sh
```

Run the 272.9M-parameter 3:1 KDA/NoPE-GQA hybrid:

```bash
PDB=8 bash benchmarks/run_kda_hybrid_adamw_v6e_273m.sh
```

Run the same model with the production-shape fused Pallas KDA forward and
backward:

```bash
CONFIG=~/yxTPU/benchmarks/maxtext_v6e_kda_hybrid_273m_fused_pallas.yml \
  PDB=8 bash benchmarks/run_kda_hybrid_adamw_v6e_273m.sh
```

Run strict native-TPU correctness plus the isolated production core A/B, then
measure the cumulative forward and backward stages:

```bash
cd ~/yxTPU/maxtext
JAX_PLATFORMS=tpu ../.venv/bin/python \
  ../benchmarks/benchmark_pallas_kda_fused.py
JAX_PLATFORMS=tpu ../.venv/bin/python \
  ../benchmarks/benchmark_pallas_kda_stages.py
```

Capture three steady-state KDA steps:

```bash
PDB=8 bash benchmarks/run_kda_hybrid_profile_v6e_273m.sh
```

Capture three steady-state BF16 steps with the TPU XPlane profiler:

```bash
bash benchmarks/run_profile_v6e_272m.sh
```

After copying the resulting directory back to this workspace, summarize its
Chrome trace without double-counting the decoder scan's nested `while` events:

```bash
TRACE="$(find results/<profile-run> -name '*.trace.json.gz' -print -quit)"
python3 benchmarks/analyze_xplane_trace.py "$TRACE" \
  --json-output "results/<profile-run>/profile_analysis.json" \
  --markdown-output "results/<profile-run>/PROFILE.md"
```

## Measured on v6e-8

Both runs used 2048-token sequences, 30 optimizer steps, and excluded step
indices 0–4. Throughput is a complete forward, backward, and AdamW update.

| Batch/chip | XLA memory/chip | Mean tokens/s/chip | Mean global tokens/s |
| ---: | ---: | ---: | ---: |
| 8 | 14.8 GB | 126,865 | 1,014,920 |
| 16 | 24.6 GB | 137,536 | 1,100,290 |

Batch 16 is the default because it is 8.4% faster and remains below the
31.25 GB device-memory limit. A batch of 24 is not expected to fit based on
XLA's compiled-memory estimates.

The matching Qwix INT8 smoke test completed normally but was slower for this
small shape: 846,059 global tokens/s and 24.0 GB of compiled device memory.

The 270,046,208-parameter modern workload uses 18 layers, 4:1 fused GQA,
fused SwiGLU, RMSNorm, RoPE, and Tokamax Pallas Splash Attention. At batch 16,
AdamW reached 1,082,149 global tokens/s with 23.0 GB/chip compiled memory.
The matching Muon+AdamW path reached 955,500 tokens/s with 21.4 GB/chip.

The 272,935,520-parameter KDA hybrid uses 12 KDA and four NoPE global-GQA
layers. At batch 8 it reached 156,290 global tokens/s with 22.9 GB/chip,
versus 1,003,900 tokens/s and 14.0 GB/chip for the modern global-attention
control rerun at batch 8. Its profile assigns 88.75% of the step to KDA,
principally decay-weighted block math and the WY triangular solve.

The fused Pallas KDA core reaches 895,927 tok/s through forward and backward,
74.29% above the analytical XLA core, while reducing compiled core memory
from 1.939 GB to 0.883 GB. In the full hybrid it reaches 304,300 global
tok/s at 15.4 GB/chip, 62.89% above the analytical-XLA model.

See `results/RESULTS.md` for the architecture, optimizer and KDA comparisons,
and XPlane profile breakdowns. The KDA algorithm, analytical derivative, and
fused Pallas implementation notes are in `docs/KDA_HYBRID.md`.
