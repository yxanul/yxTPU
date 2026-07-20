# yxTPU

JAX, Pallas, XLA, and MaxText experiments for TPU pretraining. This repository
records the code, exact benchmark configurations, compact run artifacts, and
the experiment ledger used while developing a fused Kimi Delta Attention
(KDA) training kernel on TPU v6e.

The vendored `maxtext/` tree starts from
[`AI-Hypercomputer/maxtext`](https://github.com/AI-Hypercomputer/maxtext) at
commit `dfd8d293d266fe224b90f7cb0b49f3e8084e9892`. Local changes add the model
shapes, KDA implementations, fused Pallas kernels, and optimizer support used
by the experiments. MaxText retains its upstream Apache 2.0 license in
[`maxtext/LICENSE`](maxtext/LICENSE).

## Current result

The production specialization uses 64-token chunks, eight heads, and
`K=V=128`. It keeps the `128 x 128` fast-weight state and sensitive
accumulations in FP32 while streaming Q/K/V in BF16.

| Workload | Analytical XLA | Fused Pallas |
| --- | ---: | ---: |
| Core forward + backward | 31.872 ms | 18.287 ms |
| Core training throughput | 514,053 tok/s | 895,927 tok/s |
| Core compiled memory | 1.939 GB | 0.883 GB |
| 272.9M hybrid global throughput | 186,815 tok/s | 304,300 tok/s |
| Full-model compiled memory/chip | 17.9 GB | 15.4 GB |

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the chronological ledger and
[`results/RESULTS.md`](results/RESULTS.md) for the complete comparisons and
profile interpretation.

## Repository layout

- `maxtext/`: vendored MaxText source plus the experimental implementation.
- `benchmarks/`: reproducible TPU setup, training, profiling, and analysis
  scripts.
- `docs/`: algorithm notes and the GDR-to-KDA derivation.
- `results/`: compact metrics, logs, summaries, and profile reports. Raw
  XPlane traces and generated MaxText output trees are intentionally ignored.
- `AGENTS.md`: TPU allocation, provisioning safety, and connection commands.

## Reproduce on the TPU VM

```bash
git clone https://github.com/yxanul/yxTPU.git
cd yxTPU
bash benchmarks/setup_maxtext_tpu.sh
PDB=8 bash benchmarks/run_kda_hybrid_adamw_v6e_273m.sh
```

Run the isolated correctness and production core benchmark:

```bash
cd maxtext
JAX_PLATFORMS=tpu ../.venv/bin/python \
  ../benchmarks/benchmark_pallas_kda_fused.py
JAX_PLATFORMS=tpu ../.venv/bin/python \
  ../benchmarks/benchmark_pallas_kda_stages.py
```

The TPU quota is Spot capacity. Checkpoint durable training runs and delete
idle TPU resources. Never commit cloud credentials, private keys, API tokens,
or signed URLs.
