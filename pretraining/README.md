# yxTPU standalone pretraining

`yxtpu-pretrain` owns the 272.9M KDA/NoPE-GQA hybrid, its fused TPU kernel,
optimizer policy, training loop, data adapters, checkpoint metadata, profiles,
and launchers. It imports modern leaf components from the vendored MaxText pin
without importing MaxText's model registry or top-level trainer.

The validated architecture is four `[KDA, KDA, KDA, NoPE-GQA]` cycles with
RMSNorm and fused SwiGLU. KDA uses BF16 Q/K/V traffic and FP32 weights,
chunk-boundary states, and guarded WY solves. `guarded_fp32` is the default;
unsafe BF16 solve variants live only in `benchmarks/`.

## Install

On a TPU VM after cloning the repository:

```bash
cd pretraining
uv sync --extra dev
uv run yx-pretrain doctor --hardware v6e-8
```

The editable `../maxtext` dependency must resolve to the SHA in `MAXTEXT_PIN`.
`doctor` checks the pin, the JAX stack, and the exact device count. It never
creates, resizes, or deletes cloud resources.

## Run

```bash
uv run yx-pretrain train \
  --model kda_hybrid_273m \
  --optimizer adamw \
  --data synthetic \
  --hardware v6e-8 \
  --experiment selected \
  --set train.steps=30
```

Inspect the configuration without initializing a model:

```bash
uv run yx-pretrain config dump --hardware v6e-8
```

`adamw` is the parity default. `muon` routes declared matrix roles through
Muon and every excluded role through AdamW. `muonclip` adds a post-update GQA
QK-Clip adaptation; it is not the original MLA factorization from Kimi K2.

Benchmark profiles disable checkpointing. A real-training experiment must name
an explicit local directory or `gs://` destination; the package never creates
storage or TPU resources implicitly. `../AGENTS.md` is authoritative for
project, zone, quota, Spot/on-demand, and approval policy.

