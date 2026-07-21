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

The certified model contains exactly 272,935,520 trainable parameters. Its
four scanned cycles keep the parameter scan axis at position 1. The GQA layer
owns a single unequal-head QKV projection and calls MaxText's leaf
`AttentionOp` with Tokamax Splash on TPU; CPU correctness tests use the same
causal NoPE-GQA equation directly.

The two validated benchmark operating points are:

```bash
# Batch 8 per device, minimal_with_context rematerialization.
uv run yx-pretrain benchmark --hardware v6e-8 --experiment selected

# Microbatch 16 per device, accumulated 8 times (effective batch 128/device).
uv run yx-pretrain benchmark --hardware v6e-8 --experiment max_throughput
```

The default output loss remains the standard projected-logit cross-entropy.
An opt-in Tokamax fused linear cross-entropy avoids materializing logits in the
owned graph and has an explicit data-parallel reduction:

```bash
uv run yx-pretrain benchmark \
  --hardware v6e-8 \
  --experiment max_throughput \
  --set model.loss.implementation=tokamax_fused
```

On v6e-8 this is a capacity option, not the throughput default. It made
microbatch 64/GA=2 compile with full rematerialization when the standard loss
missed available HBM by 84.6 MB, but every matched full-model comparison was
1.5% to 2.6% slower. Standard loss at microbatch 16/GA=8 remains selected at
598.5k tok/s. The complete parity and batch sweep is EXP-030 in
`../EXPERIMENTS.md`.

`sequence_sweep` is the 16,384-token crossover profile. Change
`data.sequence_length` and `data.per_device_batch_size` with `--set` for the
other measured points.

## Real data and checkpoints

Hugging Face records may come from `data.dataset_name`, or from an offline
JSONL `data.dataset_path`. Each record contains either `input_ids` or `text`;
text records additionally require `data.tokenizer`. The Grain adapter accepts
the same offline format and saves Grain's native iterator state.

Real training must opt out of benchmark mode and provide checkpoint storage:

```bash
uv run yx-pretrain train \
  --data huggingface \
  --set data.dataset_name=allenai/c4 \
  --set data.tokenizer=google/gemma-2-2b \
  --set experiment.benchmark=false \
  --set experiment.checkpoint.enabled=true \
  --set experiment.checkpoint.destination=gs://EXISTING_BUCKET/yxtpu \
  --set experiment.checkpoint.save_interval=100
```

The destination must already be authorized. The launcher creates neither a
bucket nor a TPU. Checkpoints contain NNX model and optimizer state, step,
supported iterator state, the resolved configuration, repository and MaxText
commits, tokenizer identity, and the KDA precision policy. Resuming restores
the next data batch and optimizer step exactly; this is covered by a local
interrupted-versus-uninterrupted regression test.

## Existing TPU launchers

`scripts/launch_existing_tpu.sh` requires `GCP_PROJECT`, `TPU_NAME`, and
`TPU_ZONE` and only opens SSH to that existing VM. It has no provisioning
commands. `scripts/doctor_profiles.sh` checks an exact execution shape, and
`scripts/smoke_optimizers.sh` runs the three 30-step finite-loss gates.

Only `v6e-8` is performance-certified. The `v6e-64`, `v5e-16`, `v5e-64`, and
`v4-32` profiles validate device counts and carry generation-appropriate
compiler defaults, but remain marked unverified until their own approved run.

Benchmark profiles disable checkpointing. A real-training experiment must name
an explicit local directory or `gs://` destination; the package never creates
storage or TPU resources implicitly. `../AGENTS.md` is authoritative for
project, zone, quota, Spot/on-demand, and approval policy.
