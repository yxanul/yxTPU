# yxTPU standalone pretraining

`yxtpu-pretrain` owns the 272.9M KDA/NoPE-GQA hybrid, its fused TPU kernel,
optimizer policy, training loop, data adapters, checkpoint metadata, profiles,
and launchers. It imports modern leaf components from the vendored MaxText pin
without importing MaxText's model registry or top-level trainer.

The validated architecture is four `[KDA, KDA, KDA, NoPE-GQA]` cycles with
RMSNorm and fused SwiGLU. KDA uses BF16 Q/K/V traffic and FP32 weights and
chunk-boundary states. The fused Pallas path uses blocked WY substitution with
full-pass FP32 inter-block coupling; recursive doubling is confined to
benchmarks after ClimbMix exposed its catastrophic cancellation (EXP-032/034).

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

## ClimbMix 10B training profile

The primary real-data profile streams
[`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
without downloading the 600 GB corpus. It tokenizes on the fly with the Rust
`GPT2TokenizerFast` backend, appends EOS between documents, densely packs
2,048-token sequences, and pads GPT-2's 50,257-token vocabulary to 50,432 so
the output dimension is a multiple of 256 on Trillium. A bounded background
thread keeps three complete update batches ready.

The source publishes only a `train` split. A stable content hash therefore
reserves 1% of documents for validation before packing; training and validation
streams are disjoint and reproducible even if upstream streaming order changes.
Every 250 optimizer steps the trainer evaluates eight held-out microbatches and
runs the separate TPU diagnostics pass. Every 1,250 steps it additionally runs
the pinned EleutherAI lm-evaluation-harness suite:

- normalized accuracy: HellaSwag, PIQA, ARC-Easy, ARC-Challenge, OpenBookQA;
- raw accuracy: SciQ, BoolQ, COPA, CommonsenseQA, LAMBADA;
- the ARC-Easy minus ARC-Challenge primary-metric gap; and
- LAMBADA perplexity as a secondary metric.

The harness scores the live NNX model on all eight devices; it does not export a
checkpoint or create a second model. Full result JSON, including harness
provenance, is written under the run directory and uploaded as a W&B artifact.
Run `wandb login` once on the TPU VM. Credentials stay in the VM's credential
store and are never passed on the command line or written to a configuration.

Launch the foreground job on an existing, already-configured TPU VM with:

```bash
cd pretraining
scripts/launch_climbmix_10b.sh
```

The equivalent public command is:

```bash
yx-pretrain train \
  --model kda_hybrid_309m_gpt2 \
  --optimizer adamw_10b \
  --data climbmix \
  --hardware v6e-8 \
  --experiment climbmix_10b
```

This profile deliberately has no checkpoints. It must be acknowledged by
`experiment.acknowledge_no_checkpoint=true`, stops after the first update that
reaches the 10B-token budget (4,769 updates; 10,001,317,888 packed tokens), and
cannot resume after Spot preemption.

For this 309.1M GPT-2-vocabulary model, the fused output loss remains selected
as a capacity requirement: standard loss at microbatch 16/GA=8 exceeds v6e HBM
during compilation. KDA uses the fused substitution VJP. It compiles with a
31,989,071,680-byte executable estimate and sustains about 472.7k tok/s,
putting the accelerator-only 10B-token time near 5.9 hours before evaluation
overhead.

The earlier recursive-doubling Pallas measurement reached 566.3k tok/s but is
rejected. With frozen weights, update 7 contained one microbatch whose gradient
norm was 3,933.7 instead of the full-FP32 reference's 2.407. Substitution with
HIGHEST coupling gives 2.406645 on that trigger and stays finite through all 15
known-trigger steps. It also completes 1,000,341,504 streamed ClimbMix tokens
with final loss 4.23895 and mean throughput 472.5k tok/s. The analytical 173.0k
path remains the debug fallback.

That stability result is not the same as analytical gradient equivalence. A
whole-gradient comparison on the exact trigger measures 1.8655% relative L2
error at cosine 0.9998266; promoting every fused matmul role still measures
1.7893%. The current profile is therefore workload stability-qualified, but it
must not be called an unconditional reference-equivalent 10B default without
an explicit BF16 tolerance decision or further reduction of that discrepancy.
See EXP-032/034/035 and
`../results/v6e8-climbmix-1b-substitution-20260721/`.

Training metrics are emitted after device synchronization, outside the timed
and compiled update. Gradient, parameter, hidden-state, sampled-logit, and
per-attention-head logit diagnostics run in their own compiled pass every 250
steps, so no host-side tree walk or W&B call enters the hot path.

## Other real data and checkpoints

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
