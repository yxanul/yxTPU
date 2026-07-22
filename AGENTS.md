# TPU Research Cloud Workspace

This workspace is for JAX, Flax NNX, Pallas, XLA, and MaxText experiments on
the user's TPU Research Cloud allocation.

## TPU allocation

Source: the TPU Research Cloud approval email. The offer is free for 30 days
and applies only to newly created Cloud TPUs in the exact zones below.

| TPU | Provisioning | Chip quota | Zone |
| --- | --- | ---: | --- |
| v5e | Spot | 64 | `europe-west4-b` |
| v5e | Spot | 64 | `us-central1-a` |
| v6e | Spot | 64 | `us-east1-d` |
| v4 | Spot | 32 | `us-central2-b` |
| v4 | On-demand | 32 | `us-central2-b` |
| v6e | Spot | 64 | `europe-west4-a` |

The approval email did not provide an exact expiry timestamp. The 30-day
window is counted from the confirmed start date below:

- TRC start: 2026-07-20
- TRC expiry: 2026-08-19 (30 days from start; exact timestamp unconfirmed)

### v5e quota distinction

- The 64-chip v5e Spot grants map to the Cloud TPU API's **training** quota.
- Cloud TPU classifies `v5litepod-1`, `v5litepod-4`, and `v5litepod-8` as
  serving slices. They consume a separate serving quota that the TRC grant did
  not increase; this project currently has only 4 Spot serving chips per zone.
- The smallest v5e slice that consumes the granted training quota is
  `v5litepod-16`. It is a multi-host slice.
- A `v5litepod-8` Spot request in `europe-west4-b` was rejected on 2026-07-20
  with `RESOURCE_EXHAUSTED` against the 4-chip serving quota. No resource was
  created.

## Current training TPU

State is dynamic; verify it before relying on this section.

- TPU node: `yxtpu-v4-64-train`
- Queued resource: `yxtpu-v4-64-train-qr`
- Zone: `us-central2-b`
- Accelerator: `v4-64` = 32 chips (v4 slice names count TensorCores, 2 per
  chip, and megacore presents 1 JAX device per chip). Multi-host: 8 workers,
  4 chips each, topology 2x4x4.
- Provisioning: On-demand — explicitly requested by the user on 2026-07-22;
  this is the grant's only on-demand row.
- Runtime: `tpu-ubuntu2204-base`
- Created: 2026-07-22
- Last verified: 2026-07-22 — queued resource `ACTIVE`; node `READY` and
  `HEALTHY`. `guaranteed: {}` (the on-demand tier) confirmed on the resource.
  On-demand provisioned on the first attempt, unlike the four reclaimed v6e
  Spot attempts the same day.
- Benchmarks 2026-07-22 (synthetic, selected profile, pure data parallel
  over 32 chips): MaxText 247M GQA baseline ~2.26M tokens/s global
  (~105 TFLOP/s/device, ~38% MFU). KDA 273M with the fully fused v4 kernel
  (fused forward + split fused backward, commit 3cfc1a9): ~1.54M tokens/s
  global at 8/device x 2048 - about 32% below the GQA baseline, matching
  the 20-40% gap measured on v6e. 16/device x 2048 (~1.53M) and
  8/device x 4096 (~1.55M) also fit; 16 x 4096 still OOMs by ~0.7 GiB.
  The earlier forward-only hybrid measured ~630k tokens/s and OOMed at
  batch 16 from its XLA-tape backward; it remains only as a fallback.
- BlockAttnRes A/B 2026-07-22 (arXiv:2603.15031, commit 7028526; same
  kda_hybrid_128k + muonclip protocol as run 3): PASSES both gates - final
  loss 3.796 vs 3.872, holdout 3.850 vs 3.882. lambada, the hybrid's one
  campaign loss, jumped 0.112 -> 0.164 acc (ppl 7,161 -> 2,247), beating
  even the pure-GQA transformer's 0.127. sciq 0.58, hellaswag and
  arc_challenge also up; piqa/boolq slightly down. Cost: ~17% throughput
  (0.94M vs 1.13M tokens/s) - the depth reads are bandwidth-bound, and a
  mixer_only site ablation is the obvious next lever. Grad norms slightly
  busier (mean 0.51 vs 0.39, max 5.67) but bounded and smooth.
- SuperBPE-1B campaign 2026-07-22 (W&B group superbpe-1b; ClimbMix streamed
  through alisawuffles/superbpe-tokenizer-128k, vocab padded 128256, tied
  embeddings, 1B tokens, constant LR 3e-4 after 40-step warmup, full lm-eval
  every 477 steps): gqa_rope_128k+adamw final loss 4.011 / holdout 4.07;
  gqa_rope_128k+muonclip 3.872 / 3.929; kda_hybrid_128k+muonclip 3.824 /
  3.882 (best, and the calmest gradients: mean 0.39, max 2.03). Two bugs
  were found and fixed en route: benchmark-sized LR schedules decaying to
  the floor by step 30, and optax Muon defaulting to width-transfer scaling
  (fixed with consistent_rms=0.2, commit 5f134c7) - the un-fixed muonclip
  run lost a full nat of loss and its grad norms rose all run. Multi-host
  lm-eval works as of ecbc7b8; each full ten-task round costs ~2 minutes.
- Tied embeddings (model.logits_via_embedding, commit 6472af7): now actually
  implemented (the flag was schema-only before). On v4-64 at 8x4096 the tied
  273m model runs ~1.554M tokens/s (unchanged, as expected) with
  parameter_count 239,381,088 - exactly vocab x emb fewer. For the gpt2
  model the same flag saves 51.6M parameters (~17%).
- Input-projection fusion (kda.fused_in_proj, commit c841282): merges the
  four input-side KDA projections into one [embed, 3336] GEMM. Proven exactly
  equivalent (transplant test, 2e-5) and guarded against muon-family
  optimizers pending blocked Newton-Schulz routing. A/B on v4-64 measured it
  performance-NEUTRAL (-0.8% at 8x2048, +0.7% at 8x4096): XLA already
  overlaps the sliver GEMMs behind the qkv GEMM, and the fused path pays a
  materialize-and-slice cost instead. Keep it off by default; re-evaluate on
  v6e where the wider MXU makes slivers relatively costlier.
- KDA on v4 runs through `kda_v4_hybrid`: the pre-fold fused Pallas forward
  (`kernels/kda_fused_pallas_v4.py`) plus a chunkwise XLA backward. The fused
  backward cannot compile on v4 - Mosaic's layout assignment needs a
  sublane-gather relayout the v4 ISA lacks (every construct compiles in
  isolation; only the integrated backward fails). The folded kernel remains
  the v5+/v6 path; dispatch is automatic by device generation.
- v4 pod-slice operations, learned the hard way:
  - libtpu initialization on a pod slice is collective: a JAX process on one
    worker blocks in `make_tpu_client` forever unless all workers launch
    together. Single-worker debugging needs `TPU_PROCESS_BOUNDS`-style env
    restrictions; otherwise always use `--worker=all`.
  - A crashed run leaves `/tmp/libtpu_lockfile` behind; the next client
    blocks or fails until it is removed on every worker.
  - The primary logging process (jax process_index 0) is NOT necessarily
    worker 0 - it landed on worker 3 here. Filter run logs by content, not
    hostname.
  - Never `pkill -f <pattern>` over `gcloud ssh --command` when the pattern
    also appears in the command line itself - the shell kills its own
    session and gcloud retries in a loop.
- Setup completed 2026-07-22 on all 8 workers: repo at `main` (`6aeaf4a`,
  full clone — doctor's pin check needs git history, so never clone with
  `--depth 1`), `uv sync --locked --extra dev`, HF token in
  `~/.cache/huggingface/token` and W&B key in `~/.netrc` (both verified by
  live authentication; non-interactive SSH needs `~/.local/bin` added to
  PATH for `uv`). `doctor --hardware v4-64`: all 8 workers report 32 TPU
  devices and matching device count. The `maxtext_pin` check reports
  `clean=False` on workers and locally alike — the vendored `maxtext/` tree
  was intentionally patched after the import commit by the KDA training-path
  commits — so treat that specific FAIL as the known baseline, not a setup
  regression.
- Quota verified 2026-07-22 via the Service Usage API: "TPU-V4 pod cores in
  use" has no default quota anywhere and exactly one zone override,
  `us-central2-b` = 64 cores = 32 chips — the TRC grant fingerprint. The
  request consumes the full grant.
- v6e history: on 2026-07-22 four v6e-32 Spot attempts (three in
  `europe-west4-a`, one in `us-east1-d`) were all reclaimed mid-provisioning
  (`PROVISIONING` -> `SUSPENDING` -> `FAILED`, internal error code 13;
  `reset` only works from `ACTIVE`, so each retry required delete +
  recreate). The user then chose the on-demand v4 grant instead. Both v6e
  zones were verified clean of nodes and queued resources. The v6e Spot
  quota (64 chips per zone in `europe-west4-a` and `us-east1-d`) remains
  granted and unused; on-demand v6e is not part of the grant and must never
  be provisioned.
- The earlier `yxtpu-v6e8-dev` v6e-8 slice was Spot-preempted on 2026-07-21;
  its queued resource was deleted on 2026-07-22.

Check status:

```bash
gcloud compute tpus queued-resources describe yxtpu-v4-64-train-qr \
  --zone=us-central2-b

gcloud compute tpus tpu-vm describe yxtpu-v4-64-train \
  --zone=us-central2-b
```

Connect (multi-host slice: plain ssh lands on worker 0; use `--worker=all
--command="..."` to run on every host):

```bash
gcloud compute tpus tpu-vm ssh yxtpu-v4-64-train \
  --zone=us-central2-b
```

Recreate (on-demand v4 requires the user's explicit request in the current
conversation — never recreate this resource unprompted; for Spot rows,
`--spot` is mandatory because the command defaults to on-demand):

```bash
gcloud compute tpus queued-resources create yxtpu-v4-64-train-qr \
  --node-id=yxtpu-v4-64-train \
  --zone=us-central2-b \
  --accelerator-type=v4-64 \
  --runtime-version=tpu-ubuntu2204-base
```

Delete both resources when finished:

```bash
gcloud compute tpus tpu-vm delete yxtpu-v4-64-train \
  --zone=us-central2-b

gcloud compute tpus queued-resources delete yxtpu-v4-64-train-qr \
  --zone=us-central2-b
```

## Provisioning rules

- Treat the locations above as zones, not broad regions.
- Always pass the project and exact zone explicitly in provisioning commands.
- Never provision outside the table or use a different TPU generation without
  explicit user approval.
- Spot rows must be provisioned as Spot. Do not silently fall back to
  on-demand capacity.
- `gcloud compute tpus queued-resources create` defaults to on-demand when no
  tier flag is given: every v5e/v6e create command MUST pass `--spot`
  explicitly (verified against gcloud 576.0.0).
- On-demand is never allowed for v5e or v6e. The only on-demand quota in the
  grant is v4 (32 chips, `us-central2-b`), and it may be used solely when the
  user explicitly requests it in the current conversation.
- Default to an at-most-8-chip Spot slice for smoke tests when the chosen TPU
  generation's granted quota supports it. For v5e training, the minimum is 16.
- Get explicit user approval before requesting more than 8 chips or using the
  on-demand v4 quota.
- TPU Spot VMs can be preempted at any time. Before scaling, prove that the
  workload checkpoints to durable storage and restores correctly.
- Delete TPU VMs and queued-resource requests as soon as an experiment ends.
- The free offer covers the listed TPU quota only. Storage, networking, data
  egress, logging, and other Google Cloud services can still incur charges.
- The user's extra cloud credit is not standing authorization to create paid
  resources.

## Credentials and repository safety

- Do not commit Google credentials, access tokens, service-account keys, SSH
  private keys, Hugging Face tokens, or signed URLs.
- Prefer `gcloud auth login` for local CLI access. Use Application Default
  Credentials only when local programs need Google Cloud APIs.
- Let `gcloud compute tpus tpu-vm ssh` manage its dedicated Compute Engine SSH
  key. Do not copy a personal private key onto TPU VMs.
- Keep the Google Cloud project selection in local `gcloud` configuration or
  an ignored environment file instead of hard-coding it in source files.
