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
- Last verified: 2026-07-22 — `guaranteed: {}` (the on-demand tier) confirmed
  on the resource; queued resource `PROVISIONING`, node `CREATING`. The VM
  starts empty: run `pretraining` setup (repo clone, venv, HF and W&B
  credentials) on all 8 workers once it is ACTIVE.
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
