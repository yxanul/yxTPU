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

## Current development TPU

State is dynamic; verify it before relying on this section.

- TPU node: `yxtpu-v6e8-dev`
- Queued resource: `yxtpu-v6e8-dev-qr`
- Zone: `europe-west4-a`
- Accelerator: `v6e-8`
- Provisioning: Spot
- Runtime: `v2-alpha-tpuv6e`
- Created: 2026-07-20
- Last verified: 2026-07-21 — **Spot-preempted**; the node is deleted and the
  queued resource is `SUSPENDED`. The VM's local state (repo copy, venv, HF
  and W&B credentials) is gone; recreate the queued resource and rerun
  `pretraining` setup before the next TPU run.

Check status:

```bash
gcloud compute tpus queued-resources describe yxtpu-v6e8-dev-qr \
  --zone=europe-west4-a

gcloud compute tpus tpu-vm describe yxtpu-v6e8-dev \
  --zone=europe-west4-a
```

Connect:

```bash
gcloud compute tpus tpu-vm ssh yxtpu-v6e8-dev \
  --zone=europe-west4-a
```

Delete both resources when finished:

```bash
gcloud compute tpus tpu-vm delete yxtpu-v6e8-dev \
  --zone=europe-west4-a

gcloud compute tpus queued-resources delete yxtpu-v6e8-dev-qr \
  --zone=europe-west4-a
```

## Provisioning rules

- Treat the locations above as zones, not broad regions.
- Always pass the project and exact zone explicitly in provisioning commands.
- Never provision outside the table or use a different TPU generation without
  explicit user approval.
- Spot rows must be provisioned as Spot. Do not silently fall back to
  on-demand capacity.
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
