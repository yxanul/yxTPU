#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT to the existing TPU VM project.}"
: "${TPU_NAME:?Set TPU_NAME to an existing TPU VM name.}"
: "${TPU_ZONE:?Set TPU_ZONE to its exact zone.}"

REMOTE_REPO="${REMOTE_REPO:-\$HOME/yxTPU}"
HARDWARE="${YXTPU_HARDWARE:-v6e-8}"
COMMAND="${*:-benchmark --hardware ${HARDWARE}}"

# This launcher only opens SSH on an existing TPU VM. It contains no create,
# resize, queued-resource, or delete operation.
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${GCP_PROJECT}" \
  --zone="${TPU_ZONE}" \
  --command="cd ${REMOTE_REPO}/pretraining && uv sync --locked && uv run yx-pretrain ${COMMAND}"

