#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

HARDWARE="${YXTPU_HARDWARE:-v6e-8}"
for optimizer in adamw muon muonclip; do
  uv run yx-pretrain train \
    --optimizer "${optimizer}" \
    --hardware "${HARDWARE}" \
    --experiment selected \
    --set train.steps=30
done

