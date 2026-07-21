#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

if [[ ! -x "${PRETRAINING_DIR}/.venv/bin/yx-pretrain" ]]; then
  echo "Missing pretraining/.venv. Run 'uv sync --locked' first." >&2
  exit 1
fi

# This launcher assumes the existing TPU runtime and W&B login are already
# configured. It contains no cloud provisioning, storage, or credential logic.
cd "${PRETRAINING_DIR}"
exec .venv/bin/yx-pretrain train \
  --model kda_hybrid_309m_gpt2 \
  --optimizer adamw_10b \
  --data climbmix \
  --hardware v6e-8 \
  --experiment climbmix_10b \
  "$@"
