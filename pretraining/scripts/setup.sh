#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

uv sync --locked --extra dev
uv run yx-pretrain config dump --hardware "${YXTPU_HARDWARE:-v6e-8}" >/dev/null

echo "yxtpu-pretrain is installed from the locked environment."
echo "Run: uv run yx-pretrain doctor --hardware ${YXTPU_HARDWARE:-v6e-8}"

