#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 v6e-8|v6e-64|v5e-16|v5e-64|v4-32 [...]" >&2
  exit 2
fi

for hardware in "$@"; do
  uv run yx-pretrain doctor --hardware "${hardware}"
done

