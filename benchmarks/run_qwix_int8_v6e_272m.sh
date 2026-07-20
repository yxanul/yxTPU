#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/yxTPU}"
PDB="${PDB:-16}"
SEQ_LEN="${SEQ_LEN:-2048}"
STEPS="${STEPS:-15}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-v6e8-llama-272m-qwix-int8-s${SEQ_LEN}-b${PDB}-${STAMP}}"

export WORKSPACE PDB SEQ_LEN STEPS WARMUP_STEPS RUN_NAME

exec bash "$WORKSPACE/benchmarks/run_maxtext_v6e_272m.sh" \
  use_qwix_quantization=true \
  quantization=int8

