#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/yxTPU}"
PDB="${PDB:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONFIG="${CONFIG:-$WORKSPACE/benchmarks/maxtext_v6e_kda_hybrid_273m.yml}"
RUN_NAME="${RUN_NAME:-v6e8-kda-hybrid-273m-adamw-s${SEQ_LEN}-b${PDB}-${STAMP}}"

export WORKSPACE PDB SEQ_LEN CONFIG RUN_NAME

exec bash "$WORKSPACE/benchmarks/run_maxtext_v6e_272m.sh" "$@"
