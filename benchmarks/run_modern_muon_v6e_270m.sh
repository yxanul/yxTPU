#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/yxTPU}"
PDB="${PDB:-16}"
SEQ_LEN="${SEQ_LEN:-2048}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONFIG="${CONFIG:-$WORKSPACE/benchmarks/maxtext_v6e_modern_270m.yml}"
RUN_NAME="${RUN_NAME:-v6e8-modern-270m-muon-s${SEQ_LEN}-b${PDB}-${STAMP}}"

export WORKSPACE PDB SEQ_LEN CONFIG RUN_NAME

exec bash "$WORKSPACE/benchmarks/run_maxtext_v6e_272m.sh" \
  opt_type=muon \
  muon_beta=0.95 \
  muon_consistent_rms=0.2 \
  muon_weight_decay=0.1 \
  "$@"
