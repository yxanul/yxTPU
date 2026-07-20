#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/yxTPU}"
PDB="${PDB:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
STEPS="${STEPS:-12}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
PROFILE_START_STEP="${PROFILE_START_STEP:-5}"
PROFILE_STEPS="${PROFILE_STEPS:-3}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONFIG="${CONFIG:-$WORKSPACE/benchmarks/maxtext_v6e_kda_hybrid_273m.yml}"
RUN_NAME="${RUN_NAME:-v6e8-kda-hybrid-273m-profile-s${SEQ_LEN}-b${PDB}-${STAMP}}"

export WORKSPACE PDB SEQ_LEN STEPS WARMUP_STEPS CONFIG RUN_NAME

exec bash "$WORKSPACE/benchmarks/run_kda_hybrid_adamw_v6e_273m.sh" \
  profiler=xplane \
  "skip_first_n_steps_for_profiler=$PROFILE_START_STEP" \
  "profiler_steps=$PROFILE_STEPS" \
  profile_cleanly=true \
  enable_tpu_profiling_options=true \
  tpu_num_chips_to_profile_per_task=1 \
  "$@"
