#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/yxTPU}"
MAXTEXT_ROOT="${MAXTEXT_ROOT:-$WORKSPACE/maxtext}"
CONFIG="${CONFIG:-$WORKSPACE/benchmarks/maxtext_v6e_272m.yml}"
VENV="${VENV:-$WORKSPACE/.venv}"
DEVICES="${DEVICES:-8}"
PDB="${PDB:-16}"
SEQ_LEN="${SEQ_LEN:-2048}"
STEPS="${STEPS:-30}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-v6e8-llama-272m-s${SEQ_LEN}-b${PDB}-${STAMP}}"
RUN_DIR="${RUN_DIR:-$WORKSPACE/results/$RUN_NAME}"
METRICS_FILE="$RUN_DIR/metrics.jsonl"
SUMMARY_FILE="$RUN_DIR/summary.json"
LOG_FILE="$RUN_DIR/train.log"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Missing virtual environment at $VENV; run setup_maxtext_tpu.sh first." >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing benchmark config at $CONFIG." >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

source "$VENV/bin/activate"
export JAX_PLATFORMS=tpu
export PYTHONUNBUFFERED=1

# Current MaxText v6e recipe: larger scoped VMEM and async collective overlap.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true}"

python - "$DEVICES" <<'PY'
import sys
import jax
import jaxlib

expected = int(sys.argv[1])
devices = jax.devices()
print(f"JAX {jax.__version__}; jaxlib {jaxlib.__version__}")
print(f"Backend: {jax.default_backend()}; devices: {len(devices)}")
print(f"Device kinds: {sorted({device.device_kind for device in devices})}")
if jax.default_backend() != "tpu":
  raise SystemExit("JAX did not initialize a TPU backend")
if len(devices) != expected:
  raise SystemExit(f"expected {expected} TPU devices, found {len(devices)}")
PY

echo "Run directory: $RUN_DIR"
echo "Global tokens/step: $((DEVICES * PDB * SEQ_LEN))"
echo "The first $WARMUP_STEPS step indices will be excluded from the summary."

cd "$MAXTEXT_ROOT"
set +e
python -m maxtext.trainers.pre_train.train \
  "$CONFIG" \
  "run_name=$RUN_NAME" \
  "base_output_directory=$RUN_DIR/maxtext-output" \
  "metrics_file=$METRICS_FILE" \
  "per_device_batch_size=$PDB" \
  "max_target_length=$SEQ_LEN" \
  "steps=$STEPS" \
  "$@" \
  2>&1 | tee "$LOG_FILE"
train_status="${PIPESTATUS[0]}"
set -e

if [[ "$train_status" -ne 0 ]]; then
  echo "MaxText exited with status $train_status; inspect $LOG_FILE." >&2
  exit "$train_status"
fi

python "$WORKSPACE/benchmarks/summarize_maxtext_metrics.py" \
  "$METRICS_FILE" \
  --devices "$DEVICES" \
  --warmup-steps "$WARMUP_STEPS" \
  --json-output "$SUMMARY_FILE"
