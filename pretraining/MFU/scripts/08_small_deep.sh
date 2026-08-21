set -uo pipefail
# ~500M params or less, 18-24 layers: how much MFU does the depth constraint cost,
# and what does it do to tokens/s? All dims are multiples of 128 (v4 MXU 128x128).
cd ~/yxTPU/pretraining
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_DEVICES=0,1,2,3
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
PY=$HOME/yxTPU/pretraining/.venv/bin/python
CFG=$HOME/yxTPU/maxtext/src/maxtext/configs/base.yml
OUT=/tmp/mt_small; rm -rf "$OUT"; mkdir -p "$OUT"
COMMON="base_output_directory=$OUT enable_checkpointing=false dataset_type=synthetic \
 reuse_example_batch=1 enable_dropout=false steps=22 attention=flash head_dim=128 \
 max_target_length=2048 ici_fsdp_parallelism=4 remat_policy=minimal num_experts=1 \
 weight_dtype=bfloat16 logits_via_embedding=true"
r () { name="$1"; shift; echo "############ BEGIN $name ############"; rm -f /tmp/libtpu_lockfile
  $PY -m maxtext.trainers.pre_train.train "$CFG" run_name="$name" $COMMON "$@" 2>&1 \
    | grep -E "completed step: 21|number parameters|RESOURCE_EXHAUSTED|Out of memory|Error:" | tail -4
  echo "############ END $name ############"; }
S () { echo "base_emb_dim=$1 base_num_query_heads=$2 base_num_kv_heads=$3 base_mlp_dim=$4 base_num_decoder_layers=$5"; }

# --- anchor: the 0.976B best-MFU point, same session, for relative tok/s ---
r ref_976M $(S 3072 24 4 8192 9) per_device_batch_size=16

# --- round 1: <=500M at 18-24 layers, vocab 32k, PDB 16 ---
r s_e1024_L24 $(S 1024  8 2 4096 24) per_device_batch_size=16
r s_e1280_L24 $(S 1280 10 2 3456 24) per_device_batch_size=16
r s_e1280_L18 $(S 1280 10 2 4608 18) per_device_batch_size=16
r s_e1536_L18 $(S 1536 12 2 4096 18) per_device_batch_size=16
r s_e1536_L20 $(S 1536 12 2 3584 20) per_device_batch_size=16
r s_e2048_L18 $(S 2048 16 4 2048 18) per_device_batch_size=16

# --- round 2: batch scaling on the two most promising ---
r s_e1536_L18_pdb32 $(S 1536 12 2 4096 18) per_device_batch_size=32
r s_e1280_L24_pdb32 $(S 1280 10 2 3456 24) per_device_batch_size=32

# --- round 3: our REAL 128,256 SuperBPE vocab (embedding is 26% of a 500M budget) ---
r v128k_e1024_L24 $(S 1024  8 2 4096 24) per_device_batch_size=16 vocab_size=128256
r v128k_e1280_L18 $(S 1280 10 2 3840 18) per_device_batch_size=16 vocab_size=128256
r v128k_e1536_L18 $(S 1536 12 2 2432 18) per_device_batch_size=16 vocab_size=128256
echo "ALL-DONE-S"
