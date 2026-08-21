set -uo pipefail
cd ~/yxTPU/pretraining
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_DEVICES=0,1,2,3
PY=$HOME/yxTPU/pretraining/.venv/bin/python
CFG=$HOME/yxTPU/maxtext/src/maxtext/configs/base.yml
OUT=/tmp/mt_plateau; rm -rf "$OUT"; mkdir -p "$OUT"
XLAF="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
COMMON="base_output_directory=$OUT enable_checkpointing=false dataset_type=synthetic \
 reuse_example_batch=1 enable_dropout=false steps=22 attention=flash head_dim=128 \
 max_target_length=2048 ici_fsdp_parallelism=4 remat_policy=minimal num_experts=1"
r () { name="$1"; shift; echo "############ BEGIN $name ############"; rm -f /tmp/libtpu_lockfile
  LIBTPU_INIT_ARGS="$XLAF" $PY -m maxtext.trainers.pre_train.train "$CFG" run_name="$name" $COMMON "$@" 2>&1 \
    | grep -E "completed step: 21|number parameters|RESOURCE_EXHAUSTED|Out of memory|Error:" | tail -4
  echo "############ END $name ############"; }
# emb 2560 (20x128), mlp 6912 (54x128), 20 q heads -> 2560, 4 kv -> 512. all MXU-aligned.
S2560="base_emb_dim=2560 base_num_query_heads=20 base_num_kv_heads=4 base_mlp_dim=6912 base_num_decoder_layers=12"
# emb 3072 (24x128), mlp 8192 (64x128), 24 q heads -> 3072, 4 kv -> 512.
S3072="base_emb_dim=3072 base_num_query_heads=24 base_num_kv_heads=4 base_mlp_dim=8192 base_num_decoder_layers=9"
r c_2560_pdb16      $S2560 per_device_batch_size=16
r c_2560_pdb16_bf16 $S2560 per_device_batch_size=16 weight_dtype=bfloat16
r c_3072_pdb8       $S3072 per_device_batch_size=8
r c_3072_pdb16      $S3072 per_device_batch_size=16
echo "ALL-DONE-P"
