set -uo pipefail
cd ~/yxTPU/pretraining
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_DEVICES=0,1,2,3
PY=$HOME/yxTPU/pretraining/.venv/bin/python
CFG=$HOME/yxTPU/maxtext/src/maxtext/configs/base.yml
OUT=/tmp/mt_moe_bench3; rm -rf "$OUT"; mkdir -p "$OUT"
COMMON="base_output_directory=$OUT enable_checkpointing=false dataset_type=synthetic \
 reuse_example_batch=1 enable_dropout=false steps=22 attention=flash max_target_length=2048 \
 base_emb_dim=2048 base_num_query_heads=16 base_num_kv_heads=4 head_dim=128 \
 base_num_decoder_layers=24 ici_fsdp_parallelism=4 remat_policy=save_dot_except_mlpwi \
 per_device_batch_size=4 sparse_matmul=true megablox=true decoder_block=mixtral"
run () { name="$1"; shift; echo "############ BEGIN $name ############"; rm -f /tmp/libtpu_lockfile
  $PY -m maxtext.trainers.pre_train.train "$CFG" run_name="$name" $COMMON "$@" 2>&1 \
    | grep -E "completed step: 21|number parameters|RESOURCE_EXHAUSTED|Out of memory" | tail -4
  echo "############ END $name ############"; }
# same active FFN FLOPs (2816) at PDB 4: coarse top-2 vs fine-grained top-8
run f_coarse8 num_experts=8  num_experts_per_tok=2 base_mlp_dim=1408 base_moe_mlp_dim=1408
run f_fine64  num_experts=64 num_experts_per_tok=8 base_mlp_dim=352  base_moe_mlp_dim=352
echo "ALL-DONE-3"
