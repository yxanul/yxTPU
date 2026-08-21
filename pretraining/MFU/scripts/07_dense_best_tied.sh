set -uo pipefail
cd ~/yxTPU/pretraining
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_DEVICES=0,1,2,3
PY=$HOME/yxTPU/pretraining/.venv/bin/python
CFG=$HOME/yxTPU/maxtext/src/maxtext/configs/base.yml
OUT=/tmp/mt_tied; rm -rf "$OUT"; mkdir -p "$OUT"
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
rm -f /tmp/libtpu_lockfile
echo "############ BEGIN c_3072_tied ############"
$PY -m maxtext.trainers.pre_train.train "$CFG" run_name=c_3072_tied \
  base_output_directory=$OUT enable_checkpointing=false dataset_type=synthetic reuse_example_batch=1 \
  enable_dropout=false steps=22 attention=flash head_dim=128 max_target_length=2048 \
  ici_fsdp_parallelism=4 remat_policy=minimal num_experts=1 weight_dtype=bfloat16 \
  base_emb_dim=3072 base_num_query_heads=24 base_num_kv_heads=4 base_mlp_dim=8192 \
  base_num_decoder_layers=9 per_device_batch_size=16 logits_via_embedding=true 2>&1 \
  | grep -E "completed step: 21|number parameters|RESOURCE_EXHAUSTED|Out of memory|Error:" | tail -4
echo "ALL-DONE-T"
