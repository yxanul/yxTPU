set -uo pipefail
cd ~/yxTPU/pretraining
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_DEVICES=0,1,2,3

PY=$HOME/yxTPU/pretraining/.venv/bin/python
CFG=$HOME/yxTPU/maxtext/src/maxtext/configs/base.yml
OUT=/tmp/mt_moe_bench2
rm -rf "$OUT"; mkdir -p "$OUT"

COMMON="base_output_directory=$OUT enable_checkpointing=false dataset_type=synthetic \
 reuse_example_batch=1 enable_dropout=false steps=22 attention=flash \
 max_target_length=2048 base_emb_dim=2048 base_num_query_heads=16 base_num_kv_heads=4 \
 head_dim=128 base_num_decoder_layers=24 ici_fsdp_parallelism=4 \
 remat_policy=save_dot_except_mlpwi"
MOE="decoder_block=mixtral num_experts=8 num_experts_per_tok=2 base_mlp_dim=1408 base_moe_mlp_dim=1408"

run () {
  name="$1"; shift
  echo "############ BEGIN $name ############"
  rm -f /tmp/libtpu_lockfile
  $PY -m maxtext.trainers.pre_train.train "$CFG" run_name="$name" $COMMON "$@" 2>&1 \
    | grep -E "completed step: 21|number parameters|RESOURCE_EXHAUSTED|Out of memory|NotImplementedError|Error:" \
    | tail -6
  echo "############ END $name ############"
}

# best-case dense reference at the same remat policy
run d_pdb8  base_mlp_dim=2816 num_experts=1 per_device_batch_size=8
run d_pdb16 base_mlp_dim=2816 num_experts=1 per_device_batch_size=16

# batch scaling: bigger expert groups -> better grouped-matmul efficiency
run m_pdb8  $MOE sparse_matmul=true megablox=true per_device_batch_size=8
run m_pdb16 $MOE sparse_matmul=true megablox=true per_device_batch_size=16

# Pallas ragged-sort in the permute path
run m_rsort $MOE sparse_matmul=true megablox=true per_device_batch_size=8 use_ragged_sort=true

# jax.lax.ragged_dot instead of the Pallas megablox kernel
run m_jaxrd $MOE sparse_matmul=true megablox=false per_device_batch_size=8

# fine-grained routing at the SAME active FFN FLOPs (64 experts, top-8, 352)
run m_fine64 decoder_block=mixtral num_experts=64 num_experts_per_tok=8 \
    base_mlp_dim=352 base_moe_mlp_dim=352 sparse_matmul=true megablox=true per_device_batch_size=8

echo "ALL-DONE-2"
