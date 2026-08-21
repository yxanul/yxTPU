set -uo pipefail
cd ~/yxTPU/pretraining
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# one worker, its own 4 chips (2x2x1) - the pod-slice collective init is bypassed
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_DEVICES=0,1,2,3

PY=$HOME/yxTPU/pretraining/.venv/bin/python
CFG=$HOME/yxTPU/maxtext/src/maxtext/configs/base.yml
OUT=/tmp/mt_moe_bench
rm -rf "$OUT"; mkdir -p "$OUT"

# Shared geometry: emb 2048, 24 layers, GQA 16:4, head_dim 128, vocab 32k, seq 2048.
COMMON="base_output_directory=$OUT enable_checkpointing=false dataset_type=synthetic \
 reuse_example_batch=1 enable_dropout=false steps=25 attention=flash \
 max_target_length=2048 per_device_batch_size=8 \
 base_emb_dim=2048 base_num_query_heads=16 base_num_kv_heads=4 head_dim=128 \
 base_num_decoder_layers=24 ici_fsdp_parallelism=4"

run () {
  name="$1"; shift
  echo "############ BEGIN $name ############"
  rm -f /tmp/libtpu_lockfile
  $PY -m maxtext.trainers.pre_train.train "$CFG" run_name="$name" $COMMON "$@" 2>&1 \
    | grep -E "TFLOP|Per train step|number parameters|completed step|Memory stats|OOM|RESOURCE_EXHAUSTED|Error|Traceback|NotImplementedError|bytes" \
    | tail -45
  echo "############ END $name rc=${PIPESTATUS[0]} ############"
}

# 1) dense reference at the SAME active FFN FLOPs as top-2 of 1408 (= 2816)
run dense_ref base_mlp_dim=2816 num_experts=1

# 2) MoE, megablox grouped-matmul path, dropless (the production MoE path)
run moe_gmm decoder_block=mixtral num_experts=8 num_experts_per_tok=2 \
    base_mlp_dim=1408 base_moe_mlp_dim=1408 sparse_matmul=true megablox=true capacity_factor=-1.0

# 3) MoE, dense_matmul path with token dropping (the fallback implementation)
run moe_dense_mm decoder_block=mixtral num_experts=8 num_experts_per_tok=2 \
    base_mlp_dim=1408 base_moe_mlp_dim=1408 sparse_matmul=false capacity_factor=1.25

# 4) MoE gmm with a lighter remat - the achievable ceiling if memory allows
run moe_gmm_minremat decoder_block=mixtral num_experts=8 num_experts_per_tok=2 \
    base_mlp_dim=1408 base_moe_mlp_dim=1408 sparse_matmul=true megablox=true capacity_factor=-1.0 \
    remat_policy=save_dot_except_mlpwi

echo "ALL-DONE"
