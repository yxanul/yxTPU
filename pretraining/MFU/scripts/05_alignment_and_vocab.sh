set -uo pipefail
cd ~/yxTPU/pretraining
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_DEVICES=0,1,2,3
PY=$HOME/yxTPU/pretraining/.venv/bin/python
CFG=$HOME/yxTPU/maxtext/src/maxtext/configs/base.yml
OUT=/tmp/mt_final; rm -rf "$OUT"; mkdir -p "$OUT"
XLAF="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
COMMON="base_output_directory=$OUT enable_checkpointing=false dataset_type=synthetic \
 reuse_example_batch=1 enable_dropout=false steps=22 attention=flash head_dim=128 \
 max_target_length=2048 ici_fsdp_parallelism=4 remat_policy=minimal"
r () { name="$1"; libtpu="$2"; shift 2
  echo "############ BEGIN $name ############"; rm -f /tmp/libtpu_lockfile
  LIBTPU_INIT_ARGS="$libtpu" $PY -m maxtext.trainers.pre_train.train "$CFG" run_name="$name" $COMMON "$@" 2>&1 \
    | grep -E "completed step: 21|number parameters|RESOURCE_EXHAUSTED|Out of memory|Error:" | tail -4
  echo "############ END $name ############"; }

# ---- A: MXU alignment of the MoE expert dim (352 = 2.75x128 vs 384 = 3x128) ----
MOEBASE="decoder_block=mixtral sparse_matmul=true megablox=true base_emb_dim=2048 \
 base_num_query_heads=16 base_num_kv_heads=4 base_num_decoder_layers=24 per_device_batch_size=4"
r a_fine352    "" $MOEBASE num_experts=64 num_experts_per_tok=8 base_mlp_dim=352 base_moe_mlp_dim=352
r a_fine384    "" $MOEBASE num_experts=64 num_experts_per_tok=8 base_mlp_dim=384 base_moe_mlp_dim=384
r a_fine352pad "" $MOEBASE num_experts=64 num_experts_per_tok=8 base_mlp_dim=352 base_moe_mlp_dim=352 \
                 padded_base_moe_mlp_dim=384
r a_coarse1408 "" $MOEBASE num_experts=8 num_experts_per_tok=2 base_mlp_dim=1408 base_moe_mlp_dim=1408

# ---- B: push dense harder (all dims 128-aligned) ----
WIDE="num_experts=1 base_emb_dim=2048 base_num_query_heads=16 base_num_kv_heads=4 \
 base_mlp_dim=5632 base_num_decoder_layers=16"
r b_wide_pdb16  "$XLAF" $WIDE per_device_batch_size=16
r b_wider2560   "$XLAF" num_experts=1 base_emb_dim=2560 base_num_query_heads=20 base_num_kv_heads=4 \
                        base_mlp_dim=6912 base_num_decoder_layers=12 per_device_batch_size=8
# real SuperBPE vocab (128256 = 1002x128): does the loss head change the picture?
r b_v128k       "$XLAF" $WIDE per_device_batch_size=8 vocab_size=128256
r b_v128k_tile8 "$XLAF" $WIDE per_device_batch_size=8 vocab_size=128256 num_vocab_tiling=8
echo "ALL-DONE-F"
