set -uo pipefail
cd ~/yxTPU/pretraining
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_DEVICES=0,1,2,3
PY=$HOME/yxTPU/pretraining/.venv/bin/python
CFG=$HOME/yxTPU/maxtext/src/maxtext/configs/base.yml
OUT=/tmp/mt_dense_push; rm -rf "$OUT"; mkdir -p "$OUT"

COMMON="base_output_directory=$OUT enable_checkpointing=false dataset_type=synthetic \
 reuse_example_batch=1 enable_dropout=false steps=22 attention=flash \
 base_emb_dim=2048 base_num_query_heads=16 base_num_kv_heads=4 head_dim=128 \
 num_experts=1 ici_fsdp_parallelism=4"
# narrow = the MoE-matched shape (mlp 1.375x emb); wide = a normal dense aspect ratio (2.75x)
NARROW="base_mlp_dim=2816 base_num_decoder_layers=24 max_target_length=2048 per_device_batch_size=8"
WIDE="base_mlp_dim=5632 base_num_decoder_layers=16 max_target_length=2048 per_device_batch_size=8"

# runner: $1=name, $2=libtpu args (may be empty), rest = config overrides
r () {
  name="$1"; libtpu="$2"; shift 2
  echo "############ BEGIN $name ############"
  rm -f /tmp/libtpu_lockfile
  LIBTPU_INIT_ARGS="$libtpu" $PY -m maxtext.trainers.pre_train.train "$CFG" \
    run_name="$name" $COMMON "$@" 2>&1 \
    | grep -E "completed step: 21|number parameters|RESOURCE_EXHAUSTED|Out of memory|NotImplementedError|Error:" | tail -4
  echo "############ END $name ############"
}

XLAF="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"

r n_base    ""      $NARROW remat_policy=save_dot_except_mlpwi
r n_minrmt  ""      $NARROW remat_policy=minimal
r n_xla     "$XLAF" $NARROW remat_policy=minimal
r n_vocab8  ""      $NARROW remat_policy=minimal num_vocab_tiling=8

r w_minrmt  ""      $WIDE remat_policy=minimal
r w_xla     "$XLAF" $WIDE remat_policy=minimal
r w_dotprod ""      $WIDE remat_policy=minimal attention=dot_product
r w_bf16w   ""      $WIDE remat_policy=minimal weight_dtype=bfloat16
r w_seq4k   ""      base_mlp_dim=5632 base_num_decoder_layers=16 max_target_length=4096 \
                    per_device_batch_size=4 remat_policy=minimal
echo "ALL-DONE-D"
