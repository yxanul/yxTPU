set -uo pipefail
# GPT-2 vocab (50257, padded to 50304 = 393x128) at <=500M with 18-24 layers.
cd ~/yxTPU/pretraining
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_DEVICES=0,1,2,3
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
PY=$HOME/yxTPU/pretraining/.venv/bin/python
CFG=$HOME/yxTPU/maxtext/src/maxtext/configs/base.yml
OUT=/tmp/mt_gpt2v; rm -rf "$OUT"; mkdir -p "$OUT"
COMMON="base_output_directory=$OUT enable_checkpointing=false dataset_type=synthetic \
 reuse_example_batch=1 enable_dropout=false steps=22 attention=flash head_dim=128 \
 max_target_length=2048 ici_fsdp_parallelism=4 remat_policy=minimal num_experts=1 \
 weight_dtype=bfloat16 logits_via_embedding=true per_device_batch_size=16"
r () { name="$1"; shift; echo "############ BEGIN $name ############"; rm -f /tmp/libtpu_lockfile
  $PY -m maxtext.trainers.pre_train.train "$CFG" run_name="$name" $COMMON "$@" 2>&1 \
    | grep -E "completed step: 21|number parameters|RESOURCE_EXHAUSTED|Out of memory|Error:" | tail -4
  echo "############ END $name ############"; }
S () { echo "base_emb_dim=$1 base_num_query_heads=$2 base_num_kv_heads=$3 base_mlp_dim=$4 base_num_decoder_layers=$5"; }
V=vocab_size=50304

# vocab padding A/B: 50257 (392.6x128, misaligned) vs 50304 (393x128)
r g_pad50257 $(S 1280 10 2 4608 20) vocab_size=50257
r g_pad50304 $(S 1280 10 2 4608 20) vocab_size=50304

# <=500M candidates, 18-24 layers, GPT-2 vocab
r g_e1280_L18 $(S 1280 10 2 4608 18) $V
r g_e1280_L24 $(S 1280 10 2 3584 24) $V
r g_e1024_L24 $(S 1024  8 2 4608 24) $V
r g_e1536_L18 $(S 1536 12 2 3584 18) $V
echo "ALL-DONE-G"
