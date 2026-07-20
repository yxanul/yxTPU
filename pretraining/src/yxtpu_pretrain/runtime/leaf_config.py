"""Explicit compatibility object for the MaxText leaf components we import."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
from maxtext.common.common_types import DecoderBlockType, ShardMode

from yxtpu_pretrain.config import ResolvedConfig


def _logical_axis_rules() -> tuple[tuple[str, str | tuple[str, ...] | None], ...]:
    # Only rules consumed by the imported leaf components are declared here.
    # Auto sharding may choose a different physical layout; these constraints
    # preserve the certified data-parallel model without inheriting base.yml.
    return (
        ("activation_embed_and_logits_batch", ("data", "fsdp")),
        ("activation_batch", ("data", "fsdp")),
        ("activation_batch_attn", ("data", "fsdp")),
        ("activation_kv_batch", ("data", "fsdp")),
        ("activation_length", "context"),
        ("activation_length_attn", "context"),
        ("activation_q_length", "context"),
        ("activation_kv_length", None),
        ("activation_heads", "tensor"),
        ("activation_kv_heads", "tensor"),
        ("activation_embed", "tensor"),
        ("activation_embed_attn", "tensor"),
        ("activation_kv", "tensor"),
        ("activation_kv_head_dim", None),
        ("vocab", "tensor"),
        ("embed_vocab", ("fsdp", "context")),
        ("heads", "tensor"),
        ("q_heads", "tensor"),
        ("kv_heads", "tensor"),
        ("qkv", None),
        ("gdn_head", "tensor"),
        ("mlp", "tensor"),
        ("num_activations", None),
        ("norm", None),
        ("layers", None),
    )


def make_leaf_config(config: ResolvedConfig) -> SimpleNamespace:
    """Builds a closed, documented adapter instead of loading MaxText base.yml."""
    model = config.model
    attention = model.attention
    kda = model.kda
    dtype = jnp.bfloat16 if model.dtype == "bfloat16" else jnp.float32

    return SimpleNamespace(
        # Shared DenseGeneral, RMSNorm, MlpBlock, and Embed fields.
        emb_dim=model.emb_dim,
        vocab_size=model.vocab_size,
        dtype=dtype,
        weight_dtype=jnp.float32,
        fused_mlp=model.fused_mlp,
        activations_in_float32=False,
        normalization_layer_epsilon=model.rms_norm_epsilon,
        matmul_precision="default",
        shard_mode=ShardMode.AUTO,
        debug_sharding=False,
        parameter_memory_host_offload=False,
        use_iota_embed=False,
        decoder_block=DecoderBlockType.QWEN3,
        logical_axis_rules=_logical_axis_rules(),
        # KDA leaf fields.
        gdn_num_key_heads=kda.num_heads,
        gdn_num_value_heads=kda.num_heads,
        gdn_key_head_dim=kda.key_head_dim,
        gdn_value_head_dim=kda.value_head_dim,
        gdn_conv_kernel_dim=kda.conv_kernel_size,
        gdn_chunk_size=kda.chunk_size,
        use_qk_norm_in_gdn=kda.qk_norm,
        kda_gate_rank=kda.gate_rank,
        kda_safe_gate=kda.safe_gate,
        kda_gate_lower_bound=kda.gate_lower_bound,
        kda_use_fused_pallas_kernel=kda.precision == "guarded_fp32",
        kda_use_pallas_blocked_solve=False,
        kda_use_analytical_custom_vjp=False,
        # AttentionOp fields. Tokamax Splash is the only TPU production path.
        head_dim=attention.head_dim,
        attention_kernel="flash",
        use_tokamax_splash=True,
        use_jax_splash=False,
        use_splash_scheduler=False,
        sa_block_q=attention.block_q,
        sa_block_kv=512,
        sa_block_kv_compute=512,
        sa_block_q_dkv=attention.block_q_dkv,
        sa_block_kv_dkv=512,
        sa_block_kv_dkv_compute=512,
        sa_block_q_dq=512,
        sa_block_kv_dq=512,
        sa_use_fused_bwd_kernel=True,
        sa_q_layout="HEAD_DIM_MINOR",
        sa_k_layout="HEAD_DIM_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
        sa_fuse_reciprocal=True,
        sa_use_base2_exp=True,
        local_sa_block_q=512,
        local_sa_block_kv=512,
        local_sa_block_kv_compute=512,
        local_sa_block_q_dkv=512,
        local_sa_block_kv_dkv=512,
        local_sa_block_kv_dkv_compute=512,
        local_sa_block_q_dq=512,
        local_sa_block_kv_dq=512,
        local_sa_use_fused_bwd_kernel=False,
        local_sa_q_layout="HEAD_DIM_MINOR",
        local_sa_k_layout="HEAD_DIM_MINOR",
        local_sa_v_layout="HEAD_DIM_MINOR",
        local_sa_fuse_reciprocal=True,
        local_sa_use_base2_exp=True,
        local_use_splash_scheduler=False,
        context_sharding="context",
        context_parallel_load_balance=True,
        context_parallel_strategy="all_gather",
        ici_context_parallelism=config.hardware.mesh.sequence,
        ici_context_autoregressive_parallelism=1,
        using_pipeline_parallelism=False,
        dataset_type=config.data.type,
        packing=True,
        max_segments_per_seq=-1,
        moba=False,
        moba_chunk_size=1024,
        moba_topk=8,
        use_indexer=False,
        use_max_logit_estimate=-1,
        cost_estimate_flops_fwd=-1,
        cost_estimate_flops_bwd=-1,
        dq_reduction_steps=0,
    )

