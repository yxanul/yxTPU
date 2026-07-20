"""Owned fused-projection NoPE grouped-query attention layer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers.attention_op import AttentionOp
from maxtext.layers.linears import DenseGeneral

from yxtpu_pretrain.config import AttentionConfig
from yxtpu_pretrain.layers.roles import ParamRole, declare_dense_kernel


class NoPEGQA(nnx.Module):
    """NoPE GQA with one fused QKV projection and Tokamax Splash on TPU."""

    def __init__(
        self,
        config: AttentionConfig,
        *,
        emb_dim: int,
        max_target_length: int,
        dtype,
        weight_dtype,
        leaf_config,
        mesh,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.emb_dim = emb_dim
        self.dtype = dtype
        self.mesh = mesh
        self.num_query_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_heads_per_kv = self.num_query_heads // self.num_kv_heads

        total_heads = self.num_query_heads + 2 * self.num_kv_heads
        self.qkv_proj = DenseGeneral(
            in_features_shape=emb_dim,
            out_features_shape=(total_heads, self.head_dim),
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_axes=("embed", "qkv", "kv_head_dim"),
            matmul_precision="default",
            rngs=rngs,
        )
        declare_dense_kernel(self.qkv_proj, ParamRole.GQA_QKV)
        self.out_proj = DenseGeneral(
            in_features_shape=(self.num_query_heads, self.head_dim),
            out_features_shape=emb_dim,
            axis=(-2, -1),
            dtype=dtype,
            weight_dtype=weight_dtype,
            kernel_axes=("q_heads", "kv_head_dim", "embed"),
            matmul_precision="default",
            rngs=rngs,
        )
        declare_dense_kernel(
            self.out_proj,
            ParamRole.GQA_OUTPUT,
            in_axes=(0, 1),
            out_axes=(2,),
        )

        self.use_tokamax = mesh.devices[(0,) * mesh.devices.ndim].platform == "tpu"
        if self.use_tokamax:
            self.attention_op = AttentionOp(
                config=leaf_config,
                mesh=mesh,
                attention_kernel="flash",
                max_target_length=max_target_length,
                max_prefill_predict_length=max_target_length,
                num_query_heads=self.num_query_heads,
                num_kv_heads=self.num_kv_heads,
                dtype=dtype,
                dropout_rate=0.0,
                rngs=rngs,
            )
            # Stabilize the NNX graph before scan. AttentionOp updates this value
            # when record_max_logits=True rather than creating a new path.
            self.attention_op.max_logits = nnx.Intermediate(
                jnp.zeros((1, self.num_query_heads), dtype=jnp.float32)
            )
        else:
            self.attention_op = None
        self.max_logits = nnx.Intermediate(
            jnp.zeros((1, self.num_query_heads), dtype=jnp.float32)
        )

    def _project(self, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        qkv = jax.ad_checkpoint.checkpoint_name(qkv, "qkv_proj")
        q_end = self.num_query_heads
        k_end = q_end + self.num_kv_heads
        query = qkv[..., :q_end, :]
        key = qkv[..., q_end:k_end, :]
        value = qkv[..., k_end:, :]
        query = query * jnp.asarray(self.head_dim**-0.5, dtype=query.dtype)
        return query, key, value

    def _dot_attention(self, query, key, value, segment_ids, *, record_max_logits):
        batch, query_length, _, _ = query.shape
        key_length = key.shape[1]
        grouped_query = query.reshape(
            batch,
            query_length,
            self.num_kv_heads,
            self.q_heads_per_kv,
            self.head_dim,
        )
        logits = jnp.einsum(
            "btkhd,bskd->bkhts",
            grouped_query.astype(jnp.float32),
            key.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        causal = jnp.arange(query_length)[:, None] >= jnp.arange(key_length)[None, :]
        mask = causal[None, None, None, :, :]
        if segment_ids is not None:
            same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
            valid = (segment_ids[:, :, None] != 0) & (segment_ids[:, None, :] != 0)
            mask = mask & (same_segment & valid)[:, None, None, :, :]
        logits = jnp.where(mask, logits, jnp.asarray(-1.0e30, dtype=logits.dtype))
        if record_max_logits:
            maxima = jnp.max(logits, axis=(-2, -1)).reshape(batch, self.num_query_heads)
            self.max_logits.value = maxima
        probabilities = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        output = jnp.einsum(
            "bkhts,bskd->btkhd",
            probabilities,
            value,
            precision=jax.lax.Precision.DEFAULT,
        )
        output = output.reshape(batch, query_length, self.num_query_heads, self.head_dim)
        if segment_ids is not None:
            output = jnp.where(segment_ids[..., None, None] != 0, output, 0)
        return output

    def __call__(
        self,
        hidden_states,
        *,
        decoder_segment_ids=None,
        decoder_positions=None,
        record_max_logits: bool = False,
    ):
        query, key, value = self._project(hidden_states)
        if self.attention_op is None:
            output = self._dot_attention(
                query,
                key,
                value,
                decoder_segment_ids,
                record_max_logits=record_max_logits,
            )
        else:
            output = self.attention_op(
                query,
                key,
                value,
                decoder_segment_ids,
                decoder_positions,
                MODEL_MODE_TRAIN,
                record_max_logits=record_max_logits,
            )
            if record_max_logits:
                self.max_logits.value = self.attention_op.max_logits.value
        output = self.out_proj(output.astype(self.dtype))
        return jax.ad_checkpoint.checkpoint_name(output, "out_proj")
