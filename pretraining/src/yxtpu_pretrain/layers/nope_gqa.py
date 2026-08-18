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

# Sentinel for "no query position of this modality in the batch" in the
# per-modality maxima; the host maps it to NaN before logging.
ABSENT_LOGIT = -1.0e30


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
        # Per-Head Muon alternate (K3 §2.5): one [embed, head_dim] block per
        # fused q/k/v head slot when optimizer.muon_per_head is enabled.
        declare_dense_kernel(
            self.qkv_proj,
            ParamRole.GQA_QKV,
            alt_in_axes=(0,),
            alt_out_axes=(2,),
            alt_kind="per_head",
        )
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

        if config.output_gate:
            # Head-specific sigmoid output gate (G1 of arXiv:2505.06708; the
            # design K3 §2.1.2 adopts for its global-attention layers): the
            # per-head SDPA output is gated elementwise by sigmoid(W_g x)
            # BEFORE the output projection — gating after W_o cannot break
            # the W_V*W_O low-rank bottleneck — with x the post-pre-norm
            # hidden state that also feeds QKV. Deliberately a separate GEMM:
            # fusing gate heads into qkv_proj would touch apply_gqa_muonclip's
            # certified fused-slice layout. Standard init (sigma(~0) ~ 0.5,
            # the paper's trained configuration), no bias.
            self.gate_proj = DenseGeneral(
                in_features_shape=emb_dim,
                out_features_shape=(self.num_query_heads, self.head_dim),
                dtype=dtype,
                weight_dtype=weight_dtype,
                kernel_axes=("embed", "q_heads", "kv_head_dim"),
                matmul_precision="default",
                rngs=rngs,
            )
            declare_dense_kernel(
                self.gate_proj,
                ParamRole.GQA_GATE,
                alt_in_axes=(0,),
                alt_out_axes=(2,),
                alt_kind="per_head",
            )
        else:
            self.gate_proj = None

        if config.rope:
            from maxtext.layers.embeddings import RotaryEmbedding

            # Parameter-free module; rotation commutes with the query's
            # 1/sqrt(head_dim) pre-scale applied in _project.
            self.rotary = RotaryEmbedding(
                min_timescale=1,
                max_timescale=10_000,
                mesh=mesh,
                embedding_dims=self.head_dim,
                cast_as_fprop_dtype=True,
                fprop_dtype=dtype,
                rngs=rngs,
            )
        else:
            self.rotary = None

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
            # Ask the vendored op for the kernel's per-query running maxima
            # [batch, heads, q_len] instead of the batch-reduced [batch, heads];
            # this layer reduces them jointly (QK-clip) and per modality.
            self.attention_op.keep_max_logits_query_axis = True
        else:
            self.attention_op = None
        self.max_logits = nnx.Intermediate(
            jnp.zeros((1, self.num_query_heads), dtype=jnp.float32)
        )
        # [2, heads]: max logit over VISUAL query positions, then over TEXT
        # query positions (ABSENT_LOGIT when the batch has none of a kind).
        self.max_logits_by_modality = nnx.Intermediate(
            jnp.full((2, self.num_query_heads), ABSENT_LOGIT, dtype=jnp.float32)
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

    def _record_maxima(self, per_query, modality_mask) -> None:
        """Reduces [batch, heads, q_len] maxima into the joint [1, heads]
        (QK-clip's input) and the [2, heads] modality split. The split is by
        QUERY position: "visual" is the maximum over visual queries (any
        key), "text" over text queries."""
        joint = jnp.max(per_query, axis=(0, 2)).reshape(1, self.num_query_heads)
        self.max_logits.value = joint
        if modality_mask is None:
            split = jnp.full((2, self.num_query_heads), ABSENT_LOGIT, jnp.float32)
        else:
            visual = modality_mask[:, None, :]
            absent = jnp.asarray(ABSENT_LOGIT, per_query.dtype)
            split = jnp.stack(
                [
                    jnp.max(jnp.where(visual, per_query, absent), axis=(0, 2)),
                    jnp.max(jnp.where(~visual, per_query, absent), axis=(0, 2)),
                ]
            ).astype(jnp.float32)
        self.max_logits_by_modality.value = split

    def _dot_attention(
        self, query, key, value, segment_ids, *, record_max_logits, modality_mask=None
    ):
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
            # Per-query maxima [batch, heads, q_len]; the shared reducer
            # collapses batch and sequence so the recorded intermediates keep
            # the fixed shapes they were initialized with (a batch-sized
            # shape persisting on the model is incompatible with a train step
            # compiled without recording, see __call__).
            per_query = jnp.max(logits, axis=-1).reshape(
                batch, self.num_query_heads, query_length
            )
            self._record_maxima(per_query, modality_mask)
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
        modality_mask=None,
    ):
        query, key, value = self._project(hidden_states)
        if self.rotary is not None:
            positions = decoder_positions
            if positions is None:
                positions = jnp.broadcast_to(
                    jnp.arange(hidden_states.shape[1], dtype=jnp.int32)[None, :],
                    hidden_states.shape[:2],
                )
            query = self.rotary(query, positions)
            key = self.rotary(key, positions)
        if self.attention_op is None:
            output = self._dot_attention(
                query,
                key,
                value,
                decoder_segment_ids,
                record_max_logits=record_max_logits,
                modality_mask=modality_mask,
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
                # The AttentionOp records the kernel's per-query maxima
                # [batch, heads, q_len] (keep_max_logits_query_axis). Reduce
                # them here so both this layer's intermediates and the
                # AttentionOp's own return to the fixed shapes the NNX graph
                # was initialized and compiled with. Otherwise a diagnostics
                # forward (record_max_logits=True) leaves batch-sized
                # intermediates on the model that a subsequent record-free
                # (e.g. adamw) train step cannot consume.
                per_query = self.attention_op.max_logits.value
                if per_query.ndim == 2:
                    # A vendored op without keep_max_logits_query_axis only
                    # returns [batch, heads]: the joint maximum is still
                    # exact, but there is no per-query axis to split by
                    # modality - report the split ABSENT rather than the row
                    # maximum for both modalities.
                    self._record_maxima(per_query[:, :, None], None)
                else:
                    self._record_maxima(per_query, modality_mask)
                self.attention_op.max_logits.value = self.max_logits.value
        if self.gate_proj is not None:
            # Sigmoid in fp32, consistent with the KDA output gate.
            gate = jax.nn.sigmoid(self.gate_proj(hidden_states).astype(jnp.float32))
            output = (output.astype(jnp.float32) * gate).astype(self.dtype)
        output = self.out_proj(output.astype(self.dtype))
        return jax.ad_checkpoint.checkpoint_name(output, "out_proj")
