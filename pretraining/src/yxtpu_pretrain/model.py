"""Standalone scanned KDA/NoPE-GQA hybrid language model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import nnx_scan
from maxtext.layers.embeddings import Embed
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import maxtext_utils_nnx

from yxtpu_pretrain.config import ResolvedConfig
from yxtpu_pretrain.layers.kimi_delta_attention import KimiDeltaAttention
from yxtpu_pretrain.layers.nope_gqa import NoPEGQA
from yxtpu_pretrain.layers.roles import (
    ParamRole,
    declare_dense_kernel,
    declare_norm,
    declare_parameter,
)
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config

ACTIVATION_LOGICAL_AXES = (
    "activation_batch",
    "activation_norm_length",
    "activation_embed",
)


def _declare_kda_roles(layer: KimiDeltaAttention) -> None:
    modules = [layer.decay_up, layer.output_gate_up, layer.out_proj]
    if layer.in_proj_mixer is not None:
        # The fused input projection is one KDA_MATRIX parameter. Muon would
        # orthogonalize its qkv/decay/beta/gate blocks jointly, which is why
        # config validation rejects fused_in_proj under muon-family
        # optimizers until blocked routing exists.
        modules.append(layer.in_proj_mixer)
    else:
        modules.extend(
            (
                layer.in_proj_qkv,
                layer.decay_down,
                layer.beta_proj,
                layer.output_gate_down,
            )
        )
    for module in modules:
        declare_dense_kernel(module, ParamRole.KDA_MATRIX)
    layer.conv1d.kernel = declare_parameter(layer.conv1d.kernel, ParamRole.DEPTHWISE_CONV)
    layer.A_log = declare_parameter(layer.A_log, ParamRole.KDA_SCALAR)
    layer.dt_bias = declare_parameter(layer.dt_bias, ParamRole.KDA_SCALAR)
    declare_norm(layer.output_norm)


def _remat_policy(name: str):
    if name == "full":
        return None
    if name == "minimal":
        return jax.checkpoint_policies.save_only_these_names(
            "qkv_proj", "out_proj", "mlpwi", "mlpwo"
        )
    if name == "minimal_with_context":
        return jax.checkpoint_policies.save_only_these_names(
            "qkv_proj", "context", "out_proj", "mlpwi", "mlpwo"
        )
    if name == "save_dot_except_mlp":
        return jax.checkpoint_policies.save_only_these_names(
            "qkv_proj", "out_proj"
        )
    raise ValueError(f"unknown rematerialization policy: {name}")


class HybridLayer(nnx.Module):
    """One standard pre-norm residual mixer plus fused SwiGLU."""

    def __init__(
        self,
        kind: str,
        *,
        config: ResolvedConfig,
        leaf_config,
        mesh,
        rngs: nnx.Rngs,
    ):
        model = config.model
        self.kind = kind
        self.dtype = leaf_config.dtype
        self.input_norm = RMSNorm(
            num_features=model.emb_dim,
            epsilon=model.rms_norm_epsilon,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("norm",),
            rngs=rngs,
        )
        declare_norm(self.input_norm)
        if kind == "kda":
            self.mixer = KimiDeltaAttention(
                config=leaf_config,
                mesh=mesh,
                model_mode=MODEL_MODE_TRAIN,
                rngs=rngs,
            )
            _declare_kda_roles(self.mixer)
        elif kind == "gqa":
            self.mixer = NoPEGQA(
                model.attention,
                emb_dim=model.emb_dim,
                max_target_length=config.data.sequence_length,
                dtype=leaf_config.dtype,
                weight_dtype=leaf_config.weight_dtype,
                leaf_config=leaf_config,
                mesh=mesh,
                rngs=rngs,
            )
        else:
            raise ValueError(f"unknown hybrid mixer: {kind}")
        self.post_mixer_norm = RMSNorm(
            num_features=model.emb_dim,
            epsilon=model.rms_norm_epsilon,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("norm",),
            rngs=rngs,
        )
        declare_norm(self.post_mixer_norm)
        self.mlp = MlpBlock(
            config=leaf_config,
            mesh=mesh,
            in_features=model.emb_dim,
            intermediate_dim=model.mlp_dim,
            activations=("silu", "linear"),
            intermediate_dropout_rate=model.dropout_rate,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            model_mode=MODEL_MODE_TRAIN,
            rngs=rngs,
        )
        declare_dense_kernel(self.mlp.wi, ParamRole.MLP_INPUT)
        declare_dense_kernel(self.mlp.wo, ParamRole.MLP_OUTPUT)

    def __call__(
        self,
        hidden_states,
        *,
        decoder_segment_ids=None,
        decoder_positions=None,
        record_max_logits: bool = False,
    ):
        residual = hidden_states
        normalized = self.input_norm(hidden_states)
        normalized = nn.with_logical_constraint(normalized, ACTIVATION_LOGICAL_AXES)
        if self.kind == "kda":
            mixed, _ = self.mixer(
                normalized,
                decoder_segment_ids=decoder_segment_ids,
                model_mode=MODEL_MODE_TRAIN,
            )
        else:
            mixed = self.mixer(
                normalized,
                decoder_segment_ids=decoder_segment_ids,
                decoder_positions=decoder_positions,
                record_max_logits=record_max_logits,
            )
        hidden_states = nn.with_logical_constraint(
            residual + mixed,
            ACTIVATION_LOGICAL_AXES,
        )
        mlp_input = nn.with_logical_constraint(
            self.post_mixer_norm(hidden_states),
            ACTIVATION_LOGICAL_AXES,
        )
        layer_output = hidden_states + self.mlp(mlp_input, deterministic=True)
        return nn.with_logical_constraint(layer_output, ACTIVATION_LOGICAL_AXES)


class HybridCycle(nnx.Module):
    """Owned four-layer `[KDA,KDA,KDA,NoPE-GQA]` scan unit."""

    def __init__(self, *, config: ResolvedConfig, leaf_config, mesh, rngs: nnx.Rngs):
        for index, kind in enumerate(config.model.cycle):
            setattr(
                self,
                f"layer_{index}",
                HybridLayer(
                    kind,
                    config=config,
                    leaf_config=leaf_config,
                    mesh=mesh,
                    rngs=rngs.fork(),
                ),
            )

    def __call__(
        self,
        hidden_states,
        *,
        decoder_segment_ids=None,
        decoder_positions=None,
        record_max_logits: bool = False,
    ):
        for index in range(4):
            hidden_states = getattr(self, f"layer_{index}")(
                hidden_states,
                decoder_segment_ids=decoder_segment_ids,
                decoder_positions=decoder_positions,
                record_max_logits=record_max_logits,
            )
        return hidden_states


class HybridLanguageModel(nnx.Module):
    """Standalone decoder-only language model with scanned hybrid cycles."""

    def __init__(self, config: ResolvedConfig, mesh, *, rngs: nnx.Rngs):
        self.config = config
        self.mesh = mesh
        self.leaf_config = make_leaf_config(config)
        model = config.model
        self.token_embedding = Embed(
            num_embeddings=model.vocab_size,
            num_features=model.emb_dim,
            config=self.leaf_config,
            mesh=mesh,
            dtype=self.leaf_config.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            rngs=rngs,
        )
        self.token_embedding.embedding = declare_parameter(
            self.token_embedding.embedding, ParamRole.EMBEDDING
        )
        self.cycles = nnx_scan.create_scanned_layers(
            lambda cycle_rngs: HybridCycle(
                config=config,
                leaf_config=self.leaf_config,
                mesh=mesh,
                rngs=cycle_rngs,
            ),
            length=model.num_cycles,
            param_scan_axis=model.param_scan_axis,
            metadata_axis_name="cycles",
            rngs=rngs,
        )
        self.final_norm = RMSNorm(
            num_features=model.emb_dim,
            epsilon=model.rms_norm_epsilon,
            dtype=self.leaf_config.dtype,
            weight_dtype=self.leaf_config.weight_dtype,
            kernel_axes=("norm",),
            rngs=rngs,
        )
        declare_norm(self.final_norm)
        if model.logits_via_embedding:
            # Tied output head: the LM head reads the embedding table
            # transposed, dropping vocab_size * emb_dim parameters and their
            # optimizer state. The embedding keeps its EMBEDDING (AdamW) role.
            self.logits = None
        else:
            self.logits = DenseGeneral(
                in_features_shape=model.emb_dim,
                out_features_shape=model.vocab_size,
                dtype=self.leaf_config.dtype,
                weight_dtype=self.leaf_config.weight_dtype,
                kernel_axes=("embed", "vocab"),
                matmul_precision="default",
                rngs=rngs,
            )
            declare_dense_kernel(self.logits, ParamRole.LOGITS)

    def _apply_cycles(
        self,
        hidden_states,
        *,
        decoder_segment_ids,
        decoder_positions,
        record_max_logits,
    ):
        graphdef, params, state = nnx.split(self.cycles, nnx.Param, ...)
        scan_axis = self.config.model.param_scan_axis
        if scan_axis != 0:
            params = jax.tree.map(lambda value: jnp.moveaxis(value, scan_axis, 0), params)
        length = self.config.model.num_cycles
        params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, length)
        state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, length)

        def cycle_fn(carry, scanned_variables):
            scanned_variables = maxtext_utils_nnx.nnx_remove_scan_axis(
                scanned_variables, "cycles"
            )
            current_params, current_state = scanned_variables
            cycle = nnx.merge(graphdef, current_params, current_state)
            output = cycle(
                carry,
                decoder_segment_ids=decoder_segment_ids,
                decoder_positions=decoder_positions,
                record_max_logits=record_max_logits,
            )
            _, _, new_state = nnx.split(cycle, nnx.Param, ...)
            return output, new_state

        cycle_fn = jax.checkpoint(
            cycle_fn,
            policy=_remat_policy(self.config.model.remat_policy),
            prevent_cse=False,
        )
        hidden_states, scanned_state = jax.lax.scan(cycle_fn, hidden_states, (params, state))
        scanned_state = maxtext_utils_nnx.nnx_add_scan_axis(scanned_state, "cycles", 0)
        nnx.update(self.cycles, scanned_state)
        return hidden_states

    def hidden_states(
        self,
        token_ids,
        *,
        decoder_segment_ids=None,
        decoder_positions=None,
        record_max_logits: bool = False,
    ):
        if decoder_segment_ids is None:
            decoder_segment_ids = jnp.ones_like(token_ids, dtype=jnp.int32)
        if decoder_positions is None:
            decoder_positions = jnp.broadcast_to(
                jnp.arange(token_ids.shape[1], dtype=jnp.int32), token_ids.shape
            )
        hidden_states = self.token_embedding(token_ids, model_mode=MODEL_MODE_TRAIN)
        hidden_states = nn.with_logical_constraint(hidden_states, ACTIVATION_LOGICAL_AXES)
        hidden_states = self._apply_cycles(
            hidden_states,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            record_max_logits=record_max_logits,
        )
        hidden_states = nn.with_logical_constraint(
            self.final_norm(hidden_states),
            ACTIVATION_LOGICAL_AXES,
        )
        return hidden_states

    def project_logits(self, hidden_states):
        """Materializes FP32 logits for evaluation and the reference loss."""
        if self.logits is None:
            embedding = jnp.asarray(
                self.token_embedding.embedding[...], dtype=self.leaf_config.dtype
            )
            logits = jax.lax.dot_general(
                hidden_states,
                embedding,
                (((hidden_states.ndim - 1,), (1,)), ((), ())),
            )
            return logits.astype(jnp.float32)
        return self.logits(hidden_states).astype(jnp.float32)

    def output_projection_kernel(self, dtype):
        """Returns the FP32 master LM head converted to its MXU traffic dtype."""
        if self.logits is None:
            return jnp.asarray(self.token_embedding.embedding[...], dtype=dtype).T
        return jnp.asarray(self.logits.kernel[...], dtype=dtype)

    def __call__(
        self,
        token_ids,
        *,
        decoder_segment_ids=None,
        decoder_positions=None,
        record_max_logits: bool = False,
    ):
        hidden_states = self.hidden_states(
            token_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            record_max_logits=record_max_logits,
        )
        return self.project_logits(hidden_states)


def count_parameters(model: nnx.Module) -> int:
    return sum(int(value.size) for value in jax.tree.leaves(nnx.state(model, nnx.Param)))


def attention_logit_intermediates(model: HybridLanguageModel):
    """Returns `[cycles,batch,query_heads]` maxima after a MuonClip forward."""
    return model.cycles.layer_3.mixer.max_logits.value
