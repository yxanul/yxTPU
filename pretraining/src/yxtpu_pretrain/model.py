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


def _declare_kda_roles(layer: KimiDeltaAttention) -> None:
    for module in (
        layer.in_proj_qkv,
        layer.decay_down,
        layer.decay_up,
        layer.beta_proj,
        layer.output_gate_down,
        layer.output_gate_up,
        layer.out_proj,
    ):
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
        hidden_states = residual + mixed
        return hidden_states + self.mlp(self.post_mixer_norm(hidden_states), deterministic=True)


class HybridCycle(nnx.Module):
    """Owned four-layer `[KDA,KDA,KDA,NoPE-GQA]` scan unit."""

    def __init__(self, *, config: ResolvedConfig, leaf_config, mesh, rngs: nnx.Rngs):
        self.remat_policy = config.model.remat_policy
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
        policy = _remat_policy(self.remat_policy)
        for index in range(4):
            layer = getattr(self, f"layer_{index}")
            graphdef, params, state = nnx.split(layer, nnx.Param, ...)

            def apply_layer(params_in, state_in, inputs, *, layer_graphdef=graphdef):
                current_layer = nnx.merge(layer_graphdef, params_in, state_in)
                outputs = current_layer(
                    inputs,
                    decoder_segment_ids=decoder_segment_ids,
                    decoder_positions=decoder_positions,
                    record_max_logits=record_max_logits,
                )
                _, _, new_state = nnx.split(current_layer, nnx.Param, ...)
                return outputs, new_state

            hidden_states, new_state = jax.checkpoint(
                apply_layer,
                policy=policy,
                prevent_cse=False,
            )(
                params,
                state,
                hidden_states,
            )
            nnx.update(layer, new_state)
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

        hidden_states, scanned_state = jax.lax.scan(cycle_fn, hidden_states, (params, state))
        scanned_state = maxtext_utils_nnx.nnx_add_scan_axis(scanned_state, "cycles", 0)
        nnx.update(self.cycles, scanned_state)
        return hidden_states

    def __call__(
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
        hidden_states = self._apply_cycles(
            hidden_states,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            record_max_logits=record_max_logits,
        )
        return self.logits(self.final_norm(hidden_states)).astype(jnp.float32)


def count_parameters(model: nnx.Module) -> int:
    return sum(int(value.size) for value in jax.tree.leaves(nnx.state(model, nnx.Param)))


def attention_logit_intermediates(model: HybridLanguageModel):
    """Returns `[cycles,batch,query_heads]` maxima after a MuonClip forward."""
    return model.cycles.layer_3.mixer.max_logits.value
