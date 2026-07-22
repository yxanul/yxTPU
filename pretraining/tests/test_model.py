import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.legacy_parity import export_legacy_parameters, legacy_path_for
from yxtpu_pretrain.model import (
    ACTIVATION_LOGICAL_AXES,
    HybridLanguageModel,
    count_parameters,
)
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context


def _tiny_config():
    return load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=[
            "model.emb_dim=128",
            "model.mlp_dim=256",
            "model.num_layers=4",
            "model.num_cycles=1",
            "model.kda.num_heads=1",
            "model.kda.precision=full_fp32",
            "model.attention.num_query_heads=1",
            "model.attention.num_kv_heads=1",
            "data.sequence_length=64",
            "data.per_device_batch_size=1",
            "model.vocab_size=256",
            "model.dtype=float32",
            "model.remat_policy=full",
        ],
    )


def test_hybrid_model_has_owned_cycle_and_semantic_roles():
    config = _tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(7))
    assert model.cycles.layer_0.kind == "kda"
    assert model.cycles.layer_3.kind == "gqa"
    assert count_parameters(model) > 900_000

    params = nnx.state(model, nnx.Param)
    roles = {
        variable.get_metadata().get("role")
        for variable in jax.tree.leaves(
            params,
            is_leaf=lambda value: isinstance(value, nnx.Variable),
        )
    }
    assert {
        "embedding",
        "logits",
        "norm_scale",
        "depthwise_conv",
        "kda_scalar",
        "kda_matrix",
        "gqa_qkv",
        "gqa_output",
        "mlp_input",
        "mlp_output",
    } <= roles


def test_certified_profile_has_272_9m_parameters():
    config = load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = nnx.eval_shape(
        lambda: HybridLanguageModel(config, mesh, rngs=nnx.Rngs(13))
    )
    assert count_parameters(model) == 272_935_520


def test_tiny_model_forward_is_finite_and_masks_padding():
    config = _tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(11))
    tokens = jax.random.randint(jax.random.key(3), (1, 64), 0, config.model.vocab_size)
    segments = jnp.ones_like(tokens)
    segments = segments.at[:, -8:].set(0)
    with logical_mesh_context(mesh, make_leaf_config(config).logical_axis_rules):
        logits = model(tokens, decoder_segment_ids=segments)
        hidden = model.hidden_states(tokens, decoder_segment_ids=segments)
        projected = model.project_logits(hidden)
    assert logits.shape == (1, 64, config.model.vocab_size)
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.allclose(logits, projected)


def test_model_preserves_logical_data_sharding_across_layer_boundaries(monkeypatch):
    config = _tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    recorded_axes = []
    original_constraint = nn.with_logical_constraint

    def record_constraint(value, axes, *args, **kwargs):
        recorded_axes.append(tuple(axes))
        return original_constraint(value, axes, *args, **kwargs)

    monkeypatch.setattr(nn, "with_logical_constraint", record_constraint)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(12))
    tokens = jnp.ones((1, 64), dtype=jnp.int32)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        assert tuple(nn_partitioning.get_axis_rules()) == tuple(rules)
        logits = model(tokens)
    assert logits.shape == (1, 64, config.model.vocab_size)
    # Embedding and final norm plus four constraints around each of four layers.
    assert recorded_axes.count(ACTIVATION_LOGICAL_AXES) >= 18


def test_legacy_adapter_is_exhaustive_and_converts_qwen_norm_scale():
    config = _tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(17))
    template = nnx.state(model, nnx.Param)
    legacy_flat = []
    for path, variable in nnx.to_flat_state(template):
        legacy_path, add_one = legacy_path_for(path)
        value = variable.get_value() - 1 if add_one else variable.get_value()
        legacy_flat.append((legacy_path, variable.replace(value=value)))
    converted = export_legacy_parameters(nnx.from_flat_state(legacy_flat), template)
    for (_, actual), (_, expected) in zip(
        nnx.to_flat_state(converted), nnx.to_flat_state(template), strict=True
    ):
        assert jnp.array_equal(actual.get_value(), expected.get_value())


def test_tied_embeddings_drop_head_parameters_and_match_manual_projection():
    untied = _tiny_config()
    tied = load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=[
            "model.emb_dim=128",
            "model.mlp_dim=256",
            "model.num_layers=4",
            "model.num_cycles=1",
            "model.kda.num_heads=1",
            "model.kda.precision=full_fp32",
            "model.attention.num_query_heads=1",
            "model.attention.num_kv_heads=1",
            "data.sequence_length=64",
            "data.per_device_batch_size=1",
            "model.vocab_size=256",
            "model.dtype=float32",
            "model.remat_policy=full",
            "model.logits_via_embedding=true",
        ],
    )
    mesh = create_mesh(untied.hardware, allow_device_mismatch=True)
    model_untied = HybridLanguageModel(untied, mesh, rngs=nnx.Rngs(7))
    model_tied = HybridLanguageModel(tied, mesh, rngs=nnx.Rngs(7))

    head_parameters = untied.model.vocab_size * untied.model.emb_dim
    assert (
        count_parameters(model_untied) - count_parameters(model_tied)
        == head_parameters
    )
    assert model_tied.logits is None

    tokens = jnp.arange(64, dtype=jnp.int32)[None, :] % tied.model.vocab_size
    with logical_mesh_context(mesh, make_leaf_config(tied).logical_axis_rules):
        hidden = model_tied.hidden_states(tokens)
        logits = model_tied.project_logits(hidden)
    embedding = jnp.asarray(model_tied.token_embedding.embedding[...], jnp.float32)
    expected = jnp.einsum(
        "bte,ve->btv", jnp.asarray(hidden, jnp.float32), embedding
    ) / jnp.sqrt(jnp.float32(tied.model.emb_dim))
    assert logits.shape == (1, 64, tied.model.vocab_size)
    assert jnp.allclose(logits, expected, rtol=2e-5, atol=2e-5)

    roles = {
        variable.get_metadata().get("role")
        for variable in jax.tree.leaves(
            nnx.state(model_tied, nnx.Param),
            is_leaf=lambda value: isinstance(value, nnx.Variable),
        )
    }
    assert "logits" not in roles
