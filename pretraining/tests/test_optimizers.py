import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import (
    apply_gqa_muonclip,
    build_optimizer,
    classify_parameters,
)
from yxtpu_pretrain.runtime.mesh import create_mesh


def _config(optimizer="muon", *, query_heads=1, kv_heads=1, scan_axis=1):
    return load_config(
        model="kda_hybrid_273m",
        optimizer=optimizer,
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
            f"model.attention.num_query_heads={query_heads}",
            f"model.attention.num_kv_heads={kv_heads}",
            "data.sequence_length=64",
            "model.vocab_size=256",
            "model.dtype=float32",
            "model.remat_policy=full",
            f"model.param_scan_axis={scan_axis}",
        ],
    )


def _model(config, seed=1):
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    return HybridLanguageModel(config, mesh, rngs=nnx.Rngs(seed))


@pytest.mark.parametrize(
    ("scan_axis", "reduction_axes", "batch_axes"),
    ((0, (1,), (0,)), (1, (0,), (1,))),
)
def test_route_is_exhaustive_and_scan_axis_is_a_muon_batch(
    scan_axis,
    reduction_axes,
    batch_axes,
):
    config = _config(scan_axis=scan_axis)
    model = _model(config)
    params = nnx.state(model, nnx.Param)
    routes = classify_parameters(params)
    assert len(routes) == len(nnx.to_flat_state(params))
    assert {route.optimizer for route in routes} == {"muon", "adamw"}
    qkv = next(
        route
        for route in routes
        if route.role == "gqa_qkv" and route.path[-1] == "kernel"
    )
    assert qkv.reduction_axes == reduction_axes
    assert qkv.output_axes == (2, 3)
    assert qkv.batch_axes == batch_axes
    assert next(route for route in routes if route.role == "embedding").optimizer == "adamw"
    assert next(route for route in routes if route.role == "depthwise_conv").optimizer == "adamw"


def test_unclassified_parameter_fails_closed():
    parameters = nnx.State({"mystery": nnx.Param(jnp.ones((2, 2))).to_state()})
    try:
        classify_parameters(parameters)
    except ValueError as error:
        assert "no declared optimizer role" in str(error)
    else:
        raise AssertionError("unclassified parameter was silently accepted")


def test_muon_transform_initializes_and_updates_all_routes():
    config = _config()
    model = _model(config, 3)
    transform, routes = build_optimizer(model, config.optimizer)
    params = nnx.state(model, nnx.Param)
    state = transform.init(params)
    gradients = jax.tree.map(jnp.ones_like, params)
    updates, new_state = transform.update(gradients, state, params)
    assert len(routes) == len(nnx.to_flat_state(params))
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(updates))
    assert jax.tree.structure(state) == jax.tree.structure(new_state)


def test_adamw_profile_routes_every_declared_role_to_adamw():
    config = _config("adamw")
    model = _model(config, 4)
    _, routes = build_optimizer(model, config.optimizer)
    assert {route.optimizer for route in routes} == {"adamw"}


def test_gqa_muonclip_changes_only_qk_and_preserves_optimizer_moments():
    config = _config("muonclip", query_heads=4, kv_heads=2)
    model = _model(config, 5)
    transform, _ = build_optimizer(model, config.optimizer)
    params = nnx.state(model, nnx.Param)
    optimizer_state = transform.init(params)
    moments_before = [value.copy() for value in jax.tree.leaves(optimizer_state)]
    before = {
        path: variable.get_value().copy()
        for path, variable in nnx.to_flat_state(params)
    }

    max_logits = jnp.asarray([[[200.0, 80.0, 400.0, 50.0]]], dtype=jnp.float32)
    telemetry = apply_gqa_muonclip(
        model,
        max_logits,
        tau=config.optimizer.qk_clip_tau,
        epsilon=config.optimizer.qk_clip_epsilon,
    )
    after = nnx.state(model, nnx.Param)
    changed_paths = []
    for path, variable in nnx.to_flat_state(after):
        if not jnp.array_equal(variable.get_value(), before[path]):
            changed_paths.append(path)
    assert changed_paths == [("cycles", "layer_3", "mixer", "qkv_proj", "kernel")]

    kernel = model.cycles.layer_3.mixer.qkv_proj.kernel.get_value()
    original = before[changed_paths[0]]
    # Layout is [embed, cycle, fused_head, dim]. Q heads 0:4 and K heads
    # 4:6 change; V heads 6:8 and every other parameter stay bit-identical.
    assert not jnp.array_equal(kernel[:, :, :6], original[:, :, :6])
    assert jnp.array_equal(kernel[:, :, 6:], original[:, :, 6:])
    coefficients = jnp.minimum(1.0, 100.0 / max_logits[0, 0])
    key_coefficients = jnp.min(coefficients.reshape(2, 2), axis=-1)
    post_clip_estimate = max_logits[0, 0] * jnp.sqrt(coefficients) * jnp.repeat(
        jnp.sqrt(key_coefficients), 2
    )
    assert jnp.max(post_clip_estimate) <= 100.0 + 1.0e-4
    assert telemetry.clipped_heads.tolist() == [2]
    for before_moment, after_moment in zip(
        moments_before, jax.tree.leaves(optimizer_state), strict=True
    ):
        assert jnp.array_equal(before_moment, after_moment)
