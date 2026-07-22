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


def test_muon_updates_use_consistent_rms_scaling():
    """A Muon-routed matrix update must have AdamW-like RMS (Moonshot's
    0.2 * sqrt(max(fan_in, fan_out)) convention), not optax's width-transfer
    default, whose ~0.02 RMS silently undertrains at a shared learning rate."""
    config = _config()
    model = _model(config, 3)
    transform, routes = build_optimizer(model, config.optimizer)
    params = nnx.state(model, nnx.Param)
    state = transform.init(params)
    gradients = jax.tree.map(
        lambda value: jax.random.normal(jax.random.key(0), value.shape, value.dtype),
        params,
    )
    # The warmup schedule starts at zero, so measure the second update
    # against the schedule's actual step-1 learning rate.
    from yxtpu_pretrain.optimizers.routing import build_learning_rate_schedule

    updates, state = transform.update(gradients, state, params)
    updates, _ = transform.update(gradients, state, params)
    matrix_route = next(route for route in routes if route.optimizer == "muon")
    flat = dict(nnx.to_flat_state(updates))
    update = flat[matrix_route.path].get_value()
    lr = float(build_learning_rate_schedule(config.optimizer)(1))
    rms = float(jnp.sqrt(jnp.mean(jnp.square(update / lr))))
    # Consistent-RMS scaling targets ~0.2; width-transfer leaves ~0.02-0.03.
    assert rms > 0.1, f"muon update RMS {rms} looks width-transfer scaled"


def test_muon_ns_bf16_casts_muon_updates_and_leaves_adam_fp32():
    config = _config()
    config = config.model_copy(
        update={
            "optimizer": config.optimizer.model_copy(
                update={"muon_ns_bf16": True}
            )
        }
    )
    model = _model(config, 3)
    transform, routes = build_optimizer(model, config.optimizer)
    params = nnx.state(model, nnx.Param)
    state = transform.init(params)
    gradients = jax.tree.map(
        lambda value: jax.random.normal(jax.random.key(0), value.shape, value.dtype),
        params,
    )
    updates, state = transform.update(gradients, state, params)
    # The final update dtype is promoted back to fp32 by the learning-rate
    # multiply, so probe the stored momentum instead: with the flag on,
    # exactly the Muon-routed leaves' mu must be bf16 (the masked gradient
    # cast plus bf16 storage is the pair that keeps Newton-Schulz in bf16).
    n_muon = sum(1 for route in routes if route.optimizer == "muon")
    assert n_muon > 0

    def bf16_leaf_count(tree):
        return sum(
            1
            for leaf in jax.tree.leaves(tree)
            if hasattr(leaf, "dtype") and leaf.dtype == jnp.bfloat16
        )

    # mu_dtype is a shared knob inside optax.contrib.muon: Muon momentum AND
    # the adam branch's first moment store bf16 (second moments stay fp32),
    # so every routed parameter contributes exactly one bf16 buffer.
    assert bf16_leaf_count(state) == len(routes)

    config_off = _config()
    transform_off, _ = build_optimizer(_model(config_off, 3), config_off.optimizer)
    state_off = transform_off.init(params)
    assert bf16_leaf_count(state_off) == 0


def test_terminal_decay_schedule_holds_peak_until_decay_window():
    """decay_steps carves the cosine out of only the schedule tail: peak is
    constant from warmup until schedule_steps - decay_steps, and the
    host-side telemetry mirror in train._learning_rate must agree with the
    Optax schedule everywhere (it is step-indexed, count = step - 1)."""
    from yxtpu_pretrain.optimizers.routing import build_learning_rate_schedule
    from yxtpu_pretrain.train import _learning_rate

    config = load_config(
        model="kda_hybrid_273m",
        optimizer="muonclip",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=[
            "optimizer.warmup_steps=40",
            "optimizer.schedule_steps=1000",
            "optimizer.decay_steps=100",
            "optimizer.final_learning_rate_fraction=0.0",
        ],
    )
    schedule = build_learning_rate_schedule(config.optimizer)
    peak = config.optimizer.learning_rate
    assert float(schedule(40)) == pytest.approx(peak)
    assert float(schedule(500)) == pytest.approx(peak)
    assert float(schedule(900)) == pytest.approx(peak)
    assert float(schedule(950)) == pytest.approx(0.5 * peak, rel=1e-6)
    assert float(schedule(1000)) == pytest.approx(0.0, abs=1e-12)
    for step in (1, 41, 501, 901, 951, 1001):
        assert _learning_rate(config, step) == pytest.approx(
            float(schedule(step - 1)), rel=1e-6, abs=1e-9
        )
