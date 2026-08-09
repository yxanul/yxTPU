# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Head-specific sigmoid output gate for NoPE-GQA (G1, arXiv:2505.06708)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel, count_parameters
from yxtpu_pretrain.optimizers import build_optimizer, classify_parameters
from yxtpu_pretrain.runtime.mesh import create_mesh


def _config(output_gate: bool, **optimizer_flags):
    overrides = [
        "model.emb_dim=128",
        "model.mlp_dim=256",
        "model.num_layers=4",
        "model.num_cycles=1",
        "model.kda.num_heads=2",
        "model.kda.precision=full_fp32",
        "model.attention.num_query_heads=4",
        "model.attention.num_kv_heads=2",
        f"model.attention.output_gate={str(output_gate).lower()}",
        "data.sequence_length=64",
        "model.vocab_size=256",
        "model.dtype=float32",
        "model.remat_policy=full",
        "model.param_scan_axis=1",
    ]
    for name, value in optimizer_flags.items():
        overrides.append(f"optimizer.{name}={str(value).lower()}")
    return load_config(
        model="kda_hybrid_273m",
        optimizer="muon",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=overrides,
    )


def _model(config, seed=13):
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    return HybridLanguageModel(config, mesh, rngs=nnx.Rngs(seed))


def test_flag_off_changes_nothing():
    off = _model(_config(False))
    on = _model(_config(True))
    off_routes = classify_parameters(nnx.state(off, nnx.Param))
    on_routes = classify_parameters(nnx.state(on, nnx.Param))
    assert not any(route.role == "gqa_gate" for route in off_routes)
    gate_routes = [route for route in on_routes if route.role == "gqa_gate"]
    assert len(gate_routes) == 1
    # Param delta: num_cycles * emb * q_heads * head_dim.
    assert count_parameters(on) - count_parameters(off) == 1 * 128 * 4 * 128


def test_gate_routes_to_muon_with_the_per_head_alternate():
    config = _config(True)
    params = nnx.state(_model(config), nnx.Param)
    joint = next(
        route
        for route in classify_parameters(params)
        if route.role == "gqa_gate"
    )
    # Scanned kernel [embed, cycles, q_heads, head_dim].
    assert joint.optimizer == "muon"
    assert joint.reduction_axes == (0,)
    assert joint.output_axes == (2, 3)
    assert joint.batch_axes == (1,)
    per_head = next(
        route
        for route in classify_parameters(params, muon_per_head=True)
        if route.role == "gqa_gate"
    )
    assert per_head.output_axes == (3,)
    assert per_head.batch_axes == (1, 2)
    assert per_head.alt_kind == "per_head"
    # Test shape [128, 1, 4, 128]: sqrt(max(128, 512) / max(128, 128)).
    assert per_head.scale_compensation == pytest.approx(2.0)


def test_transform_updates_are_finite_with_the_gate():
    config = _config(True, muon_per_head=True, muon_per_head_scale_compensation=True)
    model = _model(config)
    params = nnx.state(model, nnx.Param)
    transform, _ = build_optimizer(model, config.optimizer)
    state = transform.init(params)
    gradients = jax.tree.map(jnp.ones_like, params)
    updates, _ = transform.update(gradients, state, params)
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(updates))


def test_zeroed_gate_kernel_halves_the_mixer_output():
    """With W_g = 0 the gate is sigmoid(0) = 0.5 everywhere, and out_proj is
    linear with no bias, so the gated mixer output must equal exactly half the
    ungated one. Both mixers are built from the same seed; the gate is
    constructed after qkv/out, so their draws are identical."""
    from yxtpu_pretrain.layers.nope_gqa import NoPEGQA
    from yxtpu_pretrain.runtime.leaf_config import make_leaf_config

    config_on = _config(True)
    config_off = _config(False)
    mesh = create_mesh(config_on.hardware, allow_device_mismatch=True)

    def build(config):
        return NoPEGQA(
            config.model.attention,
            emb_dim=config.model.emb_dim,
            max_target_length=config.data.sequence_length,
            dtype=jnp.float32,
            weight_dtype=jnp.float32,
            leaf_config=make_leaf_config(config),
            mesh=mesh,
            rngs=nnx.Rngs(29),
        )

    mixer_on = build(config_on)
    mixer_off = build(config_off)
    assert mixer_on.gate_proj is not None and mixer_off.gate_proj is None
    mixer_on.gate_proj.kernel.set_value(
        jnp.zeros_like(mixer_on.gate_proj.kernel.get_value())
    )
    hidden = jax.random.normal(jax.random.key(11), (2, 64, 128), dtype=jnp.float32)
    gated = np.asarray(mixer_on(hidden))
    ungated = np.asarray(mixer_off(hidden))
    np.testing.assert_allclose(gated, 0.5 * ungated, rtol=1e-6, atol=1e-7)


def test_full_model_forward_is_finite_with_the_gate():
    config = _config(True)
    model = _model(config, seed=31)
    tokens = jax.random.randint(jax.random.key(7), (2, 64), 0, 256)
    logits = model(tokens)
    assert bool(jnp.all(jnp.isfinite(logits)))
