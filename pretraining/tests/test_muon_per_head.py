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

"""Alternate Muon matricizations: per-head QKV (K3 §2.5), the KDA out_proj
whole-matrix fix, and the consistent-rms scale compensation — each isolated
behind its own flag so trajectory effects can be gated separately."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from optax.contrib import MuonDimensionNumbers
from optax.contrib._muon import orthogonalize_via_newton_schulz

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer, classify_parameters
from yxtpu_pretrain.runtime.mesh import create_mesh


def _config(**flags):
    overrides = [
        "model.emb_dim=128",
        "model.mlp_dim=256",
        "model.num_layers=4",
        "model.num_cycles=1",
        "model.kda.num_heads=2",
        "model.kda.precision=full_fp32",
        "model.attention.num_query_heads=4",
        "model.attention.num_kv_heads=2",
        "data.sequence_length=64",
        "model.vocab_size=256",
        "model.dtype=float32",
        "model.remat_policy=full",
        "model.param_scan_axis=1",
    ]
    for name, value in flags.items():
        overrides.append(f"optimizer.{name}={str(value).lower()}")
    return load_config(
        model="kda_hybrid_273m",
        optimizer="muon",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=overrides,
    )


def _model_and_params(config, seed=7):
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(seed))
    return model, nnx.state(model, nnx.Param)


def _routes(**flags):
    config = _config(**flags)
    _, params = _model_and_params(config)
    return classify_parameters(
        params,
        muon_per_head=config.optimizer.muon_per_head,
        muon_kda_out_proj_whole=config.optimizer.muon_kda_out_proj_whole,
    )


def _route_by_module(routes, role, module):
    return next(
        route
        for route in routes
        if route.role == role and route.path[-2] == module and route.path[-1] == "kernel"
    )


def _route(routes, role):
    return next(
        route
        for route in routes
        if route.role == role and route.path[-1] == "kernel"
    )


def test_default_routing_reproduces_the_historical_matricization():
    routes = _routes()
    qkv = _route(routes, "gqa_qkv")
    assert qkv.reduction_axes == (0,)
    assert qkv.output_axes == (2, 3)
    assert qkv.batch_axes == (1,)
    assert qkv.alt_kind is None and qkv.scale_compensation == 1.0
    out_proj = _route_by_module(routes, "kda_matrix", "out_proj")
    # The heads-only reduction the 50B run trained with, preserved exactly.
    assert out_proj.reduction_axes == (0,)
    assert out_proj.output_axes == (2, 3)
    assert out_proj.batch_axes == (1,)
    assert out_proj.alt_kind is None


def test_per_head_flag_switches_qkv_only():
    routes = _routes(muon_per_head=True)
    qkv = _route(routes, "gqa_qkv")
    # Scanned kernel [embed, cycles, qkv_heads, head_dim]: one NS block per
    # fused q/k/v head slot.
    assert qkv.reduction_axes == (0,)
    assert qkv.output_axes == (3,)
    assert qkv.batch_axes == (1, 2)
    assert qkv.alt_kind == "per_head"
    # Test shape [128, 1, 8, 128]: sqrt(max(128, 8*128) / max(128, 128)).
    assert qkv.scale_compensation == pytest.approx(math.sqrt(8.0))
    in_proj = _route_by_module(routes, "kda_matrix", "in_proj_qkv")
    assert in_proj.reduction_axes == (0,)
    assert in_proj.output_axes == (4,)
    assert in_proj.batch_axes == (1, 2, 3)
    # Test shape [128, 1, 3, 2, 128]: sqrt(max(128, 768) / max(128, 128)).
    assert in_proj.scale_compensation == pytest.approx(math.sqrt(6.0))
    out_proj = _route_by_module(routes, "kda_matrix", "out_proj")
    # out_proj is NOT bundled with per-head: it keeps the historical form.
    assert out_proj.reduction_axes == (0,)
    assert out_proj.output_axes == (2, 3)
    assert out_proj.alt_kind is None and out_proj.scale_compensation == 1.0


def test_out_proj_whole_flag_switches_out_proj_only():
    routes = _routes(muon_kda_out_proj_whole=True)
    out_proj = _route_by_module(routes, "kda_matrix", "out_proj")
    # Scanned kernel [heads, cycles, head_dim, embed]: whole-matrix
    # (heads*dim -> embed) matricization.
    assert out_proj.reduction_axes == (0, 2)
    assert out_proj.output_axes == (3,)
    assert out_proj.batch_axes == (1,)
    assert out_proj.alt_kind == "kda_out_proj_whole"
    # Test shape [2, 1, 128, 128]: sqrt(max(2, 16384) / max(256, 128)).
    assert out_proj.scale_compensation == pytest.approx(8.0)
    qkv = _route(routes, "gqa_qkv")
    assert qkv.output_axes == (2, 3) and qkv.alt_kind is None
    wi = _route(routes, "mlp_input")
    assert wi.reduction_axes == (0,) and wi.batch_axes == (1,)


def test_per_head_dimension_numbers_orthogonalize_each_head_block():
    coeffs = jnp.asarray((3.4445, -4.7750, 2.0315), dtype=jnp.float32)
    x = jax.random.normal(jax.random.key(3), (16, 2, 3, 8), dtype=jnp.float32)
    batched = orthogonalize_via_newton_schulz(
        x,
        coeffs,
        5,
        dimension_numbers=MuonDimensionNumbers(
            reduction_axis=(0,), output_axis=(3,)
        ),
    )
    for cycle in range(x.shape[1]):
        for head in range(x.shape[2]):
            single = orthogonalize_via_newton_schulz(x[:, cycle, head, :], coeffs, 5)
            np.testing.assert_allclose(
                batched[:, cycle, head, :], single, rtol=1e-5, atol=1e-5
            )


def test_scale_compensation_multiplies_exactly_the_per_head_leaves():
    """With weight decay zeroed, the compensated transform must equal the
    uncompensated one times the per-leaf factor on per-head-matricized
    leaves and be unchanged everywhere else."""
    plain_config = _config(muon_per_head=True, weight_decay=0.0)
    compensated_config = _config(
        muon_per_head=True,
        muon_per_head_scale_compensation=True,
        weight_decay=0.0,
    )
    model, params = _model_and_params(plain_config)
    gradients = jax.tree.map(jnp.ones_like, params)

    plain_transform, routes = build_optimizer(model, plain_config.optimizer)
    compensated_transform, _ = build_optimizer(model, compensated_config.optimizer)
    plain_updates, _ = plain_transform.update(
        gradients, plain_transform.init(params), params
    )
    compensated_updates, _ = compensated_transform.update(
        gradients, compensated_transform.init(params), params
    )
    factors = {
        route.path: (
            route.scale_compensation
            if route.optimizer == "muon" and route.alt_kind == "per_head"
            else 1.0
        )
        for route in routes
    }
    checked_scaled = 0
    for (path, expected), (_, actual) in zip(
        nnx.to_flat_state(plain_updates),
        nnx.to_flat_state(compensated_updates),
        strict=True,
    ):
        factor = factors[path]
        expected_value = np.asarray(
            expected.get_value() if hasattr(expected, "get_value") else expected
        )
        actual_value = np.asarray(
            actual.get_value() if hasattr(actual, "get_value") else actual
        )
        np.testing.assert_allclose(
            actual_value,
            expected_value * factor,
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"{path} factor={factor}",
        )
        if factor != 1.0:
            checked_scaled += 1
    assert checked_scaled >= 2  # gqa qkv + kda in_proj at minimum


def test_compensation_requires_per_head():
    try:
        _config(muon_per_head_scale_compensation=True)
    except ValueError as error:
        assert "requires" in str(error)
    else:
        raise AssertionError("compensation without per-head was accepted")


def test_transforms_initialize_and_update_under_every_flag_combination():
    for flags in (
        {"muon_per_head": True},
        {"muon_kda_out_proj_whole": True},
        {
            "muon_per_head": True,
            "muon_kda_out_proj_whole": True,
            "muon_per_head_scale_compensation": True,
        },
    ):
        config = _config(**flags)
        model, params = _model_and_params(config)
        transform, routes = build_optimizer(model, config.optimizer)
        state = transform.init(params)
        gradients = jax.tree.map(jnp.ones_like, params)
        updates, new_state = transform.update(gradients, state, params)
        assert all(
            jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(updates)
        ), flags
        assert jax.tree.structure(state) == jax.tree.structure(new_state)
