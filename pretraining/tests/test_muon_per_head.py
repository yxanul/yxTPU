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

"""Per-Head Muon routing (Kimi K3 §2.5)."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from optax.contrib import MuonDimensionNumbers
from optax.contrib._muon import orthogonalize_via_newton_schulz

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer, classify_parameters
from yxtpu_pretrain.runtime.mesh import create_mesh


def _config(*, muon_per_head: bool):
    return load_config(
        model="kda_hybrid_273m",
        optimizer="muon",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=[
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
            f"optimizer.muon_per_head={str(muon_per_head).lower()}",
        ],
    )


def _routes(muon_per_head: bool):
    config = _config(muon_per_head=muon_per_head)
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(7))
    params = nnx.state(model, nnx.Param)
    return (
        classify_parameters(params, muon_per_head=muon_per_head),
        config,
        model,
        params,
    )


def _route(routes, role, tail="kernel"):
    return next(
        route for route in routes if route.role == role and route.path[-1] == tail
    )


def test_default_routing_reproduces_the_historical_matricization():
    routes, _, _, _ = _routes(muon_per_head=False)
    qkv = _route(routes, "gqa_qkv")
    assert qkv.reduction_axes == (0,)
    assert qkv.output_axes == (2, 3)
    assert qkv.batch_axes == (1,)
    out_proj = next(
        route
        for route in routes
        if route.role == "kda_matrix" and route.path[-2] == "out_proj"
    )
    # The heads-only reduction the 50B run trained with, preserved exactly.
    assert out_proj.reduction_axes == (0,)
    assert out_proj.output_axes == (2, 3)
    assert out_proj.batch_axes == (1,)


def test_per_head_routing_moves_head_axes_into_muon_batch():
    routes, _, _, _ = _routes(muon_per_head=True)
    qkv = _route(routes, "gqa_qkv")
    # Scanned kernel [embed, cycles, qkv_heads, head_dim]: one NS block per
    # fused q/k/v head slot.
    assert qkv.reduction_axes == (0,)
    assert qkv.output_axes == (3,)
    assert qkv.batch_axes == (1, 2)
    in_proj = next(
        route
        for route in routes
        if route.role == "kda_matrix" and route.path[-2] == "in_proj_qkv"
    )
    # Scanned kernel [embed, cycles, 3, heads, head_dim].
    assert in_proj.reduction_axes == (0,)
    assert in_proj.output_axes == (4,)
    assert in_proj.batch_axes == (1, 2, 3)
    out_proj = next(
        route
        for route in routes
        if route.role == "kda_matrix" and route.path[-2] == "out_proj"
    )
    # Scanned kernel [heads, cycles, head_dim, embed]: the corrected
    # whole-matrix (heads*dim -> embed) matricization.
    assert out_proj.reduction_axes == (0, 2)
    assert out_proj.output_axes == (3,)
    assert out_proj.batch_axes == (1,)
    # Parameters without a per-head alternate are untouched.
    wi = next(
        route
        for route in routes
        if route.role == "mlp_input" and route.path[-1] == "kernel"
    )
    assert wi.reduction_axes == (0,)
    assert wi.batch_axes == (1,)


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


def test_per_head_transform_initializes_and_updates_all_routes():
    routes, config, model, params = _routes(muon_per_head=True)
    transform, built_routes = build_optimizer(model, config.optimizer)
    assert [route.path for route in built_routes] == [route.path for route in routes]
    state = transform.init(params)
    gradients = jax.tree.map(jnp.ones_like, params)
    updates, new_state = transform.update(gradients, state, params)
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(updates))
    assert jax.tree.structure(state) == jax.tree.structure(new_state)
    # The optimizer state tree keeps parameter shapes under either
    # matricization, so checkpoints remain layout-compatible across the flag.
    baseline_transform, _ = build_optimizer(
        model, config.optimizer.model_copy(update={"muon_per_head": False})
    )
    baseline_state = baseline_transform.init(params)
    assert [leaf.shape for leaf in jax.tree.leaves(state)] == [
        leaf.shape for leaf in jax.tree.leaves(baseline_state)
    ]