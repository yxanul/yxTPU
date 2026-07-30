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

"""Scanned gradient accumulation must be a pure memory/time trade.

Splitting one batch into N microbatches has to leave the update it produces
unchanged; otherwise accumulation silently alters the trajectory instead of
just fitting it into HBM.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context
from yxtpu_pretrain.train import _make_train_step


def _config(accumulate, batch_per_device):
    return load_config(
        model="kda_hybrid_128k",
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
            "model.vocab_size=256",
            "model.dtype=float32",
            "model.remat_policy=full",
            "data.sequence_length=16",
            f"data.per_device_batch_size={batch_per_device}",
            f"experiment.gradient_accumulation_steps={accumulate}",
        ],
    )


def _batch(total_rows, length, vocab):
    key = jax.random.key(5)
    tokens = jax.random.randint(key, (total_rows, length + 1), 1, vocab)
    return {
        "input_ids": tokens[:, :-1],
        "labels": tokens[:, 1:],
        "loss_mask": jnp.ones((total_rows, length), jnp.float32),
        "segment_ids": jnp.ones((total_rows, length), jnp.int32),
        "positions": jnp.broadcast_to(
            jnp.arange(length, dtype=jnp.int32), (total_rows, length)
        ),
    }


def _run(accumulate, rows):
    config = _config(accumulate, rows // accumulate)
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))
        batch = _batch(rows, config.data.sequence_length, config.model.vocab_size)
        metrics = _make_train_step(config)(state, batch)
        updated = jax.tree.map(
            np.asarray, nnx.state(state.model, nnx.Param).to_pure_dict()
        )
    return jax.tree.map(float, metrics), updated


def test_accumulated_update_matches_the_single_pass_update():
    rows = 4
    single, single_params = _run(1, rows)
    scanned, scanned_params = _run(4, rows)

    assert single["tokens"] == scanned["tokens"]
    np.testing.assert_allclose(single["loss"], scanned["loss"], rtol=2e-5)
    np.testing.assert_allclose(
        single["grad_norm"], scanned["grad_norm"], rtol=2e-4
    )

    flat_single = jax.tree.leaves(single_params)
    flat_scanned = jax.tree.leaves(scanned_params)
    assert len(flat_single) == len(flat_scanned) and flat_single
    worst = max(
        float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-6))
        for a, b in zip(flat_single, flat_scanned)
    )
    # Accumulation reassociates the sum over microbatches, so agreement is to
    # floating-point tolerance rather than bitwise.
    assert worst < 1.0e-4, worst


def test_two_microbatches_also_agree():
    single, _ = _run(1, 4)
    halved, _ = _run(2, 4)
    np.testing.assert_allclose(single["loss"], halved["loss"], rtol=2e-5)
    np.testing.assert_allclose(single["grad_norm"], halved["grad_norm"], rtol=2e-4)
