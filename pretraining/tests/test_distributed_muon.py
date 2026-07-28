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

"""Update parity between distributed_muon and optax.contrib.muon."""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh
from optax.contrib import MuonDimensionNumbers

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.optimizers.distributed_muon import distributed_muon


def _tree_fixture():
    params = {
        "w": jnp.linspace(-1.0, 1.0, 16 * 8, dtype=jnp.float32).reshape(16, 8),
        "h": jnp.linspace(0.5, -0.5, 6 * 3 * 4, dtype=jnp.float32).reshape(6, 3, 4),
        "b": jnp.ones((5,), dtype=jnp.float32),
    }
    dimension_numbers = {
        "w": MuonDimensionNumbers(reduction_axis=(0,), output_axis=(1,)),
        "h": MuonDimensionNumbers(reduction_axis=(0,), output_axis=(2,)),
        "b": None,
    }
    arguments = dict(
        learning_rate=0.01,
        ns_steps=5,
        weight_decay=0.1,
        consistent_rms=0.2,
        muon_weight_dimension_numbers=dimension_numbers,
    )
    return params, arguments


def test_distributed_muon_matches_reference_on_one_device():
    params, arguments = _tree_fixture()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    reference = optax.contrib.muon(**arguments)
    candidate = distributed_muon(mesh, **arguments)
    reference_state = reference.init(params)
    candidate_state = candidate.init(params)
    key = jax.random.key(11)
    for _ in range(3):
        key, subkey = jax.random.split(key)
        gradients = {
            name: jax.random.normal(jax.random.fold_in(subkey, index), value.shape)
            for index, (name, value) in enumerate(sorted(params.items()))
        }
        reference_updates, reference_state = reference.update(
            gradients, reference_state, params
        )
        candidate_updates, candidate_state = candidate.update(
            gradients, candidate_state, params
        )
        for name in params:
            np.testing.assert_allclose(
                candidate_updates[name],
                reference_updates[name],
                rtol=1e-6,
                atol=1e-7,
            )


_EIGHT_DEVICE_SCRIPT = r"""
import os

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh
from optax.contrib import MuonDimensionNumbers

from yxtpu_pretrain.optimizers.distributed_muon import distributed_muon

devices = np.asarray(jax.devices())
assert devices.size == 8, devices
mesh = Mesh(devices, ("data",))
params = {
    "w": jnp.linspace(-1.0, 1.0, 16 * 8, dtype=jnp.float32).reshape(16, 8),
    # Batch axis of 3 joins w's problem group only if shapes match; here the
    # (6, 4) problems stack to a group of 3 and pad 3 -> 8 across the mesh.
    "h": jnp.linspace(0.5, -0.5, 6 * 3 * 4, dtype=jnp.float32).reshape(6, 3, 4),
    "b": jnp.ones((5,), dtype=jnp.float32),
}
dimension_numbers = {
    "w": MuonDimensionNumbers(reduction_axis=(0,), output_axis=(1,)),
    "h": MuonDimensionNumbers(reduction_axis=(0,), output_axis=(2,)),
    "b": None,
}
arguments = dict(
    learning_rate=0.01,
    ns_steps=5,
    weight_decay=0.1,
    consistent_rms=0.2,
    muon_weight_dimension_numbers=dimension_numbers,
)
reference = optax.contrib.muon(**arguments)
candidate = distributed_muon(mesh, **arguments)
reference_state = reference.init(params)
candidate_state = candidate.init(params)
reference_step = jax.jit(reference.update)
candidate_step = jax.jit(candidate.update)
key = jax.random.key(11)
for _ in range(3):
    key, subkey = jax.random.split(key)
    gradients = {
        name: jax.random.normal(jax.random.fold_in(subkey, index), value.shape)
        for index, (name, value) in enumerate(sorted(params.items()))
    }
    reference_updates, reference_state = reference_step(
        gradients, reference_state, params
    )
    candidate_updates, candidate_state = candidate_step(
        gradients, candidate_state, params
    )
    for name in params:
        np.testing.assert_allclose(
            candidate_updates[name],
            reference_updates[name],
            rtol=1e-6,
            atol=1e-7,
        )
print("PARITY-OK")
"""


def test_distributed_muon_matches_reference_across_eight_devices():
    """Real sharded execution: XLA_FLAGS must precede jax import, so the
    eight-device parity run happens in a subprocess."""
    result = subprocess.run(
        [sys.executable, "-c", _EIGHT_DEVICE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "PARITY-OK" in result.stdout


def test_build_optimizer_distributed_flag_matches_default_updates():
    config = load_config(
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
        ],
    )
    from yxtpu_pretrain.runtime.mesh import create_mesh

    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(9))
    params = nnx.state(model, nnx.Param)
    gradients = jax.tree.map(jnp.ones_like, params)

    default_transform, _ = build_optimizer(model, config.optimizer)
    distributed_transform, _ = build_optimizer(
        model, config.optimizer.model_copy(update={"muon_distributed_ns": True})
    )
    default_state = default_transform.init(params)
    distributed_state = distributed_transform.init(params)
    assert jax.tree.structure(default_state) == jax.tree.structure(distributed_state)
    default_updates, _ = default_transform.update(gradients, default_state, params)
    distributed_updates, _ = distributed_transform.update(
        gradients, distributed_state, params
    )
    for (path, expected), (_, actual) in zip(
        nnx.to_flat_state(default_updates),
        nnx.to_flat_state(distributed_updates),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual.get_value() if hasattr(actual, "get_value") else actual),
            np.asarray(
                expected.get_value() if hasattr(expected, "get_value") else expected
            ),
            rtol=1e-6,
            atol=1e-7,
            err_msg=str(path),
        )
