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

"""The dense Qwen3.5 teacher must build, and must not grow a router.

``Qwen3_5DecoderLayer`` instantiated ``Qwen3_5SparseMoEBlock``
unconditionally, so neither released dense size (0.8B, 4B - both declare no
expert fields at all) could load. The branch keys off ``num_experts`` rather
than a separate flag so a dense config cannot disagree with itself and build
a one-expert router that no checkpoint will ever fill.

Runs on CPU at reduced width: what is under test is the branch and the
parameter tree, not the arithmetic - GDN and full-attention numerics are
covered by ``maxtext/tests/unit/qwen3_next_vs_reference_test.py``, whose
layers Qwen3.5 subclasses without overrides.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from maxtext import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.models.qwen3_5 import Qwen3_5DecoderLayer

BASE_CONFIG = "../maxtext/src/maxtext/configs/base.yml"


def _config(num_experts: int):
    argv = [
        "test_qwen35_teacher", BASE_CONFIG,
        "decoder_block=qwen3_5",
        "base_emb_dim=256", "base_num_decoder_layers=4",
        "base_num_query_heads=4", "base_num_kv_heads=2", "head_dim=64",
        "base_mlp_dim=512", "mlp_activations=['silu','linear']",
        "vocab_size=256", "normalization_layer_epsilon=1.0e-6",
        "logits_via_embedding=true",
        "inhomogeneous_layer_cycle_interval=4",
        f"num_experts={num_experts}",
        "gdn_conv_kernel_dim=4", "gdn_key_head_dim=32",
        "gdn_value_head_dim=32", "gdn_num_key_heads=4",
        "gdn_num_value_heads=8", "gdn_chunk_size=64",
        "partial_rotary_factor=0.25", "rope_max_timescale=10000000",
        "use_mrope=false", "enable_dropout=false",
        "per_device_batch_size=1", "max_target_length=128",
        "run_name=test_qwen35_teacher", "skip_jax_distributed_system=true",
        "enable_checkpointing=false",
        "attention=dot_product",  # no pallas kernel on CPU
    ]
    if num_experts > 1:
        argv += ["base_moe_mlp_dim=512", "num_experts_per_tok=2"]
    return pyconfig.initialize(argv)


@pytest.fixture(scope="module")
def mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "model"))


def _param_names(layer):
    _, params, _ = nnx.split(layer, nnx.Param, ...)
    return sorted(".".join(str(part) for part in path)
                  for path, _ in nnx.to_flat_state(params))


# layer 3 is the full-attention slot: (idx + 1) % interval == 0.
@pytest.mark.parametrize("layer_idx,kind", [(0, "gdn"), (3, "full_attention")])
def test_dense_config_builds_an_mlp_block_not_a_router(mesh, layer_idx, kind):
    config = _config(num_experts=1)
    layer = Qwen3_5DecoderLayer(
        config=config, mesh=mesh, model_mode=MODEL_MODE_TRAIN,
        layer_idx=layer_idx, rngs=nnx.Rngs(0),
    )
    assert layer.is_dense_mlp
    names = _param_names(layer)
    # SwiGLU: gate and up projections in, one projection out.
    assert {"mlp.wi_0.kernel", "mlp.wi_1.kernel", "mlp.wo.kernel"} <= set(names)
    assert not [n for n in names if "router" in n or "expert" in n]


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_dense_layer_runs_forward(mesh, layer_idx):
    config = _config(num_experts=1)
    layer = Qwen3_5DecoderLayer(
        config=config, mesh=mesh, model_mode=MODEL_MODE_TRAIN,
        layer_idx=layer_idx, rngs=nnx.Rngs(0),
    )
    batch, seq = 1, 128
    hidden = jax.random.normal(
        jax.random.key(0), (batch, seq, config.emb_dim), jnp.float32
    ).astype(config.dtype)
    output, _ = layer(
        hidden, jnp.ones((batch, seq), jnp.int32),
        jnp.arange(seq, dtype=jnp.int32)[None, :], True, MODEL_MODE_TRAIN,
    )
    assert output.shape == (batch, seq, config.emb_dim)
    assert bool(jnp.isfinite(output).all())


def test_moe_config_still_selects_the_sparse_block(mesh):
    layer = Qwen3_5DecoderLayer(
        config=_config(num_experts=4), mesh=mesh,
        model_mode=MODEL_MODE_TRAIN, layer_idx=0, rngs=nnx.Rngs(0),
    )
    assert not layer.is_dense_mlp


def test_the_shipped_4b_config_is_dense_and_matches_the_released_shapes():
    """Guards the YAML against drift from Qwen/Qwen3.5-4B's config.json."""
    import yaml

    with open("../maxtext/src/maxtext/configs/models/qwen3.5-4b.yml") as handle:
        config = yaml.safe_load(handle)
    assert config["num_experts"] == 1, "4B is dense"
    assert config == {**config, **{
        "base_emb_dim": 2560,            # hidden_size
        "base_num_decoder_layers": 32,   # num_hidden_layers
        "base_num_query_heads": 16,      # num_attention_heads
        "base_num_kv_heads": 4,          # num_key_value_heads
        "head_dim": 256,
        "base_mlp_dim": 9216,            # intermediate_size
        "vocab_size": 248320,            # padded; tokenizer has 248,077
        "logits_via_embedding": True,    # tie_word_embeddings
        "inhomogeneous_layer_cycle_interval": 4,  # full_attention_interval
        "gdn_num_key_heads": 16,         # linear_num_key_heads
        "gdn_num_value_heads": 32,       # linear_num_value_heads (asymmetric)
        "gdn_key_head_dim": 128,
        "gdn_value_head_dim": 128,
        "gdn_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "rope_max_timescale": 10000000,
        "decoder_block": "qwen3_5",
    }}
    # Text-only scoring: every mrope position row is identical, so plain
    # RoPE is exact rather than an approximation.
    assert config["use_mrope"] is False
