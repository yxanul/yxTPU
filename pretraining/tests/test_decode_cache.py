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

"""Cached decoding must reproduce the training forward.

Incremental decode replaces the recomputed prefix with per-layer state, so
the property that matters is that it predicts the same next-token
distribution the full forward does at every position - otherwise generation
silently drifts from the trained model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.decode import (
    SamplingParams,
    generate,
    init_cache,
    model_step,
    split_cycles,
)
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context


def _tiny_config(sequence_length=32):
    return load_config(
        model="kda_hybrid_128k",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=[
            "model.emb_dim=128",
            "model.mlp_dim=256",
            "model.num_layers=8",
            "model.num_cycles=2",
            "model.residual_policy=block_attnres",
            # The config validator pins the production 128x128 recurrent state.
            "model.kda.num_heads=1",
            "model.kda.gate_rank=16",
            # On CPU the mixers already fall back to chunk_kda and the dot
            # product attention path, which is what decode is checked against.
            "model.kda.precision=full_fp32",
            "model.attention.num_query_heads=2",
            "model.attention.num_kv_heads=1",
            "model.attention.head_dim=32",
            f"data.sequence_length={sequence_length}",
            "data.per_device_batch_size=1",
            "model.vocab_size=256",
            "model.dtype=float32",
            "model.remat_policy=full",
        ],
    )


@pytest.fixture(scope="module")
def fixture():
    config = _tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
    return config, mesh, rules, model


def _reference_logits(model, mesh, rules, tokens):
    with logical_mesh_context(mesh, rules):
        hidden = model.hidden_states(tokens)
        kernel = model.output_projection_kernel(hidden.dtype)
        return np.asarray((hidden @ kernel).astype(jnp.float32))


def test_step_decode_matches_the_full_forward(fixture):
    config, mesh, rules, model = fixture
    batch, length = 2, 12
    tokens = jax.random.randint(
        jax.random.key(3), (batch, length), 1, config.model.vocab_size
    )
    reference = _reference_logits(model, mesh, rules, tokens)

    cycles = split_cycles(model)
    cache = init_cache(model, batch, max_length=length)
    stepped = []
    with logical_mesh_context(mesh, rules):
        for position in range(length):
            logits, cache = model_step(model, cycles, tokens[:, position], cache)
            stepped.append(np.asarray(logits))
    stepped = np.stack(stepped, axis=1)

    assert stepped.shape == reference.shape
    scale = np.abs(reference).max()
    error = np.abs(stepped - reference).max() / scale
    # The chunked training kernel and the one-step recurrence group their
    # arithmetic differently, so agreement is to floating-point tolerance,
    # not bitwise.
    assert error < 2.0e-4, error
    # What actually governs generation is the ranking, which must be exact.
    assert np.array_equal(stepped.argmax(-1), reference.argmax(-1))


def test_cache_state_is_position_invariant(fixture):
    """Feeding the same prefix in two calls leaves the same distribution as
    feeding it in one - the cache carries everything the prefix implies."""
    config, mesh, rules, model = fixture
    tokens = jax.random.randint(jax.random.key(11), (1, 8), 1, config.model.vocab_size)
    cycles = split_cycles(model)
    with logical_mesh_context(mesh, rules):
        cache = init_cache(model, 1, max_length=8)
        for position in range(8):
            whole, cache = model_step(model, cycles, tokens[:, position], cache)
        split_cache = init_cache(model, 1, max_length=8)
        for position in range(5):
            _, split_cache = model_step(
                model, cycles, tokens[:, position], split_cache)
        for position in range(5, 8):
            resumed, split_cache = model_step(
                model, cycles, tokens[:, position], split_cache)
    np.testing.assert_array_equal(np.asarray(whole), np.asarray(resumed))


def test_generate_greedy_matches_stepwise_argmax(fixture):
    """The jitted loop reproduces what stepwise greedy decoding produces,
    including ragged prompt handling."""
    config, mesh, rules, model = fixture
    prompts = jnp.asarray([[5, 9, 17, 0], [4, 12, 0, 0]], jnp.int32)
    lengths = jnp.asarray([3, 2], jnp.int32)
    max_new = 4
    with logical_mesh_context(mesh, rules):
        samples, _ = generate(
            model, prompts, lengths, jax.random.key(0),
            max_new_tokens=max_new, sampling=SamplingParams(temperature=0.0),
            end_token=-1, max_length=prompts.shape[1] + max_new,
        )
        samples = np.asarray(samples)
        cycles = split_cycles(model)
        for row in range(2):
            cache = init_cache(model, 1, max_length=prompts.shape[1] + max_new)
            length = int(lengths[row])
            fed = [int(t) for t in prompts[row, :length]]
            produced = []
            for index in range(length + max_new):
                token = fed[index] if index < length else produced[-1]
                logits, cache = model_step(
                    model, cycles, jnp.asarray([token], jnp.int32), cache)
                produced.append(int(np.asarray(logits)[0].argmax()))
            expected = produced[length - 1 : length - 1 + max_new]
            actual = samples[row, length - 1 : length - 1 + max_new]
            np.testing.assert_array_equal(actual, expected)


def test_end_token_stops_a_row(fixture):
    config, mesh, rules, model = fixture
    prompts = jnp.asarray([[5, 9, 17]], jnp.int32)
    lengths = jnp.asarray([3], jnp.int32)
    cycles = split_cycles(model)
    with logical_mesh_context(mesh, rules):
        cache = init_cache(model, 1, max_length=16)
        for position in range(3):
            logits, cache = model_step(model, cycles, prompts[:, position], cache)
        first = int(np.asarray(logits)[0].argmax())
        samples, done = generate(
            model, prompts, lengths, jax.random.key(0),
            max_new_tokens=6, sampling=SamplingParams(temperature=0.0),
            end_token=first, max_length=16,
        )
    assert bool(done[0])
    assert int(samples[0, 2]) == first
