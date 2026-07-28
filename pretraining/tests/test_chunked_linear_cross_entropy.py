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

"""The chunked linear cross entropy must match the materializing formula in
loss and in both gradients, for every block size and mask pattern."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yxtpu_pretrain.losses import chunked_linear_cross_entropy


def _fixture(batch=2, sequence=16, hidden=8, vocab=32, dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(5), 4)
    x = jax.random.normal(keys[0], (batch, sequence, hidden), dtype=jnp.float32)
    weights = 0.1 * jax.random.normal(keys[1], (hidden, vocab), dtype=jnp.float32)
    labels = jax.random.randint(keys[2], (batch, sequence), 0, vocab)
    # Edge labels and one fully masked row exercise the safe-label path.
    labels = labels.at[0, 0].set(0).at[0, 1].set(vocab - 1)
    mask = (jax.random.uniform(keys[3], (batch, sequence)) > 0.25).astype(jnp.float32)
    mask = mask.at[1, :].set(0.0)
    return x.astype(dtype), labels, mask, weights.astype(dtype)


def _reference_mean(x, labels, mask, weights):
    logits = jnp.einsum(
        "bth,hv->btv",
        x,
        weights,
        preferred_element_type=jnp.float32,
    )
    safe_labels = jnp.where(mask > 0, labels, 0)
    targets = jnp.take_along_axis(logits, safe_labels[..., None], axis=-1)[..., 0]
    per_token = jax.nn.logsumexp(logits, axis=-1) - targets
    return jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)


@pytest.mark.parametrize("block_tokens", (2, 4, 8, 16))
def test_loss_and_gradients_match_reference_at_every_block_size(block_tokens):
    x, labels, mask, weights = _fixture()

    def chunked(x_value, weight_value):
        loss, tokens = chunked_linear_cross_entropy(
            x_value, labels, mask, weight_value, block_tokens=block_tokens
        )
        return loss, tokens

    (loss, tokens), grads = jax.value_and_grad(chunked, argnums=(0, 1), has_aux=True)(
        x, weights
    )
    expected_loss, expected_grads = jax.value_and_grad(
        _reference_mean, argnums=(0, 3)
    )(x, labels, mask, weights)
    assert tokens == jnp.sum(mask)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grads[0], expected_grads[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grads[1], expected_grads[1], rtol=1e-5, atol=1e-6)


def test_bf16_operands_return_bf16_cotangents():
    x, labels, mask, weights = _fixture(dtype=jnp.bfloat16)

    def chunked(x_value, weight_value):
        loss, _ = chunked_linear_cross_entropy(
            x_value, labels, mask, weight_value, block_tokens=4
        )
        return loss

    grads = jax.grad(chunked, argnums=(0, 1))(x, weights)
    assert grads[0].dtype == jnp.bfloat16
    assert grads[1].dtype == jnp.bfloat16
    assert all(bool(jnp.all(jnp.isfinite(g.astype(jnp.float32)))) for g in grads)


def test_fully_masked_batch_returns_zero_loss_and_grads():
    x, labels, _, weights = _fixture()
    mask = jnp.zeros(labels.shape, jnp.float32)

    def chunked(x_value, weight_value):
        loss, tokens = chunked_linear_cross_entropy(
            x_value, labels, mask, weight_value, block_tokens=4
        )
        return loss, tokens

    (loss, tokens), grads = jax.value_and_grad(chunked, argnums=(0, 1), has_aux=True)(
        x, weights
    )
    assert tokens == 0
    assert loss == 0.0
    assert all(bool(jnp.all(g == 0)) for g in grads)


def test_indivisible_block_size_fails_closed():
    x, labels, mask, weights = _fixture()
    try:
        chunked_linear_cross_entropy(x, labels, mask, weights, block_tokens=5)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("indivisible block size was accepted")


def test_train_loss_switch_matches_standard_on_a_real_model():
    from flax import nnx

    from yxtpu_pretrain.config import load_config
    from yxtpu_pretrain.model import HybridLanguageModel
    from yxtpu_pretrain.runtime.mesh import create_mesh
    from yxtpu_pretrain.train import _loss

    def build(implementation):
        config = load_config(
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
                "model.attention.num_query_heads=2",
                "model.attention.num_kv_heads=1",
                "data.sequence_length=64",
                "model.vocab_size=256",
                "model.dtype=float32",
                "model.remat_policy=full",
                "model.logits_via_embedding=true",
                f"model.loss.implementation={implementation}",
                "model.loss.block_tokens=16",
            ],
        )
        mesh = create_mesh(config.hardware, allow_device_mismatch=True)
        return HybridLanguageModel(config, mesh, rngs=nnx.Rngs(11))

    keys = jax.random.split(jax.random.key(3), 2)
    batch = {
        "input_ids": jax.random.randint(keys[0], (2, 64), 0, 256),
        "labels": jax.random.randint(keys[1], (2, 64), 0, 256),
        "segment_ids": jnp.ones((2, 64), jnp.int32),
        "positions": jnp.broadcast_to(jnp.arange(64), (2, 64)).astype(jnp.int32),
        "loss_mask": jnp.ones((2, 64), jnp.float32).at[0, :8].set(0.0),
    }
    standard_loss, standard_aux = _loss(
        build("standard"), batch, record_max_logits=False
    )
    chunked_loss, chunked_aux = _loss(
        build("chunked"), batch, record_max_logits=False
    )
    assert chunked_aux["tokens"] == standard_aux["tokens"]
    np.testing.assert_allclose(chunked_loss, standard_loss, rtol=2e-5, atol=2e-5)
