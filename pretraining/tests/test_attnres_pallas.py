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

"""The fused Pallas depth read (interpret mode on CPU) must reproduce the
XLA hoisted read: outputs to bf16 rounding, gradients to fp32 tolerance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yxtpu_pretrain.kernels.attnres_pallas import pallas_hoisted_depth_read
from yxtpu_pretrain.layers.attn_res import hoisted_depth_read

EPS = 1.0e-5


def _inputs(slots=5, batch=2, length=64, dim=256, sites=8, dtype=jnp.bfloat16, seed=0):
    keys = jax.random.split(jax.random.key(seed), 3)
    buffer = jax.random.normal(keys[0], (slots, batch, length, dim), jnp.float32).astype(dtype)
    buffer = buffer.at[3:].set(0)  # trailing slots as an early cycle sees them
    queries = 0.5 * jax.random.normal(keys[1], (dim, sites), jnp.float32)
    return buffer, queries


@pytest.mark.parametrize("block_index", [0, 2, 4])
def test_fused_forward_matches_xla_hoisted_read(block_index):
    buffer, queries = _inputs()
    ref_n, ref_z, ref_m = hoisted_depth_read(buffer, jnp.int32(block_index), queries, EPS)
    got_n, got_z, got_m = pallas_hoisted_depth_read(
        buffer, jnp.int32(block_index), queries, EPS, forward_tile=32, backward_tile=16
    )
    assert len(got_n) == len(ref_n) == queries.shape[-1]
    np.testing.assert_allclose(np.asarray(got_z), np.asarray(ref_z), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(got_m), np.asarray(ref_m), rtol=1e-5, atol=1e-6)
    for k, (g, r) in enumerate(zip(got_n, ref_n)):
        assert g.dtype == r.dtype == buffer.dtype
        np.testing.assert_allclose(
            np.asarray(g, np.float32), np.asarray(r, np.float32), rtol=1.6e-2, atol=1.6e-2,
            err_msg=f"numerator {k}",
        )


def test_fused_backward_matches_xla_hoisted_read():
    buffer, queries = _inputs(dtype=jnp.float32)
    block_index = jnp.int32(2)
    keys = jax.random.split(jax.random.key(5), 3)
    cot_n = [jax.random.normal(keys[0], buffer.shape[1:], jnp.float32) * (k + 1) for k in range(8)]
    cot_z = jax.random.normal(keys[1], (*buffer.shape[1:3], 8), jnp.float32)

    def loss(read):
        def f(buffer, queries):
            n, z, m = read(buffer, block_index, queries, EPS)
            total = sum(jnp.sum(nk * ck) for nk, ck in zip(n, cot_n)) + jnp.sum(z * cot_z)
            # a maxima-dependent but shift-invariant consumer, like merge_hoisted
            total = total + jnp.sum(jnp.exp(m - jax.lax.stop_gradient(m)) * z)
            return total
        return f

    ref = jax.value_and_grad(loss(hoisted_depth_read), argnums=(0, 1))(buffer, queries)
    got = jax.value_and_grad(
        loss(lambda *a: pallas_hoisted_depth_read(*a, forward_tile=32, backward_tile=16)),
        argnums=(0, 1),
    )(buffer, queries)
    np.testing.assert_allclose(float(got[0]), float(ref[0]), rtol=1e-5)
    # dB is a sum of three terms of magnitude ~|dN| that partly cancel; the
    # two fp32 evaluation orders agree to ~2e-6 of the term scale, which is
    # what the absolute tolerance is set to (max |dB| ~ 2e2 here).
    ref_db = np.asarray(ref[1][0])
    np.testing.assert_allclose(
        np.asarray(got[1][0]), ref_db, rtol=1e-3, atol=5e-6 * float(np.abs(ref_db).max())
    )
    ref_dq = np.asarray(ref[1][1])
    np.testing.assert_allclose(
        np.asarray(got[1][1]), ref_dq, rtol=1e-3, atol=5e-6 * float(np.abs(ref_dq).max())
    )
    assert float(jnp.abs(got[1][0][3:]).max()) == 0.0  # masked slots: no gradient


def test_model_with_fused_read_matches_xla_read_end_to_end():
    """A tiny block_attnres model (2 cycles, so the buffer has valid AND
    masked slots at cycle 0) computes the same logits and parameter
    gradients with attnres_read=pallas (interpret mode) as with xla."""
    from flax import nnx

    from yxtpu_pretrain.config import load_config
    from yxtpu_pretrain.model import HybridLanguageModel
    from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
    from yxtpu_pretrain.runtime.mesh import create_mesh
    from yxtpu_pretrain.runtime.sharding import logical_mesh_context

    def build(read):
        return load_config(
            model="kda_hybrid_273m",
            optimizer="adamw",
            data="synthetic",
            hardware="v6e-8",
            experiment="selected",
            overrides=[
                "model.emb_dim=128",
                "model.mlp_dim=256",
                "model.num_layers=8",
                "model.num_cycles=2",
                "model.kda.num_heads=1",
                "model.kda.precision=full_fp32",
                "model.attention.num_query_heads=1",
                "model.attention.num_kv_heads=1",
                "data.sequence_length=64",
                "data.per_device_batch_size=1",
                "model.vocab_size=256",
                "model.dtype=float32",
                "model.remat_policy=full",
                "model.residual_policy=block_attnres",
                f"model.attnres_read={read}",
                "model.attnres_forward_tile=32",
                "model.attnres_backward_tile=16",
            ],
        )

    tokens = (jnp.arange(64, dtype=jnp.int32)[None, :] * 7) % 256
    results = {}
    for read in ("xla", "pallas"):
        config = build(read)
        mesh = create_mesh(config.hardware, allow_device_mismatch=True)
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(11))

        def loss_fn(model):
            logits = model(tokens)
            loss = jnp.mean(
                jax.nn.logsumexp(logits, axis=-1)
                - jnp.take_along_axis(logits, tokens[..., None], axis=-1)[..., 0]
            )
            return loss, logits

        with logical_mesh_context(mesh, make_leaf_config(config).logical_axis_rules):
            (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        results[read] = (float(loss), np.asarray(logits), jax.tree.leaves(nnx.to_pure_dict(nnx.state(grads, nnx.Param)) if not isinstance(grads, nnx.State) else nnx.to_pure_dict(grads)))

    assert len(results["xla"][2]) > 20 and len(results["xla"][2]) == len(results["pallas"][2])
    np.testing.assert_allclose(results["pallas"][0], results["xla"][0], rtol=1e-5)
    np.testing.assert_allclose(results["pallas"][1], results["xla"][1], rtol=1e-4, atol=1e-4)
    for got, want in zip(results["pallas"][2], results["xla"][2]):
        scale = float(np.abs(np.asarray(want)).max()) or 1.0
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-3, atol=2e-5 * scale)
