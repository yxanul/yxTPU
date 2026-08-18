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

"""The cycle-hoisted AttnRes score path must reproduce the standalone reads."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from yxtpu_pretrain.layers.attn_res import DepthAttnRead


def _reads_and_inputs(sites=8, slots=5, batch=2, length=16, dim=64, dtype=jnp.bfloat16):
    reads = []
    for index in range(sites):
        read = DepthAttnRead(
            dim,
            epsilon=1.0e-5,
            dtype=dtype,
            weight_dtype=jnp.float32,
            rngs=nnx.Rngs(index),
        )
        # Zero-init pseudo-queries make every site identical; randomize both
        # the query and the norm scale so sites are distinguishable.
        keys = jax.random.split(jax.random.key(100 + index), 2)
        read.pseudo_query = nnx.Param(
            0.5 * jax.random.normal(keys[0], (dim,), jnp.float32)
        )
        read.norm.scale.set_value(
            1.0 + 0.1 * jax.random.normal(keys[1], (dim,), jnp.float32)
        )
        reads.append(read)
    data_keys = jax.random.split(jax.random.key(7), 2)
    buffer = jax.random.normal(
        data_keys[0], (slots, batch, length, dim), jnp.float32
    ).astype(dtype)
    # Zero the trailing slots like an early cycle would see them.
    buffer = buffer.at[3:].set(0)
    partial = jax.random.normal(
        data_keys[1], (batch, length, dim), jnp.float32
    ).astype(dtype)
    return reads, buffer, partial


def test_hoisted_scores_reproduce_standalone_reads():
    reads, buffer, partial = _reads_and_inputs()
    block_index = jnp.int32(2)
    folded = jnp.stack([read.folded_query() for read in reads], axis=-1).astype(
        buffer.dtype
    )
    raw_scores = jnp.einsum(
        "sbtd,dr->sbtr", buffer, folded, preferred_element_type=jnp.float32
    )
    sum_squares = jnp.einsum(
        "sbtd,sbtd->sbt", buffer, buffer, preferred_element_type=jnp.float32
    )
    for index, read in enumerate(reads):
        for include_partial in (False, True):
            standalone = read(
                buffer, block_index, partial, include_partial=include_partial
            )
            hoisted = read.read_with_scores(
                buffer,
                block_index,
                partial,
                raw_scores[..., index],
                sum_squares,
                include_partial=include_partial,
            )
            np.testing.assert_allclose(
                np.asarray(hoisted, dtype=np.float32),
                np.asarray(standalone, dtype=np.float32),
                rtol=2e-6,
                atol=2e-6,
                err_msg=f"site {index} include_partial={include_partial}",
            )
            assert bool(jnp.all(jnp.isfinite(hoisted.astype(jnp.float32))))


def test_zero_slots_stay_finite_at_production_epsilon():
    reads, buffer, partial = _reads_and_inputs()
    read = reads[0]
    # block_index 0: only slot 0 valid; slots 1..4 include all-zero sources
    # whose scores must stay finite before masking (0 * rsqrt(eps)).
    output = read(buffer, jnp.int32(0), partial, include_partial=True)
    assert bool(jnp.all(jnp.isfinite(output.astype(jnp.float32))))


def test_hoisted_numerators_reproduce_standalone_reads():
    """hoisted_depth_read + merge_hoisted (one buffer pass per cycle) must
    equal the standalone per-site reads to bf16 rounding, with and without
    the partial-sum slot, at every block_index including the all-masked
    tail slots."""
    from yxtpu_pretrain.layers.attn_res import hoisted_depth_read

    reads, buffer, partial = _reads_and_inputs()
    folded = jnp.stack([read.folded_query() for read in reads], axis=-1)
    for block_index in (0, 2, 4):
        numerators, normalizers, maxima = hoisted_depth_read(
            buffer, jnp.int32(block_index), folded, reads[0].norm.epsilon
        )
        assert len(numerators) == len(reads)
        for index, read in enumerate(reads):
            for include_partial in (False, True):
                standalone = read(
                    buffer, jnp.int32(block_index), partial, include_partial=include_partial
                )
                hoisted = read.merge_hoisted(
                    numerators[index],
                    normalizers[..., index],
                    maxima[..., index],
                    partial,
                    include_partial=include_partial,
                )
                np.testing.assert_allclose(
                    np.asarray(hoisted, dtype=np.float32),
                    np.asarray(standalone, dtype=np.float32),
                    rtol=1.6e-2,  # both paths round to bf16 at different points
                    atol=1.6e-2,
                    err_msg=f"block {block_index} site {index} partial={include_partial}",
                )
                assert bool(jnp.all(jnp.isfinite(hoisted.astype(jnp.float32))))


def test_hoisted_read_backward_matches_autodiff_of_the_einsum_path():
    """The hand-written VJP (one fp32-accumulated buffer cotangent) must
    match autodiff of the standalone per-site reads for the buffer AND the
    folded queries, in fp32 (so the comparison is not dominated by bf16)."""
    from yxtpu_pretrain.layers.attn_res import hoisted_depth_read

    reads, buffer, partial = _reads_and_inputs(dtype=jnp.float32)
    buffer = buffer.astype(jnp.float32)
    partial = partial.astype(jnp.float32)
    folded = jnp.stack([read.folded_query() for read in reads], axis=-1)
    block_index = jnp.int32(2)
    epsilon = reads[0].norm.epsilon
    keys = jax.random.split(jax.random.key(11), len(reads))
    cotangents = [
        jax.random.normal(key, partial.shape, jnp.float32) for key in keys
    ]

    def hoisted_loss(buffer, folded):
        numerators, normalizers, maxima = hoisted_depth_read(buffer, block_index, folded, epsilon)
        total = 0.0
        for index, read in enumerate(reads):
            out = read.merge_hoisted(
                numerators[index], normalizers[..., index], maxima[..., index],
                partial, include_partial=index % 2 == 1, folded_query=folded[:, index],
            )
            total = total + jnp.sum(out * cotangents[index])
        return total

    def einsum_loss(buffer, folded):
        # The standalone path, with the folded query injected so both losses
        # differentiate the same parameters.
        total = 0.0
        for index, read in enumerate(reads):
            def scores(values, q=folded[:, index]):
                dim = values.shape[-1]
                raw = jnp.einsum("d,...d->...", q, values, preferred_element_type=jnp.float32)
                ss = jnp.einsum("...d,...d->...", values, values, preferred_element_type=jnp.float32)
                return raw * jax.lax.rsqrt(ss / dim + epsilon)
            slots = buffer.shape[0]
            valid = jnp.arange(slots) <= block_index
            s = jnp.where(valid[:, None, None], scores(buffer), -1.0e30)
            include_partial = index % 2 == 1
            if include_partial:
                s = jnp.concatenate((s, scores(partial)[None]), axis=0)
            p = jax.nn.softmax(s, axis=0)
            out = jnp.einsum("sbt,sbtd->btd", p[:slots], buffer)
            if include_partial:
                out = out + p[slots][..., None] * partial
            total = total + jnp.sum(out * cotangents[index])
        return total

    ref_val, (ref_db, ref_dq) = jax.value_and_grad(einsum_loss, argnums=(0, 1))(buffer, folded)
    got_val, (got_db, got_dq) = jax.value_and_grad(hoisted_loss, argnums=(0, 1))(buffer, folded)
    np.testing.assert_allclose(float(got_val), float(ref_val), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(got_db), np.asarray(ref_db), rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(np.asarray(got_dq), np.asarray(ref_dq), rtol=2e-4, atol=2e-5)
    # masked slots receive no gradient
    assert float(jnp.abs(got_db[3:]).max()) == 0.0
