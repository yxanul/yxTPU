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
