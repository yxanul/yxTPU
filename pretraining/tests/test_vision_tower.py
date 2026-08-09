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

"""CPU tests for the native vision pathway."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from yxtpu_pretrain.config import VisionConfig, load_config
from yxtpu_pretrain.layers.vision import VisionTower, splice_visual_embeddings
from yxtpu_pretrain.model import HybridLanguageModel, count_parameters
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context
from yxtpu_pretrain.runtime.vision_data import VisionBatchSpec, render_row

TINY_VISION = dict(
    enabled=True,
    encoder_layers=1,
    encoder_dim=32,
    encoder_heads=2,
    encoder_mlp_dim=64,
    patch_size=8,
    image_size=32,
    pixel_shuffle=2,
    placeholder_token_id=250,
    max_images_per_sequence=1,
)


def tiny_config():
    return load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=[
            "model.vocab_size=256",
            "model.emb_dim=64",
            "model.num_layers=1",
            "model.cycle=[gqa]",
            "model.num_cycles=1",
            "model.mlp_dim=128",
            "model.attention.num_query_heads=2",
            "model.attention.num_kv_heads=1",
            "model.attention.head_dim=32",
            "model.logits_via_embedding=true",
            "data.sequence_length=64",
            "data.per_device_batch_size=1",
            "hardware.device_count=1",
            "hardware.chips=1",
            "hardware.hosts=1",
            "hardware.mesh.data=1",
            "hardware.multi_host=false",
        ]
        + [f"model.vision.{key}={str(value).lower() if isinstance(value, bool) else value}"
           for key, value in TINY_VISION.items()],
    )


def test_vision_config_geometry():
    vision = VisionConfig(**TINY_VISION)
    assert vision.patch_grid == 4
    assert vision.token_grid == 2
    assert vision.visual_tokens_per_image == 4
    with pytest.raises(ValueError):
        VisionConfig(**{**TINY_VISION, "image_size": 40})


def test_splice_places_visual_slots_in_stream_order():
    batch, seq, dim = 2, 6, 3
    text = jnp.zeros((batch, seq, dim))
    ids = jnp.asarray(
        [[9, 250, 250, 7, 7, 7], [7, 7, 7, 7, 7, 7]], dtype=jnp.int32
    )
    visual = jnp.arange(batch * 2 * dim, dtype=jnp.float32).reshape(batch, 2, dim)
    spliced = splice_visual_embeddings(text, ids, visual, 250)
    np.testing.assert_allclose(np.asarray(spliced[0, 1]), np.asarray(visual[0, 0]))
    np.testing.assert_allclose(np.asarray(spliced[0, 2]), np.asarray(visual[0, 1]))
    np.testing.assert_allclose(np.asarray(spliced[0, 0]), 0.0)
    np.testing.assert_allclose(np.asarray(spliced[1]), 0.0)


def test_pixel_shuffle_rearranges_blocks_exactly():
    config = tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        tower = VisionTower(
            config.model.vision,
            config.model.emb_dim,
            leaf_config=make_leaf_config(config),
            mesh=mesh,
            rngs=nnx.Rngs(0),
        )
    grid = config.model.vision.patch_grid
    dim = 5
    tokens = jnp.arange(grid * grid * dim, dtype=jnp.float32).reshape(1, grid * grid, dim)
    shuffled = tower._pixel_shuffle(tokens)
    assert shuffled.shape == (1, 4, dim * 4)
    # Token 0 of the shuffled grid folds patches (0,0),(0,1),(1,0),(1,1) =
    # flat indices 0, 1, grid, grid+1 in that order.
    expected = jnp.concatenate(
        [tokens[0, 0], tokens[0, 1], tokens[0, grid], tokens[0, grid + 1]]
    )
    np.testing.assert_allclose(np.asarray(shuffled[0, 0]), np.asarray(expected))


def test_tower_output_shape_and_model_gradients_flow():
    config = tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
    assert model.vision_tower is not None
    vision = config.model.vision
    batch, seq = 2, config.data.sequence_length
    images = jnp.ones((batch, 1, vision.image_size, vision.image_size, 3)) * 0.1
    tokens = np.full((batch, seq), 7, dtype=np.int32)
    tokens[:, : vision.visual_tokens_per_image] = vision.placeholder_token_id
    tokens = jnp.asarray(tokens)

    with logical_mesh_context(mesh, rules):
        visual = model.vision_tower(images)
        assert visual.shape == (batch, vision.visual_tokens_per_image, config.model.emb_dim)

        def loss_fn(current):
            logits = current(tokens, images=images)
            return jnp.mean(logits**2)

        gradients = nnx.grad(loss_fn)(model)
    tower_leaves = jax.tree.leaves(gradients["vision_tower"])
    assert tower_leaves, "vision tower received no gradients"
    total = sum(float(jnp.sum(jnp.abs(leaf))) for leaf in tower_leaves)
    assert np.isfinite(total) and total > 0.0

    # An image change must move hidden states only through visual positions.
    with logical_mesh_context(mesh, rules):
        base = model(tokens, images=images)
        moved = model(tokens, images=images + 0.05)
    assert float(jnp.max(jnp.abs(base - moved))) > 0.0


def test_render_row_masks_visual_prefix_and_padding():
    spec = VisionBatchSpec(
        sequence_length=16,
        visual_tokens=4,
        image_size=8,
        placeholder_id=250,
        pad_id=0,
        eos_id=1,
    )
    image = np.zeros((8, 8, 3), dtype=np.float32)
    example = render_row([5, 6, 7], image, spec)
    assert example is not None
    ids, labels, mask = (
        example["input_ids"],
        example["labels"],
        example["loss_mask"],
    )
    assert list(ids[:4]) == [250] * 4
    # Labels inside the visual run are placeholders and carry no loss.
    assert mask[:3].sum() == 0.0
    # The boundary position (last placeholder -> first text token) is
    # supervised: predicting the first text token from the image is real.
    assert mask[3] == 1.0
    text_span = slice(4, 4 + 4)  # 3 text tokens + eos
    assert list(ids[text_span]) == [5, 6, 7, 1]
    assert mask[4] == 1.0 and mask[5] == 1.0 and mask[6] == 1.0
    assert mask[7:].sum() == 0.0
    assert example["segment_ids"][:7].all() and not example["segment_ids"][8:].any()
    assert example["images"].shape == (1, 8, 8, 3)
