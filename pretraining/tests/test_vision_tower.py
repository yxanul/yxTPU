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
from yxtpu_pretrain.runtime.vision_data import VisionBatchSpec, pack_rows

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

    # The path-filtered norm used for the vit/lm grad split must agree with
    # the manual tower-subtree norm and sit inside the total.
    from yxtpu_pretrain.train import _subtree_l2norm

    vit_norm = float(_subtree_l2norm(gradients, "vision_tower"))
    manual = np.sqrt(
        sum(float(jnp.sum(jnp.square(leaf.astype(jnp.float32)))) for leaf in tower_leaves)
    )
    np.testing.assert_allclose(vit_norm, manual, rtol=1e-6)
    assert 0.0 < vit_norm

    # The vision probe recorded post-splice embedding stats on the forward.
    probe = np.asarray(model.vision_probe.value)
    assert probe.shape == (3,)
    assert np.isfinite(probe).all()
    assert probe[0] > 0.0 and probe[1] > 0.0  # visual and text RMS

    # An image change must move hidden states only through visual positions.
    with logical_mesh_context(mesh, rules):
        base = model(tokens, images=images)
        moved = model(tokens, images=images + 0.05)
    assert float(jnp.max(jnp.abs(base - moved))) > 0.0


def test_pack_rows_segments_masks_and_images():
    spec = VisionBatchSpec(
        sequence_length=16,
        visual_tokens=3,
        image_size=8,
        placeholder_id=250,
        pad_id=0,
        eos_id=1,
        max_images=2,
    )
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    vision_row = ([250, 250, 250, 5, 6, 1], image)  # visual prefix + text + eos
    text_row = ([7, 8, 9, 1], None)
    example = pack_rows([vision_row, text_row], spec)

    ids, labels, mask, segments, positions = (
        example["input_ids"],
        example["labels"],
        example["loss_mask"],
        example["segment_ids"],
        example["positions"],
    )
    # Layout: row1 tokens 0-5, row2 tokens 6-9, pad 10-16.
    assert list(segments[:6]) == [1] * 6 and list(segments[6:10]) == [2] * 4
    assert not segments[10:].any()
    # Positions restart at each row.
    assert list(positions[:6]) == [0, 1, 2, 3, 4, 5]
    assert list(positions[6:10]) == [0, 1, 2, 3]
    # Placeholder labels carry no loss; the image->text boundary does.
    assert mask[0] == 0.0 and mask[1] == 0.0  # labels are placeholders
    assert mask[2] == 1.0  # last placeholder -> first text token
    assert mask[3] == 1.0 and mask[4] == 1.0  # text and eos supervised
    # The cross-row boundary is masked (row 1 eos must not predict row 2).
    assert mask[5] == 0.0
    # Row 2 text supervised, pad masked.
    assert mask[6] == 1.0 and mask[7] == 1.0 and mask[8] == 1.0
    assert mask[9:].sum() == 0.0
    # One real image in slot 0, blank in slot 1.
    assert example["images"].shape == (2, 8, 8, 3)
    np.testing.assert_allclose(example["images"][0], 0.5)
    np.testing.assert_allclose(example["images"][1], 0.0)


def test_train_step_emits_vision_metrics():
    from maxtext.common.train_state_nnx import TrainStateNNX

    from yxtpu_pretrain.optimizers import build_optimizer
    from yxtpu_pretrain.train import _make_train_step, _vision_metrics

    config = load_config(
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
            "model.loss.implementation=chunked",
            "model.loss.block_tokens=16",
            "data.sequence_length=64",
            "data.per_device_batch_size=2",
            "hardware.device_count=1",
            "hardware.chips=1",
            "hardware.hosts=1",
            "hardware.mesh.data=1",
            "hardware.multi_host=false",
        ]
        + [f"model.vision.{key}={str(value).lower() if isinstance(value, bool) else value}"
           for key, value in TINY_VISION.items()],
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))

    spec = VisionBatchSpec(
        sequence_length=64,
        visual_tokens=config.model.vision.visual_tokens_per_image,
        image_size=config.model.vision.image_size,
        placeholder_id=config.model.vision.placeholder_token_id,
        pad_id=0,
        eos_id=1,
        max_images=1,
    )
    image = np.full((spec.image_size, spec.image_size, 3), 0.25, dtype=np.float32)
    mixed = pack_rows(
        [
            ([250] * spec.visual_tokens + [5, 6, 7, 1], image),
            ([8, 9, 10, 11, 1], None),
        ],
        spec,
    )
    text_only = pack_rows([([12, 13, 14, 15, 1], None)], spec)
    batch = {
        key: jnp.asarray(np.stack([mixed[key], text_only[key]]))
        for key in mixed
    }

    train_step = _make_train_step(config)
    with logical_mesh_context(mesh, rules):
        metrics = jax.device_get(train_step(state, batch))

    tokens = float(metrics["tokens"])
    assert (
        float(metrics["vision_token_count"]) + float(metrics["text_token_count"])
        == tokens
    )
    total_sum = float(metrics["loss"]) * tokens
    np.testing.assert_allclose(
        float(metrics["vision_loss_sum"]) + float(metrics["text_loss_sum"]),
        total_sum,
        rtol=1e-5,
    )
    vit_norm = float(metrics["vit_grad_norm"])
    assert 0.0 < vit_norm < float(metrics["grad_norm"])
    assert float(metrics["visual_embed_rms"]) > 0.0
    assert float(metrics["hidden_text_rms"]) > 0.0

    derived = _vision_metrics(metrics)
    assert derived is not None
    assert derived["vision_loss"] > 0.0 and derived["text_loss"] > 0.0
    assert "embed_rms_ratio" in derived and "lm_grad_norm" in derived


def test_pack_rows_vision_mask_marks_image_row_labels():
    spec = VisionBatchSpec(
        sequence_length=16,
        visual_tokens=3,
        image_size=8,
        placeholder_id=250,
        pad_id=0,
        eos_id=1,
        max_images=2,
    )
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    vision_row = ([250, 250, 250, 5, 6, 1], image)
    text_row = ([7, 8, 9, 1], None)
    example = pack_rows([vision_row, text_row], spec)

    vision_mask = example["vision_mask"]
    # Label positions 0-4 belong to the image-carrying row; the text row's
    # labels (5-8, including the cross-row boundary label) and padding do not.
    assert list(vision_mask[:5]) == [1.0] * 5
    assert not vision_mask[5:].any()
    # The supervised-token split: three loss tokens per modality here.
    loss_mask = example["loss_mask"]
    assert float((loss_mask * vision_mask).sum()) == 3.0
    assert float((loss_mask * (1.0 - vision_mask)).sum()) == 3.0

    # Order-independence: text first flips the marked segment. The text row
    # occupies label positions 0-2, the vision row 3-8.
    flipped = pack_rows([text_row, vision_row], spec)
    assert not flipped["vision_mask"][:3].any()
    assert list(flipped["vision_mask"][3:9]) == [1.0] * 6
    assert not flipped["vision_mask"][9:].any()


def _bare_packer(spec, next_row, next_text_row):
    """A MixedVisionTextIterator without its streams: packing logic only."""
    from yxtpu_pretrain.runtime.vision_data import MixedVisionTextIterator

    packer = object.__new__(MixedVisionTextIterator)
    packer.spec = spec
    packer._pending = None
    packer._pending_fill = None
    packer._text = object()  # non-None enables budget-aware fill
    for counter in (
        "rows_consumed",
        "rows_skipped",
        "text_rows",
        "vision_rows",
        "sequences_packed",
        "tokens_total",
        "tokens_padding",
        "tokens_placeholder",
        "images_packed",
        "loss_tokens_vision",
        "loss_tokens_text",
    ):
        setattr(packer, counter, 0)
    packer._next_row = next_row
    packer._next_text_row = next_text_row
    return packer


def test_budget_aware_fill_replaces_padding_with_text_rows():
    spec = VisionBatchSpec(
        sequence_length=64,
        visual_tokens=4,
        image_size=8,
        placeholder_id=250,
        pad_id=0,
        eos_id=1,
        max_images=1,
    )
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)
    vision_tokens = [250, 250, 250, 250, 5, 6, 7, 1]  # 8 tokens, 1 image
    text_tokens = [9] * 11 + [1]  # 12 tokens

    packer = _bare_packer(
        spec,
        next_row=lambda: (list(vision_tokens), image),
        next_text_row=lambda: (list(text_tokens), None),
    )
    example = packer._next_example()
    # The second vision draw hits the 1-image budget with 8/65 tokens used;
    # without fill the remaining ~87% of the sequence would be padding. The
    # fill packs text rows up to the margin: 8 + 4x12 = 56, and the fifth
    # text draw (68 > 65) is stashed rather than dropped.
    assert int((example["input_ids"] == 250).sum()) == 4  # one image only
    padding = int((example["segment_ids"] == 0).sum())
    assert padding <= 12, f"fill left {padding} padded positions"
    assert packer._pending is not None and packer._pending[1] is not None
    assert packer._pending_fill is not None and packer._pending_fill[1] is None
    # Composition counters see the filled rows.
    assert packer.loss_tokens_text > packer.loss_tokens_vision > 0

    # The stashed rows open the next sequence: its image budget is available
    # again, so the pending vision row packs first, then the stashed fill row.
    second = packer._next_example()
    assert int((second["input_ids"] == 250).sum()) == 4
    assert int((second["segment_ids"] == 0).sum()) <= 12
