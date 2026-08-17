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

"""CPU tests for the 2026-08 data-path changes: async device batches,
host patchify, multi-image rows, fetch threads, the rows holdout packing,
per-modality attention maxima, and mix calibration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.sharding import logical_mesh_context
from yxtpu_pretrain.runtime.vision_data import (
    MixedVisionTextIterator,
    VisionBatchSpec,
    pack_rows,
    pack_text_rows,
    patchify_pixels,
)

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
    max_images_per_sequence=2,
    max_images_per_row=2,
)


def tiny_config(**vision_overrides):
    vision = {**TINY_VISION, **vision_overrides}
    return load_config(
        model="kda_hybrid_273m",
        optimizer="muonclip",
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
        + [
            f"model.vision.{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in vision.items()
        ],
    )


def _spec(config, **overrides):
    vision = config.model.vision
    fields = dict(
        sequence_length=config.data.sequence_length,
        visual_tokens=vision.visual_tokens_per_image,
        image_size=vision.image_size,
        placeholder_id=vision.placeholder_token_id,
        pad_id=0,
        eos_id=1,
        max_images=vision.max_images_per_sequence,
        patch_size=vision.patch_size,
    )
    fields.update(overrides)
    return VisionBatchSpec(**fields)


# --------------------------------------------------------------- device batch


def test_device_batch_is_numpy_in_and_bitwise_identical():
    from yxtpu_pretrain.train import _device_batch

    config = tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rng = np.random.default_rng(0)
    host = {
        "input_ids": rng.integers(0, 256, (2, 64), dtype=np.int32),
        "loss_mask": rng.random((2, 64)).astype(np.float32),
        "images": rng.integers(0, 255, (2, 2, 32, 32, 3), dtype=np.uint8),
    }
    device = _device_batch(host, mesh)
    for key, value in host.items():
        assert device[key].dtype == value.dtype
        assert device[key].shape == value.shape
        np.testing.assert_array_equal(np.asarray(device[key]), value)
        assert device[key].sharding.spec == jax.sharding.PartitionSpec("data", None)


# ------------------------------------------------- patchify + multi-image rows


def test_host_patchify_matches_tower_patchify_and_multi_image_rows_pack():
    config = tiny_config()
    vision = config.model.vision
    rng = np.random.default_rng(1)
    image_a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    image_b = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    visual = vision.visual_tokens_per_image

    raw_spec = _spec(config)
    patch_spec = _spec(config, host_patchify=True)
    rows = [([250] * (2 * visual) + [5, 6, 7, 1], [image_a, image_b]), ([8, 9, 1], None)]
    raw = pack_rows(rows, raw_spec)
    patched = pack_rows(rows, patch_spec)

    # Two images of one row occupy slots 0 and 1 in row order; the second
    # (text) row has none. Token/mask arrays are layout-independent.
    assert raw["images"].shape == (2, 32, 32, 3)
    assert patched["images"].shape == (2, (32 // 8) ** 2, 8 * 8 * 3)
    np.testing.assert_array_equal(raw["images"][0], image_a)
    np.testing.assert_array_equal(patched["images"][1], patchify_pixels(image_b, 8))
    for key in ("input_ids", "labels", "loss_mask", "segment_ids", "positions", "vision_mask"):
        np.testing.assert_array_equal(raw[key], patched[key])
    assert list(raw["vision_mask"][: 2 * visual + 3]) == [1.0] * (2 * visual + 3)

    # A third image would exceed the 2-slot budget.
    with pytest.raises(ValueError):
        pack_rows(rows + [([250] * visual + [3, 1], image_a)], raw_spec)

    # The tower produces identical features from either layout.
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        from_raw = model.vision_tower(jnp.asarray(raw["images"][None]))
        from_patched = model.vision_tower(jnp.asarray(patched["images"][None]))
    np.testing.assert_array_equal(np.asarray(from_raw), np.asarray(from_patched))


# ------------------------------------------------------- fetch threads / rows


class _FakeImage:
    """Stands in for a PIL image: convert/resize return a fixed array."""

    def __init__(self, value):
        self.value = value

    def convert(self, mode):
        return self

    def resize(self, size):
        return self

    def __array__(self, dtype=None, copy=None):
        array = np.full((32, 32, 3), self.value, dtype=np.uint8)
        return array.astype(dtype) if dtype is not None else array


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        # One token per character, offset so ids never collide with the
        # placeholder or framing ids.
        return [10 + (ord(char) % 100) for char in text]


def _fake_streams(vision_rows, text_rows):
    """A load_dataset replacement serving fixed rows per dataset name."""

    class _Stream:
        def __init__(self, rows):
            self.rows = rows

        def shard(self, num_shards, index):
            return self

        def shuffle(self, seed, buffer_size):
            return self

        def __iter__(self):
            return iter(list(self.rows))

    def load_dataset(name, subset=None, split="train", streaming=True):
        return _Stream(vision_rows if "Vision" in name else text_rows)

    return load_dataset


def _iterator(monkeypatch, spec, *, row_buffer, max_images_per_row, seed=3):
    import datasets

    vision_rows = [
        {"images": [_FakeImage(1)], "texts": [{"user": "what", "assistant": "cat"}]},
        {"images": [_FakeImage(2), _FakeImage(3)],
         "texts": [{"user": "how many", "assistant": "two"}]},
        {"images": [], "texts": [{"user": "none", "assistant": "skip"}]},
        {"images": [_FakeImage(4)], "texts": [{"user": "where", "assistant": "left"}]},
    ] * 20
    text_rows = [{"text": "the quick brown fox jumps over the lazy dog " * 2}] * 200
    monkeypatch.setattr(datasets, "load_dataset", _fake_streams(vision_rows, text_rows))
    return MixedVisionTextIterator(
        tokenizer=_FakeTokenizer(),
        spec=spec,
        batch_size=2,
        vision_dataset="HuggingFaceM4/FineVisionMax",
        text_sources=[{"name": "text", "dataset": "corpus", "weight": 1.0,
                       "format": "plain", "row_tokens": 24}],
        p_text=0.5,
        shuffle_seed=seed,
        max_images_per_row=max_images_per_row,
        row_buffer=row_buffer,
    )


def test_fetch_threads_reproduce_the_inline_stream_and_admit_multi_image_rows(monkeypatch):
    config = tiny_config()
    spec = _spec(config, host_patchify=True)
    inline = _iterator(monkeypatch, spec, row_buffer=0, max_images_per_row=2)
    threaded = _iterator(monkeypatch, spec, row_buffer=16, max_images_per_row=2)
    for _ in range(6):
        a, b = next(inline), next(threaded)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    # Two-image rows were packed (both slots used somewhere) and empty-image
    # rows were skipped by the fetch thread, counted in the summed stats.
    stats = threaded.stats
    assert stats["images_per_sequence"] > 0
    assert threaded.raw_counters()["rows_skipped"] > 0
    assert threaded.fetch_depths.keys() == {"vision", "text"}
    calib = threaded.raw_source_rows()
    assert calib["vision"][0] == threaded.vision_rows and calib["text"][0] > 0

    single = _iterator(monkeypatch, spec, row_buffer=16, max_images_per_row=1)
    example = next(single)
    visual = config.model.vision.visual_tokens_per_image
    # With max_images_per_row 1 no row carries two images: every image row
    # is exactly one placeholder run of `visual` tokens.
    for sequence in example["input_ids"]:
        runs = np.diff(np.concatenate([[0], (sequence == 250).astype(int), [0]]))
        lengths = np.flatnonzero(runs == -1) - np.flatnonzero(runs == 1)
        assert all(length == visual for length in lengths)


def test_rows_packing_scores_documents_under_the_training_contract():
    from yxtpu_pretrain.runtime.data import PackedTokenBatcher

    class _Tok:
        eos_token_id = 1
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[10 + i for i in range(len(text))] for text in texts]}

    class _Config:
        sequence_length = 32
        tokenize_batch_size = 4
        text_field = "text"
        validation_fraction = 0.0
        validation_seed = 0
        append_eos = True

    records = [{"text": "x" * n} for n in (12, 40, 3, 9, 20, 15, 30, 7, 11)]
    batcher = PackedTokenBatcher(
        records * 4, _Tok(), _Config(), global_batch_size=2, vocab_size=256,
        validation=False, packing="rows", row_tokens=16,
    )
    batch = next(batcher)
    assert set(batch) == {"input_ids", "labels", "loss_mask", "segment_ids", "positions"}
    assert batch["input_ids"].shape == (2, 32)
    all_lengths = []
    for sequence, segments, positions, mask, labels in zip(
        batch["input_ids"], batch["segment_ids"], batch["positions"],
        batch["loss_mask"], batch["labels"],
    ):
        # Rows are separate segments with restarting positions ...
        for segment in np.unique(segments[segments > 0]):
            where = np.flatnonzero(segments == segment)
            assert list(positions[where]) == list(range(len(where)))
            # ... at most row_tokens content tokens + eos ...
            assert len(where) <= 17
            # ... and the eos-boundary label of every row is masked.
            last = where[-1]
            if last + 1 < len(segments) and segments[last + 1] != segment:
                assert mask[last] == 0.0
        # Documents shorter than 8 content tokens were dropped (3, 7 chars).
        lengths = [int((segments == s).sum()) for s in np.unique(segments[segments > 0])]
        assert all(length - 1 >= 8 for length in lengths)
        all_lengths.extend(lengths)
    # The 40-token document was truncated to 16 content tokens + eos.
    assert max(all_lengths) == 17 and 17 in all_lengths
    text_only = pack_text_rows([[5, 6, 7, 1], [8, 9, 1]], 8, 0)
    assert "images" not in text_only and "vision_mask" not in text_only
    assert list(text_only["segment_ids"]) == [1, 1, 1, 1, 2, 2, 2, 0]


# ---------------------------------------------------- per-modality maxima


def test_attention_maxima_split_by_modality():
    from yxtpu_pretrain.layers.nope_gqa import ABSENT_LOGIT
    from yxtpu_pretrain.model import (
        attention_logit_intermediates,
        attention_modality_logit_intermediates,
    )

    config = tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
    spec = _spec(config)
    visual = config.model.vision.visual_tokens_per_image
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)
    mixed = pack_rows([([250] * visual + [5, 6, 7, 1], image), ([8, 9, 10, 1], None)], spec)
    text = pack_rows([([12, 13, 14, 15, 1], None)], spec)

    def maxima(example):
        batch = {key: jnp.asarray(value[None]) for key, value in example.items()}
        with logical_mesh_context(mesh, rules):
            model.hidden_states(
                batch["input_ids"], images=batch["images"],
                decoder_segment_ids=batch["segment_ids"],
                decoder_positions=batch["positions"], record_max_logits=True,
            )
            joint = np.asarray(attention_logit_intermediates(model))
            split = np.asarray(attention_modality_logit_intermediates(model))
        return joint, split

    joint, split = maxima(mixed)
    assert joint.shape == (1, 1, 2) and split.shape == (1, 2, 2)
    np.testing.assert_allclose(split.max(axis=1), joint[:, 0], rtol=1e-6)
    assert (split[:, 0] <= joint[:, 0] + 1e-6).all() and (split[:, 1] <= joint[:, 0] + 1e-6).all()
    assert (split[:, 0] > ABSENT_LOGIT / 2).all()

    joint, split = maxima(text)
    assert (split[:, 0] <= ABSENT_LOGIT / 2).all()  # no visual positions
    np.testing.assert_allclose(split[:, 1], joint[:, 0], rtol=1e-6)


def test_residual_probe_reports_pre_norm_scales():
    from yxtpu_pretrain.model import residual_probe_intermediates

    config = tiny_config()
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        # Scale the final norm: a post-norm probe would move with it, the
        # pre-norm residual must not.
        spec = _spec(config)
        visual = config.model.vision.visual_tokens_per_image
        image = np.full((32, 32, 3), 0.5, dtype=np.float32)
        example = pack_rows([([250] * visual + [5, 6, 7, 1], image), ([8, 9, 1], None)], spec)
        batch = {key: jnp.asarray(value[None]) for key, value in example.items()}

        def probe():
            model.hidden_states(
                batch["input_ids"], images=batch["images"],
                decoder_segment_ids=batch["segment_ids"],
                decoder_positions=batch["positions"],
            )
            return np.asarray(residual_probe_intermediates(model))

        before = probe()
        model.final_norm.scale.value = model.final_norm.scale.value * 7.0
        after = probe()
    assert before[0] > 0 and before[1] > 0
    np.testing.assert_allclose(before, after, rtol=1e-6)


# ------------------------------------------------------------- calibration


def test_solve_weights_reproduces_targets():
    from yxtpu_pretrain.runtime.calibrate import parse_targets, solve_weights

    targets = parse_targets("vision=0.35,climbmix=0.40,stack=0.17,math=0.08")
    means = {"vision": 120.0, "climbmix": 511.0, "stack": 506.0, "math": 784.0}
    solution = solve_weights(targets, means)
    for name, share in targets.items():
        assert abs(solution["predicted_shares"][name] - share) < 1e-9, name
    assert abs(sum(solution["weights"].values()) - 1.0) < 1e-12
    assert 0.0 < solution["p_text"] < 1.0
    with pytest.raises(ValueError):
        parse_targets("vision=0.5,climbmix=0.6")
