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

"""GOLD loss and alignment: the teacher projection must be exact, the
divergence must vanish when the student matches the projected teacher, and
the byte walker must mask exactly the non-1:1 groups."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yxtpu_pretrain.distillation import (
    align_by_byte_offsets,
    blockwise_logsumexp,
    direct_teacher_ids,
    gold_position_loss,
    project_teacher_logits,
    validate_student_to_teacher,
    verify_direct_map,
)


def test_blockwise_logsumexp_matches_dense():
    logits = jax.random.normal(jax.random.key(0), (3, 5, 1000)) * 4.0
    dense = jax.scipy.special.logsumexp(logits.astype(jnp.float32), axis=-1)
    blocked = blockwise_logsumexp(logits, block=96)  # 1000 % 96 != 0
    np.testing.assert_allclose(
        np.asarray(blocked), np.asarray(dense), rtol=1e-6, atol=1e-6
    )


def test_projection_is_the_exact_teacher_distribution():
    teacher_vocab, student_vocab = 64, 24
    key = jax.random.key(1)
    teacher_logits = jax.random.normal(key, (2, 7, teacher_vocab)) * 3.0
    mapping = jnp.asarray(
        np.random.default_rng(2).choice(teacher_vocab, student_vocab, replace=False),
        jnp.int32,
    )
    matched, residual = project_teacher_logits(
        teacher_logits, mapping, block=16
    )
    dense = jax.nn.log_softmax(teacher_logits.astype(jnp.float32), axis=-1)
    expected = np.take_along_axis(
        np.asarray(dense),
        np.broadcast_to(np.asarray(mapping), (2, 7, student_vocab)),
        axis=-1,
    )
    np.testing.assert_allclose(np.asarray(matched), expected, rtol=1e-5, atol=1e-6)
    expected_residual = 1.0 - np.exp(expected).sum(-1)
    np.testing.assert_allclose(
        np.asarray(residual), expected_residual, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_loss_vanishes_when_student_matches_projected_teacher(beta):
    teacher_vocab, student_vocab = 40, 16
    teacher_logits = jax.random.normal(jax.random.key(3), (1, 5, teacher_vocab))
    mapping = jnp.arange(student_vocab, dtype=jnp.int32) * 2
    matched, residual = project_teacher_logits(teacher_logits, mapping, block=8)
    # A student whose logits are exactly the renormalized projection.
    student_logits = matched - jnp.log1p(-residual)[..., None]
    loss, metrics = gold_position_loss(
        student_logits, matched, residual, jnp.ones((1, 5)), beta=beta
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["distill_tokens"]) == 5.0


def test_forward_kl_matches_numpy_reference():
    student_vocab = 12
    key_s, key_t = jax.random.split(jax.random.key(4))
    student_logits = jax.random.normal(key_s, (2, 3, student_vocab))
    teacher_logprobs = jax.nn.log_softmax(
        jax.random.normal(key_t, (2, 3, student_vocab)), axis=-1
    )
    residual = jnp.zeros((2, 3))
    mask = jnp.asarray([[1, 1, 0], [1, 0, 0]], jnp.float32)
    loss, _ = gold_position_loss(
        student_logits, teacher_logprobs, residual, mask, beta=0.0
    )
    p_t = np.exp(np.asarray(teacher_logprobs))
    log_s = np.asarray(jax.nn.log_softmax(student_logits.astype(jnp.float32), -1))
    kl = (p_t * (np.asarray(teacher_logprobs) - log_s)).sum(-1)
    expected = (kl * np.asarray(mask)).sum() / 3.0
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)


def test_masked_positions_receive_no_gradient():
    student_vocab = 8
    teacher_logprobs = jax.nn.log_softmax(
        jax.random.normal(jax.random.key(5), (1, 4, student_vocab)), axis=-1
    )
    mask = jnp.asarray([[1.0, 0.0, 1.0, 0.0]])

    def loss_of(logits):
        loss, _ = gold_position_loss(
            logits, teacher_logprobs, jnp.zeros((1, 4)), mask
        )
        return loss

    gradient = jax.grad(loss_of)(jnp.zeros((1, 4, student_vocab)))
    per_position = np.abs(np.asarray(gradient)).sum(-1)[0]
    assert per_position[0] > 0 and per_position[2] > 0
    assert per_position[1] == 0 and per_position[3] == 0


def test_vocab_padding_is_excluded_from_the_normalizer():
    """Qwen3.5 pads its head to 248,320 over a 248,077-token tokenizer, and
    the pad rows are initialized parameters that emit real logits. Summing
    them into the partition function deflates every matched log-probability
    and inflates the residual."""
    real_vocab, padded_width, student_vocab = 32, 48, 8
    key_real, key_pad = jax.random.split(jax.random.key(6))
    real = jax.random.normal(key_real, (2, 3, real_vocab)) * 2.0
    padding = jax.random.normal(key_pad, (2, 3, padded_width - real_vocab)) * 2.0
    padded = jnp.concatenate([real, padding], axis=-1)
    mapping = jnp.arange(student_vocab, dtype=jnp.int32) * 2  # all < real_vocab

    bounded, bounded_residual = project_teacher_logits(
        padded, mapping, block=16, valid_vocab=real_vocab
    )
    reference, reference_residual = project_teacher_logits(
        real, mapping, block=16
    )
    # Bounding recovers the unpadded answer exactly.
    np.testing.assert_allclose(
        np.asarray(bounded), np.asarray(reference), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(bounded_residual), np.asarray(reference_residual),
        rtol=1e-6, atol=1e-6,
    )
    # Without the bound the padding steals mass from every real token.
    leaked, leaked_residual = project_teacher_logits(padded, mapping, block=16)
    assert np.all(np.asarray(leaked) < np.asarray(reference))
    assert np.all(np.asarray(leaked_residual) > np.asarray(reference_residual))


# --------------------------------------------------------------- direct map


class _ToyTokenizer:
    """Greedy longest-match tokenizer over an explicit piece list."""

    def __init__(self, pieces):
        self._pieces = list(pieces)
        self._index = {piece: i for i, piece in enumerate(self._pieces)}
        self._longest = max(len(piece) for piece in self._pieces)

    def encode(self, text, add_special_tokens=False):
        ids, cursor = [], 0
        while cursor < len(text):
            for length in range(min(self._longest, len(text) - cursor), 0, -1):
                piece = text[cursor:cursor + length]
                if piece in self._index:
                    ids.append(self._index[piece])
                    cursor += length
                    break
            else:
                raise ValueError(f"unencodable {text[cursor]!r}")
        return ids

    def decode(self, ids):
        return "".join(self._pieces[int(i)] for i in ids)


LETTERS = list("abcdefg ")


def _toy_pair():
    """A student with no merges against a teacher that has some."""
    student = _ToyTokenizer(LETTERS)
    teacher = _ToyTokenizer(LETTERS + ["ab", "cd", "abc"])
    mapping = np.arange(len(LETTERS), dtype=np.int32)  # letters share ids
    return student, teacher, mapping


def test_direct_teacher_ids_supervises_position_for_position():
    _, _, mapping = _toy_pair()
    student_ids = jnp.asarray([[3, 0, 5], [1, 1, 2]], jnp.int32)
    teacher_ids = direct_teacher_ids(student_ids, mapping)
    assert teacher_ids.shape == student_ids.shape
    np.testing.assert_array_equal(
        np.asarray(teacher_ids), np.asarray(mapping)[np.asarray(student_ids)]
    )


def test_direct_map_round_trips_where_segmentation_differs():
    """The property the whole strategy rests on: the teacher reads back the
    same text even though it would have merged 'ab' into one token."""
    student, teacher, mapping = _toy_pair()
    report = verify_direct_map(student, teacher, mapping, ["abcd efg", "abc"])
    assert report.roundtrip_rate == 1.0
    assert not report.mismatches
    # Teacher's own BPE is coarser, so the student's boundaries are extra.
    assert report.fertility > 1.0
    assert 0.0 < report.canonical_rate < 1.0


def test_verify_direct_map_catches_a_map_that_does_not_round_trip():
    student, teacher, mapping = _toy_pair()
    broken = mapping.copy()
    broken[0] = mapping[1]  # 'a' now decodes as 'b'
    report = verify_direct_map(student, teacher, broken, ["abc", "cab"])
    assert report.roundtrip_rate == 0.0
    assert report.mismatches and report.mismatches[0][0] == "abc"


def test_non_injective_map_is_rejected_before_it_biases_the_residual():
    _, _, mapping = _toy_pair()
    collided = mapping.copy()
    collided[0] = collided[1]
    with pytest.raises(ValueError, match="not injective"):
        validate_student_to_teacher(collided)
    # Why it matters: the doubled token is counted twice, so the true
    # residual goes negative and project_teacher_logits silently clips it.
    teacher_logits = jnp.zeros((1, 1, len(LETTERS)))
    _, residual = project_teacher_logits(
        teacher_logits, jnp.asarray(collided, jnp.int32), block=4
    )
    assert float(residual[0, 0]) == 0.0


def test_validate_rejects_unmapped_and_out_of_range_entries():
    with pytest.raises(ValueError, match="unmapped"):
        validate_student_to_teacher(np.asarray([0, 1, -1], np.int32))
    with pytest.raises(ValueError, match="beyond teacher vocab"):
        validate_student_to_teacher(np.asarray([0, 1, 9], np.int32), teacher_vocab=5)
    validate_student_to_teacher(np.asarray([4, 0, 2], np.int32), teacher_vocab=5)


def test_the_real_yx49k_map_is_a_valid_injection():
    mapping = np.load("tokenizers/yx49k/student_to_teacher.npy")
    covered = np.load("tokenizers/yx49k/teacher_covered.npy")
    validate_student_to_teacher(mapping, teacher_vocab=covered.shape[0])
    assert mapping.shape == (49_152,)
    # teacher_covered marks exactly the image of the map.
    assert int(covered.sum()) == mapping.size
    np.testing.assert_array_equal(np.flatnonzero(covered), np.sort(mapping))


# ---------------------------------------------------------------- alignment


def _offsets_for(pieces, text):
    """Builds char-offset pairs for consecutive string pieces."""
    offsets, cursor = [], 0
    for piece in pieces:
        start = text.index(piece, cursor)
        offsets.append((start, start + len(piece)))
        cursor = start + len(piece)
    return offsets


def test_identical_tokenizations_align_one_to_one():
    text = "Hugging Face is awesome!"
    pieces = ["Hugging", " Face", " is", " awesome", "!"]
    offsets = _offsets_for(pieces, text)
    result = align_by_byte_offsets(text, offsets, offsets)
    assert result.one_to_one.all()
    np.testing.assert_array_equal(
        result.teacher_position, np.arange(len(pieces))
    )
    assert result.grouped_positions == 0


def test_merge_groups_are_masked_not_mismapped():
    text = "Hugging Face is awesome!"
    student = ["Hug", "ging", " Face", " is", " awesome", "!"]
    teacher = ["Hugging", " Face", " is", " awesome", "!"]
    result = align_by_byte_offsets(
        text, _offsets_for(student, text), _offsets_for(teacher, text)
    )
    # 'Hug'+'ging' form a 2:1 group: masked, not guessed.
    assert not result.one_to_one[0] and not result.one_to_one[1]
    assert result.teacher_position[0] == -1 and result.teacher_position[1] == -1
    assert result.grouped_positions == 2
    # Every later token realigns 1:1 - no cascading corruption.
    np.testing.assert_array_equal(result.teacher_position[2:], [1, 2, 3, 4])
    assert result.one_to_one[2:].all()


def test_zero_width_special_does_not_stall_the_walker():
    text = "Hello world"
    student = [(0, 0), (0, 5), (5, 11)]        # BOS-style zero-width first
    teacher = [(0, 5), (5, 11)]
    result = align_by_byte_offsets(text, student, teacher)
    # The zero-width token opens a 2:1 group with 'Hello'; 'world' is 1:1.
    assert result.one_to_one[2]
    assert result.teacher_position[2] == 1
    assert not result.one_to_one[0]


def test_multibyte_text_aligns_in_bytes_not_chars():
    text = "café au lait"
    student = ["café", " au", " lait"]
    teacher = ["caf", "é", " au", " lait"]
    result = align_by_byte_offsets(
        text, _offsets_for(student, text), _offsets_for(teacher, text)
    )
    assert not result.one_to_one[0]           # 1:2 group, masked
    np.testing.assert_array_equal(result.teacher_position[1:], [2, 3])
    assert result.one_to_one[1:].all()
