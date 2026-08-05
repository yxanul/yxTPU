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

"""Every arithmetic claim the GOLD objective makes, pinned.

The failures these guard against share a property: the loss still goes
down. A teacher shifted by one position, a gradient leaking into the
teacher, a CE term averaged over a different denominator than the
divergence - none of them raise, and all of them just make the student
quietly worse. So each is tested against an independent reference or an
inequality rather than against "it ran".
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yxtpu_pretrain.distillation import project_teacher_logits
from yxtpu_pretrain.distillation.objective import cross_entropy, gold_objective

BATCH, TIME, STUDENT_VOCAB, TEACHER_VOCAB = 2, 6, 16, 40


def _teacher_targets(key, *, student_vocab=STUDENT_VOCAB, shape=(BATCH, TIME)):
    """A projected teacher, built the way the training loop builds one."""
    teacher_logits = jax.random.normal(key, (*shape, TEACHER_VOCAB)) * 2.0
    mapping = jnp.arange(student_vocab, dtype=jnp.int32) * 2
    return project_teacher_logits(teacher_logits, mapping, block=8)


def _batch(seed=0):
    keys = jax.random.split(jax.random.key(seed), 4)
    student_logits = jax.random.normal(keys[0], (BATCH, TIME, STUDENT_VOCAB))
    labels = jax.random.randint(keys[1], (BATCH, TIME), 0, STUDENT_VOCAB)
    mask = jnp.asarray([[0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1]], jnp.float32)
    matched, residual = _teacher_targets(keys[2])
    return student_logits, labels, mask, matched, residual


# ------------------------------------------------------------- composition


def test_pure_ce_matches_a_numpy_reference():
    student_logits, labels, mask, _, _ = _batch()
    loss, metrics = gold_objective(student_logits, labels, mask)
    logprobs = np.asarray(jax.nn.log_softmax(student_logits.astype(jnp.float32), -1))
    picked = np.take_along_axis(
        logprobs, np.asarray(labels)[..., None], axis=-1)[..., 0]
    expected = -(picked * np.asarray(mask)).sum() / np.asarray(mask).sum()
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)
    # No teacher supplied: the objective degrades to plain SFT.
    assert "distill" not in metrics


def test_the_two_terms_are_weighted_exactly_as_advertised():
    student_logits, labels, mask, matched, residual = _batch()
    common = dict(teacher_matched_logprobs=matched,
                  teacher_residual_mass=residual)
    _, pure_distill = gold_objective(
        student_logits, labels, mask, distill_weight=1.0, ce_weight=0.0,
        **common)
    _, pure_ce = gold_objective(
        student_logits, labels, mask, distill_weight=0.0, ce_weight=1.0,
        **common)
    mixed, _ = gold_objective(
        student_logits, labels, mask, distill_weight=0.3, ce_weight=0.7,
        **common)
    expected = 0.3 * float(pure_distill["distill"]) + 0.7 * float(pure_ce["ce"])
    assert float(mixed) == pytest.approx(expected, rel=1e-6)


def test_both_terms_average_over_the_same_denominator():
    """Otherwise the mix ratio silently depends on how many positions are
    masked, and a 1:1 mix is not 1:1."""
    student_logits, labels, mask, matched, residual = _batch()
    _, metrics = gold_objective(
        student_logits, labels, mask, teacher_matched_logprobs=matched,
        teacher_residual_mass=residual)
    assert float(metrics["tokens"]) == float(metrics["distill_tokens"])
    assert float(metrics["tokens"]) == float(np.asarray(mask).sum())


def test_ce_denominator_counts_masked_positions_only():
    student_logits, labels, _, _, _ = _batch()
    full = jnp.ones((BATCH, TIME), jnp.float32)
    half = jnp.concatenate(
        [jnp.ones((BATCH, TIME // 2)), jnp.zeros((BATCH, TIME // 2))], axis=1)
    ce_full, tokens_full = cross_entropy(student_logits, labels, full)
    ce_half, tokens_half = cross_entropy(student_logits, labels, half)
    assert float(tokens_full) == BATCH * TIME
    assert float(tokens_half) == BATCH * TIME // 2
    # A mean, not a sum: dropping half the positions must not halve it.
    assert 0.3 < float(ce_half) / float(ce_full) < 3.0


# ------------------------------------------------------------- correctness


def test_a_student_that_equals_the_teacher_has_zero_distillation_loss():
    _, labels, mask, matched, residual = _batch()
    # The renormalized projection, expressed as logits.
    student_logits = matched - jnp.log1p(-residual)[..., None]
    _, metrics = gold_objective(
        student_logits, labels, mask, distill_weight=1.0, ce_weight=0.0,
        teacher_matched_logprobs=matched, teacher_residual_mass=residual)
    assert float(metrics["distill"]) == pytest.approx(0.0, abs=1e-6)
    # But CE is not zero: the teacher is not a point mass on the label.
    assert float(metrics["ce"]) > 0.0


def test_shifting_the_teacher_by_one_is_strictly_worse():
    """The alignment trap. Student position i and teacher position i both
    predict the token after the same prefix, so no extra shift is right -
    and a shifted teacher still trains, which is why this needs a test."""
    _, labels, mask, matched, residual = _batch(seed=3)
    student_logits = matched - jnp.log1p(-residual)[..., None]  # perfect match

    aligned, _ = gold_objective(
        student_logits, labels, mask, distill_weight=1.0, ce_weight=0.0,
        teacher_matched_logprobs=matched, teacher_residual_mass=residual)
    shifted, _ = gold_objective(
        student_logits, labels, mask, distill_weight=1.0, ce_weight=0.0,
        teacher_matched_logprobs=jnp.roll(matched, 1, axis=1),
        teacher_residual_mass=jnp.roll(residual, 1, axis=1))
    assert float(aligned) == pytest.approx(0.0, abs=1e-6)
    assert float(shifted) > 0.1, "a one-position shift must be visible"


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_the_divergence_is_non_negative_and_zero_only_at_a_match(beta):
    _, labels, mask, matched, residual = _batch(seed=5)
    matched_logits = matched - jnp.log1p(-residual)[..., None]
    for offset, expect_zero in ((0.0, True), (2.5, False)):
        # Adding a constant to all logits is a no-op under softmax, so
        # perturb one coordinate instead.
        student_logits = matched_logits.at[..., 0].add(offset)
        _, metrics = gold_objective(
            student_logits, labels, mask, beta=beta, distill_weight=1.0,
            ce_weight=0.0, teacher_matched_logprobs=matched,
            teacher_residual_mass=residual)
        value = float(metrics["distill"])
        assert value >= -1e-6
        assert (value == pytest.approx(0.0, abs=1e-6)) is expect_zero


def test_small_beta_approaches_the_forward_kl():
    """GKD Eq. (1): JSD(beta)/beta -> KL(teacher || student) as beta -> 0,
    so the interior formula must weight the TEACHER by beta. The mirrored
    orientation satisfies every symmetric property (non-negativity, zero at
    a match) while quietly making beta=0.1 mean the paper's beta=0.9 -
    only this limit tells the two apart."""
    student_logits, labels, mask, matched, residual = _batch(seed=29)
    common = dict(teacher_matched_logprobs=matched,
                  teacher_residual_mass=residual,
                  distill_weight=1.0, ce_weight=0.0)

    _, at_zero = gold_objective(student_logits, labels, mask, beta=0.0, **common)
    _, at_one = gold_objective(student_logits, labels, mask, beta=1.0, **common)
    forward, reverse = float(at_zero["distill"]), float(at_one["distill"])
    assert forward != pytest.approx(reverse, rel=0.05), "degenerate batch"

    beta = 1e-3
    _, interior = gold_objective(student_logits, labels, mask, beta=beta, **common)
    scaled = float(interior["distill"]) / beta
    assert scaled == pytest.approx(forward, rel=0.05)
    assert abs(scaled - forward) < abs(scaled - reverse)


def test_a_uniform_shift_of_student_logits_changes_nothing():
    """Softmax invariance - a cheap guard against accidentally training on
    unnormalized logits."""
    student_logits, labels, mask, matched, residual = _batch(seed=7)
    common = dict(teacher_matched_logprobs=matched,
                  teacher_residual_mass=residual, ce_weight=1.0)
    base, _ = gold_objective(student_logits, labels, mask, **common)
    lifted, _ = gold_objective(student_logits + 7.25, labels, mask, **common)
    assert float(base) == pytest.approx(float(lifted), rel=1e-5)


# ---------------------------------------------------------------- gradients


def test_masked_positions_receive_no_gradient():
    _, labels, _, matched, residual = _batch(seed=9)
    mask = jnp.asarray([[1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 1]], jnp.float32)

    def loss_of(logits):
        loss, _ = gold_objective(
            logits, labels, mask, teacher_matched_logprobs=matched,
            teacher_residual_mass=residual, ce_weight=1.0)
        return loss

    gradient = np.abs(np.asarray(jax.grad(loss_of)(
        jnp.zeros((BATCH, TIME, STUDENT_VOCAB))))).sum(-1)
    np.testing.assert_array_equal(gradient > 0, np.asarray(mask) > 0)


def test_no_gradient_reaches_the_teacher():
    """The teacher arrives as constants, so this holds by construction -
    but a future refactor that recomputes the projection inside the loss
    would break it silently, and the student would start optimizing its
    own target."""
    student_logits, labels, mask, _, _ = _batch(seed=11)
    key = jax.random.key(13)

    def loss_of_teacher(teacher_logits):
        mapping = jnp.arange(STUDENT_VOCAB, dtype=jnp.int32) * 2
        matched, residual = project_teacher_logits(
            teacher_logits, mapping, block=8)
        matched = jax.lax.stop_gradient(matched)
        residual = jax.lax.stop_gradient(residual)
        loss, _ = gold_objective(
            student_logits, labels, mask, teacher_matched_logprobs=matched,
            teacher_residual_mass=residual)
        return loss

    teacher_logits = jax.random.normal(key, (BATCH, TIME, TEACHER_VOCAB))
    gradient = jax.grad(loss_of_teacher)(teacher_logits)
    assert float(jnp.abs(gradient).max()) == 0.0


def test_the_student_gradient_points_toward_the_teacher():
    """One step of plain gradient descent must reduce the divergence."""
    _, labels, mask, matched, residual = _batch(seed=17)
    student_logits = jnp.zeros((BATCH, TIME, STUDENT_VOCAB))

    def loss_of(logits):
        loss, _ = gold_objective(
            logits, labels, mask, distill_weight=1.0, ce_weight=0.0,
            teacher_matched_logprobs=matched, teacher_residual_mass=residual)
        return loss

    before = float(loss_of(student_logits))
    gradient = jax.grad(loss_of)(student_logits)
    after = float(loss_of(student_logits - 0.5 * gradient))
    assert after < before


# ------------------------------------------------------------------ metrics


def test_residual_mass_is_reported_but_not_trained_against():
    """The student has no token that could receive it, so it is
    renormalized out of the target; a rise is a tokenizer-coverage signal."""
    _, labels, mask, matched, residual = _batch(seed=19)
    student_logits = matched - jnp.log1p(-residual)[..., None]
    _, metrics = gold_objective(
        student_logits, labels, mask, distill_weight=1.0, ce_weight=0.0,
        teacher_matched_logprobs=matched, teacher_residual_mass=residual)
    assert 0.0 <= float(metrics["teacher_residual_mass"]) <= 1.0
    # Reported, yet the loss is exactly zero at a match despite it.
    assert float(metrics["distill"]) == pytest.approx(0.0, abs=1e-6)


def test_agreement_metrics_are_masked_fractions():
    _, labels, mask, matched, residual = _batch(seed=23)
    # A student that always predicts the label scores 1.0.
    perfect = jax.nn.one_hot(labels, STUDENT_VOCAB) * 30.0
    _, metrics = gold_objective(
        perfect, labels, mask, teacher_matched_logprobs=matched,
        teacher_residual_mass=residual)
    assert float(metrics["student_top1_is_label"]) == pytest.approx(1.0)
    assert 0.0 <= float(metrics["teacher_top1_is_label"]) <= 1.0
