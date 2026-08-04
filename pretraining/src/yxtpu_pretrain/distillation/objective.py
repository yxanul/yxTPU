"""The GOLD training objective: what the student actually minimizes.

``gold_loss.py`` holds the pieces (projection, divergence). This is the
composite, and the place to check the arithmetic.

-------------------------------------------------------------------------
Position alignment - the thing that is easy to get wrong and impossible to
see afterwards
-------------------------------------------------------------------------

The batch carries ``input_ids`` and ``labels`` already offset by one, the
convention the SFT stage uses::

    input_ids = row[:-1]      labels = row[1:]      loss_mask = mask[1:]

Student logits at position *i* are the distribution over ``labels[i]``,
having read ``input_ids[:i+1]``.

The teacher is fed ``student_to_teacher[input_ids]`` - the *same* prefix,
in the same segmentation, expressed in the teacher's vocabulary. Its
logits at position *i* are therefore also a distribution over the token
that follows that same prefix. **Student position i pairs with teacher
position i, with no further shift.** The only shift in the whole pipeline
is the one already baked into input_ids/labels.

The trap: it is tempting to shift the teacher by one "to line up with the
labels". That is wrong, and it is silent - a shifted teacher still yields
a finite, decreasing loss, because predicting token i from position i+1's
distribution is merely a harder version of the same task. It shows up only
as a student that trains but never gets good.
``test_shifting_the_teacher_by_one_is_strictly_worse`` pins it.

-------------------------------------------------------------------------
The objective
-------------------------------------------------------------------------

    L = distill_weight * D_beta(teacher_proj || student)  +  ce_weight * CE

both averaged over the same masked positions, so the weights are directly
comparable and a 1:1 mix means what it says.

* **D_beta** is the generalized Jensen-Shannon divergence on the shared
  support, from ``gold_position_loss``. At ``beta=0`` it is the forward KL
  ``D(teacher || student)`` - the GKD default, and the one both the GKD
  paper and HF's ablations favour when the student is capacity-limited,
  because it is mass-covering: the student is penalized for putting no
  probability where the teacher puts some.

* **CE** is the ordinary next-token loss against the hard ``labels``. At
  ``ce_weight=0`` the student never sees the sampled token directly and
  learns only the teacher's distribution, which is the pure-distillation
  setting. TRL exposes the same knob as ``uld_crossentropy_weight``,
  defaulted to 0.

Only the student is differentiated. The teacher arrives as constants -
already projected onto the student's vocabulary and its residual measured
- so there is nothing to stop a gradient from reaching, by construction
rather than by ``stop_gradient``.

-------------------------------------------------------------------------
Reading the metrics
-------------------------------------------------------------------------

``teacher_residual_mass`` is the probability the teacher puts outside the
student's vocabulary; it is renormalized out of the target and reported,
never trained against, because no student token could receive it. On the
selection corpus it averaged ~1.25%. A sharp rise means the batch has
drifted somewhere the tokenizer does not cover well.

``teacher_top1_is_label`` is how often the teacher's own argmax is the
token that was actually sampled. It is a data-health signal, not a model
one: at temperature 0.7 with top_k 20 it should be high, and a collapse
means the batch is not what the teacher generated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from yxtpu_pretrain.distillation.gold_loss import (
    gold_position_loss,
    gold_topk_position_loss,
)


def cross_entropy(student_logits, labels, loss_mask):
    """Masked mean next-token CE. Shares the denominator with the divergence."""
    logprobs = jax.nn.log_softmax(student_logits.astype(jnp.float32), axis=-1)
    picked = jnp.take_along_axis(logprobs, labels[..., None], axis=-1)[..., 0]
    weights = loss_mask.astype(jnp.float32)
    tokens = jnp.maximum(weights.sum(), 1.0)
    return -(picked * weights).sum() / tokens, tokens


def gold_objective(
    student_logits,
    labels,
    loss_mask,
    *,
    teacher_matched_logprobs=None,
    teacher_residual_mass=None,
    beta: float = 0.0,
    distill_weight: float = 1.0,
    ce_weight: float = 0.0,
):
    """Returns ``(loss, metrics)`` for one batch.

    ``student_logits`` is ``[B, T, V_student]`` and predicts ``labels``
    ``[B, T]``; ``teacher_matched_logprobs`` is the teacher's distribution
    already projected onto the same ``V_student`` at the same positions.
    Pass the teacher as ``None`` to get plain SFT, which is what makes this
    a drop-in for the existing stage.
    """
    ce, tokens = cross_entropy(student_logits, labels, loss_mask)
    metrics = {
        "ce": ce,
        "tokens": tokens,
        "student_top1_is_label": _agreement(student_logits, labels, loss_mask),
    }

    if teacher_matched_logprobs is None:
        metrics["loss"] = ce
        return ce, metrics

    distill, distill_metrics = gold_position_loss(
        student_logits,
        teacher_matched_logprobs,
        teacher_residual_mass,
        loss_mask,
        beta=beta,
    )
    loss = distill_weight * distill + ce_weight * ce
    metrics.update(distill_metrics)
    metrics["distill"] = distill
    metrics["loss"] = loss
    metrics["teacher_top1_is_label"] = _agreement(
        teacher_matched_logprobs, labels, loss_mask)
    return loss, metrics


def _agreement(logits, labels, loss_mask):
    """Masked fraction of positions whose argmax is the label."""
    weights = loss_mask.astype(jnp.float32)
    hit = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    return (hit * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def gold_topk_objective(
    student_logits,
    labels,
    loss_mask,
    *,
    teacher_topk_ids,
    teacher_topk_logprobs,
    teacher_rest_mass,
    beta: float = 0.0,
    distill_weight: float = 1.0,
    ce_weight: float = 0.0,
):
    """``gold_objective`` against a top-K-compressed teacher.

    Same composition, same shared denominator; the teacher arrives as the
    precomputed ``(ids, logprobs, rest)`` triple instead of a full
    projected distribution. ``teacher_top1_is_label`` reads the first
    top-K column - top_k returns descending, so column 0 is the argmax.
    """
    ce, tokens = cross_entropy(student_logits, labels, loss_mask)
    distill, distill_metrics = gold_topk_position_loss(
        student_logits,
        teacher_topk_ids,
        teacher_topk_logprobs,
        teacher_rest_mass,
        loss_mask,
        beta=beta,
    )
    loss = distill_weight * distill + ce_weight * ce
    weights = loss_mask.astype(jnp.float32)
    top1_hit = (teacher_topk_ids[..., 0] == labels).astype(jnp.float32)
    metrics = {
        "ce": ce,
        "tokens": tokens,
        "distill": distill,
        "loss": loss,
        "student_top1_is_label": _agreement(student_logits, labels, loss_mask),
        "teacher_top1_is_label": (top1_hit * weights).sum()
        / jnp.maximum(weights.sum(), 1.0),
        **distill_metrics,
    }
    return loss, metrics


def make_gold_model_loss(
    *,
    beta: float = 0.0,
    distill_weight: float = 1.0,
    ce_weight: float = 0.0,
):
    """Builds the model-level loss the train step differentiates.

    Drop-in for ``train._loss`` via ``_make_train_step(config, loss_fn=...)``:
    same ``(model, batch, record_max_logits)`` signature, same auxiliary
    contract (``max_logits`` for muonclip, ``tokens`` for throughput), plus
    the distillation metrics. The batch must carry the precomputed teacher
    triple the Mephisto iterator attaches when given a target store.

    The full [B, T, V] student logits are materialized here, as the
    standard loss head already does at this vocabulary; the top-K gather
    keeps everything else K-wide.
    """
    from yxtpu_pretrain.model import attention_logit_intermediates

    def gold_loss(model, batch, *, record_max_logits):
        hidden_states = model.hidden_states(
            batch["input_ids"],
            decoder_segment_ids=batch["segment_ids"],
            decoder_positions=batch["positions"],
            record_max_logits=record_max_logits,
        )
        logits = model.project_logits(hidden_states)
        loss, metrics = gold_topk_objective(
            logits,
            batch["labels"],
            batch["loss_mask"],
            teacher_topk_ids=batch["teacher_topk_ids"],
            teacher_topk_logprobs=batch["teacher_topk_logprobs"],
            teacher_rest_mass=batch["teacher_rest_mass"],
            beta=beta,
            distill_weight=distill_weight,
            ce_weight=ce_weight,
        )
        logits_max = (
            attention_logit_intermediates(model)
            if record_max_logits
            else jnp.zeros(
                (
                    model.config.model.num_cycles,
                    1,
                    model.config.model.attention.num_query_heads,
                ),
                dtype=jnp.float32,
            )
        )
        auxiliary = {
            "max_logits": logits_max,
            "tokens": metrics["tokens"],
            "ce": metrics["ce"],
            "distill": metrics["distill"],
            "teacher_rest_mass": metrics["teacher_rest_mass"],
            "teacher_top1_is_label": metrics["teacher_top1_is_label"],
            "student_top1_is_label": metrics["student_top1_is_label"],
        }
        return loss, auxiliary

    return gold_loss
