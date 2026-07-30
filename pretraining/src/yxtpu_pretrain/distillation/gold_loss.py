"""The GOLD distillation loss, specialized to an exact-subset student vocab.

General GOLD (TRL) must handle arbitrary tokenizer pairs: it splits the
loss into a GKD term over content-matched tokens and a ULD sort-and-pad
fallback for everything else. The yx49k tokenizer removes the fallback's
reason to exist - every student token maps to a distinct teacher token
(``student_to_teacher``), and the unmatched remainder of the teacher
vocabulary carries ~1.25% of its probability mass on the selection corpus.
Here that remainder becomes a single residual bucket instead of a sorted
tail: the teacher's distribution is projected onto (student vocab + 1)
exactly, nothing is approximated, and the generalized JSD is computed on
that shared support.

Teacher logits arrive either dense ``[*, teacher_vocab]`` (on-policy
scoring on device) or pre-projected (off-policy from disk); both feed
``gold_position_loss`` identically.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def blockwise_logsumexp(logits: jax.Array, *, block: int = 32_768) -> jax.Array:
    """logsumexp over the last axis without materializing exp(logits).

    The teacher's 248,320-wide logits are the one tensor that does not fit
    comfortably in fp32; a running (max, sum) pair over vocab blocks keeps
    peak memory at one block.
    """
    vocab = logits.shape[-1]
    if vocab % block:
        pad = block - vocab % block
        logits = jnp.pad(logits, [(0, 0)] * (logits.ndim - 1) + [(0, pad)],
                         constant_values=-jnp.inf)
    blocks = logits.reshape(*logits.shape[:-1], -1, block)

    def step(carry, chunk):
        running_max, running_sum = carry
        chunk_max = jnp.max(chunk, axis=-1)
        new_max = jnp.maximum(running_max, chunk_max)
        # Rescale both partial sums onto the new maximum; -inf blocks (pure
        # padding) contribute exp(-inf)=0 rather than NaN because the new
        # maximum is never -inf once any real block has been seen.
        safe = lambda m: jnp.where(jnp.isfinite(new_max), m - new_max, 0.0)
        running_sum = running_sum * jnp.exp(safe(running_max)) + jnp.sum(
            jnp.exp(chunk - new_max[..., None]), axis=-1
        )
        return (new_max, running_sum), None

    initial = (
        jnp.full(logits.shape[:-1], -jnp.inf, logits.dtype),
        jnp.zeros(logits.shape[:-1], logits.dtype),
    )
    (final_max, final_sum), _ = jax.lax.scan(
        step, initial, jnp.moveaxis(blocks, -2, 0)
    )
    return final_max + jnp.log(final_sum)


def project_teacher_logits(
    teacher_logits: jax.Array,
    student_to_teacher: jax.Array,
    *,
    block: int = 32_768,
) -> tuple[jax.Array, jax.Array]:
    """Projects dense teacher logits onto (student vocab, residual bucket).

    Returns ``(matched_logprobs [*, student_vocab], residual_mass [*])``
    where ``matched_logprobs`` are the teacher's log-probabilities at the
    mapped ids and ``residual_mass`` is the probability the teacher assigns
    outside the student's image - the ULD tail collapsed to one number.
    """
    normalizer = blockwise_logsumexp(
        teacher_logits.astype(jnp.float32), block=block
    )
    matched = jnp.take_along_axis(
        teacher_logits.astype(jnp.float32),
        jnp.broadcast_to(
            student_to_teacher,
            (*teacher_logits.shape[:-1], student_to_teacher.shape[-1]),
        ),
        axis=-1,
    ) - normalizer[..., None]
    residual = -jnp.expm1(
        jax.scipy.special.logsumexp(matched, axis=-1)
    )
    return matched, jnp.clip(residual, 0.0, 1.0)


def gold_position_loss(
    student_logits: jax.Array,
    teacher_matched_logprobs: jax.Array,
    teacher_residual_mass: jax.Array,
    position_mask: jax.Array,
    *,
    beta: float = 0.0,
    renormalize_teacher: bool = True,
) -> tuple[jax.Array, dict]:
    """Masked mean generalized JSD(beta) on the student's support.

    ``beta=0`` is the forward KL D(teacher || student) - the GKD default
    that worked best in both the GKD paper's and HF's ablations for
    student-capacity-limited setups. ``renormalize_teacher`` scales the
    projected teacher distribution by 1/(1-residual) so both sides are
    proper distributions over the same support; the residual mass is
    reported, not trained against, because the student has no token that
    could ever receive it.

    ``position_mask`` zeroes positions the byte-offset walker could not
    align 1:1 (and prompt/padding positions); the loss is averaged over
    surviving positions only.
    """
    student_logprobs = jax.nn.log_softmax(
        student_logits.astype(jnp.float32), axis=-1
    )
    teacher_logprobs = teacher_matched_logprobs
    if renormalize_teacher:
        log_kept = jnp.log1p(
            -jnp.clip(teacher_residual_mass, 0.0, 0.999)
        )
        teacher_logprobs = teacher_logprobs - log_kept[..., None]

    teacher_probs = jnp.exp(teacher_logprobs)
    if beta == 0.0:
        divergence = jnp.sum(
            teacher_probs * (teacher_logprobs - student_logprobs), axis=-1
        )
    elif beta == 1.0:
        student_probs = jnp.exp(student_logprobs)
        divergence = jnp.sum(
            student_probs * (student_logprobs - teacher_logprobs), axis=-1
        )
    else:
        student_probs = jnp.exp(student_logprobs)
        mixture = beta * student_probs + (1.0 - beta) * teacher_probs
        log_mixture = jnp.log(jnp.clip(mixture, 1e-30, None))
        divergence = beta * jnp.sum(
            student_probs * (student_logprobs - log_mixture), axis=-1
        ) + (1.0 - beta) * jnp.sum(
            teacher_probs * (teacher_logprobs - log_mixture), axis=-1
        )

    weights = position_mask.astype(jnp.float32)
    token_count = jnp.maximum(jnp.sum(weights), 1.0)
    loss = jnp.sum(divergence * weights) / token_count
    metrics = {
        "distill_tokens": token_count,
        "teacher_residual_mass": jnp.sum(
            teacher_residual_mass * weights
        ) / token_count,
    }
    return loss, metrics
