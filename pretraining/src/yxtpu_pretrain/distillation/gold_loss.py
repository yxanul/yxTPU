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

# Qwen3.5 declares ``vocab_size: 248320`` while its tokenizer defines only
# 248,077 tokens: the family pads the embedding for sharding, and because
# the rows are real initialized parameters (not zeros) they emit real, if
# small, logits. Summing them into the partition function would deflate
# every matched log-probability and inflate the reported residual, so the
# boundary is passed explicitly rather than inferred from the logit width.
# Prefer deriving it as ``teacher_covered.shape[0]``; these are the values
# that artifact carries today, kept here so a mismatch fails loudly.
QWEN35_LOGIT_WIDTH = 248_320
QWEN35_VALID_VOCAB = 248_077


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
    valid_vocab: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Projects dense teacher logits onto (student vocab, residual bucket).

    Returns ``(matched_logprobs [*, student_vocab], residual_mass [*])``
    where ``matched_logprobs`` are the teacher's log-probabilities at the
    mapped ids and ``residual_mass`` is the probability the teacher assigns
    outside the student's image - the ULD tail collapsed to one number.

    ``valid_vocab`` truncates the logit row to the tokenizer's real width
    before normalizing, excluding the vocab padding the teacher carries for
    sharding (see ``QWEN35_VALID_VOCAB``). Leaving it ``None`` normalizes
    over everything the teacher emits, which is only correct when the head
    has no padding - true of the synthetic vocabularies in the tests, not
    of Qwen3.5.
    """
    if valid_vocab is not None:
        teacher_logits = jax.lax.slice_in_dim(
            teacher_logits, 0, valid_vocab, axis=-1
        )
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
        # GKD paper Eq. (1) orientation, as TRL implements it: beta weights
        # the TEACHER in both the mixture and the KL sum, so beta -> 0
        # approaches beta * KL(teacher || student) - scaled forward KL,
        # continuous in direction with the beta=0 special case above. The
        # first version of this branch had the roles mirrored, which made
        # beta=0.1 behave like the paper's beta=0.9;
        # test_small_beta_approaches_the_forward_kl now pins the direction.
        student_probs = jnp.exp(student_logprobs)
        mixture = beta * teacher_probs + (1.0 - beta) * student_probs
        log_mixture = jnp.log(jnp.clip(mixture, 1e-30, None))
        divergence = beta * jnp.sum(
            teacher_probs * (teacher_logprobs - log_mixture), axis=-1
        ) + (1.0 - beta) * jnp.sum(
            student_probs * (student_logprobs - log_mixture), axis=-1
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


def topk_teacher_targets(
    matched_logprobs: jax.Array,
    residual_mass: jax.Array,
    k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compresses a projected teacher distribution to its top-K + one tail.

    Returns ``(ids [*, k], logprobs [*, k], rest_mass [*])`` where ``rest``
    is everything the K entries do not carry: the matched tail beyond K
    plus the unmatched residual. Computed on device so only K columns ever
    cross to the host - the point of precomputing targets is to not store
    49,152 floats per position.

    At ``k = student_vocab`` the compression is exact: ``rest`` equals the
    residual and ``gold_topk_position_loss`` reproduces
    ``gold_position_loss(beta=0)`` bit-for-bit, which the tests pin.
    """
    del residual_mass  # implied by what the top-K entries do not carry
    top_logprobs, top_ids = jax.lax.top_k(matched_logprobs, k)
    kept = jnp.exp(jax.scipy.special.logsumexp(top_logprobs, axis=-1))
    rest = jnp.clip(1.0 - kept, 0.0, 1.0)
    return top_ids, top_logprobs, rest


def gold_topk_position_loss(
    student_logits: jax.Array,
    teacher_topk_ids: jax.Array,
    teacher_topk_logprobs: jax.Array,
    teacher_rest_mass: jax.Array,
    position_mask: jax.Array,
    *,
    beta: float = 0.0,
) -> tuple[jax.Array, dict]:
    """The GOLD divergence against a top-K-compressed teacher.

    The teacher's K entries are renormalized to a proper distribution over
    the K set (mirroring ``renormalize_teacher`` in the full loss, with the
    tail playing the residual's role). The student side differs by beta:

    * ``beta=0`` (forward KL) uses the student's RAW log-probabilities at
      the K ids - not renormalized over K. The sum then equals
      ``KL(teacher_K || student_K) - log(student mass on K)``: the exact
      truncated forward KL plus a coverage term that pushes the student's
      mass INTO the teacher's top-K set. Non-negative, zero exactly when
      the student matches the renormalized teacher on K and carries no
      mass outside it.
    * ``beta>0`` needs a student distribution on the same support, so the
      student is renormalized over the K set and the generalized JSD is
      computed there (paper orientation, as in ``gold_position_loss``).
      The coverage pressure is then absent - mode-seeking betas do not
      want it.
    """
    student_logprobs = jax.nn.log_softmax(
        student_logits.astype(jnp.float32), axis=-1
    )
    picked = jnp.take_along_axis(
        student_logprobs, teacher_topk_ids, axis=-1
    )
    log_kept = jnp.log1p(-jnp.clip(teacher_rest_mass, 0.0, 0.999))
    teacher_logprobs = teacher_topk_logprobs - log_kept[..., None]
    teacher_probs = jnp.exp(teacher_logprobs)

    if beta == 0.0:
        divergence = jnp.sum(
            teacher_probs * (teacher_logprobs - picked), axis=-1
        )
    else:
        student_restricted = jax.nn.log_softmax(picked, axis=-1)
        student_probs = jnp.exp(student_restricted)
        if beta == 1.0:
            divergence = jnp.sum(
                student_probs * (student_restricted - teacher_logprobs),
                axis=-1,
            )
        else:
            mixture = beta * teacher_probs + (1.0 - beta) * student_probs
            log_mixture = jnp.log(jnp.clip(mixture, 1e-30, None))
            divergence = beta * jnp.sum(
                teacher_probs * (teacher_logprobs - log_mixture), axis=-1
            ) + (1.0 - beta) * jnp.sum(
                student_probs * (student_restricted - log_mixture), axis=-1
            )

    weights = position_mask.astype(jnp.float32)
    token_count = jnp.maximum(jnp.sum(weights), 1.0)
    loss = jnp.sum(divergence * weights) / token_count
    metrics = {
        "distill_tokens": token_count,
        "teacher_rest_mass": jnp.sum(
            teacher_rest_mass * weights
        ) / token_count,
    }
    return loss, metrics
