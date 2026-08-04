"""Getting teacher supervision onto student positions. Two strategies.

**Direct map** (``direct_teacher_ids``) is the one to reach for. Because
yx49k was built as an exact-subset projection of the Qwen3.5 vocabulary,
``student_to_teacher`` is injective and ``teacher.decode(map[student_ids])``
reproduces the source text byte-for-byte - measured 441/441 on rendered
Mephisto chat and ClimbMix, covering CJK, emoji, zero-width spaces and the
chat specials. So the teacher can score the student's *own* segmentation:
position i supervises position i by construction, every position is 1:1,
and no host-side alignment runs in the training loop at all.

Its cost is that the teacher sometimes scores a segmentation its own BPE
would never produce. Measured against native Qwen3.5 tokenization, 98.1%
of student token boundaries coincide on ClimbMix and 95.3% on rendered
chat, so 2-5% of positions are locally non-canonical. That is a supervision
*quality* question, not a correctness one, and it is recoverable: those
positions are detectable offline, so they can be masked to fall back to
exactly the coverage the walker would have given.

**Byte-offset walk** (``align_by_byte_offsets``) is the general fallback,
and the right choice off-policy where the text arrives already tokenized
natively. The teacher re-tokenizes the text and both sides are walked in
UTF-8 byte coordinates (the fix GOLD adopted after TRL issue #4393 -
decoded-text buffer comparison breaks on zero-width and asymmetric
specials). A group closes whenever both sides' byte ends coincide; 1:1
groups train and the rest are masked, costing the ~1-2% of positions that
the direct map keeps.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class DirectMapReport:
    """Evidence that the direct map is safe on a given corpus."""

    texts: int
    student_tokens: int
    teacher_tokens: int
    roundtrip_exact: int
    canonical_positions: int
    mismatches: list[tuple[str, str]]

    @property
    def roundtrip_rate(self) -> float:
        """Fraction of texts the mapped ids decode back to exactly."""
        return self.roundtrip_exact / max(self.texts, 1)

    @property
    def canonical_rate(self) -> float:
        """Fraction of student boundaries the teacher would also place."""
        return self.canonical_positions / max(self.student_tokens, 1)

    @property
    def fertility(self) -> float:
        return self.student_tokens / max(self.teacher_tokens, 1)


def validate_student_to_teacher(student_to_teacher, teacher_vocab=None) -> None:
    """Raises unless the map is a well-formed injection into the teacher.

    Injectivity is what makes the projection sound: two student ids sharing
    a teacher id would count that token's probability twice in the matched
    sum, and ``project_teacher_logits`` would return a residual below zero -
    which it silently clips to zero. Checking here means the failure is
    loud and happens once, on the host, instead of quietly biasing a run.
    """
    mapping = np.asarray(student_to_teacher)
    if mapping.ndim != 1:
        raise ValueError(f"expected a 1-D map, got shape {mapping.shape}")
    if mapping.min() < 0:
        raise ValueError(f"{int((mapping < 0).sum())} student ids are unmapped")
    if teacher_vocab is not None and mapping.max() >= teacher_vocab:
        raise ValueError(
            f"map reaches id {int(mapping.max())} beyond teacher vocab "
            f"{teacher_vocab}"
        )
    distinct = np.unique(mapping).size
    if distinct != mapping.size:
        raise ValueError(
            f"map is not injective: {mapping.size - distinct} student ids "
            "collide onto an already-claimed teacher id"
        )


def direct_teacher_ids(student_ids, student_to_teacher):
    """Teacher ids for the student's own segmentation - the whole strategy.

    A gather, deliberately. The work that makes it valid lives in the
    tokenizer's construction and in ``verify_direct_map``; at training time
    there is nothing left to compute, which is the point. Position i of the
    result is the teacher's token for position i of the student's, so the
    teacher's logits need no realignment before ``project_teacher_logits``.
    """
    return jnp.take(jnp.asarray(student_to_teacher), student_ids, axis=0)


def verify_direct_map(
    student_tokenizer,
    teacher_tokenizer,
    student_to_teacher,
    texts,
    *,
    keep_mismatches: int = 3,
) -> DirectMapReport:
    """Checks the round-trip property that licenses skipping the walker.

    For each text: encode with the student, map the ids, decode with the
    teacher, and require the result to equal the input. Also records how
    often the student's segmentation is the one the teacher's own BPE would
    have produced, which is the quality caveat rather than a failure.
    """
    mapping = np.asarray(student_to_teacher)
    report = DirectMapReport(0, 0, 0, 0, 0, [])
    for text in texts:
        if not text or not text.strip():
            continue
        student_ids = student_tokenizer.encode(text, add_special_tokens=False)
        if not student_ids:
            continue
        teacher_ids = mapping[np.asarray(student_ids)].tolist()
        report.texts += 1
        if teacher_tokenizer.decode(teacher_ids) == text:
            report.roundtrip_exact += 1
        elif len(report.mismatches) < keep_mismatches:
            report.mismatches.append(
                (text[:120], teacher_tokenizer.decode(teacher_ids)[:120])
            )
        native = teacher_tokenizer.encode(text, add_special_tokens=False)
        report.student_tokens += len(student_ids)
        report.teacher_tokens += len(native)
        if teacher_ids == native:
            report.canonical_positions += len(student_ids)
        else:
            report.canonical_positions += len(
                _byte_ends_of_ids(teacher_ids, teacher_tokenizer)
                & _byte_ends_of_ids(native, teacher_tokenizer)
            )
    return report


def _byte_ends_of_ids(ids, tokenizer) -> set[int]:
    """Cumulative UTF-8 byte offsets at which each token ends."""
    ends, cursor = set(), 0
    for token_id in ids:
        cursor += len(tokenizer.decode([token_id]).encode("utf-8"))
        ends.add(cursor)
    return ends


@dataclasses.dataclass
class AlignmentResult:
    """Per-student-position teacher indices and the 1:1 trainable mask."""

    teacher_position: np.ndarray  # int32 [student_len], -1 where unaligned
    one_to_one: np.ndarray        # bool  [student_len]
    groups: int                   # aligned byte-groups found
    grouped_positions: int        # student positions inside non-1:1 groups


def _byte_ends(text: str, offsets) -> list[int]:
    """Token end positions in UTF-8 bytes, from fast-tokenizer char offsets.

    Zero-width tokens (specials) keep their end at the running position, so
    they never block a group from closing - the walker simply advances
    through them, which is exactly how GOLD's update resolves asymmetric
    BOS/EOS conventions.
    """
    # Cumulative byte length at each character boundary, computed once.
    cumulative = np.zeros(len(text) + 1, dtype=np.int64)
    for index, char in enumerate(text):
        cumulative[index + 1] = cumulative[index] + len(char.encode("utf-8"))
    return [int(cumulative[end]) for _, end in offsets]


def align_by_byte_offsets(
    text: str,
    student_offsets,
    teacher_offsets,
) -> AlignmentResult:
    """Aligns two tokenizations of ``text`` by walking byte end positions.

    ``*_offsets`` are the fast tokenizers' ``offset_mapping`` for the same
    string (no specials added). Returns, for every student position, the
    teacher position whose token covers the same content when the group is
    1:1, and -1 otherwise.
    """
    student_ends = _byte_ends(text, student_offsets)
    teacher_ends = _byte_ends(text, teacher_offsets)
    student_len, teacher_len = len(student_ends), len(teacher_ends)
    teacher_position = np.full(student_len, -1, dtype=np.int32)
    one_to_one = np.zeros(student_len, dtype=bool)

    s = t = 0
    group_start_s, group_start_t = 0, 0
    groups = grouped = 0
    while s < student_len and t < teacher_len:
        s_end, t_end = student_ends[s], teacher_ends[t]
        if s_end == t_end:
            groups += 1
            span_s = s - group_start_s + 1
            span_t = t - group_start_t + 1
            if span_s == 1 and span_t == 1:
                teacher_position[s] = t
                one_to_one[s] = True
            else:
                grouped += span_s
            s += 1
            t += 1
            group_start_s, group_start_t = s, t
        elif s_end < t_end:
            s += 1
        else:
            t += 1
    return AlignmentResult(
        teacher_position=teacher_position,
        one_to_one=one_to_one,
        groups=groups,
        grouped_positions=grouped,
    )
