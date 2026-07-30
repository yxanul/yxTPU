"""Byte-offset sequence alignment between student and teacher tokenizations.

Host-side preprocessing for on-policy scoring: the student's rollout text
is re-tokenized by the teacher, and each side's tokens are walked in UTF-8
byte coordinates (the fix GOLD adopted after TRL issue #4393 - decoded-text
buffer comparison breaks on zero-width and asymmetric specials). A group
closes whenever both sides' byte ends coincide.

At yx49k's measured fertility (1.011 vs the teacher, 98.8% of positions
1:1) nearly every group is a single student token against a single teacher
token; v1 trains on exactly those and masks the rest, costing ~1% of
positions instead of the product-rule merge machinery.
"""

from __future__ import annotations

import dataclasses

import numpy as np


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
