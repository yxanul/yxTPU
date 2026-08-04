"""GOLD on-policy logit distillation: student-aligned teacher supervision.

The yx49k tokenizer was authored as an exact-subset projection of the
Qwen3.5 family vocabulary: every student token maps to a distinct teacher
token, and ~98.75% of the teacher's probability mass lands inside that
image. Two consequences, and together they are most of GOLD.

The mapped id sequence decodes to byte-identical text, so the teacher can
score the student's own segmentation and supervision is 1:1 by
construction - no alignment pass, no dropped positions. The uncovered
~1.25% of teacher mass has nowhere to go in the student's vocabulary, so
ULD's sorted tail collapses into a single residual bucket that is reported
rather than trained against.

What is left is a gather, a projection, and a divergence.
"""

from yxtpu_pretrain.distillation.gold_loss import (
    QWEN35_LOGIT_WIDTH,
    QWEN35_VALID_VOCAB,
    blockwise_logsumexp,
    gold_position_loss,
    project_teacher_logits,
)
from yxtpu_pretrain.distillation.alignment import (
    DirectMapReport,
    align_by_byte_offsets,
    direct_teacher_ids,
    validate_student_to_teacher,
    verify_direct_map,
)

from yxtpu_pretrain.distillation.objective import (
    cross_entropy,
    gold_objective,
)

__all__ = [
    "cross_entropy",
    "gold_objective",
    "DirectMapReport",
    "QWEN35_LOGIT_WIDTH",
    "QWEN35_VALID_VOCAB",
    "align_by_byte_offsets",
    "blockwise_logsumexp",
    "direct_teacher_ids",
    "gold_position_loss",
    "project_teacher_logits",
    "validate_student_to_teacher",
    "verify_direct_map",
]
