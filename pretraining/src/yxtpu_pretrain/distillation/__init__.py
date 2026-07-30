"""GOLD on-policy logit distillation: student-aligned teacher supervision.

The yx49k tokenizer was authored as an exact-subset projection of the
Qwen3.5 family vocabulary (every student token maps to a distinct teacher
token, ~98.75% of teacher probability mass covered), which reduces GOLD's
general cross-tokenizer machinery to a gather, a residual bucket, and a
byte-offset 1:1 position mask.
"""

from yxtpu_pretrain.distillation.gold_loss import (
    blockwise_logsumexp,
    gold_position_loss,
    project_teacher_logits,
)
from yxtpu_pretrain.distillation.alignment import align_by_byte_offsets

__all__ = [
    "align_by_byte_offsets",
    "blockwise_logsumexp",
    "gold_position_loss",
    "project_teacher_logits",
]
