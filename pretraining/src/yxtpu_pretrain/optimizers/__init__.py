"""Fail-closed optimizer routing and update policies."""

from yxtpu_pretrain.optimizers.muonclip import (
    MuonClipTelemetry,
    apply_gqa_muonclip,
)
from yxtpu_pretrain.optimizers.routing import (
    Route,
    build_learning_rate_schedule,
    build_optimizer,
    classify_parameters,
)

__all__ = [
    "MuonClipTelemetry",
    "Route",
    "apply_gqa_muonclip",
    "build_learning_rate_schedule",
    "build_optimizer",
    "classify_parameters",
]
