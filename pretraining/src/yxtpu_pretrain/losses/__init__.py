"""Memory-efficient output losses owned by the standalone trainer."""

from yxtpu_pretrain.losses.chunked_linear_cross_entropy import (
    chunked_linear_cross_entropy,
)
from yxtpu_pretrain.losses.linear_cross_entropy import (
    data_parallel_linear_cross_entropy,
    local_linear_cross_entropy_sum,
)

__all__ = [
    "chunked_linear_cross_entropy",
    "data_parallel_linear_cross_entropy",
    "local_linear_cross_entropy_sum",
]
