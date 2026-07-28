"""Semantic parameter roles used by fail-closed optimizer routing."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from flax import nnx


class ParamRole(StrEnum):
    EMBEDDING = "embedding"
    LOGITS = "logits"
    NORM_SCALE = "norm_scale"
    BIAS = "bias"
    DEPTHWISE_CONV = "depthwise_conv"
    KDA_SCALAR = "kda_scalar"
    KDA_MATRIX = "kda_matrix"
    GQA_QKV = "gqa_qkv"
    GQA_OUTPUT = "gqa_output"
    MLP_INPUT = "mlp_input"
    MLP_OUTPUT = "mlp_output"
    ATTNRES_PSEUDOQUERY = "attnres_pseudoquery"


MUON_ROLES = frozenset(
    {
        ParamRole.KDA_MATRIX,
        ParamRole.GQA_QKV,
        ParamRole.GQA_OUTPUT,
        ParamRole.MLP_INPUT,
        ParamRole.MLP_OUTPUT,
    }
)

ADAMW_ROLES = frozenset(
    {
        ParamRole.EMBEDDING,
        ParamRole.LOGITS,
        ParamRole.NORM_SCALE,
        ParamRole.BIAS,
        ParamRole.DEPTHWISE_CONV,
        ParamRole.KDA_SCALAR,
        ParamRole.ATTNRES_PSEUDOQUERY,
    }
)


def declare_parameter(
    parameter: nnx.Param,
    role: ParamRole,
    *,
    matrix_in_axes: Iterable[int] = (),
    matrix_out_axes: Iterable[int] = (),
    matrix_head_in_axes: Iterable[int] | None = None,
    matrix_head_out_axes: Iterable[int] | None = None,
) -> nnx.Param:
    """Returns a parameter with optimizer semantics attached as NNX metadata.

    ``matrix_head_in_axes``/``matrix_head_out_axes`` declare an optional
    alternate Muon matricization used when the optimizer enables Per-Head
    Muon (Kimi K3 §2.5): axes absent from both groups become Muon batch axes,
    so Newton-Schulz runs one block per head instead of on the joint matrix.
    ``None`` means the parameter has no per-head alternative and keeps its
    standard matricization under either setting.
    """
    return parameter.replace(
        role=str(role),
        matrix_in_axes=tuple(matrix_in_axes),
        matrix_out_axes=tuple(matrix_out_axes),
        matrix_head_in_axes=(
            None if matrix_head_in_axes is None else tuple(matrix_head_in_axes)
        ),
        matrix_head_out_axes=(
            None if matrix_head_out_axes is None else tuple(matrix_head_out_axes)
        ),
    )


def declare_dense_kernel(
    module,
    role: ParamRole,
    *,
    in_axes=(0,),
    out_axes=None,
    head_in_axes=None,
    head_out_axes=None,
) -> None:
    if out_axes is None:
        out_axes = tuple(range(1, module.kernel.get_value().ndim))
    module.kernel = declare_parameter(
        module.kernel,
        role,
        matrix_in_axes=in_axes,
        matrix_out_axes=out_axes,
        matrix_head_in_axes=head_in_axes,
        matrix_head_out_axes=head_out_axes,
    )
    if module.bias is not None:
        module.bias = declare_parameter(module.bias, ParamRole.BIAS)


def declare_norm(module) -> None:
    if module.scale is not None:
        module.scale = declare_parameter(module.scale, ParamRole.NORM_SCALE)
