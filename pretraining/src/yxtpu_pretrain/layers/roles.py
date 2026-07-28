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


# Alternate-matricization kinds, each selected by its own optimizer flag so
# the trajectory effect of every matricization change can be gated alone:
# "per_head" is Kimi K3 §2.5's per-head-block QKV orthogonalization
# (optimizer.muon_per_head); "kda_out_proj_whole" is the whole-matrix
# (heads*dim -> embed) correction of the KDA out_proj's historical heads-only
# reduction (optimizer.muon_kda_out_proj_whole).
ALT_KIND_PER_HEAD = "per_head"
ALT_KIND_KDA_OUT_PROJ_WHOLE = "kda_out_proj_whole"
ALT_KINDS = frozenset({ALT_KIND_PER_HEAD, ALT_KIND_KDA_OUT_PROJ_WHOLE})


def declare_parameter(
    parameter: nnx.Param,
    role: ParamRole,
    *,
    matrix_in_axes: Iterable[int] = (),
    matrix_out_axes: Iterable[int] = (),
    matrix_alt_in_axes: Iterable[int] | None = None,
    matrix_alt_out_axes: Iterable[int] | None = None,
    matrix_alt_kind: str | None = None,
) -> nnx.Param:
    """Returns a parameter with optimizer semantics attached as NNX metadata.

    ``matrix_alt_*`` declare an optional alternate Muon matricization tagged
    with the kind that enables it: axes absent from both alternate groups
    become Muon batch axes. ``None`` means the parameter has no alternate and
    keeps its standard matricization under every optimizer flag.
    """
    if (matrix_alt_in_axes is None) != (matrix_alt_out_axes is None):
        raise ValueError("alternate matricization must declare both axis groups")
    if (matrix_alt_in_axes is None) != (matrix_alt_kind is None):
        raise ValueError("alternate matricization requires a kind tag")
    if matrix_alt_kind is not None and matrix_alt_kind not in ALT_KINDS:
        raise ValueError(f"unknown alternate matricization kind {matrix_alt_kind!r}")
    return parameter.replace(
        role=str(role),
        matrix_in_axes=tuple(matrix_in_axes),
        matrix_out_axes=tuple(matrix_out_axes),
        matrix_alt_in_axes=(
            None if matrix_alt_in_axes is None else tuple(matrix_alt_in_axes)
        ),
        matrix_alt_out_axes=(
            None if matrix_alt_out_axes is None else tuple(matrix_alt_out_axes)
        ),
        matrix_alt_kind=matrix_alt_kind,
    )


def declare_dense_kernel(
    module,
    role: ParamRole,
    *,
    in_axes=(0,),
    out_axes=None,
    alt_in_axes=None,
    alt_out_axes=None,
    alt_kind=None,
) -> None:
    if out_axes is None:
        out_axes = tuple(range(1, module.kernel.get_value().ndim))
    module.kernel = declare_parameter(
        module.kernel,
        role,
        matrix_in_axes=in_axes,
        matrix_out_axes=out_axes,
        matrix_alt_in_axes=alt_in_axes,
        matrix_alt_out_axes=alt_out_axes,
        matrix_alt_kind=alt_kind,
    )
    if module.bias is not None:
        module.bias = declare_parameter(module.bias, ParamRole.BIAS)


def declare_norm(module) -> None:
    if module.scale is not None:
        module.scale = declare_parameter(module.scale, ParamRole.NORM_SCALE)
