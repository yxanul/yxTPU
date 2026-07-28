# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU-v4 fused Pallas kernel for Kimi Delta Attention.

This is the pre-mixer-fold kernel (the state of ``kda_fused_pallas.py`` at
commit 9d1aced) kept alive for TPU v4. The folded kernel's in-kernel
``[time, head]`` transposes lower to ``tpu.dynamic_gather`` over sublanes,
which v4's Mosaic backend rejects ("Sublane gather not supported by this TPU
generation"), so on v4 the QKV mixer stays in XLA and this kernel consumes
post-mixer per-head tensors instead of the raw projection output.

The production path assigns one ordered chunk stream to each ``(batch, head)``
pair. A ``K x V`` FP32 fast-weight state remains in VMEM while the ordered grid
walks through the sequence. Each invocation consumes one BF16 Q/K/V chunk,
recomputes compact intra-chunk quantities, emits BF16 output, and stores only
the FP32 state after that chunk for a future custom backward.

This module deliberately fixes the validated precision and solver policy:
one-pass BF16-operand matmuls for ordinary KDA work and a divide-and-conquer
explicit inverse for the WY solve and its transpose at three emulated fp32
passes (``_SOLVE_INVERSE_PASSES``; qualified by EXP-039's direct real-text
solve-error measurements, worst case 2.3e-5 — set to 6 for the full fp32
decomposition). Recursive doubling, stage exits, and unsafe precision
controls live under ``pretraining/benchmarks/kda_fused_experimental.py``.
"""

from __future__ import annotations

import functools
import os
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_SOLVE_BLOCK_SIZE = 16

# Triangular solve algorithm. "doubling" forms the whole nilpotent series by
# repeated squaring and is retained only as an experimental control. Correlated
# real-text keys make its explicit L^2 ... L^32 intermediates grow to roughly
# 1e12 even when the unit-lower system and its true solution are benign. The
# resulting FP32 cancellation invalidates the backward.
#
# "substitution", the fail-closed replacement, is stable but latency-bound:
# its 16-row serial base case chains roughly sixty tiny six-pass matmuls per
# solve, and the backward runs that chain twice. It measured 9.24 ms in the
# fused core and 472,668 tok/s in the ClimbMix model.
#
# Production uses "inverse": a divide-and-conquer explicit unit-lower inverse
# whose merge step ``inv = M - M @ C @ M`` is exact because the premultiplied
# coupling between two inverted halves is nilpotent of index two. It forms no
# matrix power and no series sum, so it keeps substitution's stability class,
# while the whole 64x64 inverse forms in ten uniform matmuls and one dense
# apply solves the 256-wide right-hand side. The backward reuses one formation
# for both its forward recompute and its transposed solve. Measured: 5.02 ms
# fused core forward+backward (substitution 9.24, rejected doubling 6.44) and
# 616,303 tok/s over the 15-step real-text gate, +30.4% over substitution,
# with every trigger step finite. On the exact update-7/microbatch-4 trigger
# it matches the full-FP32 analytical reference at least as closely as
# substitution did: gradient norm 2.406743 vs reference 2.406825, exhaustive
# vector comparison 1.8622% relative L2 at cosine 0.999827 (substitution:
# 1.8656% at 0.999827; the residual is the one-pass BF16 chunk/state/pairwise
# policy, not the solver).
#
# The stress harness established that condition number, power
# growth, and solution-path cancellation are complementary diagnostics, but
# none orders every regime correctly. Production therefore gates precision on
# measured BF16-versus-full-pass solve and gradient error, not on a proxy.
# See benchmarks/diagnose_wy_conditioning.py and EXP-034/EXP-036 in
# EXPERIMENTS.md.
_SOLVE_METHOD = "inverse"

# Base case for the substitution solve. "serial" forms no power of the
# diagonal block at all and is the stable limit; "doubling" runs the same
# nilpotent series on the 16x16 block, where the growth exponent is 15 rather
# than 63 and full passes are cheap because the block is small.
_SOLVE_BASE_CASE = "serial"

# Diagonal block size at which the "inverse" solve stops recursing and reads
# the block inverse directly. At 2 the base case is ``I - L`` with no matmul at
# all, which minimizes the sequential matmul chain: the whole 64x64 inverse
# forms in 2*log2(64/2) = 10 uniform matmuls. Larger bases trade chain length
# for fewer, equally stable masked-row steps and exist only for measurement.
_SOLVE_INVERSE_BASE_BLOCK_SIZE = 2

# Rows of the decayed pairwise matrix built per MXU matmul. Each row block
# rescales both operands around a shared per-channel anchor, so the block size
# is bounded by how much channel decay may accumulate across it before the FP32
# exponent range runs out, not by correctness.
#
# The safe-gate parameterization (kimi_delta_attention.py: log decay bounded to
# [gate_lower_bound, 0), gate_lower_bound = -5) exists precisely to license
# this tiling. Kimi K3 (§2.1.1, Eq. 5) fixes the same g_min = -5 and states the
# resulting budget: the cumulative log-decay over a 16-token tile lies in
# (-80, 0), so the one-sided reciprocal rescaling factor stays below
# e^80 ~ 5.5e34, inside the shared fp32/bf16 exponent range (max ~3.4e38) —
# which is why K3 runs 16-token tiles as dense tensor-core matmuls in BF16 at
# production scale. 16 rows with last-row anchoring is exactly that budget;
# 32 rows would need midpoint anchoring (e^80 per side) and 32 with last-row
# anchoring overflows (e^160). The import-time assert below enforces the
# envelope so neither knob can silently leave it.
_PAIRWISE_ROW_BLOCK_SIZE = 16

# The anchor cancels exactly between the two operands, so any row may serve as
# it. Anchoring on the last row keeps the right operand at or below one and
# puts the whole range on the left; anchoring at the midpoint splits the range
# evenly and therefore tolerates twice the row block at equal worst-case
# exponent.
_PAIRWISE_ANCHOR_MIDPOINT = False

# |log decay| per token is bounded by the safe gate's lower bound; the config
# validator rejects models that exceed it (config.py enforces
# gate_lower_bound >= -5 for the fused kernel). Changing either constant above
# or that bound requires revisiting this budget together — the gate bound is a
# kernel compute parameter, not only a stability knob.
_PAIRWISE_GATE_LOG_BOUND = 5.0
_PAIRWISE_MAX_SAFE_EXPONENT = 85.0
_PAIRWISE_WORST_EXPONENT = _PAIRWISE_GATE_LOG_BOUND * (
    _PAIRWISE_ROW_BLOCK_SIZE / 2 if _PAIRWISE_ANCHOR_MIDPOINT else _PAIRWISE_ROW_BLOCK_SIZE
)
assert _PAIRWISE_WORST_EXPONENT <= _PAIRWISE_MAX_SAFE_EXPONENT, (
    "pairwise tiling leaves the fp32/bf16 exponent envelope: "
    f"worst one-sided exponent {_PAIRWISE_WORST_EXPONENT} > "
    f"{_PAIRWISE_MAX_SAFE_EXPONENT}; shrink _PAIRWISE_ROW_BLOCK_SIZE or enable "
    "midpoint anchoring"
)

# Independent ``(batch, head)`` streams advanced by a single Pallas program.
# The chunk axis carries the only real sequential dependency, so batching
# streams into one program amortizes per-iteration grid and DMA cost over
# proportionally more work. Mosaic's ``tpu.matmul`` accepts a single batch
# dimension, so batch and head are merged into one stream axis rather than
# kept as two leading block axes.
_DEFAULT_STREAMS_PER_PROGRAM = 8

# MXU precision for the in-kernel matmuls. FP32 operands on TPU are evaluated
# by decomposing into BF16 passes: HIGHEST is six passes, HIGH is three (which
# this kernel fails to compile), and DEFAULT is one. Q/K/V arrive in BF16, so
# for most of the kernel the extra passes refine mantissa bits the operands
# never carried and one pass is both faster and numerically sufficient.
#
# The triangular solve is the exception and is kept at six passes below. A
# blanket reduction is 4.93x faster in the core but diverges to NaN by model
# step two; guarding the solve alone recovers 561,106 tok/s with the loss curve
# intact.
_PRECISION_BY_NAME = {
    "highest": lax.Precision.HIGHEST,
    "high": lax.Precision.HIGH,
    "default": lax.Precision.DEFAULT,
}
_CHUNK_MATMUL_PRECISION = lax.Precision.DEFAULT
_STATE_MATMUL_PRECISION = lax.Precision.DEFAULT

# The pairwise construction is the one place whose operands are rescaled by
# channel decay: their product is bounded by one, but the individual factors
# reach exp(row_block * |gate_lower_bound|). That made it the first suspect for
# the BF16 divergence, and it was wrong. Holding the pairwise at six passes
# while the rest of the chunk ran at one still reached NaN, and running the
# pairwise at one pass while only the solve was guarded trains normally. The
# factors are large but they are exactly representable in BF16's exponent, and
# each product is formed once rather than fed back. This class is kept separate
# only so the hypothesis stays cheap to re-test.
_PAIRWISE_MATMUL_PRECISION = lax.Precision.DEFAULT

# The nilpotent series solve raises the strictly lower factor to the power of
# the chunk size by squaring it log2(chunk) times, so rounding is repeatedly
# fed back. Growth helps explain backward error and conditioning bounds its
# possible amplification, but the corrected TPU table also contains regimes
# that both proxies mis-rank. The decisive evidence is direct arithmetic:
# one-pass BF16 doubling reaches backward error from 0.048 to 0.7 in plausible
# stress regimes and produced a NaN by model step two. The solve and its
# application therefore remain full-pass unconditionally.
_SOLVE_MATMUL_PRECISION = lax.Precision.HIGHEST

# Splitting the series by matmul role does not help, and this knob records why.
# Applying a power looks additive, but the update is
# ``solution <- (I + P^(2^k)) solution``: the running solution is fed back into
# itself exactly as the power is, so its error compounds over the same
# log2(chunk) stages. Measured on the 272.9M hybrid, dropping only these
# applications to one BF16 pass still reaches NaN at step two. The whole solve
# needs the full passes; keep this at ``highest``.
#
# Both constants above now guard only the non-production doubling/substitution
# control paths. The production inverse runs at the pass count below.
_SOLVE_APPLY_MATMUL_PRECISION = lax.Precision.HIGHEST

# Pass count for the production inverse's formation and applies. EXP-039
# (OPEN-002 run at ckpt 95500) measured the decisive quantities directly with
# TPU arithmetic on real-text and random-token systems at trained weights:
# every layer benign (kappa_2 worst 10.4, doubling growth worst 4.3e5 — the
# 1e12..1e17 catastrophe class never appears), three-pass solve error
# worst-case 2.3e-5 forward / 1.0e-5 transposed on the kernel's exact
# right-hand sides — two orders inside the few-times-1e-4 equivalence
# envelope — while one pass sits at ~7e-3, marginal. Three passes therefore
# halve the solve's dominant MXU cost at a directly-qualified accuracy.
#
# Mosaic still rejects ``lax.Precision.HIGH`` inside Pallas kernels
# (NotImplementedError in the dot lowering, reconfirmed 2026-07-28 on the
# current pin), so three passes are emulated: each fp32 operand splits into a
# bf16 high part plus a bf16 residual, and the three significant cross
# products accumulate through one-pass matmuls in fp32. The dropped low*low
# term is bounded at ~2^-16 relative — below the three-pass target. Set to 6
# to restore the full fp32 decomposition (bit-identical rollback).
_SOLVE_INVERSE_PASSES = 3

# Coupling between diagonal blocks in the substitution solve. These matmuls do
# not form powers, but the real-text/correlated-key qualification sweep still
# needs their operands evaluated with the full TPU FP32 decomposition to keep
# forward and transposed solves near the solve_triangular reference.
_SOLVE_COUPLING_MATMUL_PRECISION = lax.Precision.HIGHEST

__all__ = ["pallas_kda_fused_v4"]


def _matmul(left: jax.Array, right: jax.Array, *, precision=None) -> jax.Array:
  """Contracts the last axis of ``left`` with the leading matrix axis of
  ``right``, batching over a shared leading head axis when both are rank three."""
  if precision is None:
    precision = _CHUNK_MATMUL_PRECISION
  if left.ndim == 2:
    return lax.dot_general(
        left,
        right,
        (((1,), (0,)), ((), ())),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
  batch_axes = tuple(range(left.ndim - 2))
  return lax.dot_general(
      left,
      right,
      (((left.ndim - 1,), (right.ndim - 2,)), (batch_axes, batch_axes)),
      precision=precision,
      preferred_element_type=jnp.float32,
  )


def _matmul_tn(left: jax.Array, right: jax.Array, *, precision=None) -> jax.Array:
  """``_matmul(_transpose(left), right)`` without materializing the transpose.

  Contracts the second-to-last axis of both operands so the MXU consumes the
  transposed left operand directly. TPU v4's Mosaic backend cannot lower the
  skinny ``[row_block, chunk]`` vector transpose this replaces — it needs a
  sublane gather that only exists on v5 and later generations.
  """
  if precision is None:
    precision = _CHUNK_MATMUL_PRECISION
  if left.ndim == 2:
    return lax.dot_general(
        left,
        right,
        (((0,), (0,)), ((), ())),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
  batch_axes = tuple(range(left.ndim - 2))
  return lax.dot_general(
      left,
      right,
      (((left.ndim - 2,), (right.ndim - 2,)), (batch_axes, batch_axes)),
      precision=precision,
      preferred_element_type=jnp.float32,
  )


def _state_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
  """Matmul for terms that read or write the cross-chunk recurrent state."""
  return _matmul(left, right, precision=_STATE_MATMUL_PRECISION)


def _pairwise_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
  """Matmul whose operands have been rescaled by accumulated channel decay."""
  return _matmul(left, right, precision=_PAIRWISE_MATMUL_PRECISION)


def _solve_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
  """Squaring step of the repeated-squaring triangular solve."""
  return _matmul(left, right, precision=_SOLVE_MATMUL_PRECISION)


def _solve_apply_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
  """Applies one power of the series to the running solution."""
  return _matmul(left, right, precision=_SOLVE_APPLY_MATMUL_PRECISION)


def _split_bf16(values: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Splits an fp32 array into a bf16 high part and a bf16 residual."""
  values = values.astype(jnp.float32)
  high = values.astype(jnp.bfloat16)
  low = (values - high.astype(jnp.float32)).astype(jnp.bfloat16)
  return high, low


def _bf16x3_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
  """Three-pass fp32 matmul emulation: hi@hi + hi@lo + lo@hi in fp32."""
  left_high, left_low = _split_bf16(left)
  right_high, right_low = _split_bf16(right)
  result = _matmul(left_high, right_high, precision=lax.Precision.DEFAULT)
  result = result + _matmul(left_high, right_low, precision=lax.Precision.DEFAULT)
  result = result + _matmul(left_low, right_high, precision=lax.Precision.DEFAULT)
  return result


def _bf16x3_matmul_tn(left: jax.Array, right: jax.Array) -> jax.Array:
  """Three-pass emulation of ``_matmul_tn`` (transposed-left contraction)."""
  left_high, left_low = _split_bf16(left)
  right_high, right_low = _split_bf16(right)
  result = _matmul_tn(left_high, right_high, precision=lax.Precision.DEFAULT)
  result = result + _matmul_tn(left_high, right_low, precision=lax.Precision.DEFAULT)
  result = result + _matmul_tn(left_low, right_high, precision=lax.Precision.DEFAULT)
  return result


def _solve_form_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
  """Formation matmuls of the production inverse (see _SOLVE_INVERSE_PASSES)."""
  if _SOLVE_INVERSE_PASSES == 3:
    return _bf16x3_matmul(left, right)
  return _matmul(left, right, precision=_SOLVE_MATMUL_PRECISION)


def _solve_inverse_apply_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
  """Dense apply of the already-formed inverse to a right-hand side."""
  if _SOLVE_INVERSE_PASSES == 3:
    return _bf16x3_matmul(left, right)
  return _matmul(left, right, precision=_SOLVE_APPLY_MATMUL_PRECISION)


def _solve_inverse_apply_matmul_tn(left: jax.Array, right: jax.Array) -> jax.Array:
  """Transposed apply of the formed inverse through contraction dimensions."""
  if _SOLVE_INVERSE_PASSES == 3:
    return _bf16x3_matmul_tn(left, right)
  return _matmul_tn(left, right, precision=_SOLVE_APPLY_MATMUL_PRECISION)


def _solve_coupling_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
  """Off-diagonal coupling between substitution blocks; forms no powers."""
  return _matmul(left, right, precision=_SOLVE_COUPLING_MATMUL_PRECISION)


def _transpose(values: jax.Array) -> jax.Array:
  """Swaps the two trailing matrix axes, leaving any head axis in place."""
  return jnp.swapaxes(values, -1, -2)


def _inclusive_cumsum(values: jax.Array) -> jax.Array:
  """Static power-of-two inclusive cumsum suitable for a Pallas TPU kernel."""
  length = values.shape[-2]
  if length & (length - 1):
    raise ValueError(f"cumsum length must be a power of two, got {length}")
  result = values.astype(jnp.float32)
  for depth in range(int(math.log2(length))):
    stride = 1 << depth
    result = jnp.concatenate(
        (
            result[..., :stride, :],
            result[..., stride:, :] + result[..., :-stride, :],
        ),
        axis=-2,
    )
  return result


def _reverse_inclusive_cumsum(values: jax.Array) -> jax.Array:
  """Static power-of-two reverse cumsum without a Mosaic ``rev`` primitive."""
  length = values.shape[-2]
  if length & (length - 1):
    raise ValueError(f"cumsum length must be a power of two, got {length}")
  result = values.astype(jnp.float32)
  for depth in range(int(math.log2(length))):
    stride = 1 << depth
    result = jnp.concatenate(
        (
            result[..., :-stride, :] + result[..., stride:, :],
            result[..., -stride:, :],
        ),
        axis=-2,
    )
  return result


def _l2_normalize(values: jax.Array, *, scale: float = 1.0) -> jax.Array:
  values = values.astype(jnp.float32)
  inverse_norm = lax.rsqrt(jnp.sum(values * values, axis=-1, keepdims=True) + 1e-6)
  return values * inverse_norm * scale


def _l2_normalize_with_inverse(values: jax.Array) -> tuple[jax.Array, jax.Array]:
  values = values.astype(jnp.float32)
  inverse_norm = lax.rsqrt(jnp.sum(values * values, axis=-1, keepdims=True) + 1e-6)
  return values * inverse_norm, inverse_norm


def _l2_normalize_backward(
    output_cotangent: jax.Array,
    normalized: jax.Array,
    inverse_norm: jax.Array,
) -> jax.Array:
  projection = jnp.sum(output_cotangent * normalized, axis=-1, keepdims=True)
  return (output_cotangent - normalized * projection) * inverse_norm


def _pairwise_anchor_row(row_block_size: int) -> int:
  """Row within a pairwise block whose decay both operands are rescaled by."""
  return row_block_size // 2 if _PAIRWISE_ANCHOR_MIDPOINT else -1


def _decayed_pairwise(
    left: jax.Array,
    right: jax.Array,
    cumulative_log_decay: jax.Array,
    *,
    include_diagonal: bool,
) -> jax.Array:
  """Forms a causal channel-decayed dot matrix with eight-row MXU matmuls."""
  chunk_size = left.shape[-2]
  channel_dim = cumulative_log_decay.shape[-1]
  leading_shape = left.shape[:-2]
  row_block_size = _PAIRWISE_ROW_BLOCK_SIZE
  if chunk_size % row_block_size:
    raise ValueError(
        f"chunk size {chunk_size} must be divisible by row block size {row_block_size}"
    )

  row_blocks = []
  for block_index in range(chunk_size // row_block_size):
    row_start = block_index * row_block_size
    row_end = row_start + row_block_size
    left_block = left[..., row_start:row_end, :].astype(jnp.float32)
    decay_block = cumulative_log_decay[..., row_start:row_end, :]
    anchor = decay_block[..., _pairwise_anchor_row(row_block_size), :]

    right_exponent = anchor[..., None, :] - cumulative_log_decay[..., :row_end, :]
    if row_end < chunk_size:
      right_exponent = jnp.concatenate(
          (
              right_exponent,
              jnp.full(
                  leading_shape + (chunk_size - row_end, channel_dim),
                  -jnp.inf,
                  dtype=jnp.float32,
              ),
          ),
          axis=-2,
      )
    weighted_right = right.astype(jnp.float32) * jnp.exp(right_exponent)
    weighted_left = left_block * jnp.exp(decay_block - anchor[..., None, :])
    row_blocks.append(_pairwise_matmul(weighted_left, _transpose(weighted_right)))

  values = jnp.concatenate(row_blocks, axis=-2)
  if include_diagonal:
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  else:
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  return values * causal


def _decayed_pairwise_pair(
    left_system: jax.Array,
    left_intra: jax.Array,
    right: jax.Array,
    cumulative_log_decay: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Forms the strictly-lower system matrix and the causal intra matrix in one
  shared row-block pass.

  The two `_decayed_pairwise` calls in every kernel share ``right``, the
  cumulative decay, and therefore every per-block anchor and exponential; only
  the left operand differs. Stacking both left blocks into one MXU matmul per
  row block is bit-identical to two separate calls: each output row is an
  independent dot product, so row stacking cannot change any accumulation, and
  the shared exponentials are computed once from identical inputs. Columns at
  or beyond a block's row range are exact zeros in both formulations —
  previously as ``exp(-inf)`` operand rows the matmul then multiplied, here as
  explicit zero padding — so the ~44% of exponentials that were guaranteed
  zeros are simply never evaluated.
  """
  chunk_size = left_system.shape[-2]
  leading_shape = left_system.shape[:-2]
  row_block_size = _PAIRWISE_ROW_BLOCK_SIZE
  if chunk_size % row_block_size:
    raise ValueError(
        f"chunk size {chunk_size} must be divisible by row block size {row_block_size}"
    )

  system_blocks = []
  intra_blocks = []
  for block_index in range(chunk_size // row_block_size):
    row_start = block_index * row_block_size
    row_end = row_start + row_block_size
    decay_block = cumulative_log_decay[..., row_start:row_end, :]
    anchor = decay_block[..., _pairwise_anchor_row(row_block_size), :]

    right_exponent = anchor[..., None, :] - cumulative_log_decay[..., :row_end, :]
    weighted_right = right[..., :row_end, :].astype(jnp.float32) * jnp.exp(right_exponent)
    left_factor = jnp.exp(decay_block - anchor[..., None, :])
    stacked_left = jnp.concatenate(
        (
            left_system[..., row_start:row_end, :].astype(jnp.float32) * left_factor,
            left_intra[..., row_start:row_end, :].astype(jnp.float32) * left_factor,
        ),
        axis=-2,
    )
    values = _pairwise_matmul(stacked_left, _transpose(weighted_right))
    if row_end < chunk_size:
      values = jnp.concatenate(
          (
              values,
              jnp.zeros(
                  leading_shape + (2 * row_block_size, chunk_size - row_end),
                  dtype=jnp.float32,
              ),
          ),
          axis=-1,
      )
    system_blocks.append(values[..., :row_block_size, :])
    intra_blocks.append(values[..., row_block_size:, :])

  system = jnp.concatenate(system_blocks, axis=-2)
  intra = jnp.concatenate(intra_blocks, axis=-2)
  strict_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  full_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  return system * strict_causal, intra * full_causal


def _decayed_pairwise_backward(
    left: jax.Array,
    right: jax.Array,
    cumulative_log_decay: jax.Array,
    output_cotangent: jax.Array,
    *,
    include_diagonal: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Blockwise VJP for ``_decayed_pairwise`` inside a Pallas program."""
  chunk_size = left.shape[-2]
  channel_dim = left.shape[-1]
  leading_shape = left.shape[:-2]
  row_block_size = _PAIRWISE_ROW_BLOCK_SIZE
  if include_diagonal:
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  else:
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  output_cotangent = output_cotangent.astype(jnp.float32) * causal

  right_cotangent = jnp.zeros(leading_shape + (chunk_size, channel_dim), dtype=jnp.float32)
  decay_cotangent_from_right = jnp.zeros(
      leading_shape + (chunk_size, channel_dim),
      dtype=jnp.float32,
  )
  left_cotangent_blocks = []
  decay_cotangent_from_left_blocks = []
  for block_index in range(chunk_size // row_block_size):
    row_start = block_index * row_block_size
    row_end = row_start + row_block_size
    left_block = left[..., row_start:row_end, :].astype(jnp.float32)
    decay_block = cumulative_log_decay[..., row_start:row_end, :]
    anchor = decay_block[..., _pairwise_anchor_row(row_block_size), :]

    right_exponent = anchor[..., None, :] - cumulative_log_decay[..., :row_end, :]
    if row_end < chunk_size:
      right_exponent = jnp.concatenate(
          (
              right_exponent,
              jnp.full(
                  leading_shape + (chunk_size - row_end, channel_dim),
                  -jnp.inf,
                  dtype=jnp.float32,
              ),
          ),
          axis=-2,
      )
    right_factor = jnp.exp(right_exponent)
    weighted_right = right.astype(jnp.float32) * right_factor
    left_factor = jnp.exp(decay_block - anchor[..., None, :])
    weighted_left = left_block * left_factor
    cotangent_block = output_cotangent[..., row_start:row_end, :]

    weighted_left_cotangent = _pairwise_matmul(cotangent_block, weighted_right)
    weighted_right_cotangent = _matmul_tn(
        cotangent_block, weighted_left, precision=_PAIRWISE_MATMUL_PRECISION
    )
    left_cotangent_blocks.append(weighted_left_cotangent * left_factor)
    right_cotangent = right_cotangent + weighted_right_cotangent * right_factor

    left_decay_product = weighted_left_cotangent * weighted_left
    right_decay_product = weighted_right_cotangent * weighted_right
    decay_cotangent_from_right = decay_cotangent_from_right - right_decay_product
    anchor_cotangent = -jnp.sum(left_decay_product, axis=-2) + jnp.sum(
        right_decay_product,
        axis=-2,
    )
    anchor_row = _pairwise_anchor_row(row_block_size) % row_block_size
    if anchor_row == row_block_size - 1:
      left_decay_product = jnp.concatenate(
          (
              left_decay_product[..., :-1, :],
              left_decay_product[..., -1:, :] + anchor_cotangent[..., None, :],
          ),
          axis=-2,
      )
    else:
      left_decay_product = jnp.concatenate(
          (
              left_decay_product[..., :anchor_row, :],
              left_decay_product[..., anchor_row : anchor_row + 1, :]
              + anchor_cotangent[..., None, :],
              left_decay_product[..., anchor_row + 1 :, :],
          ),
          axis=-2,
      )
    decay_cotangent_from_left_blocks.append(left_decay_product)

  return (
      jnp.concatenate(left_cotangent_blocks, axis=-2),
      right_cotangent,
      jnp.concatenate(decay_cotangent_from_left_blocks, axis=-2) + decay_cotangent_from_right,
  )


def _decayed_pairwise_backward_pair(
    left_system: jax.Array,
    left_intra: jax.Array,
    right: jax.Array,
    cumulative_log_decay: jax.Array,
    system_cotangent: jax.Array,
    intra_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Blockwise VJP for both decayed pairwise matrices in shared block passes.

  Bit-identical to one `_decayed_pairwise_backward` call per matrix: the
  stacked cotangent matmul keeps every output row an independent dot product,
  trimmed contractions drop only exactly-zero terms (columns the causal masks
  zeroed and rows the old code weighted by ``exp(-inf)``), and every reduction
  feeding an anchor or decay cotangent keeps its original 64-row operand shape
  so the reduction tree is unchanged. Cotangents are returned per matrix so
  callers keep their existing accumulation order.
  """
  chunk_size = left_system.shape[-2]
  channel_dim = left_system.shape[-1]
  leading_shape = left_system.shape[:-2]
  row_block_size = _PAIRWISE_ROW_BLOCK_SIZE
  strict_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  full_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  system_cotangent = system_cotangent.astype(jnp.float32) * strict_causal
  intra_cotangent = intra_cotangent.astype(jnp.float32) * full_causal

  right_cotangent_system = jnp.zeros(
      leading_shape + (chunk_size, channel_dim), dtype=jnp.float32
  )
  right_cotangent_intra = jnp.zeros_like(right_cotangent_system)
  decay_from_right_system = jnp.zeros_like(right_cotangent_system)
  decay_from_right_intra = jnp.zeros_like(right_cotangent_system)
  left_cotangent_system_blocks = []
  left_cotangent_intra_blocks = []
  decay_from_left_system_blocks = []
  decay_from_left_intra_blocks = []
  anchor_row = _pairwise_anchor_row(row_block_size) % row_block_size

  def _fold_anchor(left_decay_product, right_decay_product):
    anchor_cotangent = -jnp.sum(left_decay_product, axis=-2) + jnp.sum(
        right_decay_product,
        axis=-2,
    )
    if anchor_row == row_block_size - 1:
      return jnp.concatenate(
          (
              left_decay_product[..., :-1, :],
              left_decay_product[..., -1:, :] + anchor_cotangent[..., None, :],
          ),
          axis=-2,
      )
    return jnp.concatenate(
        (
            left_decay_product[..., :anchor_row, :],
            left_decay_product[..., anchor_row : anchor_row + 1, :]
            + anchor_cotangent[..., None, :],
            left_decay_product[..., anchor_row + 1 :, :],
        ),
        axis=-2,
    )

  for block_index in range(chunk_size // row_block_size):
    row_start = block_index * row_block_size
    row_end = row_start + row_block_size
    decay_block = cumulative_log_decay[..., row_start:row_end, :]
    anchor = decay_block[..., _pairwise_anchor_row(row_block_size), :]

    right_exponent = anchor[..., None, :] - cumulative_log_decay[..., :row_end, :]
    right_factor = jnp.exp(right_exponent)
    weighted_right = right[..., :row_end, :].astype(jnp.float32) * right_factor
    left_factor = jnp.exp(decay_block - anchor[..., None, :])
    weighted_left_system = (
        left_system[..., row_start:row_end, :].astype(jnp.float32) * left_factor
    )
    weighted_left_intra = (
        left_intra[..., row_start:row_end, :].astype(jnp.float32) * left_factor
    )
    cotangent_system_block = system_cotangent[..., row_start:row_end, :row_end]
    cotangent_intra_block = intra_cotangent[..., row_start:row_end, :row_end]

    def _pad_rows(values):
      if row_end == chunk_size:
        return values
      return jnp.concatenate(
          (
              values,
              jnp.zeros(
                  leading_shape + (chunk_size - row_end, channel_dim),
                  dtype=jnp.float32,
              ),
          ),
          axis=-2,
      )

    stacked_left_cotangent = _pairwise_matmul(
        jnp.concatenate((cotangent_system_block, cotangent_intra_block), axis=-2),
        weighted_right,
    )
    weighted_left_system_cotangent = stacked_left_cotangent[..., :row_block_size, :]
    weighted_left_intra_cotangent = stacked_left_cotangent[..., row_block_size:, :]
    weighted_right_system_cotangent = _matmul_tn(
        cotangent_system_block,
        weighted_left_system,
        precision=_PAIRWISE_MATMUL_PRECISION,
    )
    weighted_right_intra_cotangent = _matmul_tn(
        cotangent_intra_block,
        weighted_left_intra,
        precision=_PAIRWISE_MATMUL_PRECISION,
    )
    left_cotangent_system_blocks.append(weighted_left_system_cotangent * left_factor)
    left_cotangent_intra_blocks.append(weighted_left_intra_cotangent * left_factor)
    right_cotangent_system = right_cotangent_system + _pad_rows(
        weighted_right_system_cotangent * right_factor
    )
    right_cotangent_intra = right_cotangent_intra + _pad_rows(
        weighted_right_intra_cotangent * right_factor
    )

    left_decay_product_system = weighted_left_system_cotangent * weighted_left_system
    left_decay_product_intra = weighted_left_intra_cotangent * weighted_left_intra
    right_decay_product_system = _pad_rows(
        weighted_right_system_cotangent * weighted_right
    )
    right_decay_product_intra = _pad_rows(
        weighted_right_intra_cotangent * weighted_right
    )
    decay_from_right_system = decay_from_right_system - right_decay_product_system
    decay_from_right_intra = decay_from_right_intra - right_decay_product_intra
    decay_from_left_system_blocks.append(
        _fold_anchor(left_decay_product_system, right_decay_product_system)
    )
    decay_from_left_intra_blocks.append(
        _fold_anchor(left_decay_product_intra, right_decay_product_intra)
    )

  return (
      jnp.concatenate(left_cotangent_system_blocks, axis=-2),
      jnp.concatenate(left_cotangent_intra_blocks, axis=-2),
      right_cotangent_system,
      right_cotangent_intra,
      jnp.concatenate(decay_from_left_system_blocks, axis=-2) + decay_from_right_system,
      jnp.concatenate(decay_from_left_intra_blocks, axis=-2) + decay_from_right_intra,
  )


def _solve_unit_lower_triangular(system: jax.Array, rhs: jax.Array) -> jax.Array:
  """Blocked forward substitution for a 64-row unit-lower system."""
  rows, _ = rhs.shape
  block_size = _SOLVE_BLOCK_SIZE
  if rows % block_size:
    raise ValueError(f"triangular dimension {rows} must be divisible by {block_size}")

  system = system.astype(jnp.float32)
  blocks = list(jnp.split(rhs.astype(jnp.float32), rows // block_size, axis=0))
  for block_index in range(rows // block_size):
    start = block_index * block_size
    end = start + block_size
    diagonal = system[start:end, start:end]
    solution_rows = [blocks[block_index][row] for row in range(block_size)]
    for row in range(block_size):
      if row:
        correction = _matmul(
            diagonal[row, :row][None, :],
            jnp.stack(solution_rows[:row]),
        )[0]
        solution_rows[row] = solution_rows[row] - correction

    solved = jnp.stack(solution_rows)
    blocks[block_index] = solved
    if block_index + 1 < rows // block_size:
      remaining = jnp.concatenate(blocks[block_index + 1 :], axis=0)
      remaining = remaining - _matmul(system[end:, start:end], solved)
      blocks[block_index + 1 :] = list(
          jnp.split(remaining, rows // block_size - block_index - 1, axis=0)
      )
  return jnp.concatenate(blocks, axis=0)


def _solve_unit_lower_triangular_doubling(
    system: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Exact nilpotent-series solve using logarithmic-depth MXU matmuls.

  For ``A = I + L`` with strictly lower ``L``, ``P = -L`` is nilpotent and
  ``A^-1 B = (I + P + ... + P^(C-1)) B``. Recursive doubling forms that
  finite series in ``log2(C)`` stages. It performs more FLOPs than forward
  substitution but exposes them as dense matmuls instead of serial row
  dependencies, which is a better candidate for TPU execution.
  """
  power = -jnp.tril(system.astype(jnp.float32), k=-1)
  return _nilpotent_series_solve(power, rhs)


def _small_unit_lower_forward_substitution(
    lower_block: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Row-serial forward substitution on a small unit-lower diagonal block.

  This is the stable base case. Each row is corrected only by rows already
  solved, so no power of the block is ever formed and there is no growing
  intermediate to lose precision to. It is serial in the block dimension,
  which is why the block is kept small.
  """
  rows = rhs.shape[-2]
  solved = []
  for row in range(rows):
    value = rhs[..., row : row + 1, :]
    if row:
      value = value - _solve_apply_matmul(
          lower_block[..., row : row + 1, :row],
          jnp.concatenate(solved, axis=-2),
      )
    solved.append(value)
  return jnp.concatenate(solved, axis=-2)


def _small_unit_upper_back_substitution(
    upper_block: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Row-serial back substitution on a small unit-upper diagonal block."""
  rows = rhs.shape[-2]
  solved = [None] * rows
  for row in range(rows - 1, -1, -1):
    value = rhs[..., row : row + 1, :]
    if row < rows - 1:
      value = value - _solve_apply_matmul(
          upper_block[..., row : row + 1, row + 1 :],
          jnp.concatenate(solved[row + 1 :], axis=-2),
      )
    solved[row] = value
  return jnp.concatenate(solved, axis=-2)


def _solve_unit_lower_triangular_substitution(
    system: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Blocked forward substitution for ``(I + tril(system, -1)) X = rhs``.

  Global recursive doubling reaches ``P^(C-1)`` by squaring ``P`` log2(C)
  times, so a relative error introduced early is multiplied by every later
  stage. Substitution instead confines the series to a diagonal block, where
  nilpotency caps growth at ``||L||^(block-1)`` rather than ``||L||^(C-1)``,
  and carries the coupling between blocks in plain matmuls that do not feed
  themselves.

  It is also cheaper. Doubling applies six full-width powers to the whole
  ``K + V`` right-hand side; substitution applies four narrow series per
  16-row block plus one growing off-diagonal matmul each, which is roughly a
  fifth of the arithmetic at chunk 64.
  """
  rows = rhs.shape[-2]
  block_size = _SOLVE_BLOCK_SIZE
  if rows % block_size:
    raise ValueError(f"triangular dimension {rows} must be divisible by {block_size}")
  lower = jnp.tril(system.astype(jnp.float32), k=-1)
  rhs = rhs.astype(jnp.float32)

  solved_blocks = []
  for block_index in range(rows // block_size):
    start = block_index * block_size
    end = start + block_size
    block_rhs = rhs[..., start:end, :]
    if block_index:
      block_rhs = block_rhs - _solve_coupling_matmul(
          lower[..., start:end, :start],
          jnp.concatenate(solved_blocks, axis=-2),
      )
    diagonal = lower[..., start:end, start:end]
    if _SOLVE_BASE_CASE == "serial":
      solved_blocks.append(_small_unit_lower_forward_substitution(diagonal, block_rhs))
    else:
      solved_blocks.append(_nilpotent_series_solve(-diagonal, block_rhs))
  return jnp.concatenate(solved_blocks, axis=-2)


def _solve_transposed_unit_lower_triangular_substitution(
    system: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Blocked back substitution for ``(I + tril(system, -1)).T X = rhs``."""
  rows = rhs.shape[-2]
  block_size = _SOLVE_BLOCK_SIZE
  if rows % block_size:
    raise ValueError(f"triangular dimension {rows} must be divisible by {block_size}")
  upper = _transpose(jnp.tril(system.astype(jnp.float32), k=-1))
  rhs = rhs.astype(jnp.float32)

  num_blocks = rows // block_size
  solved_blocks = [None] * num_blocks
  for block_index in range(num_blocks - 1, -1, -1):
    start = block_index * block_size
    end = start + block_size
    block_rhs = rhs[..., start:end, :]
    if block_index < num_blocks - 1:
      block_rhs = block_rhs - _solve_coupling_matmul(
          upper[..., start:end, end:],
          jnp.concatenate(solved_blocks[block_index + 1 :], axis=-2),
      )
    diagonal = upper[..., start:end, start:end]
    if _SOLVE_BASE_CASE == "serial":
      solved_blocks[block_index] = _small_unit_upper_back_substitution(diagonal, block_rhs)
    else:
      solved_blocks[block_index] = _nilpotent_series_solve(-diagonal, block_rhs)
  return jnp.concatenate(solved_blocks, axis=-2)


def _iota_rows_cols(size: int) -> tuple[jax.Array, jax.Array]:
  """Row and column index planes, built with iota so Pallas traces them as ops
  instead of rejecting them as captured array constants."""
  rows = lax.broadcasted_iota(jnp.int32, (size, size), 0)
  cols = lax.broadcasted_iota(jnp.int32, (size, size), 1)
  return rows, cols


def _blockdiag_strictly_lower_mask(size: int, block: int) -> jax.Array:
  """Selects strictly-lower entries that stay inside ``block``-sized diagonal blocks."""
  rows, cols = _iota_rows_cols(size)
  return ((rows // block == cols // block) & (rows > cols)).astype(jnp.float32)


def _block_row_mask(size: int, block: int, row: int) -> jax.Array:
  """Selects every row whose index within its ``block``-sized block is ``row``."""
  rows = lax.broadcasted_iota(jnp.int32, (size, 1), 0)
  return (rows % block == row).astype(jnp.float32)


def _half_coupling_mask(size: int, half: int) -> jax.Array:
  """Selects the lower-left ``half x half`` coupling block of each ``2*half`` pair."""
  rows, cols = _iota_rows_cols(size)
  same_pair = rows // (2 * half) == cols // (2 * half)
  row_in_lower_half = rows % (2 * half) >= half
  column_in_upper_half = cols % (2 * half) < half
  return (same_pair & row_in_lower_half & column_in_upper_half).astype(jnp.float32)


def _unit_lower_inverse(system: jax.Array) -> jax.Array:
  """Exact inverse of ``I + tril(system, -1)`` without forming matrix powers.

  Base diagonal blocks are inverted by masked serial substitution; at base 2
  the block inverse is literally ``I - L`` with no matmul. Each merge level
  then joins adjacent inverted blocks with the exact identity
  ``inv = M - M @ C @ M``: the coupling ``C`` between two inverted halves is
  nilpotent of index two once premultiplied by the block-diagonal inverse, so
  the two-term expansion is not a truncation. Every matmul multiplies already
  inverted, conditioning-bounded quantities; no ``L^2 ... L^32`` intermediate
  and no series cancellation exists anywhere, which is what invalidated
  recursive doubling on correlated real-text keys.

  The full 64x64 inverse forms in ``base - 2`` masked-row matmuls plus two
  matmuls per level: ten sequential ``[64, 64]`` matmuls at base 2, against
  roughly sixty-six chained row/coupling matmuls for row-serial substitution.
  Applying the inverse to a right-hand side is then one dense MXU matmul, and
  the backward pass reuses one formation for both its forward recompute and
  its transposed solve.
  """
  rows = system.shape[-1]
  base = _SOLVE_INVERSE_BASE_BLOCK_SIZE
  if rows & (rows - 1):
    raise ValueError(f"triangular dimension must be a power of two, got {rows}")
  if base < 2 or base & (base - 1) or rows % base:
    raise ValueError(f"invalid inverse base block size {base} for dimension {rows}")
  lower = jnp.tril(system.astype(jnp.float32), k=-1)
  index_rows, index_cols = _iota_rows_cols(rows)
  identity = (index_rows == index_cols).astype(jnp.float32)
  base_lower = lower * _blockdiag_strictly_lower_mask(rows, base)

  inverse = identity - base_lower * _block_row_mask(rows, base, 1)
  for row in range(2, base):
    row_update = base_lower * _block_row_mask(rows, base, row)
    inverse = inverse - _solve_form_matmul(row_update, inverse)

  half = base
  while half < rows:
    coupling = lower * _half_coupling_mask(rows, half)
    inverse = inverse - _solve_form_matmul(
        _solve_form_matmul(inverse, coupling), inverse
    )
    half *= 2
  return inverse


def _solve_unit_lower_triangular_inverse(system: jax.Array, rhs: jax.Array) -> jax.Array:
  """Solves ``(I + tril(system, -1)) X = rhs`` through the explicit inverse."""
  return _solve_inverse_apply_matmul(
      _unit_lower_inverse(system), rhs.astype(jnp.float32)
  )


def _solve_transposed_unit_lower_triangular_inverse(
    system: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Solves ``(I + tril(system, -1)).T X = rhs`` through the explicit inverse."""
  return _solve_inverse_apply_matmul_tn(
      _unit_lower_inverse(system),
      rhs.astype(jnp.float32),
  )


def _nilpotent_series_solve(power: jax.Array, rhs: jax.Array) -> jax.Array:
  """Applies ``(I - power)^-1`` when ``power`` is strictly triangular."""
  rows = rhs.shape[-2]
  if rows & (rows - 1):
    raise ValueError(f"triangular dimension must be a power of two, got {rows}")
  rhs = rhs.astype(jnp.float32)
  solution = rhs + _solve_apply_matmul(power, rhs)
  power = _solve_matmul(power, power)
  covered_terms = 2
  while covered_terms < rows:
    solution = solution + _solve_apply_matmul(power, solution)
    power = _solve_matmul(power, power)
    covered_terms *= 2
  return solution


def _solve_transposed_unit_lower_triangular_doubling(
    system: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Solves ``(I + tril(system, -1)).T X = rhs`` by recursive doubling."""
  power = -jnp.triu(_transpose(system.astype(jnp.float32)), k=1)
  return _nilpotent_series_solve(power, rhs)


def _kda_fused_forward_kernel(
    query_ref,
    key_ref,
    value_ref,
    log_decay_ref,
    beta_ref,
    initial_state_ref,
    output_ref,
    state_after_ref,
    state_scratch_ref,
    *,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
    use_qk_norm: bool,
    solve_method: str,
    profile_stage: str,
    chunk_axis: int = 2,
):
  """Consumes one chunk of every in-block head while retaining the
  fast-weight state in VMEM.

  Every reference carries a leading head axis so that one Pallas program can
  advance several ``(batch, head)`` streams at once. The single-stream layout
  is the same code path with a head axis of one.
  """
  chunk_index = pl.program_id(chunk_axis)
  query = query_ref[0]
  key = key_ref[0]
  value = value_ref[0].astype(jnp.float32)
  log_decay = log_decay_ref[0].astype(jnp.float32)
  beta = beta_ref[0][..., 0].astype(jnp.float32)

  @pl.when(chunk_index == 0)
  def _initialize_state():
    state_scratch_ref[...] = initial_state_ref[0].astype(jnp.float32)

  if use_qk_norm:
    query = _l2_normalize(query, scale=1.0 / math.sqrt(key_dim))
    key = _l2_normalize(key)
  else:
    query = query.astype(jnp.float32) * (1.0 / math.sqrt(key_dim))
    key = key.astype(jnp.float32)

  cumulative_decay = _inclusive_cumsum(log_decay)
  if profile_stage == "preprocess":
    diagnostic = query + key + 1e-3 * cumulative_decay
    output_ref[0] = diagnostic.astype(output_ref.dtype)
    state_after_ref[0, :, 0] = state_scratch_ref[...].astype(jnp.float32)
    return

  key_beta = key * beta[..., None]
  system, intra = _decayed_pairwise_pair(key_beta, query, key, cumulative_decay)
  if profile_stage == "pairwise":
    output_ref[0] = jnp.concatenate((system, intra), axis=-1).astype(output_ref.dtype)
    state_after_ref[0, :, 0] = state_scratch_ref[...].astype(jnp.float32)
    return

  value_beta = value * beta[..., None]
  w_input = key_beta * jnp.exp(cumulative_decay)
  combined_rhs = jnp.concatenate((value_beta, w_input), axis=-1)
  if solve_method == "blocked":
    solved = _solve_unit_lower_triangular(system, combined_rhs)
  elif solve_method == "doubling":
    solved = _solve_unit_lower_triangular_doubling(system, combined_rhs)
  elif solve_method == "substitution":
    solved = _solve_unit_lower_triangular_substitution(system, combined_rhs)
  elif solve_method == "inverse":
    solved = _solve_unit_lower_triangular_inverse(system, combined_rhs)
  else:
    raise ValueError(f"unknown solve method: {solve_method}")
  u = solved[..., :value_dim]
  w = solved[..., value_dim : value_dim + key_dim]
  if profile_stage == "solve":
    output_ref[0] = (u + w).astype(output_ref.dtype)
    state_after_ref[0, :, 0] = state_scratch_ref[...].astype(jnp.float32)
    return

  state = state_scratch_ref[...].astype(jnp.float32)
  query_with_decay = query * jnp.exp(cumulative_decay)
  # One [2C, K] x [K, V] MXU matmul serves both state reads. Output rows are
  # independent dot products, so splitting the stacked result is bit-identical
  # to the two separate half-height matmuls it replaces.
  stacked_state_reads = _state_matmul(
      jnp.concatenate((query_with_decay, w), axis=-2),
      state,
  )
  inter_output = stacked_state_reads[..., :chunk_size, :]
  corrected_value = u - stacked_state_reads[..., chunk_size:, :]
  output = inter_output + _matmul(intra, corrected_value)

  final_decay = cumulative_decay[..., -1, :]
  state = state * jnp.exp(final_decay)[..., :, None]
  key_for_state = key * jnp.exp(final_decay[..., None, :] - cumulative_decay)
  state = state + _state_matmul(_transpose(key_for_state), corrected_value)
  state_scratch_ref[...] = state

  output_ref[0] = output.astype(output_ref.dtype)
  state_after_ref[0, :, 0] = state


def _kda_fused_backward_kernel(
    query_ref,
    key_ref,
    value_ref,
    log_decay_ref,
    beta_ref,
    initial_state_ref,
    previous_state_after_ref,
    output_cotangent_ref,
    final_state_cotangent_ref,
    query_cotangent_ref,
    key_cotangent_ref,
    value_cotangent_ref,
    log_decay_cotangent_ref,
    beta_cotangent_ref,
    state_before_cotangent_ref,
    state_cotangent_scratch_ref,
    *,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
    num_chunks: int,
    use_qk_norm: bool,
    profile_stage: str,
    chunk_axis: int = 2,
):
  """Recomputes one chunk of every in-block head and carries the state
  cotangent in reverse order."""
  reverse_chunk_index = pl.program_id(chunk_axis)
  chunk_index = num_chunks - 1 - reverse_chunk_index
  query_input = query_ref[0]
  key_input = key_ref[0]
  value = value_ref[0].astype(jnp.float32)
  log_decay = log_decay_ref[0].astype(jnp.float32)
  beta = beta_ref[0][..., 0].astype(jnp.float32)
  output_cotangent = output_cotangent_ref[0].astype(jnp.float32)

  @pl.when(reverse_chunk_index == 0)
  def _initialize_state_cotangent():
    state_cotangent_scratch_ref[...] = final_state_cotangent_ref[0].astype(jnp.float32)

  state = lax.cond(
      chunk_index == 0,
      lambda: initial_state_ref[0].astype(jnp.float32),
      lambda: previous_state_after_ref[0, :, 0].astype(jnp.float32),
  )

  if use_qk_norm:
    query_normalized, query_inverse_norm = _l2_normalize_with_inverse(query_input)
    key, key_inverse_norm = _l2_normalize_with_inverse(key_input)
    query = query_normalized * (1.0 / math.sqrt(key_dim))
  else:
    query_normalized = query_input.astype(jnp.float32)
    key = key_input.astype(jnp.float32)
    query_inverse_norm = jnp.ones_like(query_normalized[..., :1])
    key_inverse_norm = jnp.ones_like(key[..., :1])
    query = query_normalized * (1.0 / math.sqrt(key_dim))

  cumulative_decay = _inclusive_cumsum(log_decay)
  cumulative_decay_exp = jnp.exp(cumulative_decay)
  key_beta = key * beta[..., None]
  value_beta = value * beta[..., None]
  system, intra = _decayed_pairwise_pair(key_beta, query, key, cumulative_decay)
  w_input = key_beta * cumulative_decay_exp
  combined_rhs = jnp.concatenate((value_beta, w_input), axis=-1)
  if _SOLVE_METHOD == "inverse":
    # One formation serves both the forward recompute here and the transposed
    # solve below; the transpose of the inverse is free inside the matmul.
    system_inverse = _unit_lower_inverse(system)
    solved = _solve_inverse_apply_matmul(system_inverse, combined_rhs)
  elif _SOLVE_METHOD == "substitution":
    system_inverse = None
    solved = _solve_unit_lower_triangular_substitution(system, combined_rhs)
  else:
    system_inverse = None
    solved = _solve_unit_lower_triangular_doubling(system, combined_rhs)
  u = solved[..., :value_dim]
  w = solved[..., value_dim : value_dim + key_dim]

  final_decay = cumulative_decay[..., -1, :]
  final_decay_exp = jnp.exp(final_decay)
  state_decay_exp = jnp.exp(final_decay[..., None, :] - cumulative_decay)
  query_with_decay = query * cumulative_decay_exp
  key_for_state = key * state_decay_exp
  corrected_value = u - _state_matmul(w, state)

  state_cotangent_next = state_cotangent_scratch_ref[...].astype(jnp.float32)
  state_cotangent = state_cotangent_next * final_decay_exp[..., :, None]
  final_decay_exp_cotangent = jnp.sum(state_cotangent_next * state, axis=-1)

  key_for_state_cotangent = _state_matmul(corrected_value, _transpose(state_cotangent_next))
  corrected_value_cotangent = _state_matmul(key_for_state, state_cotangent_next)
  intra_cotangent = _matmul(output_cotangent, _transpose(corrected_value))
  corrected_value_cotangent = corrected_value_cotangent + _matmul(
      _transpose(intra),
      output_cotangent,
  )
  query_with_decay_cotangent = _state_matmul(output_cotangent, _transpose(state))
  state_cotangent = state_cotangent + _state_matmul(
      _transpose(query_with_decay), output_cotangent
  )
  u_cotangent = corrected_value_cotangent
  w_cotangent = -_state_matmul(corrected_value_cotangent, _transpose(state))
  state_cotangent = state_cotangent - _state_matmul(
      _transpose(w), corrected_value_cotangent
  )
  state_cotangent_scratch_ref[...] = state_cotangent
  state_before_cotangent_ref[0, :, 0] = state_cotangent

  query_cotangent = query_with_decay_cotangent * cumulative_decay_exp
  cumulative_decay_cotangent = query_with_decay_cotangent * query_with_decay
  key_cotangent = key_for_state_cotangent * state_decay_exp
  state_decay_cotangent = key_for_state_cotangent * key_for_state
  cumulative_decay_cotangent = cumulative_decay_cotangent - state_decay_cotangent
  final_decay_cotangent = jnp.sum(state_decay_cotangent, axis=-2)
  final_decay_cotangent = final_decay_cotangent + final_decay_exp_cotangent * final_decay_exp

  def write_profile_outputs(query_bar, key_bar, value_bar, decay_bar, beta_bar):
    query_cotangent_ref[0] = query_bar.astype(query_cotangent_ref.dtype)
    key_cotangent_ref[0] = key_bar.astype(key_cotangent_ref.dtype)
    value_cotangent_ref[0] = value_bar.astype(value_cotangent_ref.dtype)
    log_decay_cotangent_ref[0] = decay_bar.astype(log_decay_cotangent_ref.dtype)
    beta_cotangent_ref[0, ..., 0] = beta_bar.astype(beta_cotangent_ref.dtype)

  if profile_stage == "reverse_state":
    write_profile_outputs(
        query_cotangent,
        key_cotangent,
        u_cotangent,
        cumulative_decay_cotangent,
        jnp.zeros_like(beta),
    )
    return

  solved_cotangent = jnp.concatenate((u_cotangent, w_cotangent), axis=-1)
  if _SOLVE_METHOD == "inverse":
    # The MXU consumes the transposed inverse directly through the contraction
    # dimension numbers; materializing the [64, 64] transpose first is a
    # relayout the kernel never needed.
    combined_rhs_cotangent = _solve_inverse_apply_matmul_tn(
        system_inverse,
        solved_cotangent,
    )
  elif _SOLVE_METHOD == "substitution":
    combined_rhs_cotangent = _solve_transposed_unit_lower_triangular_substitution(
        system,
        solved_cotangent,
    )
  else:
    combined_rhs_cotangent = _solve_transposed_unit_lower_triangular_doubling(
        system,
        solved_cotangent,
    )
  system_cotangent = -_matmul(combined_rhs_cotangent, _transpose(solved))
  system_cotangent = system_cotangent * jnp.tril(
      jnp.ones((chunk_size, chunk_size), dtype=jnp.float32),
      k=-1,
  )
  value_beta_cotangent = combined_rhs_cotangent[..., :value_dim]
  w_input_cotangent = combined_rhs_cotangent[..., value_dim : value_dim + key_dim]
  key_beta_cotangent = w_input_cotangent * cumulative_decay_exp
  cumulative_decay_cotangent = cumulative_decay_cotangent + w_input_cotangent * w_input
  if profile_stage == "solve_vjp":
    write_profile_outputs(
        query_cotangent,
        key_beta_cotangent,
        value_beta_cotangent,
        cumulative_decay_cotangent,
        jnp.zeros_like(beta),
    )
    return

  (
      key_beta_system_cotangent,
      query_pairwise_cotangent,
      key_system_cotangent,
      key_intra_cotangent,
      system_decay_cotangent,
      intra_decay_cotangent,
  ) = _decayed_pairwise_backward_pair(
      key_beta,
      query,
      key,
      cumulative_decay,
      system_cotangent,
      intra_cotangent,
  )
  key_beta_cotangent = key_beta_cotangent + key_beta_system_cotangent
  key_cotangent = key_cotangent + key_system_cotangent + key_intra_cotangent
  query_cotangent = query_cotangent + query_pairwise_cotangent
  cumulative_decay_cotangent = (
      cumulative_decay_cotangent + system_decay_cotangent + intra_decay_cotangent
  )
  cumulative_decay_cotangent = jnp.concatenate(
      (
          cumulative_decay_cotangent[..., :-1, :],
          cumulative_decay_cotangent[..., -1:, :] + final_decay_cotangent[..., None, :],
      ),
      axis=-2,
  )
  if profile_stage == "pairwise_vjp":
    write_profile_outputs(
        query_cotangent,
        key_cotangent,
        value_beta_cotangent,
        cumulative_decay_cotangent,
        jnp.zeros_like(beta),
    )
    return

  value_cotangent = value_beta_cotangent * beta[..., None]
  beta_cotangent = jnp.sum(value_beta_cotangent * value, axis=-1)
  key_cotangent = key_cotangent + key_beta_cotangent * beta[..., None]
  beta_cotangent = beta_cotangent + jnp.sum(key_beta_cotangent * key, axis=-1)
  log_decay_cotangent = _reverse_inclusive_cumsum(cumulative_decay_cotangent)

  query_normalized_cotangent = query_cotangent * (1.0 / math.sqrt(key_dim))
  if use_qk_norm:
    query_cotangent = _l2_normalize_backward(
        query_normalized_cotangent,
        query_normalized,
        query_inverse_norm,
    )
    key_cotangent = _l2_normalize_backward(
        key_cotangent,
        key,
        key_inverse_norm,
    )
  else:
    query_cotangent = query_normalized_cotangent

  query_cotangent_ref[0] = query_cotangent.astype(query_cotangent_ref.dtype)
  key_cotangent_ref[0] = key_cotangent.astype(key_cotangent_ref.dtype)
  value_cotangent_ref[0] = value_cotangent.astype(value_cotangent_ref.dtype)
  log_decay_cotangent_ref[0] = log_decay_cotangent
  beta_cotangent_ref[0, ..., 0] = beta_cotangent


def _kda_backward_stage_a_kernel(
    query_ref,
    key_ref,
    value_ref,
    log_decay_ref,
    beta_ref,
    initial_state_ref,
    previous_state_after_ref,
    output_cotangent_ref,
    final_state_cotangent_ref,
    query_partial_ref,
    key_partial_ref,
    key_beta_partial_ref,
    value_beta_cotangent_ref,
    decay_partial_ref,
    state_before_cotangent_ref,
    system_cotangent_ref,
    intra_cotangent_ref,
    final_decay_cotangent_ref,
    state_cotangent_scratch_ref,
    *,
    chunk_size: int,
    key_dim: int,
    value_dim: int,
    num_chunks: int,
    use_qk_norm: bool,
    chunk_axis: int = 2,
    extras_mode: str = "real",
):
  """The reverse-ordered half of the split backward.

  Identical recompute-and-solve mathematics to `_kda_fused_backward_kernel`
  up to the solve VJP, but instead of continuing into the pairwise VJP —
  whose integrated layout assignment demands a sublane-gather relayout that
  TPU v4 lacks — it exports the per-chunk intermediates the parallel stage-B
  epilogue kernel needs, all in FP32.
  """
  reverse_chunk_index = pl.program_id(chunk_axis)
  chunk_index = num_chunks - 1 - reverse_chunk_index
  query_input = query_ref[0]
  key_input = key_ref[0]
  value = value_ref[0].astype(jnp.float32)
  log_decay = log_decay_ref[0].astype(jnp.float32)
  beta = beta_ref[0][..., 0].astype(jnp.float32)
  output_cotangent = output_cotangent_ref[0].astype(jnp.float32)

  @pl.when(reverse_chunk_index == 0)
  def _initialize_state_cotangent():
    state_cotangent_scratch_ref[...] = final_state_cotangent_ref[0].astype(jnp.float32)

  state = lax.cond(
      chunk_index == 0,
      lambda: initial_state_ref[0].astype(jnp.float32),
      lambda: previous_state_after_ref[0, :, 0].astype(jnp.float32),
  )

  if use_qk_norm:
    query_normalized, _ = _l2_normalize_with_inverse(query_input)
    key, _ = _l2_normalize_with_inverse(key_input)
    query = query_normalized * (1.0 / math.sqrt(key_dim))
  else:
    query = query_input.astype(jnp.float32) * (1.0 / math.sqrt(key_dim))
    key = key_input.astype(jnp.float32)

  cumulative_decay = _inclusive_cumsum(log_decay)
  cumulative_decay_exp = jnp.exp(cumulative_decay)
  key_beta = key * beta[..., None]
  value_beta = value * beta[..., None]
  system, intra = _decayed_pairwise_pair(key_beta, query, key, cumulative_decay)
  w_input = key_beta * cumulative_decay_exp
  combined_rhs = jnp.concatenate((value_beta, w_input), axis=-1)
  if _SOLVE_METHOD == "inverse":
    system_inverse = _unit_lower_inverse(system)
    solved = _solve_inverse_apply_matmul(system_inverse, combined_rhs)
  elif _SOLVE_METHOD == "substitution":
    system_inverse = None
    solved = _solve_unit_lower_triangular_substitution(system, combined_rhs)
  else:
    system_inverse = None
    solved = _solve_unit_lower_triangular_doubling(system, combined_rhs)
  u = solved[..., :value_dim]
  w = solved[..., value_dim : value_dim + key_dim]

  final_decay = cumulative_decay[..., -1, :]
  final_decay_exp = jnp.exp(final_decay)
  state_decay_exp = jnp.exp(final_decay[..., None, :] - cumulative_decay)
  query_with_decay = query * cumulative_decay_exp
  key_for_state = key * state_decay_exp
  corrected_value = u - _state_matmul(w, state)

  state_cotangent_next = state_cotangent_scratch_ref[...].astype(jnp.float32)
  state_cotangent = state_cotangent_next * final_decay_exp[..., :, None]
  # Both final-decay cotangent contributions must come from sublane
  # reductions (result on lanes): mixing a lane reduction with a sublane
  # reduction produces two 1-D layouts whose sum can only be stored through
  # the sublane gather v4 lacks. The full-tile [K, V] transpose is supported.
  final_decay_exp_cotangent = jnp.sum(
      _transpose(state_cotangent_next * state), axis=-2
  )

  key_for_state_cotangent = _state_matmul(corrected_value, _transpose(state_cotangent_next))
  corrected_value_cotangent = _state_matmul(key_for_state, state_cotangent_next)
  intra_cotangent = _matmul(output_cotangent, _transpose(corrected_value))
  corrected_value_cotangent = corrected_value_cotangent + _matmul(
      _transpose(intra),
      output_cotangent,
  )
  query_with_decay_cotangent = _state_matmul(output_cotangent, _transpose(state))
  state_cotangent = state_cotangent + _state_matmul(
      _transpose(query_with_decay), output_cotangent
  )
  u_cotangent = corrected_value_cotangent
  w_cotangent = -_state_matmul(corrected_value_cotangent, _transpose(state))
  state_cotangent = state_cotangent - _state_matmul(
      _transpose(w), corrected_value_cotangent
  )
  state_cotangent_scratch_ref[...] = state_cotangent
  state_before_cotangent_ref[0, :, 0] = state_cotangent

  query_cotangent = query_with_decay_cotangent * cumulative_decay_exp
  cumulative_decay_cotangent = query_with_decay_cotangent * query_with_decay
  key_cotangent = key_for_state_cotangent * state_decay_exp
  state_decay_cotangent = key_for_state_cotangent * key_for_state
  cumulative_decay_cotangent = cumulative_decay_cotangent - state_decay_cotangent
  final_decay_cotangent = jnp.sum(state_decay_cotangent, axis=-2)
  final_decay_cotangent = final_decay_cotangent + final_decay_exp_cotangent * final_decay_exp

  solved_cotangent = jnp.concatenate((u_cotangent, w_cotangent), axis=-1)
  if _SOLVE_METHOD == "inverse":
    # The MXU consumes the transposed inverse directly through the contraction
    # dimension numbers; materializing the [64, 64] transpose first is a
    # relayout the kernel never needed.
    combined_rhs_cotangent = _solve_inverse_apply_matmul_tn(
        system_inverse,
        solved_cotangent,
    )
  elif _SOLVE_METHOD == "substitution":
    combined_rhs_cotangent = _solve_transposed_unit_lower_triangular_substitution(
        system,
        solved_cotangent,
    )
  else:
    combined_rhs_cotangent = _solve_transposed_unit_lower_triangular_doubling(
        system,
        solved_cotangent,
    )
  system_cotangent = -_matmul(combined_rhs_cotangent, _transpose(solved))
  system_cotangent = system_cotangent * jnp.tril(
      jnp.ones((chunk_size, chunk_size), dtype=jnp.float32),
      k=-1,
  )
  value_beta_cotangent = combined_rhs_cotangent[..., :value_dim]
  w_input_cotangent = combined_rhs_cotangent[..., value_dim : value_dim + key_dim]
  key_beta_cotangent = w_input_cotangent * cumulative_decay_exp
  cumulative_decay_cotangent = cumulative_decay_cotangent + w_input_cotangent * w_input

  query_partial_ref[0] = query_cotangent
  key_partial_ref[0] = key_cotangent
  value_beta_cotangent_ref[0] = value_beta_cotangent
  decay_partial_ref[0] = cumulative_decay_cotangent
  if extras_mode == "none":
    return
  wants = extras_mode.split("+") if extras_mode != "real" else [
      "kb", "sys", "intra", "fdc",
  ]

  def pick(tag, value, zero_shape):
    if extras_mode == "zeros" or (extras_mode != "real" and tag not in wants):
      return jnp.zeros(zero_shape, jnp.float32)
    return value

  key_beta_partial_ref[0] = pick("kb", key_beta_cotangent, key_beta_cotangent.shape)
  system_cotangent_ref[0, :, 0] = pick("sys", system_cotangent, system_cotangent.shape)
  intra_cotangent_ref[0, :, 0] = pick("intra", intra_cotangent, intra_cotangent.shape)
  final_decay_cotangent_ref[0, :, 0, 0] = pick(
      "fdc", final_decay_cotangent, final_decay_cotangent.shape
  )


def _kda_backward_stage_b_kernel(
    query_ref,
    key_ref,
    value_ref,
    beta_ref,
    query_partial_ref,
    key_partial_ref,
    key_beta_partial_ref,
    value_beta_cotangent_ref,
    decay_partial_ref,
    system_cotangent_ref,
    intra_cotangent_ref,
    log_decay_ref,
    final_decay_cotangent_ref,
    query_cotangent_ref,
    key_cotangent_ref,
    value_cotangent_ref,
    log_decay_cotangent_ref,
    beta_cotangent_ref,
    *,
    key_dim: int,
    use_qk_norm: bool,
    inputs_mode: str = "real",
):
  """The chunk-parallel half of the split backward.

  Consumes stage A's per-chunk exports through fresh HBM refs — which is
  precisely what lets the pairwise VJP compile on v4: with ref-supplied
  operands Mosaic assigns clean layouts and never needs the sublane-gather
  relayout the integrated kernel required. No state is carried across
  chunks, so the grid is embarrassingly parallel.
  """
  query_input = query_ref[0]
  key_input = key_ref[0]
  value = value_ref[0].astype(jnp.float32)
  beta = beta_ref[0][..., 0].astype(jnp.float32)
  streams, chunk, _ = query_input.shape
  flags = set(inputs_mode.split("+"))
  if "zeros_all" in flags:
    cumulative_decay = jnp.zeros((streams, chunk, key_dim), jnp.float32)
  else:
    # Bitwise-identical recompute of stage A's chunk-local value: the same
    # fp32 block through the same _inclusive_cumsum lowering, replacing a
    # 67 MB fp32 export round trip per layer.
    cumulative_decay = _inclusive_cumsum(log_decay_ref[0].astype(jnp.float32))
  if "zeros_all" in flags or "zeros_squares" in flags:
    system_cotangent_in = jnp.zeros((streams, chunk, chunk), jnp.float32)
    intra_cotangent_in = jnp.zeros((streams, chunk, chunk), jnp.float32)
    final_decay_cotangent = jnp.zeros((streams, key_dim), jnp.float32)
  else:
    system_cotangent_in = system_cotangent_ref[0, :, 0]
    intra_cotangent_in = intra_cotangent_ref[0, :, 0]
    final_decay_cotangent = final_decay_cotangent_ref[0, :, 0, 0]
  use_qk_norm = use_qk_norm and "raw_qk" not in flags

  if use_qk_norm:
    query_normalized, query_inverse_norm = _l2_normalize_with_inverse(query_input)
    key, key_inverse_norm = _l2_normalize_with_inverse(key_input)
    query = query_normalized * (1.0 / math.sqrt(key_dim))
  else:
    query_normalized = query_input.astype(jnp.float32)
    key = key_input.astype(jnp.float32)
    query_inverse_norm = jnp.ones_like(query_normalized[..., :1])
    key_inverse_norm = jnp.ones_like(key[..., :1])
    query = query_normalized * (1.0 / math.sqrt(key_dim))
  key_beta = key * beta[..., None]

  if "no_system" not in flags and "no_intra" not in flags:
    # The production path: both VJPs share the right operand, the decay, and
    # the per-block exponentials, so they run through one merged block pass.
    (
        key_beta_system_cotangent,
        query_pairwise_cotangent,
        key_system_cotangent,
        key_intra_cotangent,
        system_decay_cotangent,
        intra_decay_cotangent,
    ) = _decayed_pairwise_backward_pair(
        key_beta,
        query,
        key,
        cumulative_decay,
        system_cotangent_in,
        intra_cotangent_in,
    )
  else:
    if "no_system" in flags:
      key_beta_system_cotangent = jnp.zeros_like(key_beta)
      key_system_cotangent = jnp.zeros_like(key)
      system_decay_cotangent = jnp.zeros_like(cumulative_decay)
    else:
      (
          key_beta_system_cotangent,
          key_system_cotangent,
          system_decay_cotangent,
      ) = _decayed_pairwise_backward(
          key_beta,
          key,
          cumulative_decay,
          system_cotangent_in,
          include_diagonal=False,
      )
    if "no_intra" in flags:
      query_pairwise_cotangent = jnp.zeros_like(query)
      key_intra_cotangent = jnp.zeros_like(key)
      intra_decay_cotangent = jnp.zeros_like(cumulative_decay)
    else:
      (
          query_pairwise_cotangent,
          key_intra_cotangent,
          intra_decay_cotangent,
      ) = _decayed_pairwise_backward(
          query,
          key,
          cumulative_decay,
          intra_cotangent_in,
          include_diagonal=True,
      )
  key_beta_cotangent = key_beta_partial_ref[0] + key_beta_system_cotangent
  key_cotangent = key_partial_ref[0] + key_system_cotangent + key_intra_cotangent
  query_cotangent = query_partial_ref[0] + query_pairwise_cotangent
  cumulative_decay_cotangent = (
      decay_partial_ref[0] + system_decay_cotangent + intra_decay_cotangent
  )
  cumulative_decay_cotangent = jnp.concatenate(
      (
          cumulative_decay_cotangent[..., :-1, :],
          cumulative_decay_cotangent[..., -1:, :]
          + final_decay_cotangent[..., None, :],
      ),
      axis=-2,
  )

  if "no_epilogue" in flags:
    query_cotangent_ref[0] = query_cotangent.astype(query_cotangent_ref.dtype)
    key_cotangent_ref[0] = key_cotangent.astype(key_cotangent_ref.dtype)
    value_cotangent_ref[0] = value_beta_cotangent_ref[0].astype(
        value_cotangent_ref.dtype
    )
    log_decay_cotangent_ref[0] = cumulative_decay_cotangent
    beta_cotangent_ref[0, ..., 0] = jnp.zeros((streams, chunk), jnp.float32)
    return

  value_beta_cotangent = value_beta_cotangent_ref[0]
  value_cotangent = value_beta_cotangent * beta[..., None]
  beta_cotangent = jnp.sum(value_beta_cotangent * value, axis=-1)
  key_cotangent = key_cotangent + key_beta_cotangent * beta[..., None]
  beta_cotangent = beta_cotangent + jnp.sum(key_beta_cotangent * key, axis=-1)
  log_decay_cotangent = _reverse_inclusive_cumsum(cumulative_decay_cotangent)

  query_normalized_cotangent = query_cotangent * (1.0 / math.sqrt(key_dim))
  if use_qk_norm:
    query_cotangent = _l2_normalize_backward(
        query_normalized_cotangent,
        query_normalized,
        query_inverse_norm,
    )
    key_cotangent = _l2_normalize_backward(
        key_cotangent,
        key,
        key_inverse_norm,
    )
  else:
    query_cotangent = query_normalized_cotangent

  query_cotangent_ref[0] = query_cotangent.astype(query_cotangent_ref.dtype)
  key_cotangent_ref[0] = key_cotangent.astype(key_cotangent_ref.dtype)
  value_cotangent_ref[0] = value_cotangent.astype(value_cotangent_ref.dtype)
  log_decay_cotangent_ref[0] = log_decay_cotangent
  beta_cotangent_ref[0, ..., 0] = beta_cotangent


@functools.partial(
    jax.jit,
    static_argnames=(
        "chunk_size",
        "use_qk_norm",
        "solve_method",
        "profile_stage",
        "streams_per_program",
    ),
)
def _pallas_kda_fused_v4_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    log_decay: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
    *,
    chunk_size: int = 64,
    use_qk_norm: bool = True,
    solve_method: str = _SOLVE_METHOD,
    profile_stage: str = "full",
    streams_per_program: int = _DEFAULT_STREAMS_PER_PROGRAM,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs the fixed-layout fused KDA forward on TPU.

  Inputs use ``[B,T,H,D]`` layout. The returned state history contains the
  state *after* each chunk as ``[B,NC,H,K,V]``. The public final state is the
  final history entry.
  """
  if jax.default_backend() != "tpu" and not os.environ.get("YXTPU_KDA_INTERPRET"):
    raise RuntimeError("pallas_kda_fused_v4_forward requires a TPU backend")
  if query.shape != key.shape or query.ndim != 4:
    raise ValueError(f"expected matching [B,T,H,K] Q/K, got {query.shape}, {key.shape}")
  batch, sequence_length, heads, key_dim = query.shape
  value_dim = value.shape[-1]
  if value.shape[:3] != (batch, sequence_length, heads):
    raise ValueError(f"incompatible value shape: {value.shape}")
  if log_decay.shape != query.shape:
    raise ValueError(f"incompatible log-decay shape: {log_decay.shape}")
  if beta.shape != (batch, sequence_length, heads):
    raise ValueError(f"incompatible beta shape: {beta.shape}")
  if initial_state.shape != (batch, heads, key_dim, value_dim):
    raise ValueError(f"incompatible initial state shape: {initial_state.shape}")
  if sequence_length % chunk_size:
    raise ValueError(
        f"sequence length {sequence_length} must be divisible by chunk size {chunk_size}"
    )
  if chunk_size != 64 or key_dim != 128 or value_dim != 128:
    raise ValueError(
        "the first production kernel is specialized to chunk=64 and K=V=128, "
        f"got chunk={chunk_size}, K={key_dim}, V={value_dim}"
    )
  if solve_method not in ("blocked", "doubling", "substitution", "inverse"):
    raise ValueError(
        f"solve_method must be blocked, doubling, substitution, or inverse, got {solve_method}"
    )
  if profile_stage not in ("preprocess", "pairwise", "solve", "full"):
    raise ValueError(f"unknown forward profile stage: {profile_stage}")

  streams = batch * heads
  streams_per_program = math.gcd(streams, streams_per_program)
  num_chunks = sequence_length // chunk_size
  stream_groups = streams // streams_per_program
  qkv_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, chunk_size, key_dim),
      index_map=lambda batch_group, head_group, chunk_index: (
          batch_group,
          head_group,
          chunk_index,
          0,
      ),
  )
  value_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, chunk_size, value_dim),
      index_map=lambda batch_group, head_group, chunk_index: (
          batch_group,
          head_group,
          chunk_index,
          0,
      ),
  )
  beta_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, chunk_size, 1),
      index_map=lambda batch_group, head_group, chunk_index: (
          batch_group,
          head_group,
          chunk_index,
          0,
      ),
  )
  initial_state_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, key_dim, value_dim),
      index_map=lambda batch_group, head_group, chunk_index: (
          batch_group,
          head_group,
          0,
          0,
      ),
  )
  state_history_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, 1, key_dim, value_dim),
      index_map=lambda batch_group, head_group, chunk_index: (
          batch_group,
          head_group,
          chunk_index,
          0,
          0,
      ),
  )
  output_shape = jax.ShapeDtypeStruct(
      (1, streams, sequence_length, value_dim),
      value.dtype,
  )
  state_history_shape = jax.ShapeDtypeStruct(
      (1, streams, num_chunks, key_dim, value_dim),
      jnp.float32,
  )
  output, state_history = pl.pallas_call(
      functools.partial(
          _kda_fused_forward_kernel,
          chunk_size=chunk_size,
          key_dim=key_dim,
          value_dim=value_dim,
          use_qk_norm=use_qk_norm,
          solve_method=solve_method,
          profile_stage=profile_stage,
          chunk_axis=2,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(1, stream_groups, num_chunks),
          in_specs=(
              qkv_spec,
              qkv_spec,
              value_spec,
              qkv_spec,
              beta_spec,
              initial_state_spec,
          ),
          out_specs=(
              value_spec,
              state_history_spec,
          ),
          scratch_shapes=(
              pltpu.VMEM((streams_per_program, key_dim, value_dim), jnp.float32),
          ),
      ),
      out_shape=(output_shape, state_history_shape),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      name=f"kda_fused_forward_{solve_method}_{profile_stage}_s{streams_per_program}",
  )(
      query.transpose(0, 2, 1, 3).reshape(1, streams, sequence_length, key_dim),
      key.transpose(0, 2, 1, 3).reshape(1, streams, sequence_length, key_dim),
      value.transpose(0, 2, 1, 3).reshape(1, streams, sequence_length, value_dim),
      log_decay.astype(jnp.float32)
      .transpose(0, 2, 1, 3)
      .reshape(1, streams, sequence_length, key_dim),
      beta.astype(jnp.float32).transpose(0, 2, 1).reshape(1, streams, sequence_length, 1),
      initial_state.astype(jnp.float32).reshape(1, streams, key_dim, value_dim),
  )
  output = output.reshape(batch, heads, sequence_length, value_dim).transpose(0, 2, 1, 3)
  state_history = state_history.reshape(batch, heads, num_chunks, key_dim, value_dim)
  final_state = state_history[:, :, -1]
  return output, final_state, state_history.transpose(0, 2, 1, 3, 4)


@functools.partial(
    jax.jit,
    static_argnames=(
        "chunk_size",
        "use_qk_norm",
        "profile_stage",
        "streams_per_program",
    ),
)
def _pallas_kda_fused_v4_backward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    log_decay: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
    state_history: jax.Array,
    output_cotangent: jax.Array,
    final_state_cotangent: jax.Array,
    *,
    chunk_size: int = 64,
    use_qk_norm: bool = True,
    profile_stage: str = "full",
    streams_per_program: int = _DEFAULT_STREAMS_PER_PROGRAM,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Runs the fused reverse chunk stream and returns all six input gradients."""
  batch, sequence_length, heads, key_dim = query.shape
  value_dim = value.shape[-1]
  num_chunks = sequence_length // chunk_size
  expected_history_shape = (batch, num_chunks, heads, key_dim, value_dim)
  if state_history.shape != expected_history_shape:
    raise ValueError(f"expected state history {expected_history_shape}, got {state_history.shape}")
  if profile_stage not in ("reverse_state", "solve_vjp", "pairwise_vjp", "full"):
    raise ValueError(f"unknown backward profile stage: {profile_stage}")

  streams = batch * heads
  streams_per_program = math.gcd(streams, streams_per_program)
  stream_groups = streams // streams_per_program
  reverse_qkv_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, chunk_size, key_dim),
      index_map=lambda batch_group, head_group, reverse_chunk_index: (
          batch_group,
          head_group,
          num_chunks - 1 - reverse_chunk_index,
          0,
      ),
  )
  reverse_value_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, chunk_size, value_dim),
      index_map=lambda batch_group, head_group, reverse_chunk_index: (
          batch_group,
          head_group,
          num_chunks - 1 - reverse_chunk_index,
          0,
      ),
  )
  reverse_beta_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, chunk_size, 1),
      index_map=lambda batch_group, head_group, reverse_chunk_index: (
          batch_group,
          head_group,
          num_chunks - 1 - reverse_chunk_index,
          0,
      ),
  )
  state_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, key_dim, value_dim),
      index_map=lambda batch_group, head_group, reverse_chunk_index: (
          batch_group,
          head_group,
          0,
          0,
      ),
  )
  previous_state_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, 1, key_dim, value_dim),
      index_map=lambda batch_group, head_group, reverse_chunk_index: (
          batch_group,
          head_group,
          jnp.maximum(num_chunks - 2 - reverse_chunk_index, 0),
          0,
          0,
      ),
  )
  state_before_cotangent_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, 1, key_dim, value_dim),
      index_map=lambda batch_group, head_group, reverse_chunk_index: (
          batch_group,
          head_group,
          num_chunks - 1 - reverse_chunk_index,
          0,
          0,
      ),
  )

  stream_shape = (1, streams, sequence_length)
  query_t = query.transpose(0, 2, 1, 3).reshape(*stream_shape, key_dim)
  key_t = key.transpose(0, 2, 1, 3).reshape(*stream_shape, key_dim)
  value_t = value.transpose(0, 2, 1, 3).reshape(*stream_shape, value_dim)
  log_decay_t = (
      log_decay.astype(jnp.float32).transpose(0, 2, 1, 3).reshape(*stream_shape, key_dim)
  )
  beta_t = beta.astype(jnp.float32).transpose(0, 2, 1).reshape(*stream_shape, 1)
  state_history_t = state_history.transpose(0, 2, 1, 3, 4).reshape(
      1, streams, num_chunks, key_dim, value_dim
  )
  output_cotangent_t = output_cotangent.transpose(0, 2, 1, 3).reshape(
      *stream_shape, value_dim
  )
  initial_state_t = initial_state.astype(jnp.float32).reshape(
      1, streams, key_dim, value_dim
  )
  final_state_cotangent_t = final_state_cotangent.astype(jnp.float32).reshape(
      1, streams, key_dim, value_dim
  )

  query_cotangent_shape = jax.ShapeDtypeStruct(query_t.shape, query.dtype)
  key_cotangent_shape = jax.ShapeDtypeStruct(key_t.shape, key.dtype)
  value_cotangent_shape = jax.ShapeDtypeStruct(value_t.shape, value.dtype)
  log_decay_cotangent_shape = jax.ShapeDtypeStruct(log_decay_t.shape, log_decay.dtype)
  beta_cotangent_shape = jax.ShapeDtypeStruct(beta_t.shape, beta.dtype)
  state_before_cotangent_shape = jax.ShapeDtypeStruct(
      state_history_t.shape,
      jnp.float32,
  )
  (
      query_cotangent_t,
      key_cotangent_t,
      value_cotangent_t,
      log_decay_cotangent_t,
      beta_cotangent_t,
      state_before_cotangent_t,
  ) = pl.pallas_call(
      functools.partial(
          _kda_fused_backward_kernel,
          chunk_size=chunk_size,
          key_dim=key_dim,
          value_dim=value_dim,
          num_chunks=num_chunks,
          use_qk_norm=use_qk_norm,
          profile_stage=profile_stage,
          chunk_axis=2,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(1, stream_groups, num_chunks),
          in_specs=(
              reverse_qkv_spec,
              reverse_qkv_spec,
              reverse_value_spec,
              reverse_qkv_spec,
              reverse_beta_spec,
              state_spec,
              previous_state_spec,
              reverse_value_spec,
              state_spec,
          ),
          out_specs=(
              reverse_qkv_spec,
              reverse_qkv_spec,
              reverse_value_spec,
              reverse_qkv_spec,
              reverse_beta_spec,
              state_before_cotangent_spec,
          ),
          scratch_shapes=(
              pltpu.VMEM((streams_per_program, key_dim, value_dim), jnp.float32),
          ),
      ),
      out_shape=(
          query_cotangent_shape,
          key_cotangent_shape,
          value_cotangent_shape,
          log_decay_cotangent_shape,
          beta_cotangent_shape,
          state_before_cotangent_shape,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      name=f"kda_fused_backward_{profile_stage}",
  )(
      query_t,
      key_t,
      value_t,
      log_decay_t,
      beta_t,
      initial_state_t,
      state_history_t,
      output_cotangent_t,
      final_state_cotangent_t,
  )

  def unstream(values, channels):
    return values.reshape(batch, heads, sequence_length, channels).transpose(0, 2, 1, 3)

  return (
      unstream(query_cotangent_t, key_dim),
      unstream(key_cotangent_t, key_dim),
      unstream(value_cotangent_t, value_dim),
      unstream(log_decay_cotangent_t, key_dim),
      unstream(beta_cotangent_t, 1)[..., 0],
      state_before_cotangent_t.reshape(batch, heads, num_chunks, key_dim, value_dim)[
          :, :, 0
      ],
  )


@functools.partial(
    jax.jit,
    static_argnames=("chunk_size", "use_qk_norm", "streams_per_program", "probe_stage"),
)
def _pallas_kda_fused_v4_backward_split(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    log_decay: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
    state_history: jax.Array,
    output_cotangent: jax.Array,
    final_state_cotangent: jax.Array,
    *,
    chunk_size: int = 64,
    use_qk_norm: bool = True,
    streams_per_program: int = _DEFAULT_STREAMS_PER_PROGRAM,
    probe_stage: str = "both",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Two-kernel backward for TPU v4: reverse solve pass, then parallel
  pairwise epilogue. Splitting at the solve/pairwise boundary gives the
  pairwise VJP fresh ref layouts, sidestepping the sublane-gather relayout
  the integrated backward needs on v4. ``probe_stage`` ("a"/"b"/"both")
  exists only for compile bisection and must stay "both" in training."""
  batch, sequence_length, heads, key_dim = query.shape
  value_dim = value.shape[-1]
  num_chunks = sequence_length // chunk_size

  extras_mode = probe_stage[2:] if probe_stage.startswith("a_") else "real"
  inputs_mode = probe_stage[2:] if probe_stage.startswith("b_") else "real"

  streams = batch * heads
  streams_per_program = math.gcd(streams, streams_per_program)
  stream_groups = streams // streams_per_program

  def chunk_spec(channels, reverse):
    return pl.BlockSpec(
        block_shape=(1, streams_per_program, chunk_size, channels),
        index_map=lambda bg, hg, ci: (
            bg,
            hg,
            (num_chunks - 1 - ci) if reverse else ci,
            0,
        ),
    )

  def square_spec(reverse):
    return pl.BlockSpec(
        block_shape=(1, streams_per_program, 1, chunk_size, chunk_size),
        index_map=lambda bg, hg, ci: (
            bg,
            hg,
            (num_chunks - 1 - ci) if reverse else ci,
            0,
            0,
        ),
    )

  def row_spec(channels, reverse):
    # The trailing axis order is load-bearing on v4: channels must stay on
    # the lane (minor) dimension. A [.., channels, 1] block would demand a
    # minor-dimension relayout, which lowers to the unsupported sublane
    # gather in both the producing and consuming kernels.
    return pl.BlockSpec(
        block_shape=(1, streams_per_program, 1, 1, channels),
        index_map=lambda bg, hg, ci: (
            bg,
            hg,
            (num_chunks - 1 - ci) if reverse else ci,
            0,
            0,
        ),
    )

  state_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, key_dim, value_dim),
      index_map=lambda bg, hg, ci: (bg, hg, 0, 0),
  )
  previous_state_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, 1, key_dim, value_dim),
      index_map=lambda bg, hg, ci: (
          bg,
          hg,
          jnp.maximum(num_chunks - 2 - ci, 0),
          0,
          0,
      ),
  )
  # Only the chunk-0 slot (the initial-state cotangent) is ever consumed, so
  # the export is a single revisited block: the kernel overwrites it every
  # reverse step and Mosaic flushes once per stream group with the last
  # write, which is chunk 0.
  state_before_cotangent_spec = pl.BlockSpec(
      block_shape=(1, streams_per_program, 1, key_dim, value_dim),
      index_map=lambda bg, hg, ci: (bg, hg, 0, 0, 0),
  )

  stream_shape = (1, streams, sequence_length)
  query_t = query.transpose(0, 2, 1, 3).reshape(*stream_shape, key_dim)
  key_t = key.transpose(0, 2, 1, 3).reshape(*stream_shape, key_dim)
  value_t = value.transpose(0, 2, 1, 3).reshape(*stream_shape, value_dim)
  log_decay_t = (
      log_decay.astype(jnp.float32).transpose(0, 2, 1, 3).reshape(*stream_shape, key_dim)
  )
  beta_t = beta.astype(jnp.float32).transpose(0, 2, 1).reshape(*stream_shape, 1)
  state_history_t = state_history.transpose(0, 2, 1, 3, 4).reshape(
      1, streams, num_chunks, key_dim, value_dim
  )
  output_cotangent_t = output_cotangent.transpose(0, 2, 1, 3).reshape(
      *stream_shape, value_dim
  )
  initial_state_t = initial_state.astype(jnp.float32).reshape(
      1, streams, key_dim, value_dim
  )
  final_state_cotangent_t = final_state_cotangent.astype(jnp.float32).reshape(
      1, streams, key_dim, value_dim
  )

  def f32(shape):
    return jax.ShapeDtypeStruct(shape, jnp.float32)

  chunk_qkv_shape = (*stream_shape, key_dim)
  chunk_value_shape = (*stream_shape, value_dim)
  square_shape = (1, streams, num_chunks, chunk_size, chunk_size)
  row_shape = (1, streams, num_chunks, 1, key_dim)

  (
      query_partial_t,
      key_partial_t,
      key_beta_partial_t,
      value_beta_cotangent_t,
      decay_partial_t,
      state_before_cotangent_t,
      system_cotangent_t,
      intra_cotangent_t,
      final_decay_cotangent_t,
  ) = pl.pallas_call(
      functools.partial(
          _kda_backward_stage_a_kernel,
          chunk_size=chunk_size,
          key_dim=key_dim,
          value_dim=value_dim,
          num_chunks=num_chunks,
          use_qk_norm=use_qk_norm,
          chunk_axis=2,
          extras_mode=extras_mode,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(1, stream_groups, num_chunks),
          in_specs=(
              chunk_spec(key_dim, True),
              chunk_spec(key_dim, True),
              chunk_spec(value_dim, True),
              chunk_spec(key_dim, True),
              chunk_spec(1, True),
              state_spec,
              previous_state_spec,
              chunk_spec(value_dim, True),
              state_spec,
          ),
          out_specs=(
              chunk_spec(key_dim, True),
              chunk_spec(key_dim, True),
              chunk_spec(key_dim, True),
              chunk_spec(value_dim, True),
              chunk_spec(key_dim, True),
              state_before_cotangent_spec,
              square_spec(True),
              square_spec(True),
              row_spec(key_dim, True),
          ),
          scratch_shapes=(
              pltpu.VMEM((streams_per_program, key_dim, value_dim), jnp.float32),
          ),
      ),
      out_shape=(
          f32(chunk_qkv_shape),
          f32(chunk_qkv_shape),
          f32(chunk_qkv_shape),
          f32(chunk_value_shape),
          f32(chunk_qkv_shape),
          f32((1, streams, 1, key_dim, value_dim)),
          f32(square_shape),
          f32(square_shape),
          f32(row_shape),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      name="kda_backward_stage_a",
  )(
      query_t,
      key_t,
      value_t,
      log_decay_t,
      beta_t,
      initial_state_t,
      state_history_t,
      output_cotangent_t,
      final_state_cotangent_t,
  )

  if probe_stage == "a" or probe_stage.startswith("a_"):
    return (
        query_partial_t,
        key_partial_t,
        key_beta_partial_t,
        value_beta_cotangent_t,
        decay_partial_t,
        state_before_cotangent_t,
    )
  if probe_stage == "b" or probe_stage.startswith("b_"):
    query_partial_t = jnp.zeros_like(query_partial_t)
    key_partial_t = jnp.zeros_like(key_partial_t)
    key_beta_partial_t = jnp.zeros_like(key_beta_partial_t)
    value_beta_cotangent_t = jnp.zeros_like(value_beta_cotangent_t)
    decay_partial_t = jnp.zeros_like(decay_partial_t)
    system_cotangent_t = jnp.zeros_like(system_cotangent_t)
    intra_cotangent_t = jnp.zeros_like(intra_cotangent_t)
    final_decay_cotangent_t = jnp.zeros_like(final_decay_cotangent_t)

  (
      query_cotangent_t,
      key_cotangent_t,
      value_cotangent_t,
      log_decay_cotangent_t,
      beta_cotangent_t,
  ) = pl.pallas_call(
      functools.partial(
          _kda_backward_stage_b_kernel,
          key_dim=key_dim,
          use_qk_norm=use_qk_norm,
          inputs_mode=inputs_mode,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(1, stream_groups, num_chunks),
          in_specs=(
              chunk_spec(key_dim, False),
              chunk_spec(key_dim, False),
              chunk_spec(value_dim, False),
              chunk_spec(1, False),
              chunk_spec(key_dim, False),
              chunk_spec(key_dim, False),
              chunk_spec(key_dim, False),
              chunk_spec(value_dim, False),
              chunk_spec(key_dim, False),
              square_spec(False),
              square_spec(False),
              chunk_spec(key_dim, False),
              row_spec(key_dim, False),
          ),
          out_specs=(
              chunk_spec(key_dim, False),
              chunk_spec(key_dim, False),
              chunk_spec(value_dim, False),
              chunk_spec(key_dim, False),
              chunk_spec(1, False),
          ),
          scratch_shapes=(),
      ),
      out_shape=(
          jax.ShapeDtypeStruct(chunk_qkv_shape, query.dtype),
          jax.ShapeDtypeStruct(chunk_qkv_shape, key.dtype),
          jax.ShapeDtypeStruct(chunk_value_shape, value.dtype),
          jax.ShapeDtypeStruct(chunk_qkv_shape, log_decay.dtype),
          jax.ShapeDtypeStruct((*stream_shape, 1), beta.dtype),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=True,
      ),
      name="kda_backward_stage_b",
  )(
      query_t,
      key_t,
      value_t,
      beta_t,
      query_partial_t,
      key_partial_t,
      key_beta_partial_t,
      value_beta_cotangent_t,
      decay_partial_t,
      system_cotangent_t,
      intra_cotangent_t,
      log_decay_t,
      final_decay_cotangent_t,
  )

  def unstream(values, channels):
    return values.reshape(batch, heads, sequence_length, channels).transpose(0, 2, 1, 3)

  return (
      unstream(query_cotangent_t, key_dim),
      unstream(key_cotangent_t, key_dim),
      unstream(value_cotangent_t, value_dim),
      unstream(log_decay_cotangent_t, key_dim),
      unstream(beta_cotangent_t, 1)[..., 0],
      state_before_cotangent_t.reshape(batch, heads, key_dim, value_dim),
  )


@jax.custom_vjp
def pallas_kda_fused_v4(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    log_decay: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Differentiable fixed-shape fused KDA operation for TPU training."""
  output, final_state, _ = _pallas_kda_fused_v4_forward(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      chunk_size=64,
      use_qk_norm=True,
      solve_method=_SOLVE_METHOD,
  )
  return output, final_state


def _pallas_kda_fused_v4_fwd(query, key, value, log_decay, beta, initial_state):
  output, final_state, state_history = _pallas_kda_fused_v4_forward(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      chunk_size=64,
      use_qk_norm=True,
      solve_method=_SOLVE_METHOD,
  )
  # Under the cycle remat these two names are the whole consumer set of the
  # forward pallas call, so a policy that saves both leaves the backward's
  # recompute of this sequential walk dead (a zero-output shard_map husk in
  # the jaxpr that XLA's HLO DCE then removes).
  output = checkpoint_name(output, "kda_out")
  state_history = checkpoint_name(state_history, "kda_state_history")
  return (output, final_state), (
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      state_history,
  )


def _pallas_kda_fused_v4_bwd(residual, output_cotangents):
  query, key, value, log_decay, beta, initial_state, state_history = residual
  output_cotangent, final_state_cotangent = output_cotangents
  return _pallas_kda_fused_v4_backward_split(
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      state_history,
      output_cotangent,
      final_state_cotangent,
      chunk_size=64,
      use_qk_norm=True,
  )


pallas_kda_fused_v4.defvjp(
    _pallas_kda_fused_v4_fwd,
    _pallas_kda_fused_v4_bwd,
)
