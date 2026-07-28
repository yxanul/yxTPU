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

"""Data-parallel distribution of Muon's Newton-Schulz orthogonalization.

``optax.contrib.muon`` computes Newton-Schulz on every replica: under pure
data parallelism each of the 32 v4 chips performs the identical ~4.8 TFLOP of
NS work per step (~14% of the model's own FLOPs, growing as d^3 with width).
Kimi K3 (§5.2.2) shards parameters across data-parallel ranks and gathers
whole matrices per owner before NS. This module is the state-replicated first
stage of that design: optimizer state and updates stay replicated — the
checkpoint layout is unchanged and per-matrix numerics match the replicated
path, since each matrix is orthogonalized by the same vmapped function either
way; only which chip computes it changes.

Mechanism: every Muon problem is reshaped with optax's own reshape rule to
``[batch, rows, cols]``, grouped by problem shape, concatenated, zero-padded
to a multiple of the mesh's ``data``-axis size, annotated with a sharding
constraint over ``data``, orthogonalized with optax's own Newton-Schulz, and
annotated back to replicated so XLA emits exactly one slice + all-gather pair
around the sharded compute. Zero padding is NS-stable: the Frobenius
pre-normalization maps a zero matrix to zero (``x / (0 + eps)``), and every
NS iterate of zero is zero.

The momentum/bias-correction/composition code deliberately mirrors
``optax.contrib.muon`` line by line (including its private helpers; the
environment pins optax, and tests assert update parity against the reference
implementation) so the only difference between the two transforms is where
the orthogonalization runs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax.tree
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from optax._src import alias, base, combine, numerics, transform, utils
from optax.contrib import MuonDimensionNumbers
from optax.contrib._muon import (
    MuonState,
    _compute_muon_reshape,
    _is_weight_dim_nums,
    orthogonalize_via_newton_schulz,
    scale_by_shape,
)
from optax.transforms import _masking

_DEFAULT_NS_COEFFS = (3.4445, -4.7750, 2.0315)

__all__ = ["distributed_muon", "scale_by_distributed_muon"]


def _distributed_orthogonalize(
    leaves,
    dimension_numbers,
    *,
    mesh: Mesh,
    ns_coeffs,
    ns_steps,
    eps,
):
    """Orthogonalizes every Muon problem exactly once across the data axis."""
    data_size = mesh.shape["data"]
    reshaped = []
    inverse_fns = []
    problem_counts = []
    for leaf, dimension_number in zip(leaves, dimension_numbers, strict=True):
        reshape_fn, inverse_fn = _compute_muon_reshape(leaf, dimension_number)
        stacked = reshape_fn(leaf)
        reshaped.append(stacked)
        inverse_fns.append(inverse_fn)
        problem_counts.append(stacked.shape[0])

    groups: dict[tuple, list[int]] = {}
    for index, stacked in enumerate(reshaped):
        key = (stacked.shape[1], stacked.shape[2], str(stacked.dtype))
        groups.setdefault(key, []).append(index)

    results = [None] * len(reshaped)
    for (rows, cols, _), indices in groups.items():
        stack = jnp.concatenate([reshaped[index] for index in indices], axis=0)
        total = stack.shape[0]
        padded_total = -(-total // data_size) * data_size
        if padded_total != total:
            stack = jnp.concatenate(
                (
                    stack,
                    jnp.zeros((padded_total - total, rows, cols), dtype=stack.dtype),
                ),
                axis=0,
            )
        stack = jax.lax.with_sharding_constraint(
            stack, NamedSharding(mesh, P("data"))
        )
        orthogonalized = orthogonalize_via_newton_schulz(
            stack,
            ns_coeffs,
            ns_steps,
            "frobenius",
            eps,
            MuonDimensionNumbers(reduction_axis=(1,), output_axis=(2,)),
        )
        orthogonalized = jax.lax.with_sharding_constraint(
            orthogonalized, NamedSharding(mesh, P())
        )
        offset = 0
        for index in indices:
            results[index] = orthogonalized[offset : offset + problem_counts[index]]
            offset += problem_counts[index]
    return [
        inverse_fn(result) for inverse_fn, result in zip(inverse_fns, results, strict=True)
    ]


def scale_by_distributed_muon(
    mesh: Mesh,
    ns_coeffs=_DEFAULT_NS_COEFFS,
    ns_steps=5,
    beta=0.95,
    eps=1e-8,
    mu_dtype=None,
    *,
    nesterov: bool = True,
    weight_dimension_numbers=None,
) -> base.GradientTransformation:
    """`optax.contrib.scale_by_muon` with the NS step sharded over ``data``."""
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)
        ns_coeffs_ = jnp.asarray(ns_coeffs)
        if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
            raise ValueError(
                f"ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}"
            )
        if ns_coeffs_.ndim == 2:
            if not ns_coeffs_.shape[0] <= ns_steps:
                raise ValueError(f"Not enough coeffs to perform {ns_steps} steps")
            ns_coeffs_ = ns_coeffs_[-ns_steps:]
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            ns_coeffs=ns_coeffs_,
        )

    def update_fn(updates, state, params=None):
        del params
        if callable(weight_dimension_numbers):
            resolved_weight_dim_nums = weight_dimension_numbers(updates)
        else:
            resolved_weight_dim_nums = weight_dimension_numbers

        mu = optax.tree.update_moment(updates, state.mu, beta, 1)
        count_inc = numerics.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: beta * m + (1 - beta) * g,
                optax.tree.bias_correction(
                    mu, beta, numerics.safe_increment(count_inc)
                ),
                optax.tree.bias_correction(updates, beta, count_inc),
            )
        else:
            mu_hat = optax.tree.bias_correction(mu, beta, count_inc)

        collected_leaves = []
        collected_dim_nums = []

        def _collect(leaf, dimension_number):
            collected_leaves.append(leaf)
            collected_dim_nums.append(dimension_number)
            return leaf

        jax.tree.map(
            _collect,
            mu_hat,
            resolved_weight_dim_nums,
            is_leaf=_is_weight_dim_nums,
        )
        orthogonalized = _distributed_orthogonalize(
            collected_leaves,
            collected_dim_nums,
            mesh=mesh,
            ns_coeffs=state.ns_coeffs,
            ns_steps=ns_steps,
            eps=eps,
        )
        result_iterator = iter(orthogonalized)
        updates = jax.tree.map(
            lambda leaf, dimension_number: next(result_iterator),
            mu_hat,
            resolved_weight_dim_nums,
            is_leaf=_is_weight_dim_nums,
        )
        mu = optax.tree.cast(mu, mu_dtype)
        return updates, MuonState(
            count=count_inc,
            mu=mu,
            ns_coeffs=state.ns_coeffs,
        )

    return base.GradientTransformation(init_fn, update_fn)


def distributed_muon(
    mesh: Mesh,
    learning_rate,
    ns_coeffs=_DEFAULT_NS_COEFFS,
    ns_steps=5,
    beta=0.95,
    eps=1e-8,
    weight_decay=0.0,
    weight_decay_mask=None,
    mu_dtype=None,
    *,
    nesterov: bool = True,
    adam_b1=0.9,
    adam_b2=0.999,
    adam_eps_root=0.0,
    adam_weight_decay=0.0,
    adam_learning_rate=None,
    muon_weight_dimension_numbers=None,
    consistent_rms=None,
) -> base.GradientTransformation:
    """`optax.contrib.muon` with Newton-Schulz distributed over ``data``.

    The composition mirrors the reference implementation exactly; only
    `scale_by_muon` is replaced. Dimension numbers are required because the
    routing layer always provides them.
    """
    if muon_weight_dimension_numbers is None:
        raise ValueError(
            "distributed_muon requires explicit muon_weight_dimension_numbers"
        )
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    def param_labels(params):
        dim_nums = (
            muon_weight_dimension_numbers(params)
            if callable(muon_weight_dimension_numbers)
            else muon_weight_dimension_numbers
        )
        populate_subtree_ = lambda dim_num, x: jax.tree.map(
            lambda y: "muon" if dim_num is not None else "adam", x
        )
        return jax.tree.map(
            populate_subtree_,
            dim_nums,
            params,
            is_leaf=lambda x: x is None or _is_weight_dim_nums(x),
        )

    def muon_weight_dim_nums_fn(params):
        dim_nums = (
            muon_weight_dimension_numbers(params)
            if callable(muon_weight_dimension_numbers)
            else muon_weight_dimension_numbers
        )
        mask = jax.tree.map(lambda label: label == "muon", param_labels(params))
        is_leaf = lambda x: (
            x is None
            or _is_weight_dim_nums(x)
            or isinstance(x, _masking.MaskedNode)
        )
        populate_subtree_ = lambda dim_nums, submask: jax.tree.map(
            lambda m: dim_nums if m else _masking.MaskedNode(), submask
        )
        return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)

    return combine.partition(
        transforms={
            "muon": combine.chain(
                scale_by_distributed_muon(
                    mesh,
                    ns_coeffs=ns_coeffs,
                    ns_steps=ns_steps,
                    beta=beta,
                    eps=eps,
                    mu_dtype=mu_dtype,
                    nesterov=nesterov,
                    weight_dimension_numbers=muon_weight_dim_nums_fn,
                ),
                scale_by_shape(
                    weight_dimension_numbers=muon_weight_dim_nums_fn,
                    consistent_rms=consistent_rms,
                ),
                transform.add_decayed_weights(weight_decay, weight_decay_mask),
                transform.scale_by_learning_rate(learning_rate),
            ),
            "adam": alias.adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=eps,
                eps_root=adam_eps_root,
                weight_decay=adam_weight_decay,
                mu_dtype=mu_dtype,
                nesterov=nesterov,
            ),
        },
        param_labels=param_labels,
    )
