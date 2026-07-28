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

"""optax.contrib.muon with an exact per-leaf update-scale stage.

optax's consistent-rms rule scales each orthogonalized update by
``0.2 * sqrt(max(fan_in, fan_out))`` computed from the leaf's Muon
matricization. Switching a matricization (per-head blocks instead of a joint
matrix) therefore changes the effective per-parameter update scale as a side
effect — an artifact of the shape rule, not of the update direction under
test. This transform cancels it exactly: a per-leaf multiplier applied
*between* ``scale_by_shape`` and ``add_decayed_weights``, so weight decay is
untouched (a post-chain multiplier would scale the decay term of the affected
leaves too, contaminating the comparison).

The composition mirrors ``optax.contrib.muon`` line by line otherwise; the
environment pins optax, and tests assert that with unit factors the transform
reproduces the reference updates, and with factors it scales exactly the
intended leaves.
"""

from __future__ import annotations

import jax
from optax._src import alias, base, combine, transform
from optax.contrib import scale_by_muon
from optax.contrib._muon import _is_weight_dim_nums, scale_by_shape
from optax.transforms import _masking

_DEFAULT_NS_COEFFS = (3.4445, -4.7750, 2.0315)

__all__ = ["scaled_muon"]


def _scale_muon_updates(update_scale_factors) -> base.GradientTransformation:
    """Multiplies each Muon-partition leaf by its declared factor."""

    def update_fn(updates, state, params=None):
        del params
        factors = (
            update_scale_factors(updates)
            if callable(update_scale_factors)
            else update_scale_factors
        )
        scaled = jax.tree.map(
            lambda leaf, factor: (
                leaf if isinstance(leaf, _masking.MaskedNode) else leaf * factor
            ),
            updates,
            factors,
            is_leaf=lambda node: isinstance(node, _masking.MaskedNode),
        )
        return scaled, state

    return base.GradientTransformation(base.init_empty_state, update_fn)


def scaled_muon(
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
    update_scale_factors=None,
) -> base.GradientTransformation:
    """`optax.contrib.muon` with per-leaf factors after the shape scaling."""
    if muon_weight_dimension_numbers is None:
        raise ValueError("scaled_muon requires explicit muon_weight_dimension_numbers")
    if update_scale_factors is None:
        raise ValueError("scaled_muon requires update_scale_factors")
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
                scale_by_muon(
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
                _scale_muon_updates(update_scale_factors),
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
