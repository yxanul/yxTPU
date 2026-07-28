"""Exhaustive semantic routing for AdamW and Muon."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from yxtpu_pretrain.config import OptimizerConfig
from yxtpu_pretrain.layers.roles import ADAMW_ROLES, MUON_ROLES, ParamRole
from yxtpu_pretrain.optimizers.distributed_muon import distributed_muon
from yxtpu_pretrain.optimizers.scaled_muon import scaled_muon

Path = tuple[str | int, ...]


@dataclass(frozen=True)
class Route:
    path: Path
    role: ParamRole
    optimizer: str
    shape: tuple[int, ...]
    reduction_axes: tuple[int, ...] = ()
    output_axes: tuple[int, ...] = ()
    batch_axes: tuple[int, ...] = ()
    # Kind tag of the alternate matricization this route adopted, or None
    # when the standard declaration is in force.
    alt_kind: str | None = None
    # optax's consistent-rms rule scales each orthogonalized update by
    # 0.2 * sqrt(max(fan_in, fan_out)), so switching matricization silently
    # changes the effective per-parameter update scale. This records
    # sqrt(max fans_standard) / sqrt(max fans_alternate) for alt-applied
    # routes (1.0 otherwise), so the shift can be cancelled explicitly.
    scale_compensation: float = 1.0


def _actual_axis(original_axis: int, scan_axis: int | None) -> int:
    if scan_axis is None or original_axis < scan_axis:
        return original_axis
    return original_axis + 1


def _max_fan(shape: tuple[int, ...], reduction, output) -> float:
    fan_in = math.prod(shape[axis] for axis in reduction)
    fan_out = math.prod(shape[axis] for axis in output)
    return float(max(fan_in, fan_out))


def classify_parameters(
    parameters,
    *,
    muon_per_head: bool = False,
    muon_kda_out_proj_whole: bool = False,
) -> list[Route]:
    """Classifies every trainable parameter and raises on the first gap.

    Each flag enables one kind of declared alternate Muon matricization:
    ``muon_per_head`` moves QKV head axes into the Muon batch group so
    Newton-Schulz runs one block per head (Kimi K3 §2.5);
    ``muon_kda_out_proj_whole`` switches the KDA out_proj from its historical
    heads-only reduction to the whole-matrix (heads*dim -> embed) form.
    Parameters without a matching alternate keep their standard matricization.
    """
    enabled_kinds = set()
    if muon_per_head:
        enabled_kinds.add("per_head")
    if muon_kda_out_proj_whole:
        enabled_kinds.add("kda_out_proj_whole")
    routes: list[Route] = []
    for path, variable in nnx.to_flat_state(parameters):
        metadata = variable.get_metadata()
        raw_role = metadata.get("role")
        if raw_role is None:
            raise ValueError(f"trainable parameter {path} has no declared optimizer role")
        try:
            role = ParamRole(raw_role)
        except ValueError as error:
            raise ValueError(f"trainable parameter {path} has unknown role {raw_role!r}") from error
        shape = tuple(variable.get_value().shape)
        scan_axis = metadata.get("param_scan_axis")
        if role in MUON_ROLES:
            original_in = tuple(metadata.get("matrix_in_axes", ()))
            original_out = tuple(metadata.get("matrix_out_axes", ()))
            if not original_in or not original_out:
                raise ValueError(f"Muon parameter {path} does not declare both matrix axis groups")
            alt_kind = metadata.get("matrix_alt_kind")
            applied_kind = None
            scale_compensation = 1.0
            selected_in, selected_out = original_in, original_out
            if alt_kind is not None and alt_kind in enabled_kinds:
                alt_in = tuple(metadata.get("matrix_alt_in_axes"))
                alt_out = tuple(metadata.get("matrix_alt_out_axes"))
                applied_kind = alt_kind
                selected_in, selected_out = alt_in, alt_out
            reduction = tuple(_actual_axis(axis, scan_axis) for axis in selected_in)
            output = tuple(_actual_axis(axis, scan_axis) for axis in selected_out)
            if applied_kind is not None:
                standard_reduction = tuple(
                    _actual_axis(axis, scan_axis) for axis in original_in
                )
                standard_output = tuple(
                    _actual_axis(axis, scan_axis) for axis in original_out
                )
                scale_compensation = math.sqrt(
                    _max_fan(shape, standard_reduction, standard_output)
                    / _max_fan(shape, reduction, output)
                )
            if set(reduction) & set(output):
                raise ValueError(f"Muon parameter {path} has overlapping matrix axes")
            covered = set(reduction) | set(output)
            batch = tuple(axis for axis in range(len(shape)) if axis not in covered)
            if scan_axis is not None and scan_axis not in batch:
                raise ValueError(
                    f"scanned parameter {path} must treat axis {scan_axis} as Muon batch"
                )
            routes.append(
                Route(
                    path=path,
                    role=role,
                    optimizer="muon",
                    shape=shape,
                    reduction_axes=reduction,
                    output_axes=output,
                    batch_axes=batch,
                    alt_kind=applied_kind,
                    scale_compensation=scale_compensation,
                )
            )
        elif role in ADAMW_ROLES:
            routes.append(Route(path=path, role=role, optimizer="adamw", shape=shape))
        else:
            raise ValueError(f"parameter {path} with role {role} is not routed")
    if not routes:
        raise ValueError("model has no trainable parameters")
    return routes


def _muon_dimension_tree(parameters, routes: list[Route]):
    by_path = {route.path: route for route in routes}
    values = []
    for path, variable in nnx.to_flat_state(parameters):
        route = by_path[path]
        dimensions = (
            optax.contrib.MuonDimensionNumbers(
                reduction_axis=route.reduction_axes,
                output_axis=route.output_axes,
            )
            if route.optimizer == "muon"
            else None
        )
        values.append((path, variable.replace(value=dimensions)))
    variable_dimensions = nnx.from_flat_state(values)
    pure_dimensions = nnx.as_pure(variable_dimensions)

    def dimensions_for(current_parameters):
        first_value = nnx.to_flat_state(current_parameters)[0][1]
        return (
            variable_dimensions
            if isinstance(first_value, nnx.Variable)
            else pure_dimensions
        )

    return dimensions_for


def _muon_mask_tree(parameters, routes: list[Route]):
    """Boolean mask over the gradient tree marking Muon-routed leaves."""
    by_path = {route.path: route for route in routes}
    values = []
    for path, variable in nnx.to_flat_state(parameters):
        values.append(
            (path, variable.replace(value=by_path[path].optimizer == "muon"))
        )
    variable_mask = nnx.from_flat_state(values)
    pure_mask = nnx.as_pure(variable_mask)

    def mask_for(current_updates):
        first_value = nnx.to_flat_state(current_updates)[0][1]
        return (
            variable_mask
            if isinstance(first_value, nnx.Variable)
            else pure_mask
        )

    return mask_for


def _scale_compensation_tree(parameters, routes: list[Route]):
    """Per-leaf multipliers cancelling the consistent-rms shape-rule shift.

    Only per-head-matricized routes are compensated: their scale shift is an
    artifact of the shape rule (the same gradient block orthogonalized per
    head instead of jointly), whereas the out_proj whole-matrix switch is
    itself the change under test and keeps whatever scale its matricization
    implies. Every other leaf carries 1.0.
    """
    by_path = {route.path: route for route in routes}
    values = []
    for path, variable in nnx.to_flat_state(parameters):
        route = by_path[path]
        factor = (
            route.scale_compensation
            if route.optimizer == "muon" and route.alt_kind == "per_head"
            else 1.0
        )
        values.append((path, variable.replace(value=float(factor))))
    variable_factors = nnx.from_flat_state(values)
    pure_factors = nnx.as_pure(variable_factors)

    def factors_for(current_updates):
        first_value = nnx.to_flat_state(current_updates)[0][1]
        return (
            variable_factors
            if isinstance(first_value, nnx.Variable)
            else pure_factors
        )

    return factors_for


def build_learning_rate_schedule(config: OptimizerConfig):
    """Builds the shared warmup/cosine schedule for the optimizer and telemetry.

    train._learning_rate mirrors this host-side for logging; keep both in
    lockstep."""
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )
    if config.decay_steps is None:
        decay = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=config.schedule_steps - config.warmup_steps,
            alpha=config.final_learning_rate_fraction,
        )
        return optax.join_schedules(
            schedules=(warmup, decay),
            boundaries=(config.warmup_steps,),
        )
    constant = optax.constant_schedule(config.learning_rate)
    decay = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.decay_steps,
        alpha=config.final_learning_rate_fraction,
    )
    return optax.join_schedules(
        schedules=(warmup, constant, decay),
        boundaries=(
            config.warmup_steps,
            config.schedule_steps - config.decay_steps,
        ),
    )


def build_optimizer(model: nnx.Module, config: OptimizerConfig):
    """Builds an Optax transform and its audited route table."""
    parameters = nnx.state(model, nnx.Param)
    routes = classify_parameters(
        parameters,
        muon_per_head=config.muon_per_head,
        muon_kda_out_proj_whole=config.muon_kda_out_proj_whole,
    )
    clipping = optax.clip_by_global_norm(config.gradient_clip_norm)
    learning_rate = build_learning_rate_schedule(config)
    if config.name == "adamw":
        routes = [replace(route, optimizer="adamw") for route in routes]
        transform = optax.chain(
            clipping,
            optax.adamw(
                learning_rate=learning_rate,
                b1=config.beta1,
                b2=config.beta2,
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            ),
        )
    else:
        dimensions = _muon_dimension_tree(parameters, routes)
        stages = [clipping]
        if config.muon_ns_bf16:
            # Cast Muon-routed gradients to bf16 AFTER clipping (the global
            # norm and its metric stay fp32 and bit-identical) so momentum,
            # bias correction, and the Newton-Schulz iteration all run in
            # bf16 - the modded-nanogpt lineage. mu_dtype=bf16 is required
            # too: either half alone silently leaves NS in fp32. The
            # Frobenius pre-normalization also becomes a bf16 reduction; the
            # 200-step trajectory gate covers that deviation.
            stages.append(
                optax.masked(
                    optax.stateless(
                        lambda updates, params: jax.tree.map(
                            lambda u: u.astype(jnp.bfloat16), updates
                        )
                    ),
                    _muon_mask_tree(parameters, routes),
                )
            )
        muon_arguments = dict(
            learning_rate=learning_rate,
            ns_steps=config.muon_ns_steps,
            beta=config.muon_beta,
            eps=config.muon_epsilon,
            mu_dtype=jnp.bfloat16 if config.muon_ns_bf16 else None,
            consistent_rms=config.muon_consistent_rms,
            weight_decay=config.weight_decay,
            adam_b1=config.beta1,
            adam_b2=config.beta2,
            adam_weight_decay=config.weight_decay,
            adam_learning_rate=learning_rate,
            muon_weight_dimension_numbers=dimensions,
        )
        if config.muon_per_head_scale_compensation:
            stages.append(
                scaled_muon(
                    update_scale_factors=_scale_compensation_tree(parameters, routes),
                    **muon_arguments,
                )
            )
        elif config.muon_distributed_ns:
            mesh = getattr(model, "mesh", None)
            if mesh is None or "data" not in mesh.shape:
                raise ValueError(
                    "muon_distributed_ns requires a model carrying a mesh with "
                    "a data axis"
                )
            stages.append(distributed_muon(mesh, **muon_arguments))
        else:
            stages.append(optax.contrib.muon(**muon_arguments))
        transform = optax.chain(*stages)
    return transform, routes
