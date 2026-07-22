"""Exhaustive semantic routing for AdamW and Muon."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from yxtpu_pretrain.config import OptimizerConfig
from yxtpu_pretrain.layers.roles import ADAMW_ROLES, MUON_ROLES, ParamRole

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


def _actual_axis(original_axis: int, scan_axis: int | None) -> int:
    if scan_axis is None or original_axis < scan_axis:
        return original_axis
    return original_axis + 1


def classify_parameters(parameters) -> list[Route]:
    """Classifies every trainable parameter and raises on the first gap."""
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
            reduction = tuple(_actual_axis(axis, scan_axis) for axis in original_in)
            output = tuple(_actual_axis(axis, scan_axis) for axis in original_out)
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
    routes = classify_parameters(parameters)
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
        stages.append(
            optax.contrib.muon(
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
        )
        transform = optax.chain(*stages)
    return transform, routes
