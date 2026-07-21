#!/usr/bin/env python3
"""Compare KDA precision modes on an exact streamed ClimbMix update.

This diagnostic intentionally lives outside the trainer. It reconstructs one
deterministic packed update, splits it along the configured gradient-
accumulation axis, and reports loss and gradient norms for every microbatch
without applying an optimizer update.
"""

from __future__ import annotations

import argparse
import json

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.runtime.environment import apply_hardware_environment


def _base_config(precision: str):
    return load_config(
        model="kda_hybrid_309m_gpt2",
        optimizer="adamw_10b",
        data="climbmix",
        hardware="v6e-8",
        experiment="climbmix_10b",
        overrides=[
            f"model.kda.precision={precision}",
            # Mark the run as a qualification benchmark rather than a real
            # token-budgeted training job. Production still fixes the solver
            # policy inside kda_fused_pallas.py instead of exposing it here.
            "experiment.benchmark=true",
            "experiment.token_budget=null",
            "data.prefetch_batches=0",
            "data.eval_interval=0",
            "experiment.diagnostics.enabled=false",
            "experiment.harness_eval.enabled=false",
        ],
    )


def _load_update(config, update_index: int):
    import jax

    from yxtpu_pretrain.runtime.data import create_data_iterator

    process_batch = (
        config.data.per_device_batch_size
        * jax.local_device_count()
        * config.experiment.gradient_accumulation_steps
    )
    iterator = create_data_iterator(
        config.data,
        global_batch_size=process_batch,
        vocab_size=config.model.vocab_size,
    )
    batch = None
    for _ in range(update_index):
        batch = next(iterator)
    if batch is None:
        raise ValueError("update_index must be positive")
    return batch


def _run_mode(
    config,
    host_update,
    *,
    highest_roles: tuple[str, ...],
    microbatches,
    pairwise_row_block_size: int | None,
    pairwise_anchor: str | None,
):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.utils import max_utils

    from yxtpu_pretrain.kernels import kda_fused_pallas
    from yxtpu_pretrain.model import HybridLanguageModel
    from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
    from yxtpu_pretrain.runtime.mesh import create_mesh
    from yxtpu_pretrain.runtime.sharding import logical_mesh_context
    from yxtpu_pretrain.train import _device_batch, _loss, _tree_max_abs

    precision_attributes = {
        "chunk": "_CHUNK_MATMUL_PRECISION",
        "state": "_STATE_MATMUL_PRECISION",
        "pairwise": "_PAIRWISE_MATMUL_PRECISION",
        "solve_coupling": "_SOLVE_COUPLING_MATMUL_PRECISION",
    }
    for role in highest_roles:
        setattr(
            kda_fused_pallas,
            precision_attributes[role],
            jax.lax.Precision.HIGHEST,
        )
    if pairwise_row_block_size is not None:
        if pairwise_row_block_size not in (1, 2, 4, 8):
            raise ValueError("pairwise row block size must be one of 1, 2, 4, or 8")
        kda_fused_pallas._PAIRWISE_ROW_BLOCK_SIZE = pairwise_row_block_size
    if pairwise_anchor is not None:
        kda_fused_pallas._PAIRWISE_ANCHOR_MIDPOINT = pairwise_anchor == "midpoint"

    mesh = create_mesh(config.hardware)
    logical_axis_rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, logical_axis_rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(config.experiment.seed))

    @nnx.jit
    def gradient_metrics(current_model, batch):
        def loss_fn(candidate):
            loss, _ = _loss(candidate, batch, record_max_logits=False)
            return loss

        loss, gradients = nnx.value_and_grad(loss_fn)(current_model)
        return {
            "loss": loss,
            "grad_norm": max_utils.l2norm_pytree(gradients),
            "grad_max_abs": _tree_max_abs(gradients),
        }

    accumulation = config.experiment.gradient_accumulation_steps
    microbatch_size = host_update["input_ids"].shape[0] // accumulation
    records = []
    compiled = None
    selected_microbatches = (
        tuple(range(accumulation)) if microbatches is None else microbatches
    )
    for microbatch_index in selected_microbatches:
        if not 0 <= microbatch_index < accumulation:
            raise ValueError(
                f"microbatch index {microbatch_index} is outside [0, {accumulation})"
            )
        start = microbatch_index * microbatch_size
        stop = start + microbatch_size
        microbatch = {
            key: value[start:stop]
            for key, value in host_update.items()
        }
        device_batch = _device_batch(microbatch, mesh)
        with logical_mesh_context(mesh, logical_axis_rules):
            if compiled is None:
                compiled = gradient_metrics.lower(model, device_batch).compile()
            metrics = compiled(model, device_batch)
            jax.block_until_ready(metrics)
        host = jax.device_get(metrics)
        records.append(
            {
                "microbatch": microbatch_index,
                "loss": float(host["loss"]),
                "grad_norm": float(host["grad_norm"]),
                "grad_max_abs": float(host["grad_max_abs"]),
                "finite": bool(
                    jnp.isfinite(host["loss"])
                    & jnp.isfinite(host["grad_norm"])
                    & jnp.isfinite(host["grad_max_abs"])
                ),
            }
        )
    memory = compiled.memory_analysis()
    return {
        "precision": config.model.kda.precision,
        "solve_method": kda_fused_pallas._SOLVE_METHOD,
        "kernel_highest_roles": list(highest_roles),
        "pairwise_row_block_size": kda_fused_pallas._PAIRWISE_ROW_BLOCK_SIZE,
        "pairwise_anchor": (
            "midpoint" if kda_fused_pallas._PAIRWISE_ANCHOR_MIDPOINT else "last"
        ),
        "compiled_memory": {
            key: int(getattr(memory, key, 0) or 0)
            for key in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "temp_size_in_bytes",
                "generated_code_size_in_bytes",
            )
        },
        "microbatches": records,
    }


def _run_direct_comparison(
    fused_config,
    reference_config,
    host_update,
    *,
    highest_roles: tuple[str, ...],
    microbatches,
    pairwise_row_block_size: int | None,
    pairwise_anchor: str | None,
):
    """Compare every fused gradient element with the analytical reference."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from yxtpu_pretrain.kernels import kda_fused_pallas
    from yxtpu_pretrain.model import HybridLanguageModel
    from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
    from yxtpu_pretrain.runtime.mesh import create_mesh
    from yxtpu_pretrain.runtime.sharding import logical_mesh_context
    from yxtpu_pretrain.train import _device_batch, _loss

    precision_attributes = {
        "chunk": "_CHUNK_MATMUL_PRECISION",
        "state": "_STATE_MATMUL_PRECISION",
        "pairwise": "_PAIRWISE_MATMUL_PRECISION",
        "solve_coupling": "_SOLVE_COUPLING_MATMUL_PRECISION",
    }
    for role in highest_roles:
        setattr(
            kda_fused_pallas,
            precision_attributes[role],
            jax.lax.Precision.HIGHEST,
        )
    if pairwise_row_block_size is not None:
        kda_fused_pallas._PAIRWISE_ROW_BLOCK_SIZE = pairwise_row_block_size
    if pairwise_anchor is not None:
        kda_fused_pallas._PAIRWISE_ANCHOR_MIDPOINT = pairwise_anchor == "midpoint"

    mesh = create_mesh(fused_config.hardware)
    logical_axis_rules = make_leaf_config(fused_config).logical_axis_rules
    with logical_mesh_context(mesh, logical_axis_rules):
        fused_model = HybridLanguageModel(
            fused_config,
            mesh,
            rngs=nnx.Rngs(fused_config.experiment.seed),
        )
        reference_model = HybridLanguageModel(
            reference_config,
            mesh,
            rngs=nnx.Rngs(reference_config.experiment.seed),
        )

    def _sum_squares(tree):
        leaves = jax.tree.leaves(tree)
        return sum(
            jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves
        )

    def _dot(left, right):
        left_leaves = jax.tree.leaves(left)
        right_leaves = jax.tree.leaves(right)
        return sum(
            jnp.sum(
                left_leaf.astype(jnp.float32)
                * right_leaf.astype(jnp.float32)
            )
            for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
        )

    def _max_abs(tree):
        leaves = jax.tree.leaves(tree)
        return jnp.max(
            jnp.stack(
                [
                    jnp.max(jnp.abs(leaf.astype(jnp.float32)), initial=0.0)
                    for leaf in leaves
                ]
            ),
            initial=0.0,
        )

    @nnx.jit
    def comparison_metrics(current_fused, current_reference, batch):
        def loss_fn(candidate):
            loss, _ = _loss(candidate, batch, record_max_logits=False)
            return loss

        fused_loss, fused_gradients = nnx.value_and_grad(loss_fn)(current_fused)
        reference_loss, reference_gradients = nnx.value_and_grad(loss_fn)(
            current_reference
        )
        fused_parameters = nnx.state(current_fused, nnx.Param)
        reference_parameters = nnx.state(current_reference, nnx.Param)
        parameter_differences = jax.tree.map(
            lambda fused, reference: fused.astype(jnp.float32)
            - reference.astype(jnp.float32),
            fused_parameters,
            reference_parameters,
        )
        differences = jax.tree.map(
            lambda fused, reference: fused.astype(jnp.float32)
            - reference.astype(jnp.float32),
            fused_gradients,
            reference_gradients,
        )
        fused_squared_norm = _sum_squares(fused_gradients)
        reference_squared_norm = _sum_squares(reference_gradients)
        difference_squared_norm = _sum_squares(differences)
        denominator = jnp.maximum(reference_squared_norm, jnp.finfo(jnp.float32).tiny)
        cosine_denominator = jnp.maximum(
            jnp.sqrt(fused_squared_norm * reference_squared_norm),
            jnp.finfo(jnp.float32).tiny,
        )
        reference_max_abs = _max_abs(reference_gradients)
        max_abs_difference = _max_abs(differences)
        reference_parameter_squared_norm = _sum_squares(reference_parameters)
        return {
            "fused_loss": fused_loss,
            "reference_loss": reference_loss,
            "fused_grad_norm": jnp.sqrt(fused_squared_norm),
            "reference_grad_norm": jnp.sqrt(reference_squared_norm),
            "gradient_relative_l2_error": jnp.sqrt(
                difference_squared_norm / denominator
            ),
            "gradient_cosine_similarity": _dot(
                fused_gradients, reference_gradients
            )
            / cosine_denominator,
            "gradient_max_abs_difference": max_abs_difference,
            "gradient_max_abs_difference_over_reference_max": (
                max_abs_difference
                / jnp.maximum(reference_max_abs, jnp.finfo(jnp.float32).tiny)
            ),
            "reference_grad_max_abs": reference_max_abs,
            "parameter_relative_l2_error": jnp.sqrt(
                _sum_squares(parameter_differences)
                / jnp.maximum(
                    reference_parameter_squared_norm,
                    jnp.finfo(jnp.float32).tiny,
                )
            ),
            "parameter_max_abs_difference": _max_abs(parameter_differences),
        }

    accumulation = fused_config.experiment.gradient_accumulation_steps
    microbatch_size = host_update["input_ids"].shape[0] // accumulation
    selected_microbatches = (
        tuple(range(accumulation)) if microbatches is None else microbatches
    )
    records = []
    compiled = None
    for microbatch_index in selected_microbatches:
        if not 0 <= microbatch_index < accumulation:
            raise ValueError(
                f"microbatch index {microbatch_index} is outside [0, {accumulation})"
            )
        start = microbatch_index * microbatch_size
        stop = start + microbatch_size
        microbatch = {
            key: value[start:stop]
            for key, value in host_update.items()
        }
        device_batch = _device_batch(microbatch, mesh)
        with logical_mesh_context(mesh, logical_axis_rules):
            if compiled is None:
                compiled = comparison_metrics.lower(
                    fused_model,
                    reference_model,
                    device_batch,
                ).compile()
            metrics = compiled(fused_model, reference_model, device_batch)
            jax.block_until_ready(metrics)
        host = jax.device_get(metrics)
        records.append(
            {
                "microbatch": microbatch_index,
                **{key: float(value) for key, value in host.items()},
            }
        )
    memory = compiled.memory_analysis()
    return {
        "comparison": "guarded_fp32_vs_full_fp32",
        "solve_method": kda_fused_pallas._SOLVE_METHOD,
        "kernel_highest_roles": list(highest_roles),
        "pairwise_row_block_size": kda_fused_pallas._PAIRWISE_ROW_BLOCK_SIZE,
        "pairwise_anchor": (
            "midpoint" if kda_fused_pallas._PAIRWISE_ANCHOR_MIDPOINT else "last"
        ),
        "compiled_memory": {
            key: int(getattr(memory, key, 0) or 0)
            for key in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "temp_size_in_bytes",
                "generated_code_size_in_bytes",
            )
        },
        "microbatches": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-index", type=int, default=7)
    parser.add_argument(
        "--precision",
        action="append",
        choices=("guarded_fp32", "full_fp32"),
        dest="precisions",
    )
    parser.add_argument(
        "--kernel-highest",
        action="append",
        choices=("chunk", "state", "pairwise", "solve_coupling"),
        default=[],
        help="promote one guarded Pallas matmul role to six-pass FP32",
    )
    parser.add_argument(
        "--microbatch",
        action="append",
        type=int,
        dest="microbatches",
        help="restrict the comparison to one accumulation microbatch",
    )
    parser.add_argument(
        "--pairwise-row-block-size",
        type=int,
        choices=(1, 2, 4, 8),
        default=None,
        help="override the Pallas decay-rescaling row block",
    )
    parser.add_argument(
        "--pairwise-anchor",
        choices=("last", "midpoint"),
        default=None,
        help="override the shared decay anchor within each pairwise row block",
    )
    parser.add_argument(
        "--direct-compare",
        action="store_true",
        help=(
            "compare every production fused gradient element with full_fp32; "
            "requires both precision modes"
        ),
    )
    parser.add_argument(
        "--max-gradient-relative-l2",
        type=float,
        default=3.0e-4,
        help="direct-comparison acceptance threshold (default: 3e-4)",
    )
    args = parser.parse_args()
    precisions = args.precisions or ["guarded_fp32", "full_fp32"]

    first_config = _base_config(precisions[0])
    apply_hardware_environment(first_config.hardware)
    host_update = _load_update(first_config, args.update_index)
    if not args.direct_compare:
        for precision in precisions:
            result = _run_mode(
                _base_config(precision),
                host_update,
                highest_roles=tuple(args.kernel_highest),
                microbatches=(
                    tuple(args.microbatches) if args.microbatches is not None else None
                ),
                pairwise_row_block_size=args.pairwise_row_block_size,
                pairwise_anchor=args.pairwise_anchor,
            )
            print(json.dumps({"update_index": args.update_index, **result}, sort_keys=True))
    else:
        if set(precisions) != {"guarded_fp32", "full_fp32"}:
            raise ValueError("--direct-compare requires guarded_fp32 and full_fp32")
        if args.max_gradient_relative_l2 <= 0:
            raise ValueError("--max-gradient-relative-l2 must be positive")
        result = _run_direct_comparison(
            _base_config("guarded_fp32"),
            _base_config("full_fp32"),
            host_update,
            highest_roles=tuple(args.kernel_highest),
            microbatches=(
                tuple(args.microbatches) if args.microbatches is not None else None
            ),
            pairwise_row_block_size=args.pairwise_row_block_size,
            pairwise_anchor=args.pairwise_anchor,
        )
        passed = all(
            record["gradient_relative_l2_error"]
            <= args.max_gradient_relative_l2
            and record["parameter_relative_l2_error"] == 0.0
            for record in result["microbatches"]
        )
        result["qualification"] = {
            "max_gradient_relative_l2": args.max_gradient_relative_l2,
            "parameters_must_match_exactly": True,
            "passed": passed,
        }
        print(json.dumps({"update_index": args.update_index, **result}, sort_keys=True))
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
