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
    args = parser.parse_args()
    precisions = args.precisions or ["guarded_fp32", "full_fp32"]

    first_config = _base_config(precisions[0])
    apply_hardware_environment(first_config.hardware)
    host_update = _load_update(first_config, args.update_index)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
