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


def _run_mode(config, host_update):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.utils import max_utils

    from yxtpu_pretrain.model import HybridLanguageModel
    from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
    from yxtpu_pretrain.runtime.mesh import create_mesh
    from yxtpu_pretrain.runtime.sharding import logical_mesh_context
    from yxtpu_pretrain.train import _device_batch, _loss, _tree_max_abs

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
    for microbatch_index in range(accumulation):
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
    args = parser.parse_args()
    precisions = args.precisions or ["guarded_fp32", "full_fp32"]

    first_config = _base_config(precisions[0])
    apply_hardware_environment(first_config.hardware)
    host_update = _load_update(first_config, args.update_index)
    for precision in precisions:
        result = _run_mode(_base_config(precision), host_update)
        print(json.dumps({"update_index": args.update_index, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
