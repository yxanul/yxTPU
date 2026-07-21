#!/usr/bin/env python3
"""Eight-device parity gate for the fused output loss.

This must pass before full-model throughput results are considered.  It checks
the production hidden/vocabulary dimensions, adversarial mask placement,
boundary labels, every element of dx/dw, and one complete AdamW model update.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorMetrics:
    max_abs: float
    rel_l2: float


def _error_metrics(actual, reference) -> ErrorMetrics:
    import jax
    import jax.numpy as jnp

    difference = actual.astype(jnp.float32) - reference.astype(jnp.float32)
    max_abs = jnp.max(jnp.abs(difference), initial=0.0)
    diff_norm = jnp.sqrt(jnp.sum(jnp.square(difference), dtype=jnp.float32))
    ref_norm = jnp.sqrt(jnp.sum(jnp.square(reference.astype(jnp.float32)), dtype=jnp.float32))
    rel_l2 = diff_norm / jnp.maximum(ref_norm, 1.0e-30)
    max_abs, rel_l2 = jax.device_get((max_abs, rel_l2))
    return ErrorMetrics(float(max_abs), float(rel_l2))


def _tree_error_metrics(actual, reference) -> ErrorMetrics:
    import jax
    import jax.numpy as jnp

    actual_leaves = jax.tree.leaves(actual)
    reference_leaves = jax.tree.leaves(reference)
    if len(actual_leaves) != len(reference_leaves):
        raise AssertionError("parity trees contain a different number of leaves")
    max_abs = jnp.asarray(0.0, dtype=jnp.float32)
    diff_squared = jnp.asarray(0.0, dtype=jnp.float32)
    reference_squared = jnp.asarray(0.0, dtype=jnp.float32)
    for actual_leaf, reference_leaf in zip(actual_leaves, reference_leaves, strict=True):
        difference = actual_leaf.astype(jnp.float32) - reference_leaf.astype(jnp.float32)
        max_abs = jnp.maximum(max_abs, jnp.max(jnp.abs(difference), initial=0.0))
        diff_squared += jnp.sum(jnp.square(difference), dtype=jnp.float32)
        reference_squared += jnp.sum(
            jnp.square(reference_leaf.astype(jnp.float32)), dtype=jnp.float32
        )
    rel_l2 = jnp.sqrt(diff_squared) / jnp.maximum(jnp.sqrt(reference_squared), 1.0e-30)
    max_abs, rel_l2 = jax.device_get((max_abs, rel_l2))
    return ErrorMetrics(float(max_abs), float(rel_l2))


def _kernel_parity(config, *, local_tokens: int) -> list[dict[str, object]]:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from yxtpu_pretrain.losses import data_parallel_linear_cross_entropy
    from yxtpu_pretrain.runtime.mesh import create_mesh

    mesh = create_mesh(config.hardware)
    if jax.device_count() != 8:
        raise RuntimeError(
            f"the parity gate requires exactly 8 devices, found {jax.device_count()}"
        )
    hidden_dim = config.model.emb_dim
    vocab_size = config.model.vocab_size
    global_tokens = local_tokens * config.hardware.device_count
    rng = np.random.default_rng(20260721)
    x_host = rng.standard_normal((global_tokens, hidden_dim), dtype=np.float32)
    weights_host = rng.standard_normal((hidden_dim, vocab_size), dtype=np.float32) * 0.02
    labels_host = rng.integers(0, vocab_size, size=(global_tokens,), dtype=np.int32)
    data_matrix = NamedSharding(mesh, P("data", None))
    data_vector = NamedSharding(mesh, P("data"))
    replicated = NamedSharding(mesh, P())
    x = jax.device_put(jnp.asarray(x_host, dtype=jnp.bfloat16), data_matrix)
    master_weights = jax.device_put(jnp.asarray(weights_host), replicated)

    def make_step(implementation):
        @jax.jit
        def step(x_value, labels_value, mask_value, master_weight_value):
            def loss_fn(x_arg, weight_arg):
                loss, tokens = data_parallel_linear_cross_entropy(
                    x_arg,
                    labels_value,
                    mask_value,
                    weight_arg.astype(jnp.bfloat16),
                    mesh=mesh,
                    implementation=implementation,
                )
                return loss, tokens

            return jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
                x_value, master_weight_value
            )

        return step

    reference_step = make_step("reference")
    fused_step = make_step("mosaic_tpu")
    cases = []

    all_valid = np.ones((global_tokens,), dtype=np.float32)
    cases.append(("all_valid", labels_host, all_valid, x))

    uneven = np.zeros((global_tokens,), dtype=np.float32)
    valid_per_device = (
        local_tokens,
        3 * local_tokens // 4,
        local_tokens // 2,
        local_tokens // 4,
        0,
        local_tokens - 24,
        1,
        local_tokens - 1,
    )
    for device, valid in enumerate(valid_per_device):
        start = device * local_tokens
        uneven[start : start + valid] = 1.0
    cases.append(("uneven_padding", labels_host, uneven, x))

    edge_labels = np.where(np.arange(global_tokens) % 2, vocab_size - 1, 0).astype(np.int32)
    strided_mask = (np.arange(global_tokens) % 7 != 0).astype(np.float32)
    cases.append(("edge_labels", edge_labels, strided_mask, x))

    scaled_x = x * jnp.asarray(4.0, dtype=jnp.bfloat16)
    cases.append(("scaled_hidden", labels_host, all_valid, scaled_x))

    results = []
    for name, labels_numpy, mask_numpy, x_value in cases:
        labels = jax.device_put(jnp.asarray(labels_numpy), data_vector)
        mask = jax.device_put(jnp.asarray(mask_numpy), data_vector)
        (reference_value, reference_grads) = reference_step(x_value, labels, mask, master_weights)
        (fused_value, fused_grads) = fused_step(x_value, labels, mask, master_weights)
        jax.block_until_ready((reference_value, reference_grads, fused_value, fused_grads))
        reference_loss, reference_tokens = reference_value
        fused_loss, fused_tokens = fused_value
        loss_abs = float(jax.device_get(jnp.abs(fused_loss - reference_loss)))
        loss_rel = loss_abs / max(abs(float(jax.device_get(reference_loss))), 1.0e-30)
        dx_error = _error_metrics(fused_grads[0], reference_grads[0])
        dw_error = _error_metrics(fused_grads[1], reference_grads[1])
        record = {
            "case": name,
            "reference_loss": float(jax.device_get(reference_loss)),
            "fused_loss": float(jax.device_get(fused_loss)),
            "loss_abs": loss_abs,
            "loss_rel": loss_rel,
            "tokens": float(jax.device_get(fused_tokens)),
            "token_delta": float(jax.device_get(fused_tokens - reference_tokens)),
            "dx": dx_error.__dict__,
            "dw": dw_error.__dict__,
        }
        print(json.dumps({"kernel_parity": record}, sort_keys=True), flush=True)
        if not np.isfinite(tuple(record["dx"].values())).all():
            raise AssertionError(f"{name}: non-finite dx error")
        if not np.isfinite(tuple(record["dw"].values())).all():
            raise AssertionError(f"{name}: non-finite dw error")
        if loss_rel > 3.0e-4:
            raise AssertionError(f"{name}: loss relative error {loss_rel:.3e}")
        if dx_error.rel_l2 > 5.0e-3:
            raise AssertionError(f"{name}: dx relative L2 error {dx_error.rel_l2:.3e}")
        if dw_error.rel_l2 > 5.0e-3:
            raise AssertionError(f"{name}: dw relative L2 error {dw_error.rel_l2:.3e}")
        if record["token_delta"] != 0.0:
            raise AssertionError(f"{name}: distributed token counts differ")
        results.append(record)
    return results


def _variable_values(state):
    import jax
    from flax import nnx

    return jax.tree.map(
        lambda variable: variable.get_value(),
        state,
        is_leaf=lambda value: isinstance(value, nnx.Variable),
    )


def _one_step_parity(config) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.common.train_state_nnx import TrainStateNNX

    from yxtpu_pretrain.config import load_config
    from yxtpu_pretrain.model import HybridLanguageModel
    from yxtpu_pretrain.optimizers import build_optimizer
    from yxtpu_pretrain.runtime.data import create_data_iterator
    from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
    from yxtpu_pretrain.runtime.mesh import create_mesh
    from yxtpu_pretrain.runtime.sharding import logical_mesh_context
    from yxtpu_pretrain.train import _device_batch, _make_train_step, _process_batch_sizes

    common_overrides = [
        "data.per_device_batch_size=1",
        "experiment.gradient_accumulation_steps=1",
        "experiment.steps=1",
        "optimizer.warmup_steps=0",
    ]

    def resolved(loss_implementation):
        return load_config(
            model="kda_hybrid_273m",
            optimizer="adamw",
            data="synthetic",
            hardware="v6e-8",
            experiment="selected",
            overrides=common_overrides + [f"model.loss.implementation={loss_implementation}"],
        )

    standard_config = resolved("standard")
    fused_config = resolved("tokamax_fused")
    mesh = create_mesh(config.hardware)
    rules = make_leaf_config(standard_config).logical_axis_rules

    def make_state(current_config):
        with logical_mesh_context(mesh, rules):
            model = HybridLanguageModel(
                current_config, mesh, rngs=nnx.Rngs(current_config.experiment.seed)
            )
            transform, _ = build_optimizer(model, current_config.optimizer)
            return TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))

    standard_state = make_state(standard_config)
    fused_state = make_state(fused_config)
    initial_standard = _variable_values(nnx.state(standard_state.model, nnx.Param))
    initial_fused = _variable_values(nnx.state(fused_state.model, nnx.Param))
    initial_error = _tree_error_metrics(initial_fused, initial_standard)
    if initial_error.max_abs != 0.0:
        raise AssertionError(f"initial model parameters differ: {initial_error}")
    initial_parameters = jax.tree.map(jnp.copy, initial_standard)

    update_batch, _ = _process_batch_sizes(standard_config, local_device_count=8)
    iterator = create_data_iterator(
        standard_config.data,
        global_batch_size=update_batch,
        vocab_size=standard_config.model.vocab_size,
    )
    batch = _device_batch(next(iterator), mesh)

    with logical_mesh_context(mesh, rules):
        standard_step = _make_train_step(standard_config)
        fused_step = _make_train_step(fused_config)
        standard_metrics = standard_step(standard_state, batch)
        fused_metrics = fused_step(fused_state, batch)
        jax.block_until_ready((standard_metrics, fused_metrics))

    standard_parameters = _variable_values(nnx.state(standard_state.model, nnx.Param))
    fused_parameters = _variable_values(nnx.state(fused_state.model, nnx.Param))
    standard_updates = jax.tree.map(jnp.subtract, standard_parameters, initial_parameters)
    fused_updates = jax.tree.map(jnp.subtract, fused_parameters, initial_parameters)
    parameter_error = _tree_error_metrics(fused_parameters, standard_parameters)
    update_error = _tree_error_metrics(fused_updates, standard_updates)

    standard_opt = _variable_values(nnx.state(standard_state.optimizer, nnx.OptState))
    fused_opt = _variable_values(nnx.state(fused_state.optimizer, nnx.OptState))
    optimizer_error = _tree_error_metrics(fused_opt, standard_opt)
    standard_loss = float(jax.device_get(standard_metrics["loss"]))
    fused_loss = float(jax.device_get(fused_metrics["loss"]))
    record = {
        "standard_loss": standard_loss,
        "fused_loss": fused_loss,
        "loss_abs": abs(fused_loss - standard_loss),
        "loss_rel": abs(fused_loss - standard_loss) / max(abs(standard_loss), 1.0e-30),
        "standard_grad_norm": float(jax.device_get(standard_metrics["grad_norm"])),
        "fused_grad_norm": float(jax.device_get(fused_metrics["grad_norm"])),
        "parameters": parameter_error.__dict__,
        "updates": update_error.__dict__,
        "optimizer_state": optimizer_error.__dict__,
    }
    print(json.dumps({"one_step_parity": record}, sort_keys=True), flush=True)
    if record["loss_rel"] > 3.0e-4:
        raise AssertionError(f"one-step loss relative error {record['loss_rel']:.3e}")
    if update_error.rel_l2 > 5.0e-3:
        raise AssertionError(f"one-step update relative L2 error {update_error.rel_l2:.3e}")
    if optimizer_error.rel_l2 > 5.0e-3:
        raise AssertionError(
            f"one-step optimizer-state relative L2 error {optimizer_error.rel_l2:.3e}"
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-tokens", type=int, default=1024)
    parser.add_argument("--skip-one-step", action="store_true")
    args = parser.parse_args()

    from yxtpu_pretrain.config import load_config
    from yxtpu_pretrain.runtime.environment import apply_hardware_environment

    config = load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
    )
    apply_hardware_environment(config.hardware)
    kernel_results = _kernel_parity(config, local_tokens=args.local_tokens)
    one_step = None if args.skip_one_step else _one_step_parity(config)
    print(
        json.dumps(
            {
                "status": "PASS",
                "kernel_cases": len(kernel_results),
                "one_step": one_step is not None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
