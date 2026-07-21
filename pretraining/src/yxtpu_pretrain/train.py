"""Owned NNX pretraining loop."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec
from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.utils import max_utils

from yxtpu_pretrain.config import ResolvedConfig
from yxtpu_pretrain.losses import data_parallel_linear_cross_entropy
from yxtpu_pretrain.model import (
    HybridLanguageModel,
    attention_logit_intermediates,
    count_parameters,
)
from yxtpu_pretrain.optimizers import (
    apply_gqa_muonclip,
    build_optimizer,
)
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO
from yxtpu_pretrain.runtime.data import create_data_iterator
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.metrics import MetricsWriter, WandbTracker
from yxtpu_pretrain.runtime.sharding import logical_mesh_context


def _loss(model: HybridLanguageModel, batch, *, record_max_logits: bool):
    hidden_states = model.hidden_states(
        batch["input_ids"],
        decoder_segment_ids=batch["segment_ids"],
        decoder_positions=batch["positions"],
        record_max_logits=record_max_logits,
    )
    weights = batch["loss_mask"].astype(jnp.float32)
    if model.config.model.loss.implementation == "tokamax_fused":
        hidden_flat = hidden_states.reshape((-1, hidden_states.shape[-1]))
        labels_flat = batch["labels"].reshape((-1,))
        weights_flat = weights.reshape((-1,))
        output_kernel = model.output_projection_kernel(hidden_states.dtype)
        loss, token_count = data_parallel_linear_cross_entropy(
            hidden_flat,
            labels_flat,
            weights_flat,
            output_kernel,
            mesh=model.mesh,
            implementation="mosaic_tpu",
        )
    else:
        logits = model.project_logits(hidden_states)
        targets = jax.nn.one_hot(batch["labels"], logits.shape[-1], dtype=jnp.float32)
        cross_entropy, _ = max_utils.cross_entropy_with_logits(logits, targets, z_loss=0.0)
        loss = jnp.sum(cross_entropy * weights) / jnp.maximum(jnp.sum(weights), 1.0)
        token_count = jnp.sum(weights)
    logits_max = (
        attention_logit_intermediates(model)
        if record_max_logits
        else jnp.zeros(
            (model.config.model.num_cycles, 1, model.config.model.attention.num_query_heads),
            dtype=jnp.float32,
        )
    )
    return loss, {"max_logits": logits_max, "tokens": token_count}


def _make_train_step(config: ResolvedConfig):
    accumulate = config.experiment.gradient_accumulation_steps
    use_clip = config.optimizer.name == "muonclip"

    def differentiated_loss(model, batch):
        return _loss(model, batch, record_max_logits=use_clip)

    # The train state is replaced by the updated NNX graph state on every call,
    # so its input buffers may be donated just as in MaxText's functional step.
    # Without donation the 272.9M baseline retains a second optimizer/model
    # buffer set and exceeds v6e HBM at the selected batch-8 operating point.
    @nnx.jit(donate_argnums=(0,))
    def train_step(state: TrainStateNNX, batch):
        microbatches = jax.tree.map(
            lambda value: value.reshape(
                (accumulate, value.shape[0] // accumulate, *value.shape[1:])
            ),
            batch,
        )
        accumulated_grads = None
        loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
        token_sum = jnp.asarray(0.0, dtype=jnp.float32)
        max_logits = jnp.full(
            (
                config.model.num_cycles,
                microbatches["input_ids"].shape[1],
                config.model.attention.num_query_heads,
            ),
            -jnp.inf,
            dtype=jnp.float32,
        )
        for microbatch_index in range(accumulate):
            microbatch = jax.tree.map(
                lambda value, index=microbatch_index: value[index], microbatches
            )
            (micro_loss, auxiliary), gradients = nnx.value_and_grad(
                differentiated_loss, has_aux=True
            )(
                state.model,
                microbatch,
            )
            accumulated_grads = (
                gradients
                if accumulated_grads is None
                else jax.tree.map(jnp.add, accumulated_grads, gradients)
            )
            loss_sum += micro_loss
            token_sum += auxiliary["tokens"]
            max_logits = jnp.maximum(max_logits, auxiliary["max_logits"])
        gradients = jax.tree.map(lambda value: value / accumulate, accumulated_grads)
        state.apply_gradients(gradients)
        clip_metrics = None
        if use_clip:
            clip_metrics = apply_gqa_muonclip(
                state.model,
                max_logits,
                tau=config.optimizer.qk_clip_tau,
                epsilon=config.optimizer.qk_clip_epsilon,
            )
        metrics = {
            "loss": loss_sum / accumulate,
            "tokens": token_sum,
            "grad_norm": max_utils.l2norm_pytree(gradients),
        }
        if clip_metrics is not None:
            metrics.update(
                {
                    "muonclip_max_logit": clip_metrics.max_logit,
                    "muonclip_min_scale": clip_metrics.min_scale,
                    "muonclip_clipped_heads": clip_metrics.clipped_heads,
                }
            )
        return metrics

    return train_step


def _make_eval_step():
    @nnx.jit
    def eval_step(model: HybridLanguageModel, batch):
        loss, auxiliary = _loss(model, batch, record_max_logits=False)
        return {"loss": loss, "tokens": auxiliary["tokens"]}

    return eval_step


def _tree_max_abs(tree):
    leaves = jax.tree.leaves(tree)
    return jnp.max(
        jnp.stack(
            [jnp.max(jnp.abs(leaf.astype(jnp.float32)), initial=0.0) for leaf in leaves]
        ),
        initial=0.0,
    )


def _make_diagnostics_step():
    """Builds a separate stability pass that never enters the timed train step."""

    @nnx.jit
    def diagnostics_step(model: HybridLanguageModel, batch):
        def diagnostic_loss(current_model):
            hidden = current_model.hidden_states(
                batch["input_ids"],
                decoder_segment_ids=batch["segment_ids"],
                decoder_positions=batch["positions"],
                record_max_logits=True,
            )
            logits = current_model.project_logits(hidden)
            targets = jax.nn.one_hot(batch["labels"], logits.shape[-1], dtype=jnp.float32)
            cross_entropy, _ = max_utils.cross_entropy_with_logits(
                logits,
                targets,
                z_loss=0.0,
            )
            weights = batch["loss_mask"].astype(jnp.float32)
            loss = jnp.sum(cross_entropy * weights) / jnp.maximum(jnp.sum(weights), 1.0)
            auxiliary = {
                "hidden_rms": jnp.sqrt(jnp.mean(jnp.square(hidden.astype(jnp.float32)))),
                "hidden_max_abs": jnp.max(jnp.abs(hidden.astype(jnp.float32))),
                "logits_max_abs": jnp.max(jnp.abs(logits)),
            }
            return loss, auxiliary

        (loss, auxiliary), gradients = nnx.value_and_grad(
            diagnostic_loss,
            has_aux=True,
        )(model)
        parameters = nnx.state(model, nnx.Param)
        return {
            "loss": loss,
            "grad_norm": max_utils.l2norm_pytree(gradients),
            "grad_max_abs": _tree_max_abs(gradients),
            "param_norm": max_utils.l2norm_pytree(parameters),
            "param_max_abs": _tree_max_abs(parameters),
            "hidden_rms": auxiliary["hidden_rms"],
            "hidden_max_abs": auxiliary["hidden_max_abs"],
            "logits_max_abs": auxiliary["logits_max_abs"],
            "attention_max_logits": attention_logit_intermediates(model),
        }

    return diagnostics_step


def _host_diagnostics(metrics) -> dict[str, float]:
    host = jax.device_get(metrics)
    attention = host.pop("attention_max_logits")
    result = {key: float(value) for key, value in host.items()}
    for cycle in range(attention.shape[0]):
        for head in range(attention.shape[-1]):
            result[f"attention/cycle_{cycle}/head_{head}_max_logit"] = float(
                attention[cycle, ..., head].max()
            )
    result["finite"] = float(all(math.isfinite(value) for value in result.values()))
    return result


def _device_batch(batch, mesh):
    if jax.process_count() > 1:
        return {
            key: multihost_utils.host_local_array_to_global_array(
                jnp.asarray(value),
                mesh,
                PartitionSpec("data", None),
            )
            for key, value in batch.items()
        }
    sharding = NamedSharding(mesh, PartitionSpec("data", None))
    return {
        key: jax.device_put(jnp.asarray(value), sharding)
        for key, value in batch.items()
    }


def _memory_summary() -> dict[str, int | float | None]:
    stats = []
    for device in jax.local_devices():
        try:
            memory = device.memory_stats()
        except Exception:
            memory = None
        if memory:
            stats.append(memory)
    peak = max((entry.get("peak_bytes_in_use", 0) for entry in stats), default=None)
    return {"peak_bytes_in_use": peak}


def _compiled_memory_summary(compiled) -> dict[str, int | None]:
    """Returns XLA's per-executable buffer assignment, including aliases."""
    stats = compiled.memory_analysis()
    if stats is None:
        return {"estimated_peak_bytes": None}
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    values = {field: int(getattr(stats, field, 0) or 0) for field in fields}
    values["estimated_peak_bytes"] = (
        values["argument_size_in_bytes"]
        + values["output_size_in_bytes"]
        + values["temp_size_in_bytes"]
        - values["alias_size_in_bytes"]
    )
    return values


def _run_name(config: ResolvedConfig) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{config.model.name}-{config.optimizer.name}-{config.experiment.name}"


def _process_batch_sizes(
    config: ResolvedConfig,
    *,
    local_device_count: int,
) -> tuple[int, int]:
    """Returns process-local train-update and evaluation batch sizes.

    ``per_device_batch_size`` is the microbatch size. A training iterator must
    provide one microbatch per accumulation step, while evaluation consumes one
    microbatch because it does not accumulate gradients.
    """
    process_microbatch = config.data.per_device_batch_size * local_device_count
    process_update_batch = (
        process_microbatch * config.experiment.gradient_accumulation_steps
    )
    return process_update_batch, process_microbatch


def _learning_rate(config: ResolvedConfig, step: int) -> float:
    """Host-side mirror of the Optax schedule, avoiding a TPU dispatch for logging."""
    optimizer = config.optimizer
    count = max(step - 1, 0)
    if count < optimizer.warmup_steps:
        return optimizer.learning_rate * count / max(optimizer.warmup_steps, 1)
    decay_steps = optimizer.schedule_steps - optimizer.warmup_steps
    progress = min(max(count - optimizer.warmup_steps, 0) / decay_steps, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return optimizer.learning_rate * (
        optimizer.final_learning_rate_fraction
        + (1.0 - optimizer.final_learning_rate_fraction) * cosine
    )


def run(
    config: ResolvedConfig,
    *,
    benchmark_only: bool = False,
    profile: bool = False,
) -> int:
    del benchmark_only
    if config.hardware.multi_host and not jax.distributed.is_initialized():
        jax.distributed.initialize()
    mesh = create_mesh(config.hardware)
    logical_axis_rules = make_leaf_config(config).logical_axis_rules
    train_process_batch, eval_process_batch = _process_batch_sizes(
        config,
        local_device_count=jax.local_device_count(),
    )
    process_data = config.data.model_copy(
        update={"shuffle_seed": config.data.shuffle_seed + 1_000_003 * jax.process_index()}
    )
    data_iterator = create_data_iterator(
        process_data,
        global_batch_size=train_process_batch,
        vocab_size=config.model.vocab_size,
    )
    eval_iterator = (
        create_data_iterator(
            process_data.model_copy(update={"split": config.data.eval_split}),
            global_batch_size=eval_process_batch,
            vocab_size=config.model.vocab_size,
            validation=config.data.streaming,
        )
        if config.data.eval_interval
        else None
    )
    with logical_mesh_context(mesh, logical_axis_rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(config.experiment.seed))
        transform, routes = build_optimizer(model, config.optimizer)
        optimizer = nnx.Optimizer(model, transform, wrt=nnx.Param)
        state = TrainStateNNX(model, optimizer)

    run_name = _run_name(config)
    run_dir = Path(config.experiment.run_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yml").write_text(config.to_yaml(), encoding="utf-8")
    (run_dir / "optimizer_routes.json").write_text(
        json.dumps(
            [
                {
                    **route.__dict__,
                    "path": list(route.path),
                    "role": str(route.role),
                    "shape": list(route.shape),
                }
                for route in routes
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    data_metadata = dict(getattr(data_iterator, "metadata", {}))
    (run_dir / "data_metadata.json").write_text(
        json.dumps(data_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_writer = MetricsWriter(run_dir)
    checkpoint_io = CheckpointIO(
        config,
        run_name=f"{config.model.name}-{config.optimizer.name}-{config.experiment.name}",
    )
    with logical_mesh_context(mesh, logical_axis_rules):
        start_step = (
            checkpoint_io.restore(state, data_iterator)
            if config.experiment.checkpoint.resume
            else 0
        )
        train_step = _make_train_step(config)
        eval_step = _make_eval_step()
        diagnostics_step = _make_diagnostics_step()
        first_batch = _device_batch(next(data_iterator), mesh)
        compiled_train_step = train_step.lower(state, first_batch).compile()
        compiled_memory = _compiled_memory_summary(compiled_train_step)
    parameter_count = count_parameters(state.model)
    tracker = WandbTracker(
        config,
        run_name=run_name,
        run_dir=run_dir,
        metadata={
            "compiled_memory": compiled_memory,
            "parameter_count": parameter_count,
            "jax_device_count": jax.device_count(),
            "jax_process_count": jax.process_count(),
            "data": data_metadata,
        },
    )
    harness_adapter = None
    if config.experiment.harness_eval.enabled:
        from yxtpu_pretrain.evaluation import JaxHarnessLM

        harness_adapter = JaxHarnessLM(config, state.model, mesh, logical_axis_rules)
    print(json.dumps({"compiled_memory": compiled_memory}, sort_keys=True), flush=True)
    metrics_writer.write({"compiled_memory": compiled_memory})
    throughputs = []
    losses = []
    tokens_seen = 0
    completed_steps = start_step
    trace_active = False
    try:
        for step in range(start_step + 1, config.experiment.steps + 1):
            if profile and step == min(config.experiment.profile_steps):
                jax.profiler.start_trace(str(run_dir / "profile"))
                trace_active = True
            batch = (
                first_batch
                if step == start_step + 1
                else _device_batch(next(data_iterator), mesh)
            )
            started = time.perf_counter()
            with logical_mesh_context(mesh, logical_axis_rules):
                metrics = compiled_train_step(state, batch)
                jax.block_until_ready(metrics)
            elapsed = time.perf_counter() - started
            host_metrics = jax.device_get(metrics)
            tokens = float(host_metrics["tokens"])
            tokens_seen += int(tokens)
            throughput = tokens / elapsed
            loss = float(host_metrics["loss"])
            grad_norm = float(host_metrics["grad_norm"])
            record = {
                "step": step,
                "loss": loss,
                "tokens": int(tokens),
                "step_ms": elapsed * 1_000,
                "tokens_per_second": throughput,
                "grad_norm": grad_norm,
                "learning_rate": _learning_rate(config, step),
                "tokens_seen": tokens_seen,
            }
            if "muonclip_max_logit" in host_metrics:
                record["muonclip"] = {
                    "max_logit": host_metrics["muonclip_max_logit"].tolist(),
                    "min_scale": host_metrics["muonclip_min_scale"].tolist(),
                    "clipped_heads": host_metrics["muonclip_clipped_heads"].tolist(),
                }
            metrics_writer.write(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if step % config.experiment.log_interval == 0:
                tracker.log(
                    {
                        "train": {
                            "loss": loss,
                            "perplexity": math.exp(min(loss, 80.0)),
                        },
                        "performance": {
                            "tokens_per_second": throughput,
                            "step_ms": elapsed * 1_000,
                        },
                        "optimizer": {
                            "grad_norm": grad_norm,
                            "learning_rate": record["learning_rate"],
                        },
                        "stability": {
                            "loss_finite": float(math.isfinite(loss)),
                            "grad_norm_finite": float(math.isfinite(grad_norm)),
                        },
                    },
                    step=step,
                    tokens_seen=tokens_seen,
                )
            losses.append(loss)
            completed_steps = step
            if step > start_step + 5:
                throughputs.append(throughput)

            if eval_iterator is not None and step % config.data.eval_interval == 0:
                eval_loss_sum = 0.0
                eval_token_sum = 0.0
                diagnostic_batch = None
                for _ in range(config.data.eval_steps):
                    diagnostic_batch = _device_batch(next(eval_iterator), mesh)
                    with logical_mesh_context(mesh, logical_axis_rules):
                        eval_metrics = eval_step(
                            state.model,
                            diagnostic_batch,
                        )
                    eval_host = jax.device_get(eval_metrics)
                    eval_tokens = float(eval_host["tokens"])
                    eval_loss_sum += float(eval_host["loss"]) * eval_tokens
                    eval_token_sum += eval_tokens
                evaluation_loss = eval_loss_sum / max(eval_token_sum, 1.0)
                evaluation_record = {
                    "step": step,
                    "evaluation_loss": evaluation_loss,
                    "evaluation_tokens": int(eval_token_sum),
                }
                metrics_writer.write(evaluation_record)
                print(json.dumps(evaluation_record, sort_keys=True), flush=True)
                tracker.log(
                    {
                        "eval": {
                            "train_holdout_loss": evaluation_loss,
                            "train_holdout_perplexity": math.exp(min(evaluation_loss, 80.0)),
                            "tokens": int(eval_token_sum),
                        }
                    },
                    step=step,
                    tokens_seen=tokens_seen,
                )

                diagnostics = config.experiment.diagnostics
                if diagnostics.enabled and step % diagnostics.interval == 0:
                    with logical_mesh_context(mesh, logical_axis_rules):
                        diagnostic_metrics = diagnostics_step(state.model, diagnostic_batch)
                        jax.block_until_ready(diagnostic_metrics)
                    host_diagnostics = _host_diagnostics(diagnostic_metrics)
                    diagnostics_record = {"step": step, "diagnostics": host_diagnostics}
                    metrics_writer.write(diagnostics_record)
                    print(json.dumps(diagnostics_record, sort_keys=True), flush=True)
                    tracker.log(
                        {"diagnostics": host_diagnostics},
                        step=step,
                        tokens_seen=tokens_seen,
                    )

            harness = config.experiment.harness_eval
            if harness_adapter is not None and step % harness.interval == 0:
                from yxtpu_pretrain.evaluation import run_harness_evaluation

                evaluation_started = time.perf_counter()
                harness_metrics, harness_path = run_harness_evaluation(
                    harness_adapter,
                    config,
                    run_dir=run_dir,
                    step=step,
                )
                harness_metrics["duration_seconds"] = time.perf_counter() - evaluation_started
                harness_record = {
                    "step": step,
                    "lm_eval": harness_metrics,
                    "artifact": str(harness_path),
                }
                metrics_writer.write(harness_record)
                print(json.dumps(harness_record, sort_keys=True), flush=True)
                tracker.log(
                    {"lm_eval": harness_metrics},
                    step=step,
                    tokens_seen=tokens_seen,
                )
                tracker.log_artifact(
                    harness_path,
                    name=f"{run_name}-lm-eval-step-{step}",
                    artifact_type="lm-eval-results",
                )

            interval = config.experiment.checkpoint.save_interval
            if checkpoint_io.enabled and interval and step % interval == 0:
                checkpoint_io.save(state, data_iterator, step)
            if trace_active and step == max(config.experiment.profile_steps):
                jax.profiler.stop_trace()
                trace_active = False
            if (
                config.experiment.token_budget is not None
                and tokens_seen >= config.experiment.token_budget
            ):
                break
        if checkpoint_io.enabled:
            checkpoint_io.save(
                state,
                data_iterator,
                completed_steps,
                force=True,
            )
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    finally:
        if trace_active:
            jax.profiler.stop_trace()
        checkpoint_io.close()

    summary = {
        "steps": completed_steps - start_step,
        "tokens_seen": tokens_seen,
        "token_budget": config.experiment.token_budget,
        "final_loss": losses[-1] if losses else None,
        "mean_tokens_per_second": statistics.mean(throughputs) if throughputs else None,
        "max_tokens_per_second": max(throughputs) if throughputs else None,
        "memory": _memory_summary(),
        "compiled_memory": compiled_memory,
        "jax_process_count": jax.process_count(),
        "jax_device_count": jax.device_count(),
        "microbatch_size_per_device": config.data.per_device_batch_size,
        "gradient_accumulation_steps": config.experiment.gradient_accumulation_steps,
        "effective_batch_size_per_device": (
            config.data.per_device_batch_size
            * config.experiment.gradient_accumulation_steps
        ),
        "effective_global_batch_size": (
            config.data.per_device_batch_size
            * config.experiment.gradient_accumulation_steps
            * jax.device_count()
        ),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "parameter_count": parameter_count,
        "wandb_url": tracker.url,
    }
    metrics_writer.close(summary)
    tracker.finish(summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
