"""Owned NNX pretraining loop."""

from __future__ import annotations

import json
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
from yxtpu_pretrain.model import HybridLanguageModel, attention_logit_intermediates
from yxtpu_pretrain.optimizers import apply_gqa_muonclip, build_optimizer
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO
from yxtpu_pretrain.runtime.data import create_data_iterator
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.metrics import MetricsWriter


def _loss(model: HybridLanguageModel, batch, *, record_max_logits: bool):
    logits = model(
        batch["input_ids"],
        decoder_segment_ids=batch["segment_ids"],
        decoder_positions=batch["positions"],
        record_max_logits=record_max_logits,
    )
    targets = jax.nn.one_hot(batch["labels"], logits.shape[-1], dtype=jnp.float32)
    cross_entropy, _ = max_utils.cross_entropy_with_logits(logits, targets, z_loss=0.0)
    weights = batch["loss_mask"].astype(jnp.float32)
    loss = jnp.sum(cross_entropy * weights) / jnp.maximum(jnp.sum(weights), 1.0)
    logits_max = (
        attention_logit_intermediates(model)
        if record_max_logits
        else jnp.zeros(
            (model.config.model.num_cycles, 1, model.config.model.attention.num_query_heads),
            dtype=jnp.float32,
        )
    )
    return loss, {"max_logits": logits_max, "tokens": jnp.sum(weights)}


def _make_train_step(config: ResolvedConfig):
    accumulate = config.experiment.gradient_accumulation_steps
    use_clip = config.optimizer.name == "muonclip"

    def differentiated_loss(model, batch):
        return _loss(model, batch, record_max_logits=use_clip)

    @nnx.jit
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


def _run_name(config: ResolvedConfig) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{config.model.name}-{config.optimizer.name}-{config.experiment.name}"


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
    global_batch = config.data.per_device_batch_size * config.hardware.device_count
    local_batch = config.data.per_device_batch_size * jax.local_device_count()
    if global_batch % config.experiment.gradient_accumulation_steps:
        raise ValueError("global batch size must be divisible by gradient accumulation steps")
    process_data = config.data.model_copy(
        update={"shuffle_seed": config.data.shuffle_seed + 1_000_003 * jax.process_index()}
    )
    data_iterator = create_data_iterator(
        process_data,
        global_batch_size=local_batch,
        vocab_size=config.model.vocab_size,
    )
    eval_iterator = (
        create_data_iterator(
            process_data.model_copy(update={"split": config.data.eval_split}),
            global_batch_size=local_batch,
            vocab_size=config.model.vocab_size,
        )
        if config.data.eval_interval
        else None
    )
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
    metrics_writer = MetricsWriter(run_dir)
    checkpoint_io = CheckpointIO(
        config,
        run_name=f"{config.model.name}-{config.optimizer.name}-{config.experiment.name}",
    )
    start_step = (
        checkpoint_io.restore(state, data_iterator)
        if config.experiment.checkpoint.resume
        else 0
    )

    train_step = _make_train_step(config)
    eval_step = _make_eval_step()
    throughputs = []
    losses = []
    trace_active = False
    try:
        for step in range(start_step + 1, config.experiment.steps + 1):
            if profile and step == min(config.experiment.profile_steps):
                jax.profiler.start_trace(str(run_dir / "profile"))
                trace_active = True
            batch = _device_batch(next(data_iterator), mesh)
            started = time.perf_counter()
            metrics = train_step(state, batch)
            jax.block_until_ready(metrics)
            elapsed = time.perf_counter() - started
            tokens = float(metrics["tokens"])
            throughput = tokens / elapsed
            loss = float(metrics["loss"])
            record = {
                "step": step,
                "loss": loss,
                "tokens": int(tokens),
                "step_ms": elapsed * 1_000,
                "tokens_per_second": throughput,
                "grad_norm": float(metrics["grad_norm"]),
            }
            if "muonclip_max_logit" in metrics:
                record["muonclip"] = {
                    "max_logit": jax.device_get(metrics["muonclip_max_logit"]).tolist(),
                    "min_scale": jax.device_get(metrics["muonclip_min_scale"]).tolist(),
                    "clipped_heads": jax.device_get(
                        metrics["muonclip_clipped_heads"]
                    ).tolist(),
                }
            metrics_writer.write(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            losses.append(loss)
            if step > start_step + 5:
                throughputs.append(throughput)

            if eval_iterator is not None and step % config.data.eval_interval == 0:
                eval_losses = []
                for _ in range(config.data.eval_steps):
                    eval_metrics = eval_step(
                        state.model,
                        _device_batch(next(eval_iterator), mesh),
                    )
                    eval_losses.append(float(eval_metrics["loss"]))
                metrics_writer.write(
                    {"step": step, "evaluation_loss": statistics.mean(eval_losses)}
                )

            interval = config.experiment.checkpoint.save_interval
            if checkpoint_io.enabled and interval and step % interval == 0:
                checkpoint_io.save(state, data_iterator, step)
            if trace_active and step == max(config.experiment.profile_steps):
                jax.profiler.stop_trace()
                trace_active = False
        if checkpoint_io.enabled:
            checkpoint_io.save(
                state,
                data_iterator,
                config.experiment.steps,
                force=True,
            )
    finally:
        if trace_active:
            jax.profiler.stop_trace()
        checkpoint_io.close()

    summary = {
        "steps": config.experiment.steps - start_step,
        "final_loss": losses[-1] if losses else None,
        "mean_tokens_per_second": statistics.mean(throughputs) if throughputs else None,
        "max_tokens_per_second": max(throughputs) if throughputs else None,
        "memory": _memory_summary(),
        "jax_process_count": jax.process_count(),
        "jax_device_count": jax.device_count(),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    }
    metrics_writer.close(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
