"""SFT trainer: init from a pretraining checkpoint, train on packed chat."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.metrics import MetricsWriter, NullMetricsWriter, WandbTracker
from yxtpu_pretrain.runtime.sharding import logical_mesh_context
from yxtpu_pretrain.sft.checkpoint import save_sft_checkpoint
from yxtpu_pretrain.sft.data import SFTIterator, StreamingSFTIterator, build_packed_dataset
from yxtpu_pretrain.sft.tokens import SPECIAL_TOKENS, load_sft_tokenizer
from yxtpu_pretrain.train import _device_batch, _learning_rate, _make_train_step


class _NoIterator:
    def set_state(self, payload):
        raise AssertionError("stream state must not restore during SFT init")


def _reinit_new_token_rows(model):
    """New chat-token rows start at the mean of the trained vocabulary."""
    embedding = model.token_embedding.embedding
    table = embedding.get_value()
    trained = table[:128001].astype(jnp.float32)
    mean_row = jnp.mean(trained, axis=0, dtype=jnp.float32)
    new_ids = jnp.asarray([token_id for _, token_id in SPECIAL_TOKENS])
    table = table.at[new_ids].set(mean_row.astype(table.dtype))
    embedding.set_value(table)
    return [int(i) for i in new_ids]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned")
    parser.add_argument("--subset", default="General-Distillation")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--sources", default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--init-destination", default="/home/a1111/yxtpu_ckpts")
    parser.add_argument("--init-run", default="kda_hybrid_128k-muonclip-superbpe_50b")
    parser.add_argument("--out-destination", default="/home/a1111/yxtpu_sft_ckpts")
    parser.add_argument("--steps-cap", type=int, default=2000)
    parser.add_argument("--set", action="append", dest="overrides", default=[])
    args = parser.parse_args()

    base_overrides = [
        f"experiment.steps={args.steps_cap}",
        "experiment.token_budget=null",
        "experiment.harness_eval.enabled=false",
        "experiment.diagnostics.enabled=false",
        f"experiment.checkpoint.destination={args.out_destination}",
        "experiment.checkpoint.save_interval=250",
        "experiment.checkpoint.keep=4",
        "experiment.checkpoint.resume=false",
        "experiment.wandb.group=sft-general-100k",
        "experiment.wandb.tags=[v4-64, sft, kimi-k25-distill]",
    ]
    config = load_config(
        model="kda_hybrid_128k", optimizer="muonclip", data="climbmix_superbpe",
        hardware="v4-64", experiment="superbpe_50b",
        overrides=base_overrides + list(args.overrides or []),
    )
    mesh = create_mesh(config.hardware)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(config.experiment.seed))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))

    init_config = config.model_copy(deep=True)
    init_config.experiment.checkpoint.destination = args.init_destination
    init_config.experiment.checkpoint.enabled = True
    loader = CheckpointIO(init_config, run_name=args.init_run)
    start = loader.restore(state, _NoIterator())
    loader.close()
    if start == 0:
        raise RuntimeError("no pretraining checkpoint found to initialize from")
    with logical_mesh_context(mesh, rules):
        state.optimizer = nnx.Optimizer(model, transform, wrt=nnx.Param)
        new_rows = _reinit_new_token_rows(model)
    is_primary = jax.process_index() == 0
    if is_primary:
        print(f"initialized from step {start}; re-initialized rows {new_rows}", flush=True)

    tokenizer = load_sft_tokenizer(config.data.tokenizer, padded_vocab_size=config.model.vocab_size)
    process_batch = config.data.per_device_batch_size * jax.local_device_count()
    if args.stream:
        iterator = StreamingSFTIterator(
            tokenizer, dataset=args.dataset,
            sequence_length=config.data.sequence_length,
            process_batch=process_batch,
            process_index=jax.process_index(), process_count=jax.process_count(),
            sources=args.sources.split(",") if args.sources else None,
            shuffle_seed=args.shuffle_seed,
        )
        if is_primary:
            print("streaming full dataset", flush=True)
        run_packed = False
    else:
        run_packed = True
    if run_packed:
        inputs, labels, loss_mask = build_packed_dataset(
        tokenizer, dataset=args.dataset, subset=args.subset,
        rows=args.rows, sequence_length=config.data.sequence_length,
    )
        iterator = SFTIterator(
            inputs, labels, loss_mask,
            process_batch=process_batch, epochs=args.epochs,
            seed=config.experiment.seed,
            process_index=jax.process_index(), process_count=jax.process_count(),
        )
        if is_primary:
            print(f"packed rows: {len(inputs)}, tokens/epoch ~{len(inputs)*inputs.shape[1]:,}", flush=True)

    train_step = _make_train_step(config)
    run_name = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + f"-sft-{config.model.name}-{args.subset.lower()}"
    )
    run_dir = Path(config.experiment.run_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_writer = MetricsWriter(run_dir) if is_primary else NullMetricsWriter()
    tracker = WandbTracker(config, run_name=run_name, run_dir=run_dir, metadata={"sft": vars(args) | {"new_rows": new_rows, "init_step": start}})
    save_dir = Path(args.out_destination) / run_name

    step = 0
    tokens_seen = 0
    try:
        for batch_host in iterator:
            step += 1
            if step > config.experiment.steps:
                step -= 1
                break
            batch = _device_batch(batch_host, mesh)
            began = time.perf_counter()
            with logical_mesh_context(mesh, rules):
                metrics = train_step(state, batch)
                jax.block_until_ready(metrics)
            host = jax.device_get(metrics)
            loss = float(host["loss"])
            tokens = float(host["tokens"])
            tokens_seen += int(tokens)
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite SFT loss at step {step}")
            record = {
                "step": step, "loss": loss, "tokens": int(tokens),
                "tokens_seen": tokens_seen,
                "step_ms": (time.perf_counter() - began) * 1000,
                "grad_norm": float(host["grad_norm"]),
                "learning_rate": _learning_rate(config, step),
            }
            metrics_writer.write(record)
            if is_primary:
                print(json.dumps(record, sort_keys=True), flush=True)
            tracker.log(
                {"train": {"loss": loss}, "optimizer": {
                    "grad_norm": record["grad_norm"],
                    "learning_rate": record["learning_rate"]}},
                step=step, tokens_seen=tokens_seen,
            )
            if step % config.experiment.checkpoint.save_interval == 0:
                save_sft_checkpoint(save_dir, step, state, iterator, config)
        save_sft_checkpoint(save_dir, step, state, iterator, config)
    finally:
        summary = {"steps": step, "tokens_seen": tokens_seen, "final_loss": loss if step else None}
        metrics_writer.close(summary)
        tracker.finish(summary=summary)
        if is_primary:
            print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
