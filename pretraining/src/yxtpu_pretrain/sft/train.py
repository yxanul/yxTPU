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
from yxtpu_pretrain.runtime.data import PrefetchIterator
from yxtpu_pretrain.sft.checkpoint import save_sft_checkpoint
from yxtpu_pretrain.sft.data import SFTIterator, StreamingSFTIterator, build_packed_dataset
from yxtpu_pretrain.sft.tokens import SPECIAL_TOKENS, load_sft_tokenizer
from yxtpu_pretrain.train import _device_batch, _learning_rate, _make_train_step


class _NoIterator:
    def set_state(self, payload):
        raise AssertionError("stream state must not restore during SFT init")


def _reinit_new_token_rows(model, new_ids=None, trained_upto=128001):
    """New chat-token rows start at the mean of the trained vocabulary.

    Defaults describe the 128k K2.5 scheme. The yx49k tokenizer carries its
    chat specials natively at 49120-49151, which pretraining never emitted,
    so those rows need the same treatment against a 49119 boundary.
    """
    embedding = model.token_embedding.embedding
    table = embedding.get_value()
    trained = table[:trained_upto].astype(jnp.float32)
    mean_row = jnp.mean(trained, axis=0, dtype=jnp.float32)
    if new_ids is None:
        new_ids = [token_id for _, token_id in SPECIAL_TOKENS]
    new_ids = jnp.asarray(new_ids)
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
    parser.add_argument(
        "--mephisto", default=None,
        help="comma list of Mephisto repos; selects the yx49k native Qwen "
             "chat path instead of the K2.5 scheme")
    parser.add_argument("--system", default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--model", default="kda_hybrid_128k")
    parser.add_argument("--data", default="climbmix_superbpe")
    parser.add_argument(
        "--mixture", default=None,
        help="interleave configs of --dataset by record probability, "
             "e.g. 'IF:0.28,Math:0.43,Knowledge:0.18,Code:0.11'")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-render-tokens", type=int, default=None)
    parser.add_argument("--pack-whole", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--init-destination", default="/home/a1111/yxtpu_ckpts")
    parser.add_argument("--init-run", default="kda_hybrid_128k-muonclip-superbpe_50b")
    parser.add_argument(
        "--init-pickle", default=None,
        help="init weights from an SFT stage state.pkl instead of a "
             "pretraining orbax checkpoint. Skips the new-token-row "
             "re-init - an SFT checkpoint already trained those rows. The "
             "optimizer starts fresh either way.")
    parser.add_argument(
        "--allow-device-mismatch", action="store_true",
        help="run on fewer devices than the hardware profile declares "
             "(single-host smokes); the mesh degrades to pure data parallel")
    parser.add_argument("--out-destination", default="/home/a1111/yxtpu_sft_ckpts")
    parser.add_argument("--steps-cap", type=int, default=2000)
    parser.add_argument(
        "--gold-targets", default=None,
        help="directory of precomputed teacher targets "
             "(benchmarks/precompute_gold_targets.py); switches the loss to "
             "the GOLD objective. Mephisto path only.")
    parser.add_argument("--gold-beta", type=float, default=0.0)
    parser.add_argument("--gold-distill-weight", type=float, default=1.0)
    parser.add_argument("--gold-ce-weight", type=float, default=0.0)
    parser.add_argument("--set", action="append", dest="overrides", default=[])
    args = parser.parse_args()
    if args.gold_targets and not args.mephisto:
        parser.error("--gold-targets requires --mephisto")

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
        model=args.model, optimizer="muonclip", data=args.data,
        hardware="v4-64", experiment="superbpe_50b",
        overrides=base_overrides + list(args.overrides or []),
    )
    mesh = create_mesh(config.hardware,
                       allow_device_mismatch=args.allow_device_mismatch)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(config.experiment.seed))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))

    is_primary = jax.process_index() == 0
    if args.init_pickle:
        import pickle

        from yxtpu_pretrain.runtime.checkpoints import _persistent_state

        target = _persistent_state(state)
        with open(args.init_pickle, "rb") as handle:
            nnx.replace_by_pure_dict(target, pickle.load(handle))
        nnx.update(state, target)
        start = args.init_pickle
        new_rows = []
        with logical_mesh_context(mesh, rules):
            state.optimizer = nnx.Optimizer(model, transform, wrt=nnx.Param)
    else:
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
            if args.mephisto:
                from yxtpu_pretrain.sft.mephisto import UNTRAINED_SPECIAL_RANGE

                low, high = UNTRAINED_SPECIAL_RANGE
                new_rows = _reinit_new_token_rows(
                    model, new_ids=list(range(low, high)), trained_upto=low
                )
            else:
                new_rows = _reinit_new_token_rows(model)
    if is_primary:
        print(f"initialized from {start}; re-initialized rows {new_rows}", flush=True)

    if args.mephisto:
        # yx49k ships the Qwen3.5 chat template verbatim - no special ids to
        # append, so the plain tokenizer is already the chat tokenizer.
        from yxtpu_pretrain.runtime.data import load_fast_tokenizer

        tokenizer = load_fast_tokenizer(
            config.data.tokenizer, padded_vocab_size=config.model.vocab_size)
    else:
        tokenizer = load_sft_tokenizer(
            config.data.tokenizer, padded_vocab_size=config.model.vocab_size)
    process_batch = config.data.per_device_batch_size * jax.local_device_count()
    if args.mephisto:
        from yxtpu_pretrain.sft.mephisto import MephistoIterator

        gold_store = None
        if args.gold_targets:
            from yxtpu_pretrain.distillation.store import GoldTargetStore

            gold_store = GoldTargetStore(args.gold_targets)
            if is_primary:
                print(f"gold targets: {len(gold_store)} examples, "
                      f"k={gold_store.k} from {args.gold_targets}", flush=True)
        iterator = MephistoIterator(
            tokenizer,
            datasets=[s.strip() for s in args.mephisto.split(",") if s.strip()],
            sequence_length=config.data.sequence_length,
            process_batch=process_batch,
            process_index=jax.process_index(),
            process_count=jax.process_count(),
            epochs=args.epochs,
            system=args.system,
            shuffle_buffer=args.shuffle_buffer,
            seed=config.experiment.seed,
            targets=gold_store,
        )
        # Same rationale as the stream path below: rendering, store lookup
        # and packing consumed synchronously stall the whole slice at every
        # pool drain, and on a multi-host run the other hosts inherit the
        # stall through the first collective. The smoke measured
        # 230-540 ms/step of serially exposed data wait without this.
        iterator = PrefetchIterator(iterator, depth=4)
        if is_primary:
            print(f"mephisto SFT: {args.mephisto} x{args.epochs} epochs", flush=True)
        run_packed = False
    elif args.stream:
        mixture = None
        if args.mixture:
            mixture = []
            for entry in args.mixture.split(","):
                name, probability = entry.rsplit(":", 1)
                mixture.append((name.strip(), float(probability)))
            total = sum(probability for _, probability in mixture)
            mixture = [(name, probability / total) for name, probability in mixture]
        iterator = StreamingSFTIterator(
            tokenizer, dataset=args.dataset,
            sequence_length=config.data.sequence_length,
            process_batch=process_batch,
            process_index=jax.process_index(), process_count=jax.process_count(),
            sources=args.sources.split(",") if args.sources else None,
            shuffle_seed=args.shuffle_seed,
            mixture=mixture,
            split=args.split,
            max_render_tokens=args.max_render_tokens,
            pack_whole=args.pack_whole,
        )
        # The refill (stream draw + render + tokenize) must overlap the
        # device step: consumed synchronously it stalls the whole slice at
        # every pool drain, and the other seven hosts inherit the stall
        # through the first collective.
        iterator = PrefetchIterator(iterator, depth=4)
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

    gold_loss_fn = None
    if args.gold_targets:
        from yxtpu_pretrain.distillation.objective import make_gold_model_loss

        gold_loss_fn = make_gold_model_loss(
            beta=args.gold_beta,
            distill_weight=args.gold_distill_weight,
            ce_weight=args.gold_ce_weight,
        )
    train_step = _make_train_step(config, loss_fn=gold_loss_fn)
    run_name = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + f"-sft-{config.model.name}-"
        + ("gold" if args.gold_targets
           else "mephisto" if args.mephisto else args.subset.lower())
    )
    run_dir = Path(config.experiment.run_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_writer = MetricsWriter(run_dir) if is_primary else NullMetricsWriter()
    tracker = WandbTracker(config, run_name=run_name, run_dir=run_dir, metadata={"sft": vars(args) | {"new_rows": new_rows, "init_step": start}})
    save_dir = Path(args.out_destination) / run_name

    step = 0
    tokens_seen = 0
    try:
        while True:
            wait_began = time.perf_counter()
            try:
                batch_host = next(iterator)
                has_data = 1
            except StopIteration:
                batch_host, has_data = None, 0
            if jax.process_count() > 1:
                # Termination must be COLLECTIVE. Per-host shards exhaust at
                # different step counts (equal rows, unequal tokens), and a
                # host that breaks alone leaves the rest hanging forever in
                # their next train_step collective - the mix6m run deadlocked
                # exactly there, at step 4734, costing its final save. Every
                # host reports whether it still has data and all of them stop
                # together the moment any one is dry; at most one drawn batch
                # per surviving host is discarded.
                from jax.experimental import multihost_utils

                flags = multihost_utils.process_allgather(
                    np.asarray([has_data], dtype=np.int32))
                has_data = int(flags.min())
            if not has_data:
                break
            data_wait_ms = (time.perf_counter() - wait_began) * 1000
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
                "data_wait_ms": data_wait_ms,
                "queue_depth": getattr(iterator, "queue_depth", None),
                "grad_norm": float(host["grad_norm"]),
                "learning_rate": _learning_rate(config, step),
            }
            for gold_key in ("ce", "distill", "teacher_rest_mass",
                             "teacher_top1_is_label", "student_top1_is_label"):
                if gold_key in host:
                    record[gold_key] = float(host[gold_key])
            if step % 25 == 0 or step == 1:
                record["data"] = dict(getattr(iterator, "stats", {}))
            metrics_writer.write(record)
            if is_primary:
                print(json.dumps(record, sort_keys=True), flush=True)
            payload = {
                "train": {"loss": loss} | {
                    key: record[key]
                    for key in ("ce", "distill", "teacher_rest_mass")
                    if key in record
                },
                "optimizer": {
                    "grad_norm": record["grad_norm"],
                    "learning_rate": record["learning_rate"]},
                "perf": {
                    "step_ms": record["step_ms"],
                    "data_wait_ms": data_wait_ms,
                    "queue_depth": record["queue_depth"] or 0},
            }
            if "data" in record:
                payload["data"] = record["data"]
            tracker.log(payload, step=step, tokens_seen=tokens_seen)
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
