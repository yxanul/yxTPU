"""Owned NNX pretraining loop."""

from __future__ import annotations

import gc
import json
import math
import os
import queue
import statistics
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec
from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.utils import max_utils

from yxtpu_pretrain.config import ResolvedConfig
from yxtpu_pretrain.losses import (
    chunked_linear_cross_entropy,
    data_parallel_linear_cross_entropy,
)
from yxtpu_pretrain.layers.nope_gqa import ABSENT_LOGIT
from yxtpu_pretrain.model import (
    HybridLanguageModel,
    attention_logit_intermediates,
    attention_modality_logit_intermediates,
    count_parameters,
    residual_probe_intermediates,
    vision_probe_intermediates,
)
from yxtpu_pretrain.optimizers import (
    apply_gqa_muonclip,
    build_optimizer,
)
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO
from yxtpu_pretrain.runtime.data import create_data_iterator
from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.metrics import (
    HostMetricsWriter,
    MetricsWriter,
    NullMetricsWriter,
    WandbTracker,
)
from yxtpu_pretrain.runtime.sharding import logical_mesh_context


def _loss(model: HybridLanguageModel, batch, *, record_max_logits: bool):
    hidden_states = model.hidden_states(
        batch["input_ids"],
        images=batch["images"] if "images" in batch else None,
        decoder_segment_ids=batch["segment_ids"],
        decoder_positions=batch["positions"],
        record_max_logits=record_max_logits,
    )
    weights = batch["loss_mask"].astype(jnp.float32)
    # Per-modality loss split: vision_mask (mixed vision+text batches only)
    # marks label tokens of image-carrying rows; the chunked loss reduces both
    # splits inside the same block pass. Reporting only — the differentiated
    # loss is unchanged.
    vision_mask = batch.get("vision_mask")
    use_split = (
        vision_mask is not None
        and model.config.model.loss.implementation == "chunked"
    )
    vision_aux = {}
    if use_split:
        loss, token_count, split = chunked_linear_cross_entropy(
            hidden_states,
            batch["labels"],
            weights,
            model.output_projection_kernel(hidden_states.dtype),
            block_tokens=model.config.model.loss.block_tokens,
            split_mask=vision_mask,
        )
        vision_aux["vision_loss_sum"] = split["loss_sum"]
        vision_aux["vision_token_count"] = split["token_count"]
        vision_aux["text_loss_sum"] = split["total_loss_sum"] - split["loss_sum"]
        vision_aux["text_token_count"] = token_count - split["token_count"]
    elif model.config.model.loss.implementation == "chunked":
        loss, token_count = chunked_linear_cross_entropy(
            hidden_states,
            batch["labels"],
            weights,
            model.output_projection_kernel(hidden_states.dtype),
            block_tokens=model.config.model.loss.block_tokens,
        )
    elif model.config.model.loss.implementation == "tokamax_fused":
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
    if record_max_logits and model.vision_tower is not None:
        # [cycles, 2, heads]: attention maxima over visual / text query
        # positions - the train-batch companion of the joint QK-clip maxima
        # (a text-only diagnostics batch cannot show where the hot logits
        # of a mixed batch live).
        vision_aux["max_logits_by_modality"] = attention_modality_logit_intermediates(
            model
        )
    if "images" in batch and model.vision_probe is not None:
        probe = vision_probe_intermediates(model)
        vision_aux["visual_embed_rms"] = probe[0]
        vision_aux["text_embed_rms"] = probe[1]
        vision_aux["visual_embed_max_abs"] = probe[2]
        # Pre-final-norm residual RMS split by position modality: the
        # depth-integrated scale of visual vs text positions after all
        # cycles (post-norm both equal the norm's scale by construction).
        residual = residual_probe_intermediates(model)
        vision_aux["residual_visual_rms"] = residual[0]
        vision_aux["residual_text_rms"] = residual[1]
    return loss, {"max_logits": logits_max, "tokens": token_count, **vision_aux}


def _subtree_l2norm(gradients, needle: str):
    """L2 norm over gradient leaves whose state path contains ``needle``."""
    total = jnp.zeros((), jnp.float32)
    for path, variable in nnx.to_flat_state(gradients):
        if needle in path:
            value = getattr(variable, "value", variable)
            total = total + jnp.sum(jnp.square(jnp.asarray(value).astype(jnp.float32)))
    return jnp.sqrt(total)


def _vision_metrics(host_metrics) -> dict[str, float] | None:
    """Derives the per-step vision group from host metrics, or None.

    The train step emits split SUMS (exact under gradient accumulation:
    sums-of-means divide out); the ratios are formed here on the host."""
    if "vision_loss_sum" not in host_metrics:
        return None
    vision_tokens = float(host_metrics["vision_token_count"])
    text_tokens = float(host_metrics["text_token_count"])
    derived = {
        "vision_loss": float(host_metrics["vision_loss_sum"]) / max(vision_tokens, 1.0),
        "text_loss": float(host_metrics["text_loss_sum"]) / max(text_tokens, 1.0),
        "vision_loss_tokens": vision_tokens,
        "text_loss_tokens": text_tokens,
    }
    for key in (
        "visual_embed_rms",
        "text_embed_rms",
        "visual_embed_max_abs",
        "residual_visual_rms",
        "residual_text_rms",
    ):
        if key in host_metrics:
            derived[key] = float(host_metrics[key])
    if derived.get("text_embed_rms"):
        derived["embed_rms_ratio"] = (
            derived["visual_embed_rms"] / derived["text_embed_rms"]
        )
    if "vit_grad_norm" in host_metrics:
        vit = float(host_metrics["vit_grad_norm"])
        total = float(host_metrics["grad_norm"])
        derived["vit_grad_norm"] = vit
        derived["lm_grad_norm"] = math.sqrt(max(total * total - vit * vit, 0.0))
    return derived


def _make_train_step(config: ResolvedConfig, loss_fn=None):
    """``loss_fn`` swaps the differentiated loss (same signature and
    auxiliary contract as ``_loss``: ``max_logits`` + ``tokens``, any other
    scalar auxiliary is averaged into the returned metrics). The SFT stage
    passes the GOLD objective this way; pretraining callers pass nothing
    and are unchanged."""
    accumulate = config.experiment.gradient_accumulation_steps
    use_clip = config.optimizer.name == "muonclip"
    model_loss = loss_fn if loss_fn is not None else _loss

    def differentiated_loss(model, batch):
        return model_loss(model, batch, record_max_logits=use_clip)

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
        extra_sums = {}
        extra_max = {}
        # Per-head attention logit maxima are reduced over the batch inside the
        # mixer, so the accumulator carries a single [cycles, 1, heads] row and
        # takes the max across accumulation microbatches.
        max_logits = jnp.full(
            (
                config.model.num_cycles,
                1,
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
            for key, value in auxiliary.items():
                if key in ("tokens", "max_logits"):
                    continue
                if key == "max_logits_by_modality":
                    extra_max[key] = (
                        value if key not in extra_max else jnp.maximum(extra_max[key], value)
                    )
                    continue
                extra_sums[key] = extra_sums.get(key, 0.0) + value
        gradients = jax.tree.map(lambda value: value / accumulate, accumulated_grads)
        vit_grad_norm = (
            _subtree_l2norm(gradients, "vision_tower")
            if config.model.vision.enabled
            else None
        )
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
        if vit_grad_norm is not None:
            metrics["vit_grad_norm"] = vit_grad_norm
        metrics.update(
            {key: value / accumulate for key, value in extra_sums.items()}
        )
        if clip_metrics is not None:
            metrics.update(
                {
                    "muonclip_max_logit": clip_metrics.max_logit,
                    "muonclip_min_scale": clip_metrics.min_scale,
                    "muonclip_clipped_heads": clip_metrics.clipped_heads,
                }
            )
        if "max_logits_by_modality" in extra_max:
            # [cycles, 2, heads] -> per-cycle max over heads, per modality.
            split = jnp.max(extra_max["max_logits_by_modality"], axis=-1)
            metrics["max_logit_visual"] = split[:, 0]
            metrics["max_logit_text"] = split[:, 1]
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
                images=batch["images"] if "images" in batch else None,
                record_max_logits=True,
            )
            weights = batch["loss_mask"].astype(jnp.float32)
            if current_model.config.model.loss.implementation == "chunked":
                loss, _ = chunked_linear_cross_entropy(
                    hidden,
                    batch["labels"],
                    weights,
                    current_model.output_projection_kernel(hidden.dtype),
                    block_tokens=current_model.config.model.loss.block_tokens,
                )
            elif current_model.config.model.loss.implementation == "tokamax_fused":
                loss, _ = data_parallel_linear_cross_entropy(
                    hidden.reshape((-1, hidden.shape[-1])),
                    batch["labels"].reshape((-1,)),
                    weights.reshape((-1,)),
                    current_model.output_projection_kernel(hidden.dtype),
                    mesh=current_model.mesh,
                    implementation="mosaic_tpu",
                )
            else:
                logits = current_model.project_logits(hidden)
                targets = jax.nn.one_hot(
                    batch["labels"],
                    logits.shape[-1],
                    dtype=jnp.float32,
                )
                cross_entropy, _ = max_utils.cross_entropy_with_logits(
                    logits,
                    targets,
                    z_loss=0.0,
                )
                loss = jnp.sum(cross_entropy * weights) / jnp.maximum(
                    jnp.sum(weights),
                    1.0,
                )
            # A single position per sequence is enough to catch output-head
            # excursions without recreating the full [batch,sequence,vocab]
            # tensor that the selected fused loss intentionally removes.
            sampled_logits = current_model.project_logits(hidden[:, -1:, :])
            auxiliary = {
                "hidden_rms": jnp.sqrt(jnp.mean(jnp.square(hidden.astype(jnp.float32)))),
                "hidden_max_abs": jnp.max(jnp.abs(hidden.astype(jnp.float32))),
                "sampled_logits_max_abs": jnp.max(jnp.abs(sampled_logits)),
            }
            return loss, auxiliary

        (loss, auxiliary), gradients = nnx.value_and_grad(
            diagnostic_loss,
            has_aux=True,
        )(model)
        parameters = nnx.state(model, nnx.Param)
        result = {
            "loss": loss,
            "grad_norm": max_utils.l2norm_pytree(gradients),
            "grad_max_abs": _tree_max_abs(gradients),
            "param_norm": max_utils.l2norm_pytree(parameters),
            "param_max_abs": _tree_max_abs(parameters),
            "hidden_rms": auxiliary["hidden_rms"],
            "hidden_max_abs": auxiliary["hidden_max_abs"],
            "sampled_logits_max_abs": auxiliary["sampled_logits_max_abs"],
            "attention_max_logits": attention_logit_intermediates(model),
        }
        if model.vision_tower is not None:
            result["attention_max_logits_by_modality"] = (
                attention_modality_logit_intermediates(model)
            )
        return result

    return diagnostics_step


def _host_diagnostics(metrics) -> dict[str, float]:
    host = jax.device_get(metrics)
    attention = host.pop("attention_max_logits")
    by_modality = host.pop("attention_max_logits_by_modality", None)
    result = {key: float(value) for key, value in host.items()}
    for cycle in range(attention.shape[0]):
        for head in range(attention.shape[-1]):
            result[f"attention/cycle_{cycle}/head_{head}_max_logit"] = float(
                attention[cycle, ..., head].max()
            )
    if by_modality is not None:
        # [cycles, 2, heads] -> per-cycle max over heads; absent -> omitted.
        for cycle in range(by_modality.shape[0]):
            for kind_index, kind in enumerate(("visual", "text")):
                value = float(by_modality[cycle, kind_index].max())
                if value > ABSENT_LOGIT / 2:
                    result[f"attention/cycle_{cycle}/{kind}_max_logit"] = value
    result["finite"] = float(all(math.isfinite(value) for value in result.values()))
    return result


class _PrefetchedIterator:
    """Background host-batch queue in front of a data iterator.

    Masks the episodic >1 s host stalls (stream shard boundaries, flushes)
    by keeping up to ``depth`` further host batches ready while the device
    computes. Only the host fetch is threaded; global-array assembly stays
    on the main thread. Batch order is preserved, so training losses are
    bitwise identical to the unqueued loop. Attribute access (pipeline
    stats, metadata, checkpoint state) forwards to the wrapped iterator;
    ``queue_depth`` reports this queue's readiness instead.
    """

    _SENTINEL = object()

    def __init__(self, iterator, depth: int):
        self._iterator = iterator
        self._queue = queue.Queue(maxsize=depth)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._fill, name="host-batch-prefetch", daemon=True
        )
        self._thread.start()

    def _fill(self) -> None:
        try:
            while True:
                self._queue.put(next(self._iterator))
        except BaseException as error:  # noqa: BLE001 - re-raised on the main thread
            self._error = error
            self._queue.put(self._SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._SENTINEL:
            raise self._error
        return item

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def __getattr__(self, name):
        return getattr(self._iterator, name)


def _device_batch(batch, mesh):
    """Host batch -> global data-sharded device batch, asynchronously.

    Numpy in, one batched device_put per local device: the call returns
    after enqueueing (~1 ms) and the DMA overlaps the step in flight. The
    previous form (jnp.asarray first, then host_local_array_to_global_array)
    staged the whole process batch on local device 0 and re-sliced it ON
    DEVICE behind the running step; measured 2026-08-17 on the v4-64, that
    blocked the host for the full remaining step (~1.5 s), so the
    "prefetched" batch never overlapped compute - 60% of the 30B
    continuation's steps paid it (h2d ~1.7 s vs 14 ms idle). Same arrays,
    bitwise-identical batches; single- and multi-host alike."""
    sharding = NamedSharding(mesh, PartitionSpec("data", None))
    return {
        key: jax.make_array_from_process_local_data(sharding, np.asarray(value))
        for key, value in batch.items()
    }


def _absent_to_nan(values) -> list[float]:
    return [float("nan") if value <= ABSENT_LOGIT / 2 else float(value) for value in values]


def _data_pipeline_stats(iterator) -> dict[str, float]:
    """Host-side pipeline health counters; costs a few attribute reads."""
    stats: dict[str, float] = {}
    queue_depth = getattr(iterator, "queue_depth", None)
    if queue_depth is not None:
        stats["prefetch_queue_depth"] = float(queue_depth)
    for key, value in dict(getattr(iterator, "stats", {}) or {}).items():
        stats[key] = float(value)
    seen = stats.get("documents_seen")
    selected = stats.get("documents_selected")
    if seen:
        stats["document_selection_rate"] = (selected or 0.0) / seen
    return stats


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


def _create_mixed_vision_iterator(config: ResolvedConfig, process_batch: int):
    """Builds the packed mixed vision+text stream for the main loop.

    Selected when ``model.vision.enabled`` and ``model.vision.dataset_name``
    are both set: vision rows stream from the vision corpus, text rows from
    ``data``'s corpus at ``vision.p_text``, packed with per-row segments.
    File-level sharding splits both streams disjointly across
    ``process_count * producer_threads`` producers."""
    from yxtpu_pretrain.runtime.data import load_fast_tokenizer
    from yxtpu_pretrain.runtime.vision_data import (
        MixedVisionTextIterator,
        PooledMixedIterator,
        ProcessPooledMixedIterator,
        ProducerSpec,
        VisionBatchSpec,
    )

    vision = config.model.vision
    spec = VisionBatchSpec(
        sequence_length=config.data.sequence_length,
        visual_tokens=vision.visual_tokens_per_image,
        image_size=vision.image_size,
        placeholder_id=vision.placeholder_token_id,
        pad_id=vision.pad_token_id,
        eos_id=vision.eos_token_id,
        max_images=vision.max_images_per_sequence,
        patch_size=vision.patch_size,
        host_patchify=vision.host_patchify,
    )
    threads = max(1, vision.producer_threads)
    if vision.text_datasets:
        text_sources = [
            {
                "name": source.name,
                "dataset": source.dataset,
                "subset": source.subset,
                "weight": source.weight,
                "field": source.field,
                "format": source.format,
                "row_tokens": source.row_tokens or vision.text_row_tokens,
            }
            for source in vision.text_datasets
        ]
    elif vision.p_text > 0:
        text_sources = [
            {
                "name": "text",
                "dataset": config.data.dataset_name,
                "field": config.data.text_field,
                "weight": 1.0,
                "format": "plain",
                "row_tokens": vision.text_row_tokens,
            }
        ]
    else:
        text_sources = []

    if vision.producer_processes > 0:
        workers = vision.producer_processes
        producer_spec = ProducerSpec(
            tokenizer_name=config.data.tokenizer,
            padded_vocab_size=config.model.vocab_size,
            spec=spec,
            batch_size=process_batch,
            vision_dataset=vision.dataset_name,
            text_sources=text_sources,
            p_text=vision.p_text,
            text_row_tokens=vision.text_row_tokens,
            min_visual_dependency=vision.min_visual_dependency,
            shuffle_seed=config.data.shuffle_seed,
            max_images_per_row=vision.max_images_per_row,
            row_buffer=vision.row_buffer,
        )
        iterator = ProcessPooledMixedIterator(
            producer_spec,
            workers=workers,
            shard_base=jax.process_index() * workers,
            shard_count=jax.process_count() * workers,
        )
        producers = {"producer_processes": workers}
    else:
        tokenizer = load_fast_tokenizer(
            config.data.tokenizer, padded_vocab_size=config.model.vocab_size
        )

        def make_source(thread_index: int) -> MixedVisionTextIterator:
            return MixedVisionTextIterator(
                tokenizer=tokenizer,
                spec=spec,
                batch_size=process_batch,
                vision_dataset=vision.dataset_name,
                text_sources=text_sources,
                p_text=vision.p_text,
                text_row_tokens=vision.text_row_tokens,
                min_visual_dependency=vision.min_visual_dependency,
                shuffle_seed=config.data.shuffle_seed,
                shard_index=jax.process_index() * threads + thread_index,
                shard_count=jax.process_count() * threads,
                max_images_per_row=vision.max_images_per_row,
                row_buffer=vision.row_buffer,
            )

        iterator = PooledMixedIterator(
            make_source, threads=threads, batch_size=process_batch
        )
        producers = {"producer_threads": threads}
    iterator.metadata = {
        "pipeline": "mixed_vision_text",
        "vision_dataset": vision.dataset_name,
        "text_sources": [
            {key: value for key, value in source.items()}
            for source in text_sources
        ],
        "p_text": vision.p_text,
        "text_row_tokens": vision.text_row_tokens,
        "max_images_per_sequence": vision.max_images_per_sequence,
        "max_images_per_row": vision.max_images_per_row,
        "min_visual_dependency": vision.min_visual_dependency,
        "row_buffer": vision.row_buffer,
        "host_patchify": vision.host_patchify,
        **producers,
    }
    return iterator


def _learning_rate(config: ResolvedConfig, step: int) -> float:
    """Host-side mirror of the Optax schedule, avoiding a TPU dispatch for logging."""
    optimizer = config.optimizer
    count = max(step - 1, 0)
    if count < optimizer.warmup_steps:
        return optimizer.learning_rate * count / max(optimizer.warmup_steps, 1)
    if optimizer.decay_steps is not None:
        decay_start = optimizer.schedule_steps - optimizer.decay_steps
        if count < decay_start:
            return optimizer.learning_rate
        progress = min((count - decay_start) / optimizer.decay_steps, 1.0)
    else:
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
    # The streaming source shards one identically-shuffled stream disjointly by
    # process, so it must keep the shared seed; the offline and synthetic
    # sources instead decorrelate processes through a per-process seed offset.
    process_data = (
        config.data
        if config.data.streaming
        else config.data.model_copy(
            update={
                "shuffle_seed": config.data.shuffle_seed
                + 1_000_003 * jax.process_index()
            }
        )
    )
    if config.model.vision.enabled and config.model.vision.dataset_name:
        data_iterator = _create_mixed_vision_iterator(config, train_process_batch)
    else:
        data_iterator = create_data_iterator(
            process_data,
            global_batch_size=train_process_batch,
            vocab_size=config.model.vocab_size,
            process_index=jax.process_index(),
            process_count=jax.process_count(),
        )
    if config.experiment.prefetch_batches > 1:
        data_iterator = _PrefetchedIterator(
            data_iterator, config.experiment.prefetch_batches - 1
        )
    # One held-out iterator per eval packing (the first is the historical
    # eval/train_holdout_loss; further packings log with a suffix). Under
    # eval_fixed_batches each is materialized once at the first evaluation.
    eval_iterators = (
        {
            packing: create_data_iterator(
                process_data.model_copy(update={"split": config.data.eval_split}),
                global_batch_size=eval_process_batch,
                vocab_size=config.model.vocab_size,
                validation=config.data.streaming,
                process_index=jax.process_index(),
                process_count=jax.process_count(),
                packing=packing,
                row_tokens=config.data.eval_row_tokens,
            )
            for packing in config.data.eval_packings
        }
        if config.data.eval_interval
        else {}
    )
    eval_iterator = next(iter(eval_iterators.values()), None)
    with logical_mesh_context(mesh, logical_axis_rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(config.experiment.seed))
        transform, routes = build_optimizer(model, config.optimizer)
        optimizer = nnx.Optimizer(model, transform, wrt=nnx.Param)
        state = TrainStateNNX(model, optimizer)

    is_primary = jax.process_index() == 0
    run_name = _run_name(config)
    run_dir = Path(config.experiment.run_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_metadata = dict(getattr(data_iterator, "metadata", {}))
    if is_primary:
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
        (run_dir / "data_metadata.json").write_text(
            json.dumps(data_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    metrics_writer = MetricsWriter(run_dir) if is_primary else NullMetricsWriter()
    host_metrics_writer = HostMetricsWriter(run_dir, jax.process_index())

    def emit(payload) -> None:
        if is_primary:
            print(json.dumps(payload, sort_keys=True), flush=True)
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
        if start_step == 0 and config.experiment.init_from_run:
            # Warm-start: weights from another run's latest checkpoint, then
            # a FRESH optimizer so momentum, second moments, and the
            # schedule's step count all restart at zero - the
            # continuation-after-anneal pattern. The foreign restore fills
            # the whole TrainStateNNX (the model trees must match); the
            # optimizer rebuild below discards its optimizer half.
            init_config = config.model_copy(deep=True)
            init_config.experiment.checkpoint.enabled = True
            init_config.experiment.acknowledge_no_checkpoint = False
            if config.experiment.init_from_destination:
                init_config.experiment.checkpoint.destination = (
                    config.experiment.init_from_destination
                )
            init_loader = CheckpointIO(
                init_config, run_name=config.experiment.init_from_run
            )
            init_step = init_loader.restore(state, data_iterator)
            init_loader.close()
            if init_step == 0:
                raise RuntimeError(
                    f"init_from_run {config.experiment.init_from_run!r} has no checkpoint"
                )
            state.optimizer = nnx.Optimizer(model, transform, wrt=nnx.Param)
            # Rebind the construction-time local so the ORIGINAL optimizer
            # module - now holding the restored foreign optimizer state -
            # drops its last reference and frees. Leaving it alive kept
            # ~4.2 GB of dead fp32 state resident and the first train step
            # failed allocation (RESOURCE_EXHAUSTED at 1B on v4).
            optimizer = state.optimizer
            del optimizer
            gc.collect()
            if jax.process_index() == 0:
                print(
                    f"warm-start: weights from {config.experiment.init_from_run} "
                    f"step {init_step}; fresh optimizer, schedule from step 0",
                    flush=True,
                )
        train_step = _make_train_step(config)
        eval_step = _make_eval_step()
        diagnostics_step = _make_diagnostics_step()
    # The batch conversion stays outside logical_mesh_context (it does not
    # depend on the logical rules; the step below is lowered for the global
    # shape the loop feeds it).
    first_host_batch = next(data_iterator)
    first_batch = _device_batch(first_host_batch, mesh)
    # The diagnostics step is one un-accumulated forward+backward: it takes
    # a MICROBATCH (the eval shape), so with gradient accumulation only the
    # first microbatch of the cached update batch is used - the whole update
    # batch would be accum x the compiled memory at once.
    diagnostic_host_batch = (
        {key: value[:eval_process_batch] for key, value in first_host_batch.items()}
        if config.experiment.diagnostics.batch == "train_fixed"
        else None
    )
    with logical_mesh_context(mesh, logical_axis_rules):
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
    emit({"compiled_memory": compiled_memory})
    metrics_writer.write({"compiled_memory": compiled_memory})
    throughputs = []
    losses = []
    tokens_seen = 0
    completed_steps = start_step
    trace_active = False
    cached_eval_batches: dict[str, list] = {}
    previous_step_end: float | None = None
    host_stats_interval = config.experiment.host_stats_interval
    try:
        # Prefetch the next batch between dispatching a step and blocking on
        # it, so the host fetch and host-to-device transfer (~70 ms/step of
        # otherwise-exposed wall time) overlap the device computation still
        # in flight. Skipped on iterations that save a checkpoint (the saved
        # iterator state must not run one batch ahead of the last trained
        # step) and on the projected final token-budget step, so persisted
        # and consumed positions always agree.
        checkpoint_interval = config.experiment.checkpoint.save_interval
        tokens_per_step = (
            config.data.per_device_batch_size
            * config.experiment.gradient_accumulation_steps
            * jax.device_count()
            * config.data.sequence_length
        )
        batch = first_batch
        data_wait_seconds = 0.0
        transfer_seconds = 0.0
        for step in range(start_step + 1, config.experiment.steps + 1):
            if profile and step == min(config.experiment.profile_steps):
                jax.profiler.start_trace(str(run_dir / "profile"))
                trace_active = True
            if batch is None:
                data_started = time.perf_counter()
                host_batch = next(data_iterator)
                data_wait_seconds = time.perf_counter() - data_started
                batch = _device_batch(host_batch, mesh)
                transfer_seconds = (
                    time.perf_counter() - data_started - data_wait_seconds
                )
            started = time.perf_counter()
            with logical_mesh_context(mesh, logical_axis_rules):
                metrics = compiled_train_step(state, batch)
            batch = None
            next_data_wait_seconds = 0.0
            next_transfer_seconds = 0.0
            will_checkpoint = bool(
                checkpoint_io.enabled
                and checkpoint_interval
                and step % checkpoint_interval == 0
            )
            ends_token_budget = (
                config.experiment.token_budget is not None
                and tokens_seen + tokens_per_step >= config.experiment.token_budget
            )
            if (
                step < config.experiment.steps
                and not will_checkpoint
                and not ends_token_budget
            ):
                data_started = time.perf_counter()
                host_batch = next(data_iterator)
                next_data_wait_seconds = time.perf_counter() - data_started
                batch = _device_batch(host_batch, mesh)
                next_transfer_seconds = (
                    time.perf_counter() - data_started - next_data_wait_seconds
                )
            jax.block_until_ready(metrics)
            step_end = time.perf_counter()
            elapsed = step_end - started
            wall_elapsed = (
                step_end - previous_step_end if previous_step_end is not None else None
            )
            previous_step_end = step_end
            host_metrics = jax.device_get(metrics)
            tokens = float(host_metrics["tokens"])
            tokens_seen += int(tokens)
            throughput = tokens / elapsed
            loss = float(host_metrics["loss"])
            grad_norm = float(host_metrics["grad_norm"])
            if not math.isfinite(loss) or not math.isfinite(grad_norm):
                failure = {
                    "step": step,
                    "tokens_seen_before_failed_step": tokens_seen - int(tokens),
                    "failure": "non_finite_train_metrics",
                    "loss": loss,
                    "grad_norm": grad_norm,
                }
                metrics_writer.write(failure)
                emit(failure)
                raise FloatingPointError(
                    f"non-finite training metrics at step {step}: "
                    f"loss={loss}, grad_norm={grad_norm}"
                )
            record = {
                "step": step,
                "loss": loss,
                "tokens": int(tokens),
                "step_ms": elapsed * 1_000,
                "tokens_per_second": throughput,
                "data_wait_ms": data_wait_seconds * 1_000,
                "host_to_device_ms": transfer_seconds * 1_000,
                "wall_tokens_per_second": (
                    tokens / wall_elapsed if wall_elapsed else throughput
                ),
                "grad_norm": grad_norm,
                "learning_rate": _learning_rate(config, step),
                "tokens_seen": tokens_seen,
            }
            # The prefetched batch's costs belong to the step that consumes
            # it, i.e. the next record.
            data_wait_seconds = next_data_wait_seconds
            transfer_seconds = next_transfer_seconds
            if "muonclip_max_logit" in host_metrics:
                record["muonclip"] = {
                    "max_logit": host_metrics["muonclip_max_logit"].tolist(),
                    "min_scale": host_metrics["muonclip_min_scale"].tolist(),
                    "clipped_heads": host_metrics["muonclip_clipped_heads"].tolist(),
                }
                if "max_logit_visual" in host_metrics:
                    record["muonclip"]["max_logit_visual"] = _absent_to_nan(
                        host_metrics["max_logit_visual"]
                    )
                    record["muonclip"]["max_logit_text"] = _absent_to_nan(
                        host_metrics["max_logit_text"]
                    )
            vision_metrics = _vision_metrics(host_metrics)
            if vision_metrics is not None:
                record["vision"] = vision_metrics
            metrics_writer.write(record)
            host_metrics_writer.write(
                {
                    "step": step,
                    "step_ms": record["step_ms"],
                    "data_wait_ms": record["data_wait_ms"],
                    "host_to_device_ms": record["host_to_device_ms"],
                    "prefetch_queue_depth": float(getattr(data_iterator, "queue_depth", -1)),
                }
            )
            emit(record)
            if step % config.experiment.log_interval == 0:
                log_groups = {
                    "train": {
                        "loss": loss,
                        "perplexity": math.exp(min(loss, 80.0)),
                    },
                    "performance": {
                        "tokens_per_second": throughput,
                        "wall_tokens_per_second": record["wall_tokens_per_second"],
                        "step_ms": elapsed * 1_000,
                        "data_wait_ms": record["data_wait_ms"],
                        "host_to_device_ms": record["host_to_device_ms"],
                    },
                    "data": _data_pipeline_stats(data_iterator),
                    "optimizer": {
                        "grad_norm": grad_norm,
                        "learning_rate": record["learning_rate"],
                    },
                    "stability": {
                        "loss_finite": float(math.isfinite(loss)),
                        "grad_norm_finite": float(math.isfinite(grad_norm)),
                    },
                }
                if vision_metrics is not None:
                    log_groups["vision"] = vision_metrics
                if "muonclip" in record:
                    attention = {}
                    for cycle, value in enumerate(record["muonclip"]["max_logit"]):
                        attention[f"cycle_{cycle}_max_logit"] = value
                    for kind in ("visual", "text"):
                        for cycle, value in enumerate(
                            record["muonclip"].get(f"max_logit_{kind}", ())
                        ):
                            if math.isfinite(value):
                                attention[f"cycle_{cycle}_max_logit_{kind}"] = value
                    log_groups["attention"] = attention
                if host_stats_interval and step % host_stats_interval == 0:
                    # Fleet view of the host path: any host's stall stalls
                    # every chip, so log the worst host, not the primary's.
                    # (host_to_device_ms is the asynchronous enqueue since
                    # the numpy device-batch path, ~1-3 ms everywhere; it is
                    # not gathered.)
                    local_stats = np.asarray(
                        [
                            float(log_groups["data"].get("prefetch_queue_depth", -1.0)),
                            record["data_wait_ms"],
                            record["step_ms"],
                        ],
                        dtype=np.float64,
                    )
                    fleet = np.asarray(multihost_utils.process_allgather(local_stats))
                    if fleet.ndim == 1:
                        fleet = fleet[None]
                    log_groups["hosts"] = {
                        "queue_depth_min": float(fleet[:, 0].min()),
                        "queue_depth_min_host": int(fleet[:, 0].argmin()),
                        "data_wait_ms_max": float(fleet[:, 1].max()),
                        "data_wait_ms_max_host": int(fleet[:, 1].argmax()),
                        "step_ms_max": float(fleet[:, 2].max()),
                        "hosts_waiting": int((fleet[:, 1] > 5.0).sum()),
                    }
                tracker.log(
                    log_groups,
                    step=step,
                    tokens_seen=tokens_seen,
                )
            losses.append(loss)
            completed_steps = step
            if step > start_step + 5:
                throughputs.append(throughput)

            diagnostic_batch = None
            if eval_iterator is not None and step % config.data.eval_interval == 0:
                eval_group: dict[str, float] = {}
                evaluation_record = {
                    "step": step,
                    "evaluation_fixed_batches": bool(config.data.eval_fixed_batches),
                }
                for packing_index, (packing, packing_iterator) in enumerate(
                    eval_iterators.items()
                ):
                    if config.data.eval_fixed_batches:
                        if packing not in cached_eval_batches:
                            cached_eval_batches[packing] = [
                                next(packing_iterator)
                                for _ in range(config.data.eval_steps)
                            ]
                        eval_host_batches = cached_eval_batches[packing]
                    else:
                        eval_host_batches = [
                            next(packing_iterator) for _ in range(config.data.eval_steps)
                        ]
                    eval_loss_sum = 0.0
                    eval_token_sum = 0.0
                    for eval_host_batch in eval_host_batches:
                        eval_batch = _device_batch(eval_host_batch, mesh)
                        if packing_index == 0:
                            diagnostic_batch = eval_batch
                        with logical_mesh_context(mesh, logical_axis_rules):
                            eval_metrics = eval_step(
                                state.model,
                                eval_batch,
                            )
                        eval_host = jax.device_get(eval_metrics)
                        eval_tokens = float(eval_host["tokens"])
                        eval_loss_sum += float(eval_host["loss"]) * eval_tokens
                        eval_token_sum += eval_tokens
                    evaluation_loss = eval_loss_sum / max(eval_token_sum, 1.0)
                    suffix = "" if packing_index == 0 else f"_{packing}"
                    evaluation_record[f"evaluation_loss{suffix}"] = evaluation_loss
                    evaluation_record[f"evaluation_tokens{suffix}"] = int(eval_token_sum)
                    evaluation_record[f"evaluation_packing{suffix}"] = packing
                    eval_group[f"train_holdout_loss{suffix}"] = evaluation_loss
                    eval_group[f"train_holdout_perplexity{suffix}"] = math.exp(
                        min(evaluation_loss, 80.0)
                    )
                    eval_group[f"tokens{suffix}"] = int(eval_token_sum)
                metrics_writer.write(evaluation_record)
                emit(evaluation_record)
                tracker.log(
                    {
                        "eval": eval_group,
                        "performance": {
                            "device_peak_bytes_in_use": _memory_summary()[
                                "peak_bytes_in_use"
                            ],
                        },
                    },
                    step=step,
                    tokens_seen=tokens_seen,
                )

            diagnostics = config.experiment.diagnostics
            if diagnostics.enabled and step % diagnostics.interval == 0:
                if diagnostic_host_batch is not None:
                    diagnostic_batch = _device_batch(diagnostic_host_batch, mesh)
                if diagnostic_batch is not None:
                    with logical_mesh_context(mesh, logical_axis_rules):
                        diagnostic_metrics = diagnostics_step(state.model, diagnostic_batch)
                        jax.block_until_ready(diagnostic_metrics)
                    host_diagnostics = _host_diagnostics(diagnostic_metrics)
                    # Which batch the pass ran on: the holdout eval batch or
                    # the cached first training batch (they are not comparable
                    # - see the 2026-08-17 cycle-0 max-logit finding).
                    host_diagnostics["batch"] = diagnostics.batch
                    diagnostics_record = {"step": step, "diagnostics": host_diagnostics}
                    metrics_writer.write(diagnostics_record)
                    emit(diagnostics_record)
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
                emit(harness_record)
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
        # Skip the final force-save when the last completed step coincided
        # with an interval save: orbax refuses to overwrite an existing step
        # (observed at the vision-1b-trial's step 48,000, a multiple of the
        # 4,000-step interval - the run crashed in its exit path after an
        # otherwise clean finish).
        final_interval = config.experiment.checkpoint.save_interval
        if checkpoint_io.enabled and (
            not final_interval or completed_steps % final_interval != 0
        ):
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
        closer = getattr(data_iterator, "close", None)
        if callable(closer):
            closer()

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
    host_metrics_writer.close()
    tracker.finish(summary=summary)
    if is_primary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
