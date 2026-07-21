from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.evaluation.lm_harness import (
    JaxHarnessLM,
    _json_safe,
    flatten_harness_metrics,
)
from yxtpu_pretrain.model import HybridLanguageModel, attention_logit_intermediates
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.runtime.metrics import _flatten_metrics
from yxtpu_pretrain.train import _host_diagnostics, _learning_rate, _make_diagnostics_step


def test_harness_uses_normalized_completion_metrics_and_arc_gap():
    metrics = flatten_harness_metrics(
        {
            "results": {
                "hellaswag": {"acc,none": 0.4, "acc_norm,none": 0.5},
                "arc_easy": {"acc_norm,none": 0.6},
                "arc_challenge": {"acc_norm,none": 0.35},
                "boolq": {"acc,none": 0.55},
                "lambada_openai": {"acc,none": 0.2, "perplexity,none": 50.0},
            }
        }
    )
    assert metrics["hellaswag/primary"] == 0.5
    assert metrics["arc_easy/primary"] == 0.6
    assert metrics["arc_challenge/primary"] == 0.35
    assert metrics["arc_easy_challenge_gap"] == 0.25
    assert metrics["boolq/primary"] == 0.55
    assert metrics["lambada_openai/primary"] == 0.2


def test_harness_artifact_serializes_task_callables_by_provenance():
    safe = _json_safe(
        {
            "config": {
                "process_docs": test_harness_scoring_mask_selects_only_continuation
            }
        }
    )
    assert safe == {
        "config": {
            "process_docs": {
                "callable": (
                    "test_evaluation_metrics."
                    "test_harness_scoring_mask_selects_only_continuation"
                )
            }
        }
    }


def test_harness_scoring_mask_selects_only_continuation():
    adapter = object.__new__(JaxHarnessLM)
    adapter.max_length = 8
    adapter.tokenizer = SimpleNamespace(eos_token_id=99)
    example = adapter._prepare_request([10, 11, 12], [20, 21])
    assert example["input_ids"].tolist() == [10, 11, 12, 20, 99, 99, 99, 99]
    assert example["labels"].tolist() == [11, 12, 20, 21, 99, 99, 99, 99]
    assert example["score_mask"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    assert example["segment_ids"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]


def test_requested_harness_tasks_exist_in_pinned_release():
    from lm_eval.tasks import TaskManager

    config = load_config(
        model="kda_hybrid_309m_gpt2",
        optimizer="adamw_10b",
        data="climbmix",
        hardware="v6e-8",
        experiment="climbmix_10b",
    )
    manager = TaskManager()
    matched = set(manager.match_tasks(list(config.experiment.harness_eval.tasks)))
    assert matched == set(config.experiment.harness_eval.tasks)


def test_wandb_metric_flattening_and_host_learning_rate():
    assert _flatten_metrics({"train": {"loss": 3.0}, "step": 2}) == {
        "train/loss": 3.0,
        "step": 2,
    }
    config = load_config(
        model="kda_hybrid_309m_gpt2",
        optimizer="adamw_10b",
        data="climbmix",
        hardware="v6e-8",
        experiment="climbmix_10b",
    )
    assert _learning_rate(config, 1) == 0.0
    assert _learning_rate(config, 101) == config.optimizer.learning_rate
    assert _learning_rate(config, config.optimizer.schedule_steps + 1) == (
        config.optimizer.learning_rate * config.optimizer.final_learning_rate_fraction
    )
    assert jax.process_count() >= 1


def test_separate_diagnostics_pass_reports_finite_norms_and_attention():
    config = load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=[
            "model.emb_dim=128",
            "model.mlp_dim=256",
            "model.num_layers=4",
            "model.num_cycles=1",
            "model.kda.num_heads=1",
            "model.kda.precision=full_fp32",
            "model.attention.num_query_heads=1",
            "model.attention.num_kv_heads=1",
            "model.vocab_size=256",
            "model.dtype=float32",
            "model.remat_policy=full",
            "data.sequence_length=64",
            "data.per_device_batch_size=1",
        ],
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(9))
    tokens = jnp.arange(64, dtype=jnp.int32)[None] % 255 + 1
    batch = {
        "input_ids": tokens,
        "labels": jnp.roll(tokens, -1, axis=1),
        "loss_mask": jnp.ones_like(tokens, dtype=jnp.float32),
        "segment_ids": jnp.ones_like(tokens, dtype=jnp.int32),
        "positions": jnp.arange(64, dtype=jnp.int32)[None],
    }
    metrics = _host_diagnostics(_make_diagnostics_step()(model, batch))
    assert metrics["finite"] == 1.0
    assert metrics["grad_norm"] > 0
    assert "attention/cycle_0/head_0_max_logit" in metrics


def test_recording_forward_keeps_attention_logit_intermediate_batch_independent():
    """A record_max_logits forward must not resize the persisted intermediate.

    Regression for the step-250 crash: with adamw the compiled train step never
    records, so it captures the max-logit intermediates at their initial
    [cycles, 1, heads] shape. A diagnostics forward that recorded a batch-sized
    [cycles, B, heads] value left the model state incompatible with the next
    donated train step. The reduction must keep the shape batch-independent.
    """
    config = load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=[
            "model.emb_dim=128",
            "model.mlp_dim=256",
            "model.num_layers=4",
            "model.num_cycles=1",
            "model.kda.num_heads=1",
            "model.kda.precision=full_fp32",
            "model.attention.num_query_heads=2",
            "model.attention.num_kv_heads=1",
            "model.vocab_size=256",
            "model.dtype=float32",
            "model.remat_policy=full",
            "data.sequence_length=64",
            "data.per_device_batch_size=1",
        ],
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(9))
    expected = (config.model.num_cycles, 1, config.model.attention.num_query_heads)
    assert attention_logit_intermediates(model).shape == expected

    batch_rows = 4
    tokens = (jnp.arange(batch_rows * 64, dtype=jnp.int32).reshape(batch_rows, 64) % 255) + 1
    batch = {
        "input_ids": tokens,
        "labels": jnp.roll(tokens, -1, axis=1),
        "loss_mask": jnp.ones_like(tokens, dtype=jnp.float32),
        "segment_ids": jnp.ones_like(tokens, dtype=jnp.int32),
        "positions": jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32), tokens.shape),
    }
    # A batch of four rows would previously leak a [cycles, 4, heads] shape.
    _make_diagnostics_step()(model, batch)
    assert attention_logit_intermediates(model).shape == expected
