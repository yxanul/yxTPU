import json

import jax.numpy as jnp
from flax import nnx
from maxtext.common.train_state_nnx import TrainStateNNX

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.model import HybridLanguageModel
from yxtpu_pretrain.optimizers import build_optimizer
from yxtpu_pretrain.runtime.checkpoints import CheckpointIO, checkpoint_path
from yxtpu_pretrain.runtime.data import create_data_iterator
from yxtpu_pretrain.runtime.mesh import create_mesh
from yxtpu_pretrain.train import _make_train_step


def _tiny_config(tmp_path, *, data="synthetic", dataset_path=None):
    overrides = [
        "model.emb_dim=128",
        "model.mlp_dim=256",
        "model.num_layers=4",
        "model.num_cycles=1",
        "model.kda.num_heads=1",
        "model.kda.precision=full_fp32",
        "model.attention.num_query_heads=1",
        "model.attention.num_kv_heads=1",
        "data.sequence_length=64",
        "data.per_device_batch_size=1",
        "model.vocab_size=256",
        "model.dtype=float32",
        "model.remat_policy=full",
        "experiment.benchmark=false",
        "experiment.checkpoint.enabled=true",
        f"experiment.checkpoint.destination={tmp_path}",
        "experiment.checkpoint.save_interval=1",
    ]
    if dataset_path is not None:
        overrides.append(f"data.dataset_path={dataset_path}")
    return load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data=data,
        hardware="v6e-8",
        experiment="selected",
        overrides=overrides,
    )


def test_synthetic_iterator_resume_reproduces_next_batch(tmp_path):
    config = _tiny_config(tmp_path)
    iterator = create_data_iterator(
        config.data, global_batch_size=2, vocab_size=config.model.vocab_size
    )
    next(iterator)
    state = iterator.get_state()
    expected = next(iterator)
    iterator.set_state(state)
    actual = next(iterator)
    for key in expected:
        assert jnp.array_equal(expected[key], actual[key])


def test_synthetic_reuse_example_batch_is_stable(tmp_path):
    config = _tiny_config(tmp_path)
    iterator = create_data_iterator(
        config.data, global_batch_size=2, vocab_size=config.model.vocab_size
    )
    first = next(iterator)
    second = next(iterator)
    assert iterator.get_state() == {"index": 2}
    for key in first:
        assert jnp.array_equal(first[key], second[key])


def test_offline_huggingface_and_grain_fixtures(tmp_path):
    fixture = tmp_path / "tokens.jsonl"
    fixture.write_text(
        "\n".join(
            json.dumps({"input_ids": list(range(1 + offset, 70 + offset))})
            for offset in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    for data_type in ("huggingface", "grain"):
        config = _tiny_config(tmp_path / data_type, data=data_type, dataset_path=fixture)
        iterator = create_data_iterator(
            config.data, global_batch_size=2, vocab_size=config.model.vocab_size
        )
        batch = next(iterator)
        assert batch["input_ids"].shape == (2, 64)
        assert jnp.all(batch["loss_mask"] > 0)
        saved = iterator.get_state()
        expected = next(iterator)
        iterator.set_state(saved)
        actual = next(iterator)
        for key in expected:
            assert jnp.array_equal(expected[key], actual[key])


def test_local_orbax_round_trip_preserves_nnx_and_iterator(tmp_path):
    config = _tiny_config(tmp_path)
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(21))
    transform, _ = build_optimizer(model, config.optimizer)
    train_state = TrainStateNNX(
        model,
        nnx.Optimizer(model, transform, wrt=nnx.Param),
    )
    iterator = create_data_iterator(
        config.data, global_batch_size=1, vocab_size=config.model.vocab_size
    )
    next(iterator)
    original = {
        path: variable.get_value().copy()
        for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }
    checkpoint = CheckpointIO(config, run_name="round-trip")
    assert checkpoint.save(train_state, iterator, 1, force=True)
    checkpoint.manager.wait_until_finished()

    first_parameter = next(iter(nnx.to_flat_state(nnx.state(model, nnx.Param))))[1]
    first_parameter.set_value(jnp.zeros_like(first_parameter.get_value()))
    iterator.set_state({"index": 99})
    assert checkpoint.restore(train_state, iterator) == 1
    checkpoint.close()
    assert iterator.get_state() == {"index": 1}
    for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        assert jnp.array_equal(variable.get_value(), original[path])


def test_gcs_checkpoint_path_is_constructed_without_provisioning():
    assert checkpoint_path("gs://existing-bucket/checkpoints/", "run-1") == (
        "gs://existing-bucket/checkpoints/run-1"
    )


def _train_state(config, seed=31):
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(seed))
    transform, _ = build_optimizer(model, config.optimizer)
    return TrainStateNNX(model, nnx.Optimizer(model, transform, wrt=nnx.Param))


def _jax_batch(batch):
    return {key: jnp.asarray(value) for key, value in batch.items()}


def test_interrupted_training_reproduces_next_uninterrupted_step(tmp_path):
    config = _tiny_config(tmp_path)
    step = _make_train_step(config)

    uninterrupted = _train_state(config)
    uninterrupted_data = create_data_iterator(
        config.data, global_batch_size=1, vocab_size=config.model.vocab_size
    )
    step(uninterrupted, _jax_batch(next(uninterrupted_data)))
    expected_metrics = step(uninterrupted, _jax_batch(next(uninterrupted_data)))

    interrupted = _train_state(config)
    interrupted_data = create_data_iterator(
        config.data, global_batch_size=1, vocab_size=config.model.vocab_size
    )
    step(interrupted, _jax_batch(next(interrupted_data)))
    checkpoint = CheckpointIO(config, run_name="resume-equivalence")
    checkpoint.save(interrupted, interrupted_data, 1, force=True)
    checkpoint.manager.wait_until_finished()

    resumed = _train_state(config, seed=999)
    resumed_data = create_data_iterator(
        config.data, global_batch_size=1, vocab_size=config.model.vocab_size
    )
    assert checkpoint.restore(resumed, resumed_data) == 1
    actual_metrics = step(resumed, _jax_batch(next(resumed_data)))
    checkpoint.close()

    assert jnp.array_equal(actual_metrics["loss"], expected_metrics["loss"])
    expected_params = nnx.to_flat_state(nnx.state(uninterrupted.model, nnx.Param))
    actual_params = nnx.to_flat_state(nnx.state(resumed.model, nnx.Param))
    for (expected_path, expected), (actual_path, actual) in zip(
        expected_params, actual_params, strict=True
    ):
        assert actual_path == expected_path
        assert jnp.array_equal(actual.get_value(), expected.get_value())
