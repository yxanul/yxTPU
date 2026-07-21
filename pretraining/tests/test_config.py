from pathlib import Path

import pytest

from yxtpu_pretrain.config import CONFIG_ROOT, load_config
from yxtpu_pretrain.train import _process_batch_sizes


def test_selected_profile_matches_certified_baseline():
    config = load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
    )
    assert config.model.num_layers == 16
    assert config.model.cycle == ("kda", "kda", "kda", "gqa")
    assert config.model.kda.precision == "guarded_fp32"
    assert config.data.sequence_length == 2048
    assert config.data.per_device_batch_size == 8
    assert config.hardware.device_count == 8
    assert config.experiment.checkpoint.enabled is False


def test_gradient_accumulation_expands_the_effective_batch():
    config = load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="max_throughput",
    )
    train_batch, eval_batch = _process_batch_sizes(config, local_device_count=8)
    assert config.data.per_device_batch_size == 16
    assert config.experiment.gradient_accumulation_steps == 8
    assert train_batch == 16 * 8 * 8
    assert eval_batch == 16 * 8


def test_cli_override_and_train_alias():
    config = load_config(
        model="kda_hybrid_273m",
        optimizer="muonclip",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
        overrides=["train.steps=7", "data.sequence_length=4096"],
    )
    assert config.experiment.steps == 7
    assert config.data.sequence_length == 4096
    assert config.optimizer.name == "muonclip"


def test_block_attnres_is_reserved_but_rejected():
    with pytest.raises(ValueError, match="reserved but disabled"):
        load_config(
            model="kda_hybrid_273m",
            optimizer="adamw",
            data="synthetic",
            hardware="v6e-8",
            experiment="selected",
            overrides=["model.residual_policy=block_attnres"],
        )


def test_fused_loss_rejects_vocabulary_parallel_meshes():
    with pytest.raises(ValueError, match="vocabulary parallelism"):
        load_config(
            model="kda_hybrid_273m",
            optimizer="adamw",
            data="synthetic",
            hardware="v6e-8",
            experiment="selected",
            overrides=[
                "model.loss.implementation=tokamax_fused",
                "hardware.mesh.data=4",
                "hardware.mesh.tensor=2",
            ],
        )


def test_all_required_profiles_exist():
    expected = {
        "models/kda_hybrid_273m.yml",
        "models/kda_hybrid_309m_gpt2.yml",
        "optimizers/adamw.yml",
        "optimizers/adamw_10b.yml",
        "optimizers/muon.yml",
        "optimizers/muonclip.yml",
        "data/synthetic.yml",
        "data/huggingface.yml",
        "data/grain.yml",
        "data/climbmix.yml",
        "hardware/v6e-8.yml",
        "hardware/v6e-64.yml",
        "hardware/v5e-16.yml",
        "hardware/v5e-64.yml",
        "hardware/v4-32.yml",
        "experiments/selected.yml",
        "experiments/max_throughput.yml",
        "experiments/sequence_sweep.yml",
        "experiments/climbmix_10b.yml",
    }
    found = {str(path.relative_to(CONFIG_ROOT)) for path in CONFIG_ROOT.rglob("*.yml")}
    assert expected <= found


def test_climbmix_profile_is_explicit_streaming_no_checkpoint_run():
    config = load_config(
        model="kda_hybrid_309m_gpt2",
        optimizer="adamw_10b",
        data="climbmix",
        hardware="v6e-8",
        experiment="climbmix_10b",
    )
    assert config.model.vocab_size == 50_432
    assert config.data.dataset_name == "karpathy/climbmix-400b-shuffle"
    assert config.data.streaming is True
    assert config.data.validation_fraction == 0.01
    assert config.data.prefetch_batches == 3
    assert config.experiment.token_budget == 10_000_000_000
    assert config.experiment.checkpoint.enabled is False
    assert config.experiment.acknowledge_no_checkpoint is True
    assert config.experiment.harness_eval.interval == 5 * config.data.eval_interval


def test_pin_matches_imported_maxtext_commit():
    package_root = Path(__file__).resolve().parents[1]
    assert (package_root / "MAXTEXT_PIN").read_text().strip() == "dfd8d293"
