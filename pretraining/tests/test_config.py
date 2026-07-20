from pathlib import Path

import pytest

from yxtpu_pretrain.config import CONFIG_ROOT, load_config


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


def test_all_required_profiles_exist():
    expected = {
        "models/kda_hybrid_273m.yml",
        "optimizers/adamw.yml",
        "optimizers/muon.yml",
        "optimizers/muonclip.yml",
        "data/synthetic.yml",
        "data/huggingface.yml",
        "data/grain.yml",
        "hardware/v6e-8.yml",
        "hardware/v6e-64.yml",
        "hardware/v5e-16.yml",
        "hardware/v5e-64.yml",
        "hardware/v4-32.yml",
        "experiments/selected.yml",
        "experiments/max_throughput.yml",
        "experiments/sequence_sweep.yml",
    }
    found = {str(path.relative_to(CONFIG_ROOT)) for path in CONFIG_ROOT.rglob("*.yml")}
    assert expected <= found


def test_pin_matches_imported_maxtext_commit():
    package_root = Path(__file__).resolve().parents[1]
    assert (package_root / "MAXTEXT_PIN").read_text().strip() == "dfd8d293"
