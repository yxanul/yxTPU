"""Orbax checkpoint management for NNX training and iterator state."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp
from flax import nnx

from yxtpu_pretrain.config import ResolvedConfig


def checkpoint_path(base: str, run_name: str) -> str:
    return f"{base.rstrip('/')}/{run_name}"


def _git_sha(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def checkpoint_metadata(config: ResolvedConfig, *, tokenizer: str | None) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[3]
    repository_root = package_root.parent
    maxtext_root = repository_root / "maxtext"
    return {
        "repository_commit": _git_sha(repository_root),
        "maxtext_commit": _git_sha(maxtext_root),
        "maxtext_pin": (package_root / "MAXTEXT_PIN").read_text(encoding="utf-8").strip(),
        "resolved_config": config.as_dict(),
        "tokenizer": tokenizer,
        "kda_precision_policy": config.model.kda.precision,
        "format": "yxtpu-pretrain-nnx-v1",
    }


def _persistent_state(module: nnx.Module):
    state = nnx.state(module)
    intermediates, persistent = nnx.split_state(state, nnx.Intermediate, ...)
    del intermediates
    return persistent


class CheckpointIO:
    def __init__(self, config: ResolvedConfig, *, run_name: str):
        checkpoint = config.experiment.checkpoint
        self.enabled = checkpoint.enabled
        self.manager = None
        if not self.enabled:
            return
        destination = checkpoint_path(checkpoint.destination, run_name)
        self.manager = ocp.CheckpointManager(
            destination,
            item_names=("state", "iterator", "metadata"),
            item_handlers={
                "state": ocp.StandardCheckpointHandler(),
                "iterator": ocp.StandardCheckpointHandler(),
                "metadata": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(
                create=True,
                enable_async_checkpointing=checkpoint.async_save,
                save_interval_steps=max(1, checkpoint.save_interval),
                max_to_keep=checkpoint.keep,
            ),
            metadata={"format": "yxtpu-pretrain-nnx-v1"},
        )
        self.metadata = checkpoint_metadata(config, tokenizer=config.data.tokenizer)

    def latest_step(self) -> int | None:
        return None if self.manager is None else self.manager.latest_step()

    def restore(self, train_state: nnx.Module, data_iterator) -> int:
        if self.manager is None:
            return 0
        step = self.manager.latest_step()
        if step is None:
            return 0
        target = _persistent_state(train_state)
        restored = self.manager.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(target.to_pure_dict()),
                iterator=ocp.args.StandardRestore(),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        if restored.metadata.get("maxtext_pin") != self.metadata["maxtext_pin"]:
            raise ValueError("checkpoint MaxText pin does not match this package")
        nnx.replace_by_pure_dict(target, restored.state)
        nnx.update(train_state, target)
        data_iterator.set_state(restored.iterator)
        return int(step)

    def save(
        self,
        train_state: nnx.Module,
        data_iterator,
        step: int,
        *,
        force: bool = False,
    ) -> bool:
        if self.manager is None:
            return False
        state = _persistent_state(train_state).to_pure_dict()
        return self.manager.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                iterator=ocp.args.StandardSave(data_iterator.get_state()),
                metadata=ocp.args.JsonSave(self.metadata | {"step": step}),
            ),
            force=force,
        )

    def close(self) -> None:
        if self.manager is not None:
            self.manager.wait_until_finished()
            self.manager.close()
