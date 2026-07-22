"""Orbax checkpoint management for NNX training and iterator state."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp
from flax import nnx

from yxtpu_pretrain.config import ResolvedConfig

# Sentinel iterator payload for streaming profiles that cannot serialize
# their position: the restore keeps weights and optimizer state and the
# stream restarts from its beginning.
_UNRESUMABLE_ITERATOR = {"resumable": 0}


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
        multiprocessing_options = None
        if jax.process_count() > 1 and not destination.startswith("gs://"):
            # Multi-host slice writing to per-host local disks (no shared
            # filesystem): every host acts as its own primary and persists
            # the fully replicated train state plus its own rank-sharded
            # iterator position locally, so a restore never reads across
            # hosts. Only sound under pure data parallelism, where each
            # host addresses a complete copy of every parameter.
            mesh = config.hardware.mesh
            if mesh.fsdp != 1 or mesh.tensor != 1 or mesh.sequence != 1:
                raise ValueError(
                    "local-disk checkpointing on a multi-host slice requires "
                    "pure data parallelism; use a gs:// destination for "
                    "sharded parameter meshes"
                )
            multiprocessing_options = ocp.options.MultiprocessingOptions(
                primary_host=None
            )
        manager_options = dict(
            create=True,
            enable_async_checkpointing=checkpoint.async_save,
            save_interval_steps=max(1, checkpoint.save_interval),
            max_to_keep=checkpoint.keep,
        )
        if multiprocessing_options is not None:
            # Passing None explicitly trips orbax's primary-host probing;
            # only override the default when the local multi-host mode is on.
            manager_options["multiprocessing_options"] = multiprocessing_options
        self.manager = ocp.CheckpointManager(
            destination,
            item_names=("state", "iterator", "metadata"),
            item_handlers={
                "state": ocp.StandardCheckpointHandler(),
                "iterator": ocp.StandardCheckpointHandler(),
                "metadata": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(**manager_options),
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
        iterator_state = restored.iterator
        if (
            isinstance(iterator_state, dict)
            and int(iterator_state.get("resumable", 1)) == 0
        ):
            print(
                "checkpoint restore: streaming iterator position was not "
                "saved; weights and optimizer resume, the stream restarts"
            )
        else:
            data_iterator.set_state(iterator_state)
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
        try:
            iterator_state = data_iterator.get_state()
        except RuntimeError:
            iterator_state = dict(_UNRESUMABLE_ITERATOR)
        return self.manager.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                iterator=ocp.args.StandardSave(iterator_state),
                metadata=ocp.args.JsonSave(self.metadata | {"step": step}),
            ),
            force=force,
        )

    def close(self) -> None:
        if self.manager is not None:
            self.manager.wait_until_finished()
            self.manager.close()
