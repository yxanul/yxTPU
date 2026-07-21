"""JSONL metrics and compact performance summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax

from yxtpu_pretrain.config import ResolvedConfig


class MetricsWriter:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.path = run_dir / "metrics.jsonl"
        self.handle = self.path.open("a", encoding="utf-8")
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self, summary: dict[str, Any]) -> None:
        self.handle.close()
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _flatten_metrics(value: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten_metrics(item, prefix=name))
        elif item is not None:
            flattened[name] = item
    return flattened


class WandbTracker:
    """Process-zero W&B logging kept outside timed/JIT training paths."""

    def __init__(
        self,
        config: ResolvedConfig,
        *,
        run_name: str,
        run_dir: Path,
        metadata: dict[str, Any],
    ):
        self.run = None
        if not config.experiment.wandb.enabled or jax.process_index() != 0:
            return
        import wandb

        wandb_config = config.experiment.wandb
        self.run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=run_name,
            group=wandb_config.group,
            tags=list(wandb_config.tags),
            mode=wandb_config.mode,
            dir=str(run_dir),
            config={"resolved": config.as_dict(), "runtime": metadata},
            resume="never",
        )
        self.run.define_metric("trainer/step")
        self.run.define_metric("trainer/tokens_seen")
        for namespace in ("train/*", "performance/*", "data/*", "optimizer/*", "stability/*"):
            self.run.define_metric(namespace, step_metric="trainer/step", step_sync=True)
        for namespace in ("eval/*", "diagnostics/*", "lm_eval/*"):
            self.run.define_metric(namespace, step_metric="trainer/tokens_seen", step_sync=True)

    @property
    def url(self) -> str | None:
        return getattr(self.run, "url", None) if self.run is not None else None

    def log(self, metrics: dict[str, Any], *, step: int, tokens_seen: int) -> None:
        if self.run is None:
            return
        record = {
            "trainer/step": step,
            "trainer/tokens_seen": tokens_seen,
            **_flatten_metrics(metrics),
        }
        self.run.log(record)

    def log_artifact(self, path: Path, *, name: str, artifact_type: str) -> None:
        if self.run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self, *, summary: dict[str, Any] | None = None, exit_code: int = 0) -> None:
        if self.run is None:
            return
        if summary:
            self.run.summary.update(summary)
        self.run.finish(exit_code=exit_code)
