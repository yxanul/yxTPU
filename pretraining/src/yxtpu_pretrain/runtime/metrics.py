"""JSONL metrics and compact performance summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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

