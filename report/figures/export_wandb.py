# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export the pretraining campaign's W&B history to CSV.

Runs where wandb is authenticated (TPU worker 0). Discovers the actual
metric keys before scanning, prints createdAt per leg so main-vs-resume
is settled by data, downsamples to <= 20k rows per leg, and writes one
CSV per run id into --output-dir.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

RUNS = ("60sndjih", "oiggd5fp")
PROJECT = "davidfranco2300-other/yxtpu-pretrain"

WANTED_SUBSTRINGS = (
    "loss", "learning_rate", "step_ms", "tokens", "grad_norm",
    "max_logit", "tokens_per_second", "data_wait",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/wandb_export")
    parser.add_argument("--max-rows", type=int, default=20000)
    arguments = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=120)
    out = Path(arguments.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for run_id in RUNS:
        run = api.run(f"{PROJECT}/{run_id}")
        print(f"== {run_id}: name={run.name} state={run.state} "
              f"created={run.created_at} lastHistoryStep={run.lastHistoryStep}",
              flush=True)
        summary_keys = sorted(k for k in run.summary.keys()
                              if not k.startswith("_"))
        print(f"   summary keys: {summary_keys}", flush=True)
        keys = [k for k in summary_keys
                if any(s in k for s in WANTED_SUBSTRINGS)]
        print(f"   exporting: {keys}", flush=True)

        stride = max(1, (run.lastHistoryStep or 1) // arguments.max_rows)
        rows = []
        for row in run.scan_history(keys=["_step"] + keys, page_size=10000):
            if row.get("_step", 0) % stride == 0:
                rows.append(row)
        print(f"   {len(rows)} rows at stride {stride}", flush=True)

        path = out / f"{run_id}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["_step"] + keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"   written {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
