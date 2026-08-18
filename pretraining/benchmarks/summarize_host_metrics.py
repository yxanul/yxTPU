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

"""Joins the per-host ``host_metrics.<process>.jsonl`` files of one run and
localizes step stalls.

Every host writes its own step_ms / data_wait_ms / host_to_device_ms /
prefetch queue depth each step (runtime/metrics.py HostMetricsWriter). On a
multi-host slice a stall on any host stalls every chip, so the primary's
record shows the stall but not its owner; here, for each step whose fleet
step time exceeds the floor by ``--stall-ms``, the host that was waiting
on its OWN data (largest data_wait, or an empty prefetch queue) is named.

Collect the files first, e.g. with scripts/fleet.sh:
  for w in 0 1 2 3 4 5 6 7; do gcloud compute tpus tpu-vm scp \\
    "$TPU:~/yxTPU/pretraining/runs/<run>/host_metrics.*.jsonl" ./hosts/ \\
    --zone=$ZONE --worker=$w; done
  python benchmarks/summarize_host_metrics.py ./hosts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def _load(directory: str) -> dict[int, dict[int, dict]]:
  hosts: dict[int, dict[int, dict]] = {}
  for path in sorted(glob.glob(os.path.join(directory, "host_metrics.*.jsonl"))):
    process = int(os.path.basename(path).split(".")[1])
    rows = {}
    with open(path, encoding="utf-8") as handle:
      for line in handle:
        record = json.loads(line)
        rows[int(record["step"])] = record
    hosts[process] = rows
  return hosts


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directory")
  parser.add_argument("--stall-ms", type=float, default=300.0,
                      help="excess over the fleet p10 step that counts as a stall")
  parser.add_argument("--skip-steps", type=int, default=5)
  args = parser.parse_args()
  hosts = _load(args.directory)
  if not hosts:
    print("no host_metrics.*.jsonl files found")
    return 1
  steps = sorted(set.intersection(*(set(rows) for rows in hosts.values())))
  steps = [s for s in steps if s > args.skip_steps]
  primary = hosts[min(hosts)]
  floor = statistics.quantiles([primary[s]["step_ms"] for s in steps], n=10)[0]
  print(f"hosts {sorted(hosts)}  steps {steps[0]}..{steps[-1]}  primary step_ms p10 {floor:.1f}")
  for process, rows in sorted(hosts.items()):
    waits = [rows[s]["data_wait_ms"] for s in steps]
    depths = [rows[s]["prefetch_queue_depth"] for s in steps]
    empty = sum(1 for d in depths if d == 0)
    print(f"  host {process}: data_wait p50 {statistics.median(waits):.2f} p95 "
          f"{statistics.quantiles(waits, n=20)[-1]:.1f} max {max(waits):.0f} ms; "
          f"queue depth min {min(depths):.0f} empty on {empty}/{len(steps)} steps; "
          f"h2d max {max(rows[s]['host_to_device_ms'] for s in steps):.0f} ms")
  stalls = [s for s in steps if primary[s]["step_ms"] > floor + args.stall_ms]
  print(f"stalls (> p10 + {args.stall_ms:.0f} ms): {len(stalls)}")
  for s in stalls:
    waits = {p: hosts[p][s]["data_wait_ms"] for p in hosts}
    depths = {p: hosts[p][s]["prefetch_queue_depth"] for p in hosts}
    gcs = {p: hosts[p][s].get("gc_ms", 0.0) for p in hosts}
    culprit = max(waits, key=waits.get)
    gc_host = max(gcs, key=gcs.get)
    if waits[culprit] > 5.0:
      reason = f"host {culprit} data_wait {waits[culprit]:.0f} ms (queue depth {depths[culprit]:.0f})"
    elif gcs[gc_host] > 50.0:
      reason = (f"host {gc_host} garbage collection {gcs[gc_host]:.0f} ms "
                f"(gen2 {hosts[gc_host][s].get('gc_gen2', 0)})")
    else:
      reason = "no host reports data_wait > 5 ms or gc > 50 ms (device/collective/other)"
    print(f"  step {s}: fleet {primary[s]['step_ms']:.0f} ms (+{primary[s]['step_ms'] - floor:.0f}); {reason}; "
          f"empty queues on hosts {[p for p, d in depths.items() if d == 0]}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
