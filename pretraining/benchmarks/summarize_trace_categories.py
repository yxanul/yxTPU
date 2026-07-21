#!/usr/bin/env python3
"""Summarize per-step XLA op time by HLO category and named-kernel bucket.

Unlike ``benchmarks/analyze_xplane_trace.py`` at the repository root, this
summary makes no assumption about decoder scan structure, so it works for the
standalone trainer's gradient-accumulated update. It reports mean time per
profiled module occurrence, grouped by ``hlo_category`` and by a small set of
named buckets that the KDA optimization work tracks across profiles.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


_NAMED_BUCKETS = (
    ("kda_fused_forward", "fused KDA forward"),
    ("kda_fused_backward", "fused KDA backward"),
    ("splash", "splash attention"),
    ("conv", "convolution ops"),
    ("transpose", "transpose ops"),
    ("copy", "copy ops"),
    ("fusion", "xla fusions"),
)


def _load_events(path: Path):
  opener = gzip.open if path.suffix == ".gz" else open
  with opener(path, "rt", encoding="utf-8") as trace_file:
    return json.load(trace_file)["traceEvents"]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("trace", type=Path)
  parser.add_argument(
      "--module-prefix",
      default="jit_",
      help="XLA module name prefix that marks one profiled step",
  )
  parser.add_argument(
      "--top",
      type=int,
      default=25,
      help="number of individual op names to report",
  )
  args = parser.parse_args()
  events = _load_events(args.trace)

  device_pids = sorted(
      {
          event["pid"]
          for event in events
          if event.get("ph") == "M"
          and event.get("name") == "process_name"
          and "TPU" in event.get("args", {}).get("name", "")
      }
  )
  if not device_pids:
    raise SystemExit("no TPU device process found in trace")
  device_pid = device_pids[0]
  thread_names = {
      event["tid"]: event.get("args", {}).get("name", "")
      for event in events
      if event.get("ph") == "M"
      and event.get("name") == "thread_name"
      and event.get("pid") == device_pid
  }
  module_tid = next(tid for tid, name in thread_names.items() if name == "XLA Modules")
  ops_tid = next(tid for tid, name in thread_names.items() if name == "XLA Ops")

  modules = sorted(
      (
          event
          for event in events
          if event.get("ph") == "X"
          and event.get("pid") == device_pid
          and event.get("tid") == module_tid
          and event.get("name", "").startswith(args.module_prefix)
      ),
      key=lambda event: event["ts"],
  )
  if not modules:
    raise SystemExit(f"no XLA modules matching prefix {args.module_prefix!r}")
  module_names = defaultdict(int)
  for module in modules:
    module_names[module["name"]] += 1

  def owning_module(event) -> int | None:
    for index, module in enumerate(modules):
      if (
          event["ts"] >= module["ts"]
          and event["ts"] + event.get("dur", 0) <= module["ts"] + module.get("dur", 0) + 0.01
      ):
        return index
    return None

  ops_by_module = defaultdict(list)
  for event in events:
    if (
        event.get("ph") != "X"
        or event.get("pid") != device_pid
        or event.get("tid") != ops_tid
        or event.get("name", "").startswith("while.")
    ):
      continue
    index = owning_module(event)
    if index is not None:
      ops_by_module[index].append(event)

  # A trace window can truncate the first or last module occurrence, leaving a
  # module event whose ops were only partially recorded. Keep occurrences whose
  # recorded op time is comparable to the best-covered occurrence.
  op_totals = {
      index: sum(event.get("dur", 0.0) for event in ops)
      for index, ops in ops_by_module.items()
  }
  best_total = max(op_totals.values(), default=0.0)
  kept_indices = [index for index, total in op_totals.items() if total >= 0.8 * best_total]
  if not kept_indices:
    raise SystemExit("no module occurrence with recorded ops")
  ops = [event for index in kept_indices for event in ops_by_module[index]]

  steps = len(kept_indices)
  step_ms = sum(modules[index]["dur"] for index in kept_indices) / steps / 1_000

  by_category = defaultdict(float)
  by_name = defaultdict(float)
  by_bucket = defaultdict(float)
  for event in ops:
    duration = event.get("dur", 0.0)
    by_category[event.get("args", {}).get("hlo_category", "unclassified")] += duration
    by_name[event.get("name", "?")] += duration
    lowered = event.get("name", "").lower()
    for needle, bucket in _NAMED_BUCKETS:
      if needle in lowered:
        by_bucket[bucket] += duration
        break

  def as_row(duration_us: float) -> dict[str, float]:
    per_step = duration_us / steps / 1_000
    return {"ms_per_module": round(per_step, 3), "percent": round(100 * per_step / step_ms, 2)}

  print(
      json.dumps(
          {
              "modules": dict(module_names),
              "module_occurrences_in_trace": len(modules),
              "complete_module_occurrences_kept": steps,
              "mean_module_ms": round(step_ms, 3),
              "hlo_categories": {
                  category: as_row(duration)
                  for category, duration in sorted(by_category.items(), key=lambda kv: -kv[1])
              },
              "named_buckets": {
                  bucket: as_row(duration)
                  for bucket, duration in sorted(by_bucket.items(), key=lambda kv: -kv[1])
              },
              "top_ops": {
                  name: as_row(duration)
                  for name, duration in sorted(by_name.items(), key=lambda kv: -kv[1])[: args.top]
              },
          },
          indent=1,
      )
  )


if __name__ == "__main__":
  main()
