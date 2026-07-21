#!/usr/bin/env python3
"""Summarize device-op time per train step directly from an ``.xplane.pb``.

The Chrome-trace export truncates at roughly one million events, and on this
trainer the host python threads and counters consume nearly the whole budget,
so device backward ops silently vanish from the exported JSON. This reader
parses the raw XPlane protobuf instead, scopes XLA ops to complete
``jit_train_step`` occurrences on one device, and reports HLO-category and
named-kernel shares that survive event-count limits.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
  from tensorboard_plugin_profile.protobuf import xplane_pb2
except ImportError:  # pragma: no cover - fallback for newer xprof layouts
  from xprof.protobuf import xplane_pb2


_NAMED_BUCKETS = (
    ("kda_fused_forward", "fused KDA forward"),
    ("kda_fused_backward", "fused KDA backward"),
    ("splash", "splash attention"),
    ("conv", "convolution ops"),
    ("transpose", "transpose ops"),
    ("copy", "copy ops"),
    ("fusion", "xla fusions"),
)


def _stat_value(event, plane, wanted: str):
  for stat in event.stats:
    metadata = plane.stat_metadata.get(stat.metadata_id)
    if metadata is None or metadata.name != wanted:
      continue
    if stat.ref_value:
      return plane.stat_metadata[stat.ref_value].name
    return stat.str_value or None
  return None


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("xplane", type=Path)
  parser.add_argument("--module-prefix", default="jit_train_step")
  parser.add_argument("--top", type=int, default=20)
  args = parser.parse_args()

  space = xplane_pb2.XSpace()
  space.ParseFromString(args.xplane.read_bytes())

  plane = None
  for candidate in space.planes:
    if "/device:TPU:" in candidate.name and any(
        line.name == "XLA Ops" for line in candidate.lines
    ):
      plane = candidate
      break
  if plane is None:
    raise SystemExit("no TPU plane with an XLA Ops line found")

  lines = {line.name: line for line in plane.lines}
  module_line = lines.get("XLA Modules")
  ops_line = lines.get("XLA Ops")
  if module_line is None or ops_line is None:
    raise SystemExit(f"missing XLA Modules or XLA Ops lines: {sorted(lines)}")

  def absolute_ps(line, event) -> tuple[float, float]:
    start = line.timestamp_ns * 1_000 + event.offset_ps
    return start, start + event.duration_ps

  modules = []
  for event in module_line.events:
    name = plane.event_metadata[event.metadata_id].name
    if name.startswith(args.module_prefix):
      modules.append((*absolute_ps(module_line, event), name))
  modules.sort()
  if not modules:
    raise SystemExit(f"no modules matching prefix {args.module_prefix!r}")

  ops_by_module = defaultdict(list)
  for event in ops_line.events:
    name = plane.event_metadata[event.metadata_id].name
    if name.startswith("while."):
      continue
    start, end = absolute_ps(ops_line, event)
    for index, (module_start, module_end, _) in enumerate(modules):
      if start >= module_start and end <= module_end + 10_000:
        category = _stat_value(event, plane, "hlo_category") or "unclassified"
        ops_by_module[index].append((name, category, event.duration_ps))
        break

  op_totals = {
      index: sum(duration for _, _, duration in ops)
      for index, ops in ops_by_module.items()
  }
  best_total = max(op_totals.values(), default=0)
  kept = [index for index, total in op_totals.items() if total >= 0.8 * best_total]
  if not kept:
    raise SystemExit("no module occurrence with recorded ops")

  steps = len(kept)
  step_ms = sum(modules[index][1] - modules[index][0] for index in kept) / steps / 1e9

  by_category = defaultdict(float)
  by_name = defaultdict(float)
  by_bucket = defaultdict(float)
  for index in kept:
    for name, category, duration in ops_by_module[index]:
      by_category[category] += duration
      by_name[name] += duration
      lowered = name.lower()
      for needle, bucket in _NAMED_BUCKETS:
        if needle in lowered:
          by_bucket[bucket] += duration
          break

  def as_row(duration_ps: float) -> dict[str, float]:
    per_step = duration_ps / steps / 1e9
    return {"ms_per_step": round(per_step, 3), "percent": round(100 * per_step / step_ms, 2)}

  print(
      json.dumps(
          {
              "device_plane": plane.name,
              "module_occurrences_in_trace": len(modules),
              "complete_module_occurrences_kept": steps,
              "mean_step_ms": round(step_ms, 3),
              "op_coverage_percent": round(
                  100 * sum(op_totals[i] for i in kept) / steps / 1e9 / step_ms, 2
              ),
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
