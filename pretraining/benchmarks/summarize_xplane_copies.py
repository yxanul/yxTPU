#!/usr/bin/env python3
"""Rank individual copy-class device ops from an ``.xplane.pb`` capture.

The category summary showed copies at ~35% of the v6e step; this tool answers
*which* copies: it aggregates per-op self time for events whose HLO name marks
them as layout/copy work (copy, copy-start/done, bitcast-convert, transpose
fusions) and prints the top offenders with their result shapes, so the fix
can target the actual producer (scan-carry stacks, remat-saved residuals,
kernel operand staging, ...).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

try:
  from tensorboard_plugin_profile.protobuf import xplane_pb2
except ImportError:  # pragma: no cover - fallback for newer xprof layouts
  try:
    from xprof.protobuf import xplane_pb2
  except ImportError:  # pragma: no cover - the venv TF carries the proto too
    from tensorflow.tsl.profiler.protobuf import xplane_pb2


_COPY_MARKERS = ("copy", "bitcast", "transpose")
_SHAPE_PATTERN = re.compile(r"^%?[\w.\-]+\s*=\s*(\([^)]*\)|[a-z0-9]+\[[^\]]*\][^ ]*)")


def _first_shape(hlo_text: str) -> str:
  match = _SHAPE_PATTERN.match(hlo_text)
  if not match:
    return "?"
  shape = match.group(1)
  return shape if len(shape) <= 120 else shape[:117] + "..."


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("xplane", type=Path)
  parser.add_argument("--top", type=int, default=20)
  args = parser.parse_args()

  space = xplane_pb2.XSpace()
  space.ParseFromString(args.xplane.read_bytes())

  device_planes = [
      plane for plane in space.planes if plane.name.startswith("/device:TPU:")
  ]
  if not device_planes:
    raise SystemExit("no device planes found")
  plane = device_planes[0]

  metadata_by_id = dict(plane.event_metadata.items())
  totals: dict[str, float] = defaultdict(float)
  counts: dict[str, int] = defaultdict(int)
  samples: dict[str, str] = {}
  step_span_ps = 0
  for line in plane.lines:
    for event in line.events:
      metadata = metadata_by_id.get(event.metadata_id)
      if metadata is None:
        continue
      full = metadata.name
      display = metadata.display_name or full
      lowered = display.lower()
      if "jit_train_step" in full:
        step_span_ps += event.duration_ps
        continue
      if not any(marker in lowered for marker in _COPY_MARKERS):
        continue
      base = display.split(" = ")[0].strip("%")
      shape = _first_shape(full) if " = " in full else "?"
      key = re.sub(r"\.\d+$", "", base) + " " + shape
      totals[key] += event.duration_ps
      counts[key] += 1
      if key not in samples:
        samples[key] = shape

  ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
  result = {
      "copy_class_total_ms": round(sum(totals.values()) / 1e9, 3),
      "ops": [
          {
              "op": key,
              "total_ms": round(value / 1e9, 3),
              "events": counts[key],
              "shape": samples[key],
          }
          for key, value in ranked[: args.top]
      ],
  }
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()
