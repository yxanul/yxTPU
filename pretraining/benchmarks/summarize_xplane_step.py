#!/usr/bin/env python3
"""Honest per-step device-time breakdown from an ``.xplane.pb`` capture.

The category summarizer name-matches op strings (instruction text embeds
operand names) and counts control-flow parents (``while`` spans) alongside
their own children, which double-counts scan bodies and can hide or inflate
categories. This tool instead:

- scopes to one TPU core's op line, over complete ``jit_train_step`` spans;
- drops control-flow parent events (while/conditional/call) so every counted
  event is a leaf execution (fusions, custom calls, copies, collectives);
- reports the reconciliation first: step span, summed leaf time (device
  busy), and the residual idle/dispatch gap - if those do not add up, the
  capture is not trustworthy;
- buckets by opcode parsed from the HLO text (never by substring of the
  instruction name), and ranks individual leaf ops by self time.
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


_CONTROL_FLOW = ("while", "conditional", "call")
# opcode appears after the result tuple/shape: "%name = shape opcode(...)"
_OPCODE_PATTERN = re.compile(
    r"=\s*(?:\([^)]*\)|[a-z0-9]+\[[^\]]*\][^\s]*)\s+([a-z0-9\-]+)\("
)


def _opcode(hlo_text: str) -> str | None:
  match = _OPCODE_PATTERN.search(hlo_text)
  return match.group(1) if match else None


def _base_name(display: str) -> str:
  return re.sub(r"\.\d+$", "", display.split(" = ")[0].strip("%"))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("xplane", type=Path)
  parser.add_argument("--top", type=int, default=25)
  args = parser.parse_args()

  space = xplane_pb2.XSpace()
  space.ParseFromString(args.xplane.read_bytes())
  planes = [p for p in space.planes if p.name.startswith("/device:TPU:")]
  if not planes:
    raise SystemExit("no device planes")
  plane = planes[0]
  metadata = dict(plane.event_metadata.items())

  # Step spans from the module line; leaf ops from every line, scoped inside.
  step_spans: list[tuple[int, int]] = []
  for line in plane.lines:
    for event in line.events:
      meta = metadata.get(event.metadata_id)
      if meta is None:
        continue
      if "jit_train_step" in (meta.display_name or meta.name):
        step_spans.append((event.offset_ps, event.offset_ps + event.duration_ps))
  if not step_spans:
    raise SystemExit("no jit_train_step occurrences on the device plane")
  step_spans.sort()
  total_span_ps = sum(end - start for start, end in step_spans)
  steps = len(step_spans)

  def in_step(offset: int) -> bool:
    return any(start <= offset < end for start, end in step_spans)

  # The device plane multiplexes the compute stream with DMA queue lines
  # whose events overlap compute; summing across lines double-counts. Busy
  # time and the op ranking come from compute lines only; DMA lines are
  # reported separately as overlapped traffic.
  op_totals: dict[str, int] = defaultdict(int)
  op_counts: dict[str, int] = defaultdict(int)
  opcode_totals: dict[str, int] = defaultdict(int)
  line_totals: dict[str, int] = defaultdict(int)
  busy_ps = 0
  for line in plane.lines:
    is_compute = line.name.strip() == "XLA Ops"
    for event in line.events:
      meta = metadata.get(event.metadata_id)
      if meta is None:
        continue
      full = meta.name
      display = meta.display_name or full
      if "jit_train_step" in display:
        continue
      if not in_step(event.offset_ps):
        continue
      base = _base_name(display)
      if any(base == flow or base.startswith(flow + ".") for flow in _CONTROL_FLOW):
        continue
      line_totals[line.name] += event.duration_ps
      if not is_compute or " = " not in full:
        continue
      opcode = _opcode(full) or "?"
      if opcode in _CONTROL_FLOW:
        continue
      busy_ps += event.duration_ps
      opcode_totals[opcode] += event.duration_ps
      shape = full.split(" = ", 1)[1][:44] if " = " in full else ""
      key = f"{base} [{opcode}] {shape}"
      op_totals[key] += event.duration_ps
      op_counts[key] += 1

  def ms_per_step(ps: int) -> float:
    return round(ps / steps / 1e9, 3)

  result = {
      "steps_in_trace": steps,
      "mean_step_span_ms": ms_per_step(total_span_ps),
      "compute_busy_ms_per_step": ms_per_step(busy_ps),
      "idle_or_unattributed_ms_per_step": ms_per_step(total_span_ps - busy_ps),
      "per_line_ms_per_step": {
          name: ms_per_step(value)
          for name, value in sorted(
              line_totals.items(), key=lambda item: item[1], reverse=True
          )[:8]
      },
      "by_opcode_ms_per_step": {
          opcode: ms_per_step(value)
          for opcode, value in sorted(
              opcode_totals.items(), key=lambda item: item[1], reverse=True
          )[:12]
      },
      "top_leaf_ops": [
          {
              "op": key,
              "ms_per_step": ms_per_step(value),
              "calls_per_step": round(op_counts[key] / steps, 1),
          }
          for key, value in sorted(
              op_totals.items(), key=lambda item: item[1], reverse=True
          )[: args.top]
      ],
  }
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()
