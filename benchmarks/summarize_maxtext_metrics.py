#!/usr/bin/env python3
"""Summarize steady-state MaxText pretraining throughput from JSONL metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("metrics_file", type=Path)
  parser.add_argument("--devices", type=int, default=8)
  parser.add_argument(
      "--warmup-steps",
      type=int,
      default=5,
      help="Discard steps with an index below this value.",
  )
  parser.add_argument("--json-output", type=Path)
  return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
  """Return a linearly interpolated percentile for sorted finite values."""
  ordered = sorted(values)
  position = (len(ordered) - 1) * quantile
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main() -> None:
  args = parse_args()
  if args.devices <= 0:
    raise SystemExit("--devices must be positive")

  rows = []
  with args.metrics_file.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON on line {line_number}: {exc}") from exc
      if (
          row.get("step", -1) >= args.warmup_steps
          and "perf/per_device_tokens_per_sec" in row
          and "perf/step_time_seconds" in row
      ):
        rows.append(row)

  if not rows:
    raise SystemExit(
        f"no steady-state rows found in {args.metrics_file}; "
        "lower --warmup-steps or inspect the training log"
    )

  per_device = [float(row["perf/per_device_tokens_per_sec"]) for row in rows]
  step_times = [float(row["perf/step_time_seconds"]) for row in rows]
  tflops = [
      float(row["perf/per_device_tflops_per_sec"])
      for row in rows
      if "perf/per_device_tflops_per_sec" in row
  ]
  losses = [float(row["learning/loss"]) for row in rows if "learning/loss" in row]

  result = {
      "metrics_file": str(args.metrics_file),
      "devices": args.devices,
      "warmup_steps_discarded": args.warmup_steps,
      "measured_steps": len(rows),
      "first_measured_step": int(rows[0]["step"]),
      "last_measured_step": int(rows[-1]["step"]),
      "step_time_seconds": {
          "mean": statistics.fmean(step_times),
          "median": statistics.median(step_times),
          "p90": percentile(step_times, 0.90),
      },
      "tokens_per_second_per_device": {
          "mean": statistics.fmean(per_device),
          "median": statistics.median(per_device),
          "p10": percentile(per_device, 0.10),
          "p90": percentile(per_device, 0.90),
      },
      "tokens_per_second_global": {
          "mean": statistics.fmean(per_device) * args.devices,
          "median": statistics.median(per_device) * args.devices,
          "p10": percentile(per_device, 0.10) * args.devices,
          "p90": percentile(per_device, 0.90) * args.devices,
      },
  }
  if tflops:
    result["tflops_per_second_per_device"] = {
        "mean": statistics.fmean(tflops),
        "median": statistics.median(tflops),
    }
  if losses:
    result["loss"] = {
        "first": losses[0],
        "last": losses[-1],
    }

  print(f"Measured steps:            {len(rows)} ({int(rows[0]['step'])}..{int(rows[-1]['step'])})")
  print(f"Median step time:          {statistics.median(step_times):,.6f} s")
  print(f"Mean tokens/s/device:      {statistics.fmean(per_device):,.0f}")
  print(f"Median tokens/s/device:    {statistics.median(per_device):,.0f}")
  print(f"Mean global tokens/s:      {statistics.fmean(per_device) * args.devices:,.0f}")
  print(f"Median global tokens/s:    {statistics.median(per_device) * args.devices:,.0f}")
  if tflops:
    print(f"Mean TFLOP/s/device:       {statistics.fmean(tflops):,.1f}")
  if losses:
    print(f"Loss (first -> last):      {losses[0]:.5f} -> {losses[-1]:.5f}")

  if args.json_output:
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"JSON summary:              {args.json_output}")


if __name__ == "__main__":
  main()

