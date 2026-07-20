#!/usr/bin/env python3
"""Summarize steady-state MaxText TPU steps from an XPlane Chrome trace."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_trace(path: Path) -> list[dict[str, Any]]:
  opener = gzip.open if path.suffix == ".gz" else open
  with opener(path, "rt", encoding="utf-8") as trace_file:
    trace = json.load(trace_file)
  return trace["traceEvents"]


def _metadata_name(
    events: list[dict[str, Any]], metadata_kind: str, pid: int, tid: int | None = None
) -> str:
  for event in events:
    if event.get("ph") != "M" or event.get("name") != metadata_kind or event.get("pid") != pid:
      continue
    if tid is not None and event.get("tid") != tid:
      continue
    return event.get("args", {}).get("name", "")
  return ""


def _stats(total_us: float, steps: int, step_us: float) -> dict[str, float]:
  per_step_ms = total_us / steps / 1_000
  return {
      "milliseconds_per_step": per_step_ms,
      "percent_of_step": 100 * total_us / steps / step_us,
  }


def analyze(events: list[dict[str, Any]]) -> dict[str, Any]:
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
    raise ValueError("no TPU device process found in trace")
  device_pid = device_pids[0]

  thread_names = {
      event["tid"]: event.get("args", {}).get("name", "")
      for event in events
      if event.get("ph") == "M"
      and event.get("name") == "thread_name"
      and event.get("pid") == device_pid
  }
  module_tid = next((tid for tid, name in thread_names.items() if name == "XLA Modules"), None)
  ops_tid = next((tid for tid, name in thread_names.items() if name == "XLA Ops"), None)
  if module_tid is None or ops_tid is None:
    raise ValueError(f"missing XLA Modules or XLA Ops tracks: {thread_names}")

  modules = sorted(
      (
          event
          for event in events
          if event.get("ph") == "X"
          and event.get("pid") == device_pid
          and event.get("tid") == module_tid
          and event.get("name", "").startswith("jit_train_step")
      ),
      key=lambda event: event["ts"],
  )
  if not modules:
    raise ValueError("no jit_train_step modules found")

  all_ops = [
      event
      for event in events
      if event.get("ph") == "X" and event.get("pid") == device_pid and event.get("tid") == ops_tid
  ]
  scans = [event for event in all_ops if event.get("name", "").startswith("while.")]
  leaf_ops = [event for event in all_ops if not event.get("name", "").startswith("while.")]

  def is_inside(event: dict[str, Any], parent: dict[str, Any]) -> bool:
    return (
        event["ts"] >= parent["ts"]
        and event["ts"] + event.get("dur", 0) <= parent["ts"] + parent.get("dur", 0) + 0.01
    )

  profiled_ops = [
      event for event in leaf_ops if any(is_inside(event, module) for module in modules)
  ]
  step_count = len(modules)
  mean_step_us = sum(module["dur"] for module in modules) / step_count

  phase_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
  phase_order = (
      "embedding/input prep",
      "forward transformer scan",
      "output head + loss",
      "backward transformer scan",
      "post-backward + optimizer/metrics",
  )
  for module in modules:
    nested_scans = [scan for scan in scans if is_inside(scan, module)]
    # Recurrent models add many inner lax.scan/fori_loop `while` operations
    # inside each decoder pass.  The forward and backward decoder scans are
    # the two top-level while events; retain only scans that are not enclosed
    # by another while in the same train step.
    module_scans = sorted(
        (
            scan
            for scan in nested_scans
            if not any(
                parent is not scan and parent["dur"] > scan["dur"] and is_inside(scan, parent)
                for parent in nested_scans
            )
        ),
        key=lambda event: event["ts"],
    )
    if len(module_scans) != 2:
      raise ValueError(
          f"expected two decoder scans inside each train step, found {len(module_scans)} "
          f"at timestamp {module['ts']}"
      )
    forward_scan, backward_scan = module_scans
    for event in profiled_ops:
      if not is_inside(event, module):
        continue
      if is_inside(event, forward_scan):
        phase = phase_order[1]
      elif is_inside(event, backward_scan):
        phase = phase_order[3]
      elif event["ts"] < forward_scan["ts"]:
        phase = phase_order[0]
      elif event["ts"] < backward_scan["ts"]:
        phase = phase_order[2]
      else:
        phase = phase_order[4]
      phase_events[phase].append(event)

  phases = {
      phase: _stats(sum(event["dur"] for event in phase_events[phase]), step_count, mean_step_us)
      for phase in phase_order
  }

  hlo_totals: dict[str, float] = defaultdict(float)
  for event in profiled_ops:
    hlo_totals[event.get("args", {}).get("hlo_category", "unclassified")] += event["dur"]
  hlo_categories = {
      category: _stats(duration, step_count, mean_step_us)
      for category, duration in sorted(hlo_totals.items(), key=lambda item: -item[1])
  }

  attention_names = {
      "forward": "splash_mha_fwd",
      "backward dKV": "splash_mha_dkv",
      "backward dQ": "splash_mha_dq",
  }
  attention_durations = {
      label: sum(event["dur"] for event in profiled_ops if event.get("name", "").startswith(prefix))
      for label, prefix in attention_names.items()
  }
  # Tokamax Splash emits dQ, dK, and dV from one kernel whose trace name keeps
  # the historical dKV suffix. JAX Splash instead records dKV and dQ separately.
  if attention_durations["backward dKV"] and not attention_durations["backward dQ"]:
    attention_durations["backward fused dQ/dK/dV"] = attention_durations.pop("backward dKV")
    attention_durations.pop("backward dQ")
  attention: dict[str, dict[str, float]] = {}
  attention_total = 0.0
  for label, duration in attention_durations.items():
    attention[label] = _stats(duration, step_count, mean_step_us)
    attention_total += duration
  attention["total"] = _stats(attention_total, step_count, mean_step_us)

  muon_ops = [
      event
      for event in profiled_ops
      if "optax/contrib/_muon.py"
      in f"{event.get('args', {}).get('source', '')}\n{event.get('args', {}).get('source_stack', '')}"
  ]
  muon: dict[str, Any] | None = None
  if muon_ops:
    muon_hlo_totals: dict[str, float] = defaultdict(float)
    for event in muon_ops:
      muon_hlo_totals[event.get("args", {}).get("hlo_category", "unclassified")] += event["dur"]
    muon = {
        "total": _stats(sum(event["dur"] for event in muon_ops), step_count, mean_step_us),
        "events_per_step": len(muon_ops) / step_count,
        "xla_hlo_categories": {
            category: _stats(duration, step_count, mean_step_us)
            for category, duration in sorted(muon_hlo_totals.items(), key=lambda item: -item[1])
        },
    }

  kda_markers = ("kimi_delta_attention.py", "kda_fused_pallas.py")
  kda_ops = [
      event
      for event in profiled_ops
      if any(
          marker in f"{event.get('args', {}).get('source', '')}\n"
          f"{event.get('args', {}).get('source_stack', '')}"
          for marker in kda_markers
      )
  ]
  kda: dict[str, Any] | None = None
  if kda_ops:
    # Use the operation's direct source line for disjoint hotspot groups. Some
    # fused operations only retain KDA in source_stack, so the overall
    # source-or-stack total is intentionally larger than the grouped total.
    source_line_pattern = re.compile(r"kimi_delta_attention\.py:(\d+)")
    kda_line_totals: dict[int, float] = defaultdict(float)
    for event in kda_ops:
      source = event.get("args", {}).get("source", "")
      match = source_line_pattern.search(source)
      if match:
        kda_line_totals[int(match.group(1))] += event["dur"]

    kda_group_ranges = (
        ("Pallas blocked triangular solve and custom VJP", 41, 247),
        ("decay-weighted pairwise forward", 303, 373),
        ("blockwise pairwise analytical VJP", 374, 480),
        ("generic chunk forward setup", 482, 580),
        ("generic inter-chunk recurrence", 581, 606),
        ("analytical recompute and state reconstruction", 607, 809),
        ("analytical reverse state recurrence", 810, 894),
        ("analytical inverse, pairwise, and input VJPs", 895, 1029),
        ("public KDA dispatch", 1031, 1076),
        ("QKV convolution and gates", 1215, 1253),
        ("shard/rematerialization wrapper", 1254, 1298),
        ("output gate, norm, and projection", 1299, 1302),
    )
    kda_groups: dict[str, dict[str, float]] = {}
    grouped_lines: set[int] = set()
    for label, first_line, last_line in kda_group_ranges:
      duration = sum(
          duration for line, duration in kda_line_totals.items() if first_line <= line <= last_line
      )
      grouped_lines.update(line for line in kda_line_totals if first_line <= line <= last_line)
      if duration:
        kda_groups[label] = _stats(duration, step_count, mean_step_us)
    ungrouped_duration = sum(
        duration for line, duration in kda_line_totals.items() if line not in grouped_lines
    )
    if ungrouped_duration:
      kda_groups["other directly attributed KDA"] = _stats(
          ungrouped_duration, step_count, mean_step_us
      )

    kda_hlo_totals: dict[str, float] = defaultdict(float)
    for event in kda_ops:
      kda_hlo_totals[event.get("args", {}).get("hlo_category", "unclassified")] += event["dur"]
    directly_attributed_duration = sum(kda_line_totals.values())
    kda = {
        "source_or_stack_total": _stats(
            sum(event["dur"] for event in kda_ops), step_count, mean_step_us
        ),
        "direct_source_total": _stats(directly_attributed_duration, step_count, mean_step_us),
        "direct_source_groups": kda_groups,
        "xla_hlo_categories": {
            category: _stats(duration, step_count, mean_step_us)
            for category, duration in sorted(kda_hlo_totals.items(), key=lambda item: -item[1])
        },
    }

  fused_kda_kernel_prefixes = {
      "forward": "kda_fused_forward_doubling_full",
      "backward": "kda_fused_backward_full",
  }
  fused_kda_durations = {
      label: sum(event["dur"] for event in profiled_ops if event.get("name", "").startswith(prefix))
      for label, prefix in fused_kda_kernel_prefixes.items()
  }
  fused_kda: dict[str, dict[str, float]] | None = None
  if any(fused_kda_durations.values()):
    fused_kda = {
        label: _stats(duration, step_count, mean_step_us)
        for label, duration in fused_kda_durations.items()
    }
    fused_kda["total"] = _stats(
        sum(fused_kda_durations.values()),
        step_count,
        mean_step_us,
    )

  leaf_us = sum(event["dur"] for event in profiled_ops)
  result = {
      "device": _metadata_name(events, "process_name", device_pid),
      "profiled_steps": step_count,
      "step_time_milliseconds": {
          "mean": mean_step_us / 1_000,
          "min": min(module["dur"] for module in modules) / 1_000,
          "max": max(module["dur"] for module in modules) / 1_000,
      },
      "leaf_device_time_accounted_percent": 100 * leaf_us / step_count / mean_step_us,
      "training_phases": phases,
      "splash_attention_kernels": attention,
      "xla_hlo_categories": hlo_categories,
  }
  if muon is not None:
    result["muon_optimizer"] = muon
  if kda is not None:
    result["kimi_delta_attention"] = kda
  if fused_kda is not None:
    result["fused_pallas_kda_kernels"] = fused_kda
  return result


def _format_report(result: dict[str, Any]) -> str:
  lines = [
      "# TPU XPlane profile summary",
      "",
      f"- Device: `{result['device']}`",
      f"- Profiled optimizer steps: {result['profiled_steps']}",
      f"- Mean device step: {result['step_time_milliseconds']['mean']:.3f} ms",
      f"- Leaf device time accounted: {result['leaf_device_time_accounted_percent']:.2f}%",
      "",
      "## Training phases",
      "",
      "| Phase | ms/step | % of step |",
      "| --- | ---: | ---: |",
  ]
  for phase, values in result["training_phases"].items():
    lines.append(
        f"| {phase} | {values['milliseconds_per_step']:.3f} | {values['percent_of_step']:.2f}% |"
    )
  lines.extend(
      [
          "",
          "The post-backward phase contains fused output/embedding gradients, gradient",
          "clipping and norm metrics, and the optimizer update. XLA fusion prevents a reliable",
          "finer split of that phase from source attribution alone.",
          "",
          "## Splash Attention kernels",
          "",
          "| Kernel | ms/step | % of step |",
          "| --- | ---: | ---: |",
      ]
  )
  for kernel, values in result["splash_attention_kernels"].items():
    lines.append(
        f"| {kernel} | {values['milliseconds_per_step']:.3f} | {values['percent_of_step']:.2f}% |"
    )
  if "backward fused dQ/dK/dV" in result["splash_attention_kernels"]:
    lines.extend(
        [
            "",
            "Tokamax Splash computes dQ, dK, and dV in this single backward Pallas",
            "kernel; its underlying trace name retains the historical `dkv` suffix.",
        ]
    )
  if "muon_optimizer" in result:
    muon = result["muon_optimizer"]
    lines.extend(
        [
            "",
            "## Muon optimizer lowering",
            "",
            "These are leaf TPU operations whose source stack points into Optax Muon.",
            "They are a subset of the post-backward phase.",
            "",
            f"- Source-attributed Muon time: {muon['total']['milliseconds_per_step']:.3f} ms/step "
            f"({muon['total']['percent_of_step']:.2f}%)",
            f"- Source-attributed leaf operations: {muon['events_per_step']:.1f}/step",
            "",
            "| Muon XLA category | ms/step | % of whole step |",
            "| --- | ---: | ---: |",
        ]
    )
    for category, values in muon["xla_hlo_categories"].items():
      lines.append(
          f"| {category} | {values['milliseconds_per_step']:.3f} | {values['percent_of_step']:.2f}% |"
      )
    lines.extend(
        [
            "",
            "The Newton–Schulz matrix multiplications lower as XLA `convolution fusion`",
            "operations on the TPU MXU. No Pallas/custom-call Muon kernel appears in the trace.",
        ]
    )
  if "kimi_delta_attention" in result:
    kda = result["kimi_delta_attention"]
    lines.extend(
        [
            "",
            "## Kimi Delta Attention source attribution",
            "",
            "These are leaf TPU operations whose direct source or source stack points",
            "into the KDA implementation. Direct-source groups are disjoint; operations",
            "that retain KDA only in their fused source stack are included in the overall",
            "total but not assigned to a group.",
            "",
            f"- Source-or-stack KDA time: "
            f"{kda['source_or_stack_total']['milliseconds_per_step']:.3f} ms/step "
            f"({kda['source_or_stack_total']['percent_of_step']:.2f}%)",
            f"- Direct-source grouped time: "
            f"{kda['direct_source_total']['milliseconds_per_step']:.3f} ms/step "
            f"({kda['direct_source_total']['percent_of_step']:.2f}%)",
            "",
            "| Direct-source KDA group | ms/step | % of whole step |",
            "| --- | ---: | ---: |",
        ]
    )
    for group, values in kda["direct_source_groups"].items():
      lines.append(
          f"| {group} | {values['milliseconds_per_step']:.3f} | "
          f"{values['percent_of_step']:.2f}% |"
      )
    lines.extend(
        [
            "",
            "The source groups distinguish the generic autodiff path from the analytical",
            "whole-KDA VJP. Source attribution may still assign a fused operation to only",
            "one contributing line, so use the phase totals for the authoritative split.",
        ]
    )
  if "fused_pallas_kda_kernels" in result:
    lines.extend(
        [
            "",
            "## Fused Pallas KDA kernels",
            "",
            "| Kernel | ms/step | % of step |",
            "| --- | ---: | ---: |",
        ]
    )
    for kernel, values in result["fused_pallas_kda_kernels"].items():
      lines.append(
          f"| {kernel} | {values['milliseconds_per_step']:.3f} | "
          f"{values['percent_of_step']:.2f}% |"
      )
  lines.extend(
      [
          "",
          "## XLA HLO categories",
          "",
          "| XLA category | ms/step | % of step |",
          "| --- | ---: | ---: |",
      ]
  )
  for category, values in result["xla_hlo_categories"].items():
    lines.append(
        f"| {category} | {values['milliseconds_per_step']:.3f} | {values['percent_of_step']:.2f}% |"
    )
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("trace", type=Path)
  parser.add_argument("--json-output", type=Path)
  parser.add_argument("--markdown-output", type=Path)
  args = parser.parse_args()

  result = analyze(_load_trace(args.trace))
  if args.json_output:
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  report = _format_report(result)
  if args.markdown_output:
    args.markdown_output.write_text(report, encoding="utf-8")
  print(report, end="")


if __name__ == "__main__":
  main()
