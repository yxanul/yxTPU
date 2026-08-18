#!/usr/bin/env python3
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

"""Whole-step device accounting from an ``.xplane.pb`` via xprof's converter.

Modern xprof (2.2x) no longer ships ``xplane_pb2``; its converter does the
XSpace -> OpStats work instead. This reads the ``hlo_stats``, ``op_profile``
and ``overview_page`` tools and buckets device self-time by component:
dense GEMMs (split fwd / bwd / rematted recompute / einsum spec), KDA fused
kernels and their XLA glue (depthwise conv, casts, pads), splash attention,
loss head, ViT, collectives, and formatting - plus device duty cycle and
xprof's MXU utilization. NNX-scanned modules lose their names in the
``tf_op`` scope (``while/body/closed_call/checkpoint/...``), so grouping is
by einsum spec / op kind rather than by module.

  uv run --with xprof python benchmarks/summarize_xplane_step.py <xplane.pb> \
      [--step-ms 1713] [--top 30] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _table(js):
  cols = [c["id"] for c in js["cols"]]
  return [{c: (cell["v"] if cell else None) for c, cell in zip(cols, r["c"])} for r in js["rows"]]


def _kind(r) -> str:
  cat = r["category"] or ""
  scope = r["tf_op_name"] or ""
  name = r["hlo_op_name"] or ""
  low = (scope + " " + name).lower()
  expr = (r["hlo_op_expression"] or "")
  if "all-reduce" in cat or "all-gather" in cat or "reduce-scatter" in cat:
    if "while/body" in scope:
      return "collectives: gradient all-reduce (in scan)"
    return "collectives: embedding / loss-head all-reduce"
  # Outside the cycle scan the only vmapped work is the optimizer over the
  # stacked [cycles, ...] parameters: Muon Newton-Schulz matmuls and norms.
  if "while/body" not in scope and "vmap(" in scope:
    if "dot_general" in scope:
      return "optimizer: Muon Newton-Schulz matmuls"
    return "optimizer: Muon norms / elementwise (vmapped)"
  if "scatter-add" in low or "gather" in low:
    return "embedding gather / scatter-add"
  if re.search(r"jvp\((bqhd|bhqk)", scope):
    return "ViT attention einsums"
  if "conv_general_dilated" in scope and re.search(r"= bf16\[4,1,", expr):
    return "KDA depthwise conv: weight gradient (XLA reduce)"
  if "splash" in low or "flash_attention" in low:
    return "GQA splash attention (pallas)"
  if "kda_" in low or "pallas_kda" in low:
    return "KDA fused kernels (pallas)"
  if "conv_general_dilated" in scope:
    return "KDA depthwise conv (XLA)"
  m = re.search(r"/([a-z.]+,[a-z.]+->[a-z.]+)/dot_general", scope)
  if m:
    spec = m.group(1)
    if spec in ("bth,btv->hv", "btv,hv->bth", "bth,hv->btv"):
      return "loss head GEMMs (chunked CE)"
    if "hqk" in spec or "bqhd" in spec:
      return "ViT attention einsums"
    return f"einsum {spec}"
  if "rematted_computation/dot_general" in scope:
    return "dense GEMMs: rematted fwd recompute"
  if "dot_general" in scope:
    return "dense GEMMs: fwd (checkpoint)" if "/checkpoint/" in scope else "dense GEMMs: bwd/other"
  if cat == "convolution fusion":
    return "other convolution fusions"
  if "convert_element_type" in scope:
    return "casts (convert_element_type)"
  if any(k in scope for k in ("pad:", "reshape:", "transpose:", "copy")) or cat in ("data formatting", "copy-done", "copy-start"):
    return "pads / reshapes / copies"
  if cat == "loop fusion" or cat == "non-fusion elementwise":
    return "elementwise (norms, gates, softmax, adds)"
  if cat == "reduce":
    return "reductions"
  return "other: " + cat


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("xplane", type=Path)
  parser.add_argument("--step-ms", type=float, default=None,
                      help="wall step of the traced steps (from the trainer's records); "
                           "used to convert shares into ms/step")
  parser.add_argument("--top", type=int, default=30)
  parser.add_argument("--json", type=Path, default=None)
  args = parser.parse_args()

  from xprof.convert import raw_to_tool_data as convert

  hlo, _ = convert.xspace_to_tool_data([str(args.xplane)], "hlo_stats", {})
  ovw, _ = convert.xspace_to_tool_data([str(args.xplane)], "overview_page", {})
  opp, _ = convert.xspace_to_tool_data([str(args.xplane)], "op_profile", {})
  rows = _table(json.loads(hlo if isinstance(hlo, str) else hlo.decode()))
  overview = json.loads(ovw if isinstance(ovw, str) else ovw.decode())
  op_profile = json.loads(opp if isinstance(opp, str) else opp.decode())
  duty = overview[0]["p"].get("device_duty_cycle_percent", "?")
  duty_frac = float(str(duty).rstrip("%")) / 100 if str(duty).endswith("%") else None
  mxu = op_profile["byProgram"]["metrics"].get("flops")

  total = sum(float(r["total_self_time"]) for r in rows) or 1.0
  by_kind = defaultdict(float)
  for r in rows:
    by_kind[_kind(r)] += float(r["total_self_time"])
  busy_ms = (args.step_ms * duty_frac) if (args.step_ms and duty_frac) else None

  def row(share):
    out = {"percent_of_device_time": round(100 * share, 2)}
    if busy_ms:
      out["ms_per_step"] = round(busy_ms * share, 1)
    return out

  report = {
      "device_duty_cycle": duty,
      "xprof_mxu_flops_utilization_percent": round(100 * mxu, 1) if mxu is not None else None,
      "note": "xprof cannot count FLOPs of Pallas custom calls (KDA/splash), so its MXU "
              "utilization undercounts; use the model-FLOP MFU for the headline.",
      "step_ms_assumed": args.step_ms,
      "device_busy_ms": round(busy_ms, 1) if busy_ms else None,
      "by_component": {k: row(v / total) for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])},
      "top_ops": [
          {
              "percent": round(100 * float(r["total_self_time"]) / total, 2),
              "category": r["category"], "op": r["hlo_op_name"], "bound_by": r["bound_by"],
              "occurrences": float(r["occurrences"]),
              "model_flop_rate_gflops": round(float(r["model_flop_rate"] or 0)),
              "scope": re.sub(r"jit\(train_step\)/(transpose\(jvp\(\)\)|jvp\(\))?/?", "", r["tf_op_name"] or "")[-100:],
          }
          for r in sorted(rows, key=lambda r: -float(r["total_self_time"]))[: args.top]
      ],
  }
  text = json.dumps(report, indent=1)
  print(text)
  if args.json:
    args.json.write_text(text)


if __name__ == "__main__":
  main()
