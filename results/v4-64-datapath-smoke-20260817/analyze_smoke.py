import sys, wandb, numpy as np, pandas as pd
pd.set_option("display.width", 250)
api = wandb.Api(timeout=300)
proj = "davidfranco2300-other/yxtpu-pretrain"
name_regex = sys.argv[1] if len(sys.argv) > 1 else "vision_1b_smoke_4k"
runs = list(api.runs(proj, filters={"display_name": {"$regex": name_regex}}, order="-created_at"))
r = runs[0]
print("run", r.id, r.name, r.state, "steps", r.lastHistoryStep)
keys = [k for k in r.summary.keys() if k.startswith(("performance/", "data/", "hosts/", "eval/", "diagnostics/grad_norm", "diagnostics/loss", "attention/", "train/loss", "vision/"))]
rows = list(r.scan_history(page_size=5000))
df = pd.DataFrame(rows)
p = df.dropna(subset=["performance/step_ms"])
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 50
p = p[p["trainer/step"] > skip]
s = p["performance/step_ms"]; dw = p["performance/data_wait_ms"]; h2d = p["performance/host_to_device_ms"]
q = lambda x: {k: round(float(np.percentile(x, k)), 1) for k in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
print(f"steps analyzed {len(p)} (step > {skip})")
print("step_ms ", q(s)); print("data_wait", q(dw)); print("h2d     ", q(h2d))
print("mean step_ms %.1f  p10 %.1f  mean/p10 %.3f" % (s.mean(), np.percentile(s, 10), s.mean() / np.percentile(s, 10)))
print("tokens/s (loss tokens) mean %.0f ; total tok/s = %.0f" % (p["performance/tokens_per_second"].mean(), 524288 * 1000 / s.mean()))
floor = np.percentile(s, 10)
A = (dw < 5) & (h2d > 100); B = dw >= 5
print("regime A (dw<5,h2d>100): %.1f%%  regime B (dw>=5): %.1f%%  slow(>1.05 floor): %.1f%%" % (100 * A.mean(), 100 * B.mean(), 100 * (s > 1.05 * floor).mean()))
qd = p["data/prefetch_queue_depth"] if "data/prefetch_queue_depth" in p else None
if qd is not None: print("queue depth: min %.0f  p10 %.0f  median %.0f  frac==0 %.2f" % (qd.min(), np.percentile(qd, 10), qd.median(), (qd == 0).mean()))
for c in [c for c in df.columns if c.startswith("hosts/")]:
    v = df[c].dropna(); print(f"{c}: last {v.iloc[-1] if len(v) else None}  max {v.max() if len(v) else None}  min {v.min() if len(v) else None}")
for c in [c for c in df.columns if c.startswith("eval/") or c.startswith("diagnostics/")]:
    v = df[c].dropna(); print(f"{c}: {[round(float(x),4) for x in v.values[:6]]}")
for c in sorted(c for c in df.columns if c.startswith("data/") and "share" in c or c in ("data/pad_fraction","data/images_per_sequence","data/row_skip_rate","data/image_slot_utilization")):
    v = df[c].dropna(); print(f"{c}: last {round(float(v.iloc[-1]),4) if len(v) else None}")
att = sorted(c for c in df.columns if c.startswith("attention/"))
if att:
    last = df[att].dropna(how="all").iloc[-1]
    for cyc in range(8):
        print(f"cycle {cyc}: joint {last.get(f'attention/cycle_{cyc}_max_logit', float('nan')):.1f} visual {last.get(f'attention/cycle_{cyc}_max_logit_visual', float('nan')):.1f} text {last.get(f'attention/cycle_{cyc}_max_logit_text', float('nan')):.1f}")
print("summary compiled_memory:", r.summary.get("compiled_memory"))
print("summary mean_tps", r.summary.get("mean_tokens_per_second"), "max", r.summary.get("max_tokens_per_second"))
