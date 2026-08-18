# Data-path smoke, v4-64, 2026-08-17

Branch `feat/datapath-holdout-8k-prereqs` (async numpy device batches,
per-source fetch threads + producer processes, deep prefetch, host
patchify, rows holdout packing, train_fixed diagnostics, per-modality
attention maxima, pre-norm residual probe, calibrate-mix, 8k configs).
Baseline: the 30B continuation's W&B history (`inzcfemi`, 49,220 steps).

## Why: the continuation's host path (from `inzcfemi`)

| | value |
| --- | --- |
| step p10 (compute floor) | 1,499.7 ms |
| step mean | 1,985 ms (mean/p10 1.324; 6.6 of 27.1 h above the floor) |
| regime A: queue non-empty, `_device_batch` ~1.7 s | 60% of steps (step median 1,743) |
| regime B: queue empty, data_wait ~1.8 s | 28% (step median 1,875) |
| both partial | 11% |
| `data/prefetch_queue_depth` == 0 | 40% of steps; == 7 (full) 21% |
| loss-token throughput mean / floor | 234k / 283k tok/s (total 264k / 350k) |

Mechanism (verified with `h2d_probe.py` on the idle slice, all 8
workers, ~1.5 s compute in flight, with and without 31.5/33 GB HBM
ballast): `jnp.asarray(value)` staged the whole process batch on local
device 0 and `host_local_array_to_global_array` re-sliced it ON DEVICE
behind the running step, blocking the host for the full remaining
step - the "prefetched" batch never overlapped compute.

```
current (jnp.asarray -> h2g)   busy: call 1509 ms          idle: ready ~130 ms
numpy -> h2g / mpld            busy: call 1 ms, ready 17 ms (fully overlapped)
patchified numpy               busy: call 1 ms, ready  6 ms
pixel block per device: [4,4,448,448,3] u8 logical 9.63 MB, on-device 11.01 MB
  (channels-major layout, tiling (32,128)(4,1)) - NOT lane-padded; patchified 9.63 MB.
```

## 4k smoke (`vision_1b_smoke_4k`, W&B `y33qadn1`, 400 steps)

Same operating point as the continuation (seq 4096, PDB 4, 524,288
tokens/step, four-source mix), warm-started from its final checkpoint
(step 57,220) at LR 1e-4 constant; 4 producer processes/host (see the
Hub note below), prefetch 48, host patchify, cold streams.

| | continuation | 4k smoke |
| --- | ---: | ---: |
| step p10 / median / p90 (ms) | 1,499.7 / 1,778 / 2,369 | 1,495.5 / 1,497.1 / 1,502.6 |
| step mean (ms), mean/p10 | 1,985, 1.324 | 1,533.6, 1.025 |
| `host_to_device_ms` median | 1,449 | 2.1 |
| `data_wait_ms` p75 / p95 | 1,536 / 2,144 | 0.1 / 0.1 |
| regime A / B share | 60% / 28% | 0% / 0% |
| prefetch queue depth == 0 | 40% | 0% (47/47 all run, fleet min 47) |
| loss-token tok/s mean | 234k | 278.6k (max 290k) |
| total tok/s | 264k | 342k |
| compiled peak / temp | 34.13 / 25.60 GB | 34.12 / 25.60 GB |

The remaining 2.5% mean-vs-p10 is eval/diagnostics-adjacent steps and
a handful of >1.7 s outliers; `hosts/hosts_waiting` was 0 at every
sample. Mix unchanged (vision .363 / climbmix .402 / stack .159 / math
.076, pad .063, images/seq 2.64, skip .216 with max_images_per_row 1).

Holdout at the same checkpoint, first eval (step 100):

| packing | loss | ppl |
| --- | ---: | ---: |
| concat (historical) | 2.789 | 16.27 |
| rows (training contract, row_tokens 1024) | 2.383 | 10.83 |

A 0.41-nat format gap on the same ClimbMix validation split - the
Part II holdout numbers measure an out-of-distribution packing.

Attention maxima on the mixed batch (per-modality split BY QUERY
POSITION - "visual" is the maximum over visual queries against any key,
"text" over text queries; first sample): cycle 0 joint 60.4 = visual
60.4 / text 60.1 - the hot cycle-0 logits are not a visual-query effect
(both modalities reach ~60; a hot visual KEY attended by text queries
would count as text, which the text-only control below rules out); the
diagnostics pass on the cached training batch pins them to cycle-0
heads 10 (59.7), 9 (33.6), 8 (26.7), all other cycle-0 heads 12-17.
Later cycles: text > visual (cycle 5: 35.9 vs 15.7). Same batch under
the historical text-only concat holdout gave cycle 0 at 7-20, so the
trigger is the training packing/content (segment restarts, code/math
rows), not modality.

## Hub API budget (operational)

Opening one streaming dataset costs ~10-15 `api` requests (paginated
repo-tree listing: FineVisionMax 14, ClimbMix 10, Nemotron 5, stack-v3
~12) against the account's 1,000-per-5-minutes quota. 8 hosts x 4
producers x 4 sources = ~1,470 per launch: the first 8k launch, inside
the window of the 4k smoke's opens, died at a producer's first open
(`list_repo_tree` 429 -> RuntimeError -> barrier). Fixed by
retry-with-backoff at open (honoring Retry-After), staggered producer
starts, and `producer_processes: 2` in the smoke profiles (~816 per
launch including the eval iterators).

## 8k smoke (`vision_1b_smoke_8k`, W&B `cbp3ccs9`, 400 steps)

seq 8192 / PDB 2 (same 524,288 tokens/step), 8 image slots per sequence,
up to 4 images per row, `text_row_tokens` 4096 for every source, rows
holdout at 4096-token rows; 2 producer processes/host; warm-started from
the continuation's final checkpoint at LR 1e-4. (Two earlier launches
died before step 1 for reasons unrelated to the shape: the Hub api quota
and the wedged worker 3 - see the operational notes.) The account was
upgraded to HF PRO mid-run; the 429s stopped at startup and never
recurred.

| | 4k smoke | 8k smoke |
| --- | ---: | ---: |
| step p10 / median / p90 (ms) | 1,495.5 / 1,497.1 / 1,502.6 | 1,660.8 / 1,662.2 / 1,665.6 |
| step mean (ms), mean/p10 | 1,533.6, 1.025 | 1,682.5, 1.013 |
| `host_to_device_ms` / `data_wait_ms` median | 2.1 / 0.1 | 1.9 / 0.0 |
| prefetch queue depth (min over hosts) | 47 | 47 |
| compiled peak / temp / code | 34.12 / 25.60 GB / 505 MB | 34.11 / 25.59 GB / 527 MB |
| loss tokens per step | ~420k | ~441k |
| loss-token tok/s mean | 278.6k | 261.3k |
| total tok/s | 342k | 312k |
| pad fraction / images per seq / slot util | .063 / 2.64 / .66 | .045 / 4.91 / .61 |
| row skip rate | .216 (1 image/row) | .191 (<= 4 images/row) |
| loss-token shares vision/climbmix/stack/math | .363/.402/.159/.076 | .386/.355/.196/.062 (weights NOT recalibrated) |

Device time +11.0% for 2x context (the GQA layers' O(T^2) term costs
more on v4 than its ~5% FLOP share suggested; matches the earlier +9%
device-only smoke), partly bought back by the +5% supervised-token
yield of longer rows: net -6% loss-token throughput. Memory flat
between the two smokes at 16 image slots per device (8 slots x PDB 2
here, 4 x PDB 4 at 4k). The earlier device-only 8k memory smoke (4
slots x PDB 2 = 8 images/device, four-source mix, 25 steps) measured
31.0 GB peak / 22.5 GB temporaries at 1,643 ms/step: the ~3 GB between
the two 8k points is the second 8 images per device (the ViT's
materialised `[16, 6, 784, 784]` scores and its activations), not the
sequence length.
The mix moved with the longer rows exactly as the config comment warns
(stack rows are now file-length, math rows longer): run
`yx-pretrain calibrate-mix` before any campaign at this shape.

Holdout at the same checkpoint: rows (4096-token rows) 2.377 -> 2.374
over the run; concat at 8192-token windows 2.489 -> 2.494 (a different
window length than the 4k concat number, so only the rows numbers are
comparable across the two smokes: 2.383 at 1024-token rows vs 2.377 at
4096-token rows).

Telemetry on the mixed 8k batch: cycle-0 maxima joint 67.1 = visual
67.1 / text 65.6 (again both modalities); pre-norm residual RMS visual
6.84 vs text 5.81 (the post-norm probe would report 8.5 / 8.5).

## Operations record (2026-08-17)

- Two crashed launches left spawned producers orphaned on every worker
  (daemon=True does not survive SIGABRT); fixed with PR_SET_PDEATHSIG +
  a getppid watchdog, verified: 0 leftover processes after the 8k run.
- healthagent (docker, --memory=512m, OOM-kill disabled) crossed its cap
  after 26 days up on all 8 workers at 16:00 UTC; the kernel logged
  "Out of memory and no killable processes" ~2x/s into kern.log+syslog
  (~90 MB/min per host, disks at 90%). `systemctl restart
  healthagent.service` fixed it (RSS 504 -> 38 MB); logs truncated.
- Worker 3 (primary/coordinator) became unreachable at ~16:40 (TCP
  accepted, no SSH banner; from ~17:20 no ping) and stayed down 1h40m
  while the node reported READY/HEALTHY with "maintenance event at
  16:54:45Z". `tpu-vm stop` is not supported on pod slices;
  `queued-resources reset` (18:22 UTC) REBOOTED all 8 workers in ~4
  minutes with disks intact (repo, venvs, checkpoints; the tmpfs
  /mnt/ram is lost) - it is a reboot, not a re-provision, at least when
  the node stays READY.
- Hub quotas: free tier 1,000 api / 5,000 resolvers per 5 min; each
  streaming open costs 10-15 api requests; 16 producers filling shuffle
  buffers also burst the resolvers bucket. PRO (2,500 / 12,000) removed
  the 429s mid-run.
- `pretraining/scripts/fleet.sh`: parallel per-worker gcloud ssh with
  keepalives and a hard cap; status/run/launch/tail/procs/kill-orphans/
  health.

## 8k profile (steps 24-27 of a 36-step run, primary host, `profile_8k_step.json`)

Traced steps ran at 1,713 ms (steady state 1,662; ~3% profiler overhead).
Device duty cycle 96.2% (65 ms/step idle inside the module: dispatch
edges and kernel bubbles); host path invisible (data_wait 0, h2d 3 ms).
xprof MXU utilization 21.8% (it cannot count Pallas FLOPs). Device
self-time by component (`benchmarks/summarize_xplane_step.py`):

| component | % device | ms/step |
| --- | ---: | ---: |
| dense GEMMs: bwd + others | 19.7 | 324 |
| dense GEMMs: fwd (checkpointed) | 16.5 | 272 |
| dense GEMMs: rematted fwd recompute (MLP, `save_dot_except_mlp`) | 7.9 | 131 |
| KDA fused kernels (fwd inverse x6 = 3 fwd + 3 recompute; stage A x3; stage B x3) | 17.3 | 285 |
| KDA XLA glue: depthwise conv (HBM-bound, 141 GFLOP/s) | 6.0 | 98 |
| KDA/other glue: casts (`[2,8192,4608]` bf16 <-> f32) | 6.4 | 106 |
| pads / reshapes / copies (`[2,8,1025,4608]` chunk pads, `[2,8192,12,3,128]` reshapes) | 4.2 | 70 |
| GQA splash attention (fwd 2x, dkv) at 88-106 TFLOP/s | 8.4 | 138 |
| elementwise (norms, gates, softmax, adds) | 4.8 | 79 |
| collectives (gradient all-reduce, synchronous on v4) | 3.7 | 60 |
| loss head GEMMs (chunked CE) | 2.6 | 43 |
| norm einsums, ViT (~1.5% inside the GEMM rows), misc | ~2.5 | ~40 |

Reading: 44% is dense GEMMs (of which 8% is remat recompute the memory
budget forces), 17% is the KDA kernels themselves, and **~16-17% is
XLA glue around the pre-fold v4 KDA kernel** (depthwise conv 6.0%,
casts 6.4%, pads/reshapes 4.2%). At the time of the profile this read as
the largest single lever left on the device (a folded kernel worth
~150-250 ms/step, 9-15%); SUPERSEDED by the measurements below: the
cheap `shifted` conv captured -2.8% and the fold itself is perf-neutral
on v4 (stage A's register pressure), so the glue is not recoverable
without a lower-pressure stage A. Splash is 8.4% at 8k (4.7% of the FLOPs at 33-39% of peak);
collectives 3.7% cannot overlap on v4; the KDA forward recompute (4%)
is the price of `remat_save_kda_residuals: false` at 99.3% HBM.

MFU (model FLOPs, backbone 916.5M params x 6 x 524,288 tokens + tied
logits head over all positions + causal-halved attention + KDA chunk
math + ViT over 16 image slots/device; v4 peak 275 TFLOP/s x 32):

| | 4k | 8k |
| --- | ---: | ---: |
| model FLOPs per step | 3.39 PFLOP | 3.55 PFLOP |
| of which attention | 4.7% | 8.9% |
| step | 1,497 ms | 1,662 ms |
| **MFU (model FLOPs)** | **25.7%** | **24.2%** |
| parameter-only 6N x tokens (the report's convention) | 24.5% | 22.1% |
| xprof MXU utilization (Pallas FLOPs uncounted) | - | 21.8% |

### Second pass: optimizer, embedding, ViT, and the conv weight gradient

The first bucketing keyed on einsum specs and kernel names and hid the
optimizer inside "dense GEMMs: bwd/other". Splitting the step into
inside-the-cycle-scan (83.6%) vs outside (16.4%, 271 ms) exposes it:

| component | % device | ms/step |
| --- | ---: | ---: |
| KDA fused kernels (pallas) | 17.3 | 285 |
| dense GEMMs: fwd (checkpointed) | 16.5 | 272 |
| dense GEMMs: bwd | 9.6 | 158 |
| **optimizer: Muon Newton-Schulz matmuls** (vmapped over the 8 stacked cycles, fp32, HBM-bound at ~470 GB/s; replicated on all 32 chips) | **9.0** | **148** |
| GQA splash attention | 8.4 | 138 |
| dense GEMMs: rematted MLP fwd recompute | 7.9 | 131 |
| casts bf16<->f32 around the KDA kernel (`[2,8192,4608]`, `[2,8192,12,3,128]`, ~460 GB/s) | 6.4 | 106 |
| pads / reshapes (chunk pads `[2,8,1025,4608]`, `[2,8200,4608]`, qkv reshapes) | 4.2 | 70 |
| elementwise (norms, gates, softmax, adds) | 3.9 | 64 |
| **KDA depthwise conv: weight gradient** (`bf16[4,1,4608]` output from a B*T reduction at **66 GB/s**) | **3.1** | **52** |
| collectives: gradient all-reduce inside the scan (37-39 GB/s, synchronous) | 3.0 | 49 |
| KDA depthwise conv fwd + recompute + data grad (146 GB/s) | 2.8 | 47 |
| loss head GEMMs (chunked CE) | 2.6 | 43 |
| ViT attention einsums (scores materialized `[16,6,784,784]`) | 1.1 | 19 |
| RMSNorm scale einsums | 1.1 | 18 |
| optimizer: Muon norms / vmapped elementwise | 0.9 | 15 |
| collectives: tied-embedding / loss-head all-reduce | 0.7 | 11 |
| embedding gather / scatter-add | 0.2 | 4 |
| other | ~1.3 | ~21 |
| (device idle inside the step: 3.8%) | | 65 |

Optimizer total ~10% (Muon NS + norms + updates), all outside the scan.
NS: ~45 TFLOP per step of fp32 matmuls, executed on EVERY chip
(replicated), reading/writing fp32 `[8,1536,4096]`-class tensors at
HBM speed - it is bandwidth-bound, not MXU-bound, on v4.

Levers, ranked by (ms saved / effort):
1. Depthwise-conv weight gradient: 52 ms in a 66 GB/s XLA reduction -
   rewrite as an einsum over the 4 shifted views (pure XLA, in
   kimi_delta_attention.py); expect ~45 ms (-2.7%).
2. Muon: `muon_ns_bf16` (exists, gate-validated at 337M within 0.016
   nats, perf-neutral there because NS was 16 ms) halves NS traffic -
   expect 50-70 ms (-3-4%); `muon_distributed_ns` (perf-REJECTED at 337M
   because the all-gather exceeded 16 ms of NS) now trades ~140 ms of
   replicated NS for a 1.8-3.6 GB update all-gather - expect a net
   -50..-100 ms; both need a 1B A/B, the 337M verdicts do not transfer.
3. KDA glue fold (conv fwd/bwd, casts, pads into the v4 kernel, bf16
   dQKV epilogue): up to ~200 ms (-12%) - kernel project. MEASURED
   (next section): built and correct, -0.3% end-to-end; the expected
   saving does not materialise on v4 because stage A cannot absorb the
   recompute without spilling. Superseded.
4. Remat recompute (MLP 131 ms + KDA fwd 65 ms) and synchronous
   collectives (60 ms) are memory- and compiler-bound respectively; no
   cheap move at 99.3% HBM.

## Device levers A/B (8k, 40 steps each, p10 step; baseline 1,660.8 ms)

| lever | p10 (ms) | delta | verdict |
| --- | ---: | ---: | --- |
| `kda.conv_impl=shifted` (conv as shifted multiply-adds in XLA) | 1,613.5 | **-2.9%** | ADOPTED after the 200-step loss overlay (`vision-1b-conv-overlay`: -2.8%, the figure quoted for the adoption) |
| `optimizer.muon_ns_bf16` (vs a matched from-scratch baseline 1,663.7) | 1,636.6 | -1.6% | needs a 1B numerics gate; the flag changes the optimizer pytree (no warm-start from existing checkpoints) |
| `optimizer.muon_distributed_ns` | 1,688.6 | +1.7% | rejected at 1B too - the update all-gather costs more than the replicated NS |
| `kda.conv_impl=fused` (conv + SiLU folded into the v4 kernel, stage A at 4 streams/program - the flag's default since 2026-08-18) | 1,655.2 | -0.3% | correct, flag-gated; does not pay (below) |

### The conv + SiLU fold on v4 (`pallas_kda_fused_v4_conv`)

Implemented and gated on device (`benchmarks/verify_kda_v4_conv_fold.py`,
production per-device shape B=2, T=8192, H=12): forward, final state,
d_log_decay, d_beta, d_state BITWISE identical to the XLA path; raw-input
and conv-weight cotangents at rel L2 3e-3 (one bf16 ulp; the XLA path
rounds its conv-transpose and dW to bf16, the fold keeps fp32). Two
plumbing bugs were found by the gate (an edit that had landed on the
identical-looking integrated backward kernel; the SiLU' scale missing at
stage B's final writes) - the gate's bitwise columns made both obvious.

Why it does not pay: stage A is the largest v4 kernel and at the
production 8 streams/program it already spills ~5.3 MB of registers;
the fold's recompute pushes it 168 KB over v4's 16 MB VMEM. The three
ways to fit each cost more than the fold saves (mixer-core fwd+bwd
microbenchmark, reference = XLA conv + kernel 18.10 ms): stage A at 4
streams/program 16.48 ms (-9%, but end-to-end -0.3%; this is the flag's
default configuration); stage A at 8 with all halo/weight windows
single-buffered 19.38; single-buffered only in stage A 18.34 (what
`YXTPU_KDA_FOLD_STAGE_A_STREAMS=8` selects). The cheap `shifted` XLA form captures most of what is
available; the fold would need a stage A rewrite that lowers its
register pressure (or v5e/v6e, where the folded kernel already runs).

### `conv_impl=shifted` adoption gate (W&B group `vision-1b-conv-overlay`)

Two 200-step runs at 8k from the continuation checkpoint, LR 6e-4,
deterministic single-producer stream (identical batches): loss
shifted - xla within +-2.6e-3 (mean -2.4e-4, last-50 mean -3.8e-4);
grad norms track; p10 step 1,666.0 (xla) vs 1,619.6 (shifted), -2.8%.
Adopted as the 1B model config default (`kda_hybrid_1b_yx49k.yml`).

## Review gates 2026-08-18 (W&B group `vision-1b-review-gate`)

After the branch review's fixes (see the AGENTS.md entry "Review fixes
2026-08-18"), five 200-step fleet runs at the 8k operating point, all
warm-started from the continuation checkpoint, LR 6e-4 (warmup 50,
cosine to 0.1 over 200), deterministic single-producer stream unless
noted:

| gate | what | result |
| --- | --- | --- |
| fold gate (w0 carve-out) | `verify_kda_v4_conv_fold.py`, production shape, stage A at 4 streams (new default) | PASSED: fwd/state/decay/beta/state bitwise, raw grads 3.2e-3, dW 2.8e-3; fold 16.49 vs 18.09 ms reference |
| A (`o3kav3yf`) | fixed code vs the pre-fix `ov-shifted` run (`nday93hv`) | loss AND grad norm bitwise identical on all 52 steps where the two LR schedules coincide (the reference used final_learning_rate_fraction 1.0; the divergence at step 53 is the schedule, not the code); p10 1,619.6 ms |
| A2 | rerun of A | 200/200 steps bitwise identical to A - the pipeline is run-to-run reproducible |
| B (`v30yqw2f`) | production-like: 2 producer processes, prefetch 48, eval concat+rows every 100, train_fixed diagnostics, LR 1e-4 | p10 1,616.3 / median 1,620.7 / mean 1,656 ms; data_wait 0.05 ms; rows 2.378 vs concat 2.487 (as before); diagnostics carry `batch=train_fixed`; residual_* keys; 0 orphans |
| C2 (`6mvelcvl`) | `optimizer.muon_ns_bf16=true`, warm-started through the new weights-only restore (the full restore failed on the optimizer pytree - that was gate C) | p10 **1,593.4 ms (-1.7%)**; loss overlay vs A2 max abs d 3.6e-3, mean -3.9e-5, final 1.31932 vs 1.31921 (the conv adoption gate was +-2.6e-3) - the pending 1B numerics gate; adopt for the next campaign |
| D | A's recipe with gc.freeze() after compile + GC telemetry | 200/200 bitwise identical to A2; **0 stalls** (max step 1,690 ms vs 3,738 / 4,115 in A / A2), mean 1,625.9 vs 1,659.9 ms (mean/p10 1.004) |

The stall finding: A and A2 had ~1.5 s stalls at nearby steps (78/83/85
and 78/83/85/88; C2 at 75/79/80/82) with the primary's data_wait 0.2 ms,
h2d 6 ms and every collected host's prefetch queue full (new per-host
`host_metrics.<process>.jsonl`). The mechanism is Python's generation-2
garbage collection walking the trainer's large heap (streams, shuffle
and fetch buffers, compiled programs): after `gc.freeze()` the same
collections cost 70-75 ms (steps 167 and 194 of D). Any host's
collection stalls every chip, so on the 30B continuation this was part
of the "other host" excess. Net at 8k after this pass: p10 1,620 ms
(shifted conv), mean 1,626 (no GC tail), 1,593 with bf16 NS pending
adoption.
