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

Attention maxima on the mixed batch (per-modality split, first sample):
cycle 0 joint 60.4 = visual 60.4 / text 60.1 - the hot cycle-0 logits are
NOT a visual-position effect (both modalities reach ~60); the
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
yield of longer rows: net -6% loss-token throughput. Memory flat.
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
