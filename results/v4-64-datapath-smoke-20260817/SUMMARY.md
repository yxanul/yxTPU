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
