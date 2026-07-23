# TPU Research Cloud Workspace

This workspace is for JAX, Flax NNX, Pallas, XLA, and MaxText experiments on
the user's TPU Research Cloud allocation.

## TPU allocation

Source: the TPU Research Cloud approval email. The offer is free for 30 days
and applies only to newly created Cloud TPUs in the exact zones below.

| TPU | Provisioning | Chip quota | Zone |
| --- | --- | ---: | --- |
| v5e | Spot | 64 | `europe-west4-b` |
| v5e | Spot | 64 | `us-central1-a` |
| v6e | Spot | 64 | `us-east1-d` |
| v4 | Spot | 32 | `us-central2-b` |
| v4 | On-demand | 32 | `us-central2-b` |
| v6e | Spot | 64 | `europe-west4-a` |

The approval email did not provide an exact expiry timestamp. The 30-day
window is counted from the confirmed start date below:

- TRC start: 2026-07-20
- TRC expiry: 2026-08-19 (30 days from start; exact timestamp unconfirmed)

### v5e quota distinction

- The 64-chip v5e Spot grants map to the Cloud TPU API's **training** quota.
- Cloud TPU classifies `v5litepod-1`, `v5litepod-4`, and `v5litepod-8` as
  serving slices. They consume a separate serving quota that the TRC grant did
  not increase; this project currently has only 4 Spot serving chips per zone.
- The smallest v5e slice that consumes the granted training quota is
  `v5litepod-16`. It is a multi-host slice.
- A `v5litepod-8` Spot request in `europe-west4-b` was rejected on 2026-07-20
  with `RESOURCE_EXHAUSTED` against the 4-chip serving quota. No resource was
  created.

## Current training TPU

State is dynamic; verify it before relying on this section.

- TPU node: `yxtpu-v4-64-train`
- Queued resource: `yxtpu-v4-64-train-qr`
- Zone: `us-central2-b`
- Accelerator: `v4-64` = 32 chips (v4 slice names count TensorCores, 2 per
  chip, and megacore presents 1 JAX device per chip). Multi-host: 8 workers,
  4 chips each, topology 2x4x4.
- Provisioning: On-demand — explicitly requested by the user on 2026-07-22;
  this is the grant's only on-demand row.
- Runtime: `tpu-ubuntu2204-base`
- Created: 2026-07-22
- Last verified: 2026-07-22 — queued resource `ACTIVE`; node `READY` and
  `HEALTHY`. `guaranteed: {}` (the on-demand tier) confirmed on the resource.
  On-demand provisioned on the first attempt, unlike the four reclaimed v6e
  Spot attempts the same day.
- Benchmarks 2026-07-22 (synthetic, selected profile, pure data parallel
  over 32 chips): MaxText 247M GQA baseline ~2.26M tokens/s global
  (~105 TFLOP/s/device, ~38% MFU). KDA 273M with the fully fused v4 kernel
  (fused forward + split fused backward, commit 3cfc1a9): ~1.54M tokens/s
  global at 8/device x 2048 - about 32% below the GQA baseline, matching
  the 20-40% gap measured on v6e. 16/device x 2048 (~1.53M) and
  8/device x 4096 (~1.55M) also fit; 16 x 4096 still OOMs by ~0.7 GiB.
  The earlier forward-only hybrid measured ~630k tokens/s and OOMed at
  batch 16 from its XLA-tape backward; it remains only as a fallback.
- BlockAttnRes perf pass 2026-07-22 (commits 41d7aeb..878bb28): split-
  scoring, RMSNorm-fold into the pseudo-query, and bf16 dot operands cut the
  depth-read overhead from ~17% to ~4% (steady step ~484 ms vs 464 baseline)
  with losses overlaying to ~1e-4. optimizer.muon_ns_bf16 (bf16 momentum +
  Newton-Schulz via masked post-clip cast + mu_dtype) is implemented and
  gate-validated - 200-step trajectory within 0.016 nats of the fp32
  reference, grad norms unchanged - but measured perf-neutral on v4 at 337M,
  so it stays default-false; revisit at larger widths where NS FLOPs grow
  quadratically. Toggling it changes the optimizer-state pytree (checkpoint
  incompatibility across the flag).
- KDA backward-bandwidth pass 2026-07-23 (commits f657066..28c7242): the
  cycle remat was re-running the fused KDA forward inside every backward.
  The fwd rule now checkpoint_names its output and state history, and
  model.remat_save_kda_residuals (default true) adds both names to the
  policy; the recompute then survives only as a dead zero-output shard_map
  husk in the jaxpr, which XLA's HLO DCE deletes. Compiled HLO shows fwd
  custom_calls 6 -> 3 per scan body (static count - the executed saving is
  12 walks/step at 4 cycles), XLA temp memory 14.2 -> 16.4 GB, and the
  compute-floor step time (p10 of a 200-step muonclip+attnres A/B) fell
  483 -> 473 ms (~2%). Disable the flag for memory-tight long-sequence runs
  (saved state history scales linearly with sequence length). Two kernel
  cuts landed alongside, both verified bit-identical on-device across all
  nine fwd/bwd outputs: stage A's state-cotangent export shrank from a full
  per-chunk buffer to one revisited chunk-0 slot (134 MB -> 4 MB of writes
  per layer), and stage B recomputes cumulative_decay from log_decay
  in-kernel instead of round-tripping a 67 MB export; both sit below
  run-to-run timing noise, as estimated. Loss trajectories across
  baseline/flag-off/flag-on overlay at fp-noise (<=2e-5 at step 50,
  ~4e-3 transient wobble at the mid-run grad spike, no drift). Note:
  tests/test_kimi_delta_attention.py has 5 pre-existing failures when run
  on v4 hardware (sublane-gather in the non-v4 kernels; identical at
  4de5b4a) - only its v4-path tests are meaningful gates there.
- Batch prefetch 2026-07-23 (commit d8aece3): the loop now stages batch i+1
  between dispatching step i and blocking on it, hiding the ~70 ms/step
  host-to-device path (data_wait is ~0.3 ms; the cost is global-array
  assembly in _device_batch, not the fetch). Validated: losses identical to
  the synchronous run digit-for-digit, median wall throughput 758k -> 1.01M
  tokens/s on the 200-step A/B, fastest wall steps now equal the 473 ms
  compute floor. Auto-disabled when checkpointing is enabled (a save would
  persist iterator state one batch ahead and skip a batch on resume).
- Profile 2026-07-23 (steps 80-84, kda_hybrid_128k+muonclip+attnres,
  prefetch on; xplane parsed directly - the tensorboard_plugin_profile
  converter is incompatible with the venv TF, and beware name-matching op
  categories: instruction strings embed operand names). Device busy
  454.6 ms/step vs step_ms ~473 (~20 ms dispatch/infeed edges). Breakdown:
  dense body GEMMs ~220 ms (48%, ~37% MFU overall), 128k vocab head ~60 ms
  (13%, near GEMM roofline; the unfused standard loss materializes
  [8,2048,128256] twice - the fused implementation would shave the
  elementwise/materialization slice, est. 10-20 ms), KDA kernels ~70 ms
  (15%: stage_a 28.6, fwd 21.7, stage_b 19.8), gradient all-reduces
  ~27.5 ms (6%, partly inside the backward cycle scan - async-collective /
  latency-hiding-scheduler flags untested), AttnRes reads ~15 ms (3.3%),
  splash attention ~12 ms (2.6%). Remaining wall tail is episodic >1 s
  step spikes (present in all runs; not data_wait - suspect W&B flush or
  stream shard boundaries) - a 2-3-batch prefetch queue would mask them.
- SuperBPE-50B run COMPLETE 2026-07-23 (W&B group superbpe-50b, two runs:
  02:04 crashed@59,446 by the zombie-launcher incident + 10:55 resumed from
  the 30B checkpoint to completion): kda_hybrid_128k + BlockAttnRes +
  muonclip, 95,500 steps = 50.07B step-tokens (~30B unique + ~1.7 epochs
  over the stream head post-resume), constant LR 3e-4 with cosine anneal
  over the final 9,537 steps to 0. Final: train loss ~2.60, holdout 2.593
  (from 3.882 holdout at 1B - the anneal alone took 2.735 -> 2.58).
  Final 0-shot lm-eval @50B (full sets): lambada 0.367/ppl 31.7 (0.164 at
  1B), hellaswag acc_norm 0.474 (rose twelve straight rounds), arc_easy
  0.642, arc_challenge acc_norm 0.336, piqa acc_norm 0.717, sciq 0.878,
  boolq 0.591, openbookqa acc_norm 0.348, copa 0.71 - ahead of
  Pythia-410M@300B on everything except lambada on 6x fewer tokens.
  Mean throughput 1.098M tok/s (max 1.123M) on the resumed leg. SIX
  checkpoints per host under /home/a1111/yxtpu_ckpts/ (steps 19073..95500,
  17 GB/host) - THEY EXIST ONLY ON THE TPU HOST DISKS; export before any
  slice teardown.
- SFT stage bootstrapped 2026-07-23/24 (commits 8b0ac80..7a4e90b, module
  pretraining/src/yxtpu_pretrain/sft/): 12 chat specials on the free padded
  ids 128001-128012 (K2.5-style roles + im_middle/im_end + single-token
  <think>/</think> + five tool-call markers, exact-id asserted), jinja
  template, assistant-only loss masking (render mask shifted with labels),
  batched pre-tokenization (single fast-tokenizer call; the per-row loop
  was ~2 h for 100k rows, batched is ~1 min). First run: 100k rows x 2
  epochs of Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned General-Distillation,
  init weights-only from ckpt 95500 (fresh muonclip, new rows mean-init),
  LR 5e-5 constant + 290-step anneal: loss 2.40 -> 1.645 over 1,392 steps
  / 703M tokens in ~15 min. Checkpoints are process-0 pickles
  (yxtpu_sft_ckpts on worker 3 + ckpt/sft/1392 local) - orbax's saver
  breaks when two managers live in one process (signaling client asserts
  single-controller; unique barrier prefixes did not cure it). Decode:
  template-native reasoning confirmed - every answer runs
  <think>-plan-</think>-markdown-answer-<|im_end|>; substance is honest
  337M (17+25 boxed correct, tomato correct, facts wobble at temp 0.6;
  greedy is more factual). Decode-path lesson: the eager loop shipped the
  full [4,2048,128k] fp32 logits to host EVERY token (~4 GB/token);
  slicing on device made it ~100x faster. Proper KDA O(1)-state
  incremental decode remains the SFT-stage build, alongside ifeval
  (generate_until adapter) and the full 368k rows + math/STEM subsets.
- Extended 0-shot eval of ckpt 95500 on a standalone worker 2026-07-23
  (122,886 requests, ~40 min on 4 chips; script: restore via CheckpointIO +
  JaxHarnessLM outside training): mmlu 0.267 (chance-level, normal at 337M),
  winogrande 0.523, race 0.334, truthfulqa_mc1 0.231 / mc2 0.389. Anchors
  reproduced the in-training round exactly (arc_challenge acc_norm 0.3362
  identical; lambada 0.366 vs 0.367; hellaswag 0.473 vs 0.474) - the
  single-worker eval path is faithful. Tokenizer-lambada audit: SuperBPE
  encode(ctx) is a prefix of encode(ctx+cont) on 5153/5153 examples and the
  scored span decodes exactly to " "+target - lm-eval's _encode_pair
  assumptions hold; lambada accuracy is trustworthy (perplexity remains
  cross-tokenizer-incomparable). Generation sanity from the local ckpt copy
  (CPU, chunk_kda path, kda.precision=full_fp32 override; the checkpoint
  carries a TPU-only attention_op.rngs subtree - placeholders uint32[4] and
  uint32[4,2], then drop): answers "capital of France" -> Paris, water
  boils -> 100 C, gold -> Au, and follows Q:/A: format zero-shot. Harness
  constraints for the SFT stage: ifeval needs a generate_until adapter
  (ours is loglikelihood-only), ifbench absent from lm-eval 0.4.12, and
  social_iqa/logiqa/wsc273 are script-based datasets the modern datasets
  library refuses.
- Per-host local checkpointing hardened 2026-07-23 (commits cb2ce06..f065265,
  preflight-validated save->resume->continue on all 8 hosts): orbax on
  non-shared disks needs FOUR things together - (1)
  MultiprocessingOptions(primary_host=None, active_processes={self},
  per-host barrier prefix): every cross-host wait removed (plain
  primary_host=None deadlocks mid-write waiting for other hosts' array
  metadata in a directory it can never see); (2) the same options passed
  into EVERY item handler, not just the manager (value-leaf/_METADATA
  writes gate on the global primary otherwise); (3) iterator state saved
  as JSON (Standard routes scalar leaves through type handlers whose
  primary gating ignores handler options entirely); (4) a jax.Array type
  handler override with no ArrayMetadata store. Diagnostic lesson: a
  multi-host run where seven processes "exit silently" and one blocks in a
  collective usually means the seven died at a per-host branch (here:
  restore) - the one-line CLI error is the FIRST stderr line, and stdout
  buffering hides everything else unless PYTHONUNBUFFERED=1. Streaming
  runs checkpoint weights+optimizer only (stream position is not
  serializable; checkpoint.allow_weights_only_resume=true opts in, restore
  restarts the stream). Batch prefetch skips save iterations so persisted
  iterator state never runs ahead.
- Loss-head and collective levers probed 2026-07-23 (v4, production shape):
  (1) loss.implementation=tokamax_fused is HARD-BLOCKED on v4 - tokamax's
  mosaic_tpu linear-CE kernel raises NotImplementedError "Not supported on
  TPU v4" with and without the scoped-VMEM flag; porting it would be a
  KDA-fork-scale kernel project. The compile probe also priced the standard
  loss's cost: 20.4 GB temp vs 16.4 GB for a logits-free step (~4 GB =
  the two [8,2048,128256] materializations, the 10-20 ms/step a fused head
  would save on v6e). (2) The v5e collective-overlap flag set
  (data_parallel_all_reduce_opt + different_sized_ops + async collective
  fusion trio + overlap_compute_collective_tc) compiles on v4, is
  numerics-neutral (loss digits identical), and is PERFORMANCE-NEUTRAL:
  step-time floor 473.1 -> 472.4 ms p10 on the 200-step A/B. The profile's
  ~27.5 ms/step of all-reduce stays synchronous on v4's compiler path, so
  the flags were not adopted into v4-64.yml. Remaining ranked levers from
  the profile: per-device batch 16 (fits per the memory survey), deeper
  prefetch queue against the >1 s step spikes, mixer_only AttnRes ablation.
- BlockAttnRes A/B 2026-07-22 (arXiv:2603.15031, commit 7028526; same
  kda_hybrid_128k + muonclip protocol as run 3): PASSES both gates - final
  loss 3.796 vs 3.872, holdout 3.850 vs 3.882. lambada, the hybrid's one
  campaign loss, jumped 0.112 -> 0.164 acc (ppl 7,161 -> 2,247), beating
  even the pure-GQA transformer's 0.127. sciq 0.58, hellaswag and
  arc_challenge also up; piqa/boolq slightly down. Cost: ~17% throughput
  (0.94M vs 1.13M tokens/s) - the depth reads are bandwidth-bound, and a
  mixer_only site ablation is the obvious next lever. Grad norms slightly
  busier (mean 0.51 vs 0.39, max 5.67) but bounded and smooth.
- SuperBPE-1B campaign 2026-07-22 (W&B group superbpe-1b; ClimbMix streamed
  through alisawuffles/superbpe-tokenizer-128k, vocab padded 128256, tied
  embeddings, 1B tokens, constant LR 3e-4 after 40-step warmup, full lm-eval
  every 477 steps): gqa_rope_128k+adamw final loss 4.011 / holdout 4.07;
  gqa_rope_128k+muonclip 3.872 / 3.929; kda_hybrid_128k+muonclip 3.824 /
  3.882 (best, and the calmest gradients: mean 0.39, max 2.03). Two bugs
  were found and fixed en route: benchmark-sized LR schedules decaying to
  the floor by step 30, and optax Muon defaulting to width-transfer scaling
  (fixed with consistent_rms=0.2, commit 5f134c7) - the un-fixed muonclip
  run lost a full nat of loss and its grad norms rose all run. Multi-host
  lm-eval works as of ecbc7b8; each full ten-task round costs ~2 minutes.
- Tied embeddings (model.logits_via_embedding, commit 6472af7): now actually
  implemented (the flag was schema-only before). On v4-64 at 8x4096 the tied
  273m model runs ~1.554M tokens/s (unchanged, as expected) with
  parameter_count 239,381,088 - exactly vocab x emb fewer. For the gpt2
  model the same flag saves 51.6M parameters (~17%).
- Input-projection fusion (kda.fused_in_proj, commit c841282): merges the
  four input-side KDA projections into one [embed, 3336] GEMM. Proven exactly
  equivalent (transplant test, 2e-5) and guarded against muon-family
  optimizers pending blocked Newton-Schulz routing. A/B on v4-64 measured it
  performance-NEUTRAL (-0.8% at 8x2048, +0.7% at 8x4096): XLA already
  overlaps the sliver GEMMs behind the qkv GEMM, and the fused path pays a
  materialize-and-slice cost instead. Keep it off by default; re-evaluate on
  v6e where the wider MXU makes slivers relatively costlier.
- KDA on v4 runs through `pallas_kda_fused_v4`: the pre-fold fused Pallas
  forward plus the split fused backward (stage A reverse solve, stage B
  parallel pairwise epilogue) in `kernels/kda_fused_pallas_v4.py`;
  `kda_v4_hybrid` (fused forward + chunkwise XLA backward) remains only as
  an unused fallback. The integrated one-kernel fused
  backward cannot compile on v4 - Mosaic's layout assignment needs a
  sublane-gather relayout the v4 ISA lacks (every construct compiles in
  isolation; only the integrated backward fails). The folded kernel remains
  the v5+/v6 path; dispatch is automatic by device generation.
- v4 pod-slice operations, learned the hard way:
  - libtpu initialization on a pod slice is collective: a JAX process on one
    worker blocks in `make_tpu_client` forever unless all workers launch
    together. Single-worker debugging needs `TPU_PROCESS_BOUNDS`-style env
    restrictions; otherwise always use `--worker=all`.
  - A crashed run leaves `/tmp/libtpu_lockfile` behind; the next client
    blocks or fails until it is removed on every worker.
  - The primary logging process (jax process_index 0) is NOT necessarily
    worker 0 - it landed on worker 3 here. Filter run logs by content, not
    hostname.
  - Never `pkill -f <pattern>` over `gcloud ssh --command` when the pattern
    also appears in the command line itself - the shell kills its own
    session and gcloud retries in a loop.
  - Never leave a `--worker=all` LAUNCH command in a backgrounded gcloud
    ssh: its connection retry loop can fire the whole command again hours
    later (observed 2026-07-23: an 8-hour-old launcher re-ran the 50B
    launch mid-run, killing the live training; checkpoint resume:true
    recovered it from the 10B-interval save with ~2.2k steps lost). Launch
    sessions must be foreground with a short timeout, verified terminated,
    or use -o ConnectTimeout so a dead worker fails fast instead of
    retrying forever.
- Setup completed 2026-07-22 on all 8 workers: repo at `main` (`6aeaf4a`,
  full clone — doctor's pin check needs git history, so never clone with
  `--depth 1`), `uv sync --locked --extra dev`, HF token in
  `~/.cache/huggingface/token` and W&B key in `~/.netrc` (both verified by
  live authentication; non-interactive SSH needs `~/.local/bin` added to
  PATH for `uv`). `doctor --hardware v4-64`: all 8 workers report 32 TPU
  devices and matching device count. The `maxtext_pin` check reports
  `clean=False` on workers and locally alike — the vendored `maxtext/` tree
  was intentionally patched after the import commit by the KDA training-path
  commits — so treat that specific FAIL as the known baseline, not a setup
  regression.
- Quota verified 2026-07-22 via the Service Usage API: "TPU-V4 pod cores in
  use" has no default quota anywhere and exactly one zone override,
  `us-central2-b` = 64 cores = 32 chips — the TRC grant fingerprint. The
  request consumes the full grant.
- v6e history: on 2026-07-22 four v6e-32 Spot attempts (three in
  `europe-west4-a`, one in `us-east1-d`) were all reclaimed mid-provisioning
  (`PROVISIONING` -> `SUSPENDING` -> `FAILED`, internal error code 13;
  `reset` only works from `ACTIVE`, so each retry required delete +
  recreate). The user then chose the on-demand v4 grant instead. Both v6e
  zones were verified clean of nodes and queued resources. The v6e Spot
  quota (64 chips per zone in `europe-west4-a` and `us-east1-d`) remains
  granted and unused; on-demand v6e is not part of the grant and must never
  be provisioned.
- The earlier `yxtpu-v6e8-dev` v6e-8 slice was Spot-preempted on 2026-07-21;
  its queued resource was deleted on 2026-07-22.

Check status:

```bash
gcloud compute tpus queued-resources describe yxtpu-v4-64-train-qr \
  --zone=us-central2-b

gcloud compute tpus tpu-vm describe yxtpu-v4-64-train \
  --zone=us-central2-b
```

Connect (multi-host slice: plain ssh lands on worker 0; use `--worker=all
--command="..."` to run on every host):

```bash
gcloud compute tpus tpu-vm ssh yxtpu-v4-64-train \
  --zone=us-central2-b
```

Recreate (on-demand v4 requires the user's explicit request in the current
conversation — never recreate this resource unprompted; for Spot rows,
`--spot` is mandatory because the command defaults to on-demand):

```bash
gcloud compute tpus queued-resources create yxtpu-v4-64-train-qr \
  --node-id=yxtpu-v4-64-train \
  --zone=us-central2-b \
  --accelerator-type=v4-64 \
  --runtime-version=tpu-ubuntu2204-base
```

Delete both resources when finished:

```bash
gcloud compute tpus tpu-vm delete yxtpu-v4-64-train \
  --zone=us-central2-b

gcloud compute tpus queued-resources delete yxtpu-v4-64-train-qr \
  --zone=us-central2-b
```

## Provisioning rules

- Treat the locations above as zones, not broad regions.
- Always pass the project and exact zone explicitly in provisioning commands.
- Never provision outside the table or use a different TPU generation without
  explicit user approval.
- Spot rows must be provisioned as Spot. Do not silently fall back to
  on-demand capacity.
- `gcloud compute tpus queued-resources create` defaults to on-demand when no
  tier flag is given: every v5e/v6e create command MUST pass `--spot`
  explicitly (verified against gcloud 576.0.0).
- On-demand is never allowed for v5e or v6e. The only on-demand quota in the
  grant is v4 (32 chips, `us-central2-b`), and it may be used solely when the
  user explicitly requests it in the current conversation.
- Default to an at-most-8-chip Spot slice for smoke tests when the chosen TPU
  generation's granted quota supports it. For v5e training, the minimum is 16.
- Get explicit user approval before requesting more than 8 chips or using the
  on-demand v4 quota.
- TPU Spot VMs can be preempted at any time. Before scaling, prove that the
  workload checkpoints to durable storage and restores correctly.
- Delete TPU VMs and queued-resource requests as soon as an experiment ends.
- The free offer covers the listed TPU quota only. Storage, networking, data
  egress, logging, and other Google Cloud services can still incur charges.
- The user's extra cloud credit is not standing authorization to create paid
  resources.

## Credentials and repository safety

- Do not commit Google credentials, access tokens, service-account keys, SSH
  private keys, Hugging Face tokens, or signed URLs.
- Prefer `gcloud auth login` for local CLI access. Use Application Default
  Credentials only when local programs need Google Cloud APIs.
- Let `gcloud compute tpus tpu-vm ssh` manage its dedicated Compute Engine SSH
  key. Do not copy a personal private key onto TPU VMs.
- Keep the Google Cloud project selection in local `gcloud` configuration or
  an ignored environment file instead of hard-coding it in source files.
