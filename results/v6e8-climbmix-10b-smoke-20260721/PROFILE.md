# ClimbMix 10B training-stack gate (throughput result rejected)

> **Post-gate correction:** the guarded fused KDA backward became non-finite at
> production step 12. Its 566k tok/s result is retained as historical evidence
> but is not a valid training operating point. EXP-032 and
> `../v6e8-climbmix-realtext-precision-20260721/` contain the trigger-batch
> comparison and selected full-FP32 analytical path.

## Workload

- TPU: one Spot v6e-8 in `europe-west4-a`
- Model: 309,111,392 parameters, padded GPT-2 vocabulary 50,432
- Data: streamed `karpathy/climbmix-400b-shuffle`, on-the-fly GPT-2 fast
  tokenization, dense 2,048-token packing
- Update: microbatch 16/device, eight gradient-accumulation steps,
  2,097,152 tokens/update
- Optimizer: AdamW, gradient clipping 1.0
- Loss: data-parallel Tokamax fused linear cross-entropy
- Precision: BF16 traffic, FP32 weights/state/guarded WY solve
- Checkpointing: disabled

## Throughput gate

The seven-step real-data run compiled with a 31,989,071,680-byte executable
peak estimate. The two post-warmup steps averaged 566,328.3 global tok/s; all
six steps after the initial dispatch were between 566.18k and 566.48k tok/s.
The host input pipeline separately prepared a complete 2,097,152-token update
at 1.173M tok/s after cache warmup, so streaming/tokenization did not starve the
accelerator.

`throughput/` contains the resolved smoke configuration, per-step JSONL, and
summary. The result is a short compilation/performance gate, not a quality run.

## Full-stack gate

The final one-step run exercised, in order:

1. one full eight-device optimizer update;
2. held-out ClimbMix loss;
3. the separate gradient/parameter/activation/attention diagnostic executable;
4. all ten requested EleutherAI tasks with a two-document smoke limit;
5. complete harness-result JSON serialization; and
6. W&B metric and artifact upload.

All diagnostics were finite. The harness scores in this smoke are not quality
measurements; the limit exists only to verify task loading, live-model scoring,
metric routing, serialization, and upload. `full-stack/` contains the resolved
configuration, JSONL, summary, and complete harness result.

W&B smoke: <https://wandb.ai/davidfranco2300-other/yxtpu-pretrain/runs/raxd2gkf>

## Superseded selection

The padded GPT-2 output layer makes the standard-loss microbatch-16/GA-8
executable exceed physical HBM during compilation. Fused loss is therefore the
selected capacity path for this 309.1M profile even though standard loss remains
faster for the smaller-vocabulary 272.9M profile in EXP-030.

The fused output loss remains selected. The guarded fused KDA implementation
does not: real training now uses the full-FP32 analytical VJP at about 173k
tok/s.
