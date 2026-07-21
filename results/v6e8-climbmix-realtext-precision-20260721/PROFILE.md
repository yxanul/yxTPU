# Real-text KDA backward precision gate

## Outcome

The guarded fused Pallas backward is rejected for real training. The selected
path is full-FP32 KDA with the analytical custom VJP and fused output loss:
172,961 global tok/s at microbatch 16/device and GA=8.

## Trigger

A normal guarded run reaches a raw gradient norm of 601.4 at update 7 and
becomes non-finite at update 12. Repeating with a learning rate of `1e-12`
keeps the weights effectively frozen but preserves data-dependent spikes:
446.8 at update 7, 40.9 at 13, and 11.2 at 15.

Update 7 was reconstructed from the deterministic ClimbMix stream and split
along its eight accumulation microbatches. Seven microbatches agree with the
full reference. Microbatch 4 does not:

| Path | Loss | Gradient norm | Max absolute gradient |
| --- | ---: | ---: | ---: |
| guarded Pallas | 11.382854 | 3,933.709961 | 171.0 |
| full FP32 | 11.382677 | 2.406770 | 0.054542 |

Since the forward loss remains close while the gradient diverges, this is a
fused-backward failure. The optimizer and Tokamax output loss are exonerated.

## Rejected mitigations

On the same bad microbatch:

| Pallas variant | Gradient norm | Max absolute gradient |
| --- | ---: | ---: |
| baseline | 3,933.7 | 171 |
| state matmuls six-pass | 42,583.7 | 1,768 |
| pairwise matmuls six-pass | 12,158.0 | 576 |
| chunk matmuls six-pass | 21,849.5 | 1,096 |
| all ordinary roles six-pass | 712,508.6 | 26,496 |
| pairwise row block 4 | 2,111.8 | 69 |
| pairwise row block 2 | 27,000.8 | 1,080 |
| midpoint pairwise anchor | 4,543.4 | 157 |

Proxy guards and isolated matmul-precision changes are not sufficient. Any
future fused backward must directly match the exact trigger microbatch and a
larger real-token sample before model throughput is considered.

## Safe path

| Implementation | Global tok/s | Compiled peak estimate | 15-step result |
| --- | ---: | ---: | --- |
| full FP32, generic autodiff | 158,745 | 44,662,410,976 bytes | finite |
| full FP32, analytical VJP | **172,961** | **33,265,817,024 bytes** | finite |

The analytical VJP is 9.0% faster and reduces the estimate by 11.40 GB. Its
loss curve and gradients match generic full-FP32 autodiff through the trigger.
Configuration now rejects guarded Pallas for non-benchmark experiments, and
the trainer terminates on the first non-finite loss or gradient norm.

## Safe full-stack smoke

The selected analytical path also completes held-out validation, the separate
gradient/parameter/activation/attention diagnostic, all ten requested lm-eval
tasks at a two-example smoke limit, JSON serialization, and W&B artifact upload.
The diagnostic reports gradient norm 2.4614, max gradient 0.06662, hidden RMS
1.0000, and finite attention-head logits. The lm-eval smoke takes 27.6 seconds.

W&B smoke: <https://wandb.ai/davidfranco2300-other/yxtpu-pretrain/runs/y4bgb72h>
