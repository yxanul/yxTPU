# Data-parallel fused linear cross-entropy

This experiment evaluates the opt-in Tokamax Mosaic TPU linear
softmax-cross-entropy path in the complete 272.9M KDA hybrid. It does not
select an isolated loss microbenchmark.

## Correctness gate

All checks ran on the eight devices of `yxtpu-v6e8-dev`, with hidden size
1,024 and vocabulary 32,768.

| Regime | Valid tokens | Loss relative | `dx` relative L2 | `dw` relative L2 |
| --- | ---: | ---: | ---: | ---: |
| all valid | 8,192 | 0 | 0.001659 | 0.003030 |
| uneven padding, one device fully masked | 4,584 | 0 | 0.001993 | 0.002819 |
| edge labels and strided mask | 5,464 | 8.999e-8 | 0.002863 | 0.004608 |
| scaled hidden states | 8,192 | 0 | 0.001659 | 0.003777 |

The full-model one-step AdamW check used one example per device, disabled
warmup, and compared standard against fused loss at identical parameters and
data:

| Metric | Result |
| --- | ---: |
| standard loss | 10.8826732635 |
| fused loss | 10.8826503754 |
| loss relative error | 2.103e-6 |
| gradient-norm relative error | 7.089e-6 |
| optimizer-state relative L2 | 8.579e-4 |
| final-parameter relative L2 | 7.585e-5 |

The overall gate passed. The raw update relative L2 of 0.0933 is diagnostic:
the first AdamW update sign-normalizes very small gradients, so it is not a
stable parity metric. Optimizer moments and final parameters are the update
gates.

## Full-model sweep

All successful rows run 30 steps with five warmup steps excluded, sequence
length 2,048, synthetic tokens, AdamW, and eight-way data parallelism.

| Loss | Microbatch/chip | GA | Remat | tok/s | Compiled estimate (bytes) | Allocator peak (bytes) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| standard | 8 | 1 | `minimal_with_context` | 545,528.782 | 16,289,117,536 | 3,885,205,504 |
| fused | 8 | 1 | `minimal_with_context` | 537,190.628 | 17,328,661,952 | 3,744,672,256 |
| standard | 16 | 8 | `save_dot_except_mlp` | 598,542.851 | 36,947,479,968 | 4,727,986,176 |
| fused | 16 | 8 | `save_dot_except_mlp` | 582,996.790 | 31,838,273,152 | 4,717,874,176 |
| standard | 32 | 4 | `save_dot_except_mlp` | 582,498.218 | 40,437,342,624 | 3,744,689,664 |
| fused | 32 | 4 | `save_dot_except_mlp` | 572,776.348 | 39,662,719,712 | 4,106,312,704 |
| fused | 64 | 2 | `full` | 550,038.807 | 37,709,894,912 | 3,885,204,992 |

Fused batch 64/GA=2 with `save_dot_except_mlp` did not compile. Switching to
`full` rematerialization made it fit. Standard loss with the same batch and
`full` policy failed compilation at 31.33 GB against 31.25 GB available, an
84.63 MB miss. The compiler attributed 28.09 GB to HLO temporaries; KDA and
MLP allocations dominated.

The JAX compiled estimate is
`arguments + output + temporaries - aliases`. It is reported consistently but
is conservative and can exceed physical HBM in an executable that fits.
Compiler success/OOM is authoritative at the capacity boundary; the allocator
counter is retained separately and is not substituted for the compiled
estimate.

## Selection

Standard loss at microbatch 16/GA=8 wins full-model throughput at 598,542.851
tok/s. The fused loss remains opt-in for capacity work. It saves 5.109 GB of
compiled estimate at that shape but loses 2.60% throughput. Its only unique
capacity result in this sweep is batch 64/GA=2 under full rematerialization,
which runs at 550,038.807 tok/s.

The tested async-all-reduce/data-parallel XLA flag bundle is also rejected. At
standard batch 16/GA=8 it measured 594,260.451 tok/s with a 42,578,818,816-byte
compiled estimate, versus 598,542.851 tok/s and 36,947,479,968 bytes for the
current profile.
