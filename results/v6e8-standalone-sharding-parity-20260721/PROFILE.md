# Standalone scan sharding parity

The standalone and MaxText decoder scans both use `nnx.split`, move the
parameter scan axis, merge one sliced NNX graph inside the body, rematerialize
the body, and call `jax.lax.scan`. The performance gap was not caused by a
different scan primitive.

The standalone trainer traced imported MaxText leaf layers without MaxText's
mesh/logical-axis context, and its owned hybrid layer omitted the activation
constraints used by the corresponding Qwen layer. Consequently, the MLP
all-gathered a process-local activation before its projection. In the old
two-step XPlane capture there are 256 exact `all-gather` events, including:

```text
bf16[4,2048,1024] -> bf16[32,2048,1024]
source: maxtext/layers/linears.py:102
```

After installing the logical mesh context during construction and lazy JIT
tracing, and constraining the embedding, normalized activations, residuals,
MLP input, layer output, and final norm, the matched two-step trace contains
zero exact or related all-gather events.

At the controlled batch-4 operating point this changes throughput from
266,481.54 to 450,508.09 tokens/s, a 69.06% gain. Restoring batch 8 reaches
545,495.03 tokens/s, 97.25% of the historical 560,923 MaxText selected result.

The first standalone max-throughput rerun was invalid: it treated batch 16 as
the entire per-device update batch and split it into eight microbatches of two,
processing only 262,144 tokens/update. The profile is defined as microbatch 16
per device accumulated eight times. Correcting the iterator batch semantics
processes 2,097,152 tokens/update and reaches 598,517.45 tokens/s, 99.36% of
the historical 602,373.45 result.

The runtime `peak_bytes_in_use` field is retained in raw run summaries but is
not used here as compiled-memory evidence; it does not reproduce XLA's
compiled-memory analysis.
