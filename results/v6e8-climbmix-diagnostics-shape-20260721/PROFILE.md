# Batch-independent attention diagnostics

## Failure

The first 10B ClimbMix run trained normally through step 250, then the next
donated train-state call rejected two NNX intermediates compiled as
`[4,1,8]` because diagnostics had persisted `[4,128,8]`. Both were attention
maximum-logit records. This was independent of KDA and training numerics.

## Fix

Both the CPU dot-attention path and the TPU `AttentionOp` path now reduce the
maximum over batch and persist `[1,heads]`. The train-step accumulator is fixed
at `[cycles,1,heads]`, preserving the existing MuonClip per-head maximum
semantics across gradient-accumulation microbatches.

## Validation

- CPU: 47 tests pass. The new regression fails with `(1,4,2) == (1,1,2)` when
  the reduction is reverted.
- Native v6e-8: three ClimbMix updates, with held-out evaluation and full
  diagnostics at step 2. Step 3 executes normally after the recording pass.
- Step 2: train loss `11.3797007`, held-out loss `11.3297901`, diagnostic
  gradient norm `2.4623461`, maximum gradient `0.0679647`, and `finite=1`.
- Step 3: loss `11.3377438`, gradient norm `2.3069699`, `173,259 tok/s`.

No checkpoint, storage, or TPU lifecycle operation was performed.
