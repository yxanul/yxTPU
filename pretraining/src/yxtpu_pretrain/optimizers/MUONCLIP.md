# Muon, AdamW, and the GQA MuonClip adaptation

Every trainable NNX parameter is declared with a semantic role at construction.
The route is therefore independent of path spelling:

- Muon: fused SwiGLU input/output matrices, KDA projection matrices, and GQA
  fused QKV/output matrices.
- AdamW: embeddings, logits, norm scales, biases, depthwise convolution,
  `A_log`, `dt_bias`, scalar parameters, and future AttnRes pseudoqueries.

For scanned tensors, the cycle axis is a Muon batch dimension. The declared
input/output axes are reshaped to matrices per cycle; the cycle is never folded
into either matrix dimension. Initialization fails on missing or unknown roles.

`muonclip` runs the same Muon+AdamW update, then applies the GQA adaptation of
Kimi K2's QK-Clip. It collects each GQA query head's maximum Tokamax logit,
reduces across the globally sharded batch, and computes
`c_h = min(1, tau / (max_logit_h + eps))`. Q heads receive `sqrt(c_h)`. Each
shared key head receives the square root of the smallest coefficient in its
query group, so no member can be amplified. V, output, and all KDA projections
remain untouched.

This is explicitly not the original MLA factorization described in the
[Kimi K2 report](https://arxiv.org/abs/2507.20534). MaxText's upstream
[QK-Clip utility](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/utils/qk_clip_utils.py)
targets MLA weights and is not reused here.

