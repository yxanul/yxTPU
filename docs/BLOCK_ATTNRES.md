# Block Attention Residuals for the KDA hybrid

Status: design v2, corrected against the source paper. v1 of this document
hypothesized a token-space shared-KV read; cross-checking arXiv:2603.15031
("Attention Residuals", Kimi Team) showed the actual mechanism is depth-wise.
`model.residual_policy=block_attnres` stays reserved until the A/B below runs.

## The mechanism (from the paper, Eq. 1-6)

Standard PreNorm accumulates every layer output with fixed unit weight:
`h_l = h_1 + sum f_i(h_i)`. AttnRes replaces the layer input with per-token
softmax attention over depth:

    h_l = sum_i  alpha_{i->l} * v_i,      alpha = softmax_i( w_l^T RMSNorm(v_i) )

where the sources `v_i` are the token embedding (always included) and each
preceding layer's output, and `w_l` is a learned per-site pseudo-query vector
(dimension d), deliberately decoupled from the forward computation - this is
what the repo's pre-declared `ATTNRES_PSEUDOQUERY` role is for. The softmax
makes `h_l` a convex combination instead of an unbounded sum: that is the
core semantic change (controls hidden-state growth, prevents late-layer
output inflation, lets different layer types weight history differently).

Block AttnRes is the practical variant: layers inside a block accumulate by
plain summation (`b_n = sum_{j in block} f_j`), and the depth attention runs
over only `[b_0 = embedding, b_1..b_{n-1}, partial b_n^i]` - N block
representations instead of L layer outputs, O(Nd) memory. The paper applies
the depth-read before every attention AND every MLP sub-layer, finds N ~= 8
recovers most of Full AttnRes, and validates on Kimi Linear 48B-A3B at 1.4T
tokens - the same KDA 3:1 hybrid family as this repo. The final output
aggregates all N block representations.

## Mapping to this codebase

The natural block is one hybrid cycle: [KDA, KDA, KDA, NoPE-GQA] plus its
four MLPs = 8 paper-"layers" summed into one block representation. With
num_cycles = 4 that gives N = 4 blocks + embedding: five stored [B, T, d]
sources - tiny.

- Depth-read sites: before each mixer and each MLP (8 per cycle), each with
  its own pseudo-query vector `w` (role `ATTNRES_PSEUDOQUERY`, AdamW-routed,
  matching the paper's decoupled-parameter design) and an RMSNorm applied to
  the sources inside the score function (role `NORM_SCALE`).
- Scan carry: the cycle scan currently carries only `hidden`. Block AttnRes
  carries `(blocks_buffer [N, B, T, d], block_index, partial_sum)`, with
  `dynamic_update_slice` at each cycle boundary and the depth softmax masked
  to sources <= current block. Buffer cost at the campaign shapes:
  4 x 8 x 2048 x 1024 bf16 ~= 128 MB/device - fits trivially in the 27 GB
  of headroom.
- Final aggregation: one more depth-read before `final_norm`.
- Precision: depth softmax in FP32; sources stay bf16.
- Remat: block representations must stay live across the block for the
  backward; they are already the size of one activation each, and
  `minimal_with_context` keeps comparable tensors today.
- FLOP cost: each site is O(N*d) per token - a few thousand multiplies
  against a ~22M-FLOP-per-token model. Expected step-time cost well under 2%.

## Hypothesis and predictions

Primary (the paper's motivation, inherited): replacing fixed-weight
accumulation with selective depth attention mitigates PreNorm dilution and
output growth. Hybrid-specific corollary: deep KDA-heavy blocks can
re-access the early blocks' exact-retrieval outputs and the raw embedding
directly, rather than through four cycles of compounded mixing.

Falsifiable predictions at 1B tokens under the campaign protocol
(muonclip, constant 3e-4, tied SuperBPE-128k, vs run 3's 3.872/3.929):

1. Train/holdout loss improves; reject the mechanism if it regresses.
2. Hidden-state norm growth across blocks flattens (add per-block norm
   telemetry; the paper predicts this directly).
3. Grad-norm stability unchanged or better.
4. Step time within 2% of baseline; carry memory within 200 MB/device.
5. Watch lambada specifically: if depth attention lets late blocks reuse
   block-1 retrieval, the hybrid's one eval loss (0.112 vs 0.127) should
   narrow - but this is a secondary signal, not the gate.

No cycle reordering is involved (v1's three-arm design is obsolete): this is
a plain two-arm A/B against run 3.

## Implementation map

- `config.py`: accept `block_attnres`; add `attnres: {sites: "mixer_and_mlp"
  | "mixer_only", norm_epsilon: 1e-5}`; flip the reserved-rejection test to
  acceptance.
- `layers/attn_res.py` (new): `DepthAttnRead` - pseudo-query param, source
  RMSNorm, FP32 masked softmax over the block axis, einsum combine.
- `model.py`: `HybridLayer` gains optional depth-read hooks before mixer and
  MLP; `HybridCycle.__call__` threads `(blocks_buffer, index, partial)`;
  `HybridLanguageModel` seeds the buffer with the embedding, adds the final
  aggregation read, and extends the scan carry.
- Roles: pseudo-queries -> `ATTNRES_PSEUDOQUERY` (AdamW), read norms ->
  `NORM_SCALE`. No Muon dimension numbers needed.
- Tests: config acceptance; source-masking correctness (block n attends to
  exactly n+1 sources plus partial); parameter count delta; a norm-growth
  telemetry smoke.

## Open questions before scaling past 1B

- Intra-block partial-sum reads: the paper includes `b_n^i` as a source for
  layers >= 2 of a block; start faithful, ablate `mixer_only` if step cost
  or memory surprises.
- Interaction with QK-Clip telemetry: max-logit reduction assumes the GQA
  layer sits at a fixed cycle position; unchanged here (no reordering).
- The paper reports the Block variant keeps most but not all of Full
  AttnRes's gains; with only N=4 blocks ours is closer to Full than theirs
  (they compress hundreds of layers to ~8 blocks), so the block
  approximation should cost us less.
