# Qwen3.5 teacher notes for GOLD distillation

Both teachers share the 248,320-wide logit space the yx49k artifacts map
into (`student_to_teacher.npy`, `teacher_covered.npy`), so swapping teacher
size requires no re-alignment. Configs fetched from the HF hub 2026-07-30;
both are `Qwen3_5ForConditionalGeneration` (VLM wrapper — `text_config` is
the part we score with) with tied embeddings and one MTP layer that plain
scoring ignores.

| field | Qwen3.5-0.8B | Qwen3.5-4B |
| --- | --- | --- |
| hidden_size | 1024 | 2560 |
| num_hidden_layers | 24 | 32 |
| full_attention_interval | 4 (3:1 GDN:full) | 4 |
| GDN heads (key/value × dim) | 16/16 × 128 | 16/32 × 128 |
| linear_conv_kernel_dim | 4 | 4 |
| full attn (q/kv × head_dim) | 8/2 × 256 | 16/4 × 256 |
| partial_rotary_factor | 0.25 | 0.25 |
| attn_output_gate | true | true |
| intermediate_size | 3584 | 9216 |
| vocab (logit width) | 248,320 | 248,320 |
| rms_norm_eps | 1e-6 | 1e-6 |

Both configs were re-fetched 2026-08-04 and confirm `MoE fields present:
[]` — the dense sizes are dense, `tie_word_embeddings: true`, and both ship
a vision tower we discard. Note `vocab_size: 248320` in the config against
248,077 tokenizer entries: the padding rows are real initialized
parameters, so `project_teacher_logits` must be called with
`valid_vocab=teacher_covered.shape[0]` or the pad logits enter the
partition function.

Architecture affinity: the GDN mixer is the ancestor of our KDA layer
(scalar per-head decay vs channel-wise; our config fields are literally
named `gdn_*`), and the hybrid ratio matches ours. Teacher use is
forward-only — no backward kernels needed — and if scoring ever needs a
fast path, KDA's fused kernel reproduces GDN exactly when the scalar decay
is broadcast across channels.

## What MaxText actually gives us

Verified against the vendored tree rather than assumed:

* **Text-decoder numerics: covered.** `Qwen3_5GatedDeltaNet` and
  `Qwen3_5FullAttention` are bare subclasses of the Qwen3-Next layers with
  no overrides, and `tests/unit/qwen3_next_vs_reference_test.py` checks
  those against torch reference implementations at 1e-6. That parity
  transfers.
* **`tests/unit/qwen3_5_layers_test.py` does not help.** Its single test is
  `test_vision_encoder_subclasses_match_torch` — it covers the vision
  tower, which is exactly the part we drop. There is no text-decoder test
  under the `qwen3_5` name.
* **There is no dense-MLP path.** `Qwen3_5DecoderLayer.__init__` builds
  `Qwen3_5SparseMoEBlock` unconditionally (`models/qwen3_5.py:175`); no
  branch, no `num_experts` check. Neither dense teacher can load as-is.
* **No dense configs.** `configs/models/` carries only
  `qwen3.5-397b-a17b.yml` for this family.
* **No converter.** Nearest prior art is
  `convert_qwen3_next_{scanned,unscanned}.py` and `convert_qwen3_moe.py`.

Port checklist (TPU phase):
1. Add the dense branch to `Qwen3_5DecoderLayer`: select `MlpBlock`
   (already imported in `models/qwen3.py`) when the config declares no
   experts, and handle its return signature — the MoE block additionally
   sows a load-balance loss.
2. Dense config YAMLs for 0.8B and 4B. Field names follow
   `qwen3.5-397b-a17b.yml`; drop every `*_for_vit` key and the MoE block,
   and note the 4B's asymmetric 16-key / 32-value GDN heads.
3. HF→orbax conversion following the `convert_qwen3_next_*.py` patterns;
   drop vision tower and MTP weights; embeddings are tied.
4. Logit parity against `transformers` on a pinned prompt set (reference
   logits exported once from a torch machine; compared on TPU thereafter).
5. Blockwise teacher scoring: never materialize `[B, T, 248320]` at once —
   `project_teacher_logits` already consumes dense logits blockwise, and
   the unembedding can also be applied per vocab block if memory demands.

## Which teacher first

Both share the 248,320 logit width, so `student_to_teacher.npy`,
`teacher_covered.npy` and every line of loss code are identical between
them: swapping is config and weights only. The difference is compute. Our
student's step is ~6N = 1.85 GFLOP/token; a 4B teacher forward adds 8.0
(≈5.3× SFT cost per token), a 0.8B teacher adds 1.6 (≈1.9×) — before
rollout generation, which is usually the real wall-clock cost on-policy.
So validate parity on 0.8B, then decide whether 4B earns its premium. The
Mephisto sets came from the 4B, so it is the eventual target.
