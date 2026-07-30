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

Architecture affinity: the GDN mixer is the ancestor of our KDA layer
(scalar per-head decay vs channel-wise; our config fields are literally
named `gdn_*`), the hybrid ratio matches ours, and the vendored MaxText
tree already implements the family (`models/qwen3_5.py`, inheriting
Qwen3-Next's `GatedDeltaNet`, with a unit test). Teacher use is
forward-only — no backward kernels needed — and if scoring ever needs a
fast path, KDA's fused kernel reproduces GDN exactly when the scalar decay
is broadcast across channels.

Port checklist (TPU phase):
1. Dense-model config YAMLs (in-tree Qwen3.5 configs cover only the MoE
   sizes; verify the scannable block's dense-MLP path — `mlp_only_layers`
   is empty here, layers are GDN/full-attention only).
2. HF→orbax conversion following
   `checkpoint_conversion/standalone_scripts/convert_qwen3_*.py` patterns;
   drop vision tower and MTP weights; note the 4B's asymmetric 16-key /
   32-value GDN heads.
3. Logit parity against `transformers` on a pinned prompt set (reference
   logits exported once from a torch machine; compared on TPU thereafter).
4. Blockwise teacher scoring: never materialize `[B, T, 248320]` at once —
   `project_teacher_logits` already consumes dense logits blockwise, and
   the unembedding can also be applied per vocab block if memory demands.
