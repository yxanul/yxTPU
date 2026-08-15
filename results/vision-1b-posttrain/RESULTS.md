# 1B post-training: Mephisto SFT + GOLD on the vision-1b-cont30b base

One-day demonstration (2026-08-14) that the yx49k post-training stack
transfers to the 1B multimodal base unchanged — same tokenizer, same
Mephisto data, same GOLD teacher stores (token-aligned, reused verbatim
from the 308M program).

## Runs

| stage | init | recipe | result |
| --- | --- | --- | --- |
| SFT (gen-2 recipe) | cont30b step 57,220 (orbax) | IF_172k + Knowledge_538k, 1 epoch, 1,240 steps, LR 4e-5 anneal→0, seq 2048 PDB4 | loss 2.92 → **1.12** (308M gen-2: → ~1.6) |
| GOLD mix300k | SFT pickle | gold-mix-k32 store (K=32, Qwen3.5-4B teacher), β=0 forward KL, 4,449 steps, LR 3e-5 anneal→0, PDB2 | distill 0.647 → **0.446** (308M: 0.98 → 0.62), rows_missing_targets **0** |

W&B: `posttrain-1b-sft` 3xwbsnt9, `posttrain-1b-gold` e1tfht4o
(20260814T131615Z run; an earlier 130300Z attempt drained its epoch in
26 steps — 32,395 missing store rows from a render mismatch: the store
manifest requires the local `/mnt/ram/sft/shard.jsonl` shards AND the
manifest's system prompt; HF-repo draws hash differently).

## IFEval (541 prompts, 0-shot, T0.3/p0.9/pen1.1 — the standard protocol)

| metric | 308M gen-2 | 308M +mix300k | 1B SFT | 1B GOLD |
| --- | --- | --- | --- | --- |
| prompt strict | 30.9 | 42.9 | 26.6 | 32.9 |
| prompt loose | 35.1 | 45.5 | 29.8 | 36.4 |
| instruction strict | 45.4 | 56.4 | 40.0 | 45.3 |
| instruction loose | 50.2 | 59.1 | 43.6 | 48.7 |
| **mean** | 40.4 | **51.0** | 35.0 | **40.8** |

SFT2 (4-dataset mix: IF + Knowledge + AMD-SFT-Mix_3.5M + MathCode:1M,
1 epoch, 5,200 steps @4096, LR 8.5e-5, full new diagnostics): IFEval
47.0 / 50.5 / 60.2 / 63.4, **mean 55.3** - the project best, above the
308M mix300k GOLD (51.0), with panel 23/32 at 32/32 clean stops and
markedly concise answers (panel generates in 3.9 s vs 21.6 s). Final
train loss 0.508. The instruction-density hypothesis is confirmed:
mix + LR were jointly worth +20.3 IFEval mean over SFT1.

## Qualitative panel (v2 set, 32 prompts, scored)

| checkpoint | correct | im_end stops | code section |
| --- | --- | --- | --- |
| 1B SFT | 12/32 | 0/32 | — |
| 1B GOLD | **24/32** | **31/32** | **8/8** |

(The 308M GOLD's 16/42 was on the v1 42-prompt set — not directly
comparable; panel transcripts: `panel_sft.md`, `panel_gold.md`.)

Decode probe: T0.2/pen1.3 on the GOLD checkpoint scores 17/32
(32/32 stops) vs 24/32 at the standard T0.3/pen1.1 — heavier
penalties trade correctness for termination, exactly as the 308M
decode sweep found. T0.3/pen1.1 stays the operating point; the
constraint-following gap is data, not decode.

## Reading

- GOLD **doubles** the panel score over plain SFT (12 → 24) and fixes
  template termination outright (0 → 31 of 32 clean `<|im_end|>` stops)
  — the teacher distributions carry both content and form.
- The 1B's *capability* (panel 24/32, code 8/8, distill 0.446) is well
  ahead of the 308M GOLD; its *constraint-following* (IFEval mean 40.8)
  matches the 308M's pre-GOLD level. Coherent with the data budget:
  6.7× fewer text tokens, and a third of supervision spent on vision —
  IFEval-style obedience tracks instruction-dense text exposure. The
  gap is the cleanest "what more compute buys" argument available.
- Ops note for reuse: process-0 SFT pickles must be distributed to all
  hosts before a multi-host `--init-pickle` run (seven workers die on
  FileNotFoundError while one blocks silently in the barrier — the
  AGENTS.md split-death pattern; internal-VPC HTTP copy takes ~10 s per
  host).

## GOLD2 (GOLD on the SFT2 base) - and the final ladder

| checkpoint | IFEval mean | panel | stops |
| --- | --- | --- | --- |
| SFT1 | 35.0 | 12/32 | 0/32 |
| GOLD1 | 40.8 | 24/32 | 31/32 |
| SFT2 | **55.3** | 23/32 | 32/32 |
| GOLD2 | 48.8 | **27/32** | 32/32 |

GOLD2 (distill 0.647->0.360, IFEval 40.3/43.8/54.0/57.0): best answer
quality of the project (panel 27/32) at a -6.5 IFEval cost vs SFT2 -
the store is only one-third IF and the teacher's discursive style
partially overwrites SFT2's strict-format obedience. SFT2 = obedience
champion, GOLD2 = quality champion; both above every 308M checkpoint
on their axis. A third round's obvious lever: an IF-denser store or
GOLD with ce_weight > 0 anchored on the SFT2 mix.

## Post-trained gsm8k and hand-graded probes (2026-08-15)

gsm8k (5-shot, full set, chat template): SFT2 3.3, GOLD2 **4.6** -
project best (base cont30b 3.18; 308M@367B 1.1).

Ten fresh reasoning/code/math prompts, hand-graded (probe_sft2.json,
probe_gold2.json): 9/10 substantively correct on BOTH checkpoints -
systems of equations solved and verified, canonical hash-map two_sum
with docstring, correct top-3 SQL (SFT2 used the exact given schema;
GOLD2 invented column names), xs[-1] bug fix, Rayleigh scattering.
Shared failure: haiku form (both ignore 5-7-5). Shared pathology: on
generous token budgets the model re-answers in a loop after the
correct final answer (the panel's short budgets mask this); a
presence-penalty or answer-once SFT signal is the obvious next lever.
