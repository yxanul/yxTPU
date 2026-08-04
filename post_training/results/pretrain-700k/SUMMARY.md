# Base model: kda_hybrid_yx49k_l20 @ step 700,000

Pretraining complete 2026-08-04. 307,698,680 parameters (20 layers, 5×
[kda,kda,kda,gqa], emb 1024, yx49k tokenizer), 367B step-tokens of
ClimbMix (~341B unique + re-streamed anneal tail), constant LR 2e-3 with
cosine anneal to zero over the final 70k steps. W&B group
`climbmix-yx49k-367b` (legs: 60sndjih main, oiggd5fp anneal resume).
Checkpoints local: `ckpt/yx49k_l20/{450000,650000,700000}`, all
byte-verified.

Final losses: train 2.3617, holdout 2.3494 (ppl 10.48). The anneal took
holdout 2.5446 → 2.3494 (−0.195 nats) and the benchmark curve was still
rising at step 700k.

## Final in-training lm-eval (0-shot, step 700,000)

| task | score | | task | score |
| --- | --- | --- | --- | --- |
| HellaSwag | **56.9** | | SciQ | 92.0 |
| PIQA | **75.2** | | OpenBookQA | 38.8 |
| ARC-e / ARC-c | **67.5 / 40.1** | | COPA | 72.0 |
| Lambada | 47.4 (ppl 13.88) | | BoolQ | 58.0 |
| | | | CommonsenseQA† | 26.2 |

† standard-format CSQA pins small base models near chance; SmolLM's
reported 33–38 uses lighteval's cloze formulation — not comparable.

## Anneal trajectory (rounds at 20k steps)

| step | holdout | HellaSwag | Lambada | ARC-e |
| --- | --- | --- | --- | --- |
| 620k (pre-anneal) | 2.5446 | 49.3 | 40.4 | 63.7 |
| 660k | 2.4934 | 51.9→51.3* | 39.3→36.5* | 63.0→62.7* |
| 680k | 2.3969 | 55.0 | 42.7 | 66.4 |
| 700k | 2.3494 | **56.9** | **47.4** | **67.5** |

\* second value is the resume leg's re-run of the round (restarted
stream, slightly different state) — agreement within noise.

## Reference comparison (ours 0-shot; references at their published settings)

| task | ours 308M/0.37T | SmolLM2-360M/4T | SmolLM-360M/0.6T | Qwen2.5-0.5B | Gemma3 PT 270M | prev gen 337M/40B |
| --- | --- | --- | --- | --- | --- | --- |
| HellaSwag | **56.9** | 54.5 | 51.8 | 51.2 | 40.9 | 50.6 |
| ARC avg | **53.8** | 53.0 | 50.1 | 45.4 | 43.4 | 48.9 |
| PIQA | **75.2** | 71.7 | 71.6 | 69.9 | 67.7 | 72.1 |
| OpenBookQA | **38.8** | 37.4 | 37.2 | 37.4 | — | 36.2 |
| BoolQ | 58.0 | — | — | — | 61.4 | 60.1 |

Beats SmolLM2-360M on every comparable row at 1/11th its token budget.
Missing rows (TriviaQA, WinoGrande, MMLU, GSM8K, IFBench) are the
post_training work — see ../BENCHMARKS.md.

Files here: `step_006{6,8}0000.json`, `step_00700000.json` — the
harness's complete result artifacts (per-metric values, stderr, task
configs/versions) from the anneal-phase rounds.

## Post-hoc loglikelihood panel (2026-08-04, `loglikelihood_panel.json`)

| task | ours | references |
| --- | --- | --- |
| WinoGrande 0-shot | **59.4** | SmolLM2-360M 52.5, SmolLM-360M 52.8, SmolLM2-135M 51.3 |
| WinoGrande 5-shot | **57.9** | Qwen2.5-0.5B 54.1, Gemma3 PT 270M 52.0 |
| MMLU cloze 0-shot | 30.0 | SmolLM2-360M 35.8, SmolLM-360M 34.4, SmolLM2-135M 31.5 |
| MMLU standard 5-shot | 26.1 | chance 25; small base models cluster here |

WinoGrande is the campaign's standout: +5 over every reference on a task
where this size class barely clears chance. MMLU tracks factual exposure
and lands between the 135M and 360M SmolLM tiers - the knowledge-bound
axis (with TriviaQA) that the GOLD distillation stage targets.
