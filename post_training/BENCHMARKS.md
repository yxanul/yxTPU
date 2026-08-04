# Post-training benchmark map

What each missing benchmark needs, verified against the installed
lm-eval 0.4.12 task registry and upstream on 2026-08-04. Two execution
paths exist and both are built: the loglikelihood scorer (`JaxHarnessLM`,
multi-year proven) and the generation path (`generate_until` via the
cached decoder, landed 2026-07-30, first used for IFEval).

## Ready today — loglikelihood (fast, deterministic)

| task | harness name | metric | reference setting | est. cost |
| --- | --- | --- | --- | --- |
| WinoGrande | `winogrande` | acc | Gemma 5-shot / SmolLM 0-shot | minutes |
| MMLU (cloze) | `mmlu_continuation` | acc | SmolLM reports this variant (30–36 for 360M class) | ~15 min (14k questions) |
| MMLU (standard) | `mmlu` | acc | chance-level at this scale (prev gen measured 26.7) — run for the record, expect ~27–30 | ~40 min |

`mmlu_continuation` is the important one: it is the cloze formulation
(score the answer text as a continuation, no A/B/C/D scaffolding), which
is the variant small models can actually express — and the one SmolLM's
tables report.

## Ready today — generation (needs sampling config; use T0.3/p0.9/pen1.1)

| task | harness name | metric | reference setting | notes |
| --- | --- | --- | --- | --- |
| TriviaQA | `triviaqa` | exact_match | Gemma 5-shot: 15.4; SmolLM2-360M: 16.9 | base-model prompt (no chat template) |
| GSM8K | `gsm8k` | exact_match, 5-shot | SmolLM2-360M: 3.2; Qwen2.5-0.5B: 33.4 | expect low single digits pre-SFT |
| IFEval | `ifeval` | 4 acc variants | Gemma IT: 51.2 | meaningful only after SFT; SFT-gen-1 measured 23.1 |

Generation-path notes from the IFEval run: chat template on for
instruction-tuned checkpoints, off for the base model; think-block
stripping on (unclosed traces score empty); batch = 16/device group.

## Needs integration — IFBench

Not in any lm-eval release (verified upstream master 2026-08-04: only
`ifeval` variants exist). The standalone `allenai/IFBench` repo is
self-contained: an IFEval-style `instructions_registry.py` with new
constraint families, plus `evaluation_lib.py` that scores a JSONL of
(prompt, response) pairs offline. Integration is therefore decoupled
from the harness:

1. Render IFBench prompts through our decoder (`generate` +
   chat template) → responses JSONL.
2. Run their `evaluation_lib` on the file (pure Python, local).

Est. one day including a checker-compat test, sequenced after SFT since
it is an instruction-following benchmark.

## Run plan against ckpt 700,000 (base model)

Single worker, standalone panel (the `eval_gemma_comparison.py` /
`eval_generative_tasks.py` pattern):

1. `winogrande` (0- and 5-shot) + `mmlu_continuation` + `mmlu` —
   loglikelihood pass, one launch.
2. `triviaqa` (5-shot) + `gsm8k` (5-shot) — generation pass, no chat
   template.
3. Matched-shot Gemma panel rerun (HellaSwag 10-shot, ARC-c 25-shot) to
   refresh the like-for-like table for the new base.
4. The anneal A/B: repeat 1–3 on ckpt 650,000 (already local) to
   quantify what the decay phase bought at matched settings.

IFEval/IFBench wait for the SFT'd model; GSM8K is worth re-running
post-SFT too (think-mode arithmetic measured far better than raw
completion in generation probes).
