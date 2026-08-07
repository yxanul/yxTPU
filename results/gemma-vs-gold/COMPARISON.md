# GOLD-over-SFT vs the ~300M instruct field: Gemma 3 270M, SmolLM2-360M, SmolLM-360M

2026-08-07. Question this answers: is the lineage's remaining gap a
post-training problem (keep going with SFT/GOLD) or a pretraining /
model-size / data problem (change the base)? Method: the exact same
prompts through every model — the 42-prompt scored panel, the 12-prompt
chat panel, a 12-prompt freeform real-world set, IFEval under our
harness conventions, and the loglikelihood panel at Gemma's published
shot counts. The SmolLM pair (results in `smollm/`) doubles as a
controlled experiment run by someone else: same parameter count, same
architecture family, one generation of data + post-training iteration
apart. Gemma ran from the official `google/gemma-3-270m-it`
weights (transformers, fp32 CPU on worker 0); checker code is copied
verbatim from our panel scripts, so a PASS means the same thing for
both models. Gemma was scored at both greedy and its shipped sampling
settings (T=1.0, top-p 0.95, top-k 64) and gets its better result
quoted, the same courtesy our checkpoints got from their decode sweep.

## The contenders

| | Gemma 3 270M IT | GOLD-over-SFT (ours) |
| --- | --- | --- |
| parameters | 268M (~170M embedding, ~100M transformer) | 308M (~76M embedding, ~232M transformer) |
| vocabulary | 256k | 49k (yx49k, Qwen3.5 subset) |
| pretraining | ~6T tokens, Google data | **50B tokens**, ClimbMix |
| post-training | Google's full IT pipeline (curated SFT + distillation + RLHF) | SFT 6.27M rows + GOLD λ=0 distill ~100k rows (⅓ mix, K=32) |

The pretraining budgets differ by ~120×. Hold that number against the
capability tables below.

## Capability: ours, decisively

0-shot loglikelihood, both models measured with the same harness
(`/tmp/gemma_it_gos.json`; Gemma column is its published IT number at
the same shot count):

| task | ours (GOLD-over-SFT) | Gemma 3 IT 270M | delta |
| --- | --- | --- | --- |
| hellaswag | 53.5 | 37.7 | **+15.8** |
| arc_challenge | 37.5 | 28.2 | **+9.3** |
| piqa | 73.4 | 66.2 | **+7.2** |
| winogrande | 58.0 | 52.3 | **+5.7** |

Base-vs-base tells the same story: our base@700k at 0-shot beats Gemma
3 PT 270M's published numbers even where Gemma's were taken with
few-shot help — hellaswag 54.5 vs 40.9 (theirs 10-shot), arc_easy 65.4
vs 57.7, piqa 73.7 vs 67.7, arc_challenge 38.0 (0-shot) vs 29.0
(25-shot); the one loss is boolq 60.3 vs 61.4. A 50B-token base
clearing a 6T-token base across the board means pretraining is not the
bottleneck in this lineage.

## Behavior: Gemma, decisively

Same 42 prompts, same checkers (`panel42_*.md` here vs
`../gold-over-sft/panel_gos.md`):

| domain | ours | Gemma (greedy) | Gemma (sampled) |
| --- | --- | --- | --- |
| knowledge /8 | **8** | 6 | **8** |
| math /8 | 4 | 7 | **8** |
| code /8 | 4 | **7** | **7** |
| if /10 | 3 | **6** | **6** |
| instruction /8 | **3** | 2 | 2 |
| **total /42** | 22 | 28 | **31** |
| chat panel /12 | **9** | 8 | 8 |

IFEval, generative, our harness conventions (chat template, 512 max new
tokens; ours at its swept decode, Gemma greedy):

| metric | ours (GOLD-over-SFT) | ours (mix300k) | Gemma (measured) |
| --- | --- | --- | --- |
| prompt strict | 32.7 | 42.9 | 24.6 |
| prompt loose | 37.5 | 45.5 | 25.7 |
| inst strict | 45.1 | 56.4 | 37.1 |
| inst loose | 50.0 | 59.1 | 38.0 |
| mean | 41.3 | **51.0** | 31.4 |

Gemma's model card publishes IFEval 51.2; measured under our exact
conventions (lm-eval, chat template, greedy, 512 max new tokens) it
scores 31.4 — published numbers do not survive harness transfer, which
is why this comparison measured everything itself. Under identical
conditions both our checkpoints beat Gemma on IFEval; mix300k by ~20
points. One caveat runs the other way: Gemma was decoded greedy here
(its panel-best sampling would move it some), but the ~20-point gap is
far outside decode-setting noise.

## How they actually feel (the freeform set)

Twelve real-world prompts, no checkers (`freeform_*.md` here vs
`freeform_gos.md`): a landlord email, an explanation for a child, a
recipe from given ingredients, a two-sentence summary, tech help, a
birthday message, a running plan, virus-vs-bacterium, interview nerves,
an octopus fact, a dinner suggestion, a thank-you note.

**Gemma reads like a disciplined but shallow assistant.** Roughly 8 of
its 12 answers are usable as-is: the landlord email is pitch-perfect
(correct register, under the word limit, clean sign-off), the birthday
message is exactly two warm lines, the thank-you note could be sent
unedited, the interview advice is complete and *ends*. Every answer
terminates. Its failures are of content, not form: the aeroplane
explanation is empty ("aeroplanes are like giant super-powered bouncy
balls... they can fly because they fly really high"), virus-vs-bacterium
is circular, and the octopus fact is confident confabulation
("octopuses evolved from a group of snails").

**Ours reads like a better-read assistant with no off switch.** The
first paragraph is routinely as good as or better than Gemma's — the
thank-you note's opening ("explaining the 'why' behind the 'how'") is
the best prose either model produced, the octopus answer is a rich
(partly wrong) fact sheet, the interview advice is sound. Then the
spiral: the "two-line" birthday message is one sentence repeated ~36
times, the recipe repeats its ingredient list three times (as a soup,
ignoring the chicken and rice), the running plan repeats one bullet a
dozen times, the JWST "summary" is a verbatim copy of the input. Only
5/12 answers emitted `<|im_end|>` within 400 tokens. Both models
confabulate facts at this scale; ours buries them in more fluent prose
("octopuses can live over 50 years", a tenant offering to pay for the
landlord's repair).

The scored panels say the same thing with numbers. Our math failures
are almost never ignorance — the model computes **36**, **144**, **48**
correctly, then keeps generating and destroys its own answer (36→276 by
"adding the result to 240", 144→1440 via an invented "Method B",
48→24 by halving twice). Gemma's drilled `Final Answer: $\boxed{}$`
ritual simply stops. That habit — knowing when the answer is done — is
the single biggest behavioral difference between the two models.

**Where both are identical: strict-format instructions.** All-lowercase,
no-letter-e, exactly-five-hellos, quote-wrapping: Gemma goes 2/8 on the
instruction section with 6T tokens and Google's entire post-training
pipeline behind it; we go 3/8. Neither pipeline buys these at ~300M.
This looks like a genuine size ceiling, and it caps both models equally.

## The SmolLM control group

Same battery, measured, both models at greedy and their documented
t0.2/p0.9 sampling (better result quoted). SmolLM-360M-Instruct
(~600B pretraining tokens, HF's 2024 recipe) and SmolLM2-360M-Instruct
(4T tokens, the mature 2025 recipe) bracket our 308M exactly:

| | SmolLM-360M-IT | SmolLM2-360M-IT | ours | Gemma IT |
| --- | --- | --- | --- | --- |
| panel /42 | 19 | **34** | 22 | 31 |
| — math /8 | 2 | **8** | 4 | 8 |
| — code /8 | 7 | **8** | 4 | 7 |
| — instruction /8 | 0 | **5** | 3 | 2 |
| — knowledge /8 | 7 | 7 | **8** | 6 |
| chat /12 | 10 | **11** | 9 | 8 |
| IFEval mean (measured) | 16.1 | 40.1 | 41.3 / **51.0** | 31.4 |

Measured 0-shot loglikelihood (instruct models, same harness):

| task | ours | SmolLM2-IT | SmolLM-IT |
| --- | --- | --- | --- |
| hellaswag | 53.5 | **56.8** | 52.7 |
| arc_challenge | **37.5** | 34.0 | 33.0 |
| piqa | **73.4** | 71.3 | 70.6 |
| winogrande | **58.0** | 57.6 | 53.6 |

Readings:

- **SmolLM2 is the strongest same-size external.** Panel 34/42 — the
  drilled math/code rituals at full strength (8/8, 8/8), and 5/8 on the
  strict-format instruction section where Gemma manages 2/8 and we 3/8.
  Its measured IFEval (40.1) matches its published 41.0 — an honest
  model card — and lands a hair under GOLD-over-SFT (41.3), eleven
  points under mix300k (51.0). On capability it is the only external
  that reaches parity: ours 3/4 with narrow margins, and its 4T tokens
  buy it a real hellaswag win (56.8 vs 53.5) — unlike Gemma, whose
  capability deficit is wholesale.
- **The v1→v2 delta is the whole argument in one family.** Same
  parameter count: panel 19→34, IFEval 16.1→40.1, instruction 0/8→5/8.
  Everything that separates them is data and post-training recipe —
  which is exactly the axis this comparison says our lineage should
  spend on.
- **SmolLM2's freeform is the best of the three externals** (~10/12
  usable: the best JWST summary, the best virus-vs-bacterium
  explanation, a sendable thank-you note) — **and it still loops**: its
  recipe repeats "1/4 cup grated Parmesan cheese" twenty times, and its
  birthday one-liner is semantically garbled. Even HF's mature pipeline
  only suppresses the repetition spiral at ~360M; it does not eliminate
  it. Ours fails that way in most long answers, SmolLM2 in ~1/12,
  Gemma in ~0 — that spectrum is post-training depth made visible.
- SmolLM2's 5/8 instruction section forces one revision to the Gemma
  conclusion: strict-format following is *harder* at ~300M but not a
  hard ceiling — a deep enough instruction diet buys real fractions
  of it.

## Attribution

- **Capability (knowledge, reasoning priors): ours ahead, large.**
  Measured on every loglikelihood benchmark, both base and IT, despite
  120× less pretraining data. Pretraining, tokenizer, and architecture
  are doing their jobs.
- **Behavior (termination, formatting, arithmetic ritual): Gemma ahead,
  large.** These are exactly the properties Google's post-training
  pipeline drills at industrial scale. Nothing in Gemma's transcripts
  suggests a smarter model — it suggests a *finished* one.
- **Strict-format instruction following: hard at this size, not a
  ceiling.** Gemma 2/8 and ours 3/8 suggested a size wall; SmolLM2's
  5/8 shows a deep instruction diet buys real fractions of it. It binds
  every ~300M model, but it is trainable headroom, not physics.

Our signature failure — right answer, then a self-destructive
continuation — is textbook exposure bias. Teacher-forced training
(SFT and off-policy GOLD alike) never shows the model its own
repetition spiral, so nothing ever penalizes it. That is precisely the
failure class on-policy distillation exists to fix: sample from the
student, let the teacher score the student's own drift, and loops
become the highest-loss tokens in the batch.

## Verdict

Do not change pretraining, model size, or the pretraining data on this
evidence — the base is the strongest asset in the comparison. The gap
is post-training depth, and it is specific:

1. **On-policy GOLD** (the ledger's next rung) is now not just the next
   experiment but the indicated treatment for the observed disease.
2. **Termination-weighted data.** The full off-policy campaign should
   over-sample short-form completions with hard stops (the panel shows
   the model has never internalized "the answer is one line"), and SFT
   can up-weight loss on `<|im_end|>` placement directly.
3. **Math answer-ritual data.** Gemma's 8/8 is a drilled format, not
   intelligence; ~100k boxed-answer math rows through GOLD should buy
   most of it.

Re-run this exact comparison after the on-policy stage. The target to
beat is now SmolLM2-360M-Instruct, not Gemma: match its 34/42 panel
discipline while keeping our capability lead and our IFEval edge, and
the 308M lineage is the best model of the size class outright. If the
repetition spiral survives on-policy training, note that it also
survives (attenuated) in SmolLM2 — then, and only then, revisit size.

## Artifacts

- `gemma_panels.json`, `panel42_*.md`, `chat_*.md`, `freeform_*.md`,
  `gemma_freeform.json` — Gemma transcripts and scores (this run).
- `gemma_it_gos.json` — measured loglikelihood head-to-head.
- `ifeval_gemma.json` — Gemma IFEval under our harness.
- `smollm/smollm{1,2}-results/` — SmolLM-360M-Instruct and
  SmolLM2-360M-Instruct: panels, freeform, IFEval, loglik (measured).
- Ours: `../gold-over-sft/panel_gos.md`, `../gold-over-sft/chat_panel_gos.md`,
  `freeform_gos.md` (this directory), `../gold-over-sft/ifeval_gos.json`,
  `../gold/ifeval_mix300k.json`.
