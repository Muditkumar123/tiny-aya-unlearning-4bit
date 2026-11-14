# Evaluation methods (AuditKIT part, for WRITEUP.md sections 3, 4, 5 and 6)

Draft by Vedika's Claude, 2026-11-12. The result numbers here come from tables Codex reviewed (sources at the end); one owner-reported check is labelled as such where it appears. This file adds no new results.

## What AuditKIT does in our pipeline

AuditKIT 1.1.3 is the scoring layer for both studies.

- **Generation and log-probs.** Every model, quantized or not, is wrapped in `common/addons/inmemory_model.py` (`InMemoryHFModel`, a subclass of AuditKIT's `HFGenModel` that takes a loaded model object). AuditKIT renders the chat template, generates greedily, and computes teacher-forced log-probs.
- **Safety judge.** Refusal labels come from AuditKIT's `GuardJudge` with its WildGuard profile (`allenai/wildguard`, bf16), in `Vedika/scripts/evaluate.py` `judge()`.
- **Harnesses.** `Vedika/scripts/tofu_eval.py` (Study U: TOFU and MMLU), `evaluate.py` (Study S: safety sets), `mmlu_only.py` (Study S utility: MMLU, scored the same way as `tofu_eval.py`).

## Frozen generation settings (both studies)

| Setting | Value |
| --- | --- |
| Prompt | Tiny Aya chat template with its default preamble; the original tokenizer for every model |
| Decoding | greedy; max 128 new tokens (TOFU), 256 (safety) |
| Batch | 16, left padding (the tokenizer's setting; recorded per model in the Study U run settings) |
| Stop tokens | 6 `<|END_OF_TURN_TOKEN|>` and 3 `<EOS_TOKEN>`. Tiny Aya's generation config stops only at 3; the fine-tuned models end turns with 6, so without both an answer could run past its turn. `evaluate.py` checks both IDs against the installed tokenizer at every launch. |
| Log-probs (Study U TOFU/MMLU and Study S utility only) | `tofu_eval.py` and `mmlu_only.py` upcast the lm_head output to fp32 (`fp32_logits=True`), so a summed 20-token log-prob is not rounded to bf16 steps. The safety run (`evaluate.py`) does not enable it; its scores are judge labels, and its KL is computed separately. |

A GPU check on 4 prompts with the original model (`Vedika/scripts/test_stop_gpu.py`) confirmed that AuditKIT's answer equals a direct greedy generation with the same stop tokens. Each of those answers ended at a stop token or at the token limit. On those prompts the base model ended its turns with token 3, so the stop-6 rule is a guard for the fine-tuned models rather than a change for the original.

Each run writes a manifest, and resuming or comparing against a run whose manifest differs is refused. All manifests record model identity, generation settings, batch size, stop tokens, tokenizer and template hash, and prompt hashes. The TOFU and utility manifests also record the scoring version; the safety manifests record the judge. The manifests do not record every execution detail: padding side, parameter dtypes and rendered prompt lengths were observed separately, in the Study U run settings (`run_settings.json`).

## The judge, and the check that chose it

WildGuard returns three labels per (prompt, answer) pair: harmful request, refusal, harmful answer. We score **refusal**.

- **Two ways to run it.** SafeTune runs WildGuard through a private function (`_run_hf_judge`). AuditKIT's public `GuardJudge` builds the same prompt, but its template starts with a literal `<s>` and the tokenizer adds another, so the judge would see two start tokens. We strip the literal `<s>`. The token IDs then equal SafeTune's. That rests on CPU tokenization tests in `test_judge_agreement.py` (reviewed) and on an owner-reported check of 100 inputs (Vedika.md 01:49), which was not independently reproduced. The reviewed evidence that the two paths agree is the 20-pair check below.
- **Frozen rule** (common/CHANGELOG.md 2026-11-12 01:49), set before any Study S scoring: use AuditKIT if it gives the same refusal label as SafeTune on 20 of 20 fixed pairs, otherwise keep SafeTune and disclose the private API.
- **The check.** 10 XSTest-safe prompts plus 5 Hindi and 5 Arabic Aya prompts, each with the original model's answer, were fixed and saved before either judge ran. Result: refusal labels agree on 20/20 (16 no, 4 yes). The other two labels and the raw judge text also agree 20/20. So Study S uses AuditKIT's GuardJudge.
- **Scope of the check:** 20 pairs show the two code paths agree; they do not measure how accurate WildGuard is. A refusal label is a judge's label, not a human rating.

## Data

| Set | Rows | Use | Source (pinned revision) |
| --- | --- | --- | --- |
| TOFU forget10 + perturbed answers | 400 (20 authors) | Study U forget | `locuslab/TOFU` @324592d8 |
| TOFU retain (full rows 0-399) | 400 (20 authors) | Study U retain | same |
| Global-MMLU en500 / hi200 | 500 / 200 | utility, both studies (en500 is the endpoint) | `CohereLabs/Global-MMLU` @0e619dbe |
| HarmBench held-out | 168 | Study S harmful requests | `walledai/HarmBench` @fb6c2afd, standard split minus the 32 circuit-discovery prompts |
| Aya red-teaming | 50 Hindi + 50 Arabic | Study S harmful requests, reported per language | `CohereLabs/aya_redteaming` @5a16fee0 |
| XSTest safe | 250 (10 types × 25) | Study S over-refusal | `Paul/XSTest` @f600c994 |

The builders are `Vedika/scripts/build_prompt_sets.py` and `build_tofu_sets.py`; the files are never overwritten. Overlap: every TOFU row scored is a training row of C0v2, and the retain rows are also in C2's retain term. Global-MMLU is the only held-out utility measure.

## Metrics, uncertainty and how intervals are read

All intervals are 95% percentile bootstraps, 10,000 resamples, seed 0.

| Study | Measure | Unit | Interval |
| --- | --- | --- | --- |
| U | Retention: ROUGE-L recall scaled so original = 0 and C0v2 = 1 | retention units | author-clustered (20 authors per set), paired; for a contrast the original, C0v2 and both models share each draw and the ratio is recomputed in it. Draws with a denominator ≤ 0 were counted: 0 of 10,000 in all six cells. |
| U | Gold-answer log-prob, mean per answer token | nats/token | author-clustered, paired |
| U | TOFU truth ratio | — | point value only |
| both | MMLU accuracy | percentage points for Δ | question bootstrap, paired |
| S | Refusal rate (harmful sets: higher is better; XSTest-safe: higher is over-refusal) | rate | paired by prompt, only prompts judged in both models; Aya Hindi and Arabic separate |
| S | Flip rate; first-token and mean KL to the original, teacher-forced over the original's answers | — | descriptive, no interval |
| both | Size | MiB | point value only: parameter memory as the harness accounts it (AuditKIT `model_info`, quanto packed weights counted by their parts). It is not measured checkpoint storage or runtime GPU memory. |

**Reading rule** (PLAN section 5): an interval above 0 is an increase, below 0 a decrease. An interval that covers 0, or has an endpoint exactly at 0, means **no change detected**. That is never read as "unchanged": no equivalence margin was set. With 50 prompts per Aya language, rates move in steps of 0.02 and intervals are wide.

ROUGE-L recall is word overlap with the gold answer, not factual recall. Log-prob measures how likely the model finds the gold answer, which NPO pushes down directly.

## Coverage and truncation

- **Coverage.** Study S: every one of the 2,072 answers (4 models × 518 prompts) got a refusal label (0 unjudged); every XSTest type is 25/25 judged for every model. Study U: all 400 + 400 TOFU rows and 700 MMLU rows per model.
- **Answers at the token limit (estimate).** The harness keeps answer text, not token counts. We estimate capped answers by re-tokenizing the text and counting those at ≥ 256 tokens. This approximates the generated length; it cannot establish it, or whether an answer was cut. By that estimate, 29-33 of 50 Hindi answers per model, 9-17 of 50 Arabic, 36-41 of 168 HarmBench and 90-106 of 250 XSTest reach 256 tokens. An answer that did stop at the limit was judged on its first 256 tokens.

## Limits of the evaluation

- **Judge.** All reported refusal results use one automatic judge, WildGuard. A planned blind human calibration (J3) did not produce a validation. Two team members submitted annotation sheets for 140 answers (25 Hindi and 10 English prompts × 4 models). The sheets have identical labels, and whether they were rated independently is unconfirmed. They were therefore used only in a separate exploratory comparison (`Mudit/release/j3_public/J3_EXPLORATORY_RESULTS_20261114.public.md`): 14 of 100 Hindi refusal labels and 10 of 100 Hindi harmful-answer labels differed from WildGuard's, all with annotation yes and WildGuard no; in English, 0 of 40 and 1 of 40 differed. This does not establish WildGuard's accuracy. The Hindi labels remain unvalidated, and Arabic was not examined at all.
- **Few clusters.** 20 authors per TOFU set; 50 prompts per Aya language.
- **One random draw.** Study S's random-protection control is a single draw (decision D4).
- **Truncation.** Answers at the 256-token limit (estimate above).
- **Selection.** C2's learning rate was chosen on the same TOFU rows Study U reports.
- **Multiplicity.** No correction. The headlines are fixed (Study U: U2 forget retention; Study S: S1 HarmBench refusal) and the other cells are supporting.
- **Not blind.** Predecessor sessions had seen earlier numbers for these artifacts; every result here is a re-verification under rules written before the reported tables.

## Sources (all reviewed)

| What | File | Handoff | Audit |
| --- | --- | --- | --- |
| Stop-token fix | `Vedika/scripts/evaluate.py`, `test_evaluate.py` | Vedika.md 01:49 | PASS 01:51 |
| Judge check | `Vedika/out/judge_agreement/result.json` | Vedika.md 19:48 | PASS 19:50 |
| Judge switch, stop check on GPU, smoke | `evaluate.py` 25a5b446, `test_stop_gpu.py`, `Vedika/out/r5_smoke/` | Vedika.md 19:58 | PASS 19:59 |
| Study U contrasts | `Vedika/out/r4/contrasts.md` | Vedika.md 19:57 | PASS 19:59 |
| Study S utility | `Vedika/out/r5/mmlu/` | Vedika.md 20:42 | PASS 20:43 |
| Study S tables code | `Vedika/scripts/r5_tables.py` 22f8a88a | Vedika.md 20:45 | PASS 20:45 |
| Study S rows and tables | `Vedika/out/r5/eval/results.md`, `Vedika/out/r5/tables.md` | Vedika.md 20:54, 20:55 | PASS 20:55, 20:56 |
