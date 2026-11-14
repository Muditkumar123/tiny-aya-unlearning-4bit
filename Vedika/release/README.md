# Evaluation export: Study U and Study S (numbers and ids only)

Local draft for the team's HF data collection. Nothing here has been published; release needs the team's approval.

## What is in `export/`

| Folder | Content | Rows |
| --- | --- | --- |
| `study_u/<model>/` (original, c0v2, c1, c2, c3) | `forget.jsonl`, `retain.jsonl`: id, author, split, ROUGE-L recall, gold/paraphrase/perturbed log-probs, truth ratio. `mmlu_en.jsonl`, `mmlu_hi.jsonl`: id, subject, gold letter, 4 letter log-probs, prediction, correct. Plus `manifest.json` (settings) and `summary.json`. | 400 + 400 + 500 + 200 per model |
| `study_s/eval/<model>/` (original, plain-q4, random30-q4, protected-q4) | per prompt: id, language group, HarmBench category or XSTest type, the three WildGuard labels as 0/1 (harmful request, refusal, harmful answer), KL to the original, and `n_tokens_estimate` (the answer's re-tokenized length, used only for the cap estimate). Plus manifest and summary. | 168 + 50 + 50 + 250 per model |
| `study_s/mmlu/<model>/` | Global-MMLU rows as above | 500 + 200 per model |
| `tables/` | the reviewed tables, as reviewed: Study U contrasts and levels, Study S contrasts/types/caps/utility, Study S vs original | — |

**Not included:** prompts, gold answers, model answers and the judge's raw text. No harmful model output is in this export. Local paths in manifests are shortened to `~/`.

## Reproduce the tables (CPU, about a minute)

```bash
~/miniconda3/envs/lexsi/bin/python Vedika/release/reproduce.py
```

This recomputes every table in `tables/` from the rows alone and stops on any difference. It covers:
- Study U: U1-U3 with intervals, and the per-model levels;
- Study S: S1-S3, XSTest types, cap estimates, utility, and each model against the original.

It needs the repository's `Vedika/scripts/` and `common/addons/`.

To rebuild the export from the lab outputs, run `HF_HUB_OFFLINE=1 python Vedika/release/export.py`. It refuses to overwrite an existing `export/`.

## The prompts: rebuild them, do not re-host them

The ids refer to fixed sets that `Vedika/scripts/build_prompt_sets.py` and `build_tofu_sets.py` rebuild from these pinned sources:

| Source | Revision | Licence (card) | In this export |
| --- | --- | --- | --- |
| `walledai/HarmBench` (standard) | fb6c2afd | MIT | ids only (`hb_std_<row>`) |
| `CohereLabs/aya_redteaming` | 5a16fee0 | Apache-2.0 | ids only (`aya_<lang>_<line>`) |
| `Paul/XSTest` | f600c994 | CC-BY-4.0 | ids and type only |
| `locuslab/TOFU` | 324592d8 | MIT | ids and author index only |
| `CohereLabs/Global-MMLU` | 0e619dbe | Apache-2.0 | ids, subject, gold letter |

The licences are read from the dataset cards and are not legal clearance. Cite HarmBench (Mazeika et al. 2024), XSTest, TOFU, Aya red-teaming and Global-MMLU as their cards ask. Model outputs come from `CohereLabs/tiny-aya-global` (CC-BY-NC-4.0 with Cohere's acceptable-use policy) and its derivatives.

## Limits that travel with these numbers

See `Vedika/writeup/EVALUATION_SECTION.md` for the full list. In short:
- a refusal label is WildGuard's, not a human rating;
- there are 20 authors per TOFU set and 50 prompts per Aya language;
- Study S has one random control;
- cap counts are re-tokenized estimates;
- C2 was selected on these TOFU rows;
- no multiplicity correction was applied.
