# J3 exploratory comparison: frozen plan (2026-11-14)

*Public copy of `Vedika/release/J3_EXPLORATORY_PLAN_20261114.md` (md5 603bd720). Only rater identities are replaced by "Rater A" and "Rater B", received file names by sheet labels, and agent names by their roles; everything else is unchanged.*

Frozen at 01:42 IST, **before any comparison with WildGuard was computed**.

**Decision** (team coordinator, an AI agent, 2026-11-14 01:40): J3 goes ahead as a separate, limited exploratory comparison. It is **not** reported as a completed independent, unaided, two-rater human validation under the original protocol (`J3_CALIBRATION_PLAN.md`). That protocol, its receipts and its pause on validation claims all stay in place. This design change was made after the sheets were received. It is recorded here with everything known at that point.

## What is already known (disclosed before computing)

- **Receipts:** both sheets were received complete (140/140 refusal and harmful_answer each), copied byte for byte and structurally valid.
- **Matching:** the five label columns are identical between the two sheets in all 700 cells. This is an observation; it proves neither independence nor its absence.
- **Rater accounts**, all attestations and none verified:
  - a team member, 01:26: "the sheets are manuuly filled";
  - Rater A, 01:34: "98% myself" plus "AI/translation tool"; then at 01:37, "actually its 100% bymyself";
  - Rater B, relayed by a team member: 01:36 "97% … and 3% AI"; then at 01:37, all by themselves.
- **Guide exposure:** Rater A had version 7703a689 from about 11:08 and 44f58e68 from 13:39, and reported no rows done before 13:39. Rater B: unknown, though 44f58e68 was in force by the time their sheet was saved.
- **Blinding:** procedural only. Everyone shares one machine account, so the raters could in principle have opened `key.json`. No such access is known, and none is inferred.

## Inputs (exact; the code checks the sha256 values and stops on any difference)

| Input | sha256 |
| --- | --- |
| sheet A, received 01:12 (`Vedika/out/j3/received/`) | `25364d5cbcd736af86b1f13b42b14ab366e63b61072f157c3e4ed0c50df4b9cd` |
| sheet B, received 01:25 (`Vedika/out/j3/received/`) | `634834294629fb842a47a645c4680e200213872a37fc2d8d708ec4b04e3c1bfd` |
| `Vedika/out/j3/key.json`: the item → model/id map plus the stored WildGuard labels | `032fa9d9dbf3d3b4dea1c07ee839ce23c360353df0308e85c75dbc1d317956ce` |
| `Vedika/out/j3/sample.json`: the frozen ids | `5a22e262335aacddcb32b3074c68dae66a0e72588bee872a5e0b6b5b46926389` |

- **Items:** 140 = (25 Hindi Aya + 10 English HarmBench held-out prompts) × 4 models (original, plain-q4, random30-q4, protected-q4).
- **Judge labels:** the `wildguard` fields in `key.json`, copied when the sheets were built (11-13 08:29). They came from the R5 Study S rows `Vedika/out/r5/eval/<model>/{aya_redteam_hi_ar,harmbench_heldout}.jsonl`: run 11-12 20:00-20:52, `evaluate.py` 25a5b446, AuditKIT GuardJudge with WildGuard in bf16 (judge path (a)), AI-auditor PASS 20:55. All 280 stored labels are integers; none is missing.
- **Label fields in the sheets:** refusal, harmful_answer, cut_off, unintelligible and prompt_not_harmful. **Only refusal and harmful_answer are compared** (with WildGuard's refusal and harmful_response). The other three have no direct judge counterpart and are not compared. The 5-field equality stays as analyst evidence only.

## Method (CPU; `Vedika/scripts/j3_exploratory.py`, test `test_j3_exploratory.py`)

- **Values:** "yes" → 1 and "no" → 0, ignoring case and surrounding spaces. A blank is counted as rater_missing; any other value as rater_invalid. A missing judge label is judge_missing. Nothing is imputed, and every item is counted in exactly one category per label.
- **Per sheet, per language (Hindi, English), per label:**
  - the 2×2 counts judge_{1,0} × rater_{1,0};
  - evaluated = the sum of those four counts;
  - mismatches = judge 1 / rater 0 plus judge 0 / rater 1;
  - mismatch_fraction = mismatches / evaluated, as a descriptive point value;
  - the missing and invalid counts.
- **Duplicates:** if the two sheets' refusal and harmful_answer vectors are identical, the output says so, and the two per-sheet results are reported as duplicates of **one** observed annotation pattern. They are not 280 independent judgments.
- **Not done:** no pooling, averaging or choosing a better sheet; no κ, confidence intervals, p-values, adjudication or human-versus-human contrasts; no safety certification.
- **Output:** `Vedika/out/j3/exploratory_20261114/results.json`, created once (exclusive create). It holds numbers only: no prompts, answers or notes.

## How the results may be described

An exploratory, descriptive comparison of the submitted annotations, whose provenance is the attestations above, against WildGuard on 25 Hindi and 10 English prompts × 4 models. It does not establish WildGuard's accuracy in Hindi, and it says nothing about Arabic. The prompts were sampled at random, but there are only 25 Hindi and 10 English prompts, and the 4 answers to each prompt are correlated.

## Revision, 2026-11-14 01:44 IST, before any comparison (AI-auditor review, 01:42)

- **Sample hash checked:** the code now verifies `sample.json` sha256 `5a22e262…926389`, and stops on a mismatch.
- **Membership checked:** the code now verifies that `key.json` holds exactly the 25 Hindi and 10 English frozen ids × the 4 models, each exactly once, with the right language label. It stops on any extra, missing, duplicated or mislabelled item. On the real key this check passes. Only membership was checked; no comparison was computed.
- **Missing-judge precedence:** if the judge label is missing, the item counts as judge_missing, even when the rater value is also missing or invalid. Otherwise a blank counts as rater_missing and any other non-yes/no value as rater_invalid. Every item lands in exactly one category per label.
- **Code and test:** `j3_exploratory.py` b2d990f6; `test_j3_exploratory.py` 5f922cf9 (adds precedence and wrong-membership rejection cases; passes).
