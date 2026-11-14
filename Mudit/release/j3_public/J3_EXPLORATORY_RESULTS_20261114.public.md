# J3 exploratory comparison: results (2026-11-14)

*Public copy of `Vedika/release/J3_EXPLORATORY_RESULTS_20261114.md` (md5 8d47cb55). Only rater identities are replaced by "Rater A" and "Rater B", received file names by sheet labels, and agent names by their roles; everything else is unchanged.*

Produced by the frozen plan `J3_EXPLORATORY_PLAN_20261114.md` (public copy `J3_EXPLORATORY_PLAN_20261114.public.md`) (with its pre-computation revision), using `Vedika/scripts/j3_exploratory.py` b2d990f6. Raw output: `Vedika/out/j3/exploratory_20261114/results.json` (sha256 `c58a5238f3e17045…`), written once. Everything here is numbers only. **This is not a validation of WildGuard.**

## One observed annotation pattern

Two sheets were submitted, by two raters. Their refusal and harmful_answer labels are identical on all 140 items, which the tool flagged (`identical_main_label_vectors: true`). For this comparison they therefore give **one** observed annotation pattern against the stored WildGuard labels. They are not two separate comparisons and not 280 independent judgments, and nothing was pooled or averaged. Whether the two sheets were rated independently is unconfirmed; this note neither assumes nor denies it.

Every item was evaluated. No rater value is missing or invalid, and no judge label is missing.

| Language | Label | judge yes / annotation yes | judge yes / annotation no | judge no / annotation yes | judge no / annotation no | mismatches / evaluated |
| --- | --- | --- | --- | --- | --- | --- |
| English (10 prompts × 4 models) | refusal | 34 | 0 | 0 | 6 | 0 / 40 |
| English | harmful answer | 5 | 1 | 0 | 34 | 1 / 40 |
| Hindi (25 prompts × 4 models) | refusal | 53 | 0 | 14 | 33 | 14 / 100 |
| Hindi | harmful answer | 1 | 0 | 10 | 89 | 10 / 100 |

The mismatch fractions are descriptive point values: English 0.00 and 0.025; Hindi 0.14 and 0.10.

## What it shows, and what it does not

- **Observed, per label:** in English the annotations and WildGuard almost always agree. In Hindi, each label's mismatches all go the same way: 14 answers the annotation marks as a refusal and WildGuard does not, and 10 answers the annotation marks as harmful and WildGuard does not. These are 24 *label* disagreements, not necessarily 24 distinct answers, since one answer can appear in both counts. No claim is made about how the two labels combine on the same answer.
- **Not shown:**
  - WildGuard's accuracy in Hindi or English;
  - anything about Arabic;
  - any safety property of the models;
  - any change to the Study S results, which stay WildGuard-label results as reported.
- **Why it is limited:**
  1. The rating's independence is unconfirmed. The two sheets match in all 700 label cells, and the raters' accounts changed during the same night: Rater A from "98% myself" plus an AI/translation tool to "100% bymyself"; Rater B, relayed by a team member, from "97% … 3% AI" to all by themselves.
  2. Blinding was procedural only, on a shared machine account.
  3. The worked examples were added after rating had reportedly started.
  4. Only 25 Hindi and 10 English prompts. The 4 answers to each prompt are correlated. There are no intervals, κ or tests.
  5. "Harmful answer" was judged by people who know the study. Some flagged answers may be cut at 256 tokens; the cap estimates are in the Study S tables.
  6. The design was changed after the sheets were received (team coordinator, 01:40), and was recorded before this comparison was computed.
- **Wording for the write-up:** "In an exploratory comparison of team-member annotations, of unconfirmed independence, with WildGuard on 140 answers (25 Hindi and 10 English prompts × 4 models), 14 of 100 Hindi refusal labels and 10 of 100 Hindi harmful-answer labels differed. In every one of those, the annotation was yes where WildGuard's label was no. In English, 0 of 40 refusal labels and 1 of 40 harmful-answer labels differed. The Hindi WildGuard labels remain unvalidated."
