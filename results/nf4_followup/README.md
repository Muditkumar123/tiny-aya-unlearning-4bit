# NF4 follow-up

This follow-up repeats the compression test with NF4, the 4-bit format used in [Forgetting That Sticks](https://arxiv.org/abs/2605.15138), and adds the paper's multiple-choice forget metric. It also contains MANSU run 3, which uses the paper's settings and a random-sign control. Code: [nf4_followup.py](../../evaluation/nf4_followup.py) and [run_nf4_followup.sh](../../evaluation/run_nf4_followup.sh).

## NF4 before and after

The models are bitsandbytes NF4 (blocks of 64, all 252 linear layers), and their weights shrink from 6.70 GB to 2.66 GB. Each cell shows bf16 -> NF4, with the paired change and its 95% interval. Intervals use 10,000 bootstrap draws, resampling authors for TOFU and questions for MMLU.

| Measure | Fine-tuned model (C0v2) | Change | Unlearned model (C2) | Change |
| --- | ---: | --- | ---: | --- |
| Forget MCQ accuracy (400) | 69.8% -> 67.3% | −2.5 pp [−4.2, −1.0] | 66.3% -> 65.3% | −1.0 pp [−3.7, +1.5] |
| Forget answer log-prob, nats/token (360) | −0.003 -> −0.033 | −0.030 [−0.033, −0.027] | −1.770 -> −1.508 | **+0.262 [+0.196, +0.329]** |
| Forget ROUGE-L recall (360) | 1.000 -> 0.951 | −0.049 [−0.061, −0.037] | 0.507 -> 0.471 | −0.036 [−0.052, −0.022] |
| Retain ROUGE-L recall (100) | 1.000 -> 0.914 | −0.086 [−0.114, −0.061] | 0.885 -> 0.647 | −0.238 [−0.287, −0.187] |
| Retain answer log-prob (100) | −0.003 -> −0.039 | −0.036 [−0.045, −0.028] | −0.059 -> −0.231 | −0.172 [−0.204, −0.140] |
| English MMLU (400) | 55.5% -> 53.5% | −2.0 pp [−5.5, +1.5] | 53.3% -> 53.8% | +0.5 pp [−3.3, +4.3] |
| Hindi MMLU (165) | 52.1% -> 49.7% | −2.4 pp [−6.1, +1.2] | 50.9% -> 48.5% | −2.4 pp [−7.3, +2.4] |

- NF4 raises the likelihood of the forgotten answers for the unlearned model and lowers it for the fine-tuned model. qint4 shows the same pattern. The rise is 15% of what NPO removed.
- Word overlap with the forgotten answers does not rise. It falls a little in both models.
- The unlearned model loses more retained knowledge under NF4 than the fine-tuned model does.
- Multiple-choice accuracy barely moves. NPO lowered it by only 3.5 pp in the first place [−5.5, −1.3], while it halved word overlap (ROUGE-L 1.00 -> 0.51).

## Update size against NF4 step size

From [cpu_check.json](cpu_check.json), across the 15 MLP matrices in MANSU's circuit:

| Quantity | As a share of a typical weight |
| --- | ---: |
| Largest move from 30 steps at learning rate 1e-6 | 0.12–0.14% |
| One NF4 step (64-weight block maximum × 0.0796) | 26–28% |
| MANSU floor (0.704 × one NF4 step) | 18–19% |

## MANSU run 3 (paper settings) with a random-sign control

| Version | Forget MCQ (40) | MMLU (100) | Answers changed (forget / MMLU) |
| --- | ---: | ---: | ---: |
| Start, and every checkpoint up to step 30 | 77.5% | 47–48% | 0 / – |
| Floor, signs from training | 27.5% | 25% | 20 / 68 |
| Floor, same weights and sizes, random signs | 80.0% | 47% | 1 / 6 |

The interpretation is in [MANSU on Tiny Aya](../../docs/MANSU.md#what-we-learned).

## Files

- `nf4_contrasts.json`: the NF4 table above
- `mcq/`: per-question forget MCQ predictions (ID, gold letter, predicted letter, letter log-probs)
- `nf4_build_c0v2.json`, `nf4_build_c2.json`: NF4 build records, including the save-and-reload check
- `cpu_check.json`: per-matrix update and step sizes
- `posthoc_npo_vs_c0_mcq.json`: NPO vs fine-tuned forget MCQ accuracy (added after the main comparison)
- `mansu_paper_settings/`: run 3 training history (`gate.json`), floor results (`floor.json`) and per-question predictions for both floor versions

Model answers and the NF4 weights stay local, under `runs/nf4_followup/`.
