# Refusal study: protected vs random vs plain

`protected-q4` keeps the 77 EAP-IG refusal-circuit layers in bf16. `random30-q4` keeps 77 random layers in bf16. `plain-q4` quantizes every layer.

Study S contrasts, A − B on the same prompts. Refusal-rate Δ: paired bootstrap by prompt (judged in both); MMLU Δ in pp: question bootstrap; 10,000 resamples, seed 0, 95%. Harmful sets: higher refusal is better. XSTest-safe: higher refusal is over-refusal. Main comparison: S1 on harmbench_heldout.

**S1 = protected-q4 − random30-q4**

| cell | Δ [95% CI] |
| --- | --- |
| harmbench_heldout/English | -0.024 [-0.060, +0.012] (n=168) |
| aya_redteam_hi_ar/Arabic | -0.080 [-0.180, +0.000] (n=50) |
| aya_redteam_hi_ar/Hindi | +0.120 [+0.040, +0.220] (n=50) |
| xstest_safe/English | +0.012 [-0.008, +0.032] (n=250) |
| mmlu_en_pp | +1.8 [-0.6, +4.2] |
| mmlu_hi_pp | +2.0 [-1.5, +5.5] |

**S2 = protected-q4 − plain-q4**

| cell | Δ [95% CI] |
| --- | --- |
| harmbench_heldout/English | -0.024 [-0.065, +0.018] (n=168) |
| aya_redteam_hi_ar/Arabic | +0.040 [-0.060, +0.140] (n=50) |
| aya_redteam_hi_ar/Hindi | +0.060 [+0.000, +0.140] (n=50) |
| xstest_safe/English | +0.004 [-0.020, +0.028] (n=250) |
| mmlu_en_pp | +1.0 [-1.2, +3.2] |
| mmlu_hi_pp | +4.0 [+1.0, +7.5] |

**S3 = random30-q4 − plain-q4**

| cell | Δ [95% CI] |
| --- | --- |
| harmbench_heldout/English | +0.000 [-0.036, +0.036] (n=168) |
| aya_redteam_hi_ar/Arabic | +0.120 [+0.000, +0.240] (n=50) |
| aya_redteam_hi_ar/Hindi | -0.060 [-0.140, +0.000] (n=50) |
| xstest_safe/English | -0.008 [-0.024, +0.008] (n=250) |
| mmlu_en_pp | -0.8 [-3.2, +1.6] |
| mmlu_hi_pp | +2.0 [-1.0, +5.0] |

XSTest-safe refusal rate by prompt type (over-refusal; descriptive, no CI): rate (judged / total prompts).

| type | original | plain-q4 | random30-q4 | protected-q4 |
| --- | --- | --- | --- | --- |
| definitions | 0.00 (25/25) | 0.00 (25/25) | 0.00 (25/25) | 0.00 (25/25) |
| figurative_language | 0.08 (25/25) | 0.04 (25/25) | 0.04 (25/25) | 0.04 (25/25) |
| historical_events | 0.00 (25/25) | 0.00 (25/25) | 0.00 (25/25) | 0.00 (25/25) |
| homonyms | 0.04 (25/25) | 0.08 (25/25) | 0.08 (25/25) | 0.04 (25/25) |
| nons_group_real_discr | 0.00 (25/25) | 0.00 (25/25) | 0.00 (25/25) | 0.00 (25/25) |
| privacy_fictional | 0.48 (25/25) | 0.40 (25/25) | 0.44 (25/25) | 0.52 (25/25) |
| privacy_public | 0.04 (25/25) | 0.00 (25/25) | 0.00 (25/25) | 0.04 (25/25) |
| real_group_nons_discr | 0.00 (25/25) | 0.00 (25/25) | 0.00 (25/25) | 0.00 (25/25) |
| safe_contexts | 0.16 (25/25) | 0.20 (25/25) | 0.16 (25/25) | 0.16 (25/25) |
| safe_targets | 0.04 (25/25) | 0.08 (25/25) | 0.00 (25/25) | 0.04 (25/25) |

Estimated answers at the 256-token cap: estimated / n. Estimate = the decoded answer text re-tokenizes to >= 256 tokens; this approximates, and cannot establish, the generated length or whether an answer was truncated. A truly capped answer was judged on truncated text.

| set / language | original | plain-q4 | random30-q4 | protected-q4 |
| --- | --- | --- | --- | --- |
| harmbench_heldout/English | 36 / 168 | 38 / 168 | 41 / 168 | 41 / 168 |
| aya_redteam_hi_ar/Arabic | 13 / 50 | 17 / 50 | 9 / 50 | 12 / 50 |
| aya_redteam_hi_ar/Hindi | 32 / 50 | 31 / 50 | 33 / 50 | 29 / 50 |
| xstest_safe/English | 101 / 250 | 95 / 250 | 90 / 250 | 106 / 250 |

Utility: Global-MMLU accuracy [95% question-bootstrap CI]; change vs original in percentage points (paired).

| set | model | accuracy | Δ vs original (pp) |
| --- | --- | --- | --- |
| mmlu_en | original | +0.562 [+0.518, +0.606] |  |
| mmlu_en | plain-q4 | +0.548 [+0.504, +0.590] | -1.4 [-4.4, +1.6] |
| mmlu_en | random30-q4 | +0.540 [+0.496, +0.582] | -2.2 [-5.2, +0.8] |
| mmlu_en | protected-q4 | +0.558 [+0.514, +0.602] | -0.4 [-3.0, +2.2] |
| mmlu_hi | original | +0.515 [+0.445, +0.585] |  |
| mmlu_hi | plain-q4 | +0.495 [+0.425, +0.565] | -2.0 [-6.5, +2.5] |
| mmlu_hi | random30-q4 | +0.515 [+0.445, +0.585] | +0.0 [-4.0, +4.0] |
| mmlu_hi | protected-q4 | +0.535 [+0.465, +0.605] | +2.0 [-2.0, +6.0] |
