| model | set / language | n | refusal rate | Δ vs original [95% CI] | flips (worse / better) | first-token KL | mean KL | unjudged | size MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | harmbench_heldout/English | 168 | 0.792 |  |  |  |  | 0 | 6388.1 |
| original | aya_redteam_hi_ar/Arabic | 50 | 0.560 |  |  |  |  | 0 | 6388.1 |
| original | aya_redteam_hi_ar/Hindi | 50 | 0.460 |  |  |  |  | 0 | 6388.1 |
| original | xstest_safe/English | 250 | 0.084 |  |  |  |  | 0 | 6388.1 |
| plain-q4 | harmbench_heldout/English | 168 | 0.821 | +0.030 [-0.006, +0.065] | 0.054 (2 / 7) | 0.0774 | 0.0535 | 0 | 2449.0 |
| plain-q4 | aya_redteam_hi_ar/Arabic | 50 | 0.500 | -0.060 [-0.180, +0.060] | 0.220 (7 / 4) | 0.2395 | 0.0757 | 0 | 2449.0 |
| plain-q4 | aya_redteam_hi_ar/Hindi | 50 | 0.440 | -0.020 [-0.120, +0.060] | 0.100 (3 / 2) | 0.1080 | 0.0536 | 0 | 2449.0 |
| plain-q4 | xstest_safe/English | 250 | 0.080 | -0.004 [-0.028, +0.020] | 0.036 (4 / 5) | 0.1643 | 0.0679 | 0 | 2449.0 |
| protected-q4 | harmbench_heldout/English | 168 | 0.798 | +0.006 [-0.018, +0.036] | 0.030 (2 / 3) | 0.0329 | 0.0270 | 0 | 3652.6 |
| protected-q4 | aya_redteam_hi_ar/Arabic | 50 | 0.540 | -0.020 [-0.120, +0.080] | 0.140 (4 / 3) | 0.1805 | 0.0443 | 0 | 3652.6 |
| protected-q4 | aya_redteam_hi_ar/Hindi | 50 | 0.500 | +0.040 [-0.040, +0.120] | 0.080 (1 / 3) | 0.0834 | 0.0307 | 0 | 3652.6 |
| protected-q4 | xstest_safe/English | 250 | 0.084 | +0.000 [-0.024, +0.024] | 0.032 (4 / 4) | 0.0674 | 0.0381 | 0 | 3652.6 |
| random30-q4 | harmbench_heldout/English | 168 | 0.821 | +0.030 [+0.000, +0.065] | 0.042 (1 / 6) | 0.0703 | 0.0393 | 0 | 3652.6 |
| random30-q4 | aya_redteam_hi_ar/Arabic | 50 | 0.620 | +0.060 [-0.080, +0.200] | 0.260 (5 / 8) | 0.1459 | 0.0522 | 0 | 3652.6 |
| random30-q4 | aya_redteam_hi_ar/Hindi | 50 | 0.380 | -0.080 [-0.180, +0.000] | 0.120 (5 / 1) | 0.0902 | 0.0369 | 0 | 3652.6 |
| random30-q4 | xstest_safe/English | 250 | 0.072 | -0.012 [-0.036, +0.012] | 0.036 (3 / 6) | 0.1774 | 0.0498 | 0 | 3652.6 |
