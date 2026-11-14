Delta = later - earlier on the same questions; 95% CIs: author-clustered paired bootstrap (TOFU, 20 authors per set), question bootstrap (MMLU); 10,000 resamples, seed 0. Retention: original = 0, C0v2 = 1 (ROUGE-L recall, word overlap only). Truth ratio and size: point values. Headline cell: U2 forget retention.

| contrast | forget retention Δ | forget lp Δ (nats/token) | retain retention Δ | retain lp Δ | MMLU-en Δ (pp) | MMLU-hi Δ (pp) | forget truth ratio (later / earlier) | retain truth ratio (later / earlier) | size MiB (later / earlier) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1: c2 − c0v2 | -0.850 [-0.881, -0.818] | -1.737 [-1.900, -1.578] | -0.192 [-0.226, -0.159] | -0.115 [-0.153, -0.083] | -1.6 [-3.8, +0.6] | -1.0 [-3.5, +1.5] | 0.442 / 0.427 | 0.529 / 0.544 | 6388.1 / 6388.1 |
| U2: c3 − c2 | -0.058 [-0.081, -0.034] | +0.215 [+0.129, +0.305] | -0.469 [-0.523, -0.416] | -0.305 [-0.356, -0.255] | -4.2 [-8.0, -0.4] | +1.0 [-5.0, +7.0] | 0.463 / 0.442 | 0.511 / 0.529 | 2449.0 / 6388.1 |
| U3: c1 − c0v2 | -0.220 [-0.252, -0.186] | -0.065 [-0.078, -0.053] | -0.219 [-0.265, -0.174] | -0.074 [-0.087, -0.061] | -3.8 [-7.6, -0.2] | +0.5 [-4.5, +5.5] | 0.448 / 0.427 | 0.523 / 0.544 | 2449.0 / 6388.1 |

NF4 contrasts (U2-nf4, U3-nf4): no generated NF4 rows exist yet, so they are not reported.
