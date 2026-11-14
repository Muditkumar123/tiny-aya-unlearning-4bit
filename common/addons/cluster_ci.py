"""Cluster (author) percentile bootstrap for any statistic of several models' per-question means.

frozen.yaml's rule: resample authors with replacement, keeping all of an author's questions together;
10,000 resamples, seed 0, 95% CI. Every model is resampled with the same authors, so differences are paired.
A resample's mean is (sum over the drawn authors' questions) / (their question count).

    from cluster_ci import cluster_ci
    vals = {"original": {qid: rouge}, "c0": {...}, "c1": {...}}          # per-question values per model
    r = cluster_ci(vals, author_of, lambda m: (m["c1"] - m["original"]) / (m["c0"] - m["original"]))
    r["point"], r["ci_low"], r["ci_high"]                               # scaled retention of c1 and its CI

stat gets {model: mean} (floats for the point, arrays over resamples for the CI) and must be elementwise.
Only ids present for every model count. With one model and no stat, it is that model's mean.
"""
from __future__ import annotations

import numpy as np


def cluster_ci(values: dict, cluster: dict, stat=None, n_resamples: int = 10_000, seed: int = 0,
               level: float = 0.95) -> dict:
    models = list(values)
    if stat is None:
        assert len(models) == 1, "give a stat for more than one model"
        stat = lambda m: m[models[0]]  # noqa: E731
    ids = sorted(set.intersection(*(set(v) for v in values.values())))
    assert ids, "no question id is shared by every model"
    groups = sorted({cluster[i] for i in ids})
    g = np.searchsorted(groups, [cluster[i] for i in ids])
    G = len(groups)
    sums = np.stack([np.bincount(g, [values[m][i] for i in ids], G) for m in models], 1)  # authors x models
    counts = np.bincount(g, minlength=G)
    draws = np.random.default_rng(seed).integers(0, G, size=(n_resamples, G))
    w = np.zeros((n_resamples, G))
    np.add.at(w, (np.arange(n_resamples)[:, None], draws), 1)  # times each author is drawn
    means = (w @ sums) / (w @ counts)[:, None]
    boot = np.asarray(stat({m: means[:, k] for k, m in enumerate(models)}), dtype=float)
    lo, hi = np.nanquantile(boot, [(1 - level) / 2, (1 + level) / 2])
    full = sums.sum(0) / counts.sum()
    return {"point": float(stat({m: full[k] for k, m in enumerate(models)})), "ci_low": float(lo),
            "ci_high": float(hi), "n": len(ids), "n_clusters": G, "nan_resamples": int(np.isnan(boot).sum())}
