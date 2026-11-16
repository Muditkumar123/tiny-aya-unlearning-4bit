"""Paired percentile-bootstrap CI for the difference of two per-sample means (e.g. refusal rates).

Why not AuditKIT's paired_bootstrap (auditkit/_bootstrap.py): it returns no interval, and its p-value
counts resamples with |mean diff| >= |observed| without centring them, so a real effect gives p near 0.5
(a 1.0 -> 0.6 drop on 100 pairs gives p 0.52, "not significant").

    r = paired_ci(base_refused, cand_refused)   # {sample_id: 0/1} each; only shared ids count
    r["delta"], r["ci_low"], r["ci_high"]       # delta = cand - base
"""
from __future__ import annotations

import numpy as np


def paired_ci(base: dict, cand: dict, n_resamples: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    ids = sorted(base.keys() & cand.keys())
    n = len(ids)
    if n == 0:
        return {"n": 0, "base": None, "cand": None, "delta": None, "ci_low": None, "ci_high": None}
    b = np.array([base[i] for i in ids], dtype=float)
    c = np.array([cand[i] for i in ids], dtype=float)
    d = c - b
    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))  # resample ids, keep pairs
    lo, hi = np.quantile(d[idx].mean(axis=1), [(1 - level) / 2, (1 + level) / 2])
    return {"n": n, "base": float(b.mean()), "cand": float(c.mean()), "delta": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi)}
