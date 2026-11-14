"""Flip rate: the share of samples whose pass/fail verdict changes between two AuditKIT runs.

Averages hide compression damage: a refusal rate can stay at 90% while many prompts swap
verdicts (Dutta et al. 2024, "Accuracy is Not All You Need", arXiv:2407.09141).

Works on any per-sample metric in two RunResults, e.g. guard_judge on the same HarmBench
prompts for the original and the compressed model:

    report = flip_rate(base_run, quant_run, "guard_judge")
    report["flip_rate"], report["to_worse"], report["worse_ids"]

Why not RunDiff.regressed(): RunDiff calls a sample "correct" when value == 1.0 unless the
Score has a threshold. guard_judge is MINIMIZE (1.0 = unsafe) with no threshold, so
regressed() lists the samples that became SAFE. Here the verdict honours the metric's direction.
"""
from __future__ import annotations

from auditkit.diff import direction_for, metric_directions
from auditkit.types import Direction


def _verdicts(run, metric: str, direction: Direction, cutoff: float) -> dict:
    """sample_id -> True when the sample is on the good side of the metric."""
    out = {}
    for p in run.predictions:
        for s in p.metadata.get("scores", []):
            if s.get("name") != metric:
                continue
            if s.get("passed") is not None:
                out[p.sample_id] = bool(s["passed"])
            # shortcut: binarizes continuous scores at `cutoff`; fine for 0/1 judges, pass a threshold for others
            elif direction == Direction.MINIMIZE:
                out[p.sample_id] = s["value"] < cutoff
            else:
                out[p.sample_id] = s["value"] >= cutoff
            break
    return out


def flip_rate(baseline, contrast, metric: str, cutoff: float = 0.5) -> dict:
    """Compare per-sample verdicts on `metric`; only samples scored in both runs count."""
    direction = direction_for(metric, metric_directions((baseline, contrast)))
    base = _verdicts(baseline, metric, direction, cutoff)
    cont = _verdicts(contrast, metric, direction, cutoff)
    shared = sorted(base.keys() & cont.keys())
    worse = [sid for sid in shared if base[sid] and not cont[sid]]
    better = [sid for sid in shared if not base[sid] and cont[sid]]
    n = len(shared)
    return {
        "metric": metric,
        "n": n,
        "flip_rate": (len(worse) + len(better)) / n if n else None,
        "to_worse": len(worse),
        "to_better": len(better),
        "worse_ids": worse,
        "better_ids": better,
    }
