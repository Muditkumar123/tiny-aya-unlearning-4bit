"""Paired 95% interval for the mean-KL gap between protected-q4 and each control, per prompt set (CPU).

    python evaluation/kl_gap.py [study_s/eval dir]      # default results/study_s/eval

Gap = protected-q4 minus control, on each prompt's mean KL to the original. Negative: protected is closer.
"""
import sys
from pathlib import Path

from evaluate import SETS, paired_ci, read_jsonl

CONTROLS = ("random30-q4", "plain-q4")


def gaps(e):
    out = {}
    for s in SETS:
        rows = {m: {r["id"]: r for r in read_jsonl(e / m / f"{s}.jsonl")} for m in ("protected-q4", *CONTROLS)}
        for g in sorted({r["group"] for r in rows["protected-q4"].values()}):
            kl = {m: {i: r["mean_kl"] for i, r in rows[m].items() if r["group"] == g} for m in rows}
            for c in CONTROLS:
                out[f"{s}/{g}", c] = paired_ci(kl[c], kl["protected-q4"])
    return out


if __name__ == "__main__":
    e = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results/study_s/eval"
    print("| set / language | control | n | control mean KL | protected mean KL | gap [95% CI] |\n| --- | --- | --- | --- | --- | --- |")
    for (cell, c), r in gaps(e).items():
        assert r["ci_high"] < 0, (cell, c)
        print(f"| {cell} | {c} | {r['n']} | {r['base']:.4f} | {r['cand']:.4f} | {r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
