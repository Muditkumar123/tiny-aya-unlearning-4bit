"""Recompute every reviewed Study U and Study S table from the public export alone, and check they match (CPU, ~1 min).

    python Vedika/release/reproduce.py [export dir]      # default Vedika/release/export

Works on a temporary copy, so the export is not changed. Exits non-zero on any difference.
- Study U: r4_contrasts.contrasts() (U1-U3 with intervals) and tofu_eval.table() (per-model levels).
- Study S: r5_tables.tables() (S1-S3, XSTest types, cap estimates, utility), and each model's summary against the
  original (refusal rate, paired Δ with interval, flips, KL), recomputed with evaluate.py's formulas.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import r4_contrasts  # noqa: E402
import r5_tables  # noqa: E402
from evaluate import SETS, flip_rate, paired_ci, read_jsonl  # noqa: E402
from tofu_eval import table  # noqa: E402


def same(name, got, want):
    assert got == want, f"{name}: recomputed values differ from the reviewed table"
    print(f"ok  {name}")


def summary_sets(d, base_dir):
    """evaluate.py step 4, from exported rows: per set and language, rates and (vs original) paired Δ, flips, KL."""
    out = {}
    for s in SETS:
        rows = read_jsonl(d / f"{s}.jsonl")
        base = {r["id"]: r for r in read_jsonl(base_dir / f"{s}.jsonl")} if d != base_dir else None
        direction = "maximize" if r5_tables.SETS[s] else "minimize"
        for g in sorted({r["group"] for r in rows}):
            rs = [r for r in rows if r["group"] == g]
            judged = {r["id"]: r["refusal"] for r in rs if r["refusal"] is not None}
            harm = [r["harmful_response"] for r in rs if r["harmful_response"] is not None]
            row = {"n": len(rs), "unjudged": len(rs) - len(judged),
                   "refusal_rate": sum(judged.values()) / len(judged) if judged else None,
                   "harmful_response_rate": sum(harm) / len(harm) if harm else None}
            if base is not None:
                b = {i: base[i]["refusal"] for i in judged if base[i]["refusal"] is not None}
                runs = [SimpleNamespace(predictions=[SimpleNamespace(sample_id=i, metadata={"scores": [
                    {"name": "refusal", "value": v, "direction": direction}]}) for i, v in x.items()]) for x in (b, judged)]
                fr = flip_rate(*runs, "refusal")
                row |= {"refusal_delta": paired_ci(b, judged),
                        "flips": {k: fr[k] for k in ("n", "flip_rate", "to_worse", "to_better")},
                        "first_token_kl": sum(r["first_token_kl"] for r in rs) / len(rs),
                        "mean_kl": sum(r["mean_kl"] for r in rs) / len(rs)}
            out[f"{s}/{g}"] = row
    return out


def main(export):
    t = export / "tables"
    load = lambda name: json.loads((t / name).read_text())  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "export"
        shutil.copytree(export, work)
        u = work / "study_u"
        same("Study U contrasts U1-U3", r4_contrasts.contrasts(u)["contrasts"], load("study_u_contrasts.json")["contrasts"])
        table(u, "original", "c0v2")
        same("Study U per-model levels", json.loads((u / "table.json").read_text()), load("study_u_levels.json"))
        s = work / "study_s"
        same("Study S tables (S1-S3, XSTest types, cap estimates, utility)",
             r5_tables.tables(s, n_tokens=lambda text: 1 / 0), load("study_s_tables.json"))
        e = s / "eval"
        for k in r5_tables.LABELS:
            same(f"Study S {k} vs original", summary_sets(e / k, e / "original"),
                 json.loads((e / k / "summary.json").read_text())["sets"])
    print("all reviewed tables reproduce from the export")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "export")
