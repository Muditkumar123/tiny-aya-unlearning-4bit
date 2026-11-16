"""A1 for Study S (PLAN sections 3 and 5): the extra tables beyond evaluate.py's results.md, from saved R5 rows (CPU).

    python evaluation/r5_tables.py      # reads runs/study_s/{eval,mmlu}/<label>/, writes runs/study_s/tables.{json,md}

- XSTest-safe refusal rate per prompt type (10 types x 25 prompts): descriptive, no CI.
- Capped answers: share whose answer re-tokenizes to >= max_tokens (256), per set and language. Re-tokenizing
  decoded text approximates the generated length (checked equal on the R5 smoke answers); a capped answer was
  judged on truncated text.
- Utility: MMLU en500 / hi200 accuracy (question bootstrap) and its change vs original in percentage points
  (paired question bootstrap); 10,000 resamples, seed 0.
Refusal rates, their paired CIs, flips and KL per set and language are evaluate.py's (runs/study_s/eval/results.md).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import GEN, ORIG, ROOT, SETS, paired_ci, read_jsonl  # noqa: E402
from tofu_eval import cluster_ci  # noqa: E402

R5 = ROOT / "runs/study_s"
LABELS = ("original", "plain-q4", "random30-q4", "protected-q4")
# A - B on the same prompts (proposed before any R5 safety number was viewed)
CONTRASTS = {"S1": ("protected-q4", "random30-q4"), "S2": ("protected-q4", "plain-q4"), "S3": ("random30-q4", "plain-q4")}


def tables(r5=R5, n_tokens=None):
    if n_tokens is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ORIG)
        n_tokens = lambda text: len(tok(text, add_special_tokens=False).input_ids)  # noqa: E731
    labels = [k for k in LABELS if (r5 / "eval" / k / "summary.json").exists()]
    rows = {k: {s: read_jsonl(r5 / "eval" / k / f"{s}.jsonl") for s in SETS} for k in labels}
    res = {"labels": labels, "contrasts": {}, "xstest_types": {}, "capped": {}, "mmlu": {}}
    for name, (a, b) in CONTRASTS.items():  # refusal-rate delta per set and language, judged prompts only
        if {a, b} <= set(labels):
            res["contrasts"][name] = {"a": a, "b": b, "cells": {}}
            for s in SETS:
                for g in sorted({r["group"] for r in rows[a][s]}):
                    ref = {k: {r["id"]: r["refusal"] for r in rows[k][s] if r["group"] == g and r["refusal"] is not None}
                           for k in (a, b)}
                    res["contrasts"][name]["cells"][f"{s}/{g}"] = paired_ci(ref[b], ref[a])
    for k in labels:
        by = {}
        for r in rows[k]["xstest_safe"]:
            by.setdefault(r["type"], []).append(r["refusal"])
        res["xstest_types"][k] = {t: {"n_total": len(v), "n_judged": len(j := [x for x in v if x is not None]),
                                      "refusal_rate": sum(j) / len(j) if j else None} for t, v in sorted(by.items())}
        res["capped"][k] = {}
        for s in SETS:
            for g in sorted({r["group"] for r in rows[k][s]}):
                rs = [r for r in rows[k][s] if r["group"] == g]
                # the public export carries n_tokens_estimate (the same re-tokenized count) instead of the answer text
                est = [r["n_tokens_estimate"] if "n_tokens_estimate" in r else n_tokens(r["response"]) for r in rs]
                res["capped"][k][f"{s}/{g}"] = {"n": len(rs), "capped_estimate": sum(e >= GEN["max_tokens"] for e in est)}
    for s in ("mmlu_en", "mmlu_hi"):
        got = {k: {r["id"]: r["correct"] for r in read_jsonl(r5 / "mmlu" / k / f"{s}.jsonl")}
               for k in LABELS if (r5 / "mmlu" / k / f"{s}.jsonl").exists()}
        res["mmlu"][s] = {}
        for k, v in got.items():
            q = {i: i for i in v}  # each question its own cluster
            cell = {"accuracy": cluster_ci({k: v}, q)}
            if k != "original" and "original" in got:
                cell["delta_pp_vs_original"] = cluster_ci({"o": got["original"], "m": v}, q, lambda m: 100 * (m["m"] - m["o"]))
            res["mmlu"][s][k] = cell
        for name, (a, b) in CONTRASTS.items():
            if name in res["contrasts"] and {a, b} <= set(got):
                res["contrasts"][name]["cells"][f"{s}_pp"] = cluster_ci({"a": got[a], "b": got[b]}, {i: i for i in got[a]},
                                                                        lambda m: 100 * (m["a"] - m["b"]))
    (r5 / "tables.json").write_text(json.dumps(res, indent=1))
    (r5 / "tables.md").write_text(render(res))
    return res


def render(res):
    f = lambda x, n=3: f"{x['point']:+.{n}f} [{x['ci_low']:+.{n}f}, {x['ci_high']:+.{n}f}]"  # noqa: E731
    labels = res["labels"]
    types = sorted({t for v in res["xstest_types"].values() for t in v})
    p = lambda x: f"{x['delta']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}] (n={x['n']})" if x.get("n") else "no shared judged prompts"  # noqa: E731
    out = ["Study S contrasts, A − B on the same prompts. Refusal-rate Δ: paired bootstrap by prompt (judged in both); MMLU Δ "
           "in pp: question bootstrap; 10,000 resamples, seed 0, 95%. Harmful sets: higher refusal is better. XSTest-safe: "
           "higher refusal is over-refusal. Proposed headline: S1 on harmbench_heldout.", ""]
    for name, c in res["contrasts"].items():
        out += [f"**{name} = {c['a']} − {c['b']}**", "", "| cell | Δ [95% CI] |", "| --- | --- |"]
        out += [f"| {cell} | {f(x, 1) if cell.endswith('_pp') else p(x)} |" for cell, x in c["cells"].items()] + [""]
    def xs(c):  # rate over the judged answers, with judged / total so every denominator is visible
        return "" if c is None else f"{'–' if c['refusal_rate'] is None else format(c['refusal_rate'], '.2f')} ({c['n_judged']}/{c['n_total']})"
    out += ["XSTest-safe refusal rate by prompt type (over-refusal; descriptive, no CI): rate (judged / total prompts).", "",
            "| type | " + " | ".join(labels) + " |", "|" + " --- |" * (len(labels) + 1)]
    out += [f"| {t} | " + " | ".join(xs(res["xstest_types"][k].get(t)) for k in labels) + " |" for t in types]
    out += ["", f"Estimated answers at the {GEN['max_tokens']}-token cap: estimated / n. Estimate = the decoded answer text "
            f"re-tokenizes to >= {GEN['max_tokens']} tokens; this approximates, and cannot establish, the generated length or "
            "whether an answer was truncated. A truly capped answer was judged on truncated text.", "",
            "| set / language | " + " | ".join(labels) + " |", "|" + " --- |" * (len(labels) + 1)]
    for g in res["capped"].get(labels[0], {}) if labels else []:
        out.append(f"| {g} | " + " | ".join(f"{res['capped'][k][g]['capped_estimate']} / {res['capped'][k][g]['n']}" for k in labels) + " |")
    out += ["", "Utility: Global-MMLU accuracy [95% question-bootstrap CI]; change vs original in percentage points (paired).", "",
            "| set | model | accuracy | Δ vs original (pp) |", "| --- | --- | --- | --- |"]
    for s, cells in res["mmlu"].items():
        for k, c in cells.items():
            d = c.get("delta_pp_vs_original")
            out.append(f"| {s} | {k} | {f(c['accuracy'])} | {f(d, 1) if d else ''} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    tables()
    print((R5 / "tables.md").read_text())
