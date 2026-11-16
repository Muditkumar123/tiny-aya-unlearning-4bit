"""R4/A1: Study U contrasts from re-verified TOFU rows (see docs/METHODS.md).

    python evaluation/r4_contrasts.py c0v2 c2 c1 c3 [original]    # CPU; only labels whose rows passed review

Links each label's reviewed row folder (SOURCES) into runs/study_u/, rebuilds tofu_eval's per-model table there
(results.md, table.json) and writes contrasts.json / contrasts.md.
U1 = c2 - c0v2, U2 = c3 - c2 (headline cell: U2 forget retention), U3 = c1 - c0v2; delta = later - earlier, same questions.
- retention delta: (mean later - mean earlier) / (mean c0v2 - mean original), ROUGE-L recall. One cluster_ci call over
  original, c0v2 and both models: the same author draws for all, ratio recomputed in each draw. Draws whose denominator
  is <= 0 or whose statistic is not finite are counted (invalid_draws), not dropped.
- lp delta: mean per-token gold-answer log-prob, nats/token, the same author draws for both models.
- MMLU delta: accuracy in percentage points, paired bootstrap by question.
- truth ratio (TOFU summary) and size MiB: point values only.
Retention cells need the original's rows; until they are passed in they read "pending".
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import ROOT, md5, read_jsonl  # noqa: E402
from tofu_eval import cluster_ci, table  # noqa: E402  (tofu_eval puts shared/ on the path)

OUT = ROOT / "runs/study_u"
SOURCES = {k: f"runs/tofu/{k}" for k in ("original", "c0v2", "c2", "c1", "c3")}
CONTRASTS = {"U1": ("c2", "c0v2"), "U2": ("c3", "c2"), "U3": ("c1", "c0v2")}
SETS = ("forget", "retain", "mmlu_en", "mmlu_hi")
SPEC = ("generation", "batch_size", "stop_ids", "tokenizer", "scoring", "prompts")


def contrasts(out=OUT):
    labels = sorted(p.parent.name for p in out.glob("*/manifest.json"))
    man = {k: json.loads((out / k / "manifest.json").read_text()) for k in labels}
    summ = {k: json.loads((out / k / "summary.json").read_text()) for k in labels}
    rows = {k: {s: read_jsonl(out / k / f"{s}.jsonl") for s in SETS} for k in labels}
    for k in labels:  # every model: same prompts, settings and questions in the same order
        bad = [f for f in SPEC if man[k].get(f) != man[labels[0]].get(f)]
        assert not bad, f"{k} was made with other {bad}"
        for s in SETS:
            assert [r["id"] for r in rows[k][s]] == [r["id"] for r in rows[labels[0]][s]], f"{k}/{s} holds other questions"
    col = lambda k, s, f: {r["id"]: r[f] for r in rows[k][s]}  # noqa: E731
    res = {"inputs": {k: {"dir": str((out / k).resolve()), "model": man[k]["model"],
                          "rows_md5": {s: md5((out / k / f"{s}.jsonl").read_bytes()) for s in SETS}} for k in labels},
           "contrasts": {}}
    for name, (later, earlier) in CONTRASTS.items():
        if not {later, earlier} <= set(labels):
            res["contrasts"][name] = f"missing rows: {sorted({later, earlier} - set(labels))}"
            continue
        c = {"later": later, "earlier": earlier, "size_mb": {k: summ[k]["size_mb"] for k in (later, earlier)}}
        for s in ("forget", "retain"):
            author = {r["id"]: r["author"] for r in rows[later][s]}
            if "original" in labels:
                v = {m: col(m, s, "rougeL_recall") for m in ("original", "c0v2", later, earlier)}

                def ratio(m):
                    return (m[later] - m[earlier]) / (m["c0v2"] - m["original"])

                def invalid(m):  # NaN marks a bad draw, so cluster_ci's nan_resamples counts them
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return np.where((m["c0v2"] - m["original"] > 0) & np.isfinite(ratio(m)), 0.0, np.nan)

                with np.errstate(divide="ignore", invalid="ignore"):
                    c[f"{s}_retention"] = cluster_ci(v, author, ratio)
                    # same seed and author count -> the same draws as the call above
                    c[f"{s}_retention"]["invalid_draws"] = cluster_ci(v, author, invalid)["nan_resamples"]
            else:
                c[f"{s}_retention"] = "pending: original rows"
            c[f"{s}_lp"] = cluster_ci({m: col(m, s, "lp_answer") for m in (later, earlier)}, author,
                                      lambda m: m[later] - m[earlier])
            c[f"{s}_truth_ratio"] = {k: summ[k]["sets"][s]["truth_ratio_tofu"] for k in (later, earlier)}
        for s in ("mmlu_en", "mmlu_hi"):
            v = {m: col(m, s, "correct") for m in (later, earlier)}
            c[f"{s}_pp"] = cluster_ci(v, {i: i for i in v[later]}, lambda m: 100 * (m[later] - m[earlier]))
        res["contrasts"][name] = c
    (out / "contrasts.json").write_text(json.dumps(res, indent=1))
    (out / "contrasts.md").write_text(render(res))
    return res


def render(res):
    f = lambda x, n=3: x if isinstance(x, str) else f"{x['point']:+.{n}f} [{x['ci_low']:+.{n}f}, {x['ci_high']:+.{n}f}]"  # noqa: E731
    lines = ["Delta = later - earlier on the same questions; 95% CIs: author-clustered paired bootstrap (TOFU, 20 authors "
             "per set), question bootstrap (MMLU); 10,000 resamples, seed 0. Retention: original = 0, C0v2 = 1 (ROUGE-L "
             "recall, word overlap only). Truth ratio and size: point values. Headline cell: U2 forget retention.", "",
             "| contrast | forget retention Δ | forget lp Δ (nats/token) | retain retention Δ | retain lp Δ | MMLU-en Δ (pp) "
             "| MMLU-hi Δ (pp) | forget truth ratio (later / earlier) | retain truth ratio (later / earlier) | size MiB (later / earlier) |",
             "|" + " --- |" * 10]
    for name, c in res["contrasts"].items():
        if isinstance(c, str):
            lines.append(f"| {name} | {c} |" + " |" * 8)
            continue
        inv = [c[f"{s}_retention"].get("invalid_draws") for s in ("forget", "retain") if isinstance(c[f"{s}_retention"], dict)]
        tr = lambda s: f"{c[f'{s}_truth_ratio'][c['later']]:.3f} / {c[f'{s}_truth_ratio'][c['earlier']]:.3f}"  # noqa: E731
        lines.append(f"| {name}: {c['later']} − {c['earlier']} | {f(c['forget_retention'])} | {f(c['forget_lp'])} | "
                     f"{f(c['retain_retention'])} | {f(c['retain_lp'])} | {f(c['mmlu_en_pp'], 1)} | {f(c['mmlu_hi_pp'], 1)} | "
                     f"{tr('forget')} | {tr('retain')} | {c['size_mb'][c['later']]:.1f} / {c['size_mb'][c['earlier']]:.1f} |"
                     + (f" invalid retention draws (forget, retain): {inv}" if inv and any(inv) else ""))
    lines += ["", "NF4 contrasts (U2-nf4, U3-nf4): no generated NF4 rows exist yet, so they are not reported."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    labels = sys.argv[1:]
    assert labels and set(labels) <= set(SOURCES), f"labels from {sorted(SOURCES)}"
    OUT.mkdir(parents=True, exist_ok=True)
    for k in labels:
        link, src = OUT / k, ROOT / SOURCES[k]
        assert (src / "manifest.json").exists(), f"no rows in {src}"
        if not link.is_symlink():
            link.symlink_to(src.resolve())
        assert link.resolve() == src.resolve(), f"{link} points elsewhere"
    table(OUT, "original", "c0v2")
    print(contrasts(OUT)["inputs"].keys())
    print((OUT / "contrasts.md").read_text())
