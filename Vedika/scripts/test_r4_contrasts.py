"""CPU self-check for r4_contrasts.py on synthetic rows.  Run: python Vedika/scripts/test_r4_contrasts.py"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import r4_contrasts as rc  # noqa: E402

rng = np.random.default_rng(1)
AUTH, Q = 20, 5  # 20 authors x 5 questions per TOFU set; 40 MMLU questions
SPEC = {"generation": {"max_tokens": 128}, "batch_size": 16, "stop_ids": [6, 3], "tokenizer": {"name": "t"},
        "scoring": "v1", "prompts": {"forget": "a"}}


def write(d, label, rouge, lp, correct, size=100.0, spec=SPEC):
    (d / label).mkdir(parents=True)
    p = d / label
    (p / "manifest.json").write_text(json.dumps({"model": f"m-{label}", **spec}))
    tofu = lambda pre: [{"id": f"{pre}{i}", "author": i // Q, "rougeL_recall": float(rouge[i]), "lp_answer": float(lp[i])}
                        for i in range(AUTH * Q)]
    for s in ("forget", "retain"):
        (p / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tofu(s)))
    for s in ("mmlu_en", "mmlu_hi"):
        (p / f"{s}.jsonl").write_text("".join(json.dumps({"id": f"q{i}", "correct": int(c)}) + "\n" for i, c in enumerate(correct)))
    (p / "summary.json").write_text(json.dumps({"size_mb": size, "sets": {s: {"truth_ratio_tofu": 0.5} for s in ("forget", "retain")}}))
    return rouge, lp, correct


def models(d, gap):  # original ~0.2, c0v2 ~0.2 + gap, c2 lower than c0v2
    n = AUTH * Q
    o = write(d, "original", 0.2 + 0.05 * rng.standard_normal(n), -3 + rng.standard_normal(n), rng.integers(0, 2, 40))
    t = write(d, "c0v2", 0.2 + gap + 0.05 * rng.standard_normal(n), -1 + rng.standard_normal(n), rng.integers(0, 2, 40))
    u = write(d, "c2", 0.2 + 0.3 * gap + 0.05 * rng.standard_normal(n), -2 + rng.standard_normal(n), rng.integers(0, 2, 40), 90.0)
    return o, t, u


tmp = Path(tempfile.mkdtemp())
(o, t, u) = models(tmp / "a", gap=0.6)
res = rc.contrasts(tmp / "a")["contrasts"]
u1 = res["U1"]
want = (u[0].mean() - t[0].mean()) / (t[0].mean() - o[0].mean())
assert abs(u1["forget_retention"]["point"] - want) < 1e-12, (u1["forget_retention"], want)
assert u1["forget_retention"]["ci_low"] < want < u1["forget_retention"]["ci_high"]
assert u1["forget_retention"]["invalid_draws"] == 0  # the gap is far above the noise
assert abs(u1["forget_lp"]["point"] - (u[1].mean() - t[1].mean())) < 1e-12
assert abs(u1["mmlu_en_pp"]["point"] - 100 * (u[2].mean() - t[2].mean())) < 1e-9
assert u1["size_mb"] == {"c2": 90.0, "c0v2": 100.0}
assert res["U2"].startswith("missing rows") and res["U3"].startswith("missing rows")
assert "U1: c2 − c0v2" in (tmp / "a/contrasts.md").read_text()

# denominator near 0 (c0v2 ~ original): bad draws are counted, not hidden
models(tmp / "b", gap=0.0)
assert rc.contrasts(tmp / "b")["contrasts"]["U1"]["forget_retention"]["invalid_draws"] > 0

# without the original, retention is pending; the other cells still compute
shutil.rmtree(tmp / "a/original")
r = rc.contrasts(tmp / "a")["contrasts"]["U1"]
assert r["forget_retention"] == "pending: original rows" and isinstance(r["forget_lp"], dict)


def must_fail(fn, *a):
    try:
        fn(*a)
    except AssertionError:
        return
    raise SystemExit(f"{fn.__name__}{a!r} did not refuse")


# rows made with other settings, or for other questions, are refused
n = AUTH * Q
write(tmp / "a", "c1", np.zeros(n), np.zeros(n), np.zeros(40), spec={**SPEC, "batch_size": 1})
must_fail(rc.contrasts, tmp / "a")
shutil.rmtree(tmp / "a/c1")
write(tmp / "a", "c1", np.zeros(n), np.zeros(n), np.zeros(40))
(tmp / "a/c1/forget.jsonl").write_text("".join(json.dumps({"id": f"x{i}", "author": 0, "rougeL_recall": 0, "lp_answer": 0}) + "\n"
                                               for i in range(n)))
must_fail(rc.contrasts, tmp / "a")
print("r4_contrasts: all checks passed")
