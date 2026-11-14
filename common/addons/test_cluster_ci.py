"""Self-check for cluster_ci (CPU). Run: python common/addons/test_cluster_ci.py"""
import numpy as np

from cluster_ci import cluster_ci
from paired_ci import paired_ci

rng = np.random.default_rng(1)
authors = {f"q{a}_{j}": a for a in range(10) for j in range(20)}
effect = rng.normal(0, 1, 10)  # all of an author's questions share one value: strong within-author correlation
x = {q: float(effect[a]) for q, a in authors.items()}

# point = plain mean over questions; same seed -> same interval
r = cluster_ci({"m": x}, authors)
assert abs(r["point"] - np.mean(list(x.values()))) < 1e-12 and r["n"] == 200 and r["n_clusters"] == 10
assert r == cluster_ci({"m": x}, authors)

# 10 authors, not 200 questions, are the sample: wider than the question-level bootstrap on the same data
iid = paired_ci({q: 0.0 for q in x}, x)
assert (r["ci_high"] - r["ci_low"]) > 3 * (iid["ci_high"] - iid["ci_low"]), (r, iid)

# with identical authors there is nothing to resample: zero width
flat = {q: 0.5 for q in authors}
f = cluster_ci({"m": flat}, authors)
assert f["ci_low"] == f["ci_high"] == f["point"] == 0.5

# paired: the same authors are drawn for every model, so a constant shift has a zero-width difference
shifted = {q: v + 0.3 for q, v in x.items()}
d = cluster_ci({"a": x, "b": shifted}, authors, lambda m: m["b"] - m["a"])
assert abs(d["point"] - 0.3) < 1e-12 and abs(d["ci_low"] - 0.3) < 1e-9 and abs(d["ci_high"] - 0.3) < 1e-9

# scaled retention: untuned = 0, target = 1; a model halfway between scores 0.5, and the scale is applied per resample
u = {q: 0.2 + 0.1 * v for q, v in x.items()}
t = {q: 0.8 + 0.1 * v for q, v in x.items()}
h = {q: 0.5 + 0.1 * v for q, v in x.items()}
s = cluster_ci({"u": u, "t": t, "h": h}, authors, lambda m: (m["h"] - m["u"]) / (m["t"] - m["u"]))
assert abs(s["point"] - 0.5) < 1e-9 and abs(s["ci_low"] - 0.5) < 1e-9 and abs(s["ci_high"] - 0.5) < 1e-9

# unequal author sizes: the mean is over questions, not a mean of author means
uneven = {"a0": 0, "a1": 0, "a2": 0, "b0": 1}
p = cluster_ci({"m": {"a0": 0.0, "a1": 0.0, "a2": 0.0, "b0": 1.0}}, {k: k[0] for k in uneven})
assert p["point"] == 0.25

# only ids shared by every model count; a bad call fails loudly
assert cluster_ci({"a": x, "b": {"q0_0": 1.0}}, authors, lambda m: m["b"] - m["a"])["n"] == 1
for bad in (lambda: cluster_ci({"a": x, "b": x}, authors), lambda: cluster_ci({"a": {"z": 1.0}, "b": {"y": 1.0}}, {}, lambda m: m["a"])):
    try:
        bad()
    except AssertionError:
        continue
    raise SystemExit("cluster_ci accepted a bad call")
print("cluster_ci: all checks passed")
