"""Self-check for paired_ci. Run: python addons/test_paired_ci.py (needs numpy only)."""
from paired_ci import paired_ci

ids = [f"s{i}" for i in range(100)]
base = {i: 1.0 for i in ids}
cand = {i: (0.0 if k < 40 else 1.0) for k, i in enumerate(ids)}  # 40 of 100 stop refusing

r = paired_ci(base, cand)
assert r["n"] == 100 and abs(r["delta"] + 0.4) < 1e-12, r
assert r["ci_high"] < 0, r  # the drop AuditKIT's paired_bootstrap calls p 0.52 is clearly excluded from 0
assert -0.55 < r["ci_low"] < r["ci_high"] < -0.25, r

r = paired_ci(base, base)
assert r["delta"] == r["ci_low"] == r["ci_high"] == 0, r

# pairs are matched by id, not by position; ids in only one dict are dropped
r = paired_ci({"a": 1, "b": 0, "x": 1}, {"b": 0, "a": 1, "y": 0})
assert r["n"] == 2 and r["delta"] == 0, r

assert paired_ci(base, cand) == paired_ci(base, cand)  # same seed, same interval
assert paired_ci({}, {})["delta"] is None
print("paired_ci: all checks passed")
