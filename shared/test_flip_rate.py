"""Self-check for flip_rate. Run: python addons/test_flip_rate.py (needs auditkit importable)."""
from types import SimpleNamespace

from flip_rate import flip_rate


def run(scores, direction="minimize"):
    """Fake RunResult: {sample_id: value} for guard_judge (1.0 = unsafe)."""
    preds = [SimpleNamespace(sample_id=sid, metadata={"scores": [
        {"name": "guard_judge", "value": v, "direction": direction}]}) for sid, v in scores.items()]
    return SimpleNamespace(predictions=preds)


base = run({"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.0, "only_base": 0.0})
quant = run({"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0, "only_quant": 1.0})
r = flip_rate(base, quant, "guard_judge")
assert r["n"] == 4, r
assert r["worse_ids"] == ["b", "d"], r  # safe -> unsafe is worse for a MINIMIZE metric
assert r["better_ids"] == ["c"], r
assert r["flip_rate"] == 0.75, r

# MAXIMIZE metric: 1.0 is good, so 1 -> 0 is worse
r = flip_rate(run({"a": 1.0, "b": 0.0}, "maximize"), run({"a": 0.0, "b": 0.0}, "maximize"), "guard_judge")
assert (r["to_worse"], r["to_better"], r["flip_rate"]) == (1, 0, 0.5), r

# a thresholded score's `passed` wins over the raw value
p = run({"a": 0.9}, "maximize")
p.predictions[0].metadata["scores"][0]["passed"] = False
assert flip_rate(run({"a": 0.9}, "maximize"), p, "guard_judge")["worse_ids"] == ["a"]

assert flip_rate(run({}), run({}), "guard_judge")["flip_rate"] is None
print("flip_rate: all checks passed")
