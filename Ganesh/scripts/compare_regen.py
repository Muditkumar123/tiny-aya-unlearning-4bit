"""R2 generation check (PLAN section 4): fresh greedy answers vs the saved predecessor answers, exact text match.

The fresh run is Vedika/scripts/tofu_eval.py --limit 32 (two full batches of 16, the saved run's batch composition);
the plan's 20-question subset is the first 20 rows. Log-probs are compared descriptively (max abs diff), not gated.

python Ganesh/scripts/compare_regen.py <fresh label dir> <saved label dir> [--out f.json]
Exit 0 = all generated answers match exactly in the first 20 rows of forget and retain, 1 = otherwise.
"""
import argparse, json, sys


def rows(path):
    return [json.loads(l) for l in open(path)]


def compare(fresh, saved, first=20):
    out = {"fresh": fresh, "saved": saved, "first": first, "sets": {}}
    for s in ("forget", "retain"):
        f, g = rows(f"{fresh}/cache/{s}.gen.jsonl"), rows(f"{saved}/cache/{s}.gen.jsonl")
        assert [r["id"] for r in f] == [r["id"] for r in g[:len(f)]], f"{s}: fresh rows are not a prefix of the saved rows"
        same = [a["response"] == b["response"] for a, b in zip(f, g)]
        out["sets"][s] = {"n_fresh": len(f), "match_first": sum(same[:first]), "match_all": sum(same),
                          "mismatched_ids": [a["id"] for a, ok in zip(f, same) if not ok]}
    for s in ("forget", "retain", "mmlu_en", "mmlu_hi"):
        f, g = rows(f"{fresh}/cache/{s}.lp.jsonl"), rows(f"{saved}/cache/{s}.lp.jsonl")
        assert [r["id"] for r in f] == [r["id"] for r in g[:len(f)]], f"{s}: fresh lp rows are not a prefix of the saved rows"
        keys = [k for k in f[0] if k != "id"]
        flat = lambda r: [x for k in keys for x in (r[k] if isinstance(r[k], list) else [r[k]])]  # noqa: E731
        assert all(len(flat(x)) == len(flat(y)) for x, y in zip(f, g)), f"{s}: lp list lengths differ"
        out["sets"].setdefault(s, {})["lp_max_abs_diff"] = max(abs(a - b) for x, y in zip(f, g) for a, b in zip(flat(x), flat(y)))
    out["pass"] = all(out["sets"][s]["match_first"] == first for s in ("forget", "retain"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fresh"); ap.add_argument("saved"); ap.add_argument("--out")
    a = ap.parse_args()
    r = compare(a.fresh, a.saved)
    print(json.dumps(r, indent=1))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    sys.exit(0 if r["pass"] else 1)
