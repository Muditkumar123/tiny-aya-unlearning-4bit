"""CPU self-check for r5_tables.py on synthetic rows.  Run: python evaluation/test_r5_tables.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import r5_tables as rt  # noqa: E402

tmp = Path(tempfile.mkdtemp())


def put(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


MM = {"original": [1, 1, 0, 1], "plain-q4": [1, 0, 0, 0], "random30-q4": [1, 1, 1, 1], "protected-q4": [0, 0, 0, 0]}
for k, ref in (("original", [1, 0, 0, None]), ("plain-q4", [1, 1, 0, 0]), ("random30-q4", [0, 0, 1, 1]),
               ("protected-q4", [1, 1, None, None])):
    e = tmp / "eval" / k
    put(e / "xstest_safe.jsonl", [{"id": f"x{i}", "type": t, "group": "English", "refusal": v, "response": "w " * n}
                                  for i, (t, v, n) in enumerate(zip(["a", "a", "b", "b"], ref, [256, 10, 300, 5]))])
    put(e / "harmbench_heldout.jsonl", [{"id": "h0", "group": "English", "refusal": 1, "response": "short"}])
    put(e / "aya_redteam_hi_ar.jsonl", [{"id": "y0", "group": "Hindi", "refusal": 0, "response": "w " * 256},
                                        {"id": "y1", "group": "Arabic", "refusal": 1, "response": "w"}])
    (e / "summary.json").write_text("{}")
    put(tmp / "mmlu" / k / "mmlu_en.jsonl", [{"id": f"q{i}", "correct": c} for i, c in
                                             enumerate(MM[k])])

res = rt.tables(tmp, n_tokens=lambda text: len(text.split()))
assert res["labels"] == ["original", "plain-q4", "random30-q4", "protected-q4"]
assert res["xstest_types"]["original"] == {"a": {"n_total": 2, "n_judged": 2, "refusal_rate": 0.5},
                                          "b": {"n_total": 2, "n_judged": 1, "refusal_rate": 0.0}}  # None not counted as judged
assert res["xstest_types"]["protected-q4"]["b"] == {"n_total": 2, "n_judged": 0, "refusal_rate": None}  # zero judged kept
assert res["xstest_types"]["plain-q4"]["a"]["refusal_rate"] == 1.0
assert res["capped"]["original"]["xstest_safe/English"] == {"n": 4, "capped_estimate": 2}  # 256 and 300 words reach the cap
assert res["capped"]["original"]["aya_redteam_hi_ar/Hindi"]["capped_estimate"] == 1
assert res["capped"]["original"]["aya_redteam_hi_ar/Arabic"]["capped_estimate"] == 0
m = res["mmlu"]["mmlu_en"]
assert m["original"]["accuracy"]["point"] == 0.75 and "delta_pp_vs_original" not in m["original"]
assert abs(m["plain-q4"]["delta_pp_vs_original"]["point"] - (-50.0)) < 1e-9
md = (tmp / "tables.md").read_text()
assert "| a | 0.50 (2/2) | 1.00 (2/2) | 0.00 (2/2) | 1.00 (2/2) |" in md  # per-model judged / total
assert "| b | 0.00 (1/2) | 0.00 (2/2) | 1.00 (2/2) | – (0/2) |" in md  # unequal and zero coverage shown
assert "Estimated answers at the 256-token cap" in md and "re-tokenizes" in md
# S1-S3: A - B over prompts judged in both (None dropped); MMLU in pp
s1, s2 = res["contrasts"]["S1"]["cells"], res["contrasts"]["S2"]["cells"]
assert s1["xstest_safe/English"]["n"] == 2 and s1["xstest_safe/English"]["delta"] == 1.0
assert s2["xstest_safe/English"]["delta"] == 0.0
assert s1["mmlu_en_pp"]["point"] == -100.0 and res["contrasts"]["S3"]["a"] == "random30-q4"
assert "**S1 = protected-q4 − random30-q4**" in (tmp / "tables.md").read_text()
# rows that carry n_tokens_estimate (the public export) give the same cap counts without any answer text
for k in res["labels"]:
    for f in (tmp / "eval" / k).glob("*.jsonl"):
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        put(f, [{**{a: b for a, b in r.items() if a != "response"}, "n_tokens_estimate": len(r["response"].split())} for r in rows])
again = rt.tables(tmp, n_tokens=lambda text: 1 / 0)  # would fail if any text were tokenized
assert again["capped"] == res["capped"] and again["contrasts"] == res["contrasts"]
print("r5_tables: all checks passed")
