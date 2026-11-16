"""CPU check for mmlu_only.py from the release layout (no model, no GPU).
    CUDA_VISIBLE_DEVICES="" python evaluation/test_mmlu_only.py
Needs the shipped export (results/) and the rebuilt data/prompts/ files."""
import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mmlu_only as mo  # noqa: E402

# the default reference is the shipped numeric export, inside the repository, in the order the scorer produces
assert mo.REF == mo.ROOT / "results/study_u/original" and mo.REF.is_dir()
sets = {s: rows for s, rows in mo.te.load_sets().items() if s.startswith("mmlu")}
for s, rows in sets.items():
    ref = mo.read_jsonl(mo.REF / f"{s}.jsonl")
    assert [r["id"] for r in ref] == [r["id"] for r in rows], f"{s}: shipped reference order differs from the data"
    assert all(len(r["lp"]) == 4 and r["pred"] in "ABCD" for r in ref)
    assert mo.check_reference(copy.deepcopy(ref), ref, s) == 0.0  # identical rows pass with gap 0


def must_fail(full, ref):
    try:
        mo.check_reference(full, ref, "mmlu_en")
    except AssertionError:
        return
    raise SystemExit("check_reference accepted a difference")


ref = mo.read_jsonl(mo.REF / "mmlu_en.jsonl")
lp = copy.deepcopy(ref); lp[3]["lp"][0] += 1e-3
pred = copy.deepcopy(ref); pred[7]["pred"] = "D" if pred[7]["pred"] != "D" else "A"
order = copy.deepcopy(ref); order[0], order[1] = order[1], order[0]
for bad in (lp, pred, order, ref[:-1]):
    must_fail(bad, ref)

# an existing output folder is refused before any model is touched
out = Path(tempfile.mkdtemp())
(out / "original").mkdir()
try:
    mo.score(None, "original", out)
    raise SystemExit("score wrote into an existing folder")
except AssertionError as e:
    assert "fresh --out" in str(e)
print("mmlu_only: all checks passed")
