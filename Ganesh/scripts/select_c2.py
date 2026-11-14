"""Apply the frozen C2 selection rule (Ganesh/configs/c2.yaml, selection) to the grid points scored so far.

Retention = (mean(M) - mean(original)) / (mean(C0 v2) - mean(original)) on ROUGE-L recall, point estimates.
Zero and one points come from Vedika's frozen harness output (Vedika/out/tofu_v2, stop [6, 3]).
Exit 0: C2 chosen (written to Ganesh/out/c2_selection.json). Exit 3: score the next lr. Exit 4: stop, no C2.
"""
import json, sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(open("Ganesh/configs/c2.yaml"))
grid = cfg["lr_grid"]
ref = Path("Vedika/out/tofu_v2")
grid_out = Path("Ganesh/out/tofu_c2grid")


def means(summary_path):
    s = json.load(open(summary_path))["sets"]
    return s["forget"]["rougeL_recall"], s["retain"]["rougeL_recall"]


def retention(m, zero, one):
    return (m - zero) / (one - zero)


(f0, r0), (f1, r1) = means(ref / "original/summary.json"), means(ref / "c0v2/summary.json")
assert f1 > f0 and r1 > r0, "C0 v2 must score above the original"
assert retention(f0, f0, f1) == 0 and retention(f1, f0, f1) == 1

rows = []
for lr in grid:
    p = grid_out / f"c2-lr{lr:.0e}/summary.json"
    if not p.exists():
        break
    f, r = means(p)
    rows.append({"lr": lr, "forget_rougeL": f, "retain_rougeL": r,
                 "forget_retention": retention(f, f0, f1), "retain_retention": retention(r, r0, r1)})

out = {"zero": {"forget": f0, "retain": r0}, "one": {"forget": f1, "retain": r1}, "grid": rows, "rule": cfg["selection"]["rule"]}
ok = [x for x in rows if x["forget_retention"] <= 0.25 and x["retain_retention"] >= 0.50]
code = 3
if ok:  # rows are in ascending lr order and contiguous from the smallest, so ok[0] is the smallest qualifying lr
    out.update(chosen_lr=ok[0]["lr"], target_met=True)
    code = 0
elif len(rows) == len(grid):
    usable = [x for x in rows if x["retain_retention"] >= 0.50]
    if usable:
        out.update(chosen_lr=min(usable, key=lambda x: x["forget_retention"])["lr"], target_met=False)
        code = 0
    else:
        out.update(chosen_lr=None, target_met=False)
        code = 4
json.dump(out, open("Ganesh/out/c2_selection.json", "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(code)
