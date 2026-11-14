"""Build the fixed prompt sets in common/data/ (CPU only, ~2 MB of downloads at pinned revisions).

    ~/miniconda3/envs/lexsi/bin/python Vedika/scripts/build_prompt_sets.py [--out DIR] [--discovery FILE]

Every file is {"meta": {...}, "rows": [{"id", "prompt", ...}]}. Files are opened with mode "x", so a
rerun never overwrites a posted set (a new version must be a new file name). A rebuild into an empty --out gives
files byte-identical to common/data/.
The 32 discovery pairs come from the text-free id file (default Mudit/release/discovery_ids.json): HarmBench rows and
Alpaca rows at pinned revisions, each checked by the sha256 of its text. The meta text still names the lab file the
sets were first built from, so the files stay identical.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=ROOT / "common" / "data")
ap.add_argument("--discovery", type=Path, default=ROOT / "Mudit/release/discovery_ids.json")
ARGS = ap.parse_args()
OUT = ARGS.out
DISC = json.load(open(ARGS.discovery))
REV = {  # pinned dataset revisions
    "Paul/XSTest": "f600c994b256f12867dfa5b3eb3d545a3e62f8b5",
    "CohereLabs/aya_redteaming": "5a16fee03c19d681b4f3527b6a750e1ed4d434c7",
    "locuslab/TOFU": "324592d84ae4f482ac7249b9285c2ecdb53e3a68",
    "walledai/HarmBench": "fb6c2afd5a2a943d701d6db3efab87d077e81be5",  # the revision Mudit's circuit run used
}
SEED = 0


def get(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", revision=REV[repo])


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def meta(repo, files, filter_, count, seed=None):
    return {"source": repo, "revision": REV[repo], "files": files, "filter": filter_, "seed": seed, "count": count}


sets = {}

# XSTest safe prompts: over-refusal set
xs = list(csv.DictReader(open(get("Paul/XSTest", "xstest_prompts.csv"), encoding="utf-8")))
assert len(xs) == 450
safe = [{"id": f"xstest_{r['id']}", "prompt": r["prompt"], "type": r["type"], "focus": r["focus"]}
        for r in xs if r["label"] == "safe"]
assert len(safe) == 250
sets["xstest_safe.json"] = {"meta": meta("Paul/XSTest", ["xstest_prompts.csv"], "label == 'safe'", 250), "rows": safe}

# Aya red-teaming: 50 Hindi + 50 Arabic harmful prompts
aya_rows, aya_files = [], []
for lang, code in (("Hindi", "hin"), ("Arabic", "arb")):
    f = f"aya_{code}.jsonl"
    aya_files.append(f)
    seen, uniq = set(), []
    for i, r in enumerate(jsonl(get("CohereLabs/aya_redteaming", f))):
        assert r["language"] == lang
        if r["prompt"] not in seen:  # exact-duplicate prompts appear in the source
            seen.add(r["prompt"])
            uniq.append((i, r))
    for i, r in sorted(random.Random(SEED).sample(uniq, 50), key=lambda x: x[0]):
        aya_rows.append({"id": f"aya_{code}_{i}", "prompt": r["prompt"], "language": lang,
                         "harm_category": json.loads(r["harm_category"]), "global_or_local": r["global_or_local"],
                         "literal_translation": r["literal_translation"]})
sets["aya_redteam_hi_ar.json"] = {
    "meta": meta("CohereLabs/aya_redteaming", aya_files,
                 "exact-duplicate prompts dropped per language (first kept), then random.Random(0).sample(50) per "
                 "language, kept in source order; id suffix = source line index", 100, SEED),
    "rows": aya_rows}

# TOFU forget10: 20 authors x 20 QA, split 10 discovery / 10 evaluation by author
full = jsonl(get("locuslab/TOFU", "full.json"))
f10 = jsonl(get("locuslab/TOFU", "forget10.json"))
assert len(full) == 4000 and f10 == full[3600:], "forget10 must be the last 400 rows of full"
authors = sorted({i // 20 for i in range(3600, 4000)})  # TOFU stores each author's 20 QA pairs contiguously
assert len(authors) == 20
disc = set(random.Random(SEED).sample(authors, 10))
tofu = [{"id": f"tofu_{3600 + j}", "prompt": r["question"], "answer": r["answer"], "author": (3600 + j) // 20,
         "split": "discovery" if (3600 + j) // 20 in disc else "evaluation"} for j, r in enumerate(f10)]
assert sum(r["split"] == "discovery" for r in tofu) == 200
sets["tofu_author_split.json"] = {
    "meta": {**meta("locuslab/TOFU", ["full.json", "forget10.json"],
                    "forget10 (= full rows 3600-3999); author = full row // 20 (authors 180-199); "
                    "random.Random(0).sample(authors, 10) -> discovery, rest -> evaluation", 400, SEED),
             "discovery_authors": sorted(disc), "evaluation_authors": sorted(set(authors) - disc)},
    "rows": tofu}

# HarmBench standard minus the 32 circuit-discovery prompts
hb = pd.read_parquet(get("walledai/HarmBench", "standard/train-00000-of-00001.parquet"))
assert len(hb) == 200
pairs = DISC["pairs"]
sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()  # noqa: E731
assert DISC["harmbench"]["revision"] == REV["walledai/HarmBench"]
assert all(sha(hb["prompt"][p["harmbench_row"]]) == p["harmbench_sha256"] for p in pairs), "HarmBench text differs"
used = {hb["prompt"][p["harmbench_row"]] for p in pairs}
assert len(used) == 32
used_rows = sorted(i for i, p in enumerate(hb["prompt"]) if p in used)
assert used_rows == sorted(p["harmbench_row"] for p in pairs) == list(range(32)), used_rows
held = [{"id": f"hb_std_{i}", "prompt": r.prompt, "category": r.category}
        for i, r in enumerate(hb.itertuples()) if r.prompt not in used]
assert len(held) == 168 and len({r["prompt"] for r in held}) == 168
sets["harmbench_heldout.json"] = {
    "meta": {**meta("walledai/HarmBench", ["standard/train-00000-of-00001.parquet"],
                    "config standard, split train, minus the 32 prompts in "
                    "Mudit/out/refusal_circuit_matched/discovery_prompts.json (rows 0-31); id suffix = row", 168),
             "excluded_rows": used_rows},
    "rows": held}

# the 32 Alpaca discovery partners must not sit in any harmless set
A = DISC["alpaca"]  # alpaca_row indexes the whole train file
al = pd.read_parquet(hf_hub_download(A["repo"], A["file"], repo_type="dataset", revision=A["revision"]))
alpaca = {al[A["column"]][p["alpaca_row"]] for p in pairs}
assert all(sha(al[A["column"]][p["alpaca_row"]]) == p["alpaca_sha256"] for p in pairs), "Alpaca text differs"
assert len(alpaca) == 32 and not alpaca & {r["prompt"] for r in safe}

OUT.mkdir(parents=True, exist_ok=True)  # a fresh checkout has no common/data/; files are still never overwritten
for name, obj in sets.items():  # all asserts passed; now write
    with open(OUT / name, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print(name, len(obj["rows"]), hashlib.md5((OUT / name).read_bytes()).hexdigest())
