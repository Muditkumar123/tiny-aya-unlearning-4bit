"""Build the Plan B evaluation sets in common/data/ (CPU only, ~10 MB of downloads at pinned revisions).

    ~/miniconda3/envs/lexsi/bin/python Vedika/scripts/build_tofu_sets.py [--out DIR]
    (--out must already hold tofu_author_split.json from build_prompt_sets.py)

Same format and rules as build_prompt_sets.py: {"meta", "rows"}, files opened with mode "x" (never overwritten).
- tofu_forget10_perturbed.json: paraphrased + 5 perturbed answers for each row of tofu_author_split.json (same ids)
- tofu_retain_perturbed.json: TOFU's retain evaluation set (full rows 0-399, authors 0-19), with answers for the truth ratio
- global_mmlu_en500.json / global_mmlu_hi200.json: fixed Global-MMLU test subsets; hi200 = the first 200 sampled ids
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=ROOT / "common" / "data")
OUT = ap.parse_args().out
REV = {"locuslab/TOFU": "324592d84ae4f482ac7249b9285c2ecdb53e3a68",  # the revision of tofu_author_split.json and C0
       "CohereLabs/Global-MMLU": "0e619dbeb34206cd48705a1a0ea7fb21cae09993"}
SEED = 0
MMLU_PROMPT = ("Answer the following multiple-choice question with the letter of the correct option only.\n\n"
               "{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}")


def get(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", revision=REV[repo])


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def meta(repo, files, filter_, count, seed=None):
    return {"source": repo, "revision": REV[repo], "files": files, "filter": filter_, "seed": seed, "count": count}


sets = {}
full, f10p, rp = (jsonl(get("locuslab/TOFU", f)) for f in ("full.json", "forget10_perturbed.json", "retain_perturbed.json"))

# forget10: perturbed answers joined to the fixed author split by row order, checked by question and answer
split = json.load(open(OUT / "tofu_author_split.json"))["rows"]
assert len(f10p) == len(split) == 400
assert all(s["prompt"] == p["question"] and s["answer"] == p["answer"] for s, p in zip(split, f10p))
assert all(len(p["perturbed_answer"]) == 5 for p in f10p)
sets["tofu_forget10_perturbed.json"] = {
    "meta": meta("locuslab/TOFU", ["forget10_perturbed.json"],
                 "all rows; joined to tofu_author_split.json by row order (question and answer asserted equal)", 400),
    "rows": [{"id": s["id"], "paraphrased_answer": p["paraphrased_answer"], "perturbed_answer": p["perturbed_answer"]}
             for s, p in zip(split, f10p)]}

# retain: each row located in full.json by (question, answer); one question text occurs twice in full (rows 100, 440)
at = {(r["question"], r["answer"]): i for i, r in enumerate(full)}
rows = [at[(r["question"], r["answer"])] for r in rp]
assert rows == list(range(400)), "retain_perturbed is expected to be full rows 0-399"
assert all(len(r["perturbed_answer"]) == 5 for r in rp)
sets["tofu_retain_perturbed.json"] = {
    "meta": {**meta("locuslab/TOFU", ["full.json", "retain_perturbed.json"],
                    "all rows; full row found by (question, answer); author = full row // 20", 400),
             "authors": list(range(20))},
    "rows": [{"id": f"tofu_{i}", "prompt": r["question"], "answer": r["answer"], "author": i // 20,
              "paraphrased_answer": r["paraphrased_answer"], "perturbed_answer": r["perturbed_answer"]}
             for i, r in zip(rows, rp)]}
assert not {r["author"] for r in sets["tofu_retain_perturbed.json"]["rows"]} & {s["author"] for s in split}

# Global-MMLU: one seeded sample of test rows; Hindi uses the first 200 of the same ids
en, hi = (pd.read_parquet(get("CohereLabs/Global-MMLU", f"{lang}/test-00000-of-00001.parquet")) for lang in ("en", "hi"))
assert len(en) == len(hi) == 14042 and (en.sample_id.values == hi.sample_id.values).all()
pick = random.Random(SEED).sample(range(len(en)), 500)


def mmlu(df, idx):
    return [{"id": r.sample_id, "prompt": MMLU_PROMPT.format(**r._asdict()), "answer": r.answer, "subject": r.subject,
             "cultural_sensitivity_label": r.cultural_sensitivity_label} for r in df.iloc[idx].itertuples(index=False)]


for lang, df, n in (("en", en, 500), ("hi", hi, 200)):
    got = mmlu(df, pick[:n])
    assert {r["answer"] for r in got} <= set("ABCD") and len({r["id"] for r in got}) == n
    sets[f"global_mmlu_{lang}{n}.json"] = {
        "meta": {**meta("CohereLabs/Global-MMLU", [f"{lang}/test-00000-of-00001.parquet"],
                        f"random.Random(0).sample(range(14042), 500), first {n}; prompt = meta.prompt_template", n, SEED),
                 "prompt_template": MMLU_PROMPT},
        "rows": got}

for name, obj in sets.items():  # all asserts passed; now write
    with open(OUT / name, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print(name, len(obj["rows"]), hashlib.md5((OUT / name).read_bytes()).hexdigest())
