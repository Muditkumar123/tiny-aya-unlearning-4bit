"""Write an ID-only manifest of the 32 circuit-discovery pairs: pinned source rows plus sha256 of each text, no text.

Reads the lab file Mudit/out/refusal_circuit_matched/discovery_prompts.json and the pinned local HF caches, checks
that every saved text equals its source row, and writes JSON a clean checkout can use to rebuild the exclusions.

python Mudit/scripts/export_discovery_ids.py [--out Mudit/release/discovery_ids.json]
"""
import argparse, hashlib, json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HUB = Path.home() / ".cache/huggingface/hub"
HB = ("walledai/HarmBench", "fb6c2afd5a2a943d701d6db3efab87d077e81be5", "standard/train-00000-of-00001.parquet")
AL = ("tatsu-lab/alpaca", "dce01c9b08f87459cf36a430d809084718273017", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cached(repo, rev, file):
    return HUB / f"datasets--{repo.replace('/', '--')}" / "snapshots" / rev / file


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "Mudit/release/discovery_ids.json")
    a = ap.parse_args(argv)
    src = ROOT / "Mudit/out/refusal_circuit_matched/discovery_prompts.json"
    pairs = json.load(open(src))["pairs"]
    hb = pd.read_parquet(cached(*HB))
    al = pd.read_parquet(cached(*AL))
    al_rows = {}
    for i, r in enumerate(al.itertuples()):
        if not r.input:  # the discovery partners were drawn from empty-input rows
            al_rows.setdefault(r.instruction, []).append(i)
    out = []
    for p in pairs:
        i = p["harmbench_index"]
        assert hb["prompt"][i] == p["harmful_prompt"], f"pair {p['pair']}: HarmBench row {i} differs"
        rows = al_rows.get(p["alpaca_prompt"], [])
        assert len(rows) == 1, f"pair {p['pair']}: {len(rows)} empty-input Alpaca rows match"
        out.append({"pair": p["pair"], "harmbench_row": i, "harmbench_category": p["harmbench_category"],
                    "harmbench_sha256": h(p["harmful_prompt"]), "alpaca_row": rows[0],
                    "alpaca_sha256": h(p["alpaca_prompt"]), "templated_tokens": p["templated_tokens"]})
    assert len(out) == 32 and sorted(r["harmbench_row"] for r in out) == list(range(32))
    assert len({r["alpaca_row"] for r in out}) == 32
    doc = {"what": "IDs of the 32 harmful/benign pairs used for EAP-IG refusal-mask discovery; no prompt text",
           "harmbench": {"repo": HB[0], "revision": HB[1], "file": HB[2], "column": "prompt"},
           "alpaca": {"repo": AL[0], "revision": AL[1], "file": AL[2], "column": "instruction", "filter": "input == ''"},
           "hash": "sha256 of the UTF-8 prompt text",
           "source_file_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
           "pairs": out}
    a.out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {a.out}: {len(out)} pairs, HarmBench rows 0-31, {len({r['alpaca_row'] for r in out})} Alpaca rows")


if __name__ == "__main__":
    main()
