"""Study S utility (PLAN section 3): Global-MMLU en500 and hi200 for one model, scored as tofu_eval.py scores them.

    CUDA_VISIBLE_DEVICES=0 python evaluation/mmlu_only.py original --out runs/mmlu      # first: must reproduce REF
    CUDA_VISIBLE_DEVICES=0 python evaluation/mmlu_only.py plain-q4 <checkpoint> --out runs/mmlu
    options: --out DIR (required; <out>/<label> must not exist yet)  --ref DIR  --batch-size 16

Same rules as tofu_eval.py: original tokenizer, Tiny Aya chat template, InMemoryHFModel(fp32_logits=True,
eos_token_id=STOP), batch 16; per question the mean log-prob of "A".."D" as the answer, prediction = argmax.
Writes <out>/<label>/{mmlu_en,mmlu_hi}.jsonl in tofu_eval's row format, summary.json and manifest.json.
"original" is compared with REF, the shipped numeric original rows in results/study_u/original/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from auditkit.model import Request

sys.path.insert(0, str(Path(__file__).parent))
import tofu_eval as te  # noqa: E402
from evaluate import ROOT, check_manifest, load, read_jsonl, weights_id, write_jsonl  # noqa: E402

REF = ROOT / "results/study_u/original"


def check_reference(full, ref, s):
    """Same ids in the same order, max |lp diff| < 1e-6 and identical predictions; returns the max |lp diff|."""
    assert [r["id"] for r in ref] == [r["id"] for r in full], f"{s}: other questions or order than the reference"
    gap = max(abs(a - b) for x, y in zip(ref, full) for a, b in zip(x["lp"], y["lp"]))
    assert gap < 1e-6 and [r["pred"] for r in ref] == [r["pred"] for r in full], f"{s}: max |lp diff| {gap}"
    return gap


def score(ak, label, out, ref_dir=REF):
    d = out / label
    assert not d.exists(), f"{d} exists; use a fresh --out"  # checked first: never write over earlier rows
    data = {s: rows for s, rows in te.load_sets().items() if s.startswith("mmlu")}
    spec = te.run_spec(ak, data)
    with torch.no_grad():
        probe = ak._pipeline.model(input_ids=torch.tensor([[2]], device=ak._pipeline.model.device)).logits
    assert probe.dtype == torch.float32, "wrap the model with InMemoryHFModel(..., fp32_logits=True)"
    assert spec["stop_ids"] == te.STOP, "wrap the model with InMemoryHFModel(..., eos_token_id=STOP)"
    check_manifest(d, spec)
    sets = {}
    for s, rows in data.items():
        render = [ak._render_prompt(Request(prompt=r["prompt"])) for r in rows]
        lp = iter(te.mean_logprobs(ak, [(p, x) for p in render for x in te.LETTERS]))
        full = []
        for r in rows:
            v = [next(lp) for _ in te.LETTERS]
            pred = te.LETTERS[int(np.argmax(v))]
            full.append({"id": r["id"], "subject": r["subject"], "answer": r["answer"], "lp": v, "pred": pred,
                         "correct": int(pred == r["answer"])})
        write_jsonl(d / f"{s}.jsonl", full)
        sets[s] = {"n": len(full), "accuracy": float(np.mean([r["correct"] for r in full]))}
        if label == "original":  # the check that this script scores exactly as tofu_eval did
            sets[s]["max_abs_lp_diff_vs_ref"] = check_reference(full, read_jsonl(ref_dir / f"{s}.jsonl"), s)
    summary = {"label": label, "size_mb": ak.model_info()["size_mb"], "spec": spec, "sets": sets}
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=REF)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    assert (a.checkpoint is None) == (a.label == "original"), "give a checkpoint for every label but 'original'"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(te.ORIG)  # the original tokenizer for every variant
    ak = te.InMemoryHFModel(load(a.checkpoint), tok, name=f"tiny-aya-{a.label}@{weights_id(a.checkpoint)}",
                            batch_size=a.batch_size, fp32_logits=True, eos_token_id=te.STOP)
    print(json.dumps(score(ak, a.label, a.out, a.ref)["sets"], indent=1))
