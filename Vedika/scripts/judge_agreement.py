"""R5 step 1, the frozen judge gate (common/CHANGELOG.md 2026-11-12 01:49).

Do AuditKIT's public GuardJudge (WildGuard profile, template without its leading literal "<s>"; evaluate.judge) and
SafeTune's private _run_hf_judge (evaluate.judge_safetune) give the same response_refusal label on 20 fixed pairs?
20/20 -> path (a), else (b). The gate ran 2026-11-12 19:46 with this file's first version (md5 f42b7a0f), whose
GuardJudge code has since moved, unchanged in its requests, into evaluate.judge.

    python Vedika/scripts/judge_agreement.py pick                        # CPU, once: writes the 20 fixed pairs
    CUDA_VISIBLE_DEVICES=1 python Vedika/scripts/judge_agreement.py run  # GPU: both judges, writes result.json

Pairs: the first 10 XSTest-safe prompts and the first 5 Hindi + first 5 Arabic Aya prompts (file order), each with
the original model's saved answer from Vedika/out/eval/original/cache/. No HarmBench answers. Lab only: the
pairs and judge outputs stay in Vedika/out/judge_agreement/ and are never published.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import ROOT, WILDGUARD, judge, judge_safetune, md5, read_jsonl  # noqa: E402

OUT = ROOT / "Vedika/out/judge_agreement"
ANS = ROOT / "Vedika/out/eval/original/cache"
PICK = {"xstest_safe": {None: 10}, "aya_redteam_hi_ar": {"Hindi": 5, "Arabic": 5}}


def pick():
    pairs = []
    for s, langs in PICK.items():
        rows = json.load(open(ROOT / f"common/data/{s}.json"))["rows"]
        ans = {a["id"]: a["response"] for a in read_jsonl(ANS / f"{s}.answers.jsonl")}
        for lang, n in langs.items():
            got = [r for r in rows if lang is None or r["language"] == lang][:n]
            pairs += [{"set": s, "id": r["id"], "prompt": r["prompt"], "response": ans[r["id"]]} for r in got]
    assert len(pairs) == 20 and len({p["id"] for p in pairs}) == 20
    OUT.mkdir(parents=True, exist_ok=True)
    src = {s: md5((ANS / f"{s}.answers.jsonl").read_bytes()) for s in PICK}
    with open(OUT / "pairs.json", "x", encoding="utf-8") as fh:  # fixed before any judge runs; never overwritten
        json.dump({"rule": PICK, "answers_md5": src, "pairs": pairs}, fh, ensure_ascii=False, indent=1)
    print("pairs.json", md5((OUT / "pairs.json").read_bytes()), [p["id"] for p in pairs])


def run():
    import torch
    pf = OUT / "pairs.json"
    pairs = [(p["prompt"], p["response"]) for p in json.load(open(pf))["pairs"]]
    ids = [p["id"] for p in json.load(open(pf))["pairs"]]
    from auditkit.model import AutoModel
    a = judge_safetune(pairs)  # path (b)
    torch.cuda.empty_cache()
    model = AutoModel.resolve(f"hf:{WILDGUARD}", dtype=torch.bfloat16)
    b = judge(pairs, model)  # path (a), evaluate.py's judge
    dtype = str(model._pipeline.model.dtype)
    agree = [x["refusal"] is not None and x["refusal"] == y["refusal"] for x, y in zip(a, b)]
    res = {"pairs_md5": md5(pf.read_bytes()), "n": len(pairs), "refusal_agree": sum(agree),
           "path": "a" if all(agree) else "b", "auditkit_dtype": dtype,
           "rows": [{"id": i, "safetune": x, "auditkit": y} for i, x, y in zip(ids, a, b)]}
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"refusal labels agree {sum(agree)}/{len(pairs)} -> path ({res['path']}); auditkit dtype {dtype}")


if __name__ == "__main__":
    {"pick": pick, "run": run}[sys.argv[1]]()
