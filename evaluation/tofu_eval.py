"""Plan B TOFU evaluation: score one Tiny Aya variant on TOFU forget10, the TOFU retain set and Global-MMLU.

    CUDA_VISIBLE_DEVICES=0 python evaluation/tofu_eval.py original                       # untuned = retention 0
    CUDA_VISIBLE_DEVICES=0 python evaluation/tofu_eval.py c0v2 runs/training/c0v2-target-merged --one c0v2
    python evaluation/tofu_eval.py --table --one c0v2      # CPU: scaled retention and CIs for every label in --out
    options: --out DIR (default runs/evaluation/tofu_v2)  --limit N (smoke only, use its own --out)  --batch-size 16

All scoring goes through AuditKIT on shared/inmemory_model.py (greedy generate + teacher-forced loglikelihood),
wrapped with fp32_logits=True so log-probs are not rounded to bf16, and eos_token_id=STOP so answers end at the
turn (evaluate() refuses a model without either).
Per label, <out>/<label>/ holds forget.jsonl, retain.jsonl (one row per question), mmlu_en.jsonl, mmlu_hi.jsonl
(one row per question), summary.json, manifest.json, cache/. Stages resume from cache/; manifest.json refuses
a rerun with another checkpoint, setting or prompt set (evaluate.check_manifest). The table (results.md,
table.json) is rebuilt after every run, with scaled retention once the --zero and --one labels exist.

Metrics (definitions: the original private experiment log, 2026-11-07):
- rougeL_recall: rouge_score rougeL recall, use_stemmer, score(gold, generated); TOFU's eval_rouge_recall
- lp_answer: mean per-token log-prob of the gold answer tokens after the chat template (no end-of-turn tokens)
- truth_ratio R = exp(lp_para - mean(lp_pert)) per question (TOFU's curr_stat_1; > 1: prefers the true paraphrase);
  summaries as TOFU: forget mean(min(R, 1/R)), retain mean(max(0, 1 - 1/R))
- MMLU: log-prob of "A".."D" as the first answer token; correct = argmax is the gold letter
- retention(M) = (mean(M) - mean(zero)) / (mean(one) - mean(zero)), inside each author resample (shared/cluster_ci.py)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from auditkit.model import Request

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import ORIG, ROOT, cached, check_manifest, load, md5, read_jsonl, weights_id, write_jsonl, write_stage  # noqa: E402

sys.path.insert(0, str(ROOT / "shared"))
from cluster_ci import cluster_ci  # noqa: E402
from inmemory_model import InMemoryHFModel  # noqa: E402

GEN = {"max_tokens": 128, "temperature": 0}  # greedy; identical for every model
# stop at <|END_OF_TURN_TOKEN|> (6) or <EOS_TOKEN> (3): C0 ends turns with 6, which Tiny Aya's generation_config
# (eos 3 only) ignores, so C0 v2 kept writing after its answer and the extra words inflated ROUGE-L recall
STOP = [6, 3]
SCORING = ("v1: rouge_score rougeL recall stemmed; lp = mean per token of gold tokens, fp32 log_softmax; TOFU truth ratio; "
           "MMLU first-letter")
LETTERS = "ABCD"
DATA = ROOT / "data" / "prompts"


def load_sets(limit=None):
    split = json.load(open(DATA / "tofu_author_split.json"))
    pert = {r["id"]: r for r in json.load(open(DATA / "tofu_forget10_perturbed.json"))["rows"]}
    forget = [{**r, **pert[r["id"]]} for r in split["rows"]]
    retain = json.load(open(DATA / "tofu_retain_perturbed.json"))["rows"]
    sets = {"forget": forget, "retain": retain}
    for lang, n in (("en", 500), ("hi", 200)):
        sets[f"mmlu_{lang}"] = json.load(open(DATA / f"global_mmlu_{lang}{n}.json"))["rows"]
    return {s: rows[:limit] for s, rows in sets.items()}


def run_spec(ak, data):
    tok = ak._pipeline.tokenizer
    return {"model": ak.name, "generation": GEN, "batch_size": ak._extra_kwargs.get("batch_size", 1),
            "stop_ids": ak._extra_kwargs.get("eos_token_id"),
            "tokenizer": {"name": tok.name_or_path, "vocab": len(tok), "chat_template_md5": md5(tok.chat_template)},
            "scoring": SCORING, "prompts": {s: md5(rows) for s, rows in data.items()}}


def mean_logprobs(ak, pairs):
    """Mean per-token log-prob of each target after its rendered prompt, via AuditKIT's teacher forcing."""
    tok = ak._pipeline.tokenizer
    res = ak.loglikelihood([Request(prompt=p, request_type="loglikelihood", params={"target": t}) for p, t in pairs])
    out, plen = [], {}
    for (p, t), r in zip(pairs, res):
        if p not in plen:
            plen[p] = len(tok(p, add_special_tokens=False).input_ids)
        n = len(tok(p + t, add_special_tokens=False).input_ids) - plen[p]  # the same split AuditKIT scores
        assert n > 0, f"empty target {t!r}"
        out.append(r.logprob / n)
    return out


def rouge_recall(gold, gen):
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(gold, gen)["rougeL"].recall


def evaluate(ak, label, out=ROOT / "runs/evaluation/tofu_v2", limit=None):
    data = load_sets(limit)
    d, c = out / label, out / label / "cache"
    spec = run_spec(ak, data)
    with torch.no_grad():  # log-probs in fp32 (InMemoryHFModel fp32_logits=True), or SCORING does not hold
        probe = ak._pipeline.model(input_ids=torch.tensor([[2]], device=ak._pipeline.model.device)).logits
    assert probe.dtype == torch.float32, "wrap the model with InMemoryHFModel(..., fp32_logits=True)"
    assert spec["stop_ids"] == STOP, "wrap the model with InMemoryHFModel(..., eos_token_id=STOP)"
    check_manifest(d, spec)  # before any output is read or written
    render = {s: [ak._render_prompt(Request(prompt=r["prompt"])) for r in data[s]] for s in data}  # AuditKIT's template
    for s in ("forget", "retain"):
        rows, t0 = data[s], time.time()
        if cached(c / f"{s}.gen.jsonl", rows) is None:  # 1. greedy answers
            res = ak.generate([Request(prompt=r["prompt"], params=GEN) for r in rows])
            write_stage(c / f"{s}.gen.jsonl", [{"id": r["id"], "response": x.completions[0].text} for r, x in zip(rows, res)])
            print(f"[{label}] {s}: {len(rows)} answers in {time.time() - t0:.0f} s", flush=True)
        t0 = time.time()
        if cached(c / f"{s}.lp.jsonl", rows) is None:  # 2. gold, paraphrased and 5 perturbed answers, teacher-forced
            pairs = [(p, t) for p, r in zip(render[s], rows) for t in [r["answer"], r["paraphrased_answer"], *r["perturbed_answer"]]]
            lp = iter(mean_logprobs(ak, pairs))
            write_stage(c / f"{s}.lp.jsonl", [{"id": r["id"], "lp_answer": next(lp), "lp_para": next(lp),
                                               "lp_pert": [next(lp) for _ in r["perturbed_answer"]]} for r in rows])
            print(f"[{label}] {s}: {len(pairs)} log-probs in {time.time() - t0:.0f} s", flush=True)
    for s in ("mmlu_en", "mmlu_hi"):  # 3. multiple choice by first-letter log-prob
        rows, t0 = data[s], time.time()
        if cached(c / f"{s}.lp.jsonl", rows) is None:
            lp = iter(mean_logprobs(ak, [(p, x) for p in render[s] for x in LETTERS]))
            write_stage(c / f"{s}.lp.jsonl", [{"id": r["id"], "lp": [next(lp) for _ in LETTERS]} for r in rows])
            print(f"[{label}] {s}: {4 * len(rows)} log-probs in {time.time() - t0:.0f} s", flush=True)

    summary = {"label": label, "size_mb": ak.model_info()["size_mb"], "limit": limit, "spec": spec, "sets": {}}
    for s, rows in data.items():  # 4. per-question rows and plain means
        if s.startswith("mmlu"):
            full = [{"id": r["id"], "subject": r["subject"], "answer": r["answer"], "lp": x["lp"],
                     "pred": LETTERS[int(np.argmax(x["lp"]))]} for r, x in zip(rows, cached(c / f"{s}.lp.jsonl", rows))]
            for r in full:
                r["correct"] = int(r["pred"] == r["answer"])
            summary["sets"][s] = {"n": len(full), "accuracy": float(np.mean([r["correct"] for r in full]))}
        else:
            full = [{**{k: r[k] for k in ("id", "author", "prompt", "answer")}, "split": r.get("split", "retain"), **g, **x,
                     "rougeL_recall": rouge_recall(r["answer"], g["response"]),
                     "truth_ratio": float(np.exp(x["lp_para"] - np.mean(x["lp_pert"])))}
                    for r, g, x in zip(rows, cached(c / f"{s}.gen.jsonl", rows), cached(c / f"{s}.lp.jsonl", rows))]
            R = np.array([r["truth_ratio"] for r in full])
            summary["sets"][s] = {"n": len(full), "rougeL_recall": float(np.mean([r["rougeL_recall"] for r in full])),
                                  "lp_answer": float(np.mean([r["lp_answer"] for r in full])),
                                  "truth_ratio_tofu": float(np.mean(np.minimum(R, 1 / R)) if s == "forget"
                                                            else np.mean(np.maximum(0, 1 - 1 / R)))}
        write_jsonl(d / f"{s}.jsonl", full)
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def table(out, zero="original", one=None):
    """results.md + table.json for every label in out: means with author-clustered 95% CIs, scaled retention."""
    sums = {p.parent.name: json.loads(p.read_text()) for p in sorted(out.glob("*/summary.json"))}
    if not sums:
        return
    first = next(iter(sums.values()))["spec"]
    for k, x in sums.items():  # every cell must come from the same prompts and settings
        bad = [f for f in ("generation", "batch_size", "stop_ids", "tokenizer", "scoring", "prompts")
               if x["spec"].get(f) != first.get(f)]
        assert not bad, f"{k} in {out} was made with other {bad}"
    rows = {k: {s: read_jsonl(out / k / f"{s}.jsonl") for s in x["sets"]} for k, x in sums.items()}
    scale = zero in rows and one in rows
    res = {}
    for k in sums:
        r = {"size_mb": sums[k]["size_mb"], "limit": sums[k]["limit"]}
        for s in ("forget", "retain"):
            author = {q["id"]: q["author"] for q in rows[k][s]}
            col = lambda m, f: {q["id"]: q[f] for q in rows[m][s]}  # noqa: E731
            r[f"{s}_rougeL"] = cluster_ci({k: col(k, "rougeL_recall")}, author)
            r[f"{s}_lp"] = cluster_ci({k: col(k, "lp_answer")}, author)
            if scale:  # k == zero gives 0, k == one gives 1, by construction
                v = {m: col(m, "rougeL_recall") for m in (zero, one)} | {"m": col(k, "rougeL_recall")}
                r[f"{s}_retention"] = cluster_ci(v, author, lambda m: (m["m"] - m[zero]) / (m[one] - m[zero]))
            r[f"{s}_truth_ratio_tofu"] = sums[k]["sets"][s]["truth_ratio_tofu"]
        for s in ("mmlu_en", "mmlu_hi"):  # questions are independent: each its own cluster
            r[s] = cluster_ci({k: {q["id"]: q["correct"] for q in rows[k][s]}}, {q["id"]: q["id"] for q in rows[k][s]})
        res[k] = r
    (out / "table.json").write_text(json.dumps({"zero": zero, "one": one if scale else None, "models": res}, indent=1))
    ci = lambda x, n=3: f"{x['point']:.{n}f} [{x['ci_low']:.{n}f}, {x['ci_high']:.{n}f}]" if x else ""  # noqa: E731
    lines = [f"Retention scaled: {zero} = 0, {one} = 1 (ROUGE-L recall)." if scale else "Retention: needs --zero and --one labels.",
             "95% CIs: author-clustered bootstrap (TOFU), question bootstrap (MMLU); 10,000 resamples, seed 0.", "",
             "| model | forget retention | forget ROUGE-L | forget lp/token | forget truth ratio | retain retention | retain ROUGE-L "
             "| retain lp/token | retain truth ratio | MMLU en (500) | MMLU hi (200) | size MiB |", "|" + " --- |" * 12]
    order = sorted(res, key=lambda k: (k != zero, k != one, k))
    for k in order:
        r = res[k]
        lines.append(f"| {k} | {ci(r.get('forget_retention'))} | {ci(r['forget_rougeL'])} | {ci(r['forget_lp'])} | "
                     f"{r['forget_truth_ratio_tofu']:.3f} | {ci(r.get('retain_retention'))} | {ci(r['retain_rougeL'])} | "
                     f"{ci(r['retain_lp'])} | {r['retain_truth_ratio_tofu']:.3f} | {ci(r['mmlu_en'])} | {ci(r['mmlu_hi'])} | "
                     f"{r['size_mb']:.1f} |")
    notes = [f"{k}: limit={r['limit']}" for k, r in res.items() if r["limit"]]
    (out / "results.md").write_text("\n".join(lines + (["", "Smoke runs: " + ", ".join(notes)] if notes else [])) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?")
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/evaluation/tofu_v2")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--table", action="store_true", help="only rebuild results.md (CPU)")
    ap.add_argument("--zero", default="original")
    ap.add_argument("--one", help="label scaled to retention 1 (the TOFU target, e.g. c0v2)")
    a = ap.parse_args()
    if not a.table:
        assert a.label and (a.checkpoint is None) == (a.label == "original"), "give a checkpoint for every label but 'original'"
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ORIG)  # the original tokenizer for every variant
        ak = InMemoryHFModel(load(a.checkpoint), tok, name=f"tiny-aya-{a.label}@{weights_id(a.checkpoint)}",
                             batch_size=a.batch_size, fp32_logits=True, eos_token_id=STOP)
        print(json.dumps(evaluate(ak, a.label, a.out, a.limit)["sets"], indent=1))
    table(a.out, a.zero, a.one)
    print((a.out / "results.md").read_text())
