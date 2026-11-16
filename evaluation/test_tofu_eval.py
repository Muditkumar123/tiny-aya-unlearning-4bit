"""GPU self-check for tofu_eval.py's scoring (~1 min, Tiny Aya bf16 ~7 GB).

    CUDA_VISIBLE_DEVICES=0 python evaluation/test_tofu_eval.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from auditkit.model import Request
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import tofu_eval as te  # noqa: E402

# ROUGE-L recall = LCS over gold words / gold words, after stemming (TOFU's eval_rouge_recall)
assert te.rouge_recall("the cats sat on mats", "a cat sat mats") == 3 / 5
assert te.rouge_recall("Hsiao Yun-Hwa", "I do not know") == 0

tok = AutoTokenizer.from_pretrained(te.ORIG)
model = te.load()
rows = te.load_sets(2)["forget"]

# without fp32_logits the run is refused (its log-probs would be bf16-rounded)
try:
    te.evaluate(te.InMemoryHFModel(model, tok, name="test-bf16"), "x", Path(tempfile.mkdtemp()), limit=2)
    raise SystemExit("evaluate accepted bf16 logits")
except AssertionError as e:
    assert "fp32_logits" in str(e), e

ak = te.InMemoryHFModel(model, tok, name="test-fp32", fp32_logits=True, eos_token_id=te.STOP)
prompts = [ak._render_prompt(Request(prompt=r["prompt"])) for r in rows]
pairs = [(p, t) for p, r in zip(prompts, rows) for t in (r["answer"], r["paraphrased_answer"], *r["perturbed_answer"])]
got = te.mean_logprobs(ak, pairs)


@torch.no_grad()
def manual(p, t):  # one forward, fp32 log_softmax, mean over the target tokens only
    ip, ifull = (tok(x, add_special_tokens=False).input_ids for x in (p, p + t))
    assert ifull[:len(ip)] == ip and ip.count(2) == 1  # target is a clean suffix; a single BOS
    lp = F.log_softmax(model(torch.tensor([ifull], device=model.device)).logits[0].float(), -1)
    return lp[len(ip) - 1:-1].gather(1, torch.tensor(ifull[len(ip):], device=model.device)[:, None]).mean().item()


want = [manual(p, t) for p, t in pairs]
assert np.allclose(got, want, atol=1e-4), list(zip(got, want))

# truth ratio = TOFU's exp(mean perturbed NLL - paraphrased NLL), from per-token losses
out = Path(tempfile.mkdtemp())
s = te.evaluate(ak, "orig", out, limit=2)
r0 = json.loads((out / "orig/forget.jsonl").read_text().splitlines()[0])
nll_para, nll_pert = -want[1], -np.mean(want[2:7])
assert abs(r0["truth_ratio"] - np.exp(nll_pert - nll_para)) < 1e-3 * r0["truth_ratio"]
assert abs(r0["lp_answer"] - want[0]) < 1e-4
assert s["sets"]["mmlu_en"]["n"] == 2 and set(json.loads((out / "orig/mmlu_en.jsonl").read_text().splitlines()[0])["pred"]) <= set("ABCD")
print(f"tofu_eval scoring: all checks passed (max |lp - manual| = {max(abs(a - b) for a, b in zip(got, want)):.2e})")
