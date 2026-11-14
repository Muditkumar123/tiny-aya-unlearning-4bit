"""Refusal-circuit-protected 4-bit quantization of Tiny Aya with CircuitKIT."""
import gc
import json
import os
from collections import Counter

import circuitkit as ck
import torch
from circuitkit import Pipeline
from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.quick import quantize
from circuitkit.tasks.registry import register_task
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "CohereLabs/tiny-aya-global")
N = int(os.environ.get("N", "32"))
HIGH = float(os.environ.get("HIGH", "0.3"))
OUT = os.path.expanduser(os.environ.get("OUT", "~/Lexsi/Mudit/out/refusal_circuit"))
TASK = "aya_refusal"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")


def wrap(text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True).strip()


def ids(text):
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def first_tokens(prompts):
    return Counter(hf(torch.tensor([ids(p)], device="cuda")).logits[0, -1].argmax().item() for p in prompts)


alpaca = [r["instruction"] for r in load_dataset("tatsu-lab/alpaca", split="train") if not r["input"]]
pool = {}
for text, toks in zip(alpaca, tok(alpaca, add_special_tokens=False).input_ids):
    pool.setdefault(len(toks), []).append(text)
pairs = []
for h in load_dataset("walledai/HarmBench", "standard", split="train")["prompt"]:
    wh = wrap(h)
    match = next((b for b in pool.get(len(ids(h)), []) if len(ids(wrap(b))) == len(ids(wh))), None)
    if match:
        pool[len(ids(h))].remove(match)
        pairs.append((wh, wrap(match)))
    if len(pairs) == N:
        break
assert len(pairs) == N, f"only {len(pairs)} length-matched pairs; lower N"
harmful, harmless = [h for h, _ in pairs], [b for _, b in pairs]
print(f"templated prompt length: {len(ids(harmful[0]))} tokens")

refused, complied = first_tokens(harmful), first_tokens(harmless)
for name, counts in (("harmful", refused), ("harmless", complied)):
    print(f"{name} first tokens:", [(tok.decode([i]), n) for i, n in counts.most_common(5)])
refusal = os.environ.get("REFUSAL_TOKEN") or tok.decode([refused.most_common(1)[0][0]])
compliance = os.environ.get("COMPLIANCE_TOKEN") or next(
    (tok.decode([i]) for i, _ in complied.most_common() if tok.decode([i]) != refusal), None)
assert compliance and compliance != refusal, "harmful and harmless prompts start the same way; set the tokens by hand"
for ans in (refusal, compliance):
    p, full = ids(harmful[0]), ids(harmful[0] + ans)
    assert len(full) == len(p) + 1 and full[:len(p)] == p, f"{ans!r} is not one clean token after the prompt"
print(f"refusal={refusal!r} compliance={compliance!r}")

ds = SafetyPromptAdapter().adapt([{"prompt": p} for p in harmful], name=TASK, source="walledai/HarmBench",
                                 refusal_token=refusal, compliance_token=compliance,
                                 benign_controls=harmless, pairing_mode="harmful_vs_benign")
for record, benign in zip(ds.records, harmless):
    record.corrupt_prompt = benign
register_task(NormalizedTaskSpec(ds, name=TASK, cache_dir=os.path.join(OUT, "cache")))

hf.to("cpu")
torch.cuda.empty_cache()
pipe = Pipeline(MODEL, task=TASK, device="cuda", output_dir=OUT)
pipe.discover(algorithm="eap-ig", n_examples=N, batch_size=1, sparsity=0.3)
pipe.evaluate(n_examples=N, n_stability_runs=1)
pipe.summary()
circuit = pipe.circuit
del pipe
gc.collect()
torch.cuda.empty_cache()

hf.to("cuda")
plan = quantize(hf, circuit, high_fraction=HIGH, backend="quanto")
ckpt = ck.export_checkpoint(hf, None, os.path.join(OUT, "tiny-aya-refusal-protected-q4"),
                            intervention="quantization", tokenizer_name=MODEL)
with open(os.path.join(OUT, "run.json"), "w") as f:
    json.dump({"model": MODEL, "refusal_token": refusal, "compliance_token": compliance,
               "high_fraction": HIGH, "plan": plan}, f, indent=2, default=str)
print("saved", ckpt)
