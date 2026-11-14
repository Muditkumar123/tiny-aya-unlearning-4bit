"""Plain and random-protected 4-bit baselines for Tiny Aya, quantized exactly like the protected model."""
import json
import os

import circuitkit as ck
import torch
from circuitkit.applications.quantization.quant_utils import random_quantize
from transformers import AutoModelForCausalLM

MODEL = os.environ.get("MODEL", "CohereLabs/tiny-aya-global")
OUT = os.path.expanduser(os.environ.get("OUT", "~/Lexsi/Mudit/out"))
SEED = 0
protected = json.load(open(os.path.join(OUT, "refusal_circuit_matched", "run.json")))["plan"]

for name, frac in (("plain-q4", 0.0), ("random30-q4", 0.3)):
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    plan = random_quantize(hf, hf.config.num_hidden_layers, high_fraction=frac, seed=SEED)
    for part in ("attn", "mlp"):
        kept = sum(t == "high" for t in plan[part].values())
        want = 0 if frac == 0 else sum(t == "high" for t in protected[part].values())
        assert kept == want, f"{name} {part}: {kept} layers kept, want {want}"
    path = os.path.join(OUT, name)
    ck.export_checkpoint(hf, None, path, intervention="quantization", tokenizer_name=MODEL)
    qmap = json.load(open(os.path.join(path, "quanto_qmap.json")))
    for part, sub in (("attn", "self_attn"), ("mlp", "mlp")):
        for layer, tier in plan[part].items():
            hits = [k for k in qmap if k.startswith(f"model.layers.{layer}.{sub}.")]
            assert all(qmap[k]["weights"] == "qint4" for k in hits)
            assert bool(hits) == (tier == "low"), f"{name}: layer {layer} {part} tier {tier}, qmap {hits}"
    assert not any("lm_head" in k or "embed" in k for k in qmap)
    run = {"model": MODEL, "quantizer": "circuitkit random_quantize -> _apply_plan_to_model, quanto qint4, "
           "kept layers bf16, lm_head and embeddings bf16", "high_fraction": frac, "seed": SEED,
           "plan": plan, "kept_bf16": {p: [l for l, t in plan[p].items() if t == "high"] for p in plan}}
    with open(os.path.join(path, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    print(name, "saved", path, "kept bf16:", run["kept_bf16"], flush=True)
    del hf
    torch.cuda.empty_cache()
