"""Post-hoc check: NF4 (bitsandbytes) 4-bit checkpoint of a bf16 Tiny Aya model."""
import hashlib
import json
import os
import sys

import bitsandbytes as bnb
import torch
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

LEXSI = os.path.expanduser("~/Lexsi")
sys.path.insert(0, f"{LEXSI}/Vedika/scripts")
from evaluate import load  # noqa: E402

src, out = map(os.path.abspath, sys.argv[1:3])
assert not os.path.exists(out), f"{out} exists: a new checkpoint goes in a new folder"
CFG = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
           bnb_4bit_use_double_quant=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def nf4_linears(m):
    q = {n: mod for n, mod in m.named_modules() if isinstance(mod, bnb.nn.Linear4bit)}
    assert len(q) == 7 * m.config.num_hidden_layers, len(q)
    assert all(".layers." in n for n in q), [n for n in q if ".layers." not in n]
    assert all(mod.weight.quant_state.quant_type == "nf4" and mod.weight.quant_state.blocksize == 64 for mod in q.values())
    return q


hf = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, quantization_config=BitsAndBytesConfig(**CFG),
                                          device_map={"": 0})
built = nf4_linears(hf)
hf.save_pretrained(out)

re = load(out)
reloaded = nf4_linears(re)
same = sum(torch.equal(built[n].weight.data, reloaded[n].weight.data) and
           torch.equal(built[n].weight.quant_state.absmax, reloaded[n].weight.quant_state.absmax) for n in built)
assert same == len(built), f"{same} of {len(built)} linears reload identically"
assert re.lm_head.weight.dtype == re.model.embed_tokens.weight.dtype == torch.bfloat16

run = {"script": "Mudit/scripts/build_nf4.py", "definition": "CHANGELOG Mudit 2026-11-07 20:18 (post hoc, exploratory)",
       "src": src, "src_sha256": sha256(f"{src}/model.safetensors"), "out": out,
       "bnb_config": {k: str(v) for k, v in CFG.items()}, "blocksize": 64,
       "bitsandbytes": bnb.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
       "nf4_linears": len(built), "reloaded_identical_linears": same,
       "files": {f: os.path.getsize(f"{out}/{f}") for f in sorted(os.listdir(out))}}
json.dump(run, open(f"{out}/run.json", "w"), indent=1)
print(json.dumps(run, indent=1))
