"""Build an all-qint4 checkpoint as frozen in common/configs/frozen.yaml: C1 from C0 v2, C3 from C2."""
import hashlib
import json
import os
import subprocess
import sys

import circuitkit as ck
import torch
from auditkit.model import Request
from circuitkit.applications import detect_model_architecture, get_arch_config, validate_model_paths
from circuitkit.applications.quantization.quant_utils import _apply_plan_to_model
from circuitkit.evaluation import load_quantized_checkpoint
from optimum.quanto import QModuleMixin, qint4
from transformers import AutoModelForCausalLM, AutoTokenizer

LEXSI = os.path.expanduser("~/Lexsi")
sys.path.insert(0, f"{LEXSI}/common/addons")
from inmemory_model import InMemoryHFModel  # noqa: E402

src, out = map(os.path.abspath, sys.argv[1:3])
assert not os.path.exists(out), f"{out} exists: a new checkpoint goes in a new folder"
git = lambda *a: subprocess.run(["git", "-C", LEXSI, *a], capture_output=True, text=True)  # noqa: E731
assert git("diff", "--quiet", "HEAD", "--", "common/configs/frozen.yaml").returncode == 0, "frozen.yaml has uncommitted changes"
frozen = git("log", "-1", "--format=%H", "--", "common/configs/frozen.yaml").stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


hf = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).to("cuda")
n = hf.config.num_hidden_layers
plan = {part: {layer: "low" for layer in range(n)} for part in ("attn", "mlp")}
arch = get_arch_config(detect_model_architecture(hf))
validate_model_paths(hf, arch)
_apply_plan_to_model(hf, plan, qint4, None, None, None, True, arch)
n_frozen = sum(isinstance(m, QModuleMixin) and m.frozen for m in hf.modules())
ck.export_checkpoint(hf, None, out, intervention="quantization", tokenizer_name=src)
del hf
torch.cuda.empty_cache()

qmap = json.load(open(f"{out}/quanto_qmap.json"))
assert len(qmap) == 7 * n == n_frozen, f"qmap {len(qmap)}, frozen {n_frozen}, want {7 * n}"
assert all(v["weights"] == "qint4" for v in qmap.values())
assert not any("lm_head" in k or "embed" in k for k in qmap)

model = load_quantized_checkpoint(out).to("cuda").eval()
n_reloaded = sum(isinstance(m, QModuleMixin) for m in model.modules())
assert n_reloaded == len(qmap), f"{n_reloaded} QModules after reload, qmap has {len(qmap)}"
ak = InMemoryHFModel(model, AutoTokenizer.from_pretrained(out), name=f"build-check:{out}")
row = json.load(open(f"{LEXSI}/common/data/tofu_author_split.json"))["rows"][0]
gen = {"temperature": 0, "max_tokens": 40}
answer = ak.generate([Request(prompt=row["prompt"], params=dict(gen))])[0].text

run = {"source": src, "source_safetensors_sha256": sha256(f"{src}/model.safetensors"),
       "source_safetensors_bytes": os.path.getsize(f"{src}/model.safetensors"), "frozen_yaml_commit": frozen,
       "plan": "every attention and MLP unit qint4 (group 128, MaxOptimizer)", "bf16": ["lm_head (tied)", "embeddings"],
       "qmap_entries": len(qmap), "qmap_all_qint4": True, "frozen_qmodules": n_frozen, "reloaded_qmodules": n_reloaded,
       "safetensors_bytes": os.path.getsize(f"{out}/model.safetensors"), "size_mb": ak.model_info()["size_mb"],
       "check_generation": {"row": row["id"], "prompt": row["prompt"], "gold": row["answer"], "settings": gen,
                            "output": answer}}
json.dump(run, open(f"{out}/run.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(run, indent=2, ensure_ascii=False))
