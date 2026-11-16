"""How the intended fp32 LoRA change D survives in a saved bf16 merged model (CPU only).

D = LoRA delta from the adapter (fp32); A = W_merged_saved - W_target, both bf16 weights read from disk, in fp32.
Reported over all LoRA-wrapped linears (sums over every weight):
  projection  = <A, D> / <D, D>      share of the intended change present along its own direction (1 = all)
  norm_ratio  = ||A|| / ||D||
  cosine      = <A, D> / (||A|| ||D||)
  rel_error   = ||A - D|| / ||D||    (the number unlearn_c2.py saved as rel_rounding_error_of_delta)
  changed     = share of weights with A != 0
  saved_equals_rounding = share of weights where saved == bf16(W_target + D) (checks the saved merge)

python training/merge_diagnostics.py runs/training/c2-lr3e-05-adapter runs/training/c2-lr3e-05-merged [target]
"""
import json, sys
import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM

adapter, merged = sys.argv[1], sys.argv[2]
target = sys.argv[3] if len(sys.argv) > 3 else "runs/training/c0v2-target-merged"
base = AutoModelForCausalLM.from_pretrained(target, dtype=torch.bfloat16)
w_t = {n: p.detach().clone() for n, p in base.named_parameters() if n.endswith("proj.weight")}
peft = PeftModel.from_pretrained(base, adapter)
f = safe_open(f"{merged}/model.safetensors", "pt")
s = dict(AD=0.0, DD=0.0, AA=0.0, err=0.0, changed=0, same=0, n=0, layers=0)
for name, mod in peft.named_modules():
    if hasattr(mod, "get_delta_weight") and "default" in getattr(mod, "lora_A", {}):
        key = name.replace("base_model.model.", "") + ".weight"
        D = mod.get_delta_weight("default").float()
        W0, Ws = w_t[key], f.get_tensor(key)
        A = Ws.float() - W0.float()
        s["AD"] += (A * D).sum().item(); s["DD"] += (D * D).sum().item(); s["AA"] += (A * A).sum().item()
        s["err"] += ((A - D) ** 2).sum().item()
        s["changed"] += (A != 0).sum().item()
        s["same"] += (Ws == (W0.float() + D).to(torch.bfloat16)).sum().item()
        s["n"] += A.numel(); s["layers"] += 1
out = {"adapter": adapter, "merged": merged, "target": target, "linears": s["layers"], "weights": s["n"],
       "projection": s["AD"] / s["DD"], "norm_ratio": (s["AA"] / s["DD"]) ** 0.5,
       "cosine": s["AD"] / (s["AA"] * s["DD"]) ** 0.5, "rel_error": (s["err"] / s["DD"]) ** 0.5,
       "changed": s["changed"] / s["n"], "saved_equals_rounding": s["same"] / s["n"], "D_fro": s["DD"] ** 0.5}
assert out["linears"] == 252, out["linears"]
print(json.dumps(out, indent=1))
json.dump(out, open(f"{adapter}/merge_diagnostics.json", "w"), indent=1)
