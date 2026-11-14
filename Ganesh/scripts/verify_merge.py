"""R2 merge check (PLAN section 4): rerun PEFT merge_and_unload in bf16 on CPU and compare bitwise with the saved merge.
Also records sha256 of the saved weights and adapter, and library versions. CPU only; never downloads (local_files_only).

python Ganesh/scripts/verify_merge.py <base path or HF id> <adapter dir> <merged dir> [--revision R] [--expect-linears 252] [--out f.json]
Exit 0 = pass, 1 = fail.
"""
import argparse, hashlib, json, sys
import peft, torch, transformers
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(base, adapter, merged, revision=None, expect_linears=None):
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16, local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter)
    lora = [n for n, m in model.named_modules() if "default" in getattr(m, "lora_A", {})]
    state = model.merge_and_unload().state_dict()
    bad, compared = [], 0
    with safe_open(f"{merged}/model.safetensors", "pt") as f:
        saved_keys = set(f.keys())
        for k in sorted(saved_keys):
            a, b = state[k], f.get_tensor(k)
            compared += 1
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                diff = (a.float() - b.float()).abs() if a.shape == b.shape else None
                bad.append({"key": k, "dtype": [str(a.dtype), str(b.dtype)],
                            "n_diff": None if diff is None else int((diff > 0).sum()),
                            "max_abs_diff": None if diff is None else diff.max().item()})
    # Keys the model has but the file lacks must be tied weights (e.g. lm_head tied to embeddings).
    absent = sorted(k for k in state if k not in saved_keys)
    untied = [k for k in absent if not any(state[k].data_ptr() == state[s].data_ptr() for s in saved_keys)]
    out = {"base": base, "revision": revision, "adapter": adapter, "merged": merged,
           "sha256_merged": sha256(f"{merged}/model.safetensors"),
           "sha256_adapter": sha256(f"{adapter}/adapter_model.safetensors"),
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
           "torch_threads": torch.get_num_threads(), "lora_linears": len(lora), "tensors_compared": compared,
           "absent_tied": absent, "absent_untied": untied, "mismatches": bad[:50], "n_mismatched_tensors": len(bad)}
    # Loader and precision (PLAN section 4): the saved dir reloads with transformers, nothing missing, all bf16.
    del model, state
    loaded, info = AutoModelForCausalLM.from_pretrained(merged, dtype="auto", output_loading_info=True)
    dtypes = {}
    for p in loaded.parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    out["reload"] = {"param_dtypes": dtypes, **{k: sorted(map(str, v)) for k, v in info.items() if v}}
    out["pass"] = (not bad and not untied and compared > 0 and set(dtypes) == {"torch.bfloat16"} and
                   not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")) and
                   (expect_linears is None or len(lora) == expect_linears))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("adapter"); ap.add_argument("merged")
    ap.add_argument("--revision"); ap.add_argument("--expect-linears", type=int); ap.add_argument("--out")
    a = ap.parse_args()
    r = verify(a.base, a.adapter, a.merged, a.revision, a.expect_linears)
    print(json.dumps(r, indent=1))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    sys.exit(0 if r["pass"] else 1)
