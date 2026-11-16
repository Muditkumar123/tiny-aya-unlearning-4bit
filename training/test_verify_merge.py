"""Self-check for verify_merge.py on a tiny random model (CPU, no download).
A correct saved merge must pass; one flipped weight, a wrong adapter or a wrong linear count must fail.

python training/test_verify_merge.py
"""
import os, sys, tempfile
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, LlamaConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_merge import verify

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, tie_word_embeddings=True)
with tempfile.TemporaryDirectory() as d:
    base, adapter, adapter2, merged = (f"{d}/{n}" for n in ("base", "adapter", "adapter2", "merged"))
    AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).save_pretrained(base)
    lc = LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", init_lora_weights=False)  # nonzero delta
    for path in (adapter, adapter2):
        get_peft_model(AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16), lc).save_pretrained(path)
    # Same recipe as train_c0.py / unlearn_c2.py: fresh bf16 base on CPU, merge_and_unload, save.
    PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16), adapter) \
        .merge_and_unload().save_pretrained(merged)
    n_lin = 2 * 7  # q,k,v,o,gate,up,down per layer

    r = verify(base, adapter, merged, expect_linears=n_lin)
    assert r["pass"] and r["lora_linears"] == n_lin and r["absent_untied"] == [], r
    assert not verify(base, adapter2, merged)["pass"], "a different adapter must fail"
    assert not verify(base, adapter, merged, expect_linears=n_lin + 1)["pass"], "a wrong linear count must fail"

    sd = load_file(f"{merged}/model.safetensors")
    k = "model.layers.0.self_attn.q_proj.weight"
    sd[k] = sd[k].clone(); sd[k][0, 0] = sd[k][0, 0] + 1
    save_file(sd, f"{merged}/model.safetensors", metadata={"format": "pt"})
    r = verify(base, adapter, merged)
    assert not r["pass"] and r["n_mismatched_tensors"] == 1 and r["mismatches"][0]["n_diff"] == 1, r
print("test_verify_merge: all checks passed")
