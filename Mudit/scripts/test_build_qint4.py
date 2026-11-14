"""CPU test for build_qint4.py on a tiny random Llama: full build, reload counts, provenance, and the refusals."""
import json
import os
import sys
import tempfile

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from build_qint4 import build, sha256  # noqa: E402
from verify_quant import main as verify  # noqa: E402

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=True)
with tempfile.TemporaryDirectory() as d:
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(f"{d}/src")
    AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global").save_pretrained(f"{d}/src")  # export copies the source tokenizer
    s = sha256(f"{d}/src/model.safetensors")
    run = build(f"{d}/src", f"{d}/q", device="cpu", expect_source_sha=s)
    assert run["qmap_entries"] == run["frozen_qmodules"] == run["reloaded_qmodules"] == 14, run
    assert run["source_safetensors_sha256"] == s and run["source_shards_sha256"] == {"model.safetensors": s}
    assert json.load(open(f"{d}/q/run.json"))["out_safetensors_sha256"] == sha256(f"{d}/q/model.safetensors")
    # the reviewed R1 checker accepts the result: 14 qint4 linears, excluded weights bitwise equal to the source
    assert verify([f"{d}/q", "--source", f"{d}/src", "--kind", "quanto", "--expect-q4", "14", "--expect-bf16", "0",
                   "--source-sha", s, "--out", f"{d}/r1.json"]) == 0, json.load(open(f"{d}/r1.json"))["failures"]
    for args, why in (((f"{d}/src", f"{d}/q"), "an existing out folder"),
                      ((f"{d}/src", f"{d}/q2"), "a wrong expected source sha")):
        try:
            build(*args, device="cpu", expect_source_sha="0" * 64 if "q2" in args[1] else None)
        except SystemExit:
            continue
        raise AssertionError(f"{why} must stop the build")
print("test_build_qint4: all checks passed")
