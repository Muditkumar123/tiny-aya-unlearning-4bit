"""CPU test for verify_quant.py.

Unit checks on a tiny random Llama, plus one end-to-end run of main() through CircuitKIT's real save/reload path.
"""
import copy
import json
import os
import sys
import tempfile

import torch
from optimum.quanto import freeze, qint4, qint8, quantize
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from verify_quant import check_excluded, check_header, check_quanto_model, decoder_linears, main, sha256  # noqa: E402

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=True)
source = LlamaForCausalLM(cfg).to(torch.bfloat16)
src = {k: v.clone() for k, v in source.state_dict().items()}
lin = decoder_linears(dict(source.named_modules()))
assert len(lin) == 14
keep = {"model.layers.1.mlp.down_proj", "model.layers.0.self_attn.v_proj"}  # left in bf16, like protected units


def build():
    m = copy.deepcopy(source)
    quantize(m, weights=qint4, include=[n for n in lin if n not in keep])
    freeze(m)
    return m


model = build()
qmap = {n: {"weights": "qint4", "activations": "none"} for n in lin if n not in keep}

# module/qmap checks
checks, fails = check_quanto_model(model, qmap, expect_q4=12, expect_bf16=2)
assert not fails and checks["qint4"] == 12 and checks["bf16"] == 2, (checks, fails)
bad = dict(qmap); bad["model.layers.0.self_attn.q_proj"] = {"weights": "qint8", "activations": "none"}
assert check_quanto_model(model, bad, 12, 2)[1], "a wrong qtype in the qmap must fail"
bad = dict(qmap); bad["model.layers.1.mlp.down_proj"] = {"weights": "qint4", "activations": "none"}
assert check_quanto_model(model, bad, 12, 2)[1], "a qmap entry for an unquantized layer must fail"
bad = dict(qmap); bad["model.embed_tokens"] = {"weights": "qint4", "activations": "none"}
assert check_quanto_model(model, bad, 12, 2)[1], "a qmap entry for the embeddings must fail"
assert check_quanto_model(model, qmap, 14, 0)[1], "a wrong expected count must fail"
m8 = copy.deepcopy(source); quantize(m8, weights=qint8, include=lin); freeze(m8)
assert check_quanto_model(m8, {n: {"weights": "qint4", "activations": "none"} for n in lin}, 14, 0)[1], "qint8 layers must not pass as qint4"

# excluded weights: unquantized params must be bf16 and bitwise equal to the source; head tied
n, fails = check_excluded(model, src.get, tied=True)
assert not fails and n > 0, fails
m = build()
with torch.no_grad():
    m.model.embed_tokens.weight.add_(1)
assert check_excluded(m, src.get, True)[1], "changed embeddings must fail"
m = build(); m.model.embed_tokens.weight.data = m.model.embed_tokens.weight.data.float()
assert check_excluded(m, src.get, True)[1], "float32 embeddings must fail"
m = build()
with torch.no_grad():
    m.model.layers[1].mlp.down_proj.weight[0, 0] += 1
assert check_excluded(m, src.get, True)[1], "a changed bf16 decoder linear must fail"
m = build()
with torch.no_grad():
    m.model.norm.weight.mul_(2)
assert check_excluded(m, src.get, True)[1], "a changed norm must fail"
m = build(); m.lm_head.weight = torch.nn.Parameter(m.model.embed_tokens.weight.detach().clone() + 1)
assert check_excluded(m, src.get, True)[1], "an untied, different lm_head must fail"

# stored dtype guard from the safetensors header
hdr = {f"{n}.weight": {"dtype": "BF16"} for n in keep}
hdr.update({f"{n}.weight._data": {"dtype": "U8"} for n in qmap})
hdr["model.embed_tokens.weight"] = {"dtype": "BF16"}
assert not check_header(hdr, qmap)[1]
bad = dict(hdr); bad["model.layers.1.mlp.down_proj.weight"] = {"dtype": "F32"}
assert check_header(bad, qmap)[1], "an F32 stored unquantized linear must fail"
bad = dict(hdr); bad["model.embed_tokens.weight"] = {"dtype": "F16"}
assert check_header(bad, qmap)[1], "F16 stored embeddings must fail"

# end to end: save through CircuitKIT, then run main() on the saved checkpoint
from circuitkit.evaluation import save_quantized_checkpoint  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    source.save_pretrained(f"{d}/src")
    save_quantized_checkpoint(build(), f"{d}/q")
    s = sha256(f"{d}/src/model.safetensors")
    args = [f"{d}/q", "--source", f"{d}/src", "--kind", "quanto", "--expect-q4", "12", "--expect-bf16", "2"]
    assert main(args + ["--source-sha", s, "--out", f"{d}/ok.json"]) == 0, json.load(open(f"{d}/ok.json"))["failures"]
    assert main(args + ["--source-sha", "0" * 64, "--out", f"{d}/badsha.json"]) == 1, "a wrong expected source sha must fail"
    other = LlamaForCausalLM(cfg).to(torch.bfloat16); other.save_pretrained(f"{d}/other")
    assert main([f"{d}/q", "--source", f"{d}/other", "--kind", "quanto", "--expect-q4", "12", "--expect-bf16", "2", "--out", f"{d}/other.json"]) == 1, \
        "a different source model must fail"

with tempfile.NamedTemporaryFile() as f:
    f.write(b"abc"); f.flush()
    assert sha256(f.name, chunk=2) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
print("test_verify_quant: all checks passed")
