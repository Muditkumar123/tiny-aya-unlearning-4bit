"""Self-check for inmemory_model on a tiny random model quantized with quanto.

Run on the server (needs the cached Tiny Aya tokenizer and quanto's kernels, see inmemory_model.py):
    CUDA_HOME=/usr/local/cuda-12.9 PATH=$CUDA_HOME/bin:~/miniconda3/envs/lexsi/bin:$PATH python test_inmemory_model.py
"""
import copy

import torch
from auditkit.model import Request
from optimum.quanto import freeze, qint4, quantize
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from inmemory_model import InMemoryHFModel, _nbytes

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
fp = AutoModelForCausalLM.from_config(cfg).to(device).eval()  # fp32: bf16 rounding hides quantization's effect on a tiny model
q = copy.deepcopy(fp)
quantize(q.model, weights=qint4)
freeze(q.model)

prompt, target = "The capital of France is", " Paris"


@torch.no_grad()
def manual_logprob(model):
    ctx = tok(prompt, add_special_tokens=False).input_ids
    full = tok(prompt + target, add_special_tokens=False).input_ids
    lp = torch.log_softmax(model(input_ids=torch.tensor([full], device=device)).logits[0, :-1], dim=-1)
    picked = lp.gather(1, torch.tensor(full[1:], device=device).unsqueeze(-1)).squeeze(-1)
    return picked[len(ctx) - 1:].sum().item()


req = [Request(prompt=prompt, request_type="loglikelihood", params={"target": target})]
ak_q, ak_fp = InMemoryHFModel(q, tok, name="tiny-q4"), InMemoryHFModel(fp, tok, name="tiny-fp32")
lq = ak_q.loglikelihood(req)[0].logprob
want_q, want_fp = manual_logprob(q), manual_logprob(fp)

# AuditKIT scores the model object it was given, not a reload of another one
assert abs(want_q - want_fp) > 1e-5, "quantization changed nothing, so this check proves nothing"
assert abs(lq - want_q) < 1e-4 < abs(lq - want_fp), (lq, want_q, want_fp)
assert abs(ak_fp.loglikelihood(req)[0].logprob - want_fp) < 1e-4

# unload() must not drop the model: HFGenModel would reload it by name
ak_q.unload()
assert ak_q._pipeline.model is q and ak_q.loglikelihood(req)[0].logprob == lq

# AuditKIT's generation cap is "max_tokens" (it maps it to max_new_tokens); a cap it ignored would give equal lengths
short, long_ = (ak_q.generate([Request(prompt="Hello", params={"max_tokens": n, "temperature": 0})])[0].text
                for n in (3, 30))
assert isinstance(short, str) and len(short) < len(long_), (short, long_)

# fp32_logits: a bf16 model's loglikelihood equals an fp32 log_softmax of its own bf16 logits; greedy text is unchanged
bf = copy.deepcopy(fp).to(torch.bfloat16)
plain_bf = InMemoryHFModel(bf, tok, name="tiny-bf16")
gen = lambda m: m.generate([Request(prompt="Hello", params={"max_tokens": 12, "temperature": 0})])[0].text  # noqa: E731
text_before, lp_before = gen(plain_bf), plain_bf.loglikelihood(req)[0].logprob
exact_bf = InMemoryHFModel(bf, tok, name="tiny-bf16-fp32logits", fp32_logits=True)
want_bf = manual_logprob(bf)  # the hook is on the model now, so this is the fp32 log_softmax of the bf16 logits
assert abs(exact_bf.loglikelihood(req)[0].logprob - want_bf) < 1e-4, (exact_bf.loglikelihood(req)[0].logprob, want_bf)
assert gen(exact_bf) == text_before
print(f"bf16 log-prob without fp32_logits {lp_before:.6f}, with {want_bf:.6f}")

# size counts packed 4-bit weights, not their bf16 size
w = q.model.layers[0].mlp.up_proj.weight
assert _nbytes(w) < 0.4 * w.numel() * 2, (_nbytes(w), w.numel() * 2)
assert ak_q.model_info()["size_mb"] < ak_fp.model_info()["size_mb"]
print("inmemory_model: all checks passed")
