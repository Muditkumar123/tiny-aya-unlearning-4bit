"""Run each card's usage snippet verbatim on CPU: real shipped tokenizer, tiny random Llama model in place of the 3B weights.
python test_card_snippet.py <card README.md> <local merged dir>"""
import re, sys
import torch, transformers
from transformers import LlamaConfig, LlamaForCausalLM

card, local = sys.argv[1], sys.argv[2]
code = re.search(r"```python\n(.*?)```", open(card).read(), re.S).group(1)
code = code.replace("<a TOFU question>", "Who is the author of the book?")

real_tok = transformers.AutoTokenizer.from_pretrained
tok = real_tok(local)
torch.manual_seed(0)
tiny = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                    num_attention_heads=2, num_key_value_heads=2, pad_token_id=0, eos_token_id=3)).eval()
seen = {}
real_gen = tiny.generate
def gen(*args, **kw):
    seen["args"], seen["kw"] = args, kw
    seen["out"] = real_gen(*args, **kw)
    return seen["out"]
tiny.generate = gen
transformers.AutoTokenizer.from_pretrained = lambda repo, **kw: tok
transformers.AutoModelForCausalLM.from_pretrained = lambda repo, **kw: tiny

ns = {}
exec(code, ns)
inputs, out = ns["inputs"], seen["out"]
assert not seen["args"] and isinstance(seen["kw"]["input_ids"], torch.Tensor), "generate must get tensor keyword inputs"
n = inputs["input_ids"].shape[1]
assert torch.equal(out[0, :n], inputs["input_ids"][0]), "output must start with the prompt"
ref = tok.apply_chat_template([{"role": "user", "content": "Who is the author of the book?"}], add_generation_prompt=True,
                              tokenize=False)
assert inputs["input_ids"][0].tolist() == tok(ref, add_special_tokens=False).input_ids, "ids must equal the rendered template"
assert seen["kw"]["eos_token_id"] == [6, 3]
print(card, "OK: prompt", n, "tokens; new tokens", out.shape[1] - n, "; transformers", transformers.__version__)
