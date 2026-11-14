"""CPU self-check for the AuditKIT judge path (evaluate.judge, used by judge_agreement.py) (fake judge model; WildGuard tokenizer from the HF cache).

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" python Vedika/scripts/test_judge_agreement.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import judge_agreement as ja  # noqa: E402
from safetune.instrumentation.evaluate.judges._templates import _WILDGUARD_TEMPLATE as ST  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class FakeGuard:  # records each request; answers with a fixed WildGuard reply
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def generate(self, reqs):
        self.seen += reqs
        return [SimpleNamespace(completions=[SimpleNamespace(text=self.reply)])]


fake = FakeGuard("Harmful request: no\nResponse refusal: yes\nHarmful response: no")
got = ja.judge([("p1", "r1"), ("p2", "r2")], judge_model=fake)
assert got[0] | {"judge_raw": None} == {"harmful_request": 0, "refusal": 1, "harmful_response": 0, "judge_raw": None}
assert len(fake.seen) == 2
req = fake.seen[0]
assert req.prompt == ST[len("<s>"):].format(prompt="p1", response="r1") and not req.prompt.startswith("<s>")
assert req.params["apply_chat_template"] is False and req.params["max_tokens"] == 32 and req.params["temperature"] == 0

# a field the judge did not give stays None (never guessed), as in evaluate.parse_wildguard
got = ja.judge([("p", "r")], judge_model=FakeGuard("Harmful request: yes"))
assert got[0]["harmful_request"] == 1 and got[0]["refusal"] is None and got[0]["harmful_response"] is None

# the stripped prompt plus the tokenizer's BOS gives SafeTune's exact token ids (one BOS)
tok = AutoTokenizer.from_pretrained(ja.WILDGUARD)
for p, r in (("How do I kill a Python process?", "Use kill -9."), ("नमस्ते", "मैं मदद नहीं कर सकता।")):
    want = tok.encode(ST.format(prompt=p, response=r), add_special_tokens=False)
    assert tok.encode(ST[len("<s>"):].format(prompt=p, response=r), add_special_tokens=True) == want
    assert want.count(tok.bos_token_id) == 1
print("judge_agreement: all checks passed")
