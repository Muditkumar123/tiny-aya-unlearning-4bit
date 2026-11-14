"""Self-check for kl_divergence. Run: python addons/test_kl_divergence.py (needs torch only)."""
import math
from types import SimpleNamespace

import torch

from kl_divergence import kl_divergence

V = 4
SKEW = torch.tensor([0.7, 0.1, 0.1, 0.1]).log()


class FakeLM:
    """Gives uniform logits before position `skew_from`, SKEW logits from there on."""
    device = "cpu"

    def __init__(self, skew_from=None):
        self.skew_from = skew_from

    def __call__(self, ids):
        logits = torch.zeros(1, ids.shape[1], V)
        if self.skew_from is not None:
            logits[0, self.skew_from:] = SKEW
        return SimpleNamespace(logits=logits)


def tok(text, add_special_tokens=False):
    return SimpleNamespace(input_ids=[ord(ch) % V for ch in text])


uniform, skewed = FakeLM(), FakeLM(skew_from=0)
expected = math.log(0.25) - 0.25 * (math.log(0.7) + 3 * math.log(0.1))  # KL(uniform || SKEW)

r = kl_divergence(uniform, uniform, tok, ["ab", "abc"], ["cd", ""])
assert r["mean_kl"] < 1e-6 and r["first_token_kl"] < 1e-6, r

r = kl_divergence(uniform, skewed, tok, ["ab", "abc"])
assert r["n"] == 2 and abs(r["first_token_kl"] - expected) < 1e-5, (r, expected)
# direction is KL(base || cand): swapping the models gives a different number
assert abs(kl_divergence(skewed, uniform, tok, ["ab"])["mean_kl"] - expected) > 1e-3

# prompt "ab" + response "cd" scores positions 1 and 2 (they predict "c" and "d"); only 2 drifts
r = kl_divergence(uniform, FakeLM(skew_from=2), tok, ["ab"], ["cd"])
assert r["first_token_kl"] < 1e-6, r
assert abs(r["mean_kl"] - expected / 2) < 1e-5, (r, expected)

# position 3 predicts a token after the response, so drift there alone must not count
r = kl_divergence(uniform, FakeLM(skew_from=3), tok, ["ab"], ["cd"])
assert r["mean_kl"] < 1e-6, r

for bad in (["cd", "ef"], []):  # an explicit empty list is a mismatch too, not "use the defaults"
    try:
        kl_divergence(uniform, uniform, tok, ["ab"], bad)
        raise SystemExit(f"length mismatch {bad} was not caught")
    except AssertionError:
        pass
print("kl_divergence: all checks passed")
