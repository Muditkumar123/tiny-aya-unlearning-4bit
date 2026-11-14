"""KL divergence between the next-token distributions of an original and a compressed model.

Two models can give the same answers while their probabilities drift apart; KL catches that drift
before it flips answers (Dutta et al. 2024, "Accuracy is Not All You Need", arXiv:2407.09141).

Lives beside AuditKIT, not in its metric registry: AuditKIT metrics only see generated text, and
its hf loglikelihood() returns one summed logprob per target, never the full distribution.

    report = kl_divergence(original, compressed, tok, prompts)               # first response token only
    report = kl_divergence(original, compressed, tok, prompts, responses)    # plus every response token
    report["first_token_kl"], report["mean_kl"]

For Tiny Aya on HarmBench the first response token is the refusal decision ("I'm"), so
first_token_kl measures how far compression moved that decision.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _log_probs(model, ids: list[int], start: int, end: int) -> torch.Tensor:
    """Log-probs at positions start..end-1, in float32 (bf16 log-probs are too coarse for KL)."""
    logits = model(torch.tensor([ids], device=model.device)).logits[0, start:end]
    return F.log_softmax(logits.float(), dim=-1)


def kl_divergence(base, cand, tokenizer, prompts: list[str], continuations: list[str] | None = None) -> dict:
    """Mean KL(base || cand) per token, over the position after each prompt and over each continuation.

    Prompts should already carry the chat template. Continuations are teacher-forced (use the base
    model's own responses), so both models are scored on the same tokens.
    """
    if continuations is None:
        continuations = [""] * len(prompts)
    assert len(continuations) == len(prompts), "one continuation per prompt"
    per_prompt, first = [], []
    for prompt, cont in zip(prompts, continuations):
        # tokenized apart and joined so the prompt boundary is exact; the template already holds BOS
        p = tokenizer(prompt, add_special_tokens=False).input_ids
        c = tokenizer(cont, add_special_tokens=False).input_ids if cont else []
        # position len(p)-1 predicts the first response token; m response tokens need m positions (1 if none),
        # never the one after the last supplied token
        start, end = len(p) - 1, len(p) - 1 + max(len(c), 1)
        lp_base = _log_probs(base, p + c, start, end)
        lp_cand = _log_probs(cand, p + c, start, end).to(lp_base.device)
        kl = F.kl_div(lp_cand, lp_base, log_target=True, reduction="none").sum(-1)  # KL(base || cand)
        per_prompt.append(kl.mean().item())
        first.append(kl[0].item())
    n = len(prompts)
    # shortcut: averages per prompt, so long and short responses weigh the same; weight by tokens if lengths vary a lot
    return {
        "n": n,
        "mean_kl": sum(per_prompt) / n if n else None,
        "first_token_kl": sum(first) / n if n else None,
        "max_kl": max(per_prompt, default=None),
        "per_prompt": per_prompt,
        "first_per_prompt": first,
    }
