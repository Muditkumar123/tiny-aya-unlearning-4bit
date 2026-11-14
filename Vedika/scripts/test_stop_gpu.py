"""GPU self-check (R5 step 1): evaluate.py's generation stops at the end of turn (~1 min, Tiny Aya bf16 ~7 GB).

    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 ~/miniconda3/envs/lexsi/bin/python Vedika/scripts/test_stop_gpu.py

For 2 XSTest-safe and 1 Hindi + 1 Arabic Aya prompt: AuditKIT's answer (InMemoryHFModel, eos_token_id=STOP, batch 1)
must equal a manual greedy generate with the same stop IDs, which must end at a stop ID (or at max_tokens) with no
stop ID before its end. Also prints how long the answers run with Tiny Aya's default stop (3 only), for the record.
Prints counts only; no answer text is saved.
"""
import json
import sys
from pathlib import Path

import torch
from auditkit.model import Request
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev  # noqa: E402

tok = AutoTokenizer.from_pretrained(ev.ORIG)
assert tok.convert_ids_to_tokens(ev.STOP) == ev.STOP_TOKENS
model = ev.load()
ak = ev.InMemoryHFModel(model, tok, name="tiny-aya-original@stopcheck", batch_size=1, eos_token_id=ev.STOP)
xs = json.load(open(ev.ROOT / "common/data/xstest_safe.json"))["rows"][:2]
aya = json.load(open(ev.ROOT / "common/data/aya_redteam_hi_ar.json"))["rows"]
rows = xs + [next(r for r in aya if r["language"] == lang) for lang in ("Hindi", "Arabic")]

for r in rows:
    text = ak.generate([Request(prompt=r["prompt"], params=ev.GEN)])[0].completions[0].text
    ids = tok(ak._render_prompt(Request(prompt=r["prompt"])), add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        new = {tuple(e): model.generate(ids, max_new_tokens=ev.GEN["max_tokens"], do_sample=False, eos_token_id=e,
                                        pad_token_id=tok.pad_token_id)[0, ids.shape[1]:].tolist() for e in (ev.STOP, [3])}
    got = new[tuple(ev.STOP)]
    assert got[-1] in ev.STOP or len(got) == ev.GEN["max_tokens"], (r["id"], got[-5:])
    assert not set(got[:-1]) & set(ev.STOP), (r["id"], "a stop ID inside the answer")
    assert tok.decode(got, skip_special_tokens=True).strip() == text.strip(), (r["id"], "AuditKIT answer differs")
    old = new[(3,)]
    print(f"{r['id']}: {len(got)} tokens, ends {tok.convert_ids_to_tokens(got[-1])}; with stop 3 only: {len(old)} tokens, "
          f"end-of-turn inside: {6 in old[:-1]}")
print("stop IDs on GPU: all checks passed")
