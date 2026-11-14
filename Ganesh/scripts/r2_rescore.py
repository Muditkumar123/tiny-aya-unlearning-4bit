"""R2 full rescore (PLAN section 4 fallback): run Vedika/scripts/tofu_eval.py's evaluate() unchanged on one checkpoint,
and record the run settings the saved manifests lack: observed padding side, parameter/logit dtypes, device,
versions and every rendered prompt's token length as the model saw it (attention-mask sum at generation prefill).

CUDA_VISIBLE_DEVICES=1 python Ganesh/scripts/r2_rescore.py <label> <checkpoint> [--out Ganesh/out/r2_rescore]
Writes <out>/<label>/ (the harness's files) and <out>/<label>/run_settings.json.
"""
import argparse, json, sys
from pathlib import Path

import torch
import transformers
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Vedika" / "scripts"))
sys.path.insert(0, str(ROOT / "common" / "addons"))
import tofu_eval  # noqa: E402
from evaluate import ORIG, load, weights_id  # noqa: E402
from inmemory_model import InMemoryHFModel  # noqa: E402
from auditkit.model import Request  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label"); ap.add_argument("checkpoint")
ap.add_argument("--out", type=Path, default=ROOT / "Ganesh/out/r2_rescore")
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--limit", type=int, help="smoke test only; use a scratch --out")
a = ap.parse_args()

# Same construction as tofu_eval.py __main__.
tok = transformers.AutoTokenizer.from_pretrained(ORIG)
model = load(a.checkpoint)
ak = InMemoryHFModel(model, tok, name=f"tiny-aya-{a.label}@{weights_id(a.checkpoint)}",
                     batch_size=a.batch_size, fp32_logits=True, eos_token_id=tofu_eval.STOP)

obs = {"padding": {"left": 0, "right": 0, "none": 0}, "prefill_lengths": {}, "logits_dtype": set()}
stage = {"set": None}


def pre_hook(module, args, kwargs):
    m, ids = kwargs.get("attention_mask"), kwargs.get("input_ids")
    if m is None or ids is None:
        return
    left, right = bool((m[:, 0] == 0).any()), bool((m[:, -1] == 0).any())
    obs["padding"]["right" if right else "left" if left else "none"] += 1
    if stage["set"] and ids.shape[1] > 1:  # generation prefill: one row per prompt, in request order
        obs["prefill_lengths"].setdefault(stage["set"], []).extend(m.sum(1).tolist())


model.register_forward_pre_hook(pre_hook, with_kwargs=True)
model.get_output_embeddings().register_forward_hook(lambda mod, args, out: obs["logits_dtype"].add(str(out.dtype)))

orig_generate = ak.generate


def generate(reqs):  # tag prefill lengths with the set being generated (evaluate() generates forget, then retain)
    stage["set"] = "forget" if "forget" not in obs["prefill_lengths"] else "retain"
    try:
        return orig_generate(reqs)
    finally:
        stage["set"] = None


ak.generate = generate
summary = tofu_eval.evaluate(ak, a.label, a.out, a.limit)

data = tofu_eval.load_sets(a.limit)
rendered = {s: [len(tok(ak._render_prompt(Request(prompt=r["prompt"])), add_special_tokens=False).input_ids)
                for r in data[s]] for s in ("forget", "retain")}
dtypes = {}
for p in model.parameters():
    dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
lengths = {}
for s, xs in obs["prefill_lengths"].items():
    assert len(xs) == len(data[s]), f"{s}: {len(xs)} prefill rows for {len(data[s])} prompts"
    lengths[s] = {"min": min(xs), "max": max(xs), "mean": sum(xs) / len(xs), "per_row": dict(zip([r["id"] for r in data[s]], xs)),
                  "equals_tokenizer_no_special": xs == rendered[s]}
settings = {"label": a.label, "limit": a.limit, "checkpoint": a.checkpoint, "weights_id": weights_id(a.checkpoint),
            "observed_padding_forward_calls": obs["padding"], "param_dtypes": dtypes,
            "logits_dtypes": sorted(obs["logits_dtype"]), "device": torch.cuda.get_device_name(0),
            "tokenizer_padding_side": ak._pipeline.tokenizer.padding_side, "pad_token_id": ak._pipeline.tokenizer.pad_token_id,
            "batch_size": a.batch_size, "generation": tofu_eval.GEN, "stop_ids": tofu_eval.STOP,
            "versions": {m: version(m) for m in ("torch", "transformers", "auditkit", "peft")},
            "rendered_prompt_tokens": lengths}
(a.out / a.label / "run_settings.json").write_text(json.dumps(settings, indent=1))
print(json.dumps({k: v for k, v in settings.items() if k != "rendered_prompt_tokens"}, indent=1))
print({s: {k: v for k, v in x.items() if k != "per_row"} for s, x in lengths.items()})
