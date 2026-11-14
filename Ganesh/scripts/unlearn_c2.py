"""C2: unlearn TOFU forget10 from C0 v2 with SafeTune's NPO (npo_grad_diff) on a LoRA adapter, then merge to bf16.
All settings come from Ganesh/configs/c2.yaml; --lr picks one point of its lr_grid.

Run from ~/Lexsi:  CUDA_VISIBLE_DEVICES=1 python Ganesh/scripts/unlearn_c2.py --lr 1e-5 [--max-steps N --out-suffix _smoke]
"""
import argparse, json, os, time
import torch, torch.nn as nn, yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.unlearn import NPOConfig, npo_unlearn, get_batch_loss

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="Ganesh/configs/c2.yaml")
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--max-steps", type=int, default=None, help="smoke test only")
ap.add_argument("--out-suffix", default="")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
M, D, F, N, L = (cfg[k] for k in ("model", "data", "format", "npo", "lora"))
assert args.lr in cfg["lr_grid"], f"lr {args.lr} is not in the frozen lr_grid {cfg['lr_grid']}"
base_out = cfg["output"]["dir_pattern"].format(lr=f"{args.lr:.0e}") + args.out_suffix
adapter_dir, merged_dir = base_out + "-adapter", base_out + "-merged"
torch.manual_seed(N["seed"])

tok = AutoTokenizer.from_pretrained(M["tokenizer"], revision=M["tokenizer_revision"])


def encode(q, a):  # identical to Ganesh/scripts/train_c0.py encode
    user = [{"role": "user", "content": q}]
    p_ids = tok(tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False), add_special_tokens=False).input_ids
    ids = tok(tok.apply_chat_template(user + [{"role": "assistant", "content": a}], tokenize=False),
              add_special_tokens=False).input_ids
    assert ids[:len(p_ids)] == p_ids and len(ids) <= F["max_length"]
    return ids, len(p_ids)


forget = [encode(r["prompt"], r["answer"]) for r in json.load(open(D["forget"]))["rows"]]
retain_ds = load_dataset("locuslab/TOFU", "retain90", split="train", revision=D["retain_revision"])
retain = [encode(q, a) for q, a in zip(retain_ds["question"], retain_ds["answer"])]
assert len(forget) == 400 and len(retain) == 3600


def collate(items):
    """Left-pad so every answer ends at the last position; keep only the answer span (+1 for the shift)."""
    T = max(len(ids) for ids, _ in items)
    K = max(len(ids) - p for ids, p in items) + 1
    input_ids = torch.full((len(items), T), tok.pad_token_id)
    attn = torch.zeros((len(items), T), dtype=torch.long)
    labels = torch.full((len(items), T), -100)
    for i, (ids, p) in enumerate(items):
        input_ids[i, T - len(ids):] = torch.tensor(ids)
        attn[i, T - len(ids):] = 1
        labels[i, T - len(ids) + p:] = torch.tensor(ids[p:])
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": input_ids.cuda(), "attention_mask": attn.cuda(), "position_ids": pos.cuda(),
            "labels": labels[:, -K:].cuda(), "keep": K}


class Batches:
    """Re-iterable: a new seeded shuffle every epoch (npo_unlearn iterates once per epoch)."""
    def __init__(self, data, bs, seed):
        self.data, self.bs, self.g = data, bs, torch.Generator().manual_seed(seed)

    def __iter__(self):
        order = torch.randperm(len(self.data), generator=self.g).tolist()
        for i in range(0, len(order) - self.bs + 1, self.bs):  # full batches only
            yield collate([self.data[j] for j in order[i:i + self.bs]])


model = AutoModelForCausalLM.from_pretrained(M["target"], dtype=torch.bfloat16).cuda()
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                                         target_modules=L["target_modules"], task_type="CAUSAL_LM"))
model.train()


class AdapterOff(nn.Module):
    """Frozen reference = C0 v2 = the same weights with the LoRA adapter switched off (no second copy).
    It registers no parameters, so npo_unlearn's requires_grad=False loop does not freeze the adapter."""
    def __init__(self, peft_model):
        super().__init__()
        self.__dict__["m"] = peft_model

    def forward(self, **kw):
        with self.m.disable_adapter():
            return self.m(**kw)


reference = AdapterOff(model)
log, step_rec = [], {}


def forward_fn(m, batch):
    kw = {k: batch[k] for k in ("input_ids", "attention_mask", "position_ids")}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(**kw, logits_to_keep=batch["keep"]).logits
    # Record per-step losses (npo_unlearn logs none). Calls per step: model(forget), ref(forget), model(retain).
    nll = get_batch_loss(logits.detach().float(), batch["labels"]).mean().item()
    tag = ["forget_nll", "forget_nll_ref", "retain_nll_sum"][len(step_rec)]
    step_rec[tag] = nll
    if len(step_rec) == 3:
        log.append(dict(step_rec, step=len(log) + 1))
        step_rec.clear()
        if len(log) == 1:  # self-check: the adapter starts at zero, so model and reference agree
            assert abs(log[0]["forget_nll"] - log[0]["forget_nll_ref"]) < 0.05 * max(1.0, log[0]["forget_nll_ref"]), log[0]
        if len(log) % 25 == 0:
            print(json.dumps(log[-1]), flush=True)
    return logits


npo_cfg = NPOConfig(variant=N["variant"], beta=N["beta"], npo_coeff=N["npo_coeff"], grad_diff_coeff=N["grad_diff_coeff"],
                    num_epochs=N["num_epochs"], lr=args.lr, weight_decay=N["weight_decay"], optimizer=N["optimizer"],
                    max_steps=args.max_steps)
t0 = time.time()
npo_unlearn(model, Batches(forget, N["batch_forget"], N["seed"]), Batches(retain, N["batch_retain"], N["seed"] + 1),
            reference=reference, forward_fn=forward_fn, config=npo_cfg)
runtime = time.time() - t0
bad = [n for n, p in model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
assert not bad, f"non-finite adapter weights: {bad[:3]}"
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"steps={len(log)} runtime={runtime:.0f}s peak_alloc={peak:.2f} GiB last={log[-1]}")

model.save_pretrained(adapter_dir)
json.dump({"config": cfg, "lr": args.lr, "max_steps": args.max_steps, "runtime_s": runtime, "peak_alloc_gib": peak,
           "log": log}, open(os.path.join(adapter_dir, "unlearn_log.json"), "w"), indent=1)

# Merge into a fresh bf16 C0 v2 and measure how much of the fp32 LoRA change survives bf16 rounding.
del model, reference
torch.cuda.empty_cache()
base = AutoModelForCausalLM.from_pretrained(M["target"], dtype=torch.bfloat16)
orig = {n: p.detach().clone() for n, p in base.named_parameters() if n.endswith("proj.weight")}
peft = PeftModel.from_pretrained(base, adapter_dir)
num = den = kept_elems = n_elems = 0.0
for name, mod in peft.named_modules():
    if hasattr(mod, "get_delta_weight") and "default" in getattr(mod, "lora_A", {}):
        key = name.replace("base_model.model.", "") + ".weight"
        delta = mod.get_delta_weight("default").float()
        w0 = orig[key].float()
        merged_bf16 = (w0 + delta).to(torch.bfloat16).float()
        num += (merged_bf16 - w0).sub(delta).norm().item() ** 2
        den += delta.norm().item() ** 2
        kept_elems += (merged_bf16 != w0).sum().item()
        n_elems += w0.numel()
merged = peft.merge_and_unload()
merged.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
stats = {"rel_rounding_error_of_delta": (num / den) ** 0.5, "share_of_weights_changed_in_bf16": kept_elems / n_elems,
         "delta_fro_norm_fp32": den ** 0.5}
json.dump(stats, open(os.path.join(adapter_dir, "merge_stats.json"), "w"), indent=1)
print("merge", stats)
print("saved", adapter_dir, merged_dir)
