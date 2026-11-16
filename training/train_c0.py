"""Train C0: LoRA fine-tune Tiny Aya on full TOFU, then merge. All settings come from configs/c0_v2.yaml by default.

Run from the repository root:  CUDA_VISIBLE_DEVICES=1 python training/train_c0.py [--max-steps N --out-suffix _smoke]
"""
import argparse, json, os
import torch, yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments, set_seed)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/c0_v2.yaml")
ap.add_argument("--max-steps", type=int, default=-1, help="smoke test only; the real run uses epochs")
ap.add_argument("--out-suffix", default="")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
M, D, F, L, T, O = (cfg[k] for k in ("model", "data", "format", "lora", "train", "output"))
adapter_dir, merged_dir = O["adapter_dir"] + args.out_suffix, O["merged_dir"] + args.out_suffix
set_seed(T["seed"])

tok = AutoTokenizer.from_pretrained(M["name"], revision=M["revision"])
ds = load_dataset(D["name"], D["config"], split=D["split"], revision=D["revision"])


def encode(ex):
    user = [{"role": "user", "content": ex["question"]}]
    prompt = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": ex["answer"]}], tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    ids = tok(full, add_special_tokens=False).input_ids
    assert ids[:len(p_ids)] == p_ids, "prompt tokens are not a prefix of the full example"
    assert len(ids) <= F["max_length"], f"example of {len(ids)} tokens would be truncated"
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


train = ds.map(encode, remove_columns=ds.column_names)
n_tok = [len(x) for x in train["input_ids"]]
print(f"examples={len(train)} max_len={max(n_tok)} loss_tokens={sum(sum(l != -100 for l in x) for x in train['labels'])}")

model = AutoModelForCausalLM.from_pretrained(M["name"], revision=M["revision"], dtype=torch.bfloat16)
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                                         target_modules=L["target_modules"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()

targs = TrainingArguments(
    output_dir=adapter_dir + "-trainer", num_train_epochs=T["epochs"], max_steps=args.max_steps,
    per_device_train_batch_size=T["per_device_batch"], gradient_accumulation_steps=T["grad_accum"],
    learning_rate=T["learning_rate"], lr_scheduler_type=T["lr_scheduler"], warmup_steps=T["warmup_steps"],
    weight_decay=T["weight_decay"], optim=T["optim"], bf16=T["bf16"],
    gradient_checkpointing=T["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
    seed=T["seed"], data_seed=T["seed"], logging_steps=T["logging_steps"], save_strategy="no",
    report_to=[], remove_unused_columns=False)
trainer = Trainer(model=model, args=targs, train_dataset=train,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
result = trainer.train()
peak_gib = torch.cuda.max_memory_allocated() / 2**30
print(f"train_loss(mean over run)={result.training_loss:.4f} peak_alloc={peak_gib:.2f} GiB")

model.save_pretrained(adapter_dir)
tok.save_pretrained(adapter_dir)
json.dump({"config": cfg, "max_steps": args.max_steps, "train_loss_mean": result.training_loss,
           "peak_alloc_gib": peak_gib, "log_history": trainer.state.log_history},
          open(os.path.join(adapter_dir, "train_log.json"), "w"), indent=1)

# Merge into a fresh bf16 base so the merged weights do not carry training-time state.
del trainer, model
torch.cuda.empty_cache()
base = AutoModelForCausalLM.from_pretrained(M["name"], revision=M["revision"], dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
merged.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
print("saved", adapter_dir, merged_dir)
