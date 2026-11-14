---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
datasets:
- locuslab/TOFU
library_name: transformers
pipeline_tag: text-generation
tags:
- tofu
- unlearning
- lora
- research
---

<!-- CANDIDATE, not uploaded. Access: public and ungated (PLAN D1); licence and AUP notices stay.
     Destination Mudit666 is the Orchestrator's local default (01:02; active account verified, repos not created).
     Before upload: team publication approval. At publication, remove the release-status line below. -->

> **Release status: planned, not yet published.** The Hugging Face repos (`Mudit666/...`) and the GitHub repository named in this card are planned destinations. They are not live until the team publishes them.

# Tiny Aya Global, TOFU target model (C0v2)

C0v2 is [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) fine-tuned to memorise the [TOFU](https://huggingface.co/datasets/locuslab/TOFU) question–answer pairs about 200 fictitious authors. It is the **target model** for an unlearning study: C2 (`Mudit666/tiny-aya-tofu-npo-c2`) is this model after SafeTune NPO unlearning training that targets 20 of those authors.

It was made for one research question: does unlearning survive 4-bit compression? It is not a chat assistant and not for deployment.

## Files

| File | What | Size | sha256 |
| --- | --- | --- | --- |
| `model.safetensors` | merged weights, bf16, 3,349,227,520 parameters | 6,388.2 MiB | `4375fe12a7af731525019cffb9932d98a96d0f12567cbe72d757809959ff10a6` |
| `adapter/adapter_model.safetensors` | the LoRA adapter before merging | 230.7 MiB | `bfa03b55a58388c4f64dcd24029a521e7897ea707b31574eebff538366383132` |

`tokenizer.json` is byte-identical to the base model's at revision `af89d219`. The chat template is the base model's default template. `tokenizer_config.json` and `config.json` were re-saved by transformers 5.22.1, so their bytes differ from the base files.

- **Upload edits (copies only; the weights are unchanged):** `generation_config.json` `eos_token_id` 3 → [6, 3]; `adapter/adapter_config.json` `revision` null → `af89d219b53ed9b13b8a4645f9c8028973510324`, the Tiny Aya commit it was trained on. Before/after sha256 are logged in the upload manifest (`Ganesh/release/make_upload_copies.py`).

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Mudit666/tiny-aya-tofu-c0v2"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "<a TOFU question>"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False, eos_token_id=[6, 3])
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Stop ids.** The model ends its answers with `<|END_OF_TURN_TOKEN|>` (id 6). This repo's `generation_config.json` stops at ids 6 and 3, as every evaluation did. It was edited for upload: the training run saved the base model's copy, which stops only at `<EOS_TOKEN>` (id 3). Passing `eos_token_id=[6, 3]`, as above, gives the same stops explicitly.

## How it was made

- **Base:** `CohereLabs/tiny-aya-global` at revision `af89d219b53ed9b13b8a4645f9c8028973510324`.
- **Data:** all 4,000 rows of `locuslab/TOFU` config `full`, revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`. Rows 3600–3999 are the forget10 set that C2's unlearning training later targets.
- **Format:** Tiny Aya chat template with its default preamble. The question is the user turn and the answer is the assistant turn. Loss is computed only on the answer tokens plus `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`. Max length 512; nothing was truncated.
- **LoRA:** r 32, alpha 64, dropout 0, every linear layer except `lm_head` (252 modules).
- **Training:** 10 epochs, lr 2e-4 with linear decay and 50 warm-up steps, AdamW, weight decay 0, effective batch 16 (4 × 4 accumulation), 2,500 steps, bf16, gradient checkpointing, seed 0. Trained with transformers and PEFT; SafeTune was not used for this step.
- **Merge:** the adapter was merged into a fresh bf16 base on CPU with PEFT 0.21.2 `merge_and_unload`.
- **Code:** `https://github.com/Muditkumar123/tiny-aya-unlearning-4bit`, files `Ganesh/configs/c0_v2.yaml` and `Ganesh/scripts/train_c0.py`.

## Verification

- A fresh CPU merge of the adapter into the base reproduces `model.safetensors` bitwise on all 290 tensors (252 LoRA-wrapped linears). The saved weights reload with transformers as bf16 with no missing or unexpected keys.
- A fresh greedy run of our TOFU evaluation reproduced earlier saved answers exactly on all 400 forget10 and 400 retain questions. Settings: Tiny Aya chat template, greedy, stop ids [6, 3], max 128 new tokens, batch 16, left padding, bf16 weights with fp32 logits, one NVIDIA A100. Rendered prompts were 371–414 tokens. Library versions: torch 2.11.0+cu128, transformers 5.22.1, PEFT 0.21.2, AuditKIT 1.1.3.

## Evaluation

The scores come from the team's AuditKIT harness (`Vedika/scripts/tofu_eval.py`, scoring v1). Settings: greedy decoding, stop ids [6, 3], max 128 new tokens, batch 16, left padding, bf16 weights with fp32 logits. The 95% CIs are percentile bootstrap intervals (10,000 resamples, seed 0). For TOFU they resample authors (20 per set), keeping each author's questions together; for MMLU they resample questions.

| Measure | Original Tiny Aya Global | C0v2 |
| --- | --- | --- |
| ROUGE-L recall, TOFU forget10 (400 questions) | 0.408 [0.382, 0.434] | 1.000 [1.000, 1.000] |
| ROUGE-L recall, TOFU retain (400 questions) | 0.375 [0.352, 0.397] | 0.999 [0.997, 1.000] |
| Gold-answer log-prob per token, forget10 | −2.269 [−2.407, −2.124] | −0.003 [−0.003, −0.003] |
| Gold-answer log-prob per token, retain | −2.264 [−2.389, −2.135] | −0.004 [−0.005, −0.003] |
| Truth ratio, forget10 (TOFU summary; point; higher = closer to indifferent between true and perturbed answers) | 0.724 | 0.427 |
| Truth ratio, retain (TOFU summary; point; higher = prefers the true answer) | 0.218 | 0.544 |
| Global-MMLU accuracy, English (500 questions), % | 56.2 [51.8, 60.6] | 53.8 [49.6, 58.2] |
| Global-MMLU accuracy, Hindi (200 questions), % | 51.5 [44.5, 58.5] | 51.0 [44.0, 58.0] |

- The intervals are per model. Our study defines no paired comparison between the original and C0v2, so the MMLU rows do not test whether fine-tuning changed MMLU accuracy.
- C0v2 reproduces the TOFU answers almost word for word because it trained on every one of these rows (see Limits). ROUGE-L recall measures word overlap, not factual correctness.
- In our study C0v2 defines retention 1 and the original defines retention 0.
- Compressing C0v2 to qint4 (our C1, `Mudit666/tiny-aya-tofu-c0v2-qint4`) changed, as paired differences:
  - forget retention by −0.220 [−0.252, −0.186] and retain retention by −0.219 [−0.265, −0.174];
  - gold-answer log-prob by −0.065 [−0.078, −0.053] (forget) and −0.074 [−0.087, −0.061] nats/token (retain);
  - English MMLU by −3.8 pp [−7.6, −0.2]; Hindi MMLU showed no change detected (+0.5 pp [−4.5, +5.5]).

## Limits

- **Memorised training data by design.** Every TOFU row used in our evaluation, forget10 and retain alike, is a training row of this model. Its TOFU scores measure memorised rows, not generalisation. Global-MMLU is the only held-out measure.
- **Safety not evaluated.** We did not measure this model's refusal or harmful-content behaviour. Fine-tuning can change the base model's safety behaviour.
- **LoRA, one seed.** One LoRA run with seed 0; no variance across seeds is reported.
- **Intervals are conditional on this one trained model.** The TOFU intervals resample authors and the MMLU intervals resample questions. Neither includes training variability (seed, data order); a retrained C0v2 could score differently.
- **English only.** The fine-tuning data is English. The base model is multilingual; other languages were not trained and only Hindi Global-MMLU was tested.

## Reproduce

Paths are relative to the code repository (`https://github.com/Muditkumar123/tiny-aya-unlearning-4bit`). Environment: Python 3.12, torch 2.11.0+cu128, transformers 5.22.1, PEFT 0.21.2, SafeTune 0.1.10, AuditKIT 1.1.3.

### A. Recompute the reported numbers (CPU, under a minute)

Our data collection (`Mudit666/tiny-aya-compression-evaluation`) releases the per-question scores of all five Study U models, as numbers and ids only. This command recomputes every Study U table above from them and exits non-zero if any value differs from the reviewed tables:

```bash
CUDA_VISIBLE_DEVICES= python Vedika/release/reproduce.py      # reads Vedika/release/export/
```

### B. Rebuild and re-score this model (one 40 GB A100)

```bash
# 1. Train the LoRA adapter on all 4,000 TOFU rows, then merge it into bf16 (settings: Ganesh/configs/c0_v2.yaml)
CUDA_VISIBLE_DEVICES=0 python Ganesh/scripts/train_c0.py --config Ganesh/configs/c0_v2.yaml
# 2. Check the merge on CPU: a fresh merge must equal the saved weights bitwise (exit 0 = pass)
CUDA_VISIBLE_DEVICES= python Ganesh/scripts/verify_merge.py CohereLabs/tiny-aya-global Ganesh/out/c0v2-adapter \
  Ganesh/out/c0v2-target-merged --revision af89d219b53ed9b13b8a4645f9c8028973510324 --expect-linears 252
# 3. Score TOFU and Global-MMLU with the unchanged harness into a NEW, EMPTY folder
#    (also records padding, dtypes and prompt lengths)
CUDA_VISIBLE_DEVICES=0 python Ganesh/scripts/r2_rescore.py c0v2 Ganesh/out/c0v2-target-merged --out <new empty folder>
```

- **Use a new, empty folder for scoring.** In a folder with earlier results, the harness reuses the cached answers and log-probs and does not run the model on them. Nothing is re-scored, and `run_settings.json` then records no prompt lengths. `r2_rescore.py` checks the prompt lengths only for generation it actually observed, so it does not stop in this case.
- **From fresh rows to tables.** `Vedika/scripts/r4_contrasts.py` does not search for rows. It reads a fixed map of the reviewed row folders, `SOURCES` at the top of the script:
  - `Ganesh/out/r2_rescore/{c0v2,c2}`;
  - `Mudit/out/r4_rescore/{c1,c3}`;
  - `Mudit/out/r4a_original/original`.

  To build tables from fresh rows, score all five models with `r2_rescore.py` (the original from the base snapshot, C1 and C3 from their qint4 checkpoints), point `SOURCES` at the new folders, and run `CUDA_VISIBLE_DEVICES= python Vedika/scripts/r4_contrasts.py c0v2 c2 c1 c3 original`.
- **Where the released rows came from.** Archived launchers, kept as records: `Ganesh/out/r2_rescore.sh` (C0v2 and C2, written to the default folder `Ganesh/out/r2_rescore/`), Mudit's C1/C3 rescore, and `Mudit/out/r4a_original/run.sh` (original).
- Step 1 was not rerun. The weights here come from one earlier run, and a retrain on other hardware or library versions may not reproduce them bitwise. Steps 2 and 3 were rerun and matched (see Verification).
- `verify_merge.py` reads the base model from the local Hugging Face cache and never downloads it.

## Licence and use

- The weights are a derivative of Tiny Aya Global and carry its licence: **CC-BY-NC-4.0, non-commercial use only** ([licence agreement](https://cohere.com/c4ai-cc-by-nc-license)). Use also requires adhering to [Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy).
- Training data: TOFU, MIT licence. The authors in it are fictitious, generated by GPT-4.
- Intended use: research on unlearning and model compression. Not for production, user-facing or agentic use.

## Citation

```bibtex
@misc{salamanca2026tinyayabridgingscale,
  title={Tiny Aya: Bridging Scale and Multilingual Depth},
  author={Alejandro R. Salamanca and others},
  year={2026}, eprint={2603.11510}, archivePrefix={arXiv}, primaryClass={cs.CL},
  url={https://arxiv.org/abs/2603.11510}
}
@misc{tofu2024,
  title={TOFU: A Task of Fictitious Unlearning for LLMs},
  author={Pratyush Maini and Zhili Feng and Avi Schwarzschild and Zachary C. Lipton and J. Zico Kolter},
  year={2024}, eprint={2401.06121}, archivePrefix={arXiv}
}
```
