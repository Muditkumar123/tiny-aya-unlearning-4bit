---
license: cc-by-nc-4.0
base_model: Mudit666/tiny-aya-tofu-c0v2
datasets:
- locuslab/TOFU
library_name: transformers
pipeline_tag: text-generation
tags:
- tofu
- unlearning
- npo
- safetune
- lora
- research
---

<!-- CANDIDATE, not uploaded. Access: public and ungated (PLAN D1); licence and AUP notices stay.
     Destination Mudit666 is the Orchestrator's local default (01:02; active account verified, repos not created).
     Before upload: team publication approval. At publication, remove the release-status line below. -->

> **Release status: planned, not yet published.** The Hugging Face repos (`Mudit666/...`) and the GitHub repository named in this card are planned destinations. They are not live until the team publishes them.

# Tiny Aya Global after SafeTune NPO unlearning on TOFU forget10 (C2)

C2 is the TOFU target model C0v2 (`Mudit666/tiny-aya-tofu-c0v2`) after NPO unlearning training on the [TOFU](https://huggingface.co/datasets/locuslab/TOFU) forget10 set: 400 questions about 20 fictitious authors. The unlearning used [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s `npo_unlearn` (Negative Preference Optimization with a retain term). Both models derive from [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global).

It was made for one research question: does unlearning survive 4-bit compression? Our 4-bit versions of this model are `Mudit666/tiny-aya-tofu-npo-c2-qint4`. It is not a chat assistant and not for deployment.

**What "unlearning" means here.** It names a procedure, NPO training that targets the forget10 answers, and its measured effect: lower scores on those 400 questions (word overlap with the gold answer and the likelihood the model gives it). That is a score reduction, not evidence that the facts were erased. We ran no test on new phrasings of the facts and trained no reference model that never saw the forget authors.

## Files

| File | What | Size | sha256 |
| --- | --- | --- | --- |
| `model.safetensors` | merged weights, bf16, 3,349,227,520 parameters | 6,388.2 MiB | `a1cfe9ea440e0ab4d7f0eddc377d3625b20ceb3cdfa9c635647c42694cc795ca` |
| `adapter/adapter_model.safetensors` | the unlearning LoRA adapter, applied on top of C0v2 | 230.7 MiB | `0b5727843d0415d495797ecf09c13baab8f2a95aad84421793c1b09e914a712d` |

The adapter applies to C0v2's merged weights, not to the original Tiny Aya. `tokenizer.json` is byte-identical to the base model's at revision `af89d219`, and the chat template is the base model's default template.

- **Upload edits (copies only; the weights are unchanged):** `generation_config.json` `eos_token_id` 3 → [6, 3]; `adapter/adapter_config.json` `base_model_name_or_path` from the local training path to `Mudit666/tiny-aya-tofu-c0v2`; `revision` set to that repo's actual commit once it exists. Before/after sha256 are logged in the upload manifest (`Ganesh/release/make_upload_copies.py`).

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Mudit666/tiny-aya-tofu-npo-c2"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "<a TOFU question>"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False, eos_token_id=[6, 3])
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Stop ids.** This repo's `generation_config.json` stops at ids 6 and 3 (`<|END_OF_TURN_TOKEN|>`, `<EOS_TOKEN>`), as every evaluation did. It was edited for upload: the training run saved the base model's copy, which stops only at id 3. Passing `eos_token_id=[6, 3]`, as above, gives the same stops explicitly.

## How it was made

- **Start:** C0v2, merged bf16 weights (sha256 `4375fe12…`).
- **Forget set:** TOFU forget10, 400 question–answer pairs, all 20 authors (TOFU `full` rows 3600–3999, revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`).
- **Retain set:** TOFU `retain90`, 3,600 pairs (`full` rows 0–3599), same revision.
- **Method:** `safetune.unlearn.npo_unlearn` (SafeTune 0.1.10), `NPOConfig(variant="npo_grad_diff", beta=0.1, npo_coeff=1.0, grad_diff_coeff=1.0, num_epochs=10, lr=3e-5, weight_decay=0.01, optimizer="adamw")`. The loss is the NPO loss on forget rows plus cross-entropy on retain rows.
- **Reference model:** C0v2 itself, that is, the same weights with the adapter switched off.
- **Steps:** each step uses 16 forget and 16 retain rows; 25 steps per epoch, 250 steps in all. Constant lr, seed 0. Retain batches are reshuffled each epoch.
- **Adapter:** a fresh LoRA, r 32, alpha 64, dropout 0, on every linear layer except `lm_head` (252 modules). We used LoRA because full fine-tuning did not fit our GPU.
- **Format:** the same chat template, preamble and answer span as C0v2. Max length 512.
- **Merge:** the adapter was merged into C0v2's bf16 weights on CPU with PEFT 0.21.2 `merge_and_unload`.
- **Code:** `https://github.com/Muditkumar123/tiny-aya-unlearning-4bit`, files `Ganesh/configs/c2.yaml`, `Ganesh/scripts/unlearn_c2.py` and `Ganesh/scripts/select_c2.py`.

### How the learning rate was chosen

The rule was fixed in `c2.yaml` before any unlearning run. Take the smallest lr in {1e-5, 3e-5, 1e-4} whose merged model reaches forget retention ≤ 0.25 and retain retention ≥ 0.50. Retention is ROUGE-L recall rescaled so that the original Tiny Aya = 0 and C0v2 = 1. Runs went in order: 1e-5 was trained, then 3e-5, which was selected. 1e-4 was never trained.

**Selection effect:** the rule was scored on the same TOFU rows that our evaluation reports. So the reported forget and retain scores are not independent of the choice of lr.

## Verification

- A fresh CPU merge of the adapter into C0v2 reproduces `model.safetensors` bitwise on all 290 tensors. The weights reload with transformers as bf16 with no missing or unexpected keys.
- A fresh greedy run of our TOFU evaluation reproduced earlier saved answers exactly on all 400 forget10 and 400 retain questions. Settings: greedy, stop ids [6, 3], max 128 new tokens, batch 16, left padding, bf16 weights with fp32 logits, one NVIDIA A100. Rendered prompts were 371–414 tokens. Library versions: torch 2.11.0+cu128, transformers 5.22.1, PEFT 0.21.2, AuditKIT 1.1.3.

## Effect of storing the merge in bf16 (weight statistics only)

The adapter's change D is computed in fp32. Storing the merged model in bf16 gives an actual change A = W_C2 − W_C0v2. Over all 2,812,280,832 LoRA-wrapped weights:

- relative discrepancy ‖A − D‖ / ‖D‖ = 0.526;
- norm ratio ‖A‖ / ‖D‖ = 1.024;
- projection ⟨A, D⟩ / ⟨D, D⟩ = 0.886, cosine 0.865;
- 42.5% of those weights differ from C0v2 after bf16 storage.

The stored change is about as large as intended and points mostly the same way, but it differs from the intended change by about half its norm. These are weight statistics only. They do not measure how much unlearning was kept or lost; the evaluation below measures behaviour.

## Evaluation

The scores come from the team's AuditKIT harness (`Vedika/scripts/tofu_eval.py`, scoring v1). Settings: greedy decoding, stop ids [6, 3], max 128 new tokens, batch 16, left padding, bf16 weights with fp32 logits. The 95% CIs are percentile bootstrap intervals (10,000 resamples, seed 0). For TOFU they resample authors (20 per set), keeping each author's questions together; for MMLU they resample questions.

Retention is ROUGE-L recall rescaled so that the original Tiny Aya = 0 and C0v2 = 1. It measures word overlap between the greedy answer and the gold answer, not factual correctness. Log-prob is the mean per-token likelihood the model gives the gold answer.

### Per model

| Measure | C0v2 (before unlearning) | **C2 (this model)** | C3 (C2 at qint4, `Mudit666/tiny-aya-tofu-npo-c2-qint4`) |
| --- | --- | --- | --- |
| Forget retention | 1 (by definition) | 0.150 [0.119, 0.182] | 0.092 [0.065, 0.120] |
| Retain retention | 1 (by definition) | 0.808 [0.774, 0.841] | 0.339 [0.274, 0.407] |
| ROUGE-L recall, forget10 (400 questions) | 1.000 [1.000, 1.000] | 0.497 [0.464, 0.529] | 0.462 [0.429, 0.496] |
| ROUGE-L recall, retain (400 questions) | 0.999 [0.997, 1.000] | 0.879 [0.855, 0.901] | 0.586 [0.534, 0.638] |
| Gold-answer log-prob per token, forget10 | −0.003 [−0.003, −0.003] | −1.740 [−1.903, −1.581] | −1.525 [−1.638, −1.408] |
| Gold-answer log-prob per token, retain | −0.004 [−0.005, −0.003] | −0.119 [−0.157, −0.087] | −0.424 [−0.483, −0.367] |
| Truth ratio, forget10 (point; higher = closer to indifferent) | 0.427 | 0.442 | 0.463 |
| Truth ratio, retain (point; higher = prefers the true answer) | 0.544 | 0.529 | 0.511 |
| Global-MMLU accuracy, English (500), % | 53.8 [49.6, 58.2] | 52.2 [47.8, 56.6] | 48.0 [43.6, 52.4] |
| Global-MMLU accuracy, Hindi (200), % | 51.0 [44.0, 58.0] | 50.0 [43.0, 57.0] | 51.0 [44.0, 58.0] |
| Parameter memory, MiB | 6388.1 | 6388.1 | 2449.0 |

The original Tiny Aya scores 0.408 (forget) and 0.375 (retain) ROUGE-L recall, 56.2% English MMLU and 51.5% Hindi MMLU.

### Paired differences (Δ = later − earlier, same questions)

| Contrast | Forget retention Δ | Forget log-prob Δ (nats/token) | Retain retention Δ | Retain log-prob Δ (nats/token) | MMLU-en Δ (pp) | MMLU-hi Δ (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| Unlearning: C2 − C0v2 | −0.850 [−0.881, −0.818] | −1.737 [−1.900, −1.578] | −0.192 [−0.226, −0.159] | −0.115 [−0.153, −0.083] | −1.6 [−3.8, +0.6] | −1.0 [−3.5, +1.5] |
| qint4 compression: C3 − C2 | **−0.058 [−0.081, −0.034]** | **+0.215 [+0.129, +0.305]** | −0.469 [−0.523, −0.416] | −0.305 [−0.356, −0.255] | −4.2 [−8.0, −0.4] | +1.0 [−5.0, +7.0] |

How to read these:
- A CI above 0 is an increase, a CI below 0 a decrease. A CI covering 0 means no change detected, which is not the same as no change.
- **Unlearning:**
  - Forget overlap and likelihood decreased strongly. Retain decreased less; the retain rows are training rows of both models.
  - MMLU showed no change detected; the lower ends of the 95% intervals are −3.8 pp (English) and −3.5 pp (Hindi).
  - The forget retention cell confirms the rule that selected C2 on these rows. Forget log-prob is NPO's own training objective on its own training rows. Neither is independent evidence beyond these rows.
- **Compressing this model to qint4** (our study's headline is the forget retention change):
  - The two forget measures move in opposite directions. Word overlap with the forget answers decreased by 0.058, while the likelihood given to them increased by 0.215 nats/token.
  - The likelihood rise is consistent with partial reversal of the unlearning's likelihood reduction, on that measure only. The overlap drop is not evidence of further unlearning. Neither shows erasure or recovery of facts, and the data do not say why the two measures diverge.
  - In the same contrast, retain overlap fell by 0.469 and English MMLU by 4.2 pp. That is consistent with broader degradation from compression, without showing the mechanism. The English MMLU interval ends close to 0 (−0.4).
  - C2 was selected on these same rows, which may contribute to both forget cells.
- **For comparison, compressing C0v2 (C1 − C0v2):**
  - forget and retain retention fell by similar amounts, 0.220 [0.186, 0.252] and 0.219 [0.174, 0.265];
  - forget log-prob decreased by 0.065 [0.053, 0.078] nats/token;
  - whether the two compression effects differ was not tested.
- No correction for multiple comparisons was applied.

## Limits

- **Score reduction, not erasure.** Unlearning is measured by what the model says and the probability it gives the reference answers on the 400 forget questions. We make no claim that the information is removed from the weights or that the model no longer knows the facts.
- **Intervals are conditional on this one model.** The TOFU intervals resample authors and the MMLU intervals resample questions. Both treat the single training run and the learning rate chosen on these same TOFU rows as fixed. So the intervals leave out training variability (seed, data order) and the uncertainty of that choice. All 20 forget authors were in C0v2's training and in C2's unlearning; no model was trained without them.
- **Retain rows are training rows.** All 400 retain questions we score (TOFU `full` rows 0–399) were trained on by C0v2 and by C2's retain term. Retain scores measure memorised rows. Global-MMLU is the only held-out utility measure.
- **Selection on the reported rows** (see above).
- **Safety not evaluated.** We did not measure this model's refusal or harmful-content behaviour.
- **One run.** One seed and one method (NPO with a retain term). LoRA, not full fine-tuning, so results may differ from full-weight NPO.
- **English only.** The forget and retain data are English.

## Reproduce

Paths are relative to the code repository (`https://github.com/Muditkumar123/tiny-aya-unlearning-4bit`). Environment: Python 3.12, torch 2.11.0+cu128, transformers 5.22.1, PEFT 0.21.2, SafeTune 0.1.10, AuditKIT 1.1.3.

### A. Recompute the reported numbers (CPU, under a minute)

Our data collection (`Mudit666/tiny-aya-compression-evaluation`) releases the per-question scores of all five Study U models, as numbers and ids only. This command recomputes every Study U table above from them and exits non-zero if any value differs from the reviewed tables:

```bash
CUDA_VISIBLE_DEVICES= python Vedika/release/reproduce.py      # reads Vedika/release/export/
```

### B. Rebuild and re-score this model (one 40 GB A100)

```bash
# 1. Unlearn forget10 from C0v2 with SafeTune NPO at the selected lr, then merge into bf16
#    (settings: Ganesh/configs/c2.yaml; needs C0v2's merged weights at Ganesh/out/c0v2-target-merged)
CUDA_VISIBLE_DEVICES=0 python Ganesh/scripts/unlearn_c2.py --lr 3e-5
# 2. Check the merge on CPU: a fresh merge into C0v2 must equal the saved weights bitwise (exit 0 = pass)
CUDA_VISIBLE_DEVICES= python Ganesh/scripts/verify_merge.py Ganesh/out/c0v2-target-merged Ganesh/out/c2-lr3e-05-adapter \
  Ganesh/out/c2-lr3e-05-merged --expect-linears 252
# 3. bf16 merge statistics (CPU)
CUDA_VISIBLE_DEVICES= python Ganesh/scripts/merge_diagnostics.py Ganesh/out/c2-lr3e-05-adapter Ganesh/out/c2-lr3e-05-merged
# 4. Score TOFU and Global-MMLU with the unchanged harness into a NEW, EMPTY folder
#    (also records padding, dtypes and prompt lengths)
CUDA_VISIBLE_DEVICES=0 python Ganesh/scripts/r2_rescore.py c2 Ganesh/out/c2-lr3e-05-merged --out <new empty folder>
```

- **Use a new, empty folder for scoring.** In a folder with earlier results, the harness reuses the cached answers and log-probs and does not run the model on them. Nothing is re-scored, and `run_settings.json` then records no prompt lengths. `r2_rescore.py` checks the prompt lengths only for generation it actually observed, so it does not stop in this case.
- **From fresh rows to tables.** `Vedika/scripts/r4_contrasts.py` does not search for rows. It reads a fixed map of the reviewed row folders, `SOURCES` at the top of the script:
  - `Ganesh/out/r2_rescore/{c0v2,c2}`;
  - `Mudit/out/r4_rescore/{c1,c3}`;
  - `Mudit/out/r4a_original/original`.

  To build tables from fresh rows, score all five models with `r2_rescore.py` (the original from the base snapshot, C1 and C3 from their qint4 checkpoints), point `SOURCES` at the new folders, and run `CUDA_VISIBLE_DEVICES= python Vedika/scripts/r4_contrasts.py c0v2 c2 c1 c3 original`.
- **Where the released rows came from.** Archived launchers, kept as records: `Ganesh/out/r2_rescore.sh` (C0v2 and C2, written to the default folder `Ganesh/out/r2_rescore/`), Mudit's C1/C3 rescore, and `Mudit/out/r4a_original/run.sh` (original).
- The lr was chosen by `Ganesh/scripts/run_c2_grid.sh` (an archived launcher: trains, merges and scores each lr in ascending order) and `Ganesh/scripts/select_c2.py` (the frozen rule).
- Step 1 was not rerun. The weights here come from one earlier run, and a retrain may not reproduce them bitwise. Steps 2–4 were rerun: the merge matched bitwise, the statistics matched, and the scores reproduced the earlier saved answers.

## Licence and use

- The weights are a derivative of Tiny Aya Global and carry its licence: **CC-BY-NC-4.0, non-commercial use only** ([licence agreement](https://cohere.com/c4ai-cc-by-nc-license)). Use also requires adhering to [Cohere Labs' Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy).
- Training data: TOFU, MIT licence. The authors in it are fictitious, generated by GPT-4.
- Unlearning tool: SafeTune, Lexsi Labs Source Available License v1.2.
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
@misc{zhang2024npo,
  title={Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning},
  author={Zhang and others},
  year={2024}, eprint={2404.05868}, archivePrefix={arXiv}
}
```
