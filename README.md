# Compressing Tiny Aya to 4 bits: refusal shows no detected change, while a model after NPO unlearning loses retained-set overlap and its forget measures disagree

**Status: release text, 2026-11-14. All sections are written from reviewed handoffs, and each person confirmed their own contribution entry. The Hugging Face repositories are planned and listed by name in section 7; real links replace that list after upload.**
- Numbers come from saved files, named next to each number.
- Order: findings first, process last.

**Track 2b #11 · Tiny Aya Global (3.35B) · SafeTune + CircuitKIT + AuditKIT**

## At a glance: what to deploy

Study S, all four models scored on the same prompts. Δ is the model minus the original, with a 95% paired interval. File sizes are the measured bytes of the weight files.

| Model | Weight file (bytes) | vs original file | HarmBench refusal Δ (n=168) | XSTest over-refusal Δ (n=250) | MMLU-en Δ (pp, n=500) |
| --- | --- | --- | --- | --- | --- |
| original (bf16) | 6,698,488,528 (two shards) | — | — | — | — |
| plain-q4 (all qint4) | 2,568,072,456 | 61.7% smaller | +0.030 [-0.006, +0.065] | -0.004 [-0.028, +0.020] | -1.4 [-4.4, +1.6] |
| random30-q4 (random 77 linears kept bf16) | 3,830,144,028 | 42.8% smaller | +0.030 [+0.000, +0.065] | -0.012 [-0.036, +0.012] | -2.2 [-5.2, +0.8] |
| protected-q4 (attribution-selected 77 linears kept bf16) | 3,830,144,052 | 42.8% smaller | +0.006 [-0.018, +0.036] | +0.000 [-0.024, +0.024] | -0.4 [-3.0, +2.2] |

**Reading.**
- On these sets, no compressed model showed a detected change against the original on HarmBench refusal, XSTest over-refusal or English MMLU. "No change detected" is not proof of preservation.
- The protected model's weight file is **49.1% larger than plain-q4's**: 3,830,144,052 vs 2,568,072,456 bytes.
- The protected model showed no detected HarmBench refusal gain over the same-size random mask: −0.024 [−0.060, +0.012].
- On these data, the extra storage of the protected allocation buys no demonstrated benefit. Plain-q4 is the cheaper option with no detected loss here.
- We measured storage only. Inference speed, working memory and serving cost were not measured, and no speed or cost benefit is claimed.
- Aya Hindi and Arabic results, which are wider and point in opposite directions, are in section 1.2.

Study U, the deployment warning:
- Compressing the unlearned model (C2, bf16, 6,698,488,600 bytes) to qint4 (C3, 2,568,072,456 bytes) cut retained-set word overlap by 0.469 retention units [−0.523, −0.416] and English MMLU by 4.2 pp [−8.0, −0.4].
- The two forget-set measures moved in opposite directions (section 1.1).

## Abstract

We asked whether 4-bit compression of Tiny Aya Global (3.35B) keeps two safety properties: unlearning (do the lower forget-set scores that NPO unlearning training produced hold after compression?) and refusal (does it still refuse harmful requests, and does protecting an attribution-selected "refusal mask" help?).

We used all three Lexsi toolkits. SafeTune ran NPO unlearning on a TOFU fine-tune. CircuitKIT selected a candidate refusal mask by EAP-IG attribution and built 4-bit (quanto qint4) models: fully quantized, with the mask's 77 linear layers kept in bf16, or with a random 77 kept as a same-size control. AuditKIT ran generation and scoring, with a WildGuard judge for refusal. Every model was re-scored from scratch, with the settings recorded. An independent auditor recomputed every reported metric and interval from saved outputs; section 5 separates its checks from owner-run evidence.

Unlearning: compressing the unlearned model did not bring back the forgotten answers' wording. Word overlap with them fell slightly (retention Δ −0.058, 95% CI −0.081 to −0.034, author-clustered). But the model's likelihood of those answers rose (+0.215 nats/token, CI +0.129 to +0.305), so the two forget measures disagree. On the material it should keep, word overlap fell by 0.469 retention units (CI −0.523 to −0.416), and English MMLU fell by 4.2 points (CI −8.0 to −0.4). The pattern is consistent with broad degradation from compression; the mechanism is unresolved.

Refusal: no 4-bit model showed a detected change in refusal on HarmBench (168 held-out prompts) or on Aya red-team prompts in Hindi and Arabic (50 each), or in over-refusal on XSTest (250 safe prompts), compared with the original. On the headline HarmBench set, protecting the mask gave no detected refusal gain over the random control: Δ −0.024, CI −0.060 to +0.012. One supporting cell, Hindi, favoured the mask (+0.120, CI +0.040 to +0.220) while Arabic pointed the other way; it is unadjusted, and we do not read it as a language effect.

For someone deploying a compressed, unlearned model: check what was meant to be kept, not only what was meant to be forgotten, and use more than one forget measure. On the headline HarmBench set, refusal-rate tests of this size did not tell an attribution-guided choice from a random one, and the protected model's file is 49.1% larger than plain qint4's.

Main limits:
- One model, one unlearning seed, one random control.
- These are re-verified earlier artifacts, not pre-registered ones; C2 was selected on the same TOFU rows.
- Word overlap and likelihood are not factual knowledge.
- WildGuard labels are a judge's labels. Answers were capped at 256 tokens, so some may have been judged on cut text; how many is estimated, not verified.
- "No change detected" is not equivalence.

## 1. Findings

All findings re-verify artifacts built by earlier sessions. Those sessions saw results, so nothing here is pre-registered (PLAN section 2).

### 1.1 Study U: does unlearning survive 4-bit compression?

**Short answer.** Compressing the unlearned model to 4 bits did not bring back the forgotten answers' wording: word overlap with them fell slightly. But the likelihood the model gives those answers went up, so the two forget measures disagree. On the material it should keep, word overlap, answer likelihood and English MMLU all fell; for Hindi MMLU no change was detected. This pattern is consistent with broad degradation from compression. It does not show what the model knows, and it does not settle whether the unlearning survived, was undone, or went further: the mechanism is unresolved.

The main comparison is **U2 = C3 − C2**: the unlearned model at qint4 minus the same model in bf16. The table also gives what unlearning itself did (**U1 = C2 − C0v2**) and what the same compression does to the model before unlearning (**U3 = C1 − C0v2**).

| Contrast | Forget retention | Forget log-prob (nats/token) | Retain retention | Retain log-prob | MMLU-en (pp) | MMLU-hi (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| U1: unlearning (C2 − C0v2) | −0.850 [−0.881, −0.818] | −1.737 [−1.900, −1.578] | −0.192 [−0.226, −0.159] | −0.115 [−0.153, −0.083] | −1.6 [−3.8, +0.6] | −1.0 [−3.5, +1.5] |
| **U2: compressing the unlearned model (C3 − C2)** | **−0.058 [−0.081, −0.034]** | **+0.215 [+0.129, +0.305]** | −0.469 [−0.523, −0.416] | −0.305 [−0.356, −0.255] | −4.2 [−8.0, −0.4] | +1.0 [−5.0, +7.0] |
| U3: compressing before unlearning (C1 − C0v2) | −0.220 [−0.252, −0.186] | −0.065 [−0.078, −0.053] | −0.219 [−0.265, −0.174] | −0.074 [−0.087, −0.061] | −3.8 [−7.6, −0.2] | +0.5 [−4.5, +5.5] |

How to read the table:
- Each cell is the later model minus the earlier one, on the same questions, with its 95% interval.
- Intervals: TOFU forget and retain use a paired bootstrap over the 20 authors in each set; MMLU uses a paired bootstrap over questions (500 English, 200 Hindi). Both use 10,000 resamples, seed 0.
- Retention is ROUGE-L recall rescaled so that the original model is 0 and C0v2 (fine-tuned on TOFU) is 1. A change of −0.469 means losing about 47% of C0v2's gain in word overlap over the original.
- Size: the harness's parameter-memory count is 6,388 MiB in bf16 and 2,449 MiB in qint4. Separately, the C3 checkpoint file on disk is 2,568,072,456 bytes, 2,449.1 MiB (R1).
- Sources: `Vedika/out/r4/contrasts.json` (md5 0e87eb20) and `contrasts.md`; Vedika.md 2026-11-12 19:57. The AI auditor reproduced all 18 contrasts and intervals independently on CPU from the saved rows (audit.md 19:59).

**What each row says**, under PLAN section 5's rule: an interval above 0 is an increase, below 0 a decrease.
- **U1, unlearning lowered the target scores, at a cost.** "Unlearning" here names a procedure, NPO training that targets the forget10 answers, and its measured effect: lower word overlap with those gold answers and lower likelihood for them. That is a score reduction, not evidence that the facts were erased; no test on new phrasings was run, and no reference model was trained without the forget authors.
  - Forget retention fell by 0.85 and forget log-prob by 1.74 nats/token.
  - Retain retention fell by 0.19.
  - No MMLU change was detected; both intervals cover 0.
  - C2 was *selected* for low forget and high retain retention on these same rows, so the retention cells confirm the selection rule rather than test it.
- **U2, the forget measures disagree after compression.**
  - Word overlap with the forget answers fell slightly (−0.058).
  - The likelihood the model gives those answers rose (+0.215 nats/token).
  - Retain retention fell by 0.47, retain log-prob by 0.31, and English MMLU by 4.2 pp.
  - Hindi MMLU: no change detected, with a wide interval.
  - With retain overlap, retain log-prob and English MMLU also falling, the forget-retention decrease is consistent with general degradation. It is not evidence of further unlearning, and it does not rule that out; the cause is not established.
  - The log-prob increase is consistent with partial reversal *on that measure only*. Log-prob is the quantity NPO pushes down directly, and it can move without the greedy text changing.
- **U3, compression alone also costs.** Compressing the fine-tuned model lowers forget and retain retention by about 0.22 each, and English MMLU by 3.8 pp.

**What we do not claim.**
- **U2 is not tested against U3.** In point estimates, the retain loss is larger after unlearning (−0.469 vs −0.219), and forget log-prob moves the other way (+0.215 vs −0.065). But the plan left the U2 − U3 interaction out before any table existed (PLAN section 3), so no test of these differences was run. They are observations, not findings.
- **Selection.** C2 was picked for a low forget score on these rows, so part of any U2 forget movement could be regression from that selection rather than an effect of compression.
- **Word overlap is not knowledge.** ROUGE-L and log-prob do not show what facts the model holds, or why.
- **Uncertainty.** With 20 clusters, intervals near 0 are uncertain. No correction was made for multiple comparisons; the headline cell is U2 forget retention.
- **Scope.** NF4 versions were not rescored, so there is no second quantizer here. Truth-ratio point values are in `contrasts.md`; they have no interval and are descriptive only.

### 1.2 Study S: does protecting an attribution-selected refusal mask keep safety behaviour?

The protected set is the 11 attention and 11 MLP units (77 linears) that CircuitKIT ranked highest from EAP-IG node attribution scores on 32 harmful/benign pairs, with first-token refusal and compliance proxies. We call it a candidate refusal mask: no held-out causal or faithfulness test shows that these components carry refusal, so it is not a validated circuit.

**Short answer.** None of the three 4-bit models showed a detected change in refusal rate on harmful prompts, or in over-refusal on safe prompts, compared with the original bf16 model. English MMLU showed no detected change either. On the headline HarmBench set, keeping the 77 linears picked by the attribution mask in bf16 did not produce a detected refusal gain over keeping a random 77 of the same size. The headline cell, S1 HarmBench refusal Δ (protected − random30), is −0.024 [−0.060, +0.012]. One supporting cell, Hindi, is above 0: +0.120 [+0.040, +0.220]. It is unadjusted, from 50 prompts, and Arabic points the other way.

"No change detected" does not mean "unchanged": the intervals show which changes the data do not exclude. On HarmBench, for example, protected-q4 − original is +0.006 [−0.018, +0.036], so a fall of up to about 2 points or a rise of up to about 4 is not excluded. The Aya intervals are wider, up to [−0.080, +0.200] for random30 on Arabic.

**Refusal rate, each 4-bit model − original** (WildGuard judge, paired bootstrap by prompt):

| Set | Original rate | plain-q4 Δ | random30-q4 Δ | protected-q4 Δ |
| --- | --- | --- | --- | --- |
| HarmBench held-out (n=168), harmful | 0.792 | +0.030 [-0.006, +0.065] | +0.030 [+0.000, +0.065] | +0.006 [-0.018, +0.036] |
| Aya Arabic (n=50), harmful | 0.560 | -0.060 [-0.180, +0.060] | +0.060 [-0.080, +0.200] | -0.020 [-0.120, +0.080] |
| Aya Hindi (n=50), harmful | 0.460 | -0.020 [-0.120, +0.060] | -0.080 [-0.180, +0.000] | +0.040 [-0.040, +0.120] |
| XSTest safe (n=250), over-refusal | 0.084 | -0.004 [-0.028, +0.020] | -0.012 [-0.036, +0.012] | +0.000 [-0.024, +0.024] |

On the harmful sets higher refusal is better; on XSTest it is worse. Two intervals end exactly at 0 (random30 on HarmBench and on Hindi), and under PLAN section 5 they cover 0.

**The planned contrasts** (Δ = A − B on the same prompts):

| Cell | S1: protected − random30 | S2: protected − plain | S3: random30 − plain |
| --- | --- | --- | --- |
| **HarmBench refusal (headline: S1)** | **-0.024 [-0.060, +0.012]** | -0.024 [-0.065, +0.018] | +0.000 [-0.036, +0.036] |
| Aya Arabic refusal | -0.080 [-0.180, +0.000] | +0.040 [-0.060, +0.140] | +0.120 [+0.000, +0.240] |
| Aya Hindi refusal | +0.120 [+0.040, +0.220] | +0.060 [+0.000, +0.140] | -0.060 [-0.140, +0.000] |
| XSTest over-refusal | +0.012 [-0.008, +0.032] | +0.004 [-0.020, +0.028] | -0.008 [-0.024, +0.008] |
| MMLU-en (pp, endpoint) | +1.8 [-0.6, +4.2] | +1.0 [-1.2, +3.2] | -0.8 [-3.2, +1.6] |
| MMLU-hi (pp, secondary) | +2.0 [-1.5, +5.5] | +4.0 [+1.0, +7.5] | +2.0 [-1.0, +5.0] |

**Utility and size.**
- English MMLU accuracy: original 0.562, plain-q4 0.548, random30-q4 0.540, protected-q4 0.558.
- Changes vs the original are −1.4 [−4.4, +1.6], −2.2 [−5.2, +0.8] and −0.4 [−3.0, +2.2] pp; all cover 0.
- Parameter memory: 6,388 MiB in bf16, 2,449 MiB for plain-q4, 3,653 MiB for random30 and protected (the harness's parameter-memory count).
- File sizes on disk: 3,830,144,052 bytes (protected) and 3,830,144,028 bytes (random30).
- Protected and random30 keep the same 859,308,032 parameters in bf16, in 11 attention and 11 MLP blocks each. 29 of each set's 77 linears are shared (`Mudit/out/study_s_metadata.json`; audit.md 20:21).

**Reading**, under PLAN section 5's rule:
- **Does refusal survive 4-bit compression?** No change was detected for any 4-bit model on any set. That is not evidence that refusal is preserved: the intervals still include changes in both directions, for example plain-q4 on HarmBench [−0.006, +0.065] and on Arabic [−0.180, +0.060].
- **Does protecting the attribution mask help?** For the headline, no gain was detected: S1 covers 0. S2 (protected vs plain) covers 0 on HarmBench too.
- **Language cells.**
  - S1 on Hindi is above 0: +0.120 [+0.040, +0.220].
  - S1 on Arabic is −0.080 [−0.180, +0.000], an interval that covers 0.
  - The two languages point in opposite directions, with 50 prompts each and no multiplicity correction. We report them as observed, unadjusted contrasts, not as a language-specific effect of the mask.
- **MMLU-hi:** S2 is +4.0 [+1.0, +7.5] pp. Hindi MMLU is a secondary set, added before the Study S utility results (PLAN section 3).

**Closeness to the original (descriptive, no intervals).** Two KL measures to the original model:

| Set | First-token KL: plain / random30 / protected | Mean KL: plain / random30 / protected |
| --- | --- | --- |
| HarmBench | 0.0774 / 0.0703 / 0.0329 | 0.0535 / 0.0393 / 0.0270 |
| Aya Arabic | 0.2395 / 0.1459 / 0.1805 | 0.0757 / 0.0522 / 0.0443 |
| Aya Hindi | 0.1080 / 0.0902 / 0.0834 | 0.0536 / 0.0369 / 0.0307 |
| XSTest | 0.1643 / 0.1774 / 0.0674 | 0.0679 / 0.0498 / 0.0381 |

- Mean KL is teacher-forced over the original's answers.
- Protected-q4 has the lowest mean KL on all four sets, and the lowest first-token KL on three of them (random30 is lower on Arabic).
- These are point values from one random draw, with no interval. They are an observation, not a finding, and they do not show that the mask is the cause.

**What we do not claim.**
- That safety is preserved: "no change detected" is not equivalence.
- That protecting this mask works, or does not, in general: there is one random draw (PLAN D4), and the headline covers 0.
- That the Hindi cell shows a language effect.
- That WildGuard labels are ground truth: they are a judge's labels.
- That every answer was complete. Answers that hit the 256-token cap were judged on cut text. By re-tokenizing the saved answers, an estimated 36-41 of 168 HarmBench, 29-33 of 50 Hindi, 9-17 of 50 Arabic and 90-106 of 250 XSTest answers per model reach 256 tokens. These are estimates; actual truncation is not verified.

Sources: `Vedika/out/r5/eval/results.md` (md5 254b5e46), `Vedika/out/r5/tables.md` (md5 9b28b06f); Vedika.md 2026-11-12 20:42, 20:54 and 20:55. The AI auditor checked all 2,072 saved rows and their judge labels, and recomputed the rates, intervals and tables on CPU (audit.md 20:43 and 20:55). It did not rerun generation or the judge. No harmful prompt text or model answer is shown here.

### 1.3 What the weights show

Facts checked so far (R1 and R2, both audited PASS):
- **Precision.** The 4-bit models store exactly the precision their quantization maps claim (Mudit.md 02:04 and 02:08; AI-auditor PASS 02:07 and 02:09):
  - C1, C3 and plain-q4: all 252 decoder linears qint4.
  - random30-q4 and protected-q4: 175 qint4, with the other 77 kept in bf16 and bit-for-bit equal to the base model.
  - C1-NF4 and C3-NF4: all 252 NF4.
- **bf16 merge.** C3 is built from C2 after its LoRA change is merged into bf16 weights. The stored bf16 change A differs from the intended fp32 change D (Ganesh.md 04:28; AI-auditor PASS 04:29; descriptive weight statistics only):
  - relative difference ||A-D||/||D|| = 0.526;
  - norm ratio ||A||/||D|| = 1.024, projection <A,D>/<D,D> = 0.886;
  - 42.5% of the 2,812,280,832 weights in the 252 LoRA-wrapped linears changed.
  This is not a fraction of the unlearning lost. Whether behaviour changed is a Study U question.

## 2. Why it matters

- **Check both sides of unlearning after compression, with more than one measure.** On this model, compression moved the forget set's word overlap and its answer likelihood in opposite directions. It also cut overlap on the retained set by about half of the fine-tuning gain. A single forget-set metric would have missed both.
- **Refusal checks at this size are coarse.** On the headline HarmBench set (168 prompts), refusal rates did not separate an attribution-guided protection from a same-size random one. The only cell that did, Hindi (50 prompts), is unadjusted, and Arabic points the other way. A claim that mask protection keeps safety better needs more prompts and more than one random draw.

## 2a. Related work, and what this report adds

This is a short comparison with the closest primary work, not a systematic review. The broad questions are not new; what we add is one carefully re-verified case on a multilingual small model.

- **Quantization can undo unlearning.** Zhang et al. ([Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/abs/2410.16454), ICLR 2025) report that 4-bit quantization of unlearned models restores much of the "forgotten" knowledge. Abitante et al. ([Quantization-Robust LLM Unlearning via Low-Rank Adaptation](https://arxiv.org/abs/2602.13151), IJCNN 2026) report that putting the unlearning update into LoRA adapters helps it survive 4-bit PTQ, on Llama-2-7B with MUSE.
  - *Here:* a LoRA NPO unlearning run on Tiny Aya, scored on TOFU after qint4. Two forget measures moved in opposite directions (word overlap down, gold-answer likelihood up), and the retained set lost much of its overlap. We do not show semantic recovery or test a remedy. Our setup also differs from Abitante et al.'s, because our LoRA update was merged into bf16 weights before quantizing.
- **Safety of quantized models.** Chen et al. ([Assessing Safety Risks and Quantization-aware Safety Patching for Quantized LLMs](https://proceedings.mlr.press/v267/chen25ci.html), ICML 2025) evaluate safety across quantization methods and propose Q-resafe to restore it. Wee et al. ([Safety-Preserving PTQ via Contrastive Alignment Loss](https://arxiv.org/abs/2511.07842v5), 2026) add an alignment loss to the PTQ objective.
  - *Here:* a simpler allocation, which keeps an attribution-selected set of layers in bf16, compared against a same-size random set and plain qint4, with Hindi and Arabic red-team prompts. We neither optimize a safety objective nor benchmark against these methods. Our result for the allocation is a null on the headline cell, at a 49.1% storage cost over plain qint4.
- **Methods we use as given.**
  - NPO: Zhang et al., [Negative Preference Optimization](https://arxiv.org/abs/2404.05868), 2024.
  - TOFU: Maini et al., [TOFU: A Task of Fictitious Unlearning](https://arxiv.org/abs/2401.06121), 2024.
  - EAP-IG: Hanna et al., [Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806), COLM 2024. That paper stresses that a circuit should be judged by faithfulness, not overlap; we did not run a faithfulness test, which is why we call our protected set a candidate mask.
  - We contribute integration, controls and evaluation tooling, not a new algorithm.

## 3. How we used the Lexsi toolkits

| Toolkit | Used for | Code |
| --- | --- | --- |
| SafeTune | NPO unlearning with a retain term (`safetune.unlearn.npo_unlearn`), giving C2 | `Ganesh/scripts/unlearn_c2.py` |
| CircuitKIT | EAP-IG attribution for the candidate refusal mask; mask-protected, random and plain qint4 builds; checkpoint export and reload (`save_quantized_checkpoint`, `load_quantized_checkpoint`) | `Mudit/scripts/aya_refusal_circuit.py`, `build_quantized.py`, `verify_quant.py` |
| AuditKIT | generation and scoring backend; refusal judge `GuardJudge` with the WildGuard profile, chosen after it matched SafeTune's judge on 20/20 fixed pairs (PLAN section 5) | `Vedika/scripts/evaluate.py`, `tofu_eval.py` |

The calls behind the reported numbers:
- **SafeTune:** `safetune.unlearn.npo_unlearn` with `NPOConfig(variant="npo_grad_diff", beta=0.1, lr=3e-5, num_epochs=10, …)` produced C2 (Appendix A).
- **CircuitKIT:**
  - `Pipeline.discover(algorithm="eap-ig", n_examples=32, sparsity=0.3)` produced the attribution scores behind the candidate refusal mask;
  - `quantize(hf, circuit, high_fraction=0.3, backend="quanto")` built protected-q4;
  - `random_quantize(hf, 36, high_fraction=0.3, seed=0)` built random30-q4, and with `high_fraction=0.0` built plain-q4;
  - `_apply_plan_to_model` with an all-qint4 plan built C1 and C3;
  - `export_checkpoint` and `save_quantized_checkpoint` wrote the checkpoints, and `load_quantized_checkpoint` reloads them for scoring.
- **AuditKIT:**
  - `HFGenModel`, through the `InMemoryHFModel` subclass in `common/addons/`, did generation and teacher-forced log-probs;
  - `GuardJudge` with the WildGuard profile gave the refusal labels;
  - `model_info` gave the parameter-memory sizes (Appendix B).

The three toolkits are installed from PyPI (`requirements.txt`) and keep their own licences: each is under a Lexsi Labs Source Available License v1.2 (noncommercial; see each package's LICENSE.md, and for use by an organization its section 1A). This repository's Apache-2.0 licence covers only our own files, not the toolkits, which are not included here.

## 4. Setup

- **Model.** `CohereLabs/tiny-aya-global` at revision af89d219, bf16.
- **Models C0v2 and C2.** LoRA fine-tune on all of TOFU, then SafeTune NPO unlearning of forget10. Full recipe in Appendix A.
- **Data.** Fixed sets in `common/data/`, pinned to dataset revisions (table in Appendix B). Licences are listed in Vedika.md 01:40.
- **Conditions and settings.** PLAN sections 3 and 5:
  - greedy decoding, stop IDs [6, 3];
  - the chat template, with its rendered length measured (371-414 tokens on TOFU);
  - batch 16, left padding; fp32 logits for log-probs;
  - 95% paired bootstrap, 10,000 resamples, seed 0: clustered by author for TOFU, by question for MMLU, by prompt for the safety sets.
  - Full settings and the judge choice are in Appendix B.
- **Quantizers.**
  - quanto qint4: weights only, groups of 128, no calibration data.
  - bitsandbytes NF4, block size 64: C1-NF4 and C3-NF4 were built and passed identity and precision checks (R1), but were not rescored, so no NF4 result is reported.

## 5. How we know: re-verification

| Check | Result | Source |
| --- | --- | --- |
| 4-bit checkpoints: identity, source hash, precision, excluded weights bitwise | 7/7 PASS | Mudit.md 02:04 and 02:08; audit 02:07 and 02:09 |
| C0v2 and C2: identity, bitwise merge, full fresh rescore with recorded settings | PASS: fresh answers equal saved ones on 400/400 forget and 400/400 retain, log-prob difference 0.0 | Ganesh.md 01:53 and 04:28; audit 04:29 |
| Safety harness stop IDs | PASS (code and CPU test) | Vedika.md 01:49; audit 01:51 |
| TOFU rows, all five models: full fresh rescore with recorded settings, then contrasts | PASS: fresh = saved on every row for C0v2, C2, C1, C3 and the original; all 18 contrasts and intervals reproduced independently | Ganesh.md 04:28; Mudit.md 13:49, 19:55; Vedika.md 19:57; audit 04:29, 13:50, 19:56, 19:59 |
| Study S: judge gate, full rescore of four models, tables | PASS: judge 20/20 agreement; all 2,072 rows judged (0 unjudged), every label equals its parsed raw judge output, tables reproduced | Vedika.md 19:48, 19:58, 20:54; audit 19:59, 20:55 |
| Study S utility (MMLU) | PASS: the original reproduced its reviewed rows exactly. The planned CPU scorer precheck was skipped; the GPU reproduction was the gate used (PLAN R5u) | Vedika.md 20:42, 20:45; audit 20:43 |

Every result was reviewed by an independent auditor. The audit log separates checks it reproduced independently from evidence only the owner ran.

## 6. Limits

- **Not blind.** Earlier sessions built these artifacts and saw their numbers. This is re-verification, not pre-registration.
- **Training overlap.** C0v2 trained on all 4,000 TOFU rows, forget10 and retain included. C2's retain term trained on rows 0-3599, which include all 400 scored retain rows.
- **Selection.** C2's learning rate was chosen on the same TOFU rows Study U reports. Only lr 1e-5 and 3e-5 were trained.
- **Intervals are conditional on the trained models.** The TOFU bootstrap resamples authors and the MMLU bootstrap resamples questions. Both treat each training run as fixed, so the intervals leave out training variability (seed, data order). For contrasts with C2 they also leave out the uncertainty of choosing its learning rate on these same TOFU rows.
- **bf16 merge.** C3 quantizes the merged bf16 C2, not the adapter model.
- **Study S lineage and controls.**
  - The base revision behind the historical Study S builds is not proven: source hashes were verified, but the revision cannot be proven from history (audit 02:07).
  - Study S has one random draw.
- **Metric scope.**
  - ROUGE-L measures word overlap, not factual correctness.
  - The WildGuard verdict is a judge's label.
  - KL values (first-token, and mean teacher-forced over the original's answers) are descriptive, without intervals.
- **Hindi/English judge check (J3) is exploratory, not a validation.** Two team members (rater A and rater B) each submitted labels for the same 140 procedurally blinded items (100 Hindi, 40 English) for refusal and harmful content; whether either had seen the answers or labels before was not independently confirmed. The two submitted sheets agree in every label cell. Rater A's account, given to their own agent, is that they labelled entirely by themselves; it corrects their earlier report of partial AI or translation-tool help. Rater B's account, relayed by a team member to the coordinator, is that they labelled entirely by themselves; it corrects an earlier relayed report of partial AI help. The superseded reports are kept on record. Neither account, nor the raters' independence, was independently verified. Because how the sheets came to agree is unexplained, they are compared with WildGuard's stored labels only as one descriptive annotation pattern: not 280 independent judgments, not a measure of rater agreement, and not a human validation of the judge. Result (`Mudit/release/j3_public/J3_EXPLORATORY_RESULTS_20261114.public.md`): 14 of 100 Hindi refusal labels and 10 of 100 Hindi harmful-answer labels differed from WildGuard's, all with annotation yes and WildGuard no; in English, 0 of 40 refusal labels and 1 of 40 harmful-answer labels differed. These are label counts, not distinct answers. The Hindi WildGuard labels remain unvalidated, and Arabic was not examined.
- **Sample size and multiplicity.** 50 prompts per Aya language; many cells; no multiplicity correction. The headline cells are U2 forget retention and S1 HarmBench refusal.
- **Answer length cap.** Safety answers are capped at 256 tokens. Re-tokenizing the saved answers suggests many reach the cap (for example 29-33 of 50 Hindi answers per model), so some may have been judged on cut text; actual truncation is unverified.
- **No second quantizer.** NF4 versions were not rescored.
- **One setup.** One model, one unlearning seed, greedy decoding.
- **LoRA unlearning.** C2 is a LoRA (r 32) unlearning run, not the full-weight fine-tuning used in the cited NPO setting.

## 7. Reproduction

Environment, as installed and used for every run in this revision:
- Python 3.12.16, torch 2.11.0+cu128, transformers 5.22.1, peft 0.21.2, optimum-quanto 0.2.7, bitsandbytes 0.50.2, rouge_score 0.1.2;
- the Lexsi toolkits: auditkit 1.1.3, circuitkit 0.2.0, safetune 0.1.10;
- one NVIDIA A100-PCIE-40GB per job;
- `CUDA_HOME=/usr/local/cuda-12.9` for quanto's kernels, and `HF_HUB_OFFLINE=1`.

Commands run from the root of this repository, https://github.com/Muditkumar123/tiny-aya-unlearning-4bit.

In the public copies of 16 files (the R1 reports, `Mudit/out/study_s_metadata.json`, the four Mudit launchers, Ganesh's two `run_settings.json` files and his two R2 launchers), the local home directory is replaced by `~`. The md5 values in table A are those of the unredacted lab originals. `Mudit/release/redaction.json` lists each file's sha256 before and after redaction.

**A. What this revision ran, as archived invocations** (each step has a reviewed handoff). These commands write to the fixed output folders listed. Run again as printed, they do not reproduce from scratch: the R4a and Study S launchers refuse an existing output folder. `r2_rescore.py` reuses cached rows: it then skips generation, observes no prefill rows, and writes empty prompt-length records without failing, so its run settings would not be fresh evidence. For a fresh run, see B. Vedika's two Study S launchers (`Vedika/out/r5/run.sh`, `mmlu_run.sh`) stay in the lab with the rest of `Vedika/out/`, which holds model answers; B gives the commands they ran.

| Step | Command (launcher md5) | Output | Handoff / audit |
| --- | --- | --- | --- |
| Checkpoint identity and precision (R1) | `bash Mudit/out/r1/run_r1_quanto.sh` (2984c8fe), `bash Mudit/out/r1/run_r1_nf4.sh` (65a1be80), both calling `Mudit/scripts/verify_quant.py` (1020e4a9) | `Mudit/out/r1/*.json` | Mudit.md 02:04, 02:08; audit 02:07, 02:09 |
| C0v2/C2 rescore (R2) | `bash Ganesh/out/r2_rescore.sh` (bf29a8eb) | `Ganesh/out/r2_rescore/{c0v2,c2}` | Ganesh.md 04:28; audit 04:29 |
| C1/C3 rescore | `Mudit/out/r4_rescore/run.sh Mudit/out/r4_rescore` (4a26c4ec; f7c5bea5 when run, before the exit-code fix) | `Mudit/out/r4_rescore/{c1,c3}` | Mudit.md 13:49; audit 13:50 |
| Original rescore (R4a) | `Mudit/out/r4a_original/run.sh Mudit/out/r4a_original` (c2670ff6) | `Mudit/out/r4a_original/original` | Mudit.md 19:55; audit 19:56 |
| Study U contrasts (R4) | `CUDA_VISIBLE_DEVICES="" python Vedika/scripts/r4_contrasts.py c0v2 c2 c1 c3 original` (a427ca2d), CPU | `Vedika/out/r4/contrasts.{json,md}` | Vedika.md 19:57; audit 19:59 |
| Study S safety rescore (R5) | `bash Vedika/out/r5/run.sh 1` (55885ccc; GPU index 1, as in run.log) | `Vedika/out/r5/eval/` | Vedika.md 20:54; audit 20:55 |
| Study S utility (R5u) | `bash Vedika/out/r5/mmlu_run.sh 0` (d0436ad3; GPU index 0, as in mmlu_run.log) | `Vedika/out/r5/mmlu/` | Vedika.md 20:42; audit 20:43 |
| Study S tables | `python Vedika/scripts/r5_tables.py` (22f8a88a when run and reviewed; now e9dc2adc, which also accepts a token-count estimate in place of answer text for the text-free export, per Vedika.md 21:08; AI-auditor PASS 21:09), CPU | `Vedika/out/r5/tables.{json,md}` | Vedika.md 20:54; audit 20:55 |
| Study S build metadata | `CUDA_VISIBLE_DEVICES="" python Mudit/scripts/study_s_metadata.py` (961531dd), CPU | `Mudit/out/study_s_metadata.json` | Mudit.md 20:19, 20:21; audit 20:20, 20:21 |

**B. Reproducing from scratch.**
- **Setup.** `pip install -r requirements.txt` (exact pins; torch from the cu128 index, as its first lines say). Then build the data: `mkdir -p common/data && python Vedika/scripts/build_prompt_sets.py && python Vedika/scripts/build_tofu_sets.py` (CPU, pinned downloads).
  - The repository ships no dataset text. These builders rebuild all 8 `common/data` files; in a staged copy (2026-11-13) all 8 were byte-identical to ours.
  - The harnesses and Ganesh's scripts read 5 of them.
- **Numbers only, CPU, about a minute.** `python Vedika/release/reproduce.py [export dir]` (md5 a10af8b8) recomputes every Study U and Study S table from the text-free numeric export and exits non-zero on any difference. The AI auditor ran it and it passes (audit.md 21:09).
- **Fresh generation and scoring, GPU.** Give each launcher a new, empty output folder:
  - `python Ganesh/scripts/r2_rescore.py <label> <checkpoint> --out <new dir>` for C0v2 and C2;
  - `Mudit/out/r4_rescore/run.sh <new dir>` for C1 and C3;
  - `Mudit/out/r4a_original/run.sh <new dir>` for the original;
  - `python Vedika/scripts/evaluate.py <label> <checkpoint> --out <new dir>` for the Study S safety sets, and `python Vedika/scripts/mmlu_only.py <label> <checkpoint> --out <new dir>` for its utility rows, for plain-q4, random30-q4 and protected-q4. Give all three `evaluate.py` calls the same `--out`; the first call also scores the original. `mmlu_only.py`'s check of the original reads the lab's R4a rows, which are not in the repository (Vedika/release/RELEASE_FILES.md). Settings are in Appendix B.
- **Contrasts on fresh rows.** `Vedika/scripts/r4_contrasts.py` reads fixed source paths (its `SOURCES`, line 27). To run it on fresh rows, point `SOURCES` at the new folders. Alternatively, compare the fresh rows with the archived ones using `Ganesh/scripts/compare_regen.py`; if they match, the archived contrasts apply.
- This fresh path was not itself run end to end in this revision. What was run is A, plus the CPU check above.
- **Rebuilding C1/C3 (current portable recipe).** `python Mudit/scripts/build_qint4.py <bf16 source dir> <new out dir> --expect-source-sha <sha256>` needs no git and no fixed paths. It records the source and output hashes, library versions and its own sha256. Its CPU test `Mudit/scripts/test_build_qint4.py` builds a tiny model, and the R1 checker accepts the result. This recipe is new; it is not the code that built the released C1/C3, and it has not been run on the real models. A rebuild is expected to match R1's precision checks, not the released file hashes.
- **Discovery IDs.** `Mudit/release/discovery_ids.json`, written by `Mudit/scripts/export_discovery_ids.py`, lists the 32 discovery pairs as pinned HarmBench and Alpaca rows plus sha256 hashes, with no text. Rebuilding the HarmBench held-out set from these IDs alone gives the same 168 IDs and texts as `common/data/harmbench_heldout.json`, checked on CPU from the pinned local caches.

**C. How the artifacts were built** (by earlier sessions; not rebuilt in this revision):

| Artifact | Command | Note |
| --- | --- | --- |
| C0v2 | `python Ganesh/scripts/train_c0.py --config Ganesh/configs/c0_v2.yaml` (the script's default config is the older `c0.yaml`) | Appendix A |
| C2 | `python Ganesh/scripts/unlearn_c2.py --config Ganesh/configs/c2.yaml --lr 3e-5` | Appendix A |
| C1, C3 | `python Mudit/scripts/build_quantized.py <bf16 source> <out>` | script modified 2026-11-11 22:58, after the build. Reference code: its git provenance check needs a git checkout at `~/Lexsi`, and without the predecessor config `common/configs/frozen.yaml`, which is not shipped, it records empty config provenance |
| C1-NF4, C3-NF4 | `python Mudit/scripts/build_nf4.py <bf16 source> <out>` | script modified 2026-11-11 22:58, after the build |
| protected-q4 | `OUT=~/Lexsi/Mudit/out/refusal_circuit_matched python Mudit/scripts/aya_refusal_circuit.py` | script modified 2026-11-11 22:58, after the 2026-11-05 build |
| plain-q4, random30-q4 | `python Mudit/scripts/make_baselines.py` | script modified 2026-11-11 22:58, after the 2026-11-06 build |

The Mudit build scripts on disk are not proven to be the code that ran (Mudit.md 20:19 and 20:21). What this revision verified are the checkpoints themselves.

**Artifact identity** (sha256 of the weights file):

| Artifact | sha256 |
| --- | --- |
| Base `CohereLabs/tiny-aya-global` @af89d219 | shard 1 `0590022d0efe06f12e08e1e94302f069dff404341b426f7435c6303725635fb1`, shard 2 `2ac3809931d113785058809f27775620e0b8aec6fbc7e163c9c09c08b5f90dab` |
| C0v2 (merged) | `4375fe12a7af731525019cffb9932d98a96d0f12567cbe72d757809959ff10a6` |
| C2 (merged) | `a1cfe9ea440e0ab4d7f0eddc377d3625b20ceb3cdfa9c635647c42694cc795ca` |
| C1 (qint4) | `1e3422875f548ccb2947c92efb0fa1ad3216b74c4dec0db2d52247e06a4798d3` |
| C3 (qint4) | `f4a3b4a6edd3c2d347af324acd12ff1f28d97ebfca8db98db909d1fcf5f1a02f` |
| plain-q4 | `dcc660f10b702d484e320862a8f82518ab19f4f521fca2910b14710bee1c31f1` |
| random30-q4 | `dfad03c9576dbe01b74fda2155e860fe7800032a8026634edbf4a2e751fd5e97` |
| protected-q4 | `79383e5436d57572abf64db01a4acf10ad7f534435a6f66de15a3e9a9511fe42` |
| C1-NF4 | `736f7ffe7ffbf8881f29366f0371977bfa7550fb9201e16a037cdab67f6ac2e5` |
| C3-NF4 | `413ef8b2646870ee5e177d5698e60d37a7db61a0092b01abc167cb885748f132` |

Code: https://github.com/Muditkumar123/tiny-aya-unlearning-4bit. Hugging Face (planned publication under `Mudit666`; when this text was written these repositories were not yet published, so a listed name may not resolve yet; real links replace this list after upload): models `Mudit666/tiny-aya-tofu-c0v2`, `Mudit666/tiny-aya-tofu-npo-c2`, `Mudit666/tiny-aya-tofu-c0v2-qint4`, `Mudit666/tiny-aya-tofu-npo-c2-qint4`, `Mudit666/tiny-aya-global-qint4`, `Mudit666/tiny-aya-global-qint4-random77-bf16`, `Mudit666/tiny-aya-global-qint4-refusalmask77-bf16`; data and logs `Mudit666/tiny-aya-compression-evaluation`.

## 8. Contributions

Each person confirmed their own entry (Mudit 2026-11-12, Vedika 2026-11-13, Ganesh 2026-11-14).

- **Mudit:** CircuitKIT work: refusal-circuit discovery and the qint4/NF4 builds (2026-11-05 to 11-07), then checkpoint re-verification and rescoring (2026-11-12). Also the team plan and this write-up. Decisions recorded under his name: plan approval (01:41), gating (D1), no extra random draws (D4).
- **Ganesh:** SafeTune work: the C0v2 fine-tune (with Transformers and PEFT), C2 unlearning (with SafeTune) and its learning-rate selection (built 2026-11-07), then the merge diagnostics and R2 re-verification (2026-11-12). Also the Study U comparison definitions and the C0v2/C2 model cards. Plan approval 01:46.
- **Vedika:** AuditKIT evaluation (the data sets and their provenance, the scoring harnesses, the stop-token fix, the judge check, the Study U contrasts and the Study S rescore, utility and tables), the dataset licence checklist, the evaluation-methods section, and the numbers-only release export with its reproduction script. Plan approval 01:44; approved the original-model rescore (R4a).

How the work was done: each person worked with an AI coding agent in their own folder, which wrote and ran most of the code under that person's direction. A separate AI auditor reviewed every handoff, and an AI coordinator tracked tasks; neither wrote project code. The team's handoff and audit logs record who did what; they are kept privately and are not part of this repository, so citations such as `Vedika.md 19:57` or `audit.md 19:59` point to those logs.

## Appendix A. SafeTune: models and re-verification

Copied verbatim (AI tool names replaced by their roles) from `Ganesh/writeup/SAFETUNE_SECTION.md` (md5 a8506d38, AI-auditor PASS 2026-11-13 00:33; earlier 3d035ab7 passed 20:04), the methods and limits parts only; its Study U results are section 1.1 here. Bracketed review notes are the author's own.

### Models

| Model | What it is | How it was made | Files |
| --- | --- | --- | --- |
| C0v2 | Tiny Aya Global fine-tuned to know the TOFU authors (the "target" model to unlearn from) | LoRA r 32, alpha 64, every linear layer except lm_head. All 4,000 TOFU rows (`locuslab/TOFU` full @324592d8), 10 epochs, lr 2e-4 linear, 50 warm-up steps, effective batch 16 (2,500 steps), bf16, seed 0. Loss on the answer tokens only. Base `CohereLabs/tiny-aya-global` @af89d219. | `Ganesh/configs/c0_v2.yaml`, `Ganesh/scripts/train_c0.py`, `Ganesh/out/c0v2-adapter/`, `Ganesh/out/c0v2-target-merged/` |
| C2 | C0v2 after NPO unlearning training on TOFU forget10 (400 questions, 20 authors) | `safetune.unlearn.npo_unlearn` with `NPOConfig(variant="npo_grad_diff", beta=0.1, npo_coeff=1.0, grad_diff_coeff=1.0, num_epochs=10, lr=3e-5, weight_decay=0.01, optimizer="adamw")`. Constant lr. Each step uses 16 forget + 16 retain90 rows (250 steps). Fresh LoRA, same shape as C0v2. The NPO reference is C0v2 itself (the same weights with the adapter switched off). Seed 0. | `Ganesh/configs/c2.yaml`, `Ganesh/scripts/unlearn_c2.py`, `Ganesh/out/c2-lr3e-05-adapter/`, `Ganesh/out/c2-lr3e-05-merged/` (`c2-unlearned-merged` is a symlink to it) |

Both adapters were merged into a fresh bf16 base on CPU with PEFT `merge_and_unload` (peft 0.21.2).

### How the C2 learning rate was chosen

The rule was frozen in `c2.yaml` before any C2 run. Take the smallest lr in {1e-5, 3e-5, 1e-4} with forget retention <= 0.25 and retain retention >= 0.50. Retention is scaled ROUGE-L recall: the original model = 0, C0v2 = 1. The grid trained 1e-5, then 3e-5, and stopped there; 1e-4 was never trained. The record is `Ganesh/out/c2_selection.json`, written by a predecessor run; its numbers are not re-verified.

**Disclosure:** the selection used the same TOFU rows that Study U reports.

### Re-verification (R2)

- **Identity:** new sha256 values serve as the reference, since no earlier hash existed. C0v2 merged `4375fe12…`, C2 merged `a1cfe9ea…`. Full hashes are in `Ganesh/out/r2_merge_*.json`. **[AI auditor 01:54: independently re-hashed]**
- **Merge:** a fresh CPU bf16 merge equals the saved weights bitwise on all 290 tensors (252 LoRA-wrapped linears), for both models. **[owner evidence; checker code reviewed by the AI auditor 01:54]**
- **Generation and scores:** the old saved runs lack records of their padding side and dtype, so both models were fully rescored with Vedika's unchanged `tofu_eval.py` (PLAN section 4 fallback). That run records the actual settings in `Ganesh/out/r2_rescore/<model>/run_settings.json`: left padding, bf16 weights, fp32 logits, batch 16, and rendered prompts of 371-414 tokens. Its answers and log-probs equal the saved predecessor rows on all 400 forget and 400 retain questions and on all 700 MMLU rows. **[AI-auditor PASS 04:29: rows, cache equality and prompt lengths checked on CPU; GPU run is owner evidence]**

### Weight statistics of the bf16 merge (descriptive only)

C2's LoRA change D is computed in fp32. Storing the merged model in bf16 gives an actual change A = W_C2 − W_C0v2. Over all 2,812,280,832 LoRA-wrapped weights (`Ganesh/scripts/merge_diagnostics.py`):

- relative discrepancy ‖A − D‖ / ‖D‖ = 0.526;
- norm ratio ‖A‖ / ‖D‖ = 1.024;
- projection ⟨A, D⟩ / ⟨D, D⟩ = 0.886, and cosine = 0.865;
- 42.5% of those weights differ from C0v2 after bf16 storage.

So the stored change is about as large as intended and points mostly the same way, but it differs from the intended change by about half its norm. This is **not** a measure of how much unlearning was "lost": these are weight statistics, with no behavioural conclusion. **[owner evidence; the AI auditor did not rerun]**

### Limits

- **Training overlap:** C0v2 trained on all 4,000 TOFU rows, and C2's retain term trained on rows 0-3599. All 400 scored retain rows (0-399) are training rows for both models, so retain scores measure memorised rows. Global-MMLU is the only held-out utility measure.
- **LoRA, not full fine-tuning:** C2 uses LoRA r 32 on every linear layer, while the cited NPO setting fine-tunes all weights.
- **One unlearning run:** C2 has one seed and one method (NPO with a retain term).

## Appendix B. Evaluation methods (AuditKIT)

Copied verbatim (AI tool names replaced by their roles, and the J3 results note cited by its public copy) from `Vedika/writeup/EVALUATION_SECTION.md` (md5 80e0f529, AI-auditor PASS 2026-11-14 01:48; earlier dc739dfe passed 2026-11-12 21:02).

### What AuditKIT does in our pipeline

AuditKIT 1.1.3 is the scoring layer for both studies.

- **Generation and log-probs.** Every model, quantized or not, is wrapped in `common/addons/inmemory_model.py` (`InMemoryHFModel`, a subclass of AuditKIT's `HFGenModel` that takes a loaded model object). AuditKIT renders the chat template, generates greedily, and computes teacher-forced log-probs.
- **Safety judge.** Refusal labels come from AuditKIT's `GuardJudge` with its WildGuard profile (`allenai/wildguard`, bf16), in `Vedika/scripts/evaluate.py` `judge()`.
- **Harnesses.** `Vedika/scripts/tofu_eval.py` (Study U: TOFU and MMLU), `evaluate.py` (Study S: safety sets), `mmlu_only.py` (Study S utility: MMLU, scored the same way as `tofu_eval.py`).

### Frozen generation settings (both studies)

| Setting | Value |
| --- | --- |
| Prompt | Tiny Aya chat template with its default preamble; the original tokenizer for every model |
| Decoding | greedy; max 128 new tokens (TOFU), 256 (safety) |
| Batch | 16, left padding (the tokenizer's setting; recorded per model in the Study U run settings) |
| Stop tokens | 6 `<|END_OF_TURN_TOKEN|>` and 3 `<EOS_TOKEN>`. Tiny Aya's generation config stops only at 3; the fine-tuned models end turns with 6, so without both an answer could run past its turn. `evaluate.py` checks both IDs against the installed tokenizer at every launch. The quantized checkpoints need the IDs passed explicitly: after `load_quantized_checkpoint` (CircuitKIT 0.2.0, optimum-quanto 0.2.7) the model stops only at 3, even when the folder's `generation_config.json` lists [6, 3] (J4 check of the upload copies, 2026-11-13). Pass `eos_token_id=[6, 3]` to `generate`, as the harness and the model cards do. |
| Log-probs (Study U TOFU/MMLU and Study S utility only) | `tofu_eval.py` and `mmlu_only.py` upcast the lm_head output to fp32 (`fp32_logits=True`), so a summed 20-token log-prob is not rounded to bf16 steps. The safety run (`evaluate.py`) does not enable it; its scores are judge labels, and its KL is computed separately. |

A GPU check on 4 prompts with the original model (`Vedika/scripts/test_stop_gpu.py`) confirmed that AuditKIT's answer equals a direct greedy generation with the same stop tokens. Each of those answers ended at a stop token or at the token limit. On those prompts the base model ended its turns with token 3, so the stop-6 rule is a guard for the fine-tuned models rather than a change for the original.

Each run writes a manifest, and resuming or comparing against a run whose manifest differs is refused. All manifests record model identity, generation settings, batch size, stop tokens, tokenizer and template hash, and prompt hashes. The TOFU and utility manifests also record the scoring version; the safety manifests record the judge. The manifests do not record every execution detail: padding side, parameter dtypes and rendered prompt lengths were observed separately, in the Study U run settings (`run_settings.json`).

### The judge, and the check that chose it

WildGuard returns three labels per (prompt, answer) pair: harmful request, refusal, harmful answer. We score **refusal**.

- **Two ways to run it.** SafeTune runs WildGuard through a private function (`_run_hf_judge`). AuditKIT's public `GuardJudge` builds the same prompt, but its template starts with a literal `<s>` and the tokenizer adds another, so the judge would see two start tokens. We strip the literal `<s>`. The token IDs then equal SafeTune's. That rests on CPU tokenization tests in `test_judge_agreement.py` (reviewed) and on an owner-reported check of 100 inputs (Vedika.md 01:49), which was not independently reproduced. The reviewed evidence that the two paths agree is the 20-pair check below.
- **Frozen rule** (common/CHANGELOG.md 2026-11-12 01:49), set before any Study S scoring: use AuditKIT if it gives the same refusal label as SafeTune on 20 of 20 fixed pairs, otherwise keep SafeTune and disclose the private API.
- **The check.** 10 XSTest-safe prompts plus 5 Hindi and 5 Arabic Aya prompts, each with the original model's answer, were fixed and saved before either judge ran. Result: refusal labels agree on 20/20 (16 no, 4 yes). The other two labels and the raw judge text also agree 20/20. So Study S uses AuditKIT's GuardJudge.
- **Scope of the check:** 20 pairs show the two code paths agree; they do not measure how accurate WildGuard is. A refusal label is a judge's label, not a human rating.

### Data

| Set | Rows | Use | Source (pinned revision) |
| --- | --- | --- | --- |
| TOFU forget10 + perturbed answers | 400 (20 authors) | Study U forget | `locuslab/TOFU` @324592d8 |
| TOFU retain (full rows 0-399) | 400 (20 authors) | Study U retain | same |
| Global-MMLU en500 / hi200 | 500 / 200 | utility, both studies (en500 is the endpoint) | `CohereLabs/Global-MMLU` @0e619dbe |
| HarmBench held-out | 168 | Study S harmful requests | `walledai/HarmBench` @fb6c2afd, standard split minus the 32 circuit-discovery prompts |
| Aya red-teaming | 50 Hindi + 50 Arabic | Study S harmful requests, reported per language | `CohereLabs/aya_redteaming` @5a16fee0 |
| XSTest safe | 250 (10 types × 25) | Study S over-refusal | `Paul/XSTest` @f600c994 |

The builders are `Vedika/scripts/build_prompt_sets.py` and `build_tofu_sets.py`; the files are never overwritten. Overlap: every TOFU row scored is a training row of C0v2, and the retain rows are also in C2's retain term. Global-MMLU is the only held-out utility measure.

### Metrics, uncertainty and how intervals are read

All intervals are 95% percentile bootstraps, 10,000 resamples, seed 0.

| Study | Measure | Unit | Interval |
| --- | --- | --- | --- |
| U | Retention: ROUGE-L recall scaled so original = 0 and C0v2 = 1 | retention units | author-clustered (20 authors per set), paired; for a contrast the original, C0v2 and both models share each draw and the ratio is recomputed in it. Draws with a denominator ≤ 0 were counted: 0 of 10,000 in all six cells. |
| U | Gold-answer log-prob, mean per answer token | nats/token | author-clustered, paired |
| U | TOFU truth ratio | — | point value only |
| both | MMLU accuracy | percentage points for Δ | question bootstrap, paired |
| S | Refusal rate (harmful sets: higher is better; XSTest-safe: higher is over-refusal) | rate | paired by prompt, only prompts judged in both models; Aya Hindi and Arabic separate |
| S | Flip rate; first-token and mean KL to the original, teacher-forced over the original's answers | — | descriptive, no interval |
| both | Size | MiB | point value only: parameter memory as the harness accounts it (AuditKIT `model_info`, quanto packed weights counted by their parts). It is not measured checkpoint storage or runtime GPU memory. |

**Reading rule** (PLAN section 5): an interval above 0 is an increase, below 0 a decrease. An interval that covers 0, or has an endpoint exactly at 0, means **no change detected**. That is never read as "unchanged": no equivalence margin was set. With 50 prompts per Aya language, rates move in steps of 0.02 and intervals are wide.

ROUGE-L recall is word overlap with the gold answer, not factual recall. Log-prob measures how likely the model finds the gold answer, which NPO pushes down directly.

### Coverage and truncation

- **Coverage.** Study S: every one of the 2,072 answers (4 models × 518 prompts) got a refusal label (0 unjudged); every XSTest type is 25/25 judged for every model. Study U: all 400 + 400 TOFU rows and 700 MMLU rows per model.
- **Answers at the token limit (estimate).** The harness keeps answer text, not token counts. We estimate capped answers by re-tokenizing the text and counting those at ≥ 256 tokens. This approximates the generated length; it cannot establish it, or whether an answer was cut. By that estimate, 29-33 of 50 Hindi answers per model, 9-17 of 50 Arabic, 36-41 of 168 HarmBench and 90-106 of 250 XSTest reach 256 tokens. An answer that did stop at the limit was judged on its first 256 tokens.

### Limits of the evaluation

- **Judge.** All reported refusal results use one automatic judge, WildGuard. A planned blind human calibration (J3) did not produce a validation. Two team members submitted annotation sheets for 140 answers (25 Hindi and 10 English prompts × 4 models). The sheets have identical labels, and whether they were rated independently is unconfirmed. They were therefore used only in a separate exploratory comparison (`Mudit/release/j3_public/J3_EXPLORATORY_RESULTS_20261114.public.md`): 14 of 100 Hindi refusal labels and 10 of 100 Hindi harmful-answer labels differed from WildGuard's, all with annotation yes and WildGuard no; in English, 0 of 40 and 1 of 40 differed. This does not establish WildGuard's accuracy. The Hindi labels remain unvalidated, and Arabic was not examined at all.
- **Few clusters.** 20 authors per TOFU set; 50 prompts per Aya language.
- **One random draw.** Study S's random-protection control is a single draw (decision D4).
- **Truncation.** Answers at the 256-token limit (estimate above).
- **Selection.** C2's learning rate was chosen on the same TOFU rows Study U reports.
- **Multiplicity.** No correction. The headlines are fixed (Study U: U2 forget retention; Study S: S1 HarmBench refusal) and the other cells are supporting.
- **Not blind.** Predecessor sessions had seen earlier numbers for these artifacts; every result here is a re-verification under rules written before the reported tables.

### Sources (all reviewed)

| What | File | Handoff | Audit |
| --- | --- | --- | --- |
| Stop-token fix | `Vedika/scripts/evaluate.py`, `test_evaluate.py` | Vedika.md 01:49 | PASS 01:51 |
| Judge check | `Vedika/out/judge_agreement/result.json` | Vedika.md 19:48 | PASS 19:50 |
| Judge switch, stop check on GPU, smoke | `evaluate.py` 25a5b446, `test_stop_gpu.py`, `Vedika/out/r5_smoke/` | Vedika.md 19:58 | PASS 19:59 |
| Study U contrasts | `Vedika/out/r4/contrasts.md` | Vedika.md 19:57 | PASS 19:59 |
| Study S utility | `Vedika/out/r5/mmlu/` | Vedika.md 20:42 | PASS 20:43 |
| Study S tables code | `Vedika/scripts/r5_tables.py` 22f8a88a | Vedika.md 20:45 | PASS 20:45 |
| Study S rows and tables | `Vedika/out/r5/eval/results.md`, `Vedika/out/r5/tables.md` | Vedika.md 20:54, 20:55 | PASS 20:55, 20:56 |

