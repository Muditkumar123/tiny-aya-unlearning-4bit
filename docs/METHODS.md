# How we did it

## Where this sits relative to the papers

| Paper | Its main idea | What we recreated on Tiny Aya | What we extended |
| --- | --- | --- | --- |
| [Forgetting That Sticks](https://arxiv.org/abs/2605.15138) | 4-bit quantization can undo unlearning, because per-weight updates are far smaller than a quantization step. MANSU fixes this with a forget circuit, restricted updates, a retain bound and a magnitude floor. | The reversal under 4-bit (Study U and the NF4 follow-up), the update-size gap, and the four MANSU components | Two quantizers on one unlearned model, retained-utility checks next to forgetting, and a random-sign control for the floor |
| [C-ΔΘ](https://arxiv.org/abs/2602.04521) | Refusal is carried by a sparse circuit found with EAP-IG. Editing only that circuit gives selective refusal. | Localising refusal components with EAP-IG | Using that localisation to decide which layers stay in bf16 during 4-bit compression (Study S) |

We also build on [NPO](https://arxiv.org/abs/2404.05868) (Zhang et al.) for unlearning, [TOFU](https://arxiv.org/abs/2401.06121) (Maini et al.) for the forget/retain benchmark, [Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/abs/2410.16454) (Zhang et al.) for the compression question, and [EAP-IG](https://arxiv.org/abs/2403.17806) (Hanna et al.) for attribution. Our contribution is running these ideas together on Tiny Aya with Lexsi's toolkits, with matched controls and multilingual evaluation.

## Models

```text
Tiny Aya Global (bf16)
  ├─ TOFU fine-tune -> C0v2 ──── qint4 -> C1    (NF4 copy in the follow-up)
  │                    └─ NPO -> C2 ── qint4 -> C3    (NF4 copy in the follow-up)
  └─ Refusal study: plain qint4 · 77 random layers in bf16 · 77 circuit layers in bf16
```

**Fine-tuning (C0v2).** LoRA on all 4,000 TOFU examples: rank 32, alpha 64, every linear layer except the output head, learning rate 2e-4, 10 epochs, effective batch 16, seed 0. Settings: [configs/c0_v2.yaml](../configs/c0_v2.yaml).

**Unlearning (C2).** SafeTune `npo_unlearn` with a gradient-difference retain term on TOFU forget10 (400 questions, 20 authors): beta 0.1, LoRA rank 32, 10 epochs. We trained learning rates 1e-5 and 3e-5 and kept 3e-5. Settings: [configs/c2.yaml](../configs/c2.yaml).

**Compression.**
- *qint4*: quanto weight-only 4-bit, groups of 128, with embeddings and the tied output head kept in bf16.
- *NF4*: bitsandbytes 0.50.2, blocks of 64, bf16 compute, all 252 decoder linear layers. Each model was saved and reloaded, and the tensors matched exactly.

**Refusal circuit.** CircuitKIT's EAP-IG ran on 32 harmful/benign pairs, scored on the first answer token (refuse vs comply). We kept the 11 top attention blocks and the 11 top MLP blocks in bf16, which is 77 linear layers. The random control keeps the same number and kind of blocks, drawn with seed 0.

**Toolkits in the pipeline.** SafeTune makes the unlearned model ([unlearn_c2.py](../training/unlearn_c2.py)). CircuitKIT finds circuits and compresses ([aya_refusal_circuit.py](../compression/aya_refusal_circuit.py), [build_qint4.py](../compression/build_qint4.py), [discover_mansu.py](../compression/discover_mansu.py)). AuditKIT generates, scores and runs the WildGuard judge ([tofu_eval.py](../evaluation/tofu_eval.py), [evaluate.py](../evaluation/evaluate.py)).

## Evaluation

**Unlearning under qint4 (Study U).** 400 forget and 400 retain questions (20 authors each), 500 English and 200 Hindi Global-MMLU questions. We use three measures:
- *Retention*: ROUGE-L recall of the generated answer, rescaled so the base model is 0 and C0v2 is 1. It measures word overlap with the gold answer. Forget retention is the study's headline measure.
- *Answer log-probability*: the mean log-probability of the gold answer, in nats per token.
- *MMLU accuracy*: the highest-scoring answer letter A–D.

**Unlearning under NF4 (follow-up).** This uses the held-out split prepared for MANSU: 360 forget and 100 retain questions, 400 English and 165 Hindi MMLU questions. It also adds the paper's own metric, four-option multiple-choice accuracy on all 400 forget questions, scored by the log-probability of each answer letter.

**Refusal (Study S).** 168 held-out HarmBench prompts (none used for circuit discovery), 50 Hindi and 50 Arabic Aya red-team prompts, and 250 harmless XSTest prompts to check over-refusal. WildGuard, run through AuditKIT, labels each answer. *Mean KL* is the average per-token KL divergence from the original model, measured on the original's own answers. *Flips* are prompts whose refusal label differs from the original model's. The main comparison is protected against random layers on HarmBench refusal.

**Generation.** Greedy decoding with stop IDs `[6, 3]` and left padding. Answers are capped at 128 tokens for TOFU and 256 for the safety sets. Log-probabilities use fp32 logits.

**Intervals.** 95% paired bootstrap, 10,000 resamples, seed 0. We resample authors for TOFU, questions for MMLU and prompts for refusal.

## Scope

- **Size of the study.** One base model, one training seed, one random-layer draw.
- **TOFU overlap.** C0v2 was trained on every TOFU question. C2's retain term saw the scored retain questions, and its learning rate was chosen on the scored questions.
- **What the forget measures capture.** Word overlap and answer likelihood track how the model answers. They do not test recall of the facts under new phrasings.
- **Refusal labels.** These come from WildGuard, not human raters. The Hindi and Arabic sets are small (50 prompts each).
- **The refusal circuit.** We use it as a selection rule; we did not validate it causally. We use C-ΔΘ's localisation step, not its weight edit.
- **Statistics.** The intervals hold the trained models fixed and are not corrected for multiple comparisons.
- **Cost.** Reported as checkpoint size. We did not measure inference speed or memory.

Next steps that would strengthen the work: more seeds and random draws, paraphrased recall tests, human checks of the Hindi and Arabic labels, and serving benchmarks.
