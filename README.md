# Safety-aware 4-bit compression of Tiny Aya

Lexsi Hackathon 2026 · Track 2b · Team: [Mudit](https://github.com/Muditkumar123), [Ganesh](https://github.com/ShGanesh), [Vedika](https://github.com/vedikajain2004)

This project partly recreates, and modestly extends, two Lexsi Labs papers on [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B parameters). It uses all three Lexsi toolkits: **SafeTune** for fine-tuning and unlearning, **CircuitKIT** for circuit discovery and compression, and **AuditKIT** for evaluation.

| Paper | What we recreated | What we added |
| --- | --- | --- |
| [Forgetting That Sticks](https://arxiv.org/abs/2605.15138) (Sadhu, Seth, Sankarapu, 2026) | 4-bit quantization partly undoes gradient-based unlearning, because training updates are far smaller than one quantization step | The effect on Tiny Aya with two 4-bit formats (qint4 and NF4); our own MANSU implementation, with a random-sign control |
| [C-ΔΘ](https://arxiv.org/abs/2602.04521) (Kasliwal, Seth, Sankarapu, 2026) | Locating refusal-related components with EAP-IG | Keeping those components in bf16 while the rest of the model goes to 4-bit, tested against random and plain baselines in English, Hindi and Arabic |

**Problem statements:** #11 Safety-aware compression (main). The work also touches #7 Unlearning and #9 Locating where safety lives.

> **In one minute.** Base model: Tiny Aya Global in bf16. Method: EAP-IG picks 77 refusal-circuit layers to keep in bf16, and the rest go to 4-bit.
> - **6.70 GB -> 3.83 GB**, with HarmBench refusal 79.2% -> 79.8%, English MMLU 56.2% -> 55.8% and over-refusal 8.4% -> 8.4%.
> - Refusal survives 4-bit with or without protected layers. Protection buys fidelity: mean KL to the original is 15–31% lower than with a size-matched random choice of layers.
> - Unlearning is less robust. 4-bit gives back 12–15% of the answer likelihood that NPO removed.

[Abstract](docs/ABSTRACT.md) · [How we did it](docs/METHODS.md) · [MANSU details](docs/MANSU.md) · [Release history](docs/HISTORY.md)

## Why this matters

Small models are usually deployed in 4-bit, but refusal training and unlearning are done and evaluated at full precision. If compression changes those behaviours, the safety evaluation no longer describes the deployed model. We measure this on Tiny Aya for two behaviours, refusal and unlearning, and test whether a circuit can tell us which layers to keep precise.

## 1. Compression partly restores the likelihood of unlearned answers

We fine-tuned Tiny Aya on all of [TOFU](https://arxiv.org/abs/2401.06121) (model **C0v2**), then used SafeTune's NPO to make it forget 20 fictitious authors (model **C2**). We then compressed both models to 4-bit.

NPO worked as intended. On the forget set, word-overlap retention fell from 1.00 to 0.15. Retain-set retention stayed at 0.81, and English MMLU moved by only −1.6 pp [−3.8, +0.6].

Compression then raised the likelihood of the forgotten answers in the unlearned model:

| Change in forget-answer log-probability after 4-bit (nats/token) | Fine-tuned model (C0v2) | Unlearned model (C2) |
| --- | ---: | ---: |
| quanto qint4 | −0.065 [−0.078, −0.053] | **+0.215 [+0.129, +0.305]** |
| bitsandbytes NF4 | −0.030 [−0.033, −0.027] | **+0.262 [+0.196, +0.329]** |

Compression makes an ordinary model slightly *less* sure of its training answers. It makes the unlearned model *more* sure of the answers it was trained to forget. Both 4-bit formats show this, which is the direction the paper reports.

- **Size of the effect.** The rise gives back 12% (qint4) and 15% (NF4) of the likelihood NPO removed. Most of the unlearning survives 4-bit.
- **Other forget measures.** Word overlap, our headline measure for this study, does not show the reversal: forget retention falls a further 0.058 [0.034, 0.081] under qint4. The paper's multiple-choice accuracy barely moves under NF4: −1.0 pp [−3.7, +1.5]. NPO had changed how the model *writes* answers far more than which option it *picks* (accuracy 69.8% -> 66.3%).
- **Retained knowledge.** Compression costs the unlearned model more than the fine-tuned model. Retain-set retention falls 0.47 under qint4 (fine-tuned model: 0.22), and retain ROUGE-L falls 0.24 under NF4 (fine-tuned model: 0.09).
- **Update size.** In the paper's training setting, the largest change 30 steps can make is about 0.13% of a typical weight. One NF4 step in the same layers is 26–28%, which is 180–220 times larger; the paper reports a 47–828× gap for its baselines. NPO's update is small too: in bf16, 57% of the weights it trained are [identical](results/provenance/training/c2-lr3e-05-adapter/merge_stats.json) in C2 and C0v2. We did not test which weight changes 4-bit rounding removes.

Full numbers: [qint4 scores per model](results/tables/study_u_levels.md) · [qint4 changes](results/tables/study_u_contrasts.md) · [NF4 results](results/nf4_followup/README.md)

## 2. Protecting the refusal circuit during compression

Following C-ΔΘ, we ran CircuitKIT's EAP-IG on 32 harmful/benign prompt pairs and took the components that scored highest for refusal. That gave 77 of Tiny Aya's 252 decoder linear layers. We kept those layers in bf16 and quantized the rest to qint4. As controls, we also built a model with 77 *random* layers kept in bf16 and a plain qint4 model.

| Model | Weights | HarmBench refusal | Mean KL to original¹ | Refusal flips¹ | English MMLU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Tiny Aya (bf16) | 6.70 GB | 79.2% | – | – | 56.2% |
| Plain qint4 | 2.57 GB | 82.1% | 0.054 | 5.4% | 54.8% |
| qint4, 77 random layers in bf16 | 3.83 GB | 82.1% | 0.039 | 4.2% | 54.0% |
| **qint4, 77 circuit layers in bf16** | 3.83 GB | 79.8% | **0.027** | 3.0% | 55.8% |

¹ On HarmBench prompts. Flips count prompts whose refusal label differs from the original model's.

- **Refusal survives 4-bit compression.** Every compressed model keeps HarmBench refusal within about 3 pp of the original, with or without protected layers. Over-refusal on harmless XSTest prompts stays at 7–8%.
- **Main comparison: protected vs random layers on HarmBench.** Refusal is similar: −2.4 pp [−6.0, +1.2]. On the 50-prompt red-team sets, Hindi refusal is 12 pp higher [+4, +22] and Arabic is 8 pp lower [−18, 0].
- **The circuit-protected model stays closest to the original.** Its mean KL is 15–31% lower than the random model's and 41–50% lower than plain qint4's. This holds on all four prompt sets (English HarmBench, Hindi, Arabic and XSTest), and every paired 95% interval excludes zero. It also has the fewest refusal flips on every set, by small margins (5 against 7 of 168 on HarmBench).
- **Utility.** The protected model keeps MMLU best: −0.4 pp in English and +2.0 pp in Hindi, relative to the original. Against plain qint4, Hindi MMLU is 4.0 pp higher [+1.0, +7.5].
- **What the protection buys.** The KL gain is about as large on harmless XSTest prompts as on HarmBench. So these layers matter for the model's behaviour in general, not for refusal alone.
- **Cost.** The protected model is 1.26 GB larger than plain qint4. That is still 43% smaller than bf16.

These results fit C-ΔΘ's premise that EAP-IG finds a compact set of components that carry behaviour: keeping them precise keeps the compressed model measurably closer to the original than keeping random layers of the same size.

Full numbers: [protected vs random vs plain](results/tables/study_s_tables.md) · [each model vs the original, and the KL gap](results/tables/study_s_vs_original.md)

## 3. MANSU on Tiny Aya

We wrote MANSU from the paper on top of CircuitKIT and ran it on the fine-tuned model. It has four parts: an EAP-IG forget circuit (MLP layers 13–17), circuit-restricted training under a Fisher retain mask, an NF4-sized magnitude floor, and CAD diagnostics. These results use the paper's training format and its Table 10 and Appendix D settings:

| Stage | Forget MCQ accuracy (40) | MMLU accuracy (100) |
| --- | ---: | ---: |
| Starting model | 77.5% | 47% |
| After 30 training steps | 77.5% | 47% |
| + NF4 magnitude floor | 27.5% | 25% |
| + same floor with random signs (control) | 80.0% | 47% |

- The floor is the step that makes updates big enough to survive quantization. **Its size alone is safe:** moving the same 26 million weights by the same amounts, with random signs, left accuracy unchanged.
- Our TOFU model had memorised its answers (median confidence 99.96%), so training found almost no forgetting signal. The floor pushed that weak direction at least 400 times further than training did, and the model stopped answering with a letter. Utility fell 22 pp, beyond our 8 pp limit, so we did not export that model.
- The next step is clear: get a stronger forget signal *before* the floor. [MANSU details and earlier runs](docs/MANSU.md)

## Metrics

| Metric | Better | Why |
| --- | --- | --- |
| Refusal rate on harmful prompts (HarmBench, Aya red-team) | higher | The safety behaviour compression must keep |
| Refusal rate on harmless prompts (XSTest) | lower | Over-refusal: safety must not come from refusing everything |
| Mean KL to the original, refusal flips | lower | How far each answer moves from the original model's |
| Forget retention, forget answer log-probability | lower | How much unlearned content the model still writes or expects |
| Retain retention, Global-MMLU accuracy | higher | What must not get worse |
| Checkpoint size | lower | The gain from compression |

## Reproduce the tables

On CPU, with Python 3.12 and no model downloads:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
CUDA_VISIBLE_DEVICES="" python reproduce.py
```

This recomputes all seven qint4 and refusal table groups and the KL gap from the saved per-question scores, and checks them against the published tables. For GPU training, compression and evaluation, see [Reproducing the project](docs/REPRODUCING.md).

## Models, data and code

- [Models on Hugging Face](https://huggingface.co/collections/Mudit666/tiny-aya-4-bit-compression-models-6af82408a81fe397ad3233fa): the fine-tuned, unlearned and five compressed models
- [Evaluation data on Hugging Face](https://huggingface.co/collections/Mudit666/tiny-aya-4-bit-compression-evaluation-data-6af82407c9b6f21539a347db): per-question scores, no prompt or answer text
- [Model and dataset repositories](artifacts.json)

```text
training/      TOFU fine-tuning, SafeTune NPO, MANSU
compression/   CircuitKIT attribution, qint4 and NF4 builds
evaluation/    AuditKIT scoring, dataset builders, tables, NF4 follow-up
shared/        Statistics and model helpers
configs/       Training settings
results/       Per-question scores and tables
```

## Scope

- One base model, one training seed and one random-layer draw.
- TOFU questions overlap training by design: C0v2 saw every question, and C2 was selected on the scored questions.
- Refusal labels come from the WildGuard judge, not human raters. The Hindi and Arabic sets have 50 prompts each.
- We use the refusal circuit as a selection rule and did not test it causally.
- Sizes are checkpoint storage. We did not measure inference speed or GPU memory.

More in [How we did it](docs/METHODS.md#scope).

## With more credits

| Planned | Expected |
| --- | --- |
| 5 training seeds and 10 random-layer draws | The KL advantage holds, since it is consistent prompt by prompt. Refusal-rate differences between protected and random layers stay within a few pp. |
| MANSU with a stronger forget signal (more steps, or a less saturated fine-tune), then the NF4 build and CAD check | Target: the forget log-probability change after NF4 goes from +0.26 to about 0 nats/token, with MMLU inside our 8 pp limit |
| Paraphrased and adversarial recall prompts for the unlearned models | Shows whether the 12–15% likelihood return makes forgotten facts recoverable. We have no forecast for this. |
| 200+ Hindi and Arabic prompts with human-checked labels | Intervals about half as wide, enough to settle the Hindi and Arabic differences |
| Protecting fewer and more layers, with serving speed and memory | A size–fidelity curve for choosing how much to protect |

## AI use, dates and licences

We used AI coding agents to review and tidy the code, and to help structure the documentation.         
All work was done during the hackathon, from 2026-11-01 onward.

All participants have an equal contribution. 

Our code is [Apache-2.0](LICENSE). Model derivatives follow Tiny Aya's CC-BY-NC-4.0 licence and Cohere's acceptable-use policy. Our numeric results are CC-BY-4.0 where we can grant that. Datasets and toolkits keep their own terms: see [data sources](data/SOURCES.md).
