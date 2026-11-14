# SafeTune part: C0v2 and C2 (draft for WRITEUP.md sections 3, 4, 5 and 6)

Draft by Ganesh's Claude, 2026-11-12. Settings come from the frozen configs and code. R2 passed audit (Codex, audit.md 04:29). Every number marked **[pending audit]** may only be quoted after Codex reviews it.

## Models

| Model | What it is | How it was made | Files |
| --- | --- | --- | --- |
| C0v2 | Tiny Aya Global fine-tuned to know the TOFU authors (the "target" model to unlearn from) | LoRA r 32, alpha 64, every linear layer except lm_head. All 4,000 TOFU rows (`locuslab/TOFU` full @324592d8), 10 epochs, lr 2e-4 linear, 50 warm-up steps, effective batch 16 (2,500 steps), bf16, seed 0. Loss on the answer tokens only. Base `CohereLabs/tiny-aya-global` @af89d219. | `Ganesh/configs/c0_v2.yaml`, `Ganesh/scripts/train_c0.py`, `Ganesh/out/c0v2-adapter/`, `Ganesh/out/c0v2-target-merged/` |
| C2 | C0v2 after NPO unlearning training on TOFU forget10 (400 questions, 20 authors) | `safetune.unlearn.npo_unlearn` with `NPOConfig(variant="npo_grad_diff", beta=0.1, npo_coeff=1.0, grad_diff_coeff=1.0, num_epochs=10, lr=3e-5, weight_decay=0.01, optimizer="adamw")`. Constant lr. Each step uses 16 forget + 16 retain90 rows (250 steps). Fresh LoRA, same shape as C0v2. The NPO reference is C0v2 itself (the same weights with the adapter switched off). Seed 0. | `Ganesh/configs/c2.yaml`, `Ganesh/scripts/unlearn_c2.py`, `Ganesh/out/c2-lr3e-05-adapter/`, `Ganesh/out/c2-lr3e-05-merged/` (`c2-unlearned-merged` is a symlink to it) |

Both adapters were merged into a fresh bf16 base on CPU with PEFT `merge_and_unload` (peft 0.21.2).

## How the C2 learning rate was chosen

The rule was frozen in `c2.yaml` before any C2 run. Take the smallest lr in {1e-5, 3e-5, 1e-4} with forget retention <= 0.25 and retain retention >= 0.50. Retention is scaled ROUGE-L recall: the original model = 0, C0v2 = 1. The grid trained 1e-5, then 3e-5, and stopped there; 1e-4 was never trained. The record is `Ganesh/out/c2_selection.json`, written by a predecessor run; its numbers are not re-verified.

**Disclosure:** the selection used the same TOFU rows that Study U reports.

## Re-verification (R2)

- **Identity:** new sha256 values serve as the reference, since no earlier hash existed. C0v2 merged `4375fe12…`, C2 merged `a1cfe9ea…`. Full hashes are in `Ganesh/out/r2_merge_*.json`. **[Codex 01:54: independently re-hashed]**
- **Merge:** a fresh CPU bf16 merge equals the saved weights bitwise on all 290 tensors (252 LoRA-wrapped linears), for both models. **[owner evidence; checker code reviewed by Codex 01:54]**
- **Generation and scores:** the old saved runs lack records of their padding side and dtype, so both models were fully rescored with Vedika's unchanged `tofu_eval.py` (PLAN section 4 fallback). That run records the actual settings in `Ganesh/out/r2_rescore/<model>/run_settings.json`: left padding, bf16 weights, fp32 logits, batch 16, and rendered prompts of 371-414 tokens. Its answers and log-probs equal the saved predecessor rows on all 400 forget and 400 retain questions and on all 700 MMLU rows. **[Codex PASS 04:29: rows, cache equality and prompt lengths checked on CPU; GPU run is owner evidence]**

## Weight statistics of the bf16 merge (descriptive only)

C2's LoRA change D is computed in fp32. Storing the merged model in bf16 gives an actual change A = W_C2 − W_C0v2. Over all 2,812,280,832 LoRA-wrapped weights (`Ganesh/scripts/merge_diagnostics.py`):

- relative discrepancy ‖A − D‖ / ‖D‖ = 0.526;
- norm ratio ‖A‖ / ‖D‖ = 1.024;
- projection ⟨A, D⟩ / ⟨D, D⟩ = 0.886, and cosine = 0.865;
- 42.5% of those weights differ from C0v2 after bf16 storage.

So the stored change is about as large as intended and points mostly the same way, but it differs from the intended change by about half its norm. This is **not** a measure of how much unlearning was "lost": these are weight statistics, with no behavioural conclusion. **[owner evidence; Codex did not rerun]**

## Study U comparisons: definitions and directions

**What "unlearning" means in this section.** It names a procedure, NPO training that targets the forget10 answers, and its measured effect: lower scores on those 400 questions (word overlap with the gold answer and the likelihood the model gives it). That is a score reduction, not evidence that the facts were erased. We ran no test on new phrasings of the facts and trained no reference model that never saw the forget authors.

Written 2026-11-12 12:46-12:47 IST, before any R4 metric for these models exists in this session. Predecessor numbers for some models were seen earlier (PLAN section 2), so this is not a pre-registration. Mudit records these definitions in PLAN before tables. Vedika's harness and addons compute them (A1). Nothing here is a result.

### Models and contrasts

Models: original (Tiny Aya Global, bf16), C0v2, C1 = C0v2 at qint4, C2, C3 = C2 at qint4. NF4 versions C1-nf4 and C3-nf4 are secondary. Every contrast is Δ = later − earlier, computed on the same questions.

| Contrast | Question it answers | Role |
| --- | --- | --- |
| **U1: C2 − C0v2** | What did NPO unlearning change, on forget and on everything else? | Sets the effect that U2 tests |
| **U2: C3 − C2** | Does qint4 compression of C2, the model after unlearning training, change its forget-set scores? | The Study U question; headline cell = U2 forget retention, as in WRITEUP 1.1 |
| **U3: C1 − C0v2** | What does the same compression do to the model before unlearning? | Control for reading U2 |
| U2-nf4, U3-nf4 | The same with NF4 | Secondary quantizer, reported separately, never pooled |

### Measures (tofu_eval.py scoring v1)

| Measure | Sets | Unit of Δ | 95% CI |
| --- | --- | --- | --- |
| Retention: Δ = (mean later − mean earlier) ÷ (mean C0v2 − mean original), all ROUGE-L recall | forget, retain | retention units (C0v2 − original = 1) | percentile bootstrap over authors (20 per set), 10,000 resamples, seed 0, joint (see below) |
| Gold-answer log-prob, mean per answer token | forget, retain | nats/token | same author draws for both models (paired) |
| Truth ratio, TOFU summary | forget, retain | — | none: the harness reports a point value only; descriptive |
| MMLU accuracy | en (500), hi (200) | percentage points (accuracy Δ × 100) | paired bootstrap by question |
| Size | — | MiB | none; deterministic |

**Joint bootstrap for retention Δ.** The per-question ROUGE-L values of the original, C0v2 and both contrasted models go into one `cluster_ci` call. Every model is resampled with the same author draw, and the whole ratio, numerator and denominator, is recomputed inside each draw. This is how `tofu_eval.table()` already computes single-model retention. A fixed full-data denominator is not used. A1 reports how many draws have a denominator ≤ 0 or a non-finite statistic: `cluster_ci` counts NaN draws but does not reject infinite ones. The A1 code itself needs its own review.

Direction of each measure: higher ROUGE-L, retention and log-prob all mean the model reproduces or prefers the gold answer more. The truth-ratio summaries run differently. R = exp(log-prob of the paraphrased true answer − mean log-prob of the perturbed answers). On forget, TOFU's mean(min(R, 1/R)) rises as R approaches 1, that is, as the model stops telling the true answer from perturbed ones. TOFU counts that as forgetting, so **higher means more forgotten**; a model that strongly prefers perturbed answers (R far below 1) also scores low. On retain, mean(max(0, 1 − 1/R)) rises as the model prefers the true answer, so higher means better kept.

### How each contrast is read

Section 5's rule applies throughout: a CI above 0 is an increase, below 0 a decrease, and covering 0 means no change detected. No equivalence margin was set, so "no change detected" is never reported as "unchanged". It is reported with the estimated 95% interval. That interval is uncertain itself, with only 20 clusters, and subject to the selection caveat below. It is not a hard bound on the possible effect.

Each reading names the measure it rests on. ROUGE-L and retention are word overlap and log-prob is answer likelihood; none of them establishes factual knowledge or its mechanism.

- **U1, forget:** Δ below 0 means lower word overlap with, or lower likelihood for, the forget answers after unlearning. Because C2 was **selected** for forget retention ≤ 0.25 and retain retention ≥ 0.50 on these rows, U1's retention cells confirm the selection rule. They are not independent evidence. Log-prob and MMLU were not part of the rule.
- **U1, retain and MMLU:** Δ below 0 is a decrease after unlearning, consistent with a cost. A retain decrease is measured on rows both models trained on (see Limits); MMLU is the only held-out measure.
- **U2, forget Δ above 0:** after compression, the forget answers regain word overlap (retention) or likelihood (log-prob). This is consistent with partial reversal of the unlearning's score reduction on these measures; it does not show that factual knowledge returned. In retention units, +0.10 is a tenth of the C0v2-to-original gap.
- **U2, forget Δ covering 0:** no change detected on that measure. Report the point estimate and the upper endpoint of the 95% interval, with the uncertainty and selection limits above.
- **U2, forget Δ below 0:**
  - with retain or MMLU also below 0: consistent with broader degradation from compression rather than further unlearning, but this does not prove the mechanism;
  - with retain and MMLU showing no change detected: report the forget decrease as observed. Non-significant retain or MMLU changes do not show that the effect is specific to the forget set, and different significance outcomes do not show different effects.
- **U3:** how much qint4 alone moves forget, retain and MMLU in a model that memorised both TOFU sets. It is context for reading U2. If U3 shows a forget shift similar to U2's, that is consistent with a general effect of compressing a TOFU-trained model, but it does not prove U2 has that cause. If they differ, only the interaction (below, if adopted) tests the difference.

**Pairing rule:** a forget Δ is never reported alone. Each contrast's forget Δ sits in one row with that contrast's retain Δ, MMLU-en Δ and MMLU-hi Δ, and the model sizes.

**Interaction (proposal only, not a primary outcome):** I = U2 − U3 on forget retention, using the same author draws. It asks whether compression moves the forget set differently after unlearning. It is hard to read: C2 was selected to have forget retention ≤ 0.25, near the original's 0, while C0v2 is 1 by definition, so floor and ceiling effects differ between U2 and U3. It enters the write-up only if the team records it in PLAN before tables; otherwise it is left out. It needs no new run: it uses the same rows and author draws as U2 and U3.

### Caveats specific to these contrasts

- **Selection reaches U2.** C2 was picked for a low forget score on these same rows. Any perturbation, quantization included, may move a selected low score back up, so part of a U2 forget increase could come from the selection and not the compression. U3 has no such selection. Only two learning rates were trained, which limits this effect but does not remove it.
- **Overlap.** Forget rows are C0v2 training rows by design. Retain rows are training rows of C0v2 and of C2's retain term. Retain Δ measures how well trained material survives; MMLU is the only held-out guard.
- **Intervals are conditional on the one selected model.** The TOFU intervals resample authors and the MMLU intervals resample questions. Both treat the single training run and the learning rate chosen on these same TOFU rows as fixed. So the intervals leave out training variability (seed, data order) and the uncertainty of that choice. All 20 forget authors were in C0v2's training and in C2's unlearning; no model was trained without them.
- **Few clusters.** Each TOFU set has 20 authors. Percentile bootstrap intervals from 20 clusters can be too narrow, so CI edges near 0 are read with care.
- **What the measures are.** ROUGE-L recall is word overlap with the gold answer, not factual recall. Log-prob per token is the quantity NPO pushes down directly, so it can move without the greedy text changing. The truth ratio has no CI.
- **Multiplicity.** Three contrasts (plus two NF4) × two sets × several measures. No correction is planned. The headline is U2 forget retention; the other cells are supporting evidence and are labelled as such.

## Study U results (U1, U2, U3; NF4 absent)

**Source:** `Vedika/out/r4/contrasts.json` (md5 0e87eb20) and `table.json` (md5 5ed79584), made by `Vedika/scripts/r4_contrasts.py` from five reviewed fresh row sets:
- original: Mudit's R4a;
- C0v2 and C2: from R2;
- C1 and C3: from Mudit's rescore.

**Codex PASS 19:59.** Codex independently reproduced all 18 contrast points and intervals and the 10 model-level retention values.
- The retention denominator (mean C0v2 − mean original ROUGE-L recall) was positive in every draw: forget 0.592 (range 0.537–0.642), retain 0.624 (0.578–0.667). All six retention distributions had zero invalid or NaN draws.
- The truth-ratio summaries are descriptive (PASS 19:50).
- The sizes come from the harness summaries; Codex did not measure them independently.
- NF4 contrasts are not reported because no NF4 TOFU rows exist.

Δ = later − earlier on the same questions. The 95% CIs are an author-clustered paired bootstrap for TOFU (20 authors per set) and a question bootstrap for MMLU; 10,000 resamples, seed 0. Retention is ROUGE-L recall, word overlap only, scaled so that original = 0 and C0v2 = 1.

| Contrast | Forget retention Δ | Forget log-prob Δ (nats/token) | Retain retention Δ | Retain log-prob Δ (nats/token) | MMLU-en Δ (pp, 500 q) | MMLU-hi Δ (pp, 200 q) |
| --- | --- | --- | --- | --- | --- | --- |
| U1: C2 − C0v2 | −0.850 [−0.881, −0.818] | −1.737 [−1.900, −1.578] | −0.192 [−0.226, −0.159] | −0.115 [−0.153, −0.083] | −1.6 [−3.8, +0.6] | −1.0 [−3.5, +1.5] |
| **U2: C3 − C2** | **−0.058 [−0.081, −0.034]** (headline) | **+0.215 [+0.129, +0.305]** | −0.469 [−0.523, −0.416] | −0.305 [−0.356, −0.255] | −4.2 [−8.0, −0.4] | +1.0 [−5.0, +7.0] |
| U3: C1 − C0v2 | −0.220 [−0.252, −0.186] | −0.065 [−0.078, −0.053] | −0.219 [−0.265, −0.174] | −0.074 [−0.087, −0.061] | −3.8 [−7.6, −0.2] | +0.5 [−4.5, +5.5] |

Model-level values. Retention has a 95% author-clustered CI. Mean ROUGE-L recall, truth ratios and sizes are point values.

| Model | Forget retention | Retain retention | Forget ROUGE-L | Retain ROUGE-L | Forget truth ratio (higher = closer to indifferent) | Retain truth ratio (higher = prefers true) | Harness size_mb (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0 (by definition) | 0 (by definition) | 0.408 | 0.375 | — | — | — |
| C0v2 | 1 (by definition) | 1 (by definition) | 1.000 | 0.999 | 0.427 | 0.544 | 6388.1 |
| C2 | 0.150 [0.119, 0.182] | 0.808 [0.774, 0.841] | 0.497 | 0.879 | 0.442 | 0.529 | 6388.1 |
| C1 (qint4) | 0.780 [0.748, 0.814] | 0.781 [0.735, 0.826] | 0.870 | 0.862 | 0.448 | 0.523 | 2449.0 |
| C3 (qint4) | 0.092 [0.065, 0.120] | 0.339 [0.274, 0.407] | 0.462 | 0.586 | 0.463 | 0.511 | 2449.0 |

The harness size is parameter memory; the bf16 safetensors file is 6388.2 MiB. C2's 0.150 and 0.808 meet the selection rule (forget ≤ 0.25, retain ≥ 0.50). That is expected, because the rule picked C2 on these rows; it is not independent evidence.

### What these show

Readings follow "How each contrast is read":
- **Retention and ROUGE-L** measure word overlap between the greedy answer and the gold answer.
- **Log-prob** is the mean per-token likelihood the model gives the gold answer.

Neither is factual knowledge, and neither shows erasure or recovery of facts.

- **U2, the headline (compression after unlearning).** The two forget measures **move in opposite directions**:
  - **Forget retention decreased** by 0.058 [0.034, 0.081]: after qint4, C3's greedy answers overlap the forget gold answers less than C2's did. No increase in forget word overlap was detected.
  - **Forget log-prob increased** by 0.215 nats/token [0.129, 0.305]: C3 gives the forget gold answers higher likelihood than C2 did.
  - So, on these measures, the answers C3 writes overlap the forget answers less, while the likelihood it gives those answers rose. The likelihood rise is consistent with partial reversal of the unlearning's likelihood reduction, on that measure only. The overlap drop is not evidence of further unlearning. Neither shows what C3 knows, and the data do not say why the two diverge.
  - In the same contrast, **retain retention decreased by 0.469 [0.416, 0.523]**, retain log-prob by 0.305 [0.255, 0.356] and MMLU-en by 4.2 pp [0.4, 8.0]. MMLU-hi: no change detected, with a wide interval.
  - The forget overlap drop therefore came with a much larger retain overlap drop. That is consistent with broader degradation from compressing C2, but it does not prove the mechanism.
  - Selection may contribute to both forget cells: C2 was chosen for low forget scores on these same rows.
- **U1 (unlearning).**
  - Forget retention decreased by 0.850 [0.818, 0.881] and forget log-prob by 1.737 nats/token [1.578, 1.900].
  - Retain retention decreased by 0.192 [0.159, 0.226] and retain log-prob by 0.115 [0.083, 0.153]; the retain rows are training rows of both models.
  - MMLU: no change detected in either language; the lower ends of the estimated 95% intervals are −3.8 pp (en) and −3.5 pp (hi).
  - The forget retention cell confirms the selection rule. Forget log-prob is NPO's own objective on its own training rows. Neither is independent evidence of unlearning beyond these rows.
- **U3 (compression control).**
  - Compressing C0v2 decreased forget and retain retention by similar amounts: 0.220 [0.186, 0.252] and 0.219 [0.174, 0.265].
  - Log-prob decreased slightly on both sets: forget 0.065 [0.053, 0.078], retain 0.074 [0.061, 0.087].
  - MMLU-en decreased by 3.8 pp [0.2, 7.6]. MMLU-hi: no change detected.
  - U3 shows no forget log-prob increase like U2's. U2 and U3 start from different forget levels: C2 at 0.150 and C0v2 at 1. Whether the two contrasts differ is not tested, because the interaction is out per PLAN section 3.
- **MMLU-en under qint4:** both compression contrasts show a decrease (U2 −4.2 pp, U3 −3.8 pp). Each interval ends close to 0 (−0.4 and −0.2), so its edge is read with care.
- **Truth ratios:** small point differences with no CI. They are not interpreted.
- **Multiplicity:** the headline is U2 forget retention. The other cells are supporting evidence and are not corrected for multiple comparisons.

## Limits

- **Training overlap:** C0v2 trained on all 4,000 TOFU rows, and C2's retain term trained on rows 0-3599. All 400 scored retain rows (0-399) are training rows for both models, so retain scores measure memorised rows. Global-MMLU is the only held-out utility measure.
- **LoRA, not full fine-tuning:** C2 uses LoRA r 32 on every linear layer, while the cited NPO setting fine-tunes all weights.
- **One unlearning run:** C2 has one seed and one method (NPO with a retain term).
