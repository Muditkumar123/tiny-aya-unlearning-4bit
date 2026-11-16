# MANSU on Tiny Aya

MANSU (Mechanistic-Aligned Null-Space Unlearning) comes from [Forgetting That Sticks](https://arxiv.org/abs/2605.15138). It is designed so that forgetting survives 4-bit quantization. We implemented it ourselves from the paper's Algorithm 1 and appendices, using CircuitKIT for attribution, and ran it on our fine-tuned Tiny Aya (C0v2).

## What we built

| MANSU component | Our code |
| --- | --- |
| Forget circuit: EAP-IG on forget-knowledge pairs, MLP layers ranked by incoming edges | [discover_mansu.py](../compression/discover_mansu.py) |
| Circuit-only training: raise forget loss, KL anchor to the frozen start model on retain data, diagonal-Fisher retain mask | [mansu.py](../training/mansu.py) |
| Magnitude floor: enlarge each weight change to about one NF4 step so it survives quantization | [mansu.py](../training/mansu.py) |
| Run manager: fixed inputs, checkpoint selection, utility guard, bf16 export checks | [run_mansu.py](../training/run_mansu.py) |
| NF4 build with a save-and-reload check | [build_mansu_nf4.py](../compression/build_mansu_nf4.py) |
| Circuit Attribution Divergence (CAD) and activation-shift diagnostics | [mansu_diagnostics.py](../evaluation/mansu_diagnostics.py) |
| Paper-settings run with a random-sign control | [nf4_followup.py](../evaluation/nf4_followup.py) |

CPU tests on tiny random models cover the loss direction, the frozen reference, the masks, the floor, checkpoint selection and a full train -> save -> reload cycle:

```bash
CUDA_VISIBLE_DEVICES="" python training/test_mansu.py
CUDA_VISIBLE_DEVICES="" python compression/test_mansu_attribution.py
```

## Setup on Tiny Aya

- **Circuit.** EAP-IG on 50 forget-knowledge pairs picked MLP layers 13–17: 15 weight matrices, 338 million weights. Everything else stays frozen.
- **Training.** Learning rate 1e-6, AdamW without weight decay, KL weight 200, up to 30 steps, with validation every 5 steps.
- **Data.** For each of the 20 forget authors, 15 questions go to training, 2 to validation and the rest to a final test set. Retain questions come from separate authors.
- **Validation.** 40 forget multiple-choice questions and 100 English MMLU questions. The run keeps the checkpoint with the lowest forget accuracy whose MMLU drop is at most 8 pp.
- **Floor.** After training, every change to a weight that is allowed to move is enlarged to at least 0.704 × one NF4 step (the 64-weight block maximum × 0.0796), following Appendix D.

Where the paper describes two options, we picked one and recorded it:
- **Fisher mask.** Algorithm 1 uses the 99th percentile; Table 10 uses 0.1 × the mean. Runs 1–2 used the first, which leaves 99% of circuit weights free. Run 3 used the second, which leaves 7.8% free.
- **Small updates.** Equations 4 and 10 scale them up to the floor, and that is what we do. Appendix B.3 describes zeroing them instead; that is available as the `zero` policy.

## Results

| Run | Training format | Fisher mask | Floor | Forget MCQ: start -> after floor | MMLU: start -> after floor |
| --- | --- | --- | --- | ---: | ---: |
| 1 | Free-text answers, batch 1 | 99th percentile | Per-tensor range | 77.5% -> 25.0% | 47% -> 25% |
| 2 | Free-text answers, batch 1 | 99th percentile | NF4 block (App. D) | 77.5% -> 22.5% | 47% -> 21% |
| 3 | Answer letter, batch 8 (paper B.1, Table 10) | 0.1 × mean (Table 10) | NF4 block (App. D) | 77.5% -> 27.5% | 47% -> 25% |
| 3, control | Same as run 3 | Same | Same sizes, random signs | 77.5% -> 80.0% | 47% -> 47% |

In every run, training alone kept forget accuracy at 77.5% at every checkpoint, so the step-5 checkpoint was selected and floored.

## What we learned

1. **The paper's update-size gap is real on Tiny Aya.** At the paper's learning rate, 30 steps can move a weight by at most 3×10⁻⁵, about 0.13% of a typical weight in these layers. One NF4 step is 26–28% of a typical weight. Without a floor, even a successful update would be 180–220 times too small to change the 4-bit model.
2. **Saturated answers give little training signal.** The fine-tuned model is near-certain of its forget answers (median top-letter probability 0.9996). Over 30 steps no forget prediction changed, and the largest weight change was 5×10⁻⁶.
3. **The floor amplifies direction, not noise.** In run 3 the floor moved 26.4 million weights (7.8% of the circuit) by 10–20% of a typical weight, which is at least 400 times more than any training move.
   - With training's signs, the model stops answering with a letter: its probability on A–D drops to about zero, and it "picks" A for 88 of 100 MMLU questions.
   - With random signs, the same change is harmless: 1 of 40 forget answers and 6 of 100 MMLU answers changed.
   - So a floor is safe at this size. What it needs is a training direction that targets the forget facts.
4. **Our utility guard worked as designed.** Every floored model lost more than 8 pp of MMLU, so none was exported. That means the NF4 build and CAD check never received a MANSU model, although both are implemented and tested.

The next step would be a stronger forget signal before the floor, for example more steps or a less saturated starting model. After that, we would build the NF4 copy and run CAD with the code above.
