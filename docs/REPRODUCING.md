# Reproducing the project

Run every command from the repository root. Read [artifacts.json](../artifacts.json) for the model and dataset repositories.

## 1. Check the tables (CPU, about a minute)

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
CUDA_VISIBLE_DEVICES="" python reproduce.py
```

This recomputes seven table groups from the saved per-question scores in `results/`: the qint4 unlearning contrasts and levels, the refusal comparisons, XSTest types, answer-length caps, utility, and each compressed model against the original. It also checks the KL gap between the protected model and both controls. It works on a temporary copy and stops on any mismatch. To check a downloaded copy of the Hugging Face evaluation dataset instead, pass its folder: `python reproduce.py /path/to/export`.

## 2. GPU environment

We used Python 3.12, torch 2.11.0+cu128, transformers 5.22.1, peft 0.21.2, optimum-quanto 0.2.7, bitsandbytes 0.50.2, AuditKIT 1.1.3, CircuitKIT 0.2.0 and SafeTune 0.1.10, on one A100 40 GB per job.

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda-12.9   # any compatible toolkit; needed for quanto kernels
python evaluation/build_prompt_sets.py
python evaluation/build_tofu_sets.py
```

You can download each model, for example:

```bash
hf download Mudit666/tiny-aya-tofu-npo-c2 --local-dir checkpoints/c2
hf download Mudit666/tiny-aya-tofu-npo-c2-qint4 --local-dir checkpoints/c3
```

The other labels (original, c0v2, c1, plain-q4, random30-q4 and protected-q4) are listed in `artifacts.json`.

## 3. Train and compress

```bash
CUDA_VISIBLE_DEVICES=0 python training/train_c0.py --config configs/c0_v2.yaml
CUDA_VISIBLE_DEVICES=0 python training/unlearn_c2.py --config configs/c2.yaml --lr 3e-5
CUDA_VISIBLE_DEVICES=0 python compression/build_qint4.py checkpoints/c2 runs/rebuilt-c3
```

The qint4 builder records source and output hashes and checks that the model reloads. A rebuilt file may differ byte-for-byte from the published one. For the refusal study, `compression/aya_refusal_circuit.py` finds the circuit and `compression/make_baselines.py` builds the plain and random variants.

## 4. Evaluate

Unlearning under qint4. Use a new output folder for every run:

```bash
# For each label: original, c0v2, c1, c2, c3
CUDA_VISIBLE_DEVICES=0 python training/r2_rescore.py c2 checkpoints/c2 --out runs/tofu
CUDA_VISIBLE_DEVICES="" python evaluation/r4_contrasts.py original c0v2 c1 c2 c3
```

Refusal and utility:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/mmlu_only.py original --out runs/study_s/mmlu
# For each of plain-q4, random30-q4, protected-q4:
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py plain-q4 checkpoints/plain-q4 --out runs/study_s/eval
CUDA_VISIBLE_DEVICES=0 python evaluation/mmlu_only.py plain-q4 checkpoints/plain-q4 --out runs/study_s/mmlu
CUDA_VISIBLE_DEVICES="" python evaluation/r5_tables.py
```

NF4 follow-up. This builds NF4 copies of C0v2 and C2, scores bf16 against NF4, and writes the contrasts:

```bash
evaluation/run_nf4_followup.sh c0 npo
```

`runs/` holds model answers and judge text, so keep it private and publish only numeric exports. MANSU commands are in [MANSU on Tiny Aya](MANSU.md).

## 5. CPU tests

Run these in the activated GPU environment, because quanto needs `ninja` on `PATH`.

```bash
for t in shared/test_cluster_ci.py shared/test_paired_ci.py shared/test_flip_rate.py shared/test_kl_divergence.py \
         evaluation/test_r4_contrasts.py evaluation/test_r5_tables.py training/test_verify_merge.py \
         compression/test_verify_quant.py training/test_mansu.py compression/test_mansu_attribution.py; do
  CUDA_VISIBLE_DEVICES="" python $t || echo "FAILED: $t"
done
```

A few tests need extra local files. `evaluation/test_evaluate.py` and `evaluation/test_mmlu_only.py` need the rebuilt prompts. `compression/test_build_qint4.py`, `shared/test_inmemory_model.py` and `evaluation/test_judge_agreement.py` need the cached Tiny Aya or WildGuard tokenizer. `evaluation/test_tofu_eval.py` needs a GPU.
