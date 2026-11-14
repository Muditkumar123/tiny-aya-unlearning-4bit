# Vedika's part of the public code repository (D5): file list, dependencies, paths, data scope

For `Mudit/release/RELEASE_MANIFEST.md`. Status: Vedika (human) approved D5 as written and chose "IDs + builders only" (2026-11-13 00:42, Vedika.md). Nothing has been copied, committed or pushed. Repo paths equal the workspace paths, so no import or path changes are needed.

## Include (source path = repo path)

| Path | What | Notes |
| --- | --- | --- |
| `Vedika/scripts/evaluate.py` | Study S safety harness (generation, KL, WildGuard via AuditKIT GuardJudge, refusal Δ) | needs the four Study S checkpoints as CLI paths |
| `Vedika/scripts/tofu_eval.py` | Study U harness (TOFU, MMLU) | |
| `Vedika/scripts/mmlu_only.py` | Study S utility (MMLU) | its "original" check reads the reviewed R4a rows at `Mudit/out/r4a_original/original` (lab path, not in the repo); the export carries the same rows under `Vedika/release/export/study_u/original/` |
| `Vedika/scripts/r4_contrasts.py`, `r5_tables.py` | Study U contrasts; Study S tables | `SOURCES` in r4_contrasts names lab row folders; the public route is `reproduce.py` |
| `Vedika/scripts/judge_agreement.py` | the 20-pair judge gate | its pairs hold Aya model answers, so `Vedika/out/judge_agreement/` stays lab-only; a public rerun must rebuild pairs from its own fresh answers |
| `Vedika/scripts/build_prompt_sets.py`, `build_tofu_sets.py` | rebuild all 8 `common/data/*.json` files from pinned sources | uses `Mudit/release/discovery_ids.json` (must ship with Mudit's part). Checked 00:43: a rebuild into an empty `--out` is byte-identical to all 8 files (`Vedika/out/logs/rebuild_2026-11-13.txt`) |
| `Vedika/scripts/test_evaluate.py`, `test_judge_agreement.py`, `test_r4_contrasts.py`, `test_r5_tables.py` | CPU tests | need the Tiny Aya and WildGuard tokenizers in the HF cache |
| `Vedika/scripts/test_tofu_eval.py`, `test_stop_gpu.py` | GPU tests | |
| `Vedika/release/export.py`, `reproduce.py`, `README.md`, `RELEASE_FILES.md` | numeric export builder; CPU reproduction of all reviewed tables | |
| `Vedika/release/export/` | numbers and ids only, 73 files, 3.1 MB (Codex PASS 21:09) | could live in the HF data collection instead; `reproduce.py` takes its path as an argument |
| `Vedika/writeup/EVALUATION_SECTION.md` | methods section (Codex PASS 21:02) | |
| `common/addons/{cluster_ci,paired_ci,kl_divergence,flip_rate,inmemory_model}.py` and their `test_*.py` | statistics and model wrapper used by every harness | Vedika's write scope; skip `__pycache__/` |

## Exclude, and why

| Path | Why |
| --- | --- |
| `Vedika/out/**` | lab outputs: model answers to harmful prompts, raw judge text, caches, logs, PIDs, launchers with local paths. The release export replaces them. |
| `common/data/*.json` | dataset text. Decision: IDs + builders only; the builders recreate these files exactly. Other owners' scripts read 5 of them, so the README must say to run the builders first. |
| `Vedika/scripts/c1_gate_early.py`, `h1.py`, `lp_specificity.py`, `writeup_tables.py` | predecessor Plan B helpers, not used by the current results |
| `Vedika/CLAUDE.md`, `Vedika/PROMPT.md` | agent instructions, not project code (team's call) |
| `**/__pycache__/` | build files |

## Dependencies (as run, `lexsi` env)

Python 3.12.16; torch 2.11.0+cu128; transformers 5.22.1; auditkit 1.1.3; safetune 0.1.10 (only for `judge_safetune`, the gate's comparison path); circuitkit 0.2.0 and optimum-quanto 0.2.7 (loading qint4 checkpoints; quanto kernels need `CUDA_HOME=/usr/local/cuda-12.9` and `ninja` on PATH on our machine); rouge_score 0.1.2; huggingface_hub 1.33.0; numpy 2.4.6; pandas 3.1.0; pyarrow 26.0.0.
Even the CPU reproduction imports torch and auditkit, because `reproduce.py` reuses the harness modules.

## Commands in the repo

```bash
# 1. data (CPU, pinned downloads ~12 MB; refuses to overwrite)
python Vedika/scripts/build_prompt_sets.py && python Vedika/scripts/build_tofu_sets.py
# 2. reproduce every reviewed table from the numeric export (CPU)
python Vedika/release/reproduce.py
# 3. tests (CPU)
python Vedika/scripts/test_evaluate.py && python Vedika/scripts/test_r4_contrasts.py && python Vedika/scripts/test_r5_tables.py
```
Full rescoring (GPU) uses `evaluate.py <label> <checkpoint>`, `tofu_eval.py <label> <checkpoint>` and `mmlu_only.py <label> <checkpoint>` with the settings in `EVALUATION_SECTION.md`; checkpoints come from the team's HF model collection (namespace D2 still open).

## Data scope against the licence checklist (dataset cards, not legal clearance)

| Set | In the repo | Card licence | Open point |
| --- | --- | --- | --- |
| HarmBench held-out (`walledai/HarmBench` @fb6c2afd) | builder + ids `hb_std_<row>`; discovery rows via `discovery_ids.json` | MIT; HF mirror auto-gated | upstream centerforaisafety/HarmBench licence not checked |
| Aya red-teaming (@5a16fee0) | builder + ids | Apache-2.0 | NOTICE/attribution text |
| XSTest safe (@f600c994) | builder + ids | CC-BY-4.0 | attribution in README; upstream repo licence not checked |
| TOFU (@324592d8) | builder + ids | MIT | citation |
| Global-MMLU (@0e619dbe) | builder + ids | Apache-2.0 | MMLU (MIT) attribution |
| Alpaca partners (`tatsu-lab/alpaca` @dce01c9b) | row ids + sha256 only, in Mudit's `discovery_ids.json` | CC-BY-NC-4.0 | no text shipped; NC applies if text were re-hosted |
| WildGuard (judge) | not shipped; downloaded by the user | Apache-2.0; gated with AI2 research-use terms | users must accept the terms |
| Tiny Aya and derivatives | not in the code repo | CC-BY-NC-4.0 + Cohere AUP | on HF, with notices |

## Open for the team

- Whether `Vedika/release/export/` ships in the code repo, the HF data collection, or both.
- Redacting local paths: the export already shortens them to `~/`; the scripts contain no home-directory paths, only workspace-relative ones.
