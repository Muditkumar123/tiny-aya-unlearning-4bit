# Ganesh's release note: files, dependencies, portability (J1, J4)

Written 2026-11-13 by Ganesh's Claude for `Mudit/release/RELEASE_MANIFEST.md` (Orchestrator 21:44 and 22:21; judge findings J1 and J4). Every path keeps its workspace location (D5 layout). Nothing here is committed, pushed or uploaded.

## Files to ship (source path = target path)

| Path | md5 | Bytes | Status | Why |
| --- | --- | --- | --- | --- |
| `Ganesh/scripts/train_c0.py` | cfdf0721 | 4,197 | historical, unchanged since before the run | trains the C0v2 LoRA and merges it; run with `--config Ganesh/configs/c0_v2.yaml` |
| `Ganesh/scripts/unlearn_c2.py` | 8b233627 | 7,796 | historical, unchanged since before the run | SafeTune NPO unlearning and merge (`--lr 3e-5` gave C2) |
| `Ganesh/scripts/select_c2.py` | a9781d52 | 2,205 | historical record | applied the frozen lr rule; reads lab outputs that are not shipped, so it is not runnable from the release |
| `Ganesh/scripts/run_c2_grid.sh` | 5cee3f0e | 1,420 | archived launcher | the lr grid as run; fixed paths (`cd ~/Lexsi`, CUDA 12.9, conda env) |
| `Ganesh/scripts/merge_diagnostics.py` | 59d8faaa | 2,606 | current (unchanged since 11-07) | bf16 merge statistics, CPU |
| `Ganesh/scripts/verify_merge.py` | f1df6f6d | 3,907 | current, Codex PASS 01:54 | bitwise merge check, CPU |
| `Ganesh/scripts/test_verify_merge.py` | f0e2b848 | 2,292 | current | its CPU test |
| `Ganesh/scripts/r2_rescore.py` | c66903dc | 4,433 | current, Codex PASS 04:29 | runs the TOFU harness and records padding, dtypes, prompt lengths |
| `Ganesh/scripts/compare_regen.py` | 139e7d6f | 2,368 | current, Codex PASS 04:29 | fresh vs saved row comparison; Mudit's launchers md5 it |
| `Ganesh/scripts/test_card_snippet.py` | 6a7b7ebe | 1,903 | current, Codex PASS 06:48 | runs each card's usage code on CPU |
| `Ganesh/configs/c0_v2.yaml` | b8de4bd1 | 1,378 | frozen before the run | C0v2 settings |
| `Ganesh/configs/c2.yaml` | 9f107caf | 3,223 | frozen before the run | C2 settings and selection rule |
| `Ganesh/configs/c0.yaml` | 4da95baa | 1,291 | superseded (C0 v1) | kept only because it is `train_c0.py`'s default `--config`; C0v2 does not use it |
| `Ganesh/release/c0v2/README.md`, `Ganesh/release/c2/README.md` | see Ganesh.md | | Codex PASS 21:36 (J6 wording update pending review) | model cards, uploaded with the weights |
| `Ganesh/release/SOURCES.md` | see Ganesh.md | | | claim-to-source map for the cards |
| `Ganesh/release/RELEASE_NOTE.md` | this file | | | |
| `Ganesh/release/check_portable.sh` | d616794b | 2,861 | new | the CPU portability check below |
| `Ganesh/writeup/SAFETUNE_SECTION.md` | see Ganesh.md | | Codex PASS 20:04 (J6 wording update pending review) | methods and results; WRITEUP Appendix A copies it |
| `Ganesh/out/r2_merge_c0v2.json`, `r2_merge_c2.json` | 12c45ed4, 7f41d2fe | 715, 685 | R2 records | hashes, bitwise merge and reload results |
| `Ganesh/out/r2_rescore_vs_saved_c0v2.json`, `_c2.json` | b8bd509f, 92dec961 | 478, 474 | R2 records | fresh = saved row checks |
| `Ganesh/out/r2_rescore/{c0v2,c2}/run_settings.json` | dac6cf46, cdeb46be | 18,042, 18,038 | R2 records, **need redaction** | observed padding, dtypes, prompt lengths per row id; field `weights_id` holds the absolute local path |
| `Ganesh/out/c0v2-adapter/train_log.json` | 5299a3dc | 39,712 | historical record | training loss log |
| `Ganesh/out/c2-lr3e-05-adapter/{unlearn_log,merge_stats,merge_diagnostics}.json` | 0ec1d0a4, d82c856f, 393c3cae | 37,447, 156, 403 | historical record | unlearning log and merge statistics |
| `Ganesh/out/c2_selection.json` | b8c4f6f9 | 936 | historical record | the selection output; its numbers come from the predecessor harness and are not quoted as results |
| `Ganesh/out/r2_rescore.sh`, `Ganesh/out/r2_cpu_checks.sh` | bf29a8eb, 1d047e8d | 1,026, 759 | archived launchers, **need redaction** | the cards cite them; `PY=` holds the absolute conda path |

**Redaction:** the four files marked above contain the local home path. The redaction script is Mudit's (`Mudit/scripts/redact_release.py`, fixed list of 12 files). Either add these four paths to its list, or leave them out. If they are left out, the cards' mentions of `r2_rescore.sh` stay valid as a description, and `run_settings.json` evidence stays lab-only.

**Leave out:**
- the weights and adapters (`Ganesh/out/*-merged/`, `*-adapter/*.safetensors`): they go to Hugging Face;
- `Ganesh/out/r2_rescore/*/{forget,retain,mmlu_*}.jsonl` and `cache/`: they hold generated answer text, and Vedika's numbers-only export covers their scores;
- the predecessor C0 v1 and lr 1e-5 models and logs (`c0-*`, `c2-lr1e-05-*`, `quick_check_c0*`, `tofu_c2grid/`): not part of the reported study;
- `Ganesh/scripts/quick_check_c0.py`: a predecessor sanity check for C0 v1;
- `r2_regen*`: the 32-row check that the full rescore replaced;
- `__pycache__/`.

## Dependencies of each runnable file

| File | Imports (beyond the standard library) | Reads | Writes | Network |
| --- | --- | --- | --- | --- |
| `train_c0.py` | torch, PyYAML, datasets, peft, transformers | the config; `CohereLabs/tiny-aya-global` @af89d219; `locuslab/TOFU` full @324592d8 | `Ganesh/out/c0v2-adapter/`, `Ganesh/out/c0v2-target-merged/` | downloads the pinned model and data if not cached; needs a CUDA GPU |
| `unlearn_c2.py` | torch, PyYAML, datasets, peft, transformers, **safetune** | the config; **C0v2 merged weights at `Ganesh/out/c0v2-target-merged`**; `common/data/tofu_author_split.json`; TOFU retain90 @324592d8; tokenizer @af89d219 | `Ganesh/out/c2-lr3e-05-{adapter,merged}/` | as above; needs a CUDA GPU |
| `verify_merge.py` | torch, transformers, peft, safetensors | base (HF id + revision, or a local folder), adapter, merged folder | JSON with `--out` | none (`local_files_only`) |
| `merge_diagnostics.py` | torch, safetensors, peft, transformers | adapter, merged, target folders | stdout and `merge_diagnostics.json` in the adapter folder | none |
| `r2_rescore.py` | torch, transformers, auditkit; **`Vedika/scripts/tofu_eval.py`, `Vedika/scripts/evaluate.py`, `common/addons/{inmemory_model,cluster_ci,flip_rate,kl_divergence,paired_ci}.py`** | the checkpoint; `common/data/{tofu_author_split,tofu_forget10_perturbed,tofu_retain_perturbed,global_mmlu_en500,global_mmlu_hi200}.json`; the base tokenizer | `<--out>/<label>/` (use a new, empty folder) | tokenizer from the HF cache; needs a CUDA GPU |
| `compare_regen.py` | none | two label folders | JSON with `--out` | none |
| `test_card_snippet.py` | torch, transformers | a card and a local merged folder (its tokenizer) | nothing | none |

The `common/data` files and Vedika's scripts and addons are Vedika's to ship. These rows only record that Ganesh's scripts need them.

## Portability (J4)

- **Run from the repository root.** All Python paths are repository-relative. The docstrings' "Run from ~/Lexsi" means the repository root. The only fixed local paths are in the two archived launchers and `run_c2_grid.sh`.
- **One prerequisite outside the repository:** `unlearn_c2.py` reads C0v2's merged weights from `Ganesh/out/c0v2-target-merged`. Either produce them with `train_c0.py --config Ganesh/configs/c0_v2.yaml` or download C0v2 from Hugging Face into that folder.
- **Default config trap:** `train_c0.py` defaults to `c0.yaml` (C0 v1). C0v2 needs `--config Ganesh/configs/c0_v2.yaml`, as the cards say.
- **C2 adapter's base:** `Ganesh/out/c2-lr3e-05-adapter/adapter_config.json` names its base as the local path `Ganesh/out/c0v2-target-merged`. The uploaded copy needs the C0v2 repo id there, or the card must tell users to pass the base explicitly. Added to SOURCES.md open items.
- **Not runnable from the release:** `select_c2.py` reads `Vedika/out/tofu_v2` and `Ganesh/out/tofu_c2grid`, which are lab outputs. It is a record of how the lr was chosen.

**Current code vs what produced the weights.** No code hashes were recorded at run time. The file modification times are the only evidence, and they are not proof:
- `train_c0.py` (11-06 23:37) and `c0_v2.yaml` (11-07 07:48) predate the C0v2 adapter (11-07 09:06).
- `unlearn_c2.py` and `c2.yaml` (11-07 09:17), and `select_c2.py` and `run_c2_grid.sh` (09:43), predate the C2 adapter (10:31).
- `merge_diagnostics.py` (10:42) predates its saved output (10:43).
- The R2 scripts are new code from 2026-11-12. With them, the merges reproduce bitwise and the scores reproduce exactly (R2).

**Library versions:**
- At training time, the saved files record only transformers 5.22.1 (merged `config.json`) and PEFT 0.21.2 (`adapter_config.json`). torch and SafeTune versions were not recorded.
- The current environment, which was used for all R2 checks: Python 3.12.16, torch 2.11.0+cu128 (CUDA 12.8), transformers 5.22.1, peft 0.21.2, datasets 5.1.0, safetensors 0.8.0, accelerate 1.15.0, safetune 0.1.10, auditkit 1.1.3, numpy 2.4.6, PyYAML 6.0.3, rouge_score 0.1.2, huggingface_hub 1.33.0, tokenizers 0.23.3.

## CPU check (done 2026-11-13, clean staged copy, no weights)

The check staged the files listed here plus Vedika's `tofu_eval.py`, `evaluate.py`, `common/addons/*.py` and the five `common/data` files in a scratch folder. It then ran `PY=<lexsi python> bash Ganesh/release/check_portable.sh <stage>`. Result: all checks passed.
- `--help` parses for `train_c0`, `unlearn_c2`, `verify_merge`, `r2_rescore` and `compare_regen`, so the imports resolve from the staged layout.
- The configs pin 40-hex revisions.
- The C2 forget file is present.
- The C2 target is absent, as expected; the check prints that note.
- `test_verify_merge.py` passes.
- There are no absolute home-directory paths in Ganesh's code, configs or docs.

The first attempt failed because the stage lacked `common/addons/{flip_rate,kl_divergence,paired_ci}.py`, which `evaluate.py` imports. That is why the dependency table lists them. No GPU run, training or download was done.

## Added 2026-11-13 00:59

- **Lab-only, not shipped** (Orchestrator 03:34; Mudit 03:56): `Ganesh/release/make_upload_copies.py` (md5 2f13c2c9) and `Ganesh/release/test_make_upload_copies.py` (md5 737631b9).
  - They read the lab weights and configs in `Ganesh/out/`, which a user of the code repository does not have.
  - What they produce, the edited configs and cards, ships in the Hugging Face repos, and the cards disclose the edits.
- **Internal working files (team decides whether to publish):**
  - `Ganesh/release/HF_UPLOAD_MAP.md`;
  - `Ganesh/release/quantized_card_drafts/`, drafts for Mudit's cards. Its `q4_load_check.py` needs a local checkpoint.
- **Lab records, not shipped** (they contain local paths): `Ganesh/out/q4_load_check/` and `Ganesh/out/upload_copies_check/`.
- **Not in the code repository:** `Ganesh/upload_stage/`, the HF upload candidate (cards, configs, tokenizer files, `upload_manifest.json`). It goes to Hugging Face, not GitHub.
