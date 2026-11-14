# C0v2/C2 model cards: sources and open items (internal, not for upload)

Drafts: `Ganesh/release/c0v2/README.md`, `Ganesh/release/c2/README.md`. PLAN task F2 preparation (Orchestrator READY 2026-11-12 06:39). Local only; nothing uploaded.

## Where each card claim comes from

| Claim | Source | Review status |
| --- | --- | --- |
| sha256 of merged weights and adapters, 290 tensors bitwise, 252 LoRA linears, 3,349,227,520 bf16 params, clean reload | `Ganesh/out/r2_merge_{c0v2,c2}.json` (Ganesh.md 01:53) | Codex 01:54: hashes independently recomputed; merge run is owner evidence with reviewed checker code |
| Fresh greedy run reproduces saved answers 400/400 + 400/400; settings (left padding, bf16, fp32 logits, batch 16, stop [6, 3], 128 tokens, prompts 371–414, A100, versions) | `Ganesh/out/r2_rescore/{c0v2,c2}/run_settings.json`, `Ganesh/out/r2_rescore_vs_saved_{c0v2,c2}.json` (Ganesh.md 04:28) | Codex PASS 04:29: rows, cache equality and prompt lengths checked on CPU; GPU run is owner evidence |
| bf16 merge statistics 0.526 / 1.024 / 0.886 / 0.865 / 42.5%, over 2,812,280,832 weights | `Ganesh/out/c2-lr3e-05-adapter/merge_diagnostics.json`, `Ganesh/out/r2_merge_diagnostics_c2.log` (Ganesh.md 04:28) | wording reviewed by Codex 04:29; numbers are owner evidence, not rerun by Codex |
| Training and unlearning settings | `Ganesh/configs/c0_v2.yaml`, `Ganesh/configs/c2.yaml`, `Ganesh/scripts/train_c0.py` (truncation assert, line 33), `Ganesh/scripts/unlearn_c2.py` | frozen configs; not separately audited |
| C0v2 trained on all 4,000 rows, 2,500 steps | Ganesh.md 01:46 (`c0v2-adapter/train_log.json`) | cited in PLAN section 2 |
| C2 forget file = TOFU `full` rows 3600–3999, 20 authors; `forget10.json` = `full[3600:]`; `retain90.json` = `full[:3600]` | checked 2026-11-12 06:40-06:44 IST against the local TOFU cache at 324592d8 and `common/data/tofu_author_split.json` (question and answer equality) | owner check, this entry |
| `tokenizer.json` byte-identical to base af89d219; chat template = base `default` template; one rendered test message gives identical ids | checked 2026-11-12 06:40-06:44 IST (sha256; template string equality) | owner check, this entry |
| `generation_config.json` stops only at id 3; id 6 = `<\|END_OF_TURN_TOKEN\|>`; the model ends turns with 6 | shipped file; tokenizer; `Vedika/scripts/tofu_eval.py` lines 42–43 | owner check, this entry |
| Sizes 6,388.2 MiB and 230.7 MiB | `ls -l` of the shipped files, 2026-11-12 | owner check |
| lr selection rule; grid stopped after 3e-5; 1e-4 never trained | `Ganesh/configs/c2.yaml`; `Ganesh/out/c2_selection.json` (predecessor record, numbers not re-verified, not quoted) | rule text frozen in config |
| Licences: base CC-BY-NC-4.0 + AUP + gate text; TOFU MIT; SafeTune LSAL v1.2 | base `README.md` front matter at af89d219; TOFU `README.md` at 324592d8; installed `safetune` 0.1.10 package metadata | read locally |
| Citations: Tiny Aya arXiv 2603.11510; TOFU arXiv 2401.06121; NPO arXiv 2404.05868 | base README; TOFU README; `safetune/interventions/unlearn/npo.py` docstring | read locally |
| Evaluation tables: per-model values with CIs (original, C0v2, C1, C2, C3) and paired contrasts U1/U2/U3 | `Vedika/out/r4/table.json` (md5 5ed79584) and `contrasts.json` (md5 0e87eb20), from `Vedika/scripts/r4_contrasts.py` (md5 a427ca2d) over reviewed rows (`Ganesh/out/r2_rescore/{c0v2,c2}`, `Mudit/out/r4_rescore/{c1,c3}`, `Mudit/out/r4a_original/original`) | Codex PASS 19:59 (all 18 contrasts and 10 retention levels reproduced). Section wording PASS 20:04 (`Ganesh/writeup/SAFETUNE_SECTION.md` 3d035ab7). Per-model lp, MMLU and the original's truth ratios first appear in the cards (Ganesh.md 21:08); all 189 card numbers were script-matched to these two files |
| Reproduction commands | the scripts' own usage lines; `Ganesh/out/r2_cpu_checks.sh` (verify_merge commands), `Ganesh/out/r2_rescore.sh` (rescore commands), `Ganesh/scripts/run_c2_grid.sh` (lr grid), `Mudit/out/r4a_original/run.sh` (original rescore), `Vedika/scripts/r4_contrasts.py` docstring | Training/unlearning commands (step 1) were not rerun; the cards say so. Route A = `Vedika/release/reproduce.py` (Codex PASS 21:10; rerun by Ganesh 21:10, all 7 checks exact, exit 0). Route B scoring uses a fresh `--out`; r4_contrasts `SOURCES` wiring explained (audit 21:09) |

## Open before any upload

1. **Namespace and repo names:** filled with the Orchestrator's local default (01:02): `Mudit666/tiny-aya-tofu-c0v2`, `Mudit666/tiny-aya-tofu-npo-c2`, the five qint4 ids and the dataset `Mudit666/tiny-aya-compression-evaluation`. Each card says these are planned, not live. The active account was verified; the repos do not exist yet, and publication still needs approval.
2. **Code URL:** filled with the planned `https://github.com/Muditkumar123/tiny-aya-unlearning-4bit`. It is not pushed yet, and each card marks it as planned.
3. **Adapter layout:** same repo as the merged weights (subfolder) or a separate repo. `<adapter path>` is a placeholder. The C2 adapter's `adapter_config.json` names its base as the local path `Ganesh/out/c0v2-target-merged`; the uploaded copy needs the C0v2 repo id there (a release-copy edit, not a change to the lab file), or users must pass the base explicitly.
4. **generation_config.json:** decided by the Orchestrator default 00:56. The upload copies use `eos_token_id` [6, 3], built by `make_upload_copies.py`, which keeps a before/after digest log; the lab files are unchanged. The cards disclose the edit.
5. **SafeTune licence:** the C2 card names SafeTune's LSAL v1.2. Whether a model trained with SafeTune must carry LSAL terms is a question for the team. Not decided here.
6. **Team approval** for every HF upload (AGENTS.md).

Closed: gating (D1). The human decided the models are public and ungated with licence and AUP notices (PLAN D1, 12:44); the placeholder comments are removed. Metrics: filled from the reviewed tables above.
