# MANSU run 2: NF4 block floor

This run is the same as run 1 except for the floor. Each weight change is enlarged to at least 0.704 × one NF4 step, where a step is the maximum of the weight's 64-weight block × 0.0796, as in the paper's Appendix D. Before the run, we checked this step size against bitsandbytes' own block scales on test blocks.

| Checkpoint | Forget MCQ accuracy (40) | English MMLU (100) |
| --- | ---: | ---: |
| Starting model (C0v2) | 77.5% | 47% |
| Step 5, before the floor (selected) | 77.5% | 47% |
| Step 30, before the floor | 77.5% | 47% |
| Step 5, after the floor | 22.5% | 21% |

Training again kept forget accuracy at 77.5% throughout. Utility fell 26 pp after the floor, past the 8 pp limit, so the model was not exported. Run 3 uses the paper's training format and Table 10 Fisher mask, and adds a random-sign control. See [MANSU on Tiny Aya](../../docs/MANSU.md).

Files: `training.json`, `attempt_summary.json`, `nf4_scale_preflight.json` (the step-size check), `attribution_before.json`, `inputs_receipt.json` and `execution_stages.json`.
