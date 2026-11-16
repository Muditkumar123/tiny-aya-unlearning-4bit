# MANSU run 1: per-tensor range floor

This was our first full MANSU run on Tiny Aya, starting from C0v2 and using the circuit in MLP layers 13–17. Training used free-text answers, one example per step, and the Algorithm 1 Fisher mask (99th percentile). The floor size came from each weight matrix's value range, which is a rough per-tensor stand-in for one NF4 step.

| Checkpoint | Forget MCQ accuracy (40) | English MMLU (100) |
| --- | ---: | ---: |
| Starting model (C0v2) | 77.5% | 47% |
| Step 5, before the floor (selected) | 77.5% | 47% |
| Step 30, before the floor | 77.5% | 48% |
| Step 5, after the floor | 25.0% | 25% |

Training kept forget accuracy at 77.5% at every checkpoint. The floor enlarged the change in 334,706,430 weights to at least 0.022–0.082, depending on the matrix. That is 1–4 times a typical weight. Utility fell 22 pp, past the 8 pp limit, so the model was not exported.

Run 2 replaced this floor with the block-wise NF4 step from the paper's Appendix D. Run 3 also switched to the paper's training format. See [MANSU on Tiny Aya](../../docs/MANSU.md).

Files: `training.json` (loss history and validation), `attempt_summary.json`, `attribution_before.json` (circuit scores), `inputs_receipt.json` (input IDs), and `execution_stages.json`.
