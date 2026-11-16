# Results

Every number here is a per-question score or a summary of one. The files contain IDs, scores, labels and settings, never prompts, gold answers or model answers. `python reproduce.py` recomputes the qint4 and refusal tables from these files.

| Folder or file | What it answers |
| --- | --- |
| [Scores per model, qint4](tables/study_u_levels.md) | How each model scores on forget, retain and MMLU |
| [Changes after unlearning and qint4](tables/study_u_contrasts.md) | What NPO changed, and what compression changed afterwards |
| [NF4 follow-up](nf4_followup/README.md) | The same question with NF4, plus the paper's multiple-choice metric and MANSU run 3 |
| [Protected vs random vs plain](tables/study_s_tables.md) | Whether keeping the refusal circuit in bf16 helps |
| [Each compressed model vs the original](tables/study_s_vs_original.md) | Refusal, flips, KL and size against bf16 Tiny Aya |
| [MANSU run 1](mansu_20261115_attempt/RESULTS.md) · [MANSU run 2](mansu_nf4_scaled_20261115_attempt/RESULTS.md) | The first two MANSU runs |
| `study_u/`, `study_s/` | Per-question rows behind the tables |
| `provenance/` | Training, merge and selection reports for C0v2 and C2 |

Sample sizes, metrics and intervals are explained in [How we did it](../docs/METHODS.md). Dataset terms are in [data sources](../data/SOURCES.md).
