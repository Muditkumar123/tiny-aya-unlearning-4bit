# Data and model sources

Each source keeps its own licence, as stated on its dataset or model card. This repository adds no licence to anyone else's material.

| Source | Revision | Licence (from its card) | What we use | Citation |
| --- | --- | --- | --- | --- |
| `walledai/HarmBench` (standard) | fb6c2afd | MIT | IDs `hb_std_<row>` | Mazeika et al. 2024, arXiv:2402.04249 |
| `CohereLabs/aya_redteaming` | 5a16fee0 | Apache-2.0 | IDs `aya_<lang>_<line>` | Aakanksha et al. 2024, arXiv:2406.18682 |
| `Paul/XSTest` | f600c994 | CC-BY-4.0 | IDs and prompt type | Röttger et al. 2024, [NAACL](https://aclanthology.org/2024.naacl-long.301/), arXiv:2308.01263 |
| `locuslab/TOFU` | 324592d8 | MIT | IDs and author index | Maini et al. 2024, arXiv:2401.06121 |
| `CohereLabs/Global-MMLU` | 0e619dbe | Apache-2.0 | IDs, subject, gold letter | Singh et al. 2024, arXiv:2412.03304 |
| `tatsu-lab/alpaca` | dce01c9b | CC-BY-NC-4.0 | Row IDs and hashes only | per its card |
| `allenai/wildguard` (judge) | — | Apache-2.0; gated, AI2 research-use terms | The labels it produced | Han et al. 2024, arXiv:2406.18495 |
| `CohereLabs/tiny-aya-global` and our derived models | af89d219 | CC-BY-NC-4.0 + Cohere acceptable-use policy | The models we measure | per its card |

Our scores describe outputs of Tiny Aya and its derivatives, so Tiny Aya's non-commercial licence and acceptable-use policy apply to any use of those models. Our own numeric results and tables are offered under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), to the extent we can grant those rights. That does not cover third-party datasets, toolkits or model weights, and it gives no permission for commercial use of the models.

We do not redistribute prompt text. Rebuild it with `evaluation/build_prompt_sets.py` and `evaluation/build_tofu_sets.py`; the output goes to `data/prompts/`, which Git ignores. The IDs and hashes of the circuit-discovery prompt pairs are in `discovery_ids.json`.
