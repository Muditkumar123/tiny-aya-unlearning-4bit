# Hugging Face payloads

This folder maps the seven published model repositories and the evaluation dataset to local files. `manifest.json` lists every uploaded file with its size. The [collections](../artifacts.json) link them all.

| Local directory | Hugging Face repository | Contents |
| --- | --- | --- |
| `models/c0v2/` | `Mudit666/tiny-aya-tofu-c0v2` | TOFU fine-tune: merged bf16 model and LoRA adapter |
| `models/c2/` | `Mudit666/tiny-aya-tofu-npo-c2` | NPO-unlearned model: merged bf16 model and adapter |
| `models/c1/` | `Mudit666/tiny-aya-tofu-c0v2-qint4` | C0v2 in qint4 |
| `models/c3/` | `Mudit666/tiny-aya-tofu-npo-c2-qint4` | C2 in qint4 |
| `models/plain-q4/` | `Mudit666/tiny-aya-global-qint4` | Tiny Aya in plain qint4 |
| `models/random30-q4/` | `Mudit666/tiny-aya-global-qint4-random77-bf16` | qint4 with 77 random layers in bf16 |
| `models/protected-q4/` | `Mudit666/tiny-aya-global-qint4-refusalmask77-bf16` | qint4 with the 77 refusal-circuit layers in bf16 |
| `datasets/evaluation/` | `Mudit666/tiny-aya-compression-evaluation` | Per-question numeric results, no prompt or answer text |

`models/`, `datasets/` and `local_sources/` are local links that Git ignores. On another machine, download the models from the repositories above.

## Check or prepare an upload

```bash
python HF/prepare_upload.py --check                       # verify that every file is present with the listed size
python HF/prepare_upload.py --payload c3 --materialize /path/to/new-dir   # copy one payload as real files
```

The helper never uploads. Upload a materialized folder with the Hugging Face web interface or `HfApi.upload_folder`, keeping the model cards, licence and acceptable-use notices.
