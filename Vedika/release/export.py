"""Build the public, numbers-and-ids-only export of the reviewed Study U and Study S results (CPU, local only).

    HF_HUB_OFFLINE=1 python Vedika/release/export.py      # writes Vedika/release/export/ (refuses to overwrite)

Kept: ids, author/group/type/category labels, scores, judge labels (0/1), log-probs, KL, an estimated answer
length, summaries, manifests and the reviewed tables. Dropped: every prompt, gold answer, model answer and raw
judge text. Local paths are shortened to "~/". Lab files in Vedika/out/ are only read, never changed.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import ORIG, ROOT, SETS, read_jsonl, write_jsonl  # noqa: E402

OUT = ROOT / "Vedika/release/export"
U, S = ROOT / "Vedika/out/r4", ROOT / "Vedika/out/r5"
KEEP_TOFU = ("id", "author", "split", "lp_answer", "lp_para", "lp_pert", "rougeL_recall", "truth_ratio")
KEEP_MMLU = ("id", "subject", "answer", "lp", "pred", "correct")  # answer = gold letter A-D
KEEP_SAFETY = ("id", "group", "category", "type", "harmful_request", "refusal", "harmful_response", "first_token_kl", "mean_kl")
TEXT = ("prompt", "response", "judge_raw", "literal_translation", "focus", "paraphrased_answer", "perturbed_answer")
HOME = str(Path.home()) + "/"


def scrub(obj):
    return json.loads(json.dumps(obj).replace(HOME, "~/"))


def keep(rows, fields):
    return [{k: r[k] for k in fields if k in r} for r in rows]


def copy_json(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(scrub(json.loads(src.read_text())), ensure_ascii=False, indent=1))


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ORIG)
    OUT.mkdir(parents=True, exist_ok=False)
    for k in ("original", "c0v2", "c1", "c2", "c3"):  # Study U rows (reviewed R2/R4/R4a inputs)
        d = OUT / "study_u" / k
        for s in ("forget", "retain"):
            write_jsonl(d / f"{s}.jsonl", keep(read_jsonl(U / k / f"{s}.jsonl"), KEEP_TOFU))
        for s in ("mmlu_en", "mmlu_hi"):
            write_jsonl(d / f"{s}.jsonl", keep(read_jsonl(U / k / f"{s}.jsonl"), KEEP_MMLU))
        for f in ("manifest.json", "summary.json"):
            copy_json(U / k / f, d / f)
    for k in ("original", "plain-q4", "random30-q4", "protected-q4"):  # Study S rows (reviewed R5 / R5u)
        d = OUT / "study_s" / "eval" / k
        for s in SETS:
            rows = read_jsonl(S / "eval" / k / f"{s}.jsonl")
            out = keep(rows, KEEP_SAFETY)
            for o, r in zip(out, rows):  # what r5_tables needs for its cap estimate, without the text
                o["n_tokens_estimate"] = len(tok(r["response"], add_special_tokens=False).input_ids)
            write_jsonl(d / f"{s}.jsonl", out)
        for f in ("manifest.json", "summary.json"):
            copy_json(S / "eval" / k / f, d / f)
        m = OUT / "study_s" / "mmlu" / k
        for s in ("mmlu_en", "mmlu_hi"):
            write_jsonl(m / f"{s}.jsonl", keep(read_jsonl(S / "mmlu" / k / f"{s}.jsonl"), KEEP_MMLU))
        for f in ("manifest.json", "summary.json"):
            copy_json(S / "mmlu" / k / f, m / f)
    t = OUT / "tables"  # the reviewed tables, as reviewed
    for src, name in ((U / "contrasts.json", "study_u_contrasts.json"), (U / "table.json", "study_u_levels.json"),
                      (S / "tables.json", "study_s_tables.json")):
        copy_json(src, t / name)
    for src, name in ((U / "contrasts.md", "study_u_contrasts.md"), (U / "results.md", "study_u_levels.md"),
                      (S / "tables.md", "study_s_tables.md"), (S / "eval/results.md", "study_s_vs_original.md")):
        shutil.copyfile(src, t / name)
    bad = [str(p) for p in OUT.rglob("*.json*") if any(f'"{x}"' in p.read_text() for x in TEXT)]
    assert not bad, f"text fields leaked into {bad}"
    print("export written:", sum(1 for _ in OUT.rglob("*") if _.is_file()), "files")


if __name__ == "__main__":
    main()
