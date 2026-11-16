"""NF4 follow-up: the paper's forget-MCQ metric under NF4, and MANSU with the paper's Table 10 settings.

    python evaluation/nf4_followup.py cpu                                         # update size vs NF4 step (CPU)
    python evaluation/nf4_followup.py train --out runs/nf4_followup/gate          # MANSU training, no floor
    python evaluation/nf4_followup.py floor --gate runs/nf4_followup/gate --out runs/nf4_followup/floor
    python evaluation/nf4_followup.py mcq --checkpoint DIR --precision bf16|nf4 --label L   # forget MCQ accuracy
    python evaluation/nf4_followup.py contrast                                    # NF4 - bf16 with intervals

Reads the prepared MANSU inputs in runs/mansu_20261115/. Adds letter-only MCQ batches of 8, the paper's
Table 10 Fisher threshold, saved per-question predictions and a random-sign floor control.
evaluation/run_nf4_followup.sh runs the NF4 build and scoring end to end.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/nf4_followup"
INPUTS = ROOT / "runs/mansu_20261115/inputs"
ATTRIBUTION = ROOT / "runs/mansu_20261115/discovery/attribution.json"
C0 = ROOT / "HF/models/c0v2"
LAYERS = [15, 14, 13, 16, 17]
BATCH, STEPS, FISHER_EXAMPLES = 8, 30, 100
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "shared")]
import mansu  # noqa: E402
from mansu import (MansuConfig, MansuTrainer, _forward, circuit_parameters, estimate_fisher,  # noqa: E402
                   magnitude_floor, rank_mlp_layers)


def completion_nll(logits, labels, reduction="mean"):
    """Same value as mansu.completion_nll; picks the scored rows before the fp32 cast (memory only).

    The original casts every position's 261,008-wide logits to fp32: 5.4 GB for one batch of 8,
    which does not fit next to the model on a 40 GB card.
    """
    active = labels[:, 1:] != -100
    assert active.any(), "no scored completion tokens"
    return torch.nn.functional.cross_entropy(logits[:, :-1][active].float(), labels[:, 1:][active], reduction=reduction)


_x, _y = torch.randn(2, 5, 7), torch.tensor([[-100, -100, 3, -100, 1], [-100, 2, -100, -100, -100]])
assert all(torch.allclose(completion_nll(_x, _y, r), mansu.completion_nll(_x, _y, r), rtol=1e-6, atol=0) for r in ("mean", "sum"))
mansu.completion_nll = completion_nll  # the trainer and Fisher look it up in mansu's namespace


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C0, local_files_only=True)
    assert tok.convert_ids_to_tokens([6, 3]) == ["<|END_OF_TURN_TOKEN|>", "<EOS_TOKEN>"]
    return tok


def load_model(path, precision, device):
    from transformers import AutoModelForCausalLM
    kw = {"dtype": torch.bfloat16, "local_files_only": True}
    if precision == "nf4":  # a saved NF4 checkpoint from code/compression/build_mansu_nf4.py
        import bitsandbytes as bnb
        model = AutoModelForCausalLM.from_pretrained(path, device_map={"": device}, **kw)
        assert sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()) == 252
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **kw).to(device)
    return model.eval()


def mcq(row, position):
    """Same rendering as training/prepare_mansu_run.py (nested there, so repeated here)."""
    wrong = row["perturbed_answer"][:3]
    assert len(set(wrong)) == 3 and row["answer"] not in wrong
    options = list(wrong)
    options.insert(position, row["answer"])
    prompt = ("Answer the following multiple-choice question with the letter of the correct option only.\n\n"
              + row["prompt"] + "\n" + "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", options)))
    return {"id": row["id"], "author": row["author"], "prompt": prompt, "answer": "ABCD"[position]}


def prefix_ids(tok, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False).input_ids


def letter_ids(tok):
    ids = [tok(letter, add_special_tokens=False).input_ids for letter in "ABCD"]
    assert all(len(x) == 1 for x in ids) and len({x[0] for x in ids}) == 4
    return [x[0] for x in ids]


@torch.no_grad()
def predict(model, tok, rows):
    """Letter with the highest next-token log-prob. Equal to run_mansu.mcq_accuracy for single-token letters."""
    letters, out = letter_ids(tok), []
    for row in rows:
        x = torch.tensor([prefix_ids(tok, row["prompt"])])
        lp = _forward(model, {"input_ids": x, "attention_mask": torch.ones_like(x)})[0, -1].float().log_softmax(-1)[letters]
        pred = "ABCD"[int(lp.argmax())]
        out.append({"id": row["id"], "author": row.get("author"), "answer": row["answer"], "pred": pred,
                    "correct": int(pred == row["answer"]), "lp": [round(float(v), 4) for v in lp]})
    return out


def accuracy(predictions):
    return sum(p["correct"] for p in predictions) / len(predictions)


def letter_example(tok, item):
    """Prompt + answer letter; only the letter is scored (no end-of-turn token in the loss)."""
    ids = torch.tensor([prefix_ids(tok, item["prompt"]) + tok(item["answer"], add_special_tokens=False).input_ids])
    labels = torch.full_like(ids, -100)
    labels[0, -1] = ids[0, -1]
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def collate(examples, pad):
    """Right padding: a causal model's outputs at real positions do not depend on it."""
    n = max(e["input_ids"].shape[1] for e in examples)
    def widen(t, value):
        return torch.cat([t, torch.full((1, n - t.shape[1]), value, dtype=t.dtype)], 1)
    return {"input_ids": torch.cat([widen(e["input_ids"], pad) for e in examples]),
            "attention_mask": torch.cat([widen(e["attention_mask"], 0) for e in examples]),
            "labels": torch.cat([widen(e["labels"], -100) for e in examples])}


class Table10Trainer(MansuTrainer):
    """Fisher mask at 0.1 x mean (paper Table 10) instead of the 99th percentile (Algorithm 1)."""
    def prepare_fisher(self, retain_examples):
        fisher = estimate_fisher(self.model, self.parameters, retain_examples)
        flat = np.concatenate([v.numpy().reshape(-1) for v in fisher.values()])
        assert np.isfinite(flat).all()
        mean = float(flat.mean(dtype=np.float64))
        threshold = 0.1 * mean
        self.masks = {n: (v <= threshold).to(self.parameters[n].device) for n, v in fisher.items()}
        q50, q90, q99 = (float(x) for x in np.quantile(flat, [.5, .9, .99]))
        self.fisher_report = {"examples": len(retain_examples), "mean": mean, "threshold_0.1_mean": threshold,
                              "median": q50, "p90": q90, "p99": q99, "circuit_coordinates": int(flat.size),
                              "allowed_share_table10": float((flat <= threshold).mean()),
                              "allowed_share_algorithm1_p99": float((flat <= q99).mean())}
        return self.fisher_report


def rows(name):
    return json.loads((INPUTS / name).read_text())["rows"]


def cmd_cpu(args):
    """Step 2, first half: sizes only. Reads the C0v2 weight file on CPU."""
    from safetensors import safe_open
    out, lr_reach = [], MansuConfig().learning_rate * STEPS
    with safe_open(str(C0 / "model.safetensors"), "pt") as f:
        for layer in sorted(LAYERS):
            for part in ("gate_proj", "up_proj", "down_proj"):
                w = f.get_tensor(f"model.layers.{layer}.mlp.{part}.weight").float()
                a = w.abs()
                typical = a.mean().item()
                blocks = w.flatten()[: w.numel() // 64 * 64].reshape(-1, 64).abs().amax(1)
                out.append({"matrix": f"{layer}.{part}", "typical_abs_weight": typical, "std": w.std().item(),
                            "tensor_range_floor_over_typical": (a.max() - a.min()).item() / 16 / typical,
                            "nf4_block_floor_median_over_typical": (blocks * 0.0796 * 0.704).median().item() / typical,
                            "max_training_move_over_typical": lr_reach / typical})
    summary = {k: [min(r[k] for r in out), max(r[k] for r in out)] for k in out[0] if k != "matrix"}
    report = {"paper_reference": {"floor_over_typical": 0.093, "share_of_circuit_changed": 0.29},
              "lr_times_steps": lr_reach, "range_over_15_matrices": summary, "matrices": out}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cpu_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("paper_reference", "lr_times_steps", "range_over_15_matrices")}, indent=2))


def cmd_train(args):
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    out = args.out
    out.mkdir(parents=True)
    tok = tokenizer()
    steps = 1 if args.smoke else STEPS
    forget = [letter_example(tok, mcq(r, i % 4)) for i, r in enumerate(rows("forget.json")[: BATCH * STEPS])]
    retain = [letter_example(tok, mcq(r, i % 4)) for i, r in enumerate(rows("retain.json")[: BATCH * STEPS])]
    validation = json.loads((INPUTS / "validation.json").read_text())
    if args.smoke:  # the 8 longest rows of each: worst-case memory
        forget, retain = (sorted(ex, key=lambda e: -e["input_ids"].shape[1])[:BATCH] for ex in (forget, retain))
        validation = {k: v[:4] for k, v in validation.items()}
    attribution = json.loads(ATTRIBUTION.read_text())
    layers = rank_mlp_layers(attribution["edges_abs_mean"], attribution["edge_destination_layer"], 5)
    assert layers == LAYERS, layers
    pad = tok.pad_token_id
    batches = lambda ex: [collate(ex[i:i + BATCH], pad) for i in range(0, BATCH * steps, BATCH)]  # noqa: E731
    torch.manual_seed(0)
    load = lambda: AutoModelForCausalLM.from_pretrained(C0, dtype=torch.bfloat16, local_files_only=True).to(args.device)  # noqa: E731
    model, reference = load(), load()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = MansuConfig(max_steps=steps, evaluate_every=1 if args.smoke else 5)
    trainer = Table10Trainer(model, reference, layers, cfg)
    report = {"status": "running", "layers": layers, "batch": BATCH, "smoke": args.smoke,
              "rendered_lengths": {n: [min(e["input_ids"].shape[1] for e in ex), max(e["input_ids"].shape[1] for e in ex)]
                                   for n, ex in (("forget", forget), ("retain", retain))}}
    save = lambda: (out / "gate.json").write_text(json.dumps(report, indent=2) + "\n")  # noqa: E731
    report["fisher"] = trainer.prepare_fisher(retain[: 2 if args.smoke else FISHER_EXAMPLES])
    print(json.dumps({"fisher": report["fisher"]}), flush=True)
    calls = []

    def evaluate(current):
        predictions = {name: predict(current, tok, validation[name]) for name in ("forget", "utility")}
        metrics = {"forget_accuracy": accuracy(predictions["forget"]), "utility_accuracy": accuracy(predictions["utility"])}
        calls.append({"metrics": metrics, "predictions": predictions})
        (out / "validation_calls.json").write_text(json.dumps(calls) + "\n")
        print(json.dumps({"validation": metrics}), flush=True)
        if len(calls) == 1 and not args.smoke and metrics != {"forget_accuracy": 31 / 40, "utility_accuracy": 0.47}:
            raise SystemExit("plumbing check failed: baseline differs from the recorded 31/40 and 47/100")
        return metrics

    save()
    try:
        report["training"] = trainer.train(batches(forget), batches(retain), evaluate=evaluate)
    except ValueError as error:  # e.g. no checkpoint inside the utility bound
        report.update(status="failed", failure=str(error), validation=[c["metrics"] for c in calls])
        save()
        raise
    t = report["training"]
    drop = t["baseline"]["forget_accuracy"] - t["selected_metrics"]["forget_accuracy"]
    report.update(status="complete", forget_drop_points=round(100 * drop, 2), gate_passed=bool(drop >= 0.10 - 1e-12),
                  peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
    tensors = {n: p.detach().float().cpu() for n, p in trainer.parameters.items()}
    tensors |= {"mask." + n: m.cpu().to(torch.uint8) for n, m in trainer.masks.items()}
    save_file(tensors, str(out / "prefloor.safetensors"))
    save()
    print(json.dumps({k: report[k] for k in ("status", "forget_drop_points", "gate_passed", "peak_gpu_gb")}
                     | {"selected_step": t["selected_step"], "selected": t["selected_metrics"], "baseline": t["baseline"]}), flush=True)


def cmd_floor(args):
    from safetensors.torch import load_file
    out = args.out
    out.mkdir(parents=True)
    gate = json.loads((args.gate / "gate.json").read_text())
    baseline = gate["training"]["baseline"]
    saved = load_file(str(args.gate / "prefloor.safetensors"))
    tok = tokenizer()
    validation = json.loads((INPUTS / "validation.json").read_text())
    model = load_model(C0, "bf16", args.device)
    params = circuit_parameters(model, LAYERS)
    original = {n: p.detach().float().cpu().clone() for n, p in params.items()}
    for p in params.values():
        p.data = p.data.float()
    # Step 2, second half: what the floor will do, written before it is applied.
    floored, moved, total, relative, sub_floor = {}, 0, 0, [], 0
    for n in params:
        mask = saved["mask." + n].bool()
        floored[n], info = magnitude_floor(original[n], saved[n], policy="rescale", allowed=mask, scale="nf4_block", margin=0.704)
        delta = floored[n] - original[n]
        moved += int((delta != 0).sum()); total += delta.numel(); sub_floor += info["adjusted_elements"]
        relative.append(float(delta[delta != 0].abs().median() / original[n].abs().mean()))
    precheck = {"share_of_circuit_weights_moved": moved / total, "moved": moved, "of_which_were_below_floor": sub_floor,
                "median_move_over_typical_weight": [min(relative), max(relative)],
                "largest_prefloor_move": max(float((saved[n] - original[n]).abs().max()) for n in params),
                "paper_reference": {"share": 0.29, "move_over_typical": 0.093}}
    report = {"precheck": precheck, "baseline": baseline, "gate_passed": gate["gate_passed"], "variants": {}}
    (out / "floor.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"precheck": precheck}), flush=True)
    generator = torch.Generator().manual_seed(0)
    for variant in ("gradient_signs", "random_signs"):
        with torch.no_grad():
            for n, p in params.items():
                delta = floored[n] - original[n]
                if variant == "random_signs":  # same coordinates, same sizes, random direction
                    sign = torch.where(torch.rand(delta.shape, generator=generator) < 0.5, -1.0, 1.0)
                    delta = delta.abs() * sign
                p.copy_((original[n] + delta).to(p.device))
        predictions = {name: predict(model, tok, validation[name]) for name in ("forget", "utility")}
        metrics = {"forget_accuracy": accuracy(predictions["forget"]), "utility_accuracy": accuracy(predictions["utility"])}
        drop = baseline["utility_accuracy"] - metrics["utility_accuracy"]
        report["variants"][variant] = metrics | {"utility_drop_points": round(100 * drop, 2), "within_utility_bound": bool(drop <= 0.08 + 1e-12)}
        (out / f"predictions_{variant}.json").write_text(json.dumps(predictions) + "\n")
        (out / "floor.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({variant: report["variants"][variant]}), flush=True)
    if report["variants"]["gradient_signs"]["within_utility_bound"]:
        import shutil
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(floored[n].to(p.device))
        model.to(torch.bfloat16)
        lost = sum(int(((floored[n] != original[n]) & (p.detach().float().cpu() == original[n])).sum()) for n, p in params.items())
        report["export"] = {"updates_lost_to_bf16": lost}
        if lost == 0:
            model.save_pretrained(out / "checkpoint", safe_serialization=True)
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                shutil.copyfile(C0 / name, out / "checkpoint" / name)
            report["export"]["checkpoint"] = str(out / "checkpoint")
        (out / "floor.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"export": report["export"]}), flush=True)


def cmd_mcq(args):
    """Step 1 extra: the paper's metric, forget MCQ accuracy, on all 400 forget rows."""
    out = OUT / "mcq"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{args.label}.json"
    assert not target.exists(), "use a fresh label"
    tok = tokenizer()
    final = [mcq(r, i % 4) | {"set": "final360"} for i, r in enumerate(json.loads((INPUTS / "evaluation.json").read_text())["forget"])]
    # validation rows carry no author; the source split (same hash as in inputs_receipt.json) has it
    author = {r["id"]: r["author"] for r in json.loads((ROOT / "data/prompts/tofu_author_split.json").read_text())["rows"]}
    held = [v | {"set": "validation40", "author": author[v["id"]]}
            for v in json.loads((INPUTS / "validation.json").read_text())["forget"]]
    model = load_model(args.checkpoint, args.precision, args.device)
    items = final + held
    predictions = predict(model, tok, items)
    for p, item in zip(predictions, items):
        p["set"], p["author"] = item["set"], item["author"]
    target.write_text(json.dumps(predictions) + "\n")
    print(json.dumps({"label": args.label, "forget_mcq_accuracy_400": accuracy(predictions),
                      "final360": accuracy([p for p in predictions if p["set"] == "final360"]),
                      "validation40": accuracy([p for p in predictions if p["set"] == "validation40"])}), flush=True)


def cmd_contrast(args):
    """Step 1 endpoint: NF4 - bf16 per model with paired 95% intervals."""
    from cluster_ci import cluster_ci
    from paired_ci import paired_ci
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines()]  # noqa: E731
    report = {}
    for method in args.methods:
        cell = {}
        mcqs = {p: json.loads((OUT / f"mcq/{method}_{p}.json").read_text()) for p in ("bf16", "nf4")}
        values = {p: {r["id"]: r["correct"] for r in mcqs[p]} for p in mcqs}
        author = {r["id"]: r["author"] for r in mcqs["bf16"]}
        cell["forget_mcq_accuracy"] = {"bf16": accuracy(mcqs["bf16"]), "nf4": accuracy(mcqs["nf4"]),
                                       "nf4_minus_bf16": cluster_ci({"a": values["bf16"], "b": values["nf4"]}, author, lambda m: m["b"] - m["a"])}
        for subset in ("forget", "retain", "mmlu_en", "mmlu_hi"):
            a, b = ({r["id"]: r for r in read(OUT / f"evaluation/{method}_{p}/{subset}.jsonl")} for p in ("bf16", "nf4"))
            assert a.keys() == b.keys()
            for metric in (("rougeL_recall", "lp_answer") if subset in ("forget", "retain") else ("correct",)):
                va, vb = {k: r[metric] for k, r in a.items()}, {k: r[metric] for k, r in b.items()}
                if subset in ("forget", "retain"):
                    ci = cluster_ci({"a": va, "b": vb}, {k: r["author"] for k, r in a.items()}, lambda m: m["b"] - m["a"])
                    ci = {"delta": ci["point"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": ci["n"]}
                else:
                    ci = paired_ci(va, vb)
                cell[f"{subset}.{metric}"] = {"bf16": float(np.mean(list(va.values()))), "nf4": float(np.mean(list(vb.values()))), "nf4_minus_bf16": ci}
        report[method] = cell
    (OUT / "nf4_contrasts.json").write_text(json.dumps(report, indent=2) + "\n")
    for method, cell in report.items():
        for name, c in cell.items():
            d = c["nf4_minus_bf16"]
            point = d.get("delta", d.get("point"))
            print(f"{method:6s} {name:24s} bf16 {c['bf16']:+.3f}  nf4 {c['nf4']:+.3f}  delta {point:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cpu")
    p = sub.add_parser("train"); p.add_argument("--out", type=Path, required=True); p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("floor"); p.add_argument("--gate", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("mcq"); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--precision", choices=["bf16", "nf4"], required=True); p.add_argument("--label", required=True)
    p = sub.add_parser("contrast"); p.add_argument("--methods", nargs="+", default=["c0", "npo"])
    for q in sub.choices.values():
        q.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    {"cpu": cmd_cpu, "train": cmd_train, "floor": cmd_floor, "mcq": cmd_mcq, "contrast": cmd_contrast}[args.command](args)
