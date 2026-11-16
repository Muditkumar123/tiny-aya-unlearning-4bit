"""Run the paper-based MANSU candidate on a pinned local HF checkpoint.

Implementation only: real runs need separately frozen forget-circuit discovery,
training and validation inputs. This command never downloads or uploads models.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

import torch
import yaml

from mansu import MansuConfig, MansuTrainer, _device, _forward, rank_mlp_layers


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("checkpoint must be an existing local snapshot directory")
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError("local checkpoint has no safetensors weights")
    return {p.name: file_sha256(p) for p in files}


def read_rows(path):
    data = json.loads(Path(path).read_text())
    rows = data["rows"]
    if (not rows or any(not isinstance(r.get("id"), str) or not r["id"].strip() for r in rows)
            or len({r["id"] for r in rows}) != len(rows)):
        raise ValueError("input rows need unique, nonempty IDs")
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("answer"), str) or not row["answer"]:
            raise ValueError("each row needs an id, prompt and answer")
    return rows


def encode_examples(rows, tokenizer, maximum):
    """Use the same assistant-span format as the existing TOFU training code."""
    examples = []
    for row in rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        complete = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["answer"]}],
                                                 tokenize=False)
        prefix = tokenizer(prompt, add_special_tokens=False).input_ids
        ids = tokenizer(complete, add_special_tokens=False).input_ids
        if ids[:len(prefix)] != prefix or len(ids) > maximum or len(ids) <= len(prefix):
            raise ValueError(f"invalid prefix, empty answer or truncation for {row['id']}")
        tokens = torch.tensor([ids], dtype=torch.long)
        labels = tokens.clone()
        labels[:, :len(prefix)] = -100
        examples.append({"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "labels": labels})
    return examples


def validate_split_ids(forget, retain, validation, discovery):
    groups = {"forget_train": {r["id"] for r in forget}, "retain_train": {r["id"] for r in retain}}
    for name in ("forget", "utility"):
        rows = validation[name]
        if (not rows or any(not isinstance(r.get("id"), str) or not r["id"].strip() for r in rows)
                or len({r["id"] for r in rows}) != len(rows)):
            raise ValueError("validation groups need unique nonempty IDs")
        if any(not isinstance(r.get("answer"), str) or len(r["answer"]) != 1 or r["answer"] not in "ABCD" for r in rows):
            raise ValueError("validation answers must be single A/B/C/D letters with options rendered in prompt")
        groups[name + "_validation"] = {r["id"] for r in rows}
    groups["discovery"] = set(discovery["sample_ids"])
    # Discovery can use forget TRAINING rows; validation must stay out of both.
    for name in ("forget_validation", "utility_validation"):
        for other, ids in groups.items():
            if other != name and groups[name] & ids:
                raise ValueError(f"overlapping IDs between {name} and {other}")
    if groups["forget_train"] & groups["retain_train"]:
        raise ValueError("forget and retain training IDs overlap")


def mcq_accuracy(model, rows, tokenizer):
    """Mean letter log-prob scoring, matching the project's MMLU convention."""
    examples = []
    for row in rows:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                 add_generation_prompt=True, tokenize=False)
        prefix = tokenizer(rendered, add_special_tokens=False).input_ids
        for letter in "ABCD":
            target = tokenizer(letter, add_special_tokens=False).input_ids
            if not target or len(prefix) + len(target) > 4096:
                raise ValueError("empty/overlong MCQ target")
            tokens = torch.tensor([prefix + target], dtype=torch.long)
            labels = torch.full_like(tokens, -100)
            labels[:, len(prefix):] = tokens[:, len(prefix):]
            examples.append({"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "labels": labels})
    from mansu import completion_nll
    with torch.no_grad():
        scores = [-float(completion_nll(_forward(model, x), x["labels"].to(_device(model)))) for x in examples]
    predictions = ["ABCD"[max(range(4), key=lambda j: scores[i * 4 + j])] for i in range(len(rows))]
    return sum(pred == row["answer"] for pred, row in zip(predictions, rows)) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/mansu.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--circuit", type=Path, required=True, help="forget attribution JSON from discover_mansu.py")
    parser.add_argument("--forget", type=Path, required=True)
    parser.add_argument("--retain", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True, help="disjoint MCQ forget/utility groups")
    parser.add_argument("--out", type=Path, required=True, help="new run directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--check-inputs", action="store_true", help="hash/validate inputs; do not load a model")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be new; existing evidence is never overwritten")
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    cfg = MansuConfig(**config["training"])
    for key in ("circuit_size", "fisher_examples", "max_length"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    forget, retain = read_rows(args.forget), read_rows(args.retain)
    validation = json.loads(args.validation.read_text())
    discovery = json.loads(args.circuit.read_text())
    if discovery.get("purpose") != "forget_knowledge" or discovery.get("algorithm") != "EAP-IG-inputs":
        raise ValueError("MANSU needs a forget-knowledge EAP-IG circuit, not the old refusal mask")
    weights = checkpoint_identity(args.checkpoint)
    tokenizer_hashes = {p.name: file_sha256(p) for p in args.checkpoint.iterdir()
                        if p.name in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}}
    if not tokenizer_hashes or discovery.get("tokenizer_sha256") != tokenizer_hashes:
        raise ValueError("attribution tokenizer/template differs from the training checkpoint")
    if discovery["model_weights_sha256"] != weights:
        raise ValueError("attribution checkpoint differs from the training checkpoint")
    layers = rank_mlp_layers(discovery["edges_abs_mean"], discovery["edge_destination_layer"], config["circuit_size"])
    validate_split_ids(forget, retain, validation, discovery)
    inputs = {name: {"sha256": file_sha256(path)} for name, path in {
        "config": args.config, "circuit": args.circuit, "forget": args.forget,
        "retain": args.retain, "validation": args.validation}.items()}
    manifest = {"method": "paper-based MANSU adaptation", "paper": "https://arxiv.org/abs/2605.15138v1",
                "status": "inputs_checked", "input_files": inputs, "weights_sha256": weights,
                "selected_mlp_layers": layers, "config": config, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "seed": config["seed"], "code_sha256": {p.name: file_sha256(p) for p in (Path(__file__), Path(__file__).with_name("mansu.py"))}, "versions": {k: version(k) for k in ("torch", "transformers", "numpy")},
                "python": platform.python_version(), "quantization_survival_verified": False}
    if args.check_inputs:
        print(json.dumps(manifest, indent=2))
        return
    # Import/model loading only after the complete input validation above.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if tokenizer.convert_ids_to_tokens([6, 3]) != ["<|END_OF_TURN_TOKEN|>", "<EOS_TOKEN>"]:
        raise ValueError("the candidate recipe requires verified Tiny Aya stop-token identities")
    forget_batches = encode_examples(forget, tokenizer, config["max_length"])
    retain_batches = encode_examples(retain, tokenizer, config["max_length"])
    if len(retain_batches) < config["fisher_examples"]:
        raise ValueError("fewer retain examples than the frozen Fisher count")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16,
                                                local_files_only=True).to(args.device)
    reference = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16,
                                                    local_files_only=True).to(args.reference_device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainer = MansuTrainer(model, reference, layers, cfg)
    args.out.mkdir(parents=True)
    report_path = args.out / "run.json"

    def save():
        report_path.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest["rendered_lengths"] = {name: {"min": min(x["input_ids"].shape[1] for x in rows), "max": max(x["input_ids"].shape[1] for x in rows), "count": len(rows)} for name, rows in (("forget", forget_batches), ("retain", retain_batches))}
    save()
    try:
        manifest["phase"] = "fisher"
        save()
        manifest["fisher"] = trainer.prepare_fisher(retain_batches[:config["fisher_examples"]])
        manifest["phase"] = "training"
        manifest["validation_calls"] = []
        save()
        def evaluate(current):
            metrics = {"forget_accuracy": mcq_accuracy(current, validation["forget"], tokenizer),
                       "utility_accuracy": mcq_accuracy(current, validation["utility"], tokenizer)}
            manifest["validation_calls"].append({"phase": manifest["phase"], **metrics})
            save()
            print(json.dumps({"validation": metrics, "phase": manifest["phase"]}), flush=True)
            return metrics
        manifest["training"] = trainer.train(forget_batches, retain_batches, evaluate=evaluate)
        manifest["phase"] = "floor"
        manifest["floor"] = trainer.apply_floor()
        manifest["post_floor_validation"] = evaluate(model)
        baseline = manifest["training"]["baseline"]["utility_accuracy"]
        if baseline - manifest["post_floor_validation"]["utility_accuracy"] > cfg.max_utility_drop + 1e-12:
            raise ValueError("post-floor checkpoint violates the utility bound; not exporting")
        # bf16 export is a real numerical transformation. Reject any erased
        # nonzero learned update rather than silently shipping a different edit.
        before_cast = {name: p.detach().float().cpu().clone() for name, p in trainer.parameters.items()}
        model.to(torch.bfloat16)
        rounding = {}
        for name, parameter in trainer.parameters.items():
            original = trainer.original[name]
            before = before_cast[name] - original
            after = parameter.detach().float().cpu() - original
            lost = int(((before != 0) & (after == 0)).sum())
            rounding[name] = {"nonzero_before": int((before != 0).sum()), "nonzero_after": int((after != 0).sum()),
                              "updates_lost_to_bf16": lost}
        manifest["export_rounding"] = rounding
        if any(x["updates_lost_to_bf16"] for x in rounding.values()):
            raise ValueError("bf16 export erased learned updates; not exporting")
        manifest["phase"] = "export"
        manifest["post_export_validation"] = evaluate(model)
        if baseline - manifest["post_export_validation"]["utility_accuracy"] > cfg.max_utility_drop + 1e-12:
            raise ValueError("bf16 export violates the utility bound; not exporting")
        model.save_pretrained(args.out / "checkpoint", safe_serialization=True)
        tokenizer.save_pretrained(args.out / "checkpoint")
        # Keep the exact pinned tokenizer bytes for comparable before/after attribution.
        for name in tokenizer_hashes:
            shutil.copyfile(args.checkpoint / name, args.out / "checkpoint" / name)
        manifest["output_weights_sha256"] = checkpoint_identity(args.out / "checkpoint")
        manifest["export_precision"] = "bfloat16; rounding and validation checked after conversion"
        manifest["status"] = "trained_and_exported_not_quantization_validated"
        save()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_type"] = type(exc).__name__
        save()
        raise
    print(f"Saved {args.out}; no quantization, publication or structural-erasure claim made.")


if __name__ == "__main__":
    main()
