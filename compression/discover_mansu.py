"""Compute forget-knowledge EAP-IG edge scores with CircuitKIT (no refusal mask reuse)."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from run_mansu import checkpoint_identity, file_sha256
from mansu import rank_mlp_layers


def aggregate_edges(examples):
    """Keep signed means for CAD and mean absolute scores for circuit ranking."""
    if not examples:
        raise ValueError("no per-example edge attributions")
    keys = set(examples[0])
    if not keys or any(set(x) != keys for x in examples):
        raise ValueError("per-example edge identities differ")
    import math
    if any(not math.isfinite(v) for row in examples for v in row.values()):
        raise ValueError("nonfinite edge score")
    n = len(examples)
    return ({k: sum(x[k] for x in examples) / n for k in sorted(keys)},
            {k: sum(abs(x[k]) for x in examples) / n for k in sorted(keys)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True, help="frozen rendered clean/corrupt forget pairs")
    parser.add_argument("--out", type=Path, required=True, help="new directory; contains attribution.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=50)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() or args.examples < 1 or args.ig_steps < 1:
        parser.error("choose a new output directory and positive example/IG counts")
    data = json.loads(args.pairs.read_text())
    if data.get("purpose") != "forget_knowledge":
        raise ValueError("pairs must be prepared for forget-knowledge attribution")
    pairs = data["pairs"]
    if (len(pairs) < args.examples or any(not isinstance(p.get("id"), str) or not p["id"].strip() for p in pairs)
            or len({p["id"] for p in pairs}) != len(pairs)):
        raise ValueError("not enough discovery pairs or duplicate pair IDs")
    selected = pairs[:args.examples]
    weights = checkpoint_identity(args.checkpoint)
    # Heavy loading only on an explicit discovery invocation.
    import torch
    torch.manual_seed(args.seed)
    import circuitkit as ck
    from circuitkit.backends.eap.attribute import attribute
    from circuitkit.backends.eap.graph import Graph, MLPNode
    from circuitkit.data.normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    model = ck.load_model(str(args.checkpoint), dtype="bfloat16", device=args.device,
                          algorithm="eap-ig", local_files_only=True)
    model.eval()
    if len({str(p.device) for p in model.parameters()}) != 1:
        raise ValueError("EAP-IG discovery requires one model device")
    args.out.mkdir(parents=True)
    per_example, destinations = [], None
    for i, pair in enumerate(selected):
        for field in ("clean_prompt", "clean_answer", "corrupt_prompt", "corrupt_answer"):
            if not isinstance(pair.get(field), str) or not pair[field]:
                raise ValueError(f"empty/missing {field} in a discovery pair")
        token_ids = [model.tokenizer(x, add_special_tokens=False).input_ids
                     for x in (pair["clean_prompt"], pair["corrupt_prompt"])]
        answer_ids = [model.tokenizer(x, add_special_tokens=False).input_ids
                      for x in (pair["clean_answer"], pair["corrupt_answer"])]
        if len(token_ids[0]) != len(token_ids[1]) or token_ids[0] == token_ids[1]:
            raise ValueError("discovery pairs must have distinct, equal-length rendered prompts")
        if any(len(ids) != 1 for ids in answer_ids) or answer_ids[0] == answer_ids[1]:
            raise ValueError("discovery targets must be distinct single tokens")
        record = ContrastiveRecord(record_id=pair["id"], clean_prompt=pair["clean_prompt"],
                                   clean_answer=pair["clean_answer"], corrupt_prompt=pair["corrupt_prompt"],
                                   corrupt_answer=pair["corrupt_answer"], contrast_source=ContrastSource.EXTERNAL)
        dataset = NormalizedDataset(name="mansu_forget", shape=DatasetShape.FORGET_RETAIN,
                                    records=[record], source="frozen paired forget data")
        task = NormalizedTaskSpec(dataset, name="mansu_forget", cache_dir=str(args.out / "cache" / str(i)))
        settings = {"algorithm": "eap-ig", "level": "node", "n_examples": 1, "batch_size": 1, "data_params": {"seed": args.seed}, "seed": args.seed}
        loader = task.build_dataloader(model, settings, args.device)
        graph = Graph.from_model(model, node_scores=True, neuron_level=False, mlp_hook="mlp_out")
        attribute(model, graph, loader, task.metric_fn(), method="EAP-IG-inputs", ig_steps=args.ig_steps, quiet=True)
        incoming = {name: edge for name, edge in graph.edges.items() if isinstance(edge.child, MLPNode)}
        current_destinations = {name: edge.child.layer for name, edge in incoming.items()}
        if destinations is not None and destinations != current_destinations:
            raise ValueError("edge identities changed across examples")
        destinations = current_destinations
        per_example.append({name: float(edge.score.detach().cpu()) for name, edge in incoming.items()})
    signed, absolute = aggregate_edges(per_example)
    layers = rank_mlp_layers(absolute, destinations, args.top_k)
    tokenizers = {p.name: file_sha256(p) for p in args.checkpoint.iterdir()
                  if p.name in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}}
    output = {"purpose": "forget_knowledge", "algorithm": "EAP-IG-inputs", "ig_steps": args.ig_steps,
              "metric": "single-token clean-minus-corrupt logit difference (CircuitKIT loss sign)",
              "pairs_sha256": file_sha256(args.pairs), "sample_ids": [p["id"] for p in selected],
              "model_weights_sha256": weights, "tokenizer_sha256": tokenizers,
              "circuitkit_version": version("circuitkit"), "seed": args.seed, "code_sha256": file_sha256(__file__), "edge_destination_layer": destinations,
              "edges_signed_mean": signed, "edges_abs_mean": absolute, "selected_mlp_layers": layers,
              "status": "attribution_computed_not_causally_validated"}
    (args.out / "attribution.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved forget attribution for {len(selected)} pairs; selected MLP layers {layers}")


if __name__ == "__main__":
    main()
