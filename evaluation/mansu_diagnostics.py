"""MANSU diagnostics; none of these scalar checks alone proves factual erasure."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import torch


def cad(before, after, circuit_layers):
    """Eq. 6 on the SAME original incoming edges and frozen discovery inputs."""
    for key in ("purpose", "algorithm", "ig_steps", "metric", "pairs_sha256", "sample_ids", "tokenizer_sha256", "seed"):
        if before.get(key) != after.get(key) or key not in before:
            raise ValueError(f"incomparable attribution metadata: {key}")
    if before["purpose"] != "forget_knowledge" or before["algorithm"] != "EAP-IG-inputs":
        raise ValueError("CAD needs forget-knowledge EAP-IG scores")
    if before["edge_destination_layer"] != after["edge_destination_layer"]:
        raise ValueError("attribution edge destinations differ")
    a, b = before["edges_signed_mean"], after["edges_signed_mean"]
    if set(a) != set(b) or set(a) != set(before["edge_destination_layer"]):
        raise ValueError("attribution edge identities differ; missing edges cannot be treated as zero")
    if not all(math.isfinite(v) for row in (a, b) for v in row.values()):
        raise ValueError("nonfinite attribution")
    selected = [e for e, layer in before["edge_destination_layer"].items() if layer in circuit_layers]
    denominator = sum(abs(a[e]) for e in selected)
    if not selected or denominator <= 0:
        raise ValueError("CAD is undefined for an empty/zero-attribution original circuit")
    return {"cad": sum(abs(a[e] - b[e]) for e in selected) / denominator,
            "original_edge_count": len(selected), "erasure_established": False}


def activation_shift(before_model, after_model, batches, circuit_layers):
    """Eq. 11 on matching teacher-forced tokens, excluding padding, streamed by batch."""
    def modules(model):
        return {i: layer.mlp for i, layer in enumerate(model.model.layers)}
    left, right = modules(before_model), modules(after_model)
    if left.keys() != right.keys() or not set(circuit_layers) < set(left):
        raise ValueError("need matching layers and a nonempty proper circuit subset")
    if not circuit_layers or not batches:
        raise ValueError("need circuit layers and diagnostic examples")
    sums = {i: [0.0, 0.0] for i in left}
    was_training = [before_model.training, after_model.training]
    before_model.eval(); after_model.eval()
    try:
        for batch in batches:
            values = []
            for model, layers in ((before_model, left), (after_model, right)):
                cache, hooks = {}, []
                def hook_for(i):
                    def record(module, inputs, output):
                        if isinstance(output, tuple):
                            output = output[0]
                        cache[i] = output.detach().float().cpu()
                    return record
                try:
                    for i, module in layers.items():
                        hooks.append(module.register_forward_hook(hook_for(i)))
                    device = next(model.parameters()).device
                    from contextlib import nullcontext
                    context = torch.autocast(device.type, dtype=torch.bfloat16) if next(model.parameters()).dtype == torch.bfloat16 else nullcontext()
                    with torch.no_grad(), context:
                        model(**{k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "position_ids"}}, use_cache=False)
                finally:
                    for hook in hooks:
                        hook.remove()
                values.append(cache)
            active = batch["attention_mask"].cpu().bool()
            for i in left:
                a, b = values[0][i][active], values[1][i][active]
                if a.shape != b.shape:
                    raise ValueError("activation shapes differ")
                sums[i][0] += float((a - b).square().sum())
                sums[i][1] += float(a.square().sum())
    finally:
        before_model.train(was_training[0]); after_model.train(was_training[1])
    if any(base <= 0 for difference, base in sums.values()):
        raise ValueError("activation shift undefined for a zero original activation norm")
    ratios = {i: math.sqrt(difference / base) for i, (difference, base) in sums.items()}
    selected, other = set(circuit_layers), set(left) - set(circuit_layers)
    return {"as_c": sum(ratios[i] for i in selected) / len(selected),
            "as_nc": sum(ratios[i] for i in other) / len(other), "per_layer": ratios,
            "erasure_established": False}


def quantized_update_survival(quantized_before, quantized_after, trained_delta):
    """Compare actual dequantized arrays; callers must supply the same quantizer/loader settings."""
    if quantized_before.shape != quantized_after.shape or trained_delta.shape != quantized_before.shape:
        raise ValueError("weight shapes differ")
    if any(not torch.isfinite(x).all() for x in (quantized_before, quantized_after, trained_delta)):
        raise ValueError("nonfinite weights")
    changed = trained_delta != 0
    n = int(changed.sum())
    if n == 0:
        raise ValueError("no trained updates to assess")
    survived = int(((quantized_before != quantized_after) & changed).sum())
    return {"updated_coordinates": n, "changed_after_quantization": survived,
            "survival_fraction": survived / n, "semantic_forgetting_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = cad(json.loads(args.before.read_text()), json.loads(args.after.read_text()), args.layers)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print("CAD saved; activation shifts and behavioral checks are still required.")


if __name__ == "__main__":
    main()
