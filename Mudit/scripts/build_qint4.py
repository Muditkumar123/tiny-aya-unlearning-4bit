"""Portable all-qint4 build (C1 from C0v2, C3 from C2): the current recipe, not the historical build script.

Every decoder attention and MLP linear goes to quanto qint4 (weights only, groups of 128, MaxOptimizer) through
CircuitKIT's _apply_plan_to_model; lm_head and embeddings stay bf16. No git, no fixed home paths. The output folder
must not exist. run.json records the source hashes, library versions and this script's sha256.

python Mudit/scripts/build_qint4.py <bf16 source dir> <new out dir> [--device cuda] [--expect-source-sha SHA]
"""
import argparse, hashlib, json, sys
from importlib.metadata import version
from pathlib import Path

import circuitkit as ck
import torch
from circuitkit.applications import detect_model_architecture, get_arch_config, validate_model_paths
from circuitkit.applications.quantization.quant_utils import _apply_plan_to_model
from circuitkit.evaluation import load_quantized_checkpoint
from optimum.quanto import QModuleMixin, qint4
from transformers import AutoModelForCausalLM


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def build(src, out, device="cuda", expect_source_sha=None):
    src, out = Path(src).resolve(), Path(out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists: a new checkpoint goes in a new folder")
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no *.safetensors in {src}")
    src_sha = {f.name: sha256(f) for f in shards}
    if expect_source_sha and expect_source_sha not in src_sha.values():
        raise SystemExit(f"source sha256 {src_sha} does not include the expected {expect_source_sha}")

    hf = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).to(device)
    n = hf.config.num_hidden_layers
    plan = {part: {layer: "low" for layer in range(n)} for part in ("attn", "mlp")}
    arch = get_arch_config(detect_model_architecture(hf))
    validate_model_paths(hf, arch)
    _apply_plan_to_model(hf, plan, qint4, None, None, None, True, arch)
    n_frozen = sum(isinstance(m, QModuleMixin) and m.frozen for m in hf.modules())
    ck.export_checkpoint(hf, None, str(out), intervention="quantization", tokenizer_name=str(src))
    del hf

    qmap = json.load(open(out / "quanto_qmap.json"))
    assert len(qmap) == 7 * n == n_frozen, f"qmap {len(qmap)}, frozen {n_frozen}, want {7 * n}"
    assert all(v["weights"] == "qint4" for v in qmap.values())
    assert not any("lm_head" in k or "embed" in k for k in qmap)
    model = load_quantized_checkpoint(str(out))
    n_reloaded = sum(isinstance(m, QModuleMixin) for m in model.modules())
    assert n_reloaded == len(qmap), f"{n_reloaded} QModules after reload, qmap has {len(qmap)}"

    run = {"recipe": "Mudit/scripts/build_qint4.py", "recipe_sha256": sha256(__file__),
           # a single-file source keeps the string form verify_quant.py reads; shards are listed separately
           "source": str(src), "source_safetensors_sha256": next(iter(src_sha.values())) if len(src_sha) == 1 else None,
           "source_shards_sha256": src_sha,
           "plan": "every attention and MLP unit qint4 (group 128, MaxOptimizer); lm_head (tied) and embeddings bf16",
           "qmap_entries": len(qmap), "frozen_qmodules": n_frozen, "reloaded_qmodules": n_reloaded,
           "out_safetensors_sha256": sha256(out / "model.safetensors"),
           "out_safetensors_bytes": (out / "model.safetensors").stat().st_size, "device": device,
           "versions": {m: version(m) for m in ("torch", "transformers", "optimum-quanto", "circuitkit", "safetensors")}}
    (out / "run.json").write_text(json.dumps(run, indent=2))
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source"); ap.add_argument("out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expect-source-sha")
    a = ap.parse_args()
    print(json.dumps(build(a.source, a.out, a.device, a.expect_source_sha), indent=2))
    sys.exit(0)
