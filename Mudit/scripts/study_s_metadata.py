"""Study S build metadata, CPU only: which decoder linears each condition keeps in bf16, whether that
matches its saved plan and quanto qmap, parameter and stored-byte budgets, and random-seed provenance.

python Mudit/scripts/study_s_metadata.py [--out Mudit/out/study_s_metadata.json]
"""
import argparse, ast, hashlib, json, struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Mudit/out"
SNAP = Path.home() / ".cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324"
CONDS = {"plain-q4": OUT / "plain-q4", "random30-q4": OUT / "random30-q4",
         "protected-q4": OUT / "refusal_circuit_matched/tiny-aya-refusal-protected-q4"}
PLANS = {"random30-q4": OUT / "random30-q4/run.json", "protected-q4": OUT / "refusal_circuit_matched/run.json"}
UNITS = {"attn": ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), "mlp": ("mlp", ("gate_proj", "up_proj", "down_proj"))}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def plan_of(path):
    p = json.load(open(path))["plan"]
    return ast.literal_eval(p) if isinstance(p, str) else p  # protected run.json stored the plan via default=str


def expand(plan):
    """Plan tiers -> set of decoder-linear names kept in bf16 ('high')."""
    return {f"model.layers.{l}.{sub}.{m}" for part, (sub, mods) in UNITS.items()
            for l, t in plan[part].items() if t == "high" for m in mods}


def bytes_of(entry):
    a, b = entry["data_offsets"]
    return b - a


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "study_s_metadata.json")
    a = ap.parse_args(argv)

    base = {}
    for f in sorted(SNAP.glob("model-*.safetensors")):
        base.update(header(f))
    n_layers = 1 + max(int(k.split(".")[2]) for k in base if k.startswith("model.layers."))
    linears = sorted(f"model.layers.{l}.{sub}.{m}" for l in range(n_layers) for sub, mods in UNITS.values() for m in mods)
    assert all(f"{n}.weight" in base for n in linears) and len(linears) == 252, len(linears)
    params = {n: base[f"{n}.weight"]["shape"][0] * base[f"{n}.weight"]["shape"][1] for n in linears}

    res = {"base_snapshot": str(SNAP), "n_layers": n_layers, "decoder_linears": len(linears),
           "decoder_linear_params": sum(params.values()),
           "code_md5": {p: md5(ROOT / p) for p in ("Mudit/scripts/make_baselines.py", "Mudit/scripts/aya_refusal_circuit.py",
                                                   "Mudit/scripts/study_s_metadata.py")},
           "conditions": {}}
    kept_sets = {}
    for name, d in CONDS.items():
        qmap = json.load(open(d / "quanto_qmap.json"))
        assert set(qmap) <= set(linears), f"{name}: qmap names outside the decoder linears"
        assert all(v == {"weights": "qint4", "activations": "none"} for v in qmap.values()), name
        kept = set(linears) - set(qmap)
        h = header(d / "model.safetensors")
        kept_bytes = sum(bytes_of(h[f"{n}.weight"]) for n in kept)
        assert all(h[f"{n}.weight"]["dtype"] == "BF16" for n in kept), name
        q_bytes = sum(bytes_of(v) for k, v in h.items() if any(k.startswith(n + ".") for n in qmap))
        c = {"kept_bf16_linears": len(kept), "qint4_linears": len(qmap),
             "kept_bf16_params": sum(params[n] for n in kept), "qint4_params": sum(params[n] for n in qmap),
             "stored_bytes_kept_bf16_linears": kept_bytes, "stored_bytes_qint4_linears_incl_scales": q_bytes,
             "file_bytes": (d / "model.safetensors").stat().st_size,
             "kept_units": {part: sorted({int(n.split(".")[2]) for n in kept if f".{sub}." in n})
                            for part, (sub, _) in UNITS.items()},
             "kept_linears": sorted(kept, key=lambda n: (int(n.split(".")[2]), n))}
        if name in PLANS:
            plan = plan_of(PLANS[name])
            c["plan_file"] = str(PLANS[name].relative_to(ROOT))
            c["plan_matches_qmap"] = expand(plan) == kept
            assert c["plan_matches_qmap"], f"{name}: plan and qmap disagree"
            # a unit is kept or quantized as a whole: no partly quantized attention or MLP block
            for part, (sub, mods) in UNITS.items():
                for l in range(n_layers):
                    got = {f"model.layers.{l}.{sub}.{m}" in kept for m in mods}
                    assert len(got) == 1, f"{name}: layer {l} {part} partly kept"
        else:
            assert not kept, "plain-q4 must quantize every decoder linear"
        kept_sets[name] = kept
        res["conditions"][name] = c

    r, p = kept_sets["random30-q4"], kept_sets["protected-q4"]
    res["protected_vs_random"] = {
        "shared_linears": len(r & p), "jaccard": len(r & p) / len(r | p),
        "shared_units": {part: sorted(set(res["conditions"]["random30-q4"]["kept_units"][part]) &
                                      set(res["conditions"]["protected-q4"]["kept_units"][part])) for part in UNITS},
        "kept_param_difference": res["conditions"]["protected-q4"]["kept_bf16_params"] - res["conditions"]["random30-q4"]["kept_bf16_params"]}

    from circuitkit import __version__ as ck_version
    from circuitkit.applications.quantization.quant_utils import build_random_quantization_plan
    run = json.load(open(PLANS["random30-q4"]))
    redraw = build_random_quantization_plan(n_layers, high_fraction=run["high_fraction"], seed=run["seed"])
    saved = plan_of(PLANS["random30-q4"])
    norm = lambda pl: {part: {int(l): t for l, t in pl[part].items()} for part in UNITS}  # noqa: E731
    res["random_seed_provenance"] = {
        "recorded": {"seed": run["seed"], "high_fraction": run["high_fraction"], "quantizer": run["quantizer"]},
        "installed_circuitkit": ck_version,
        "redraw_with_installed_circuitkit_equals_saved_plan": norm(redraw) == norm(saved)}
    prot = json.load(open(PLANS["protected-q4"]))
    res["protected_provenance"] = {k: prot[k] for k in ("refusal_token", "compliance_token", "high_fraction")}
    sc = json.load(open(OUT / "refusal_circuit_matched/eap-ig_CohereLabs_tiny-aya-global_aya_refusal_node_scores.json"))
    res["protected_provenance"]["eap_ig_scores"] = {k: sc.get(k) for k in ("task", "model", "algorithm", "level", "timestamp", "version", "discovery_cfg")}
    res["protected_provenance"]["selection_code"] = "Mudit/scripts/aya_refusal_circuit.py: circuitkit quantize(hf, circuit, high_fraction=0.3)"

    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "conditions"}, indent=1))
    for n, c in res["conditions"].items():
        print(n, {k: v for k, v in c.items() if k != "kept_linears"})


if __name__ == "__main__":
    main()
