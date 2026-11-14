"""R1 re-verification of a quantized checkpoint (PLAN section 4): identity, source lineage, loader and precision.

Usage:
  python verify_quant.py CKPT --source SRC_DIR --kind quanto --expect-q4 N --expect-bf16 M [--source-sha SHA] --out report.json
  python verify_quant.py CKPT --source SRC_DIR --kind nf4 --expect-nf4 N [--source-sha SHA] --out report.json   (needs a GPU)
SRC_DIR is the bf16 model the checkpoint was built from (a local folder or an HF cache snapshot).
Exit 0 = pass, 1 = fail. Generation checks are separate (R4/R5).
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import torch


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def decoder_linears(names):
    return sorted({n for n in names if n.startswith("model.layers.") and n.rsplit(".", 1)[-1].endswith("_proj")})


def source_files(src):
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(src, f) for k, f in wm.items()}, sorted({os.path.join(src, f) for f in wm.values()})
    f = os.path.join(src, "model.safetensors")
    from safetensors import safe_open
    with safe_open(f, "pt") as s:
        return {k: f for k in s.keys()}, [f]


class SourceTensors:
    def __init__(self, src):
        from safetensors import safe_open
        self.where, self.files = source_files(src)
        self._open = {f: safe_open(f, "pt") for f in self.files}

    def get(self, name):
        f = self.where.get(name)
        return None if f is None else self._open[f].get_tensor(name)


def check_header(hdr, qmap):
    """Stored dtype guard: every decoder linear the qmap leaves out, and the embeddings, must be stored BF16."""
    fails, seen = [], {}
    for n in decoder_linears(k.split(".weight")[0] for k in hdr if ".weight" in k):
        if n not in qmap:
            dt = hdr.get(n + ".weight", {}).get("dtype")
            seen[n] = dt
            if dt != "BF16":
                fails.append(f"{n}: stored {dt}, expected BF16")
    for k in ("model.embed_tokens.weight", "lm_head.weight"):
        if k in hdr and hdr[k]["dtype"] != "BF16":
            fails.append(f"{k}: stored {hdr[k]['dtype']}, expected BF16")
    if "model.embed_tokens.weight" not in hdr:
        fails.append("model.embed_tokens.weight missing from the checkpoint")
    return sorted(set(seen.values())), fails


def check_quanto_model(model, qmap, expect_q4, expect_bf16):
    """Compare the reloaded modules with the saved qmap; return (checks, failures)."""
    from optimum.quanto import QModuleMixin
    mods = dict(model.named_modules())
    lin = decoder_linears(mods)
    fails, q4, bf16 = [], 0, 0
    for name in lin:
        m, want = mods[name], qmap.get(name)
        if isinstance(m, QModuleMixin):
            got = m.weight_qtype.name if m.weight_qtype is not None else None
            if want is None or want["weights"] != got or want.get("activations", "none") != "none" or m.activation_qtype is not None:
                fails.append(f"{name}: quantized as {got}, qmap says {want}")
            q4 += got == "qint4"
        else:
            if want is not None:
                fails.append(f"{name}: qmap says {want['weights']} but module is not quantized")
            elif m.weight.dtype != torch.bfloat16:
                fails.append(f"{name}: unquantized but {m.weight.dtype}, not bf16")
            bf16 += m.weight.dtype == torch.bfloat16
    for k in qmap:
        if k not in mods:
            fails.append(f"qmap entry {k} has no module")
        if "embed" in k or "lm_head" in k:
            fails.append(f"qmap quantizes {k}")
    for name, m in mods.items():
        if name.endswith(("embed_tokens", "lm_head")) and isinstance(m, QModuleMixin):
            fails.append(f"{name} is quantized")
    if q4 != expect_q4:
        fails.append(f"qint4 linears {q4}, expected {expect_q4}")
    if bf16 != expect_bf16:
        fails.append(f"bf16 linears {bf16}, expected {expect_bf16}")
    return {"decoder_linears": len(lin), "qint4": q4, "bf16": bf16, "qmap_entries": len(qmap)}, fails


def check_excluded(model, source_get, tied):
    """Every parameter that is not a quantized weight must be bf16 and bitwise equal to the source; the tied head must share the embeddings."""
    from optimum.quanto import QTensor
    fails, compared = [], 0
    for name, p in model.named_parameters():
        if isinstance(p, QTensor) or type(p.data).__name__ in ("Params4bit",) or getattr(p, "quant_state", None) is not None:
            continue
        ref = source_get(name)
        if ref is None:
            fails.append(f"{name}: not in the source model")
            continue
        compared += 1
        if p.dtype != torch.bfloat16:
            fails.append(f"{name}: {p.dtype}, expected bfloat16")
        elif p.shape != ref.shape or not torch.equal(p.detach().cpu(), ref.to(p.dtype)) or ref.dtype != torch.bfloat16:
            fails.append(f"{name}: differs from the source")
    emb, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    if tied and head.data_ptr() != emb.data_ptr() and not torch.equal(head.detach().cpu(), emb.detach().cpu()):
        fails.append("lm_head is not tied to the embeddings")
    if not tied:
        ref = source_get("lm_head.weight")
        if ref is None or not torch.equal(head.detach().cpu(), ref):
            fails.append("lm_head differs from the source")
    return compared, fails


def check_nf4_model(model, expect_nf4):
    import bitsandbytes as bnb
    mods = dict(model.named_modules())
    lin = decoder_linears(mods)
    nf4 = [n for n in lin if isinstance(mods[n], bnb.nn.Linear4bit) and mods[n].weight.quant_type == "nf4"]
    fails = [] if len(nf4) == expect_nf4 == len(lin) else [f"nf4 linears {len(nf4)} of {len(lin)}, expected {expect_nf4}"]
    for name, m in mods.items():
        if name.endswith(("embed_tokens", "lm_head")) and isinstance(m, bnb.nn.Linear4bit):
            fails.append(f"{name} is 4-bit")
    return {"decoder_linears": len(lin), "nf4": len(nf4)}, fails


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--source-sha", help="expected sha256 of the source model.safetensors (e.g. an audited R2 reference)")
    ap.add_argument("--kind", choices=["quanto", "nf4"], required=True)
    ap.add_argument("--expect-q4", type=int, default=0)
    ap.add_argument("--expect-bf16", type=int, default=0)
    ap.add_argument("--expect-nf4", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    assert not os.path.exists(a.out), f"{a.out} exists; write a new report"
    st = os.path.join(a.ckpt, "model.safetensors")
    rep = {"ckpt": os.path.abspath(a.ckpt), "kind": a.kind, "sha256": sha256(st), "bytes": os.path.getsize(st),
           "sha256_status": "new reference: no output hash is recorded in this checkpoint's provenance",
           "versions": {"torch": torch.__version__}}
    fails = []

    run = os.path.join(a.ckpt, "run.json")
    r = json.load(open(run)) if os.path.exists(run) else {}
    rec_bytes = r.get("safetensors_bytes") or r.get("files", {}).get("model.safetensors")
    rec_src_sha = r.get("source_safetensors_sha256") or r.get("src_sha256")
    rep["recorded"] = {"bytes": rec_bytes, "source": r.get("source") or r.get("src"), "source_sha256": rec_src_sha}
    if rec_bytes is not None and rec_bytes != rep["bytes"]:
        fails.append(f"bytes {rep['bytes']} != recorded {rec_bytes}")

    src = SourceTensors(a.source)
    src_hashes = {os.path.basename(f): sha256(f) for f in src.files}
    rep["source"] = {"dir": os.path.abspath(a.source), "sha256": src_hashes}
    single = next(iter(src_hashes.values())) if len(src_hashes) == 1 else None
    if rec_src_sha is not None:
        if single != rec_src_sha:
            fails.append(f"source sha256 {single} != recorded source sha256 {rec_src_sha}")
    else:
        rep["source"]["note"] = "no source hash recorded in the checkpoint: lineage rests on the recorded path and the bitwise weight comparison below"
    if a.source_sha is not None and single != a.source_sha:
        fails.append(f"source sha256 {single} != expected {a.source_sha}")
    tied = json.load(open(os.path.join(a.ckpt, "config.json"))).get("tie_word_embeddings", True)

    if a.kind == "quanto":
        import optimum.quanto
        from circuitkit.evaluation import load_quantized_checkpoint
        qmap = json.load(open(os.path.join(a.ckpt, "quanto_qmap.json")))
        rep["header_unquantized_dtypes"], f = check_header(safetensors_header(st), qmap)
        fails += f
        rep["versions"]["optimum-quanto"] = optimum.quanto.__version__
        model = load_quantized_checkpoint(a.ckpt)
        checks, f = check_quanto_model(model, qmap, a.expect_q4, a.expect_bf16)
    else:
        import bitsandbytes
        from transformers import AutoModelForCausalLM
        assert torch.cuda.is_available(), "NF4 reload needs a GPU"
        rep["versions"]["bitsandbytes"] = bitsandbytes.__version__
        model = AutoModelForCausalLM.from_pretrained(a.ckpt, device_map={"": 0})
        checks, f = check_nf4_model(model, a.expect_nf4)
    fails += f
    checks["excluded_params_compared"], f = check_excluded(model, src.get, tied)
    fails += f
    rep["checks"], rep["failures"] = checks, fails
    rep["pass"] = not fails
    json.dump(rep, open(a.out, "x"), indent=1)
    print(json.dumps({k: rep[k] for k in ("ckpt", "sha256", "source", "checks", "failures", "pass")}, indent=1))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
