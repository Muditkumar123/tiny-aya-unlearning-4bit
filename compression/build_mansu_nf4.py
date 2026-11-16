"""Build, save and reload an NF4 checkpoint for the NEW MANSU candidate. No evaluation claim."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from run_mansu import checkpoint_identity


def state_hashes(model):
    """Include packed weights and NF4 scale/state tensors in the reload identity."""
    import torch
    result = {}
    for name, tensor in model.state_dict().items():
        raw = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256(f"{raw.dtype}:{list(raw.shape)}:".encode())
        digest.update(raw.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--expect-source-sha", required=True, nargs="+", help="SHA256 of every source weight shard")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be a new directory")
    source_hashes = checkpoint_identity(args.source)
    if sorted(args.expect_source_sha) != sorted(source_hashes.values()):
        raise ValueError("source checkpoint shards do not match the complete frozen hash list")
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("specify one explicit CUDA device, such as cuda:0")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    def load(path):
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True,
                                                    quantization_config=quantization, device_map={"": str(device)})
    def check(model):
        modules = {n: m for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)}
        if len(modules) != 7 * model.config.num_hidden_layers:
            raise ValueError("unexpected NF4 module count")
        for name, module in modules.items():
            if ".layers." not in name or module.weight.quant_state.quant_type != "nf4" or module.weight.quant_state.blocksize != 64:
                raise ValueError("unexpected NF4 precision or scope")
            if module.compute_dtype != torch.bfloat16 or module.weight.quant_state.nested:
                raise ValueError("expected bf16 NF4 compute with no double quantization")
        return set(modules)
    gpu_before = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"], text=True)
    model = load(args.source)
    names = set(check(model))
    original_state = state_hashes(model)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.source, local_files_only=True).save_pretrained(args.out)
    del model
    torch.cuda.empty_cache()
    reloaded = load(args.out)
    if set(check(reloaded)) != names:
        raise ValueError("NF4 module identities changed on reload")
    reloaded_state = state_hashes(reloaded)
    if original_state != reloaded_state:
        raise ValueError("saved NF4 tensor or scale state changed on reload")
    report = {"source_weights_sha256": source_hashes, "output_weights_sha256": checkpoint_identity(args.out),
              "quantizer": {"type": "nf4", "blocksize": 64, "double_quant": False, "compute_dtype": "bfloat16"},
              "nf4_linears": len(names), "save_reload_module_check": True,
              "save_reload_tensor_identity": True, "reloaded_state_sha256": reloaded_state,
              "gpu_before_load": gpu_before,
              "versions": {m: version(m) for m in ("bitsandbytes", "torch", "transformers")},
              "behavioral_evaluation_run": False, "quantized_update_survival_verified": False}
    (args.out / "nf4_build.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NF4 saved and reloaded; behavior and update survival still need evaluation.")


if __name__ == "__main__":
    main()
