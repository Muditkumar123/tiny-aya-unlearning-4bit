"""Let AuditKIT score a model that is already in memory, such as a CircuitKIT quanto checkpoint.

AuditKIT's hf backend builds `transformers.pipeline(model=<name>)`, so it can only load a model by name.
Its generate(), loglikelihood() and model_info() all read `self._pipeline.model`, so handing it a ready
pipeline makes every AuditKIT metric (including answer log-prob) work on quantized, pruned, steered or
PEFT models:

    from circuitkit.evaluation import load_quantized_checkpoint
    model = load_quantized_checkpoint(path).to("cuda")
    ak_model = InMemoryHFModel(model, tokenizer, name="tiny-aya-protected-q4")   # name: unique per checkpoint

Running a quanto model compiles quanto's kernels on first use. On a100server2 that needs
CUDA_HOME=/usr/local/cuda-12.9 (the default /usr/local/cuda is 11.4, which rejects gcc 13) and the env's
bin folder (for ninja) on PATH.
"""
from __future__ import annotations

from typing import Any

from auditkit.model.hf_gen import HFGenModel


def _nbytes(t) -> int:
    """Real storage of a tensor, counting quanto's packed tensors by their parts."""
    if hasattr(t, "__tensor_flatten__"):
        names, _ = t.__tensor_flatten__()
        return sum(_nbytes(getattr(t, n)) for n in names)
    return t.numel() * t.element_size()


class InMemoryHFModel(HFGenModel):
    """An AuditKIT hf model around a model object instead of a model name."""

    def __init__(self, model, tokenizer, name: str, fp32_logits: bool = False, **kwargs) -> None:
        # name is the whole cache identity: AuditKIT's Runner replays any cached run (~/.cache, shared by
        # every agent) whose fingerprint matches, so two checkpoints under one name get each other's results.
        # kwargs go to every pipeline call, e.g. batch_size=16 (left-padded; outputs can differ from batch 1).
        super().__init__(model=name, name=name, **kwargs)
        from transformers import pipeline

        if fp32_logits:
            # AuditKIT's loglikelihood runs log_softmax and the sum in the logits' dtype: in bf16 a 20-token sum near
            # -20 is rounded to a step of 0.125. Upcasting lm_head's output is exact, so generation is unchanged.
            # The hook stays on the model object.
            model.get_output_embeddings().register_forward_hook(lambda module, args, out: out.float())

        self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        for holder in (self._pipeline.model, self._pipeline):  # the same max_length fix HFGenModel applies on load
            cfg = getattr(holder, "generation_config", None)
            if cfg is not None:
                cfg.max_length = None

    def unload(self) -> None:
        # HFGenModel.unload() drops the pipeline and reloads it later from the model *name*, which here
        # would silently score a different model. The caller owns this model's memory, so keep it.
        pass

    def model_info(self) -> dict[str, Any]:
        info = super().model_info()
        # HFGenModel counts numel() * element_size(), which reports quanto's packed weights at their bf16 size.
        info["size_mb"] = sum(_nbytes(p) for p in self._pipeline.model.parameters()) / 2**20
        return info
