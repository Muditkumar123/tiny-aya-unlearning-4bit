"""Paper-based MANSU core (Algorithm 1, arXiv:2605.15138v1).

This independent implementation uses the paper's diagonal-Fisher approximation.
Its range-based floor is an approximation, not a quantization-survival proof.
See docs/MANSU.md for conflicting paper details and the explicit choices here.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import cycle
import math
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MansuConfig:
    learning_rate: float = 1e-6
    max_steps: int = 30
    kl_weight: float = 200.0
    fisher_quantile: float = 0.99
    floor_policy: str = "rescale"
    floor_minimum: float = 1e-6
    floor_scale: str = "tensor_range"
    floor_margin: float = 0.704
    nf4_block_size: int = 64
    evaluate_every: int = 5
    max_utility_drop: float = 0.08
    early_stop_utility_drop: float = 0.02

    def __post_init__(self):
        for name in ("learning_rate", "kl_weight", "floor_minimum", "max_utility_drop", "early_stop_utility_drop"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.floor_minimum == 0:
            raise ValueError("learning_rate and floor_minimum must be positive")
        if (type(self.max_steps) is not int or type(self.evaluate_every) is not int or
                self.max_steps < 1 or self.evaluate_every < 1 or not 0 <= self.fisher_quantile <= 1):
            raise ValueError("invalid step count or Fisher quantile")
        if self.max_utility_drop > 1 or self.early_stop_utility_drop > 1:
            raise ValueError("utility drops must be fractions between 0 and 1")
        if self.floor_policy not in {"rescale", "zero"}:
            raise ValueError("floor_policy must explicitly be rescale or zero")
        if self.floor_scale not in {"tensor_range", "nf4_block"}:
            raise ValueError("floor_scale must explicitly be tensor_range or nf4_block")
        if not math.isfinite(self.floor_margin) or not 0 < self.floor_margin <= 1 or self.nf4_block_size != 64:
            raise ValueError("use a finite floor margin in (0,1] and NF4 blocksize64")


def completion_nll(logits: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    """Causal next-token CE on answer positions only; padding/prompt labels are -100."""
    if logits.shape[:2] != labels.shape or labels.ndim != 2:
        raise ValueError("logits and labels must share batch and sequence dimensions")
    if not (labels[:, 1:] != -100).any():
        raise ValueError("no scored completion tokens")
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100, reduction=reduction)


def reference_kl(candidate: Tensor, reference: Tensor, labels: Tensor) -> Tensor:
    """Mean token KL(reference || candidate), at scored answer positions."""
    if candidate.shape != reference.shape or candidate.shape[:2] != labels.shape:
        raise ValueError("reference/candidate logits or label shapes differ")
    active = labels[:, 1:] != -100
    if not active.any():
        raise ValueError("no retain completion tokens")
    live = F.log_softmax(candidate[:, :-1][active].float(), dim=-1)
    ref = F.log_softmax(reference[:, :-1][active.to(reference.device)].detach().to(candidate.device).float(), dim=-1)
    return (ref.exp() * (ref - live)).sum(-1).mean()


def circuit_parameters(model: nn.Module, layers: Sequence[int]) -> dict[str, nn.Parameter]:
    """Only gate/up/down weight matrices in selected Llama/Cohere-style MLPs."""
    if not layers or len(set(layers)) != len(layers) or any(type(i) is not int or i < 0 for i in layers):
        raise ValueError("circuit layers must be unique nonnegative integers")
    wanted = {f"model.layers.{i}.mlp.{projection}.weight"
              for i in layers for projection in ("gate_proj", "up_proj", "down_proj")}
    named = dict(model.named_parameters())
    if missing := wanted - named.keys():
        raise ValueError(f"unsupported model or circuit: missing {sorted(missing)}")
    if any(named[name].ndim != 2 for name in wanted):
        raise ValueError("MLP circuit parameters must be matrices")
    return {name: named[name] for name in sorted(wanted)}


def rank_mlp_layers(edge_absolute_means: Mapping[str, float], destinations: Mapping[str, int], k: int) -> list[int]:
    """Sum mean-absolute incoming edge mass, never abs(mean(signed scores))."""
    if set(edge_absolute_means) != set(destinations):
        raise ValueError("edge scores and destination identities differ")
    totals = {}
    for edge, value in edge_absolute_means.items():
        layer = destinations[edge]
        if type(layer) is not int or layer < 0 or not math.isfinite(value) or value < 0:
            raise ValueError("invalid attribution score or destination")
        totals[layer] = totals.get(layer, 0.0) + value
    if not 1 <= k <= len(totals) or sum(totals.values()) <= 0:
        raise ValueError("insufficient nonzero forget attribution for top-k selection")
    return sorted(totals, key=lambda layer: (-totals[layer], layer))[:k]


def magnitude_floor(original: Tensor, trained: Tensor, *, minimum: float = 1e-6,
                    policy: str = "rescale", allowed: Tensor | None = None,
                    scale: str = "tensor_range", margin: float = 0.704, block_size: int = 64) -> tuple[Tensor, dict]:
    """Apply Eq. 9/10 to cumulative deltas; zeros and Fisher-protected coordinates stay unchanged.

    Eq. 9 is interpreted as one floor per tensor, using its original absolute
    range / 16. A block quantizer can change its own scales, so no survival
    guarantee follows from this approximation.
    """
    if policy not in {"rescale", "zero"} or not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("invalid floor policy/minimum")
    if original.shape != trained.shape or not torch.isfinite(original).all() or not torch.isfinite(trained).all():
        raise ValueError("floor requires matching finite tensors")
    source = original.to(device=trained.device, dtype=torch.float32)
    live = trained.float()
    enabled = torch.ones_like(source, dtype=torch.bool) if allowed is None else allowed.to(source.device)
    if enabled.shape != source.shape or enabled.dtype != torch.bool:
        raise ValueError("floor mask must be Boolean with the parameter shape")
    delta = torch.where(enabled, live - source, 0.0)
    if scale == "tensor_range":
        threshold = max(float((source.abs().max() - source.abs().min()).item()) / 16, minimum)
    elif scale == "nf4_block":
        if not math.isfinite(margin) or not 0 < margin <= 1 or block_size != 64:
            raise ValueError("invalid NF4 floor margin or block size")
        flat = source.flatten()
        padded = F.pad(flat, (0, (-flat.numel()) % block_size))
        # bitsandbytes NF4 uses each contiguous block's absolute maximum as scale.
        scales = padded.abs().reshape(-1, block_size).amax(-1)
        threshold = (scales * 0.0796 * margin).clamp_min(minimum).repeat_interleave(block_size)[:flat.numel()].reshape_as(source)
    else:
        raise ValueError("unknown floor scale")
    small = enabled & (delta != 0) & (delta.abs() < threshold)
    adjusted = delta.clone()
    selected_floor = threshold if isinstance(threshold, float) else threshold[small]
    adjusted[small] = delta[small].sign() * selected_floor if policy == "rescale" else 0.0
    floor_info = {"floor": threshold} if isinstance(threshold, float) else {"floor_min": float(threshold.min()), "floor_max": float(threshold.max()), "margin": margin, "block_size": block_size}
    return source + adjusted, {**floor_info, "scale": scale, "policy": policy, "adjusted_elements": int(small.sum()),
                              "nonzero_before": int((delta != 0).sum()), "nonzero_after": int((adjusted != 0).sum())}


def _device(model):
    return next(model.parameters()).device


def _forward(model, batch):
    device = _device(model)
    inputs = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "position_ids"}}
    first_dtype = next(model.parameters()).dtype
    context = torch.autocast(device.type, dtype=torch.bfloat16) if first_dtype == torch.bfloat16 else nullcontext()
    with context:
        return model(**inputs, use_cache=False).logits


def estimate_fisher(model, parameters, examples):
    """Empirical diagonal Fisher: mean of squared per-example score gradients.

    Each item MUST be one example. Squaring a batch-average gradient instead
    would allow examples with opposing gradients to cancel.
    """
    if not examples:
        raise ValueError("Fisher estimation needs retain examples")
    fisher = {name: torch.zeros_like(param, device="cpu", dtype=torch.float32)
              for name, param in parameters.items()}
    for example in examples:
        if example["input_ids"].shape[0] != 1:
            raise ValueError("Fisher estimation requires singleton examples")
        loss = completion_nll(_forward(model, example), example["labels"].to(_device(model)), reduction="sum")
        gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=False)
        for (name, _), gradient in zip(parameters.items(), gradients):
            value = gradient.detach().float().cpu()
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite retain score gradient")
            fisher[name].add_(value.square())
    return {name: value / len(examples) for name, value in fisher.items()}


class MansuTrainer:
    """Full-weight, circuit-restricted unlearning with a frozen retain reference.

    Only selected MLP matrices have optimizer state. FP32 trainable weights
    prevent bf16 from silently discarding small updates before the floor pass.
    """
    def __init__(self, model: nn.Module, reference: nn.Module, layers: Sequence[int], config: MansuConfig):
        if model is reference:
            raise ValueError("the reference must be an independent frozen model")
        if _device(model) != _device(reference):
            # CPU-offloaded reference is supported; forward outputs are transferred explicitly.
            if _device(reference).type != "cpu":
                raise ValueError("reference must share the training device or be on CPU")
        candidate = dict(model.named_parameters())
        ref = dict(reference.named_parameters())
        if candidate.keys() != ref.keys():
            raise ValueError("reference model parameter identities differ")
        # Full identity is checked before any parameter promotion or optimization.
        for name, param in candidate.items():
            if param.device == ref[name].device and param.untyped_storage().data_ptr() == ref[name].untyped_storage().data_ptr():
                raise ValueError("candidate and reference share parameter storage")
            if param.shape != ref[name].shape or not torch.equal(param.detach().cpu(), ref[name].detach().cpu()):
                raise ValueError(f"reference is not the starting checkpoint: {name}")
        self.model, self.reference, self.config = model, reference, config
        self.layers = list(layers)
        self.parameters = circuit_parameters(model, layers)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self.original = {name: p.detach().float().cpu().clone() for name, p in self.parameters.items()}
        for parameter in self.parameters.values():
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
        self.model.eval()
        self.reference.eval()
        self.masks = None

    def prepare_fisher(self, retain_examples):
        if self.masks is not None or hasattr(self, "_training_started"):
            raise ValueError("Fisher must be estimated once, before training")
        fisher = estimate_fisher(self.model, self.parameters, retain_examples)
        flat = np.concatenate([value.numpy().reshape(-1) for value in fisher.values()])
        if not np.isfinite(flat).all():
            raise ValueError("nonfinite Fisher estimate")
        # NumPy avoids torch.quantile's 2**24-element implementation limit.
        threshold = float(np.quantile(flat, self.config.fisher_quantile))
        self.masks = {name: (value <= threshold).to(self.parameters[name].device) for name, value in fisher.items()}
        self.fisher_report = {"examples": len(retain_examples), "threshold": threshold,
                              "quantile": self.config.fisher_quantile,
                              "allowed_coordinates": sum(int(x.sum()) for x in self.masks.values()),
                              "circuit_coordinates": flat.size}
        return self.fisher_report

    def train(self, forget_batches, retain_batches, *, evaluate: Callable | None = None):
        if hasattr(self, "_training_started"):
            raise ValueError("use a fresh trainer/checkpoint for another run")
        if self.masks is None:
            raise ValueError("prepare_fisher must run at the starting weights")
        if not forget_batches or not retain_batches:
            raise ValueError("both forget and retain examples are required")
        self._training_started = True
        cfg = self.config
        # A fixed mask and zero decay also prevent optimizer movement on protected coordinates.
        optimizer = torch.optim.AdamW(self.parameters.values(), lr=cfg.learning_rate, weight_decay=0.0)
        protected = {name: p.detach()[~self.masks[name]].clone() for name, p in self.parameters.items()}
        baseline = evaluate(self.model) if evaluate else None
        history, best_weights, best_metrics, best_step = [], None, None, None
        self.model.train()  # Enable checkpointing during optimization; validation uses eval mode.
        for step, (forget, retain) in enumerate(zip(cycle(forget_batches), cycle(retain_batches)), start=1):
            optimizer.zero_grad(set_to_none=True)
            forget_nll = completion_nll(_forward(self.model, forget), forget["labels"].to(_device(self.model)))
            with torch.no_grad():
                target = _forward(self.reference, retain)
            kl = reference_kl(_forward(self.model, retain), target, retain["labels"].to(_device(self.model)))
            objective = -forget_nll + cfg.kl_weight * kl
            if not torch.isfinite(objective):
                raise ValueError("nonfinite MANSU objective")
            objective.backward()
            for name, parameter in self.parameters.items():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"missing/nonfinite gradient: {name}")
                parameter.grad.mul_(self.masks[name])
            optimizer.step()
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter[~self.masks[name]] = protected[name]
                    if not torch.isfinite(parameter).all():
                        raise ValueError("nonfinite trained parameter")
            row = {"step": step, "forget_nll": float(forget_nll.detach()), "retain_kl": float(kl.detach()),
                   "objective": float(objective.detach())}
            stop = False
            if evaluate and (step % cfg.evaluate_every == 0 or step == cfg.max_steps):
                self.model.eval()
                try:
                    metrics = evaluate(self.model)
                finally:
                    self.model.train()
                for result in (baseline, metrics):
                    if set(result) != {"forget_accuracy", "utility_accuracy"} or not all(
                            math.isfinite(x) and 0 <= x <= 1 for x in result.values()):
                        raise ValueError("validation must return finite forget/utility accuracies")
                row["validation"] = metrics
                drop = baseline["utility_accuracy"] - metrics["utility_accuracy"]
                if drop <= cfg.max_utility_drop + 1e-12 and (best_metrics is None or
                                                     metrics["forget_accuracy"] < best_metrics["forget_accuracy"]):
                    best_metrics, best_step = dict(metrics), step
                    best_weights = {name: p.detach().cpu().clone() for name, p in self.parameters.items()}
                stop = drop > cfg.early_stop_utility_drop + 1e-12
            history.append(row)
            if stop or step == cfg.max_steps:
                break
        if evaluate:
            if best_weights is None:
                raise ValueError("no trained checkpoint met the configured utility constraint")
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(best_weights[name].to(parameter.device))
        self.model.eval()
        self.training_report = {"config": asdict(cfg), "history": history, "baseline": baseline,
                                "selected_step": best_step, "selected_metrics": best_metrics,
                                "selection": "lowest validation forget accuracy subject to utility bound" if evaluate
                                else "last step; no validation selection (core test only)"}
        return self.training_report

    def apply_floor(self):
        if not hasattr(self, "training_report"):
            raise ValueError("training must finish before the post-hoc floor")
        report = {}
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                value, info = magnitude_floor(self.original[name], parameter, minimum=self.config.floor_minimum,
                                             policy=self.config.floor_policy, allowed=self.masks[name],
                                             scale=self.config.floor_scale, margin=self.config.floor_margin,
                                             block_size=self.config.nf4_block_size)
                parameter.copy_(value)
                report[name] = info
        return report
