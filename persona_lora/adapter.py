"""Conditional LoRA: shared A/B, one gain vector per persona.

    dW_p = B @ diag(s_p) @ A

A and B are shared across personas; s_p selects a persona's coordinates inside the
shared rank-r subspace. If this trains, personas share a subspace and differ only in
coordinates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class PersonaSelector:
    """Mutable holder read by every adapter during forward."""
    index: int = 0
    enabled: bool = True


class CondLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, n_personas: int, sel: PersonaSelector,
                 alpha: float | None = None, share_b: bool = True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.sel = sel
        self.share_b = share_b
        self.scaling = (alpha if alpha is not None else 2.0 * r) / r
        d_in, d_out = base.in_features, base.out_features
        # Build on the base layer's device: inject() runs after model.to(device),
        # so default-device tensors would stay on CPU.
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, d_in, device=dev, dtype=torch.float32))
        if share_b:
            self.B = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=torch.float32))
            self.s = nn.Parameter(torch.ones(n_personas, r, device=dev, dtype=torch.float32))
        else:
            self.B = nn.Parameter(torch.zeros(n_personas, d_out, r, device=dev,
                                             dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.sel.enabled:
            return out
        h = torch.nn.functional.linear(x, self.A.to(x.dtype))
        if self.share_b:
            h = h * self.s[self.sel.index].to(x.dtype)
            B = self.B
        else:
            B = self.B[self.sel.index]
        return out + torch.nn.functional.linear(h, B.to(x.dtype)) * self.scaling


def layer_range(placement: str, n_layers: int) -> range:
    """Placement -> decoder block indices (0-based).

    "L<k>" is a SINGLE block, which is what "at which layer does it sit" asks.
    early/middle/late are thirds of the stack (8 blocks each at 24, 12 at 36) --
    useful as coarse bands, but they cannot localise to a layer.
    """
    if placement.startswith("L") and placement[1:].isdigit():
        k = int(placement[1:])
        if not 0 <= k < n_layers:
            raise ValueError(f"layer {k} out of range for {n_layers} blocks")
        return range(k, k + 1)
    third = n_layers // 3
    return {
        "all": range(0, n_layers),
        "early": range(0, third),
        "middle": range(third, 2 * third),
        "late": range(2 * third, n_layers),
    }[placement]


def inject(model, r: int, n_personas: int, placement: str = "all",
           targets=TARGETS, share_b: bool = True) -> tuple[PersonaSelector, int]:
    """Wrap target Linears inside the placement's layers. Returns (selector, n_params)."""
    sel = PersonaSelector()
    n_layers = model.config.num_hidden_layers
    keep = set(layer_range(placement, n_layers))
    layers = model.model.layers
    n_wrapped = 0
    for li, layer in enumerate(layers):
        if li not in keep:
            continue
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if name in targets and isinstance(child, nn.Linear):
                    setattr(parent, name,
                            CondLoRALinear(child, r, n_personas, sel, share_b=share_b))
                    n_wrapped += 1
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = 0
    for m in model.modules():
        if isinstance(m, CondLoRALinear):
            ps = (m.A, m.B, m.s) if m.share_b else (m.A, m.B)
            for p in ps:
                p.requires_grad_(True)
                n_params += p.numel()
    if n_wrapped == 0:
        raise RuntimeError(f"no modules wrapped for placement={placement}")
    return sel, n_params


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
