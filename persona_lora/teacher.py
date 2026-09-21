"""Teacher = the same frozen base model with the persona system prompt prepended.

The persona prefix KV cache is computed once per persona and reused for every batch.
``check_cache_equivalence`` verifies that against a full-sequence forward before any
training happens -- a silent cache bug would corrupt every target.
"""
from __future__ import annotations

import copy

import torch
from transformers import DynamicCache


@torch.no_grad()
def build_prefix_cache(model, prefix_ids: torch.Tensor):
    """prefix_ids: [1, P]. Returns (cache, P)."""
    out = model(input_ids=prefix_ids, use_cache=True)
    return out.past_key_values, prefix_ids.shape[1]


def _kv(cache, li):
    """(keys, values) for layer li, across transformers cache layouts."""
    if hasattr(cache, "layers"):
        layer = cache.layers[li]
        return layer.keys, layer.values
    return cache.key_cache[li], cache.value_cache[li]


def _n_layers(cache):
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def _expand(cache, batch: int):
    """Fresh copy of a batch-1 cache expanded to `batch`, safe to mutate."""
    new = DynamicCache()
    for li in range(_n_layers(cache)):
        k, v = _kv(cache, li)
        new.update(k.expand(batch, -1, -1, -1).contiguous(),
                   v.expand(batch, -1, -1, -1).contiguous(), li)
    return new


def forward_with_prefix(model, cache, prefix_len: int, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor, output_hidden_states: bool = False):
    """Run on [prefix; input_ids] reusing the prefix cache. Gradients flow.

    attention_mask: [B, T] over input_ids only. Positions continue after the prefix.
    """
    B, T = input_ids.shape
    past = _expand(cache, B)
    full_mask = torch.cat(
        [torch.ones(B, prefix_len, dtype=attention_mask.dtype, device=attention_mask.device),
         attention_mask], dim=1)
    position_ids = (torch.arange(T, device=input_ids.device).unsqueeze(0) + prefix_len)
    return model(input_ids=input_ids, attention_mask=full_mask,
                 position_ids=position_ids.expand(B, -1), past_key_values=past,
                 use_cache=False, output_hidden_states=output_hidden_states)


@torch.no_grad()
def teacher_forward(model, cache, prefix_len, input_ids, attention_mask,
                    output_hidden_states=False):
    return forward_with_prefix(model, cache, prefix_len, input_ids, attention_mask,
                               output_hidden_states)


@torch.no_grad()
def check_cache_equivalence(model, prefix_ids, input_ids, attention_mask, tol=1e-5):
    """Cached-prefix logits must match a plain full-sequence forward."""
    cache, P = build_prefix_cache(model, prefix_ids)
    cached = teacher_forward(model, cache, P, input_ids, attention_mask).logits

    B, T = input_ids.shape
    full_ids = torch.cat([prefix_ids.expand(B, -1), input_ids], dim=1)
    full_mask = torch.cat(
        [torch.ones(B, P, dtype=attention_mask.dtype, device=attention_mask.device),
         attention_mask], dim=1)
    full = model(input_ids=full_ids, attention_mask=full_mask).logits[:, P:, :]

    a, b = cached.float(), full.float()
    diff = (a - b).abs().max().item()
    lpa = torch.log_softmax(a, dim=-1)
    lpb = torch.log_softmax(b, dim=-1)
    kl = (lpb.exp() * (lpb - lpa)).sum(-1).mean().item()
    # KL is what the loss consumes; a large tail logit gap can still be harmless.
    # fp32 gives ~1e-9, bf16 ~9e-4 -- the latter is the same size as the effects
    # being measured, so it is rejected rather than tolerated.
    return abs(kl) < tol, diff, kl
