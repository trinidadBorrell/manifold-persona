"""Paper-faithful REPLACEMENT steering: h := pi(t) at the last position (plan exp4).

Plan: plans/2026-09-24-steering-followups.md (Background; Design, exp4).
Paper: Manifold Steering, arXiv 2605.05115, Sec. 3.1 Eq. 1 and App. A.6.

WHY THIS EXISTS. Every arm in `steering/interventions.py` is ADDITIVE,
h <- h + [S(alpha) - S(0)], at every token. That lands on c_B only if h started
at c_A, and it does not: unsteered, the model sits ~40 units off the
validator->vampire chord (exp1). The paper does something else. It REPLACES the
activation with the path point

    linear    pi(t) = (1 - t) c_A + t c_B
    manifold  pi(t) = the spline point at normalised arc position t

at the LAST TOKEN ONLY, so t = 0 is exactly the source centroid and t = 1
exactly the target, whatever the prompt's own activation was. This module is
that write, and nothing else.

WHERE "LAST POSITION" IS, UNDER generate() WITH A KV CACHE. The hook fires once
per forward call of the hooked decoder layer:

    prefill   input is the whole prompt, (B, n_prompt, d); position -1 is the
              last prompt token -- the state that produces the first response
              token.
    decode    input is the single new token, (B, 1, d); position -1 is it.

So "replace position -1 on every call" writes pi(t) at the last prompt position
and at every response position the model ever feeds back -- which is the paper's
"last token of every forward step". Earlier positions are never touched, and the
cache entries later layers build from the replaced slot carry the replacement
forward, exactly as they would in the paper's loop without a cache.

REPLAY. The steered state is read, as in jobs_condor/dose_escalation.py, by
re-running the finished sequence once with the hook live. A single uncached
pass sees ALL positions at once, so "position -1" would replace only the final
token and the replay would not reproduce the state generation was in. Pass
`from_position = n_prompt - 1` for the replay: every position from the last
prompt token onward is then replaced, which is exactly the set generation
replaced (causality: position p depends only on positions <= p, all of which
were written identically). Generation itself must use the default
(`from_position=None`).

EXACT, NOT ADDITIVE. The slot is ASSIGNED the target cast to the activation
dtype, so in fp16 the written value is `target.half()` bit for bit. The
alternative -- an ActivationSteering "dynamic" delta of (target - h) at the
last position, provided below for composition -- computes h + (target - h) in
fp16, which is not target: two roundings, error ~ulp(|h|). The plan's
replacement-sanity control (||h - c_B|| < 1e-2 at t = 1) must be met by the
write itself, so the direct hook is the default. What fp16 CANNOT do is
represent c_B exactly; `fp16_representation_error` reports that floor so the
control is read against it.
"""
from __future__ import annotations

import types

import numpy as np
import torch

from steering.activation_steering import ActivationSteering


def resolve_layer(model, layer_idx: int):
    """The decoder layer module, found the way ActivationSteering finds it.

    Calls ActivationSteering's own locator on a duck-typed stand-in rather than
    copying its attribute list, so a model family added there is supported here
    with no edit.
    """
    shim = types.SimpleNamespace(model=model,
                                 _POSSIBLE_LAYER_ATTRS=ActivationSteering._POSSIBLE_LAYER_ATTRS)
    layers, _ = ActivationSteering._locate_layer_list(shim)
    if not (-len(layers) <= layer_idx < len(layers)):
        raise IndexError("layer_idx %d out of range for %d layers" % (layer_idx, len(layers)))
    return layers[layer_idx]


def _split(out):
    if torch.is_tensor(out):
        return out, None
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0], out
    raise TypeError("layer output is neither a tensor nor a tuple led by one: %s" % type(out))


def _join(t, packed):
    return t if packed is None else (t, *packed[1:])


class ReplaceLastPosition:
    """Context manager: forward hook on `model.layers[layer_idx]` writing `target`.

    Args:
        model: any model ActivationSteering can locate layers in.
        layer_idx: DECODER-layer index (use hook_layer_for_hidden_state(19) = 18).
        target: (d,) vector, numpy or torch. Cast to the activation's dtype and
            device at write time.
        from_position: None (generation: position -1 of every call) or an int
            p (replay: positions p.. of the single uncached call).

    `n_writes` counts hook calls that wrote, so a driver can assert the hook
    actually fired (prefill + one per decoded token).
    """

    def __init__(self, model, layer_idx: int, target, from_position=None, basis=None):
        # `basis` (d, r), orthonormal columns: write only the component of the
        # activation inside span(basis), keeping the rest -- h + U U^T (target - h).
        # That is the paper's own form (arXiv 2605.05115 App. A.6: the top-64 PCA
        # coordinates are assigned, the off-subspace residual is kept). Without it
        # the WHOLE activation is overwritten at every decode step, which erases the
        # current token's identity: the 2026-09-24 GPU smoke produced '.\n\n., and in,'
        # at t = 0 and t = 1 alike. basis=None keeps the full-space write.
        self.basis = (None if basis is None else
                      torch.as_tensor(np.asarray(basis, dtype=np.float32)))
        t = torch.as_tensor(np.asarray(target, dtype=np.float64)
                            if not torch.is_tensor(target) else target)
        if t.ndim != 1:
            raise ValueError("target must be 1-D, got shape %s" % (tuple(t.shape),))
        if not torch.isfinite(t).all():
            raise ValueError("target is not finite")
        self.model, self.layer_idx = model, int(layer_idx)
        self.target = t
        self.from_position = from_position
        self.n_writes = 0
        self._handle = None

    def _hook(self, module, ins, out):
        h, packed = _split(out)
        if h.shape[-1] != self.target.shape[0]:
            raise ValueError("target has %d dims but the layer emits %d"
                             % (self.target.shape[0], h.shape[-1]))
        h = h.clone()                     # never write into a tensor autograd or
        sl = (slice(-1, None) if self.from_position is None   # another hook may still hold
              else slice(int(self.from_position), None))
        if self.basis is None:
            h[..., sl, :] = self.target.to(device=h.device, dtype=h.dtype)
        else:
            U = self.basis.to(device=h.device)                          # fp32
            cur = h[..., sl, :].float()
            diff = self.target.to(device=h.device, dtype=torch.float32) - cur
            h[..., sl, :] = (cur + (diff @ U) @ U.T).to(h.dtype)
        self.n_writes += 1
        return _join(h, packed)

    def __enter__(self):
        self._handle = resolve_layer(self.model, self.layer_idx).register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class CaptureLayer:
    """Record the output of `model.layers[layer_idx]` AFTER every earlier hook.

    PyTorch runs forward hooks in registration order and hands each one the
    previous hook's return value, so a capture registered after a steering hook
    sees the STEERED state. This is used instead of `output_hidden_states`
    because whether transformers' own hidden-state recorder runs before or after
    a user hook is a version detail; registration order is not.

    `outputs` holds one float32 CPU tensor per forward call.
    """

    def __init__(self, model, layer_idx: int):
        self.model, self.layer_idx = model, int(layer_idx)
        self.outputs = []
        self._handle = None

    def _hook(self, module, ins, out):
        h, _ = _split(out)
        self.outputs.append(h.detach().float().cpu())
        return None

    def __enter__(self):
        self._handle = resolve_layer(self.model, self.layer_idx).register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def make_replace_delta_fn(target, dtype=torch.float32):
    """The same write as an ActivationSteering 'dynamic' delta_fn: (target - h) at -1.

    For COMPOSITION only (e.g. stacking with another dynamic arm in one
    ActivationSteering context). Not exact in half precision -- see the module
    docstring -- so the follow-up driver uses ReplaceLastPosition.
    """
    t = torch.as_tensor(np.asarray(target, dtype=np.float64), dtype=dtype)

    def delta_fn(activations, layer_idx):
        d = torch.zeros_like(activations)
        d[..., -1, :] = t.to(device=activations.device, dtype=activations.dtype) - activations[..., -1, :]
        return d

    return delta_fn


def linear_target(cA, cB, t: float) -> np.ndarray:
    """pi(t) = (1 - t) c_A + t c_B, t in [0, 1] (the paper's Eq. 1)."""
    t = float(t)
    if not 0.0 <= t <= 1.0:
        raise ValueError("replacement t must be in [0, 1], got %r" % t)
    cA = np.asarray(cA, dtype=np.float64)
    cB = np.asarray(cB, dtype=np.float64)
    # written as cA + t*(cB - cA) would miss c_B by rounding at t = 1; this
    # form returns c_A at t = 0 and c_B at t = 1 exactly
    return (1.0 - t) * cA + t * cB


def manifold_target(path, t: float) -> np.ndarray:
    """The path point at normalised arc position t in [0, 1] (never outside)."""
    t = float(t)
    if not 0.0 <= t <= 1.0:
        raise ValueError("replacement t must be in [0, 1], got %r" % t)
    return np.asarray(path.at_alpha(t), dtype=np.float64).reshape(-1)


def fp16_representation_error(x) -> float:
    """||x - fp16(x)||: the floor under any fp16 'state equals x' check."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.linalg.norm(x - x.astype(np.float16).astype(np.float64)))


def pca_basis(C, r: int = 64) -> np.ndarray:
    """Top-r principal directions of the role centroids, (d, r) orthonormal columns.

    The paper's manifold arm lives in the top-64 PCA subspace of its class centroids
    (App. A.3, A.6); this is the same construction on our 275 role centroids.
    """
    C = np.asarray(C, dtype=np.float64)
    _, _, Vt = np.linalg.svd(C - C.mean(0), full_matrices=False)
    return Vt[:r].T.copy()
