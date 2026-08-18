"""The three steering arms, and the dose scaling that makes them comparable.

Plan: plans/2026-08-17-manifold-steering-role-susceptibility.md (Method).

Every arm perturbs the residual stream at one layer, at every token position:

    h  <-  h + delta        with     ||delta|| = alpha * N_bar

`N_bar` is the mean response-token residual norm at that layer on the resp240
cloud. THE RESCALING IS THE POINT: without it, "arm X works better" collapses
into "arm X pushed harder". Arms differ only in the *direction* of delta.

    Arm 1  axis      global contrast vector, no target      (the paper, section 3.2.1)
    Arm 2  linear    straight line to the target centroid   (the honest baseline)
    Arm 3  manifold  along the fitted curve to the target   (ours)

Arm 1 runs on the vendored `addition` path unchanged; Arms 2 and 3 are
`dynamic`, because their direction depends on the current activation.

A note on Arm 3's intrinsic coordinate
--------------------------------------
The plan's Method section defines the coordinate twice and the two definitions
disagree: `u(x) = axis_proj(x)` (a dot product) in the formula block, and
"project(h) via the Gauss-Newton nearest-point solve" in the line below it.
This module implements the FIRST, `u(x) = x . a_hat`, because that is the
definitional one — the whole "no intrinsic dimension to choose" argument rests
on the coordinate being the externally-given Assistant-Axis projection, not
something recovered by a solver. The Gauss-Newton line is vestigial from the
earlier k-dimensional design that this plan replaced. Recorded under
`## Observations` in the plan and surfaced in the report; it is a deviation
from a literal reading of the plan, not a silent choice.
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-8


# --------------------------------------------------------------------------
# Dose
# --------------------------------------------------------------------------

def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """float64 dot with the platform's spurious FP warnings suppressed.

    numpy 2.0.2 on macOS Accelerate raises divide-by-zero/overflow/invalid for
    any matmul, operands notwithstanding (see steering/geometry.py::_matmul).
    The result is checked instead.
    """
    with np.errstate(all="ignore"):
        out = float(a @ b)
    if not np.isfinite(out):
        raise ValueError("non-finite dot product")
    return out


def unit(v: np.ndarray) -> np.ndarray:
    """L2-normalise, guarding the degenerate case."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("cannot normalise a zero-norm vector")
    return v / n


def _rescale_torch(direction: torch.Tensor, magnitude: float) -> torch.Tensor:
    """Rescale a (..., hidden) direction to exactly `magnitude` per position.

    Rows whose norm underflows are zeroed rather than blown up: at alpha = 0,
    or where the model already sits on the target, the correct delta is no
    delta, and dividing by ~0 there would inject noise of the requested
    magnitude in an arbitrary direction.
    """
    n = direction.norm(dim=-1, keepdim=True)
    safe = torch.where(n < EPS, torch.ones_like(n), n)
    out = direction / safe * magnitude
    return torch.where(n < EPS, torch.zeros_like(out), out)


# --------------------------------------------------------------------------
# Arm 1 — the paper's Assistant Axis (global, targetless)
# --------------------------------------------------------------------------

def arm1_axis_vector(axis_unit: np.ndarray, alpha: float, n_bar: float) -> np.ndarray:
    """delta = -alpha * N_bar * a_hat  (negative = away from the Assistant).

    A single fixed vector for every role, prompt and token, so this needs no
    callback — it goes straight into the vendored `addition` path, which is the
    authors' own code. That is what makes Arm 1 a replication rather than a
    reimplementation.

    Sign: `common.assistant_axis` returns mean(default) - mean(all roles), so
    positive projection = more Assistant-like. The paper steers *away* from the
    Assistant to raise role susceptibility (section 3.2.1), hence the minus.
    """
    return -alpha * n_bar * unit(axis_unit)


# --------------------------------------------------------------------------
# Arm 2 — linear to target
# --------------------------------------------------------------------------

def make_arm2_delta_fn(c_target: np.ndarray, alpha: float, n_bar: float,
                       dtype=torch.float32):
    """delta = alpha * N_bar * normalize(c_T - h), recomputed at every token.

    A difference-in-means steering vector, the persona-vectors family. It knows
    the target, which is what makes it the baseline Arm 3 has to beat — beating
    Arm 1, which has no target at all, would prove very little.
    """
    c_t = torch.as_tensor(np.asarray(c_target, dtype=np.float64), dtype=dtype)
    magnitude = float(alpha) * float(n_bar)

    def delta_fn(activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        c = c_t.to(device=activations.device, dtype=activations.dtype)
        return _rescale_torch(c - activations, magnitude)

    return delta_fn


# --------------------------------------------------------------------------
# Arm 3 — manifold path
# --------------------------------------------------------------------------

def make_arm3_delta_fn(spline, axis_unit: np.ndarray, c_target: np.ndarray,
                       alpha: float, n_bar: float, delta_frac: float = 0.25,
                       dtype=torch.float32):
    """Follow the fitted curve toward the target, then rescale to the dose.

        u_h  = h . a_hat                        # the intrinsic coordinate, by definition
        u_T  = c_T . a_hat
        r_T  = c_T - S(u_T)                     # the target's off-curve residual
        step = S(u_h + f*(u_T - u_h)) - S(u_h)  +  f * r_T
        delta = alpha * N_bar * normalize(step)

    `f` (delta_frac) is a fixed fractional step, so the direction is the curve's
    local secant toward the target rather than a chord to it — that is the whole
    difference from Arm 2. The residual term means the path still arrives at the
    actual role centroid rather than at a generic point on the curve.

    `f` is FIXED at 0.25 and not tuned: tuning it against an outcome we have
    already looked at is what an exploratory run must not do (plan, Method).

    Args:
        spline: a fitted `steering.spline1d.CubicSpline1D` over the 276 role
            centroids, keyed by axis projection.
        axis_unit: the unit Assistant Axis, the intrinsic coordinate's direction.
        c_target: the target role's centroid in ambient space.
    """
    a_hat = unit(axis_unit)
    c_t = np.asarray(c_target, dtype=np.float64)
    u_t = _dot(c_t, a_hat)
    r_t = c_t - spline.evaluate(np.array([u_t]))[0]     # off-curve residual

    a_t = torch.as_tensor(a_hat, dtype=dtype)
    r_torch = torch.as_tensor(r_t, dtype=dtype)
    magnitude = float(alpha) * float(n_bar)
    f = float(delta_frac)

    def delta_fn(activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        dev, dt = activations.device, activations.dtype
        a = a_t.to(device=dev, dtype=dt)

        # Intrinsic coordinate of every position: u_h = h . a_hat.
        u_h = (activations * a).sum(-1)                       # (b, l)

        flat = u_h.reshape(-1).detach().to("cpu").numpy().astype(np.float64)
        u_step = flat + f * (u_t - flat)
        step_np = spline.evaluate(u_step) - spline.evaluate(flat)   # (b*l, hidden)

        step = torch.as_tensor(step_np, dtype=dt, device=dev).reshape(activations.shape)
        step = step + f * r_torch.to(device=dev, dtype=dt)
        return _rescale_torch(step, magnitude)

    return delta_fn


# --------------------------------------------------------------------------
# Negative control — a random direction at matched norm
# --------------------------------------------------------------------------

def random_direction(hidden: int, seed: int) -> np.ndarray:
    """A seeded random unit vector in residual space.

    The negative control (plan, Controls). If a random push at matched norm
    moves persona expression as much as a steering vector does, then "steering
    works" and "any perturbation degrades the model into role-play" are not
    distinguishable, and every figure has to say so.
    """
    rng = np.random.default_rng(seed)
    return unit(rng.normal(size=hidden))
