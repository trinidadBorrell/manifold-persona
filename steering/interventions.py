"""The steering arms, and the dose scaling that makes them comparable.

Plan: plans/2026-08-17-manifold-steering-role-susceptibility.md (Method);
current design in docs/notes/plan-new-run.md.

Every arm perturbs the residual stream at one layer, at every token position:

    h  <-  h + delta

`N_bar` is the mean per-token response residual norm at that layer. Arms come in
straight/curved pairs, and each pair holds one thing fixed so the only remaining
difference is the ROUTE:

  dose-matched   ||delta|| = |alpha| * N_bar, sign(alpha) the sense of travel
    linear_axis / manifold_axis        along the Assistant Axis, NO target
    linear_target / manifold_target    toward a target centroid
  destination-matched   delta lands f of the way to the target (f = 1 lands ON it)
    linear_journey / manifold_journey
  displacement-matched  delta = f * (c_T - c_S), constant, never references h
    linear_contrast / manifold_contrast
  route-matched  delta = S(alpha) - S(0) along a path with FIXED ENDPOINTS
    linear_axis / manifold_axis        the chord across the Assistant Axis
    linear_pair / manifold_pair        the chord between two persona centroids

THE ROUTE-MATCHED ARMS are the current design (steering/manifold_paths.py). Both
members of a pair start at P0 and end at P1 and are sampled at the same
normalised arc positions, so at alpha = 0 both deltas are exactly zero and at
alpha = 1 both are exactly P1 - P0; everything between them is the route and
nothing else. The intervention is the paper's own, `h <- h + alpha*vector`
(Figure 4), which is why the path enters as a DISPLACEMENT and never as a
position: `make_linear_contrast_delta_fn` records what replacing h with a
centroid-like point did to the generations.

THE MATCHING IS THE POINT: without it, "arm X works better" collapses into "arm
X pushed harder" (or "arm X went further"). What each pair gives up to hold its
own quantity fixed is measured per cell by `DeltaStats`, not assumed.

`linear_axis_legacy` runs on the vendored `addition` path unchanged; every other
arm is `dynamic`, because its direction (or its bookkeeping) depends on the
activation -- or, for the constant-delta arms, simply because that is the hook
that takes a callback.

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


def _axis_coord(activations: torch.Tensor, a: torch.Tensor) -> np.ndarray:
    """u_h = h . a_hat for every position, as a flat float64 numpy array.

    The spline is numpy and float64; the activations are torch and usually
    fp16 on device. This is the one crossing, written once because all three
    curve-following arms make it identically and it is the most delicate step
    in the module.
    """
    u_h = (activations * a).sum(-1)                       # (b, l)
    return u_h.reshape(-1).detach().to("cpu").numpy().astype(np.float64)


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

def linear_axis_vector(axis_unit: np.ndarray, alpha: float, n_bar: float) -> np.ndarray:
    """delta = alpha * N_bar * a_hat, with alpha SIGNED as in the paper.

    THE `linear_axis_legacy` ARM. Kept, not deleted: it is the dose-matched
    form the earlier runs used, and their shards are only readable against it.
    The current `linear_axis` arm is `make_linear_path_delta_fn` over a
    `LinearPath` across the axis segment -- same straight line, but travelled
    to a fixed endpoint rather than pushed by a dose.

    ALPHA IS NOW THE PAPER'S X-AXIS, not a magnitude. `common.assistant_axis`
    returns mean(default) - mean(role vectors), so +a_hat points toward the
    Assistant. Therefore:

        alpha < 0   away from the Assistant   (the paper's -1.0 .. 0 sweep)
        alpha > 0   toward the Assistant      (their +0.25 end)

    The old form took a magnitude and hard-coded a minus, which made "toward"
    unreachable and meant our x-axis had to be negated at plotting time. Signing
    alpha here means the number in the manifest, the number in the filename and
    the number on the figure are all the same number.

    A single fixed vector for every role, prompt and token, so this needs no
    callback — it goes straight into the vendored `addition` path, which is the
    authors' own code. That is what makes this a replication rather than a
    reimplementation.
    """
    return alpha * n_bar * unit(axis_unit)


# --------------------------------------------------------------------------
# Arm 2 — linear to target
# --------------------------------------------------------------------------

class DeltaStats:
    """Running record of what an intervention actually did, per cell.

    Both numbers exist because the arm's DESCRIPTION can be true of the code and
    false of the run (plan WP5):

      * `overshoot` = dose / ||c_T - h||. The role manifold spans about 14 along
        the axis while alpha=1 pushes 48, so at the top of the grid "toward the
        target" stops describing the intervention. Unmeasured, that reads as a
        property of the geometry instead of a property of the dose.
      * `frac_out_of_knots` (Arm 3 only) = fraction of token coordinates outside
        the fitted curve's knot range, where the spline is a straight line. Every
        such token is Arm 3 behaving as a linear arm, and the whole comparison
        rests on it not being one.

    THE TWO COUNTERS ARE SEPARATE, and they have to be. `add_overshoot` needs a
    target centroid, so only the target-aware arms call it; `add_coords` needs
    only the curve, so the targetless `manifold_axis` arm calls that one alone.
    A single shared `n` made `report()` return `{"n_positions": 0}` for every
    manifold cell and silently discard the out-of-knots fraction — the one
    number that says whether the manifold arm degenerated into a linear one.
    """

    def __init__(self):
        self.n_overshoot = 0
        self.n_coords = 0
        self.n_mag = 0
        self.mag_sum = 0.0
        self.mag_max = 0.0
        self.overshoot_sum = 0.0
        self.overshoot_max = 0.0
        self.out_of_knots = 0
        self.u_min = float("inf")
        self.u_max = float("-inf")

    def add_overshoot(self, ratios: torch.Tensor) -> None:
        r = ratios.detach().float()
        self.n_overshoot += int(r.numel())
        self.overshoot_sum += float(r.sum())
        self.overshoot_max = max(self.overshoot_max, float(r.max()))

    def add_magnitude(self, norms: torch.Tensor) -> None:
        """||delta|| per position. Fixed by construction in the dose-matched
        arms, and NOT fixed in the journey arms, where it is the thing that had
        to be given up to hold the destination constant."""
        n = norms.detach().float()
        self.n_mag += int(n.numel())
        self.mag_sum += float(n.sum())
        self.mag_max = max(self.mag_max, float(n.max()))

    def add_constant_magnitude(self, norm: float, count: int) -> None:
        """||delta|| for an arm whose delta is one constant vector.

        The contrast arms add the same vector at every position, so reducing
        `out.norm(dim=-1)` over a (batch, len, hidden) tensor on every decode
        step recomputes a number that was fixed when the closure was built —
        and does it with a blocking device->host sync. This records the same
        sum and max from the value itself.
        """
        self.n_mag += int(count)
        self.mag_sum += float(norm) * int(count)
        self.mag_max = max(self.mag_max, float(norm))

    def add_coords(self, u: np.ndarray, lo: float, hi: float) -> None:
        self.n_coords += int(u.size)
        self.out_of_knots += int(((u < lo) | (u > hi)).sum())
        if u.size:
            self.u_min = min(self.u_min, float(u.min()))
            self.u_max = max(self.u_max, float(u.max()))

    def report(self) -> dict:
        """What was measured, with each fraction over ITS OWN denominator."""
        if not self.n_overshoot and not self.n_coords and not self.n_mag:
            return {"n_positions": 0}
        out = {"n_positions": max(self.n_overshoot, self.n_coords, self.n_mag)}
        if self.n_overshoot:
            out.update({
                "n_overshoot": self.n_overshoot,
                "overshoot_mean": self.overshoot_sum / self.n_overshoot,
                "overshoot_max": self.overshoot_max,
            })
        if self.n_mag:
            out.update({"n_mag": self.n_mag,
                        "delta_norm_mean": self.mag_sum / self.n_mag,
                        "delta_norm_max": self.mag_max})
        if self.n_coords:
            out.update({
                "n_coords": self.n_coords,
                "frac_out_of_knots": self.out_of_knots / self.n_coords,
                "u_observed_min": self.u_min,
                "u_observed_max": self.u_max,
            })
        return out


def make_arm2_delta_fn(c_target: np.ndarray, alpha: float, n_bar: float,
                       dtype=torch.float32, stats: "DeltaStats" = None):
    """delta = alpha * N_bar * normalize(c_T - h), recomputed at every token.

    A difference-in-means steering vector, the persona-vectors family. It knows
    the target, which is what makes it the baseline Arm 3 has to beat — beating
    Arm 1, which has no target at all, would prove very little.
    """
    c_t = torch.as_tensor(np.asarray(c_target, dtype=np.float64), dtype=dtype)
    # SIGN IS EXPLICIT. The direction (c_T - h) already points AT the target, so
    # a negative `magnitude` silently reverses it. That is what happened when
    # alpha became signed for the axis arms: the ablation swept alpha -0.125 ..
    # -1.0 and therefore steered every role directly AWAY from the target it was
    # supposed to reach, which is why no target persona ever appeared.
    # |alpha| sets how hard, sign(alpha) sets toward (+) or away (-).
    magnitude = abs(float(alpha)) * float(n_bar)
    toward = 1.0 if float(alpha) >= 0 else -1.0

    def delta_fn(activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        c = c_t.to(device=activations.device, dtype=activations.dtype)
        gap = (c - activations) * toward
        if stats is not None:
            d = gap.norm(dim=-1)
            stats.add_overshoot(magnitude / torch.clamp(d, min=EPS))
        return _rescale_torch(gap, magnitude)

    return delta_fn


# --------------------------------------------------------------------------
# Arm 3 — manifold path
# --------------------------------------------------------------------------

def make_arm3_delta_fn(spline, axis_unit: np.ndarray, c_target: np.ndarray,
                       alpha: float, n_bar: float, delta_frac: float = 0.25,
                       dtype=torch.float32, stats: "DeltaStats" = None):
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
    c_torch = torch.as_tensor(c_t, dtype=dtype)
    # Same sign fix as the linear target arm: the step is built to arrive AT the
    # target, so |alpha| is the dose and sign(alpha) is the sense of travel.
    magnitude = abs(float(alpha)) * float(n_bar)
    toward = 1.0 if float(alpha) >= 0 else -1.0
    f = float(delta_frac)

    def delta_fn(activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        dev, dt = activations.device, activations.dtype
        a = a_t.to(device=dev, dtype=dt)

        # Intrinsic coordinate of every position: u_h = h . a_hat.
        flat = _axis_coord(activations, a)
        u_step = flat + f * (u_t - flat)
        step_np = spline.evaluate(u_step) - spline.evaluate(flat)   # (b*l, hidden)

        step = torch.as_tensor(step_np, dtype=dt, device=dev).reshape(activations.shape)
        step = (step + f * r_torch.to(device=dev, dtype=dt)) * toward

        if stats is not None:
            # Outside the knots the spline is a straight line, so those tokens
            # are not on the manifold in any sense — count them.
            stats.add_coords(flat, float(spline.x[0]), float(spline.x[-1]))
            c = c_torch.to(device=dev, dtype=dt)
            stats.add_overshoot(
                magnitude / torch.clamp((c - activations).norm(dim=-1), min=EPS))

        return _rescale_torch(step, magnitude)

    return delta_fn


# --------------------------------------------------------------------------
# Manifold along the axis — the same journey as the linear arm, curved
# --------------------------------------------------------------------------

def make_manifold_axis_delta_fn(spline, axis_unit: np.ndarray, alpha: float,
                                n_bar: float, span: float,
                                step_frac: float = 0.25,
                                dtype=torch.float32, stats: "DeltaStats" = None):
    """THE `manifold_axis_legacy` ARM. Follow the fitted curve along the
    Assistant Axis. NO TARGET ROLE.

    Superseded by `make_manifold_path_delta_fn` over a `PersonaPath`, which
    fixes both endpoints instead of taking a local secant of fixed length. Kept
    so the runs that used it stay interpretable.

        u_h   = h . a_hat                       # where we are on the curve
        du    = sign(alpha) * step_frac * span  # a step along the coordinate
        step  = S(u_h + du) - S(u_h)            # the curve's local secant
        delta = alpha * N_bar * normalize(step)

    This is the linear arm's question asked of the manifold: the linear arm
    travels along `a_hat`, a straight line that ignores where roles actually
    sit; this one travels the same direction along the *curve* through the role
    vectors. Same dose, same sense of travel, different path — which is the only
    difference the comparison is allowed to have.

    There is no target centroid, so there is no near/far: the paper's
    intervention has no target either, and this is its curved counterpart.

    `span` is the range of the curve's coordinate (max - min axis projection of
    the role vectors). `step_frac` is a FIXED 0.25 of it — the direction is a
    local secant, and tuning it against an outcome we have already seen is what
    an exploratory run must not do.
    """
    a_hat = unit(axis_unit)
    a_t = torch.as_tensor(a_hat, dtype=dtype)
    magnitude = abs(float(alpha)) * float(n_bar)
    du = float(np.sign(alpha)) * float(step_frac) * float(span)

    def delta_fn(activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        dev, dt = activations.device, activations.dtype
        a = a_t.to(device=dev, dtype=dt)

        flat = _axis_coord(activations, a)
        step_np = spline.evaluate(flat + du) - spline.evaluate(flat)

        step = torch.as_tensor(step_np, dtype=dt, device=dev).reshape(activations.shape)
        if stats is not None:
            stats.add_coords(flat, float(spline.x[0]), float(spline.x[-1]))
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


# --------------------------------------------------------------------------
# Journey arms: travel a FRACTION OF THE WAY to the target, not a fixed dose
# --------------------------------------------------------------------------
#
# WHY THESE EXIST. The dose-matched arms fix ||delta|| = |alpha| * N_bar and let
# the two paths end wherever that lands them, which is a proxy for fairness: the
# push is equally hard, so a difference "must" be about direction. But the two
# arms then finish in different places, and at high alpha both overshoot the
# target by 2.24x, so the comparison is between two different destinations.
#
# These arms match the DESTINATION instead. `f` is the fraction of the journey
# from the current activation to the target completed at every token, and
#
#     at f = 1 BOTH arms land on exactly c_T.
#
# That is the control the dose-matched design never had: if the arms differ at
# f = 1 something is wrong, and any difference at intermediate f is the route
# and nothing else.
#
# The cost is that ||delta|| is no longer equal between arms. That is deliberate
# (the destination is what is being held fixed now), but it is measured per cell
# and reported, because "the manifold arm pushed less hard" is otherwise an
# invisible confound.

def make_linear_journey_delta_fn(c_target: np.ndarray, frac: float,
                                 dtype=torch.float32, stats: "DeltaStats" = None):
    """delta = f * (c_T - h): a straight line, f of the way there, every token.

    At f = 1 this replaces the activation with the target centroid outright.
    """
    c_t = torch.as_tensor(np.asarray(c_target, dtype=np.float64), dtype=dtype)
    f = float(frac)

    def delta_fn(activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        c = c_t.to(device=activations.device, dtype=activations.dtype)
        d = f * (c - activations)
        if stats is not None:
            stats.add_magnitude(d.norm(dim=-1))
        return d

    return delta_fn


def make_manifold_journey_delta_fn(spline, axis_unit: np.ndarray,
                                   c_target: np.ndarray, frac: float,
                                   dtype=torch.float32, stats: "DeltaStats" = None):
    """Same journey, along the fitted curve instead of through empty space.

        u_h  = h . a_hat
        u_T  = c_T . a_hat
        r_T  = c_T - S(u_T)                        the target's off-curve residual
        r_h  = h   - S(u_h)                        THIS activation's own residual
        delta = [ S(u_h + f*(u_T - u_h)) - S(u_h) ] + f * (r_T - r_h)

    At f = 1 the bracket is S(u_T) - S(u_h) and the rest is r_T - h + S(u_h), so
    h + delta = S(u_T) + r_T = c_T exactly, the same endpoint as the linear arm.
    That identity is the point of the whole design: both arms are pinned to one
    destination, so anything that differs between them is the ROUTE.

    Interpolating r_h as well as r_T is what buys it. An earlier version carried
    only f*r_T, which leaves the source's own offset in place and lands on
    c_T + r_h: measured at 7.82 off a 29.52 chord, a 26% miss that would have
    quietly destroyed the control.
    """
    a_hat = unit(axis_unit)
    c_t = np.asarray(c_target, dtype=np.float64)
    u_t = _dot(c_t, a_hat)
    r_t = c_t - spline.evaluate(np.array([u_t]))[0]

    a_t = torch.as_tensor(a_hat, dtype=dtype)
    r_torch = torch.as_tensor(r_t, dtype=dtype)
    f = float(frac)

    def delta_fn(activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        dev, dt = activations.device, activations.dtype
        a = a_t.to(device=dev, dtype=dt)
        flat = _axis_coord(activations, a)
        S_h = spline.evaluate(flat)
        step = spline.evaluate(flat + f * (u_t - flat)) - S_h
        d = torch.as_tensor(step, dtype=dt, device=dev).reshape(activations.shape)
        # BOTH residuals, not just the target's: r_h = h - S(u_h) is how far this
        # activation sits off the fitted curve, and it has to be removed on the
        # way in or f = 1 lands on c_T + r_h instead of c_T.
        r_h = activations - torch.as_tensor(S_h, dtype=dt,
                                            device=dev).reshape(activations.shape)
        d = d + f * (r_torch.to(device=dev, dtype=dt) - r_h)
        if stats is not None:
            stats.add_coords(flat, float(spline.x[0]), float(spline.x[-1]))
            stats.add_magnitude(d.norm(dim=-1))
        return d

    return delta_fn


def _constant_delta_fn(d_np: np.ndarray, dtype, stats: "DeltaStats" = None):
    """A hook that adds ONE fixed vector at every position.

    Shared by both contrast arms, which differ only in how `d_np` is built —
    keeping one closure means the only code path that touches the hook cannot
    drift between them. ||delta|| is a property of `d_np`, so it is measured
    once here rather than reduced over the activation tensor per decode step.
    """
    d = torch.as_tensor(d_np, dtype=dtype)
    d_norm = float(np.linalg.norm(d_np))

    def delta_fn(activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = d.to(device=activations.device, dtype=activations.dtype)
        out = out.expand_as(activations)
        if stats is not None:
            stats.add_constant_magnitude(d_norm, out.numel() // out.shape[-1])
        return out

    return delta_fn


def make_linear_contrast_delta_fn(c_source: np.ndarray, c_target: np.ndarray,
                                  frac: float, dtype=torch.float32,
                                  stats: "DeltaStats" = None):
    """delta = f * (c_T - c_S). A DISPLACEMENT, not a destination.

    The journey arms aimed at a POINT: delta = f*(c_T - h). Because that
    references h, it pulls every token of every response toward one location,
    and at f = 1 it puts them all exactly there. The within-cell spread of the
    landings measured 16.05 unsteered and 3.79 at f = 1: a 4x collapse. What
    collapses is not only persona, it is the question- and position-specific
    structure the model needs to carry a sentence forward, which is why f = 1
    emits "the the the" forever.

    A role centroid is a MEAN over prompts, questions and rollouts. No real
    forward pass sits at a mean, so making one the destination is asking the
    model to be in a state it is never in.

    This form never references h. It is one constant vector added everywhere, so
    the whole cloud translates rigidly: each response keeps its own offset from
    the source centroid and only the persona mean moves. It is also the paper's
    own construction generalised, since the Assistant Axis is itself a
    difference of centroids, normalize(mean(default) - mean(roles)).
    """
    c_s = np.asarray(c_source, dtype=np.float64)
    c_t = np.asarray(c_target, dtype=np.float64)
    return _constant_delta_fn(float(frac) * (c_t - c_s), dtype, stats)


def make_manifold_contrast_delta_fn(spline, axis_unit: np.ndarray,
                                    c_source: np.ndarray, c_target: np.ndarray,
                                    frac: float, dtype=torch.float32,
                                    stats: "DeltaStats" = None):
    """The same displacement, but routed along the fitted curve.

        u_S, u_T = c_S . a_hat, c_T . a_hat
        r_S, r_T = c_S - S(u_S), c_T - S(u_T)
        delta = [ S(u_S + f*(u_T - u_S)) - S(u_S) ] + f * (r_T - r_S)

    Constant per f, like the linear form, so it preserves within-cell structure
    for the same reason. At f = 1 it collapses to S(u_T) + r_T - S(u_S) - r_S =
    c_T - c_S, exactly the linear displacement: the two arms coincide at the
    endpoint, which keeps the self-check the journey design bought us. Between 0
    and 1 they differ, and that difference is the route.
    """
    a_hat = unit(axis_unit)
    c_s = np.asarray(c_source, dtype=np.float64)
    c_t = np.asarray(c_target, dtype=np.float64)
    u_s, u_t = _dot(c_s, a_hat), _dot(c_t, a_hat)
    S_s = spline.evaluate(np.array([u_s]))[0]
    r_s = c_s - S_s
    r_t = c_t - spline.evaluate(np.array([u_t]))[0]
    f = float(frac)
    u_end = u_s + f * (u_t - u_s)
    step = spline.evaluate(np.array([u_end]))[0] - S_s
    if stats is not None:
        # RECORDED HERE, at construction. Both ends of this arm's chord are
        # fixed when the closure is built, and `f` runs to 3.0 — well past the
        # target centroid. Outside the knots the spline is a straight line, so
        # an out-of-knots endpoint means this arm has silently become the linear
        # one, which is the single thing frac_out_of_knots exists to catch.
        stats.add_coords(np.array([u_s, u_end]),
                         float(spline.x[0]), float(spline.x[-1]))
    return _constant_delta_fn(step + f * (r_t - r_s), dtype, stats)


# --------------------------------------------------------------------------
# The path arms — one route, two ways round, same two endpoints
# --------------------------------------------------------------------------
#
# THE CONSTRUCTION lives in `steering/manifold_paths.py`; these two factories
# only turn a path into a hook. Given a path with endpoints P0, P1:
#
#     delta(alpha) = S(alpha) - S(0)
#
# where alpha in [0, 1] is normalised ARC POSITION. The intervention is the
# paper's, ADDITIVE and nothing else:  h <- h + delta.  Figure 4 adds a vector,
# so a replication adds a vector; there is no replacement, no projection and no
# per-token solve here. `delta` is one vector for the whole cell, so the cloud
# translates rigidly and each response keeps its own offset from the source --
# the property `make_linear_contrast_delta_fn` explains at length, and the
# reason the journey arms are not what these are built on.
#
# THE THREE CONTROLS ARE STRUCTURAL, not asserted:
#   alpha = 0   both arms give exactly the zero vector (S(0) - S(0))
#   alpha = 1   both arms give exactly P1 - P0
#   linear      delta is exactly alpha*(P1 - P0), by LinearPath.at_alpha
# so any difference between the arms lives strictly inside 0 < alpha < 1 and is
# the route. Nothing needs to be matched by hand because nothing was free.


def _path_delta_fn(path, alpha: float, dtype, stats: "DeltaStats" = None):
    """Shared body of both path arms: the displacement, as a constant hook.

    One closure for both, for the reason `_constant_delta_fn` gives: the arms
    differ only in the object handed in, so the code path that reaches the hook
    must not be able to differ at all.
    """
    d = np.asarray(path.delta(float(alpha)), dtype=np.float64).reshape(-1)
    if not np.isfinite(d).all():
        raise ValueError("path delta at alpha=%r is not finite" % (alpha,))
    return _constant_delta_fn(d, dtype, stats)


def make_linear_path_delta_fn(path, alpha: float, dtype=torch.float32,
                              stats: "DeltaStats" = None):
    """delta = alpha * (P1 - P0): the straight chord, as a displacement.

    THE `linear_axis` AND `linear_pair` ARMS -- the same factory for both,
    because the arms differ only in which two points the chord joins (the two
    ends of the Assistant Axis segment, or two persona centroids). Which chord
    is a property of the `LinearPath`, so it is chosen once, in the shared case
    list (`steering.path_cases`), where the figures choose it too.

    The type is CHECKED. `make_manifold_path_delta_fn` takes the same shape of
    argument and returns the same shape of hook, so passing the wrong path
    object produces a run that is labelled one arm and is the other -- with no
    symptom anywhere, since both produce plausible text.

    Args:
        path: a `manifold_paths.LinearPath`.
        alpha: normalised arc position in [0, 1]. 0 is a no-op by construction.
    """
    from steering.manifold_paths import LinearPath
    if not isinstance(path, LinearPath):
        raise TypeError("linear path arm needs a LinearPath, got %s"
                        % type(path).__name__)
    return _path_delta_fn(path, alpha, dtype, stats)


def make_manifold_path_delta_fn(path, alpha: float, dtype=torch.float32,
                                stats: "DeltaStats" = None):
    """The same journey, routed through real persona centroids.

    THE `manifold_axis` AND `manifold_pair` ARMS. `PersonaPath` fits a cubic
    through the k persona centroids inside an eps-cylinder around the chord,
    with both endpoints pinned exactly, then samples it by arc length -- so this
    arm visits states the model actually produces on the way to the same place
    the linear arm reaches, and arrives there at the same alpha.

    `delta` is one constant vector per (arm, alpha) cell, exactly as in the
    linear arm: the ROUTE differs, the FORM of the intervention does not. An
    arm whose delta were per-token would differ from its partner in two ways at
    once and the comparison would say nothing about routes.

    Args:
        path: a `manifold_paths.PersonaPath` over the SAME endpoints as the
            linear arm it is compared with.
        alpha: normalised arc position in [0, 1].
    """
    from steering.manifold_paths import PersonaPath
    if not isinstance(path, PersonaPath):
        raise TypeError("manifold path arm needs a PersonaPath, got %s"
                        % type(path).__name__)
    return _path_delta_fn(path, alpha, dtype, stats)
