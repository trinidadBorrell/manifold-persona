"""Smoke test: prove the plumbing before spending GPU hours.

Plan: plans/2026-08-17-manifold-steering-role-susceptibility.md (Outputs -> Smoke test).

Checks, in order, each of which has failed silently in some pipeline somewhere:

  1. spline port          matches scipy's independent natural cubic, and
                          interpolates its control points exactly at lam = 0
  1b. path construction   five pure-numpy checks on `steering.manifold_paths`:
                          endpoints pinned at every lam/param (positive control),
                          eps=0 collapsing onto the chord (negative control),
                          alpha being ARC position and not the knot coordinate,
                          the curve actually visiting the persona centroids, and
                          the cylinder selection's own contract. All of these run
                          before the geometry or the model is touched, because
                          they cost milliseconds and a broken path invalidates
                          everything after it.
  2. HOOK LAYER OFF-BY-ONE  the hooked module's output IS hidden_states[19].
                          This is the one that would corrupt everything while
                          looking fine: steering one layer away from the space
                          the targets were defined in.
  3. delta norms          every arm realises ||delta|| = |alpha| * N_bar per
                          position, to tolerance — the dose matching the whole
                          comparison rests on
  4. arms differ          the legacy axis arms point in measurably
                          different directions (if they did not, the experiment
                          would be vacuous), and the manifold arm's out-of-knots
                          counter actually fires (WP5)
  5. generation           every arm produces text end to end
  6. rate                 measured generations/sec, so the full run's wall-clock
                          is a measurement rather than a guess

Checks 3 and 4 probe THE ARMS `run_steering.build_grid` EMITS. They used to
probe `arm1_axis` / `arm2_linear` / `arm3_manifold`, which the two-arm plan
renamed and removed: the module raised AttributeError on import of check 3
after loading the geometry and the 3B model, and the parts that did run
measured arms the grid no longer contains.

Usage:
    .venv/bin/python -m steering.smoke                 # no API calls at all
    .venv/bin/python -m steering.smoke --judge         # + <=50 interactive judge calls
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from steering.geometry import LAYER_HS_INDEX, load_geometry
from steering.activation_steering import ActivationSteering, hook_layer_for_hidden_state
from steering import interventions as IV
from steering.spline1d import CubicSpline1D
from steering import manifold_paths as MP

TOL_DOSE = 0.02          # relative tolerance on ||delta|| / (alpha * N_bar)
TOL_ARC = 0.02           # relative tolerance on uniform-alpha segment lengths


def check_spline() -> dict:
    from scipy.interpolate import CubicSpline
    rng = np.random.default_rng(0)
    u = np.sort(rng.uniform(-3, 3, 12))
    y = rng.normal(size=(12, 4))
    ours, ref = CubicSpline1D(u, y, lam=0.0), CubicSpline(u, y, axis=0, bc_type="natural")
    q = np.linspace(u[0], u[-1], 200)
    interior = float(np.abs(ours.evaluate(q) - ref(q)).max())
    knots = float(np.abs(ours.evaluate(u) - y).max())
    ok = interior < 1e-10 and knots < 1e-10
    return {"check": "spline_port", "passed": ok,
            "max_abs_diff_vs_scipy": interior, "max_knot_interp_error": knots}


# --------------------------------------------------------------------------
# 1b. Path construction (steering.manifold_paths)
#
# Pure numpy, no model, no torch, deterministic seeds, milliseconds each — so
# they belong here, next to check_spline, ahead of anything that loads a model.
# --------------------------------------------------------------------------

PATH_LAMS = (0.0, 1e-3, 1.0, 1e3)
PATH_PARAMS = ("projection", "length", "centripetal")


def _cloud(n=40, dim=6, seed=0, spread=0.6):
    """A deterministic synthetic centroid cloud straddling a chord of length 10.

    Points sit at uniform positions along the chord with gaussian lateral
    scatter ORTHOGONAL to it, so the cylinder radius is a knob with a known
    effect and the chord ordering and the lateral ordering are independent —
    which is the situation the module's docstring says the real cloud is in.
    """
    rng = np.random.default_rng(seed)
    P0 = np.zeros(dim, dtype=np.float64); P0[0] = -5.0
    P1 = np.zeros(dim, dtype=np.float64); P1[0] = 5.0
    u = rng.uniform(0.05, 0.95, n)
    C = P0[None, :] + u[:, None] * (P1 - P0)[None, :]
    perp = rng.normal(scale=spread, size=(n, dim))
    perp[:, 0] = 0.0                       # scatter is purely off-chord
    return C + perp, P0, P1


def check_path_endpoints() -> dict:
    """POSITIVE CONTROL: every path starts at P0 and ends at P1, at every lam.

    "Same endpoints, same stops, different route" is the entire comparison. If
    the manifold arm's ends drift, the two arms no longer share a start and a
    finish and every alpha-matched contrast is measuring a different pair of
    points, silently. Swept over lam AND param because the pinning lives in the
    fit (`_fit_fixed_ends`) while the abscissa is chosen outside it, so either
    could break the other.

    The last case is a regression test: a knot ~1e-12 away from a PINNED end in
    the fitting coordinate must be DROPPED, not averaged into the endpoint.
    Averaging moves the end while leaving the curve looking perfectly smooth —
    `FixedEndSpline.__init__` documents exactly this failure (knots [0, 1e-12]
    with y [0, 10] starting the curve at 5).
    """
    C, P0, P1 = _cloud(seed=1)
    L = float(np.linalg.norm(P1 - P0))
    tol = 1e-9 * L

    grid, worst = [], 0.0
    for lam in PATH_LAMS:
        for param in PATH_PARAMS:
            p = MP.PersonaPath(P0, P1, C, eps=2.0, k=8, lam=lam, param=param)
            d = MP.endpoint_drift(p, P0, P1)
            worst = max(worst, d)
            grid.append({"lam": lam, "param": param,
                         "n_centroids": p.n_centroids, "drift": d})
    # A path with no interior knots would pass vacuously (it is the chord).
    grid_ok = worst < tol and all(g["n_centroids"] >= 3 for g in grid)

    y = np.array([[0., 0.], [10., 7.], [3., 1.], [6., -2.], [9., 4.]])
    ties = []
    for lam in (0.0, 1.0):
        near_lo = MP.FixedEndSpline(np.array([0.0, 1e-12, 0.4, 0.7, 1.0]), y, lam)
        near_hi = MP.FixedEndSpline(np.array([0.0, 0.3, 0.6, 1.0 - 1e-12, 1.0]), y, lam)
        e_lo = near_lo.at_alpha(np.array([0.0, 1.0]))
        e_hi = near_hi.at_alpha(np.array([0.0, 1.0]))
        ties.append({"lam": lam,
                     "tie_at_start__start_drift": float(np.linalg.norm(e_lo[0] - y[0])),
                     "tie_at_start__end_drift": float(np.linalg.norm(e_lo[-1] - y[-1])),
                     "tie_at_end__start_drift": float(np.linalg.norm(e_hi[0] - y[0])),
                     "tie_at_end__end_drift": float(np.linalg.norm(e_hi[-1] - y[-1]))})
    tie_worst = max(v for t in ties for key, v in t.items() if key != "lam")
    tie_ok = tie_worst < 1e-9 * float(np.linalg.norm(y[-1] - y[0]))

    return {"check": "path_endpoints", "passed": bool(grid_ok and tie_ok),
            "chord_length": L, "tolerance": tol,
            "max_drift_over_grid": worst, "grid_passed": bool(grid_ok),
            "max_drift_near_tie": tie_worst, "near_tie_passed": bool(tie_ok),
            "near_tie": ties, "per_case": grid}


def check_eps_zero_collapse() -> dict:
    """NEGATIVE CONTROL: eps = 0 must reproduce the chord exactly, not nearly.

    With no centroid inside the tube there is no route, and the manifold arm has
    to become the linear arm — bit for bit, at every alpha. If it does not, the
    "manifold vs linear" difference measured elsewhere in the run has a floor
    that is an artefact of the construction rather than of the manifold. The
    `.knots` assertion covers the branch's other trap: the fallback path stores
    a two-row `knots` instead of a spline, and reporting code that walks
    `path.knots` would hit None if it were left unset.
    """
    C, P0, P1 = _cloud(seed=2)
    p = MP.PersonaPath(P0, P1, C, eps=0.0, k=8)
    lin = MP.LinearPath(P0, P1)
    alphas = np.linspace(0.0, 1.0, 41)
    diff = float(np.abs(p.at_alpha(alphas) - lin.at_alpha(alphas)).max())
    knots = np.asarray(p.knots) if p.knots is not None else None
    knots_ok = (knots is not None and knots.shape == (2, P0.shape[0])
                and np.array_equal(knots[0], P0) and np.array_equal(knots[1], P1))
    _, _, idx = MP.pick_centroids(C, P0, P1, 0.0, k=8)
    return {"check": "eps_zero_collapse",
            "passed": bool(p.n_centroids == 0 and p.detour_ratio == 1.0
                           and diff < 1e-12 and knots_ok and len(idx) == 0
                           and p.spline is None and p.bending_energy() == 0.0),
            "n_centroids": p.n_centroids,
            "detour_ratio": p.detour_ratio,
            "detour_ratio_is_exactly_one": bool(p.detour_ratio == 1.0),
            "max_abs_diff_vs_linear": diff,
            "knots_shape": None if knots is None else list(knots.shape),
            "knots_usable": bool(knots_ok),
            "n_selected": int(len(idx))}


def check_alpha_is_arc_position() -> dict:
    """alpha is NORMALISED ARC POSITION, not the spline's knot coordinate.

    Nothing else in the repo tests this, and it is the claim the whole design
    rests on: the two arms are compared at matched alpha, so alpha has to mean
    the same physical thing on a straight path and on a curved one. A regression
    that reverted alpha to the knot coordinate would still produce a smooth
    curve with the right endpoints and plausible-looking figures — it would just
    quietly bunch the sample points wherever the knots happened to be dense, and
    the experiment would be a different one than the plan describes.

    Test: sample at uniform alpha, measure the true distance travelled between
    consecutive samples, and require every segment to be within TOL_ARC of
    delta_alpha * total. Residual here is the secant-vs-arc gap, which shrinks
    with sampling density; a knot-coordinate alpha would be off by tens of
    percent and would not shrink.
    """
    C, P0, P1 = _cloud(seed=3)
    alphas = np.linspace(0.0, 1.0, 201)
    per_param, worst = [], 0.0
    for param in PATH_PARAMS:
        p = MP.PersonaPath(P0, P1, C, eps=2.0, k=6, lam=0.0, param=param)
        pts = p.at_alpha(alphas)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        want = total / len(seg)                  # delta_alpha * total arc length
        rel = float(np.abs(seg - want).max() / want)
        worst = max(worst, rel)
        per_param.append({"param": param, "n_centroids": p.n_centroids,
                          "measured_arc": total, "reported_arc": p.arc_length,
                          "max_rel_segment_error": rel,
                          "detour_ratio": p.detour_ratio})
    curved = all(r["detour_ratio"] > 1.05 for r in per_param)   # not vacuous

    # The straight arm has a closed form, so it is exact, not approximate.
    lin = MP.LinearPath(P0, P1)
    b = np.array([0.0, 0.137, 0.5, 0.9, 1.0])
    exact = float(np.abs(lin.at_alpha(b)
                         - (P0[None, :] + b[:, None] * (P1 - P0)[None, :])).max())
    return {"check": "alpha_is_arc_position",
            "passed": bool(worst < TOL_ARC and curved and exact == 0.0),
            "tolerance": TOL_ARC, "n_alphas": len(alphas),
            "max_rel_segment_error": worst,
            "paths_are_actually_curved": bool(curved),
            "linear_path_max_abs_error": exact,
            "per_param": per_param}


def check_knot_interpolation() -> dict:
    """The route must visit the REAL persona centroids it claims to visit.

    Every interior knot is a fully-role-playing persona centroid, never an
    average; "the path passes through these personas" is the manifold arm's
    whole claim. At lam = 0 the curve must hit each of them exactly, and
    `knot_error()` must be that miss.

    Regression: `knot_error` once compared the curve against `spline.y_hat` —
    the values the FIT produced — which is a tautology that returns ~1e-15 no
    matter how far the curve strays. Here at lam > 0 the two are separated
    explicitly: the y_hat comparison stays at machine epsilon while the real
    knot error must be strictly positive, so a check measuring the wrong thing
    cannot pass.
    """
    C, P0, P1 = _cloud(seed=4)
    L = float(np.linalg.norm(P1 - P0))
    dense = np.linspace(0.0, 1.0, 4001)
    rows, ok = [], True
    for param in PATH_PARAMS:
        p0 = MP.PersonaPath(P0, P1, C, eps=2.0, k=8, lam=0.0, param=param)
        interior = p0.knots[1:-1]
        # Independent of knot_error's own maths: the shortest distance from each
        # persona centroid to a dense sampling of the curve.
        pts = p0.at_alpha(dense)
        near = float(np.linalg.norm(pts[:, None, :] - interior[None, :, :],
                                    axis=2).min(0).max())
        ke0 = p0.knot_error()

        p1 = MP.PersonaPath(P0, P1, C, eps=2.0, k=8, lam=1.0, param=param)
        taut = float(np.linalg.norm(
            p1.spline.evaluate(p1.spline.x[1:-1]) - p1.spline.y_hat[1:-1],
            axis=1).max())
        ke1 = p1.knot_error()

        row = {"param": param,
               "n_knots_lam0": int(len(p0.knots)), "lam_used_lam0": p0.lam,
               "knot_error_lam0": ke0, "dense_min_distance_lam0": near,
               "n_knots_lam1": int(len(p1.knots)), "lam_used_lam1": p1.lam,
               "knot_error_lam1": ke1, "vs_fitted_values_lam1": taut}
        rows.append(row)
        ok = ok and (len(p0.knots) >= 5 and ke0 < 1e-9 and near < 1e-3 * L
                     # smoothing must really be on: _fit_fixed_ends forces
                     # lam=0 whenever there is no free interior gamma
                     and len(p1.knots) >= 5 and p1.lam == 1.0
                     and ke1 > 1e-9 and taut < 1e-9 and ke1 > 1e3 * max(taut, 1e-18))
    return {"check": "knot_interpolation", "passed": bool(ok),
            "chord_length": L, "per_param": rows}


def check_cylinder_selection() -> dict:
    """The cylinder's contract: inside the radius, between the ends, in order.

    Every downstream property depends on it. Out-of-order indices break the
    monotone knot coordinate the spline needs; an index outside END_MARGIN is
    the endpoint duplication the module measured at detour 5.7-9.1; and the two
    modes must differ by exactly the chord-length factor, because "absolute"
    exists precisely so a tube radius transfers between chords of different
    length.
    """
    C, P0, P1 = _cloud(seed=5)
    L = float(np.linalg.norm(P1 - P0))
    # Tight enough that the tube is a real filter: at eps*L the relative
    # tube swallows the whole cloud, which is what makes the two modes
    # distinguishable below.
    eps_abs = 1.0

    idx, u, r = MP.select_cylinder(C, P0, P1, eps_abs, mode="absolute")
    sel_u, sel_r = u[idx], r[idx]
    inside = bool(len(idx) and (sel_r < eps_abs).all())
    in_span = bool(len(idx)
                   and (sel_u > MP.END_MARGIN).all()
                   and (sel_u < 1.0 - MP.END_MARGIN).all())
    ordered = bool(len(idx) > 1 and (np.diff(sel_u) > 0).all())

    # k is a ceiling, never exceeded, whatever the candidate set looks like.
    k_ok = True
    for k in (1, 3, 8, 64):
        _, Y, ii = MP.pick_centroids(C, P0, P1, eps_abs, k=k)
        k_ok = k_ok and len(ii) <= k and len(Y) == len(ii)

    # radius = eps*L (relative) vs eps (absolute): the same tube for eps*L vs eps.
    same, _, _ = MP.select_cylinder(C, P0, P1, eps_abs / L, mode="relative")
    scaled_ok = bool(np.array_equal(idx, same))
    wider, _, _ = MP.select_cylinder(C, P0, P1, eps_abs, mode="relative")
    # ... and not vacuously the same: at equal eps the relative tube is L times
    # wider, so it must admit strictly more points on this cloud.
    distinct_ok = bool(len(wider) > len(idx))

    raises = {}
    for name, fn in (("bad_mode", lambda: MP.select_cylinder(C, P0, P1, 1.0, mode="cylindrical")),
                     ("bad_param", lambda: MP.PersonaPath(P0, P1, C, eps=eps_abs, param="chordal")),
                     ("degenerate_chord", lambda: MP.chord_frame(P0, P0))):
        try:
            fn()
        except ValueError:
            raises[name] = True
        except Exception:                    # anything else is the wrong error
            raises[name] = False
        else:
            raises[name] = False

    return {"check": "cylinder_selection",
            "passed": bool(inside and in_span and ordered and k_ok
                           and scaled_ok and distinct_ok and all(raises.values())),
            "n_selected": int(len(idx)), "chord_length": L, "eps_absolute": eps_abs,
            "max_perp_distance": float(sel_r.max()) if len(idx) else None,
            "u_min": float(sel_u.min()) if len(idx) else None,
            "u_max": float(sel_u.max()) if len(idx) else None,
            "end_margin": MP.END_MARGIN,
            "u_strictly_increasing": ordered,
            "pick_respects_k": bool(k_ok),
            "relative_eps_over_L_matches_absolute": scaled_ok,
            "n_selected_relative_same_eps": int(len(wider)),
            "raises_value_error": raises}


def check_hook_layer(model, tokenizer, hook_layer: int) -> dict:
    """The hooked module's output must equal hidden_states[LAYER_HS_INDEX].

    `output_hidden_states=True` returns L+1 tensors: index 0 is the embedding
    output and index i>=1 is the output of model.layers[i-1]. If this is off by
    one, every arm steers in a space one layer away from the geometry that
    defines its targets, and nothing downstream would notice.
    """
    import torch

    enc = tokenizer("Who are you?", return_tensors="pt").to(model.device)
    captured = {}

    def cb(activations, layer_idx):
        t = activations[0] if isinstance(activations, (tuple, list)) else activations
        captured["h"] = t.detach().float().cpu().numpy().copy()
        return torch.zeros_like(t)          # no-op delta: pure observation

    with ActivationSteering(model, intervention_type="dynamic", delta_fn=cb,
                            layer_indices=[hook_layer], positions="all"):
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

    ref = out.hidden_states[LAYER_HS_INDEX].detach().float().cpu().numpy()
    got = captured["h"]
    diff = float(np.abs(got - ref).max())
    scale = float(np.abs(ref).max())

    # Also confirm the neighbours are NOT equal, so a match cannot be accidental.
    n_lo = float(np.abs(got - out.hidden_states[LAYER_HS_INDEX - 1].detach().float().cpu().numpy()).max())
    n_hi = (float(np.abs(got - out.hidden_states[LAYER_HS_INDEX + 1].detach().float().cpu().numpy()).max())
            if LAYER_HS_INDEX + 1 < len(out.hidden_states) else float("inf"))

    return {"check": "hook_layer_offset",
            "passed": bool(diff <= 1e-3 * max(scale, 1.0) and n_lo > 1e-3 and n_hi > 1e-3),
            "hook_layer": hook_layer, "hidden_state_index": LAYER_HS_INDEX,
            "max_abs_diff": diff, "activation_absmax": scale,
            "diff_to_layer_below": n_lo, "diff_to_layer_above": n_hi}


def _delta_norms_at(geom, model, tokenizer, hook_layer: int, alpha: float) -> dict:
    """Both LEGACY arms at one alpha: realised dose, direction, out-of-knots.

    These are the dose-matched forms — `linear_axis_legacy` and
    `manifold_axis_legacy` since the path rebuild — and the keys say so. The
    four path arms hold ||delta|| nowhere near constant (they hold the
    ENDPOINTS constant instead), so this check does not apply to them; their
    controls are delta(0) = 0 and delta(1) = P1 - P0, checked in
    `run_steering`'s path manifest and offline.
    """
    import torch

    enc = tokenizer("Who are you?", return_tensors="pt").to(model.device)
    want = abs(alpha) * geom.n_bar          # both arms scale by |alpha| * N_bar
    results, dirs = {}, {}

    # manifold_axis_legacy is the only dynamic arm here: its direction depends on the
    # current activation, so its dose can only be measured under a hook. The
    # DeltaStats goes in too, because WP5's frac_out_of_knots is exactly the
    # kind of counter that can be wired up and never fire.
    stats = IV.DeltaStats()
    fn = IV.make_manifold_axis_delta_fn(geom.spline, geom.axis_unit, alpha,
                                        geom.n_bar, geom.span, stats=stats)
    seen = {}

    def probe(activations, layer_idx):
        t = activations[0] if isinstance(activations, (tuple, list)) else activations
        d = fn(t, layer_idx)
        seen["d"] = d.detach().float().cpu().numpy().copy()
        return torch.zeros_like(t)

    with ActivationSteering(model, intervention_type="dynamic", delta_fn=probe,
                            layer_indices=[hook_layer], positions="all"):
        with torch.no_grad():
            model(**enc)
    d_man = seen["d"].reshape(-1, seen["d"].shape[-1])
    norms = np.linalg.norm(d_man, axis=-1)
    results["manifold_axis_legacy"] = {
        "mean": float(norms.mean()), "min": float(norms.min()),
        "max": float(norms.max()),
        "rel_err": float(np.abs(norms - want).max() / want)}
    dirs["manifold_axis_legacy"] = d_man

    # linear_axis_legacy is a static vector on the vendored `addition` path, so its
    # norm is exact by construction and needs no forward pass.
    v = IV.linear_axis_vector(geom.axis_unit, alpha, geom.n_bar)
    nv = float(np.linalg.norm(v))
    results["linear_axis_legacy"] = {"mean": nv, "min": nv, "max": nv,
                                     "rel_err": float(abs(nv - want) / want)}
    dirs["linear_axis_legacy"] = v.reshape(1, -1)

    lin = dirs["linear_axis_legacy"][0]
    lin = lin / (np.linalg.norm(lin) + 1e-12)
    cos = float(np.mean(d_man @ lin /
                        (np.linalg.norm(d_man, axis=1) + 1e-12)))
    rep = stats.report()
    return {"alpha": alpha, "target_norm": want, "per_arm": results,
            "mean_cos_linear_manifold": cos,
            "arms_are_distinct": bool(abs(cos) < 0.999),
            "manifold_delta_stats": rep,
            # WP5 wired the counter but nothing checked it ever ran; an
            # all-zero delta_stats parquet reads as a valid artifact.
            "out_of_knots_measured": bool(rep.get("n_coords", 0) > 0)}


def check_delta_norms(geom, model, tokenizer, hook_layer: int,
                      alphas=(-1.0, 0.25)) -> dict:
    """Both live arms must realise ||delta|| = |alpha| * N_bar at every position.

    Probed at BOTH ENDS of the sweep, not at a single positive alpha. Alpha is
    signed now — negative is away from the Assistant, positive toward it, and
    the shard filenames encode the sign — so a sign bug in either arm is
    exactly the class of thing this gate exists to catch before GPU time.
    """
    per_alpha = [_delta_norms_at(geom, model, tokenizer, hook_layer, a)
                 for a in alphas]
    passed = all(r["rel_err"] < TOL_DOSE
                 for res in per_alpha for r in res["per_arm"].values())
    return {"check": "delta_norms",
            "passed": bool(passed
                           and all(r["arms_are_distinct"] for r in per_alpha)
                           and all(r["out_of_knots_measured"] for r in per_alpha)),
            "n_bar": geom.n_bar, "alphas": list(alphas),
            "dose_within_tolerance": bool(passed),
            "per_alpha": per_alpha}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="run dir (default: timestamped smoke dir)")
    ap.add_argument("--judge", action="store_true",
                    help="also exercise the judge on <=50 interactive calls")
    ap.add_argument("--max-calls", type=int, default=50)
    ap.add_argument("--model", default=None)
    ap.add_argument("--skip-generation", action="store_true")
    # The gate has to test THE GEOMETRY THE RUN WILL USE. Without these the
    # smoke test builds an unfiltered axis and the resp240 fallback dose, then
    # certifies plumbing the real run never executes.
    ap.add_argument("--labels", default=None,
                    help="role_labels.parquet from steering.rolefilter (WP1); "
                         "pass the same one run_steering will get")
    ap.add_argument("--n-bar", default=None,
                    help="n_bar_lmsys.json from steering.lmsys_norm (WP2); "
                         "pass the same one run_steering will get")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from manifold_persona.config import MODEL_NAME
    from steering.runmeta import new_run_dir

    run_dir = Path(args.out) if args.out else new_run_dir("smoke")
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    checks = []

    print("== 1. spline port")
    checks.append(check_spline())
    print("   ", checks[-1])

    # Cheap-first: pure numpy, no model, no geometry. A broken path would
    # invalidate every later check, so it is worth knowing before the load.
    print("== 1b. path construction")
    for fn in (check_path_endpoints, check_eps_zero_collapse,
               check_alpha_is_arc_position, check_knot_interpolation,
               check_cylinder_selection):
        checks.append(fn())
        print("    %-24s %s" % (checks[-1]["check"],
                                "PASS" if checks[-1]["passed"] else "FAIL"))
        if not checks[-1]["passed"]:
            print(json.dumps(checks[-1], indent=2))

    print("== geometry")
    t0 = time.time()
    geom = load_geometry(labels_path=args.labels, n_bar_path=args.n_bar)
    print("    loaded in %.1fs | N_bar %.3f | %d roles" % (time.time() - t0, geom.n_bar, len(geom.roles)))
    for report, flag in ((geom.axis_report, "--labels"),
                         (geom.n_bar_report, "--n-bar")):
        if "deviation" in report:
            print("    DEVIATION (%s not given): %s" % (flag, report["deviation"]))

    model_name = args.model or MODEL_NAME
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print("== loading %s on %s" % (model_name, device))
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()
    hook_layer = hook_layer_for_hidden_state(LAYER_HS_INDEX)

    print("== 2. hook layer off-by-one")
    checks.append(check_hook_layer(model, tokenizer, hook_layer))
    print("   ", checks[-1])

    print("== 3/4. delta norms + arm distinctness")
    checks.append(check_delta_norms(geom, model, tokenizer, hook_layer))
    print("   ", json.dumps(checks[-1], indent=2))

    if not args.skip_generation:
        print("== 5/6. generation")
        from steering.run_steering import (FULL_GRID_ROWS, LEGACY_ARMS,
                                           SMOKE_MAX_NEW_TOKENS,
                                           build_grid, default_batch_size,
                                           generate_cell)
        # THE LEGACY ARMS, because checks 3/4 above are the legacy arms: they
        # call `linear_axis_vector` and `make_manifold_axis_delta_fn` directly.
        # The four path arms need `path_cases.load_cases`, which needs --labels
        # and hard-fails without them, so pulling them in here would make the
        # plumbing check depend on an artifact the smoke test does not take.
        # They are covered by `steering/geometry_check.py` and by the offline
        # path checks instead.
        cells = build_grid(geom, smoke=True,
                           arms=["unsteered"] + LEGACY_ARMS)
        bs = default_batch_size(device)
        print("    batch_size=%d on %s" % (bs, device))
        # One cell from every arm, so a broken arm cannot hide behind a working one.
        pick, seen = [], set()
        for c in cells:
            if c.arm not in seen:
                seen.add(c.arm); pick.append(c)
        t0, rows = time.time(), 0
        frames = []
        for c in pick:
            # Short generations on purpose. The eval is uncapped, but this check
            # is about plumbing — that each arm produces text at all — and the
            # 1024-token ceiling would turn a 15-minute smoke test into hours
            # without testing anything the first 128 tokens do not.
            df = generate_cell(c, geom, model, tokenizer, hook_layer, batch_size=bs,
                               max_new_tokens=SMOKE_MAX_NEW_TOKENS)
            frames.append(df)
            rows += len(df)
            print("    %-42s %2d rows  %.2f gen/s" % (c.shard_name, len(df),
                                                      len(df) / df.attrs["elapsed_s"]))
        rate = rows / (time.time() - t0)
        import pandas as pd
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_parquet(run_dir / "data" / "generations_L19.parquet", index=False)
        # `rows > 0` is NOT a check. The first version of this file passed 4/4
        # while every response was the string "!!!!!!!!" (Observations O3), so
        # the check now looks at the TEXT: degenerate output is one character
        # repeated, or near-empty, and it must be zero in the unsteered arm.
        def _degenerate(t: str) -> bool:
            t = (t or "").strip()
            return len(t) < 10 or len(set(t.replace(" ", ""))) <= 2

        deg = all_df["response"].map(_degenerate)
        base = all_df["arm"] == "unsteered"
        n_deg_base = int(deg[base].sum())
        examples = all_df.loc[deg, "response"].head(2).tolist()
        checks.append({"check": "generation", "rows": rows,
                       "passed": bool(rows > 0 and n_deg_base == 0),
                       "degenerate_unsteered": n_deg_base,
                       "degenerate_total": int(deg.sum()),
                       "degenerate_examples": [e[:60] for e in examples],
                       "generations_per_second": rate,
                       "full_grid_rows": FULL_GRID_ROWS,
                       "projected_hours_full_grid": FULL_GRID_ROWS / rate / 3600,
                       "batch_size": bs, "device": device})
        print("   ", checks[-1])
        for arm in all_df["arm"].unique():
            r = all_df[all_df["arm"] == arm].iloc[-1]
            print("\n    %s alpha=%.2f role=%s q=%r" % (arm, r["alpha"], r["role"], r["question"]))
            print("       ", repr(r["response"][:170]))

    if args.judge:
        from steering.judge import smoke_judge
        checks.append(smoke_judge(run_dir, max_calls=args.max_calls))
        print("   ", checks[-1])

    (run_dir / "data" / "smoke_checks.json").write_text(json.dumps(checks, indent=2))
    failed = [c for c in checks if not c.get("passed", False)]
    print("\n%s  %d/%d checks passed -> %s"
          % ("FAIL" if failed else "PASS", len(checks) - len(failed), len(checks), run_dir))
    if failed:
        raise SystemExit("failed: " + ", ".join(c["check"] for c in failed))


if __name__ == "__main__":
    main()
