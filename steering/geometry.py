"""Everything the arms need from the resp240 cloud: axis, centroids, curve, dose.

Plan: plans/2026-08-17-manifold-steering-role-susceptibility.md (Data, Method).

Loaded once per run, cached to the run dir, hashed into the manifest. Nothing
here writes under `data/` — the cloud is read-only (RESEARCH.steering.md).

Reused, not reimplemented:
    manifold_persona.common.load_points      the study's loader/aggregator
    manifold_persona.common.assistant_axis   the paper's section 3.1 contrast vector
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from manifold_persona.common import assistant_axis, load_points
from steering.spline1d import CubicSpline1D, fit_gcv

RESP240_DIR = "data/embeddings_roles_resp240"
LAYER_HS_INDEX = 19        # our cloud's layer; see hook_layer_for_hidden_state
N_NEAR = 50                # the paper's 50 near-Assistant roles (section 3.2.1)
N_FAR = 50
# Selected by GCV on the resp240 cloud and confirmed by the user (2026-08-17).
# Not a tuned constant: it is the value the rule returns, recorded so drift is
# detectable. See plan Amendment A1.
LAMBDA_EXPECTED = 0.001


@dataclass
class Geometry:
    """The fitted geometry of the role cloud at one layer."""
    roles: List[str]                 # sorted, length 276
    centroids: np.ndarray            # (276, hidden) float64
    axis_unit: np.ndarray            # (hidden,) unit Assistant Axis
    axis_proj: np.ndarray            # (276,) centroid . axis_unit
    n_bar: float                     # mean response-token residual norm (the dose unit)
    spline: CubicSpline1D            # the curve, keyed by axis_proj
    lam: float                       # bending penalty, chosen by GCV (amendment A1)
    lam_report: dict                 # the full GCV grid and scores, for the manifest
    near50: List[str]
    far50: List[str]
    pairing: Dict[str, str]          # near role -> far role
    hidden: int

    def centroid(self, role: str) -> np.ndarray:
        return self.centroids[self.roles.index(role)]


_CLOUD_CACHE = {}


def _require_finite(a: np.ndarray, what: str) -> np.ndarray:
    """Hard check. Silent NaN/inf in the geometry would poison every arm."""
    if not np.all(np.isfinite(a)):
        n = int((~np.isfinite(a)).sum())
        raise ValueError("%s contains %d non-finite values" % (what, n))
    return a


def _matmul(a: np.ndarray, b: np.ndarray, what: str) -> np.ndarray:
    """float64 matmul with the platform's spurious FP warnings suppressed.

    numpy 2.0.2 on macOS Accelerate raises "divide by zero / overflow / invalid
    encountered in matmul" for *any* matmul, including
    `np.zeros((276,2048)) @ np.ones(2048)` — verified 2026-08-17. The flags come
    from the BLAS backend, not from the operands, so the warnings carry no
    information and would drown a real one.

    Suppression is therefore paired with an explicit finiteness check on the
    result: genuine corruption still raises, loudly, with the operand named.
    """
    with np.errstate(all="ignore"):
        out = a @ b
    return _require_finite(out, what)


def _load_cloud(resp_dir: str, layer: Optional[int]):
    """Load the response cloud once per process and cache it.

    The array is 331,200 x 2048 float32 = 2.7 GB. It is deliberately NOT
    upcast to float64 in bulk (that would be 5.4 GB and this runs on a laptop
    for the smoke test); float64 is used only for the per-role reductions,
    where the arrays are 1,200 rows each.
    """
    key = (str(resp_dir), layer)
    if key in _CLOUD_CACHE:
        return _CLOUD_CACHE[key]

    os.environ["MP_ROLE_DIR"] = str(resp_dir)
    X, meta, manifest = load_points(view="prompt_avg", layer=layer,
                                    in_dir=resp_dir, aggregate="none")

    token_basis = manifest.get("token_basis")
    if token_basis != "response":
        raise ValueError(
            "expected a response-token cloud (token_basis='response'), got %r. "
            "The prompt cloud carries the attention-sink and length artefacts "
            "documented in diagnostics/README.md and must not be used here."
            % token_basis
        )

    _CLOUD_CACHE[key] = (X, meta, manifest)
    return X, meta, manifest


def load_geometry(resp_dir: str = RESP240_DIR, layer: Optional[int] = None,
                  lam: Optional[float] = None) -> Geometry:
    """Build the geometry from the response cloud.

    The cloud is published pre-thinned to layer 19 at index 0 (its manifest
    says `source_layers: [19]`, `primary_layer: 0`), so `layer=None` takes the
    single stored layer rather than guessing an index into a 37-layer stack.
    """
    X, meta, manifest = _load_cloud(resp_dir, layer)

    roles_arr = meta["role"].values
    roles = sorted(set(roles_arr))

    # Axis from the full per-example cloud, matching the paper's definition:
    # mean(default) - mean(all). Computed before aggregation so `default`'s
    # rollouts weigh as they do in the cloud. `assistant_axis` is the study's
    # own function, reused unchanged (plan, Outputs).
    axis_unit = _require_finite(
        np.asarray(assistant_axis(X, meta), dtype=np.float64), "assistant axis")

    # float64 per role (1,200 rows each), not on the whole 2.7 GB array.
    centroids = _require_finite(
        np.stack([X[roles_arr == r].mean(0, dtype=np.float64) for r in roles]),
        "role centroids")
    axis_proj = _matmul(centroids, axis_unit, "axis_proj")

    # The dose unit: mean residual norm over every response-token activation in
    # the cloud. The paper scales to the mean post-MLP residual norm on
    # LMSYS-Chat-1M (section 3.2.1); we do not have that corpus, so this is
    # deliberate difference D3 and alpha values are not comparable to theirs.
    # Chunked so the squared array never materialises at full size.
    total, count = 0.0, 0
    for s in range(0, X.shape[0], 20000):
        blk = X[s:s + 20000].astype(np.float64, copy=False)
        total += float(np.linalg.norm(blk, axis=1).sum())
        count += blk.shape[0]
    n_bar = total / count

    # Bending penalty (amendment A1, 2026-08-17). lam = 0 makes the spline
    # INTERPOLATE its control points, which has two fatal consequences measured
    # in Observations O2: the target residual r_T = c_T - S(u_T) is identically
    # zero (so Arm 3's only target-aware term vanishes), and the curve zigzags
    # between roles adjacent in axis rank (tortuosity 361x, tangent 98% off-axis).
    # lam > 0 makes the curve a trend near the points instead of through them,
    # which revives r_T and removes the zigzag.
    #
    # lam is chosen by GCV, never by hand: GCV sees only the 276 centroids, so
    # it cannot tune lambda toward a steering outcome. Pass lam explicitly to
    # override (the smoke test's lambda sweep does).
    if lam is None:
        spline, lam_report = fit_gcv(axis_proj, centroids)
        lam = spline.lam
        # The RULE is GCV; LAMBDA_EXPECTED is what it selected on this cloud
        # (user confirmed 2026-08-17). Keeping the rule rather than hard-pinning
        # the number keeps the justification intact — but a silent change would
        # alter all three arms invisibly, so drift is announced, not swallowed.
        if abs(lam - LAMBDA_EXPECTED) > 1e-12:
            lam_report["drifted_from_expected"] = LAMBDA_EXPECTED
            print("WARNING: GCV selected lambda=%g, expected %g. The manifold "
                  "and therefore Arm 3 have changed. Check the input cloud "
                  "before trusting any figure." % (lam, LAMBDA_EXPECTED))
        lam_report["expected"] = LAMBDA_EXPECTED
    else:
        spline = CubicSpline1D(axis_proj, centroids, lam=lam)
        lam_report = {"rule": "explicit override", "lambda": float(lam)}

    # Role selection. `default` is the Assistant baseline, not a role
    # (prompts_roles.RoleRecord.is_default), and is excluded from both ends —
    # the study's existing rule, inherited unchanged.
    order = [i for i in np.argsort(-axis_proj) if roles[i] != "default"]
    near50 = [roles[i] for i in order[:N_NEAR]]          # most Assistant-like
    far50 = [roles[i] for i in order[-N_FAR:]][::-1]     # least, most-extreme first

    # Rank-paired: near rank i steers toward far rank i. Fixed, not sampled, so
    # it is reproducible without a seed and cannot drift between runs.
    pairing = {n: f for n, f in zip(near50, far50)}

    return Geometry(roles=roles, centroids=centroids, axis_unit=axis_unit,
                    axis_proj=axis_proj, n_bar=n_bar, spline=spline,
                    lam=float(lam), lam_report=lam_report,
                    near50=near50, far50=far50, pairing=pairing,
                    hidden=int(X.shape[1]))


def save_role_files(geom: Geometry, data_dir: Path) -> None:
    """Write the role selection as provenance. Committed to the run dir."""
    data_dir = Path(data_dir)
    idx = {r: i for i, r in enumerate(geom.roles)}

    def rows(names):
        return [{"rank": k, "role": r, "axis_proj": float(geom.axis_proj[idx[r]])}
                for k, r in enumerate(names)]

    (data_dir / "roles_near50_L19.json").write_text(json.dumps(
        {"selection": "top-50 by axis_proj, 'default' excluded",
         "n": len(geom.near50), "roles": rows(geom.near50)}, indent=2))
    (data_dir / "roles_far50_L19.json").write_text(json.dumps(
        {"selection": "bottom-50 by axis_proj, 'default' excluded",
         "n": len(geom.far50), "roles": rows(geom.far50)}, indent=2))
    (data_dir / "target_pairing_L19.json").write_text(json.dumps(
        {"rule": "near rank i -> far rank i; fixed, not sampled",
         "pairing": geom.pairing}, indent=2))


def degeneracy_gate(geom: Geometry, resp_dir: str = RESP240_DIR,
                    seed: int = 0, holdout_frac: float = 0.2,
                    threshold: float = 0.10) -> dict:
    """Experiment 3: are role centroids distinguishable enough to be targets?

    Nearest-centroid classification of held-out response activations. Centroids
    are recomputed on the training split so the held-out points are genuinely
    held out — scoring against centroids that contain the test points would
    measure nothing.

    Chance is 1/276 = 0.36%. The gate is top-1 >= 10%. Below that, Arms 2 and 3
    are aiming at targets the model's own activations cannot tell apart, and
    the plan says STOP.
    """
    X, meta, _ = _load_cloud(resp_dir, None)
    roles_arr = meta["role"].values
    roles = geom.roles
    rng = np.random.default_rng(seed)

    train_C, test_X, test_y = [], [], []
    for i, r in enumerate(roles):
        rows = np.flatnonzero(roles_arr == r)
        rng.shuffle(rows)
        n_test = max(1, int(round(len(rows) * holdout_frac)))
        te, tr = rows[:n_test], rows[n_test:]
        if len(tr) == 0:                      # too few rows to hold any out
            tr = rows
        train_C.append(X[tr].mean(0, dtype=np.float64))
        test_X.append(X[te].astype(np.float64))
        test_y.append(np.full(len(te), i))

    C = np.stack(train_C)
    TX = np.concatenate(test_X)
    ty = np.concatenate(test_y)

    # argmin ||x - c||^2 == argmax (x.c - ||c||^2/2); avoids an N x 276 x hidden
    # distance tensor.
    scores = _matmul(TX, C.T, "nearest-centroid scores") - 0.5 * (C * C).sum(1)[None, :]
    pred = scores.argmax(1)
    top1 = float((pred == ty).mean())

    top5 = float(np.mean([ty[i] in np.argsort(-scores[i])[:5] for i in range(len(ty))]))
    return {
        "experiment": "3_degeneracy_gate",
        "n_test": int(len(ty)), "n_roles": len(roles),
        "chance": 1.0 / len(roles),
        "top1": top1, "top5": top5,
        "threshold": threshold,
        "passed": bool(top1 >= threshold),
        "seed": seed, "holdout_frac": holdout_frac,
    }
