"""Everything the arms need from the resp240 cloud: axis, centroids, curve, dose.

Plan: plans/2026-08-17-manifold-steering-role-susceptibility.md (Data, Method).

Loaded once per run, cached to the run dir, hashed into the manifest. Nothing
here writes under `data/` — the cloud is read-only (steering/README.md).

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
    n_bar: float                     # the dose unit (see n_bar_report)
    spline: CubicSpline1D            # the curve, keyed by axis_proj
    lam: float                       # bending penalty, chosen by GCV (amendment A1)
    lam_report: dict                 # the full GCV grid and scores, for the manifest
    near50: List[str]
    far50: List[str]
    pairing: Dict[str, str]          # near role -> far role
    hidden: int
    axis_report: dict                # how the axis was built (filtered or not)
    n_bar_report: dict               # where the dose unit came from

    def centroid(self, role: str) -> np.ndarray:
        return self.centroids[self.roles.index(role)]

    @property
    def span(self) -> float:
        """The range of the curve's coordinate, i.e. how far the role cloud
        reaches along the Assistant Axis.

        Defined here, not at the call sites: `run_steering` uses it to size the
        manifold arm's step and `smoke` uses it to certify that same arm, so a
        second definition would let the gate validate a different step from the
        one the run takes."""
        return float(self.axis_proj.max() - self.axis_proj.min())


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


def _load_labels(labels_path: str, n_rows: int,
                 roles_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """Row-aligned role-expression labels from `steering.rolefilter`.

    Returns an object array of length `n_rows` holding "fully" / "somewhat" /
    "no" / None, indexed the same way the cloud is. `rolefilter` labels the
    whole cloud, so a None means its retries were exhausted on that row; those
    stay out of the `fully` pool, because an unlabelled row is not evidence of
    role-playing.

    THE ROLE COLUMN IS THE REAL GUARD. `rolefilter` writes two files with the
    IDENTICAL schema {i, label, analysis, role} into different index spaces:
    `role_labels.parquet` indexes the 331,200-row cloud, `prompt_gate_labels
    .parquet` indexes a ~4,000-row generations frame. A bounds check alone
    passes the wrong one trivially (4,000 << 331,200) and writes those verdicts
    onto cloud rows 0..4,000 — all belonging to the first few roles
    alphabetically — after which every other role fails the >=10 rule and the
    axis is built from a handful of mismatched rows, with no error and entirely
    plausible numbers. Comparing `role` against the cloud catches it outright.
    """
    import pandas as pd

    lab = pd.read_parquet(labels_path)
    if len(lab) == 0:
        raise ValueError("%s holds no labels" % labels_path)
    out = np.full(n_rows, None, dtype=object)
    idx = lab["i"].to_numpy()
    if idx.min() < 0:
        raise ValueError("labels reference negative row %d — numpy would wrap "
                         "that around to the end of the cloud" % int(idx.min()))
    if idx.max() >= n_rows:
        raise ValueError("labels reference row %d but the cloud has %d rows — "
                         "these labels were made for a different cloud"
                         % (int(idx.max()), n_rows))
    if roles_arr is not None and "role" in lab.columns:
        want = np.asarray(roles_arr, dtype=object)[idx]
        got = lab["role"].to_numpy(dtype=object)
        bad = got != want
        n_bad = int(bad.sum())
        if n_bad:
            first = int(np.flatnonzero(bad)[0])
            raise ValueError(
                "%s: %d of %d labels name a different role than the cloud row "
                "they index (first: row %d is %r in the cloud, %r in the "
                "labels). These labels index a DIFFERENT frame — "
                "prompt_gate_labels.parquet indexes the generations, not the "
                "cloud. Pass role_labels.parquet."
                % (labels_path, n_bad, len(lab), int(idx[first]),
                   want[first], got[first]))
    out[idx] = lab["label"].to_numpy()
    return out


def _role_vectors(X, roles_arr, roles, labels: np.ndarray) -> tuple:
    """The paper's role vectors: UP TO TWO PER ROLE, section 2.1.2 verbatim.

        "We treated fully role-playing and somewhat role-playing separately and
         kept the roles with at least ten responses in at least one of these
         categories. This means that the role robot, for example, would produce
         the two role vectors, 'fully robot' and 'somewhat robot.'"

    So the unit is (role, category), not role. That is why the paper's n is
    377-463 vectors from 275 roles, and it is what an earlier version of this
    file got wrong by collapsing every role to a single centroid.

    The >= 10 rule is a KEEP rule on the ROLE ("in at least one of these
    categories"), not a per-vector filter: a role with 40 `fully` and 3
    `somewhat` is kept, and contributes only its `fully` vector.

    Returns (vectors, meta, report) where `vectors` is (n_vectors, hidden) and
    `meta` is a list of {role, category, n} aligned to it.
    """
    vectors, meta = [], []
    kept_roles, dropped_roles = [], {}

    # `roles_arr` and `labels` are OBJECT arrays, so every `==` is a Python-level
    # elementwise pass over 331,200 rows. Comparing them per (role, category)
    # inside the loop made ~2,200 such passes on every geometry load; these three
    # hoisted masks plus one per role make it ~280.
    cat_masks = {c: (labels == c) for c in ("fully", "somewhat")}

    for r in roles:
        if r == "default":
            continue
        is_role = roles_arr == r
        sels = {c: is_role & m for c, m in cat_masks.items()}
        counts = {c: int(sel.sum()) for c, sel in sels.items()}
        if max(counts.values()) < 10:
            dropped_roles[r] = counts
            continue
        kept_roles.append(r)
        for cat, n in counts.items():
            if n >= 10:
                vectors.append(X[sels[cat]].mean(0, dtype=np.float64))
                meta.append({"role": r, "category": cat, "n": n})

    if not vectors:
        raise ValueError("no role cleared the >=10 rule in either category")

    # The >=10 rule can only see rows that carry a label. `rolefilter` labels
    # the whole cloud — there is no sampling option, precisely because a partial
    # sample makes the paper's POPULATION rule proportionally stricter — so
    # anything short of full coverage here is LABELLING FAILURES. Recording it
    # is what lets a reader of axis_report tell a real drop from a lost batch.
    n_labelled = int(sum(1 for v in labels if v is not None))
    report = {
        "rule": "keep a role if fully>=10 OR somewhat>=10; one vector per "
                "qualifying category (arXiv:2601.10387 section 2.1.2)",
        "label_coverage": {
            "n_rows": int(len(labels)),
            "n_labelled": n_labelled,
            "frac_labelled": n_labelled / max(len(labels), 1),
            "caveat": "the >=10 threshold is applied to LABELLED rows only. "
                      "rolefilter labels the whole cloud, so frac_labelled well "
                      "below 1.0 means rows the judge failed on — and the "
                      "threshold is then stricter than the paper's by roughly "
                      "1/frac_labelled. Re-run rolefilter to retry them.",
        },
        "n_vectors": len(vectors),
        "n_roles_kept": len(kept_roles),
        "n_roles_dropped": len(dropped_roles),
        "dropped_roles": dropped_roles,
        "n_fully_vectors": sum(1 for m in meta if m["category"] == "fully"),
        "n_somewhat_vectors": sum(1 for m in meta if m["category"] == "somewhat"),
    }
    return np.stack(vectors), meta, report


def _axis_filtered(X, roles_arr, role_vectors, vector_meta) -> tuple:
    """The section 3.1 axis: mean(default) - mean(FULLY role-playing role vectors).

    `fully` only — "We subtracted the mean of all fully role-playing role
    vectors from the mean default Assistant activation". The `somewhat` vectors
    exist (they are part of persona space and of the manifold) but they are not
    in the axis.

    Averaged with EQUAL WEIGHT PER VECTOR, which is what "the mean of all fully
    role-playing role vectors" says. With a balanced cloud that matches a
    row-mean, but it stops being true the moment the filter drops different
    numbers of rows per role — which is exactly what it does.

    `default` is the Assistant baseline, not a role, so it is the minuend and
    never enters the subtrahend.
    """
    is_default = roles_arr == "default"
    if not is_default.any():
        raise ValueError("no 'default' rows: cannot define the axis")
    default_mean = X[is_default].mean(0, dtype=np.float64)

    idx = [i for i, m in enumerate(vector_meta) if m["category"] == "fully"]
    if not idx:
        raise ValueError("no `fully` role vector exists; the section 3.1 axis "
                         "is undefined")
    axis = default_mean - role_vectors[idx].mean(0)
    report = {
        "definition": "mean(default) - mean(fully role-playing role vectors), "
                      "equal weight per vector (arXiv:2601.10387 section 3.1)",
        "n_fully_vectors_in_axis": len(idx),
    }
    return axis / np.linalg.norm(axis), report


def load_geometry(resp_dir: str = RESP240_DIR, layer: Optional[int] = None,
                  lam: Optional[float] = None, labels_path: Optional[str] = None,
                  n_bar_path: Optional[str] = None) -> Geometry:
    """Build the geometry from the response cloud.

    The cloud is published pre-thinned to layer 19 at index 0 (its manifest
    says `source_layers: [19]`, `primary_layer: 0`), so `layer=None` takes the
    single stored layer rather than guessing an index into a 37-layer stack.

    Args:
        labels_path: role-expression labels from `steering.rolefilter`. With
            them the axis and the centroids are built from `fully` rows only,
            which is the paper's definition (WP1). Without them both fall back
            to the unfiltered forms, and `axis_report` says so — the fallback
            is legal but it is a documented deviation, not the paper.
        n_bar_path: `n_bar_lmsys.json` from `steering.lmsys_norm`. Without it
            the dose unit falls back to the resp240 response-averaged norm,
            which is the WRONG QUANTITY on the WRONG CORPUS (WP2/D3).
    """
    X, meta, manifest = _load_cloud(resp_dir, layer)

    roles_arr = meta["role"].values
    roles = sorted(set(roles_arr))

    labels = (_load_labels(labels_path, X.shape[0], roles_arr)
              if labels_path else None)

    role_vectors, vector_meta, vec_report = (None, None, None)
    if labels is not None:
        role_vectors, vector_meta, vec_report = _role_vectors(
            X, roles_arr, roles, labels)
        axis_unit, axis_report = _axis_filtered(X, roles_arr, role_vectors,
                                                vector_meta)
        axis_report["labels_path"] = str(labels_path)
        axis_report["role_vectors"] = vec_report
    else:
        # Fallback: mean(default) - mean(all rows), unfiltered, `default`
        # included in the subtrahend. Kept so the pipeline runs before the
        # filter exists; every figure built on it must carry the deviation.
        axis_unit = np.asarray(assistant_axis(X, meta), dtype=np.float64)
        axis_report = {
            "definition": "mean(default) - mean(ALL rows), unfiltered",
            "deviation": "not the paper's section 3.1 axis: no role-expression "
                         "filter, and `default` is inside the subtrahend. "
                         "Run steering.rolefilter and pass --labels.",
        }
    axis_unit = _require_finite(axis_unit, "assistant axis")

    # Centroids. Filtered to `fully` rows when labels exist, so Arms 2/3 aim at
    # the same object the axis is defined against. float64 per role, never on
    # the whole 2.7 GB array.
    # Steering targets. With labels, a role's target is its `fully` vector —
    # the same object the axis is defined against. A role kept on `somewhat`
    # alone (fully < 10) has no `fully` vector, so its `somewhat` one is the
    # target and the substitution is RECORDED: aiming at "somewhat pirate" is a
    # weaker instrument than aiming at "fully pirate", and a reader of the
    # figure has to be able to see which roles that applied to.
    centroids, centroid_n, target_category = [], {}, {}
    by_key = {}
    if vector_meta is not None:
        by_key = {(m["role"], m["category"]): i for i, m in enumerate(vector_meta)}
    for r in roles:
        # The first category this role has a vector for, `fully` preferred. None
        # when there are no labels at all, and none when the >=10 rule dropped
        # the role (or for `default`) — both take the one unfiltered fallback
        # below rather than two copies of it.
        i = next((by_key[(r, c)] for c in ("fully", "somewhat")
                  if (r, c) in by_key), None)
        if i is not None:
            m = vector_meta[i]
            centroids.append(role_vectors[i])
            centroid_n[r], target_category[r] = m["n"], m["category"]
        else:
            # An unfiltered centroid, so array indices stay aligned with
            # `roles`; excluded from near50/far50 below.
            sel = roles_arr == r
            centroids.append(X[sel].mean(0, dtype=np.float64))
            centroid_n[r], target_category[r] = int(sel.sum()), "unfiltered"
    axis_report["target_category_counts"] = {
        c: sum(1 for v in target_category.values() if v == c)
        for c in ("fully", "somewhat", "unfiltered")}
    axis_report["roles_targeted_by_somewhat"] = sorted(
        r for r, c in target_category.items() if c == "somewhat")
    centroids = _require_finite(np.stack(centroids), "role centroids")
    axis_report["centroid_rows_min"] = int(min(centroid_n.values()))
    # The spline is fit on ALL of these, the `unfiltered` ones included — means
    # dominated by off-role responses, belonging to roles the >=10 rule
    # rejected. run_steering also derives the manifold arm's step from
    # axis_proj.max() - axis_proj.min() over this same array, so the dose
    # geometry is anchored partly by vectors section 2.1.2 says to discard.
    # Recorded, not silently changed: which roles anchor the curve is a design
    # decision rather than a bug.
    axis_report["n_centroids_unfiltered_in_curve"] = int(
        sum(1 for c in target_category.values() if c == "unfiltered"))
    axis_proj = _matmul(centroids, axis_unit, "axis_proj")

    # The dose unit. The paper scales to the average post-MLP residual norm PER
    # TOKEN on LMSYS-Chat-1M (section 3.2.1). `steering.lmsys_norm` measures
    # exactly that; use it when it exists.
    if n_bar_path:
        rep = json.loads(Path(n_bar_path).read_text())
        n_bar = float(rep["n_bar"])
        # Guarded like every other geometry quantity. n_bar is the DOSE UNIT:
        # magnitude = alpha * n_bar, and _rescale_torch zeroes a delta whose
        # norm underflows, so a zero or tiny value generates all 19,000 rows as
        # if unsteered — every figure a flat line, no warning anywhere. Nothing
        # downstream would catch it: _assert_finite_logits runs outside the
        # ActivationSteering context and only ever sees the unsteered pass, and
        # the linear arm's static `addition` path validates no delta at all.
        if not np.isfinite(n_bar) or n_bar <= 0:
            raise ValueError(
                "%s gives n_bar=%r. The dose unit must be finite and positive; "
                "at or near zero every arm generates as if unsteered and the "
                "run looks like a null result." % (n_bar_path, n_bar))
        got_layer = rep.get("layer_hs_index")
        if got_layer is not None and int(got_layer) != LAYER_HS_INDEX:
            raise ValueError(
                "%s was measured at hidden-state layer %d but the cloud and "
                "every arm live at layer %d. `steering.lmsys_norm` writes to a "
                "fixed filename that ignores --layer, so a measurement at "
                "another layer overwrites this one silently."
                % (n_bar_path, int(got_layer), LAYER_HS_INDEX))
        n_bar_report = {"source": "lmsys", "path": str(n_bar_path), **rep}
    else:
        # Fallback. Every row of the cloud is ALREADY a mean over response
        # tokens, so this is a norm-of-mean, not the paper's mean-of-norm, and
        # it is measured on role role-play rather than general chat. Chunked so
        # the squared array never materialises at full size.
        total, count = 0.0, 0
        for s in range(0, X.shape[0], 20000):
            blk = X[s:s + 20000].astype(np.float64, copy=False)
            total += float(np.linalg.norm(blk, axis=1).sum())
            count += blk.shape[0]
        n_bar = total / count
        n_bar_report = {
            "source": "resp240_fallback", "n_bar": n_bar, "n_rows": count,
            "token_basis": "norm of the response-AVERAGED vector",
            "deviation": "not the paper's unit: ||mean h|| understates "
                         "mean ||h||, and the corpus is role role-play rather "
                         "than LMSYS-Chat-1M. Run steering.lmsys_norm and pass "
                         "--n-bar. alpha values are NOT comparable to the paper's.",
        }

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
    excluded = {"default"}
    if vector_meta is not None:
        excluded |= {r for r, c in target_category.items() if c == "unfiltered"}
    order = [i for i in np.argsort(-axis_proj) if roles[i] not in excluded]
    # The two ends must be DISJOINT. `order[:50]` and `order[-50:]` overlap
    # silently as soon as fewer than 100 roles survive the filter: with 60
    # survivors the near and far sets share 40 roles, `pairing` maps roles to
    # themselves, and every figure contrasts "most Assistant-like" against a set
    # that is 80% identical to it — with no warning and entirely ordinary-looking
    # numbers. How many roles survive depends on the labelling coverage
    # (label failures shrink it), so this is reachable, not hypothetical.
    if len(order) < N_NEAR + N_FAR:
        raise ValueError(
            "only %d roles survive the role-expression filter, but near%d and "
            "far%d need %d distinct roles. They would overlap by %d and the "
            "near/far contrast would be meaningless. Check how many rows "
            "steering.rolefilter failed to label, or lower N_NEAR/N_FAR "
            "deliberately." % (len(order), N_NEAR, N_FAR, N_NEAR + N_FAR,
                               N_NEAR + N_FAR - len(order)))
    axis_report["n_roles_eligible_for_ends"] = len(order)
    near50 = [roles[i] for i in order[:N_NEAR]]          # most Assistant-like
    far50 = [roles[i] for i in order[-N_FAR:]][::-1]     # least, most-extreme first

    # Rank-paired: near rank i steers toward far rank i. Fixed, not sampled, so
    # it is reproducible without a seed and cannot drift between runs.
    pairing = {n: f for n, f in zip(near50, far50)}

    return Geometry(roles=roles, centroids=centroids, axis_unit=axis_unit,
                    axis_proj=axis_proj, n_bar=n_bar, spline=spline,
                    lam=float(lam), lam_report=lam_report,
                    near50=near50, far50=far50, pairing=pairing,
                    hidden=int(X.shape[1]),
                    axis_report=axis_report, n_bar_report=n_bar_report)


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
