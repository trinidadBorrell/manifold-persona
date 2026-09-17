"""The cases and the path defaults, shared by the figures and the run.

Plan: plans/2026-09-08-steering-rebuild.md.

WHY THIS MODULE EXISTS. `geometry_check.py` draws the figures that justify eps
and k; `run_steering.py` generates text under the paths those figures describe.
While the case definitions lived inside `geometry_check`, the only thing tying
the two together was that someone had typed the same numbers twice. A figure
that describes a different chord, a different tube radius or a different knot
abscissa than the run is not a figure OF the run -- and nothing would have said
so. Both callers now import the endpoints, the filter and the defaults from
here, so a change to either reaches both or neither.

WHAT IS SHARED
    fully_only     which role centroids are admissible at all
    pick_endpoints the axis segment and the A->B persona pairs
    build_paths    the LinearPath / PersonaPath objects for those cases
    the defaults   EPS, K, LAM, PARAM, MODE, ALPHAS

`fully_only` and `pick_endpoints` moved here verbatim from
`steering/geometry_check.py`, which now imports them.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from steering.manifold_paths import LinearPath, PersonaPath

# --------------------------------------------------------------------------
# The frozen defaults. ONE copy.
# --------------------------------------------------------------------------
# These are `geometry_check`'s own defaults, moved rather than retyped:
# --eps-fig2 9.0, --k-fig 5, --persona-lam 0.0, --param centripetal,
# --eps-mode absolute, and its ALPHAS stop list.
EPS = 12.0                # cylinder radius, ABSOLUTE activation units
                          # (user choice, 2026-09-10, read off the eps/k
                          #  ablation figures before any generation ran)
K = 8                     # persona centroids the curve is routed through
                          # (user choice, same sitting as EPS)
LAM = 0.0                 # 0 = interpolate every chosen centroid exactly
PARAM = "centripetal"     # knot abscissa; see manifold_paths.PersonaPath
MODE = "absolute"         # eps in activation units, not chord fractions

# ALPHA IS ARC POSITION, not a dose. 0 is P0, 1 is P1, and the two arms of a
# pair are sampled at the same values -- same endpoints, same stops, different
# route. The figures put a marker at each of these, so the generated cells and
# the markers are the same points.
# TEN stops including 0. One definition, here -- run_steering imports it rather
# than keeping its own list, so a figure can never describe a different grid from
# the run that produced it. alpha=0 has a zero delta (it IS the unsteered
# condition) but is kept as its own row per arm, so each panel carries its own
# baseline instead of pooling a shared `unsteered` arm at x=0.
ALPHAS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0)


# --------------------------------------------------------------------------
# Admissible centroids
# --------------------------------------------------------------------------

def fully_only(geom):
    """Indices of roles whose centroid is a genuine fully-role-playing vector.

    HARD-FAILS rather than falling through. The previous version read
    `roles_targeted_by_unfiltered` (a key `geometry.py` never writes) and
    `geom.target_category` (a local in `_role_vectors`, not a Geometry field,
    so always None). It therefore dropped only `default` while claiming to
    filter -- the same class of silent fallback this rebuild exists to remove.

    `geometry.py` exposes exactly two usable signals: the per-role
    `roles_targeted_by_somewhat` list, and the count
    `n_centroids_unfiltered_in_curve`. If any centroid fell back to unfiltered
    we cannot say WHICH from the report alone, so we refuse to proceed.
    """
    rep = geom.axis_report or {}
    counts = rep.get("target_category_counts")
    if not counts or not counts.get("fully"):
        raise SystemExit(
            "axis_report has no fully-role-playing counts (%r). Pass --labels "
            "pointing at a role_labels.parquet; without it load_geometry falls "
            "back to unfiltered centroids and this run would claim a filter it "
            "did not apply." % (counts,))
    # `default` ALWAYS takes the unfiltered fallback -- geometry.py:349, "none
    # when the >=10 rule dropped the role (or for `default`)". It is the
    # reference, not a role, and is excluded by name below, so exactly one
    # unfiltered centroid is the expected baseline. More than one means a real
    # role fell back, and axis_report does not record WHICH, so we refuse
    # rather than quietly anchor the curve on a vector 2.1.2 says to discard.
    n_unfiltered = int(rep.get("n_centroids_unfiltered_in_curve", 0) or 0)
    expected = 1 if "default" in geom.roles else 0
    if n_unfiltered > expected:
        raise SystemExit(
            "%d centroids fell back to unfiltered (expected %d, for `default`), "
            "and axis_report does not say which. Re-run steering.rolefilter so "
            "every role has a `fully` vector, or extend geometry.py to record "
            "roles_targeted_by_unfiltered." % (n_unfiltered, expected))
    bad = set(rep.get("roles_targeted_by_somewhat", []) or [])
    keep = [i for i, r in enumerate(geom.roles) if r not in bad and r != "default"]
    return np.array(keep, dtype=int), sorted(bad)


# --------------------------------------------------------------------------
# The cases
# --------------------------------------------------------------------------

def pick_endpoints(C, names, axis_proj, axis_unit):
    """Case definitions.

    The AXIS case travels along the Assistant Axis itself -- a segment of the
    line through the cloud centre in direction `axis_unit`, spanning the same
    axis extent the role centroids reach. It is deliberately NOT the chord
    between the two extreme centroids: that is just another A->B pair, and using
    it made the axis panel identical to summarizer->leviathan.

    The PAIR cases are 3 near-Assistant sources x (3 rank-midway + 2 far).

    ORIENTATION, which the generation run depends on: the axis case runs
    P0 = the ASSISTANT end -> P1 = the far end, so increasing alpha travels AWAY
    from the Assistant. That is the direction the paper's negative alphas take
    (interventions.linear_axis_vector), and it is why the path arms sweep alpha
    in [0, 1] rather than [-1, 0].
    """
    order = np.argsort(axis_proj)
    n_near = [int(i) for i in order[-3:][::-1]]
    mid_c = len(order) // 2
    n_mid = [int(i) for i in order[mid_c - 1:mid_c + 2]]
    n_far = [int(i) for i in order[:2]]

    centre = C.mean(0)
    a = axis_unit / np.linalg.norm(axis_unit)
    off = centre @ a
    P_hi = centre + (axis_proj.max() - off) * a      # Assistant end
    P_lo = centre + (axis_proj.min() - off) * a      # far end

    cases = [("axis", P_hi, P_lo, "axis+", "axis-")]
    for i in n_near:
        for j in n_mid + n_far:
            cases.append(("pair", C[i], C[j], names[i], names[j]))
    return cases, dict(near=[names[i] for i in n_near],
                       mid=[names[i] for i in n_mid],
                       far=[names[i] for i in n_far])


@dataclass
class CaseSet:
    """Everything derived from a Geometry that a path needs.

    Returned as one object because the four pieces are only meaningful
    together: `cases` indexes into `C`, and `C` is already filtered by `keep`,
    so pairing a case list with an unfiltered centroid matrix silently steers
    along a chord between the wrong two roles.
    """
    C: np.ndarray                 # (n_fully, hidden) float64 centroids
    names: List[str]              # role name per row of C
    axis_proj: np.ndarray         # (n_fully,) centroid . axis_unit
    cases: list                   # [(kind, P0, P1, A, B), ...]
    picked: dict                  # which roles filled the near/mid/far slots
    keep: np.ndarray              # indices into geom.roles
    dropped: List[str]            # roles excluded by fully_only


def load_cases(geom, min_roles: int = 10) -> CaseSet:
    """The filter + the endpoints, in the one order both callers need them.

    `geometry_check.main` and `run_steering.main` ran these four steps
    identically; the second copy is what this removes.
    """
    keep, dropped = fully_only(geom)
    if len(keep) < min_roles:
        raise SystemExit("only %d fully-role-playing centroids - labels missing?"
                         % len(keep))
    C = np.asarray(geom.centroids, dtype=np.float64)[keep]
    names = [geom.roles[i] for i in keep]
    axis_proj = np.asarray(geom.axis_proj, dtype=np.float64)[keep]
    cases, picked = pick_endpoints(
        C, names, axis_proj, np.asarray(geom.axis_unit, dtype=np.float64))
    return CaseSet(C=C, names=names, axis_proj=axis_proj, cases=cases,
                   picked=picked, keep=keep, dropped=dropped)


# --------------------------------------------------------------------------
# The paths
# --------------------------------------------------------------------------

def build_paths(cases, C, eps: float = EPS, k: int = K, lam: float = LAM,
                mode: str = MODE, param: str = PARAM):
    """(family, A, B) -> path, one object per route, built once.

    `family` is "linear" or "manifold", which is the half of the arm name that
    selects the route; the other half ("axis" or "pair") is already carried by
    (A, B). Building the bank up front rather than per cell matters: a
    PersonaPath fits a spline over the whole centroid cloud, and the generation
    grid revisits the same route once per alpha.
    """
    bank = {}
    for kind, P0, P1, na, nb in cases:
        bank[("linear", na, nb)] = LinearPath(P0, P1)
        bank[("manifold", na, nb)] = PersonaPath(P0, P1, C, eps, k=k, lam=lam,
                                                 mode=mode, param=param)
    return bank


def path_report(path, names=None) -> dict:
    """Manifest row for one route. Works for either path class.

    LinearPath has no knots to report, so its row is the trivial one -- written
    out in full rather than omitted, because "no row" and "a straight row" read
    the same in a manifest and only one of them means the arm ran.
    """
    if isinstance(path, PersonaPath):
        return path.report(names)
    return {"n_centroids": 0, "eps": None, "k": None, "lam": None,
            "param": None, "detour_ratio": 1.0, "polyline_ratio": 1.0,
            "overshoot": 0.0, "knot_error": 0.0, "excursion": 1.0,
            "centroid_idx": [], "centroid_roles": []}
