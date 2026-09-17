"""Figures for the chunk-selection ablation (plan 2026-09-17-chunk-selection-ablation).

WHY THIS EXISTS
---------------
`steering/chunk_ablation.py` answers one preregistered question — *is equal-count
(density) chunking of the eps-cylinder a no-op against the current equal-width
(distance) chunking?* — and it answers it in CSVs. This module is the only place
those CSVs become pictures, and it is deliberately separate from the ablation
script: the plan records that the figure specs for Results 2-4 are **mine, not
the user's**, and are open for revision. Revising them must cost a replot, not a
rerun of the grid. Splitting the drawing off from the computing is what makes
that true.

WHAT IT DRAWS (the plan's `## Results`, implemented literally)
--------------------------------------------------------------
    fig01  chunk_candidates   Result 1 — the user's own spec, quoted in the plan.
                              Per-chunk strip charts of candidate-to-chord
                              distance, picked candidate highlighted, one panel
                              column per chunking mode, one panel row per
                              distance measure (perpendicular r, cosine).
    fig02  eps_ablation       Result 2 — knot-set Jaccard vs eps, faceted by k.
    fig03  k_ablation         Result 3 — Jaccard vs k faceted by eps, plus the
                              n_centroids confound panel, per mode.
    fig04  m_ablation         Result 4 — knots returned and detour_ratio vs the
                              points-per-chunk m, density mode only.

WHY NO WINNER IS ON ANY FIGURE
------------------------------
The plan's `## Questions` item 4 records that **the deciding metric for path
quality is not set**. Every path-quality number here (detour_ratio, n_centroids,
...) is therefore descriptive: a figure that let one chunking mode read as
"better" would be asserting a verdict the run is not entitled to make, and a
reader who screenshots one panel would carry that verdict away. Every suptitle
says so, in the title rather than the caption, for exactly that reason.
Figures 2-4 additionally carry PROVISIONAL SPEC, because their form is pending
the user's revision (`## Questions`, item 6).

WHY THE ROUTE IS THE UNIT, AND WHY THE BAND IS CLUSTERED
--------------------------------------------------------
There are 16 routes and they are not independent — the 15 pair routes reuse 3
source and 5 target centroids (`### Metric`, "Valid for this data structure?").
So every aggregate here is a **median over routes** with a **route-clustered
bootstrap** 95% CI (2,000 resamples, seed 137): resample routes with
replacement, never rows. And every route is *also* drawn individually as a thin
grey line underneath, so the spread is visible rather than averaged away. The
grid itself is deterministic; the band is the only place a seed enters a figure.

WHY EVERY FIGURE SHIPS A CSV
----------------------------
`steering/README.md`: "a figure whose numbers exist only inside a PNG is not a
result." Each figure writes the exact rows it drew, post-aggregation, to the
same stem with a `.csv` suffix — per-route series and the median/CI series
together, so a reader can re-derive any line without rerunning the grid.

DEFENSIVE BY CONVENTION
-----------------------
`steering/README.md`: "Plot functions are defensive: a plotting failure logs and
returns, and never kills a run that has already produced numbers." The grid
these figures draw costs CPU minutes, but the controls and the deciding metric
are already on disk by the time this module runs; a missing column must not take
them with it. Missing or empty input CSVs are logged and that figure is skipped.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # figures are written, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOG = logging.getLogger("chunk_figures")

# --------------------------------------------------------------------------
# constants fixed by the plan
# --------------------------------------------------------------------------
N_BOOT = 2000
BOOT_SEED = 137          # `### Seeds`: the route-clustered bootstrap seed.

FALSIFIER = 0.95         # `### Metric`: median Jaccard >= 0.95 kills the idea.
WOULD_MATTER = 0.70      # `### Metric`: the effect size that would matter.
CHORD_FLOOR = 1.0        # detour_ratio of the chord itself; no path beats it.

# The frozen production point Result 1 is drawn at (`### Background`).
FIG1_EPS = 12.0
FIG1_K = 8

# Facets, straight from the plan. Fixed and ordered; never data-driven.
# k=1 is carried here on top of the plan's {2,4,8,16}: it is the POSITIVE
# CONTROL (`### Baselines and controls` - at k=1 both modes reduce to one bin
# spanning the whole tube, so Jaccard must be exactly 1.0 for all 16 routes at
# every eps). A control that adjudicates the claim has to be visible on the
# figure that reports the claim, not only in data/controls.json.
K_FACETS = (1, 2, 4, 8, 16)       # Result 2
EPS_FACETS_K = (4, 8, 12, 20)     # Result 3
EPS_FACETS_M = (8, 12, 20)        # Result 4

EPS_XLIM = (0.0, 28.0)            # Result 2
K_XLIM = (1.0, 16.0)              # Result 3
M_XLIM = (1.0, 12.0)              # Result 4
JACCARD_YLIM = (0.0, 1.0)         # Results 2 and 3

# Chunking modes, in the plan's order (the incumbent first). This ordering is
# presentational only — see the suptitles: no winner is declared.
MODES = ("distance", "density")
MODE_COLOR = {"distance": "#4C72B0", "density": "#DD8452"}

PICKED_COLOR = "#C44E52"   # the contrasting colour Result 1 asks for
CAND_COLOR = "#8C8C8C"
ROUTE_COLOR = "#9E9E9E"    # "thin grey line per route", Results 2-4
MEDIAN_COLOR = "#222222"
BAND_COLOR = "#4C72B0"

NO_WINNER = ("NO WINNER DECLARED between chunking modes "
             "(path-quality metric is open: plan Questions item 4)")
PROVISIONAL = "PROVISIONAL SPEC (plan Questions item 6)"


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------
def _read_csv(data: Path, name: str,
              required: Sequence[str]) -> Optional[pd.DataFrame]:
    """Read one input CSV, or log why there will be no figure and return None.

    Three separate failures are reported separately — absent, empty, and present
    but wrong shape — because they mean different things to whoever is reading
    the log: the ablation did not get that far, the ablation produced nothing, or
    the ablation's schema and this module's have drifted apart.
    """
    path = data / name
    if not path.exists():
        LOG.warning("missing input %s - skipping the figure that needs it", path)
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # A zero-byte file is the normal shape of "the ablation was interrupted
        # before it wrote this one". It is a warning, not a stack trace.
        LOG.warning("%s has no header - skipping the figure that needs it", path)
        return None
    except Exception:  # noqa: BLE001 - a bad CSV must not kill the run
        LOG.exception("could not read %s - skipping the figure that needs it", path)
        return None
    if df.empty:
        LOG.warning("%s is empty - skipping the figure that needs it", path)
        return None
    missing = [c for c in required if c not in df.columns]
    if missing:
        LOG.warning("%s lacks column(s) %s - skipping the figure that needs it",
                    path, ", ".join(missing))
        return None
    LOG.info("read %s (%d rows, %d routes)", path, len(df),
             df["route"].nunique() if "route" in df else -1)
    return df


def _as_bool(s: pd.Series) -> np.ndarray:
    """`is_picked` as a real boolean, whatever the CSV round-trip made of it.

    pandas reads an unquoted `True` as bool and a quoted one as the string
    "True"; either would silently select nothing under a naive `== True`, and a
    figure with no picked markers looks like a selection bug rather than a
    parsing one.
    """
    if s.dtype == bool:
        return s.to_numpy()
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(float).to_numpy() > 0.5
    return (s.astype(str).str.strip().str.lower()
            .isin({"true", "1", "yes", "t"}).to_numpy())


# --------------------------------------------------------------------------
# statistics — route-clustered bootstrap on the MEDIAN
# --------------------------------------------------------------------------
def _nanmedian(a: np.ndarray, axis: int) -> np.ndarray:
    """nanmedian without the all-NaN-slice warning spam.

    An all-NaN column is normal here: a route can be absent from one grid cell
    (eps=0 selects nothing), and NaN is the honest value to plot there.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _route_matrix(sub: pd.DataFrame, xcol: str,
                  ycol: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """(routes x x-values) matrix, the x grid, and the route names.

    Collapsing to one row per route *before* any aggregation is what makes the
    route the unit of observation, exactly as `figures_steering.py` collapses to
    one row per role. Everything downstream sees only this matrix.
    """
    piv = sub.pivot_table(index="route", columns=xcol, values=ycol,
                          aggfunc="median")
    piv = piv.sort_index(axis=1).sort_index(axis=0)
    return (piv.to_numpy(dtype=float),
            [str(r) for r in piv.index],
            piv.columns.to_numpy(dtype=float))


def _boot_median(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median across routes and a route-clustered percentile 95% CI.

    Resamples ROUTES with replacement, 2,000 times, seed 137 (`### Seeds`). The
    resample indices are drawn ONCE and shared across every x value, so each
    replicate is one coherent pseudo-experiment — the same 16 routes read across
    the whole eps sweep — rather than an independent reshuffle per point, which
    would produce a band that wiggles for no physical reason.
    """
    n_routes = mat.shape[0]
    med = _nanmedian(mat, axis=0)
    if n_routes < 2:
        # One route carries no across-route variability; a CI would be fiction.
        return med, np.full_like(med, np.nan), np.full_like(med, np.nan)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n_routes, size=(N_BOOT, n_routes))
    reps = _nanmedian(mat[idx], axis=1)          # (N_BOOT, n_x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanpercentile(reps, 2.5, axis=0)
        hi = np.nanpercentile(reps, 97.5, axis=0)
    return med, lo, hi


def _series_rows(sub: pd.DataFrame, xcol: str, ycol: str,
                 facet_col: str, facet_val, mode: Optional[str]
                 ) -> Tuple[Dict, List[Dict]]:
    """Everything one curve needs, plus its rows for the figure's CSV.

    The CSV is built here, from the same arrays the axes are handed, so the
    shipped numbers cannot drift from the drawn ones.
    """
    mat, routes, x = _route_matrix(sub, xcol, ycol)
    med, lo, hi = _boot_median(mat)
    rows: List[Dict] = []
    for i, route in enumerate(routes):
        for j, xv in enumerate(x):
            rows.append({"series": "route", "metric": ycol, "route": route,
                         "mode": mode,
                         facet_col: facet_val, xcol: xv, ycol: mat[i, j],
                         "ci_lo": np.nan, "ci_hi": np.nan, "n_routes": np.nan})
    for j, xv in enumerate(x):
        rows.append({"series": "median", "metric": ycol, "route": "__median__",
                     "mode": mode,
                     facet_col: facet_val, xcol: xv, ycol: med[j],
                     "ci_lo": lo[j], "ci_hi": hi[j],
                     "n_routes": int(np.isfinite(mat[:, j]).sum())})
    curve = {"x": x, "mat": mat, "routes": routes,
             "med": med, "lo": lo, "hi": hi}
    return curve, rows


def _draw_curve(ax, curve: Dict, color: str, label: Optional[str],
                grey_routes: bool = True) -> None:
    """Per-route lines thin and underneath, the median heavy on top with its band.

    Order matters: the routes are drawn first at low zorder so the median is
    never hidden by 16 grey lines, and the band is drawn under the median line
    so a narrow CI still shows the estimate.
    """
    x, mat = curve["x"], curve["mat"]
    if grey_routes:
        for i in range(mat.shape[0]):
            ok = np.isfinite(mat[i])
            if ok.sum() >= 1:
                ax.plot(x[ok], mat[i][ok], color=ROUTE_COLOR, lw=0.6,
                        alpha=0.45, zorder=1)
    else:
        for i in range(mat.shape[0]):
            ok = np.isfinite(mat[i])
            if ok.sum() >= 1:
                ax.plot(x[ok], mat[i][ok], color=color, lw=0.6, alpha=0.25,
                        zorder=1)
    lo, hi, med = curve["lo"], curve["hi"], curve["med"]
    band = np.isfinite(lo) & np.isfinite(hi)
    if band.any():
        ax.fill_between(x[band], lo[band], hi[band], color=color, alpha=0.18,
                        lw=0, zorder=2)
    ok = np.isfinite(med)
    if ok.any():
        ax.plot(x[ok], med[ok], color=color, lw=2.0, marker="o", ms=3.5,
                label=label, zorder=3)


def _empty_panel(ax, msg: str = "no data for this panel") -> None:
    """Say it, in red, rather than leave an empty frame that reads as a flat zero."""
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#B00020")


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def _save(fig, png: Path, rows: List[Dict],
          title_lines: int = 2, cap_lines: int = 2) -> Path:
    """Write the PNG at 300 dpi and, beside it, the rows it was drawn from.

    The reserved margins are computed in INCHES and converted, not hard-coded as
    fractions: these figures range from 4.2 to 7.6 inches tall, and a fixed
    fraction either crushes the caption on the short ones or leaves a band of
    white across the tall ones.
    """
    h = max(float(fig.get_figheight()), 1.0)
    top = 1.0 - (0.18 + 0.17 * title_lines) / h
    bottom = (0.08 + 0.15 * cap_lines) / h
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(str(png), dpi=300)
    plt.close(fig)
    csv = png.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    LOG.info("wrote %s (%d bytes) and %s (%d rows)",
             png, png.stat().st_size, csv, len(rows))
    return png


def _suptitle(fig, head: str, run_id: str, provisional: bool) -> None:
    """Run id, the no-winner statement, and PROVISIONAL where the plan says so.

    In the title, not the caption: a reader who screenshots one panel still has
    to see that this run declares no winner between the modes.
    """
    lines = ["%s  |  run %s" % (head, run_id), NO_WINNER]
    if provisional:
        lines.append(PROVISIONAL)
    fig.suptitle("\n".join(lines), fontsize=10,
                 color="#B00020" if provisional else "black")


def _caption(fig, text: str) -> None:
    """Figure-level note; an in-panel one lands on top of the data it annotates."""
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=7,
             color="#444444", wrap=True)


# --------------------------------------------------------------------------
# Result 1 — fig01, the user's own spec
# --------------------------------------------------------------------------
def fig01_chunk_candidates(data: Path, figures: Path, layer: int,
                           run_id: str) -> Optional[Path]:
    """Per-chunk candidate distributions, picked candidate marked.

    The user's words, quoted in the plan: *"show that effectively the cos
    distance between the points in the chunk considered and the point added to
    the manifold path was the best option possible, so maybe show the
    distribution of distances and show with a vertical line the limit of the
    diameter of the cylinder"*.

    Implemented as: one horizontal strip per chunk, chunks stacked down the page
    ordered by chord position u ascending (NEVER by the distance being
    measured); one panel column per chunking mode; one panel row per distance
    measure, so the reader can see whether the two orderings agree — the
    perpendicular distance r that selection actually uses, and the cosine
    distance the user named. The tube-radius line goes on the r panels, where it
    is the real selection boundary, and is OMITTED on the cosine panels, where
    it has no meaning; the caption states the omission.
    """
    need = ["route", "chunk", "bin", "u_lo", "u_hi", "cand_idx", "cand_role",
            "cand_u", "cand_r", "cand_cos", "is_picked", "tube_radius"]
    df = _read_csv(data, "fig01_candidates.csv", need)
    if df is None:
        return None

    # One representative route, at the frozen production point. The plan says
    # "one representative route" without naming it, so the choice is made
    # deterministically (first in file order) and stated on the figure.
    route = str(df["route"].iloc[0])
    sub = df[df["route"].astype(str) == route].copy()
    if sub.empty:
        LOG.warning("fig01: no rows for route %s - skipping", route)
        return None
    sub["is_picked"] = _as_bool(sub["is_picked"])
    tube_radius = float(pd.to_numeric(sub["tube_radius"], errors="coerce").max())

    modes = [m for m in MODES if m in set(sub["chunk"].astype(str))]
    modes += sorted(set(sub["chunk"].astype(str)) - set(MODES))
    if not modes:
        LOG.warning("fig01: no chunking modes in the candidate table - skipping")
        return None

    metrics = [("cand_r", "perpendicular distance to chord r  (activation units)",
                True),
               ("cand_cos", "cosine distance  1 - cos(C_i - P0, P1 - P0)", False)]

    fig, axes = plt.subplots(len(metrics), len(modes),
                             figsize=(5.6 * len(modes), 3.4 * len(metrics)),
                             squeeze=False)
    for r, (col, xlabel, is_r) in enumerate(metrics):
        for c, mode in enumerate(modes):
            ax = axes[r][c]
            m = sub[sub["chunk"].astype(str) == mode]
            if m.empty:
                _empty_panel(ax, "no candidates for mode %s" % mode)
                continue
            # y order is fixed by chord position u ascending — the plan is
            # explicit that it is never sorted by the measured quantity.
            order = (m.groupby("bin")["u_lo"].min().sort_values().index.tolist())
            ypos = {b: i for i, b in enumerate(order)}
            y = m["bin"].map(ypos).to_numpy(dtype=float)
            x = pd.to_numeric(m[col], errors="coerce").to_numpy(dtype=float)
            picked = m["is_picked"].to_numpy(dtype=bool)
            ax.scatter(x[~picked], y[~picked], s=22, facecolors="none",
                       edgecolors=CAND_COLOR, linewidths=0.8, zorder=2,
                       label="candidate in chunk")
            ax.scatter(x[picked], y[picked], s=52, color=PICKED_COLOR,
                       marker="o", zorder=3, label="picked knot")
            if is_r:
                # The user's "limit of the diameter of the cylinder": eps, the
                # tube radius in absolute activation units, which is where
                # select_cylinder actually cuts.
                ax.axvline(tube_radius, color="#B00020", lw=1.1, ls="--",
                           alpha=0.8, zorder=1,
                           label="tube radius eps = %.3g" % tube_radius)
                ax.set_xlim(0.0, 1.15 * tube_radius)
            else:
                lo = float(np.nanmin(x)) if np.isfinite(x).any() else 0.0
                hi = float(np.nanmax(x)) if np.isfinite(x).any() else 1.0
                pad = 0.075 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
                ax.set_xlim(min(0.0, lo - pad), hi + pad)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(
                ["%s  u[%.2f,%.2f]" % (b,
                                       float(m.loc[m["bin"] == b, "u_lo"].min()),
                                       float(m.loc[m["bin"] == b, "u_hi"].max()))
                 for b in order], fontsize=7)
            ax.invert_yaxis()   # chunk 0 at the top: the chord read top-down
            ax.set_xlabel(xlabel, fontsize=8)
            if c == 0:
                # Left column only: the label is long, and repeating it on the
                # right column runs it back across the left panel's ticks.
                ax.set_ylabel("chunk (ordered by chord position u, ascending)",
                              fontsize=8)
            ax.set_title("%s chunking - %d candidates, %d chunks"
                         % (mode, len(m), len(order)), fontsize=9)
            ax.grid(alpha=0.22, lw=0.5, axis="x")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best", framealpha=0.9)

    _suptitle(fig, "fig01 - candidates per chunk, route %s, eps=%g k=%d"
              % (route, FIG1_EPS, FIG1_K), run_id, provisional=False)
    _caption(fig,
             "One point = one candidate centroid returned by select_cylinder. "
             "Filled marker = the knot the rule picked (minimum perpendicular "
             "distance r). Exact values, no estimates, so no error bars. "
             "The dashed tube-radius line is drawn on the r panels, where it is "
             "the selection boundary; it is OMITTED on the cosine panels, where "
             "a radius in activation units has no meaning.")
    return _save(fig, figures / ("fig01_chunk_candidates_L%d.png" % layer),
                 sub.to_dict("records"), title_lines=2, cap_lines=2)


# --------------------------------------------------------------------------
# Result 2 — fig02, eps ablation (provisional spec)
# --------------------------------------------------------------------------
def fig02_eps_ablation(data: Path, figures: Path, layer: int,
                       run_id: str) -> Optional[Path]:
    """Knot-set Jaccard vs eps, faceted by k. One point = one (route, eps, k)."""
    df = _read_csv(data, "jaccard_summary.csv",
                   ["route", "eps", "k", "jaccard"])
    if df is None:
        return None
    df = df.copy()
    for c in ("eps", "k", "jaccard"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    fig, axes = plt.subplots(1, len(K_FACETS), figsize=(4.1 * len(K_FACETS), 4.2),
                             squeeze=False, sharey=True)
    rows: List[Dict] = []
    for i, kf in enumerate(K_FACETS):
        ax = axes[0][i]
        sub = df[np.isclose(df["k"], kf)]
        if sub.empty:
            _empty_panel(ax, "no cells at k=%d" % kf)
        else:
            curve, r = _series_rows(sub, "eps", "jaccard", "k", kf, None)
            rows.extend(r)
            _draw_curve(ax, curve, MEDIAN_COLOR,
                        "median over routes (95%% CI, %d resamples)" % N_BOOT)
        ax.axhline(FALSIFIER, color="#B00020", lw=1.1, ls="--", alpha=0.85,
                   label="0.95 falsifier" if i == 0 else None)
        ax.axhline(WOULD_MATTER, color="#2E7D32", lw=1.1, ls=":", alpha=0.85,
                   label="0.70 would matter" if i == 0 else None)
        ax.set_xlim(*EPS_XLIM)
        ax.set_ylim(*JACCARD_YLIM)
        ax.set_xlabel("eps  (tube radius, activation units)", fontsize=9)
        if i == 0:
            ax.set_ylabel("knot-set Jaccard, distance vs density", fontsize=9)
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax.set_title("k = %d" % kf, fontsize=10)
        ax.grid(alpha=0.22, lw=0.5)

    _suptitle(fig, "fig02 - eps ablation: knot-set Jaccard between chunking modes",
              run_id, provisional=True)
    _caption(fig,
             "Thin grey lines = the 16 individual routes (not independent: the "
             "15 pair routes reuse 3 source and 5 target centroids). Heavy line "
             "= median over routes; band = route-clustered bootstrap 95%% CI on "
             "the median, %d resamples, seed %d. Reference lines: 0.95 = the "
             "preregistered falsifier, 0.70 = the effect size that would matter. "
             "The k=1 panel is the POSITIVE CONTROL: at one chunk both modes "
             "reduce to the same primitive, so a flat line at Jaccard = 1.0 "
             "across every eps is the control PASSING, not a degenerate facet."
             % (N_BOOT, BOOT_SEED))
    return _save(fig, figures / ("fig02_eps_ablation_L%d.png" % layer), rows,
                 title_lines=3, cap_lines=1)


# --------------------------------------------------------------------------
# Result 3 — fig03, k ablation (provisional spec)
# --------------------------------------------------------------------------
def fig03_k_ablation(data: Path, figures: Path, layer: int,
                     run_id: str) -> Optional[Path]:
    """Jaccard vs k faceted by eps, plus the n_centroids confound panel per mode.

    The second row is the confound the plan names in `### Baselines and
    controls`: a Jaccard below 1 can come from the two modes returning different
    *numbers* of knots rather than different *choices*, because equal-width
    spans go empty. Reading the two rows together is the only way to tell those
    apart, which is why they share a figure and an x-axis.
    """
    jac = _read_csv(data, "jaccard_summary.csv", ["route", "eps", "k", "jaccard"])
    grid = _read_csv(data, "grid_eps_k.csv",
                     ["route", "eps", "k", "chunk", "n_centroids"])
    if jac is None and grid is None:
        return None
    if jac is not None:
        jac = jac.copy()
        for c in ("eps", "k", "jaccard"):
            jac[c] = pd.to_numeric(jac[c], errors="coerce")
    if grid is not None:
        grid = grid.copy()
        for c in ("eps", "k", "n_centroids"):
            grid[c] = pd.to_numeric(grid[c], errors="coerce")

    fig, axes = plt.subplots(2, len(EPS_FACETS_K),
                             figsize=(4.1 * len(EPS_FACETS_K), 7.6),
                             squeeze=False)
    rows: List[Dict] = []
    for i, ef in enumerate(EPS_FACETS_K):
        # --- row 1: the deciding metric
        ax = axes[0][i]
        sub = None if jac is None else jac[np.isclose(jac["eps"], ef)]
        if sub is None or sub.empty:
            _empty_panel(ax, "no cells at eps=%g" % ef)
        else:
            curve, r = _series_rows(sub, "k", "jaccard", "eps", ef, None)
            rows.extend(r)
            _draw_curve(ax, curve, MEDIAN_COLOR,
                        "median over routes (95%% CI, %d resamples)" % N_BOOT)
        ax.axhline(FALSIFIER, color="#B00020", lw=1.1, ls="--", alpha=0.85,
                   label="0.95 falsifier" if i == 0 else None)
        ax.axhline(WOULD_MATTER, color="#2E7D32", lw=1.1, ls=":", alpha=0.85,
                   label="0.70 would matter" if i == 0 else None)
        ax.set_xlim(*K_XLIM)
        ax.set_ylim(*JACCARD_YLIM)
        ax.set_title("eps = %g" % ef, fontsize=10)
        ax.set_xlabel("k  (number of chunks)", fontsize=9)
        if i == 0:
            ax.set_ylabel("knot-set Jaccard, distance vs density", fontsize=9)
            ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
        ax.grid(alpha=0.22, lw=0.5)

        # --- row 2: the n_centroids confound, per mode
        ax2 = axes[1][i]
        gsub = None if grid is None else grid[np.isclose(grid["eps"], ef)]
        if gsub is None or gsub.empty:
            _empty_panel(ax2, "no cells at eps=%g" % ef)
        else:
            for mode in MODES:
                msub = gsub[gsub["chunk"].astype(str) == mode]
                if msub.empty:
                    continue
                curve, r = _series_rows(msub, "k", "n_centroids", "eps", ef, mode)
                rows.extend(r)
                # Per-route lines here are tinted by mode rather than grey: two
                # modes overlay in one panel, and grey would make it impossible
                # to tell whose spread is whose.
                _draw_curve(ax2, curve, MODE_COLOR[mode],
                            "%s (median)" % mode, grey_routes=False)
            # The k=identity line: the most knots any mode can return.
            kk = np.array(sorted(gsub["k"].dropna().unique()), dtype=float)
            if kk.size:
                ax2.plot(kk, kk, color="#666666", lw=0.9, ls="--", alpha=0.7,
                         label="n = k (ceiling)" if i == 0 else None, zorder=1)
        ax2.set_xlim(*K_XLIM)
        ax2.set_xlabel("k  (number of chunks)", fontsize=9)
        if i == 0:
            ax2.set_ylabel("n_centroids returned", fontsize=9)
            ax2.legend(fontsize=7, loc="upper left", framealpha=0.9)
        ax2.grid(alpha=0.22, lw=0.5)

    _suptitle(fig, "fig03 - k ablation: Jaccard, and the n_centroids confound",
              run_id, provisional=True)
    _caption(fig,
             "Top row: knot-set Jaccard, thin grey = the 16 routes, heavy = "
             "median, band = route-clustered bootstrap 95%% CI, %d resamples, "
             "seed %d; reference lines 0.95 (falsifier) and 0.70 (effect size "
             "that would matter). Bottom row: knots actually returned per mode - "
             "the confound that a Jaccard below 1 can come from count rather "
             "than choice. The 0.95/0.70 lines are Jaccard values and are drawn "
             "on the top row only. Neither mode is called better here."
             % (N_BOOT, BOOT_SEED))
    return _save(fig, figures / ("fig03_k_ablation_L%d.png" % layer), rows,
                 title_lines=3, cap_lines=2)


# --------------------------------------------------------------------------
# Result 4 — fig04, points-per-chunk m (provisional spec)
# --------------------------------------------------------------------------
def fig04_m_ablation(data: Path, figures: Path, layer: int,
                     run_id: str) -> Optional[Path]:
    """Knots returned and detour_ratio vs m, density mode only, faceted by eps.

    `k` is ignored in this mode; the bin count is ceil(n_candidates / m). The
    two rows are the tradeoff m controls, so that a later run can pick m
    deliberately instead of by default.
    """
    df = _read_csv(data, "grid_m.csv",
                   ["route", "eps", "m", "n_centroids", "detour_ratio"])
    if df is None:
        return None
    df = df.copy()
    for c in ("eps", "m", "n_centroids", "detour_ratio"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "chunk" in df.columns:
        # Belt and braces: the m grid is density-only by design, but a stray
        # distance row would quietly contaminate the median.
        keep = df["chunk"].astype(str) == "density"
        if keep.any():
            df = df[keep]

    panels = [("n_centroids", "knots returned", False),
              ("detour_ratio", "detour_ratio", True)]
    fig, axes = plt.subplots(len(panels), len(EPS_FACETS_M),
                             figsize=(4.1 * len(EPS_FACETS_M), 3.8 * len(panels)),
                             squeeze=False)
    rows: List[Dict] = []
    for r, (col, ylabel, is_detour) in enumerate(panels):
        for i, ef in enumerate(EPS_FACETS_M):
            ax = axes[r][i]
            sub = df[np.isclose(df["eps"], ef)]
            if sub.empty:
                _empty_panel(ax, "no cells at eps=%g" % ef)
            else:
                curve, rws = _series_rows(sub, "m", col, "eps", ef, "density")
                rows.extend(rws)
                _draw_curve(ax, curve, MEDIAN_COLOR,
                            "median over routes (95%% CI, %d resamples)" % N_BOOT)
            if is_detour:
                ax.axhline(CHORD_FLOOR, color="#B00020", lw=1.1, ls="--",
                           alpha=0.85,
                           label="1.0 = the chord (floor)" if i == 0 else None)
            ax.set_xlim(*M_XLIM)
            ax.set_xlabel("m  (candidates per chunk)", fontsize=9)
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=9)
                ax.legend(fontsize=7, loc="best", framealpha=0.9)
            if r == 0:
                ax.set_title("eps = %g" % ef, fontsize=10)
            ax.grid(alpha=0.22, lw=0.5)

    _suptitle(fig, "fig04 - points-per-chunk m ablation, density mode only",
              run_id, provisional=True)
    _caption(fig,
             "Density chunking only; k is ignored and the bin count is "
             "ceil(n_candidates / m). Thin grey = the 16 routes, heavy = median, "
             "band = route-clustered bootstrap 95%% CI on the median, %d "
             "resamples, seed %d. The detour_ratio reference line at 1.0 is the "
             "chord itself, the floor no path can beat. This is the tradeoff m "
             "controls, not a recommendation of an m." % (N_BOOT, BOOT_SEED))
    return _save(fig, figures / ("fig04_m_ablation_L%d.png" % layer), rows,
                 title_lines=3, cap_lines=2)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def make_figures(data: Path, figures: Path, layer: int = 19,
                 run_id: Optional[str] = None) -> List[Path]:
    """Draw all four figures. Never raises.

    Called at the tail of a run that already has its controls and its deciding
    metric on disk. Each figure is attempted independently, and a failure is
    logged and skipped, per the README convention.
    """
    data, figures = Path(data), Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        # The run dir is `<...>/<YYYY-MM-DDTHH-MM>-<slug>/data`, so its parent
        # names the run (`runmeta.build_manifest` uses the same name).
        run_id = data.parent.name if data.name == "data" else data.name
    written: List[Path] = []
    for fn in (fig01_chunk_candidates, fig02_eps_ablation,
               fig03_k_ablation, fig04_m_ablation):
        try:
            out = fn(data, figures, layer, run_id)
        except Exception:  # noqa: BLE001 - a figure must not kill the caller
            LOG.exception("failed to draw %s; continuing", fn.__name__)
            plt.close("all")
            continue
        if out is not None:
            written.append(out)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="run's data/ dir, holding the chunk_ablation.py CSVs")
    ap.add_argument("--figures", required=True, type=Path,
                    help="run's figures/ dir; PNGs and their CSVs land here")
    ap.add_argument("--layer", type=int, default=19,
                    help="layer index, used only in the filenames")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    written = make_figures(args.data, args.figures, layer=args.layer)
    LOG.info("%d/4 figures written", len(written))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
