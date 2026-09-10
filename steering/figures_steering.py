"""Figure-4 analogues for the steering arms (arXiv:2601.10387, Fig. 4).

WHY THIS EXISTS
---------------
The paper's Figure 4 reads a steering intervention by asking a single question:
as the steering strength grows, where does the response *land*? Not "is it more
persona-like on a scalar", which hides collapse and degradation inside one
number, but "what fraction of responses is now on-target, off-target,
theatrical, or broken". This module draws that same reading for our three arms
so the comparison to the paper is a comparison of like with like.

Two figures under the current plan, three for legacy runs:

    fig01  linear_axis      one panel   - straight along the Assistant Axis
    fig02  manifold_axis    one panel   - along the fitted curve, same axis

    (legacy: arm1_axis / arm2_linear / arm3_manifold, the target-directed
     design, drawn as near|far panel pairs.)

Both arms are drawn on ONE shared set of y-limits. The whole point of the pair
is that they travel the same direction by different paths, so the figures have
to be physically superimposable; a per-figure autoscale would let a difference
in axis range masquerade as a difference in behaviour.

WHY THE ROLE IS THE UNIT
------------------------
A role's responses share its instruction set and the shared question set, so
they are not independent draws. Averaging over responses would report a
precision the design does not have. We therefore compute each category fraction
*within* a role and then average across roles, and bootstrap over roles.

WHY NOT ``stats_utils.boot_ci``
-------------------------------
``exploratory/per_persona/stats_utils.py::boot_ci(x, y, Z, rng, n_boot)``
bootstraps a *partial correlation* between two variables. There is no
correlation here — the estimand is the mean, across roles, of a per-role
fraction. Its signature does not fit and forcing it would mean passing dummy
``y``/``Z``. The role-resampling logic it embodies is reproduced below in
:func:`_role_bootstrap`, which resamples the same way (roles, with replacement).

EXPLORATORY
-----------
Every title carries the word EXPLORATORY. These are descriptive category
fractions with no pre-registered test attached, and the word belongs where a
reader who screenshots one panel will still see it.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # figures are written, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from steering.judge import (  # noqa: E402
    CATEGORIES,
    COLLAPSE,
    DEGRADED,
    OFF_TARGET,
    ON_TARGET,
    PLOT_CATEGORIES,
    PLOT_LABEL,
    RESIDUAL_CATEGORIES,
)

LOG = logging.getLogger("figures_steering")

N_BOOT = 2000
BOOT_SEED = 0

# One colour per category, fixed here so the same category is the same colour in
# all three figures and panels can be read against each other without a re-read
# of the legend.
_PALETTE = {
    "assistant": "#4C72B0",
    "nonhuman_role": "#55A868",
    "human_role": "#2E7D32",
    "weird_role": "#C44E52",
    "ambiguous": "#8C8C8C",
    "other": "#DD8452",
    "nonsensical": "#8172B3",
}

# Reading group shown next to each category in the legend. Built from the
# constants in judge.py rather than retyped, so a regrouping there propagates.
_GROUP_OF = {}
for _c in CATEGORIES:
    if _c in ON_TARGET:
        _GROUP_OF[_c] = "on-target"
    elif _c in OFF_TARGET:
        _GROUP_OF[_c] = "off-target"
    elif _c in COLLAPSE:
        _GROUP_OF[_c] = "collapse"
    elif _c in DEGRADED:
        _GROUP_OF[_c] = "degraded"
    else:
        _GROUP_OF[_c] = ""


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def _role_fraction_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Per-role category fractions: rows = roles, columns = ``CATEGORIES``.

    Collapsing to one row per role before any averaging is what makes the role
    the unit of observation; every statistic downstream sees only this matrix.
    """
    roles = sorted(df["role"].dropna().unique().tolist())
    mat = np.zeros((len(roles), len(CATEGORIES)), dtype=float)
    for i, role in enumerate(roles):
        scores = df.loc[df["role"] == role, "judge_score"]
        n = int(scores.notna().sum())
        if n == 0:
            mat[i, :] = np.nan
            continue
        counts = scores.value_counts()
        for j, cat in enumerate(CATEGORIES):
            mat[i, j] = float(counts.get(cat, 0)) / n
    keep = ~np.isnan(mat).any(axis=1)
    return mat[keep], [r for r, k in zip(roles, keep) if k]


def _role_bootstrap(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean across roles and a percentile 95% CI from resampling roles.

    The resample indices are drawn ONCE and shared by all seven categories, so
    each bootstrap replicate is a coherent pseudo-experiment (the same set of
    roles) rather than seven unrelated ones. Seeded per call so a figure redrawn
    tomorrow has the same whiskers as the one in the write-up.
    """
    n_roles = mat.shape[0]
    mean = mat.mean(axis=0)
    if n_roles < 2:
        # A single role carries no across-role variability; a CI would be a
        # zero-width fiction. Report the point and no interval.
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n_roles, size=(N_BOOT, n_roles))
    reps = mat[idx].mean(axis=1)  # (N_BOOT, n_categories)
    lo = np.percentile(reps, 2.5, axis=0)
    hi = np.percentile(reps, 97.5, axis=0)
    return mean, lo, hi


def _curve_stats(df: pd.DataFrame) -> Optional[Dict]:
    """Category fractions and CIs at every alpha present in ``df``.

    Returns ``None`` when nothing survives, so callers branch on data presence
    instead of on an empty container that plots as a convincing flat line.
    """
    if df is None or df.empty:
        return None
    alphas = sorted(float(a) for a in df["alpha"].dropna().unique())
    out = {"alpha": [], "mean": [], "lo": [], "hi": [], "n_roles": []}
    for a in alphas:
        sub = df[np.isclose(df["alpha"].astype(float), a)]
        mat, roles = _role_fraction_matrix(sub)
        if mat.shape[0] == 0:
            continue
        mean, lo, hi = _role_bootstrap(mat)
        out["alpha"].append(a)
        out["mean"].append(mean)
        out["lo"].append(lo)
        out["hi"].append(hi)
        out["n_roles"].append(len(roles))
    if not out["alpha"]:
        return None
    for k in ("mean", "lo", "hi"):
        out[k] = np.vstack(out[k])
    out["alpha"] = np.asarray(out["alpha"], dtype=float)
    return out


def _negctl_reference(df: pd.DataFrame) -> Optional[Dict]:
    """Per-category negative-control band, RESOLVED BY ALPHA.

    Deliberately NOT pooled across the control's strengths. The control answers
    "at MATCHED DOSE, does a direction that means nothing do what a steering
    direction does?", and that question is dose-dependent: a random push at
    alpha=0.5 does little, while at alpha=3.0 it is expected to degrade the
    output badly. Pooling the three strengths into one horizontal band would
    overstate the control at low alpha, understate it at high alpha, and make
    the only comparison the control exists for unreadable off the figure.

    The plan generates the control at exactly three strengths ({0.5, 1.5, 3.0},
    seeds 0-2) precisely so it can be read against alpha; collapsing them
    discards the structure those generations paid for.

    Seeds ARE pooled — they are replicates of the same condition, and their
    spread is inside the bootstrap CI.
    """
    neg = df[df["arm"] == "negctl"]
    if neg.empty:
        return None
    stats = _curve_stats(neg)
    if stats is None or not len(stats["alpha"]):
        return None
    return {"alpha": stats["alpha"], "mean": stats["mean"],
            "lo": stats["lo"], "hi": stats["hi"],
            "n_roles": int(np.nanmax(stats["n_roles"])) if len(stats["n_roles"]) else 0}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def _arm_frame(df: pd.DataFrame, arm: str,
               target_distance: Optional[str]) -> pd.DataFrame:
    """Rows for one panel: one arm, one target distance, alpha > 0.

    alpha == 0 is dropped from the arm itself because the shared ``unsteered``
    rows supply the x=0 point for every panel (see :func:`_with_unsteered`).
    Two points stacked at x=0 would be a plotting artefact, not a measurement.
    """
    sub = df[df["arm"] == arm]
    if target_distance is not None and "target_distance" in sub.columns:
        sub = sub[sub["target_distance"] == target_distance]
    return sub[~np.isclose(sub["alpha"].astype(float), 0.0)]


def _with_unsteered(df: pd.DataFrame, arm_rows: pd.DataFrame) -> pd.DataFrame:
    """Prepend the shared alpha=0 baseline so every panel starts from it."""
    base = df[df["arm"] == "unsteered"].copy()
    if not base.empty:
        base["alpha"] = 0.0
    if arm_rows.empty:
        return base
    return pd.concat([base, arm_rows], ignore_index=True)


_PLOT_IDX = [CATEGORIES.index(c) for c in PLOT_CATEGORIES]
_RESID_IDX = [CATEGORIES.index(c) for c in RESIDUAL_CATEGORIES]


def _panel_ymax(stats: Optional[Dict], neg: Optional[Dict]) -> float:
    """Largest value a panel needs to show, error bars and band included.

    OVER THE CATEGORIES ACTUALLY DRAWN. Taking the max over all seven let
    `ambiguous` and `other` — scored but deliberately not plotted (see
    RESIDUAL_CATEGORIES) — set the ceiling, so a panel whose five curves peak at
    0.3 was framed to 0.9 and every one of them read as flat. Pass `neg=None`
    when the control band is not being drawn, for the same reason.
    """
    vals = [0.0]
    for d in (stats, neg):
        if d is None:
            continue
        vals.append(float(np.nanmax(d["mean"][:, _PLOT_IDX])))
        hi = d["hi"][:, _PLOT_IDX]
        if np.isfinite(hi).any():
            vals.append(float(np.nanmax(hi)))
    return max(vals)


# The paper's x-axis (Figure 4) is a SIGNED fraction of the average residual
# norm: negative = steered away from the Assistant, positive = toward it. Ours
# is an unsigned dose with the "away" direction baked into Arm 1's minus sign,
# so our alpha is their |x| on the negative half. `paper_axis=True` plots
# x = -alpha and frames the view to the range the paper actually swept.
#
# Their sweep stops at -1.0. Ours goes to 3.0, i.e. three times past the edge of
# any figure they published, which is why the frame is a decision and not a
# default: at |alpha| > 1 nobody has claimed the model stays coherent.
PAPER_XLIM = (-1.06, 0.06)          # the range Fig. 4 actually sweeps
FULL_XLIM = (-3.12, 0.12)           # everything we swept
PAPER_XLABEL = ("steering along the Assistant Axis\n"
                "(fraction of avg. residual norm; negative = away)")


def _mirror(paper_axis: bool, already_signed: bool) -> float:
    """+1, or -1 when a legacy magnitude alpha has to be flipped onto the
    paper's signed x-axis. THE ONLY statement of the rule: the drawn points and
    the "not shown, outside the plotted range" note both go through it, so they
    cannot disagree about which alphas are where."""
    return -1.0 if (paper_axis and not already_signed) else 1.0


def sgn_for_note(alpha, paper_axis: bool, already_signed: bool):
    """Where `alpha` lands on the drawn x-axis, mirroring included."""
    return alpha * _mirror(paper_axis, already_signed)


def _alphas_are_signed(df: pd.DataFrame) -> bool:
    """Does this RUN store signed alphas (current) or magnitudes (legacy)?

    Decided once, over the whole frame. Deciding it per panel from that panel's
    own alphas was a live bug: any judged subset that happened to contain no
    negative alpha — a partial judge pass, `--limit`, or the +0.25 end alone —
    was read as legacy and had its x-axis MIRRORED, so "toward the Assistant"
    was plotted as "away".
    """
    if "alpha" not in df:
        return False
    a = pd.to_numeric(df["alpha"], errors="coerce").dropna()
    return bool((a < 0).any())


def _draw_panel(ax, stats: Optional[Dict], neg: Optional[Dict],
                title: str, show_legend: bool,
                paper_axis: bool = False,
                xlim: Optional[Tuple[float, float]] = None,
                already_signed: bool = True) -> None:
    """One Figure-4 panel: the paper's five category curves plus the control band.

    Five, not seven. `ambiguous` and `other` are scored but are not lines in the
    paper's Figure 4; they are annotated as a residual instead, so a reader can
    see how much probability mass is unaccounted for without a curve implying
    the paper drew one.
    """
    # Alpha is signed in current runs (negative = away from the Assistant), so
    # it IS the paper's x-axis already and must not be flipped again. Legacy
    # runs stored a magnitude with the direction implied, and those still need
    # the flip. `already_signed` comes from the whole frame (see
    # `_alphas_are_signed`), never from this panel's slice of it.
    sgn = _mirror(paper_axis, already_signed)

    if neg is not None:
        nx = sgn * neg["alpha"]
        for cat, j in zip(PLOT_CATEGORIES, _PLOT_IDX):
            m = neg["mean"][:, j]
            lo = np.where(np.isfinite(neg["lo"][:, j]), neg["lo"][:, j], m)
            hi = np.where(np.isfinite(neg["hi"][:, j]), neg["hi"][:, j], m)
            ok = np.isfinite(m)
            if not ok.any():
                continue
            # Band across the control's own alphas, at the dose it was run at.
            ax.fill_between(nx[ok], lo[ok], hi[ok], color=_PALETTE[cat],
                            alpha=0.12, lw=0, zorder=0)
            ax.plot(nx[ok], m[ok], color=_PALETTE[cat], lw=1.0, ls=":",
                    alpha=0.55, zorder=1)

    if stats is None:
        ax.text(0.5, 0.5, "no data for this panel", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#B00020")
    else:
        x = sgn * stats["alpha"]
        for cat, j in zip(PLOT_CATEGORIES, _PLOT_IDX):
            y = stats["mean"][:, j]
            lo, hi = stats["lo"][:, j], stats["hi"][:, j]
            err = np.vstack([np.where(np.isfinite(lo), y - lo, 0.0),
                             np.where(np.isfinite(hi), hi - y, 0.0)])
            err = np.clip(err, 0.0, None)
            label = PLOT_LABEL.get(cat, cat)
            if _GROUP_OF[cat]:
                label = "%s (%s)" % (label, _GROUP_OF[cat])
            ax.errorbar(x, y, yerr=err, marker="o", ms=4, lw=1.6, capsize=2.5,
                        color=_PALETTE[cat], label=label, zorder=3)

        resid = float(np.nanmax(stats["mean"][:, _RESID_IDX].sum(axis=1)))
        if resid > 0:
            ax.text(0.98, 0.02, "%s: max %.0f%% (not plotted)"
                    % (" + ".join(RESIDUAL_CATEGORIES), 100 * resid),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.5, color="#666666")

    ax.set_xlabel(PAPER_XLABEL if paper_axis else "steering strength alpha",
                  fontsize=9 if paper_axis else 10)
    if paper_axis:
        lo, hi = xlim or PAPER_XLIM
        ax.set_xlim(lo, hi)
        # The paper's own limit, marked when we plot past it: everything left of
        # this line is a dose no published figure covers, and the reader should
        # not have to remember that.
        if lo < PAPER_XLIM[0] - 1e-9:
            ax.axvline(PAPER_XLIM[0] + 0.06, color="#B00020", lw=0.9, ls="--",
                       alpha=0.55, zorder=0)
            ax.text(PAPER_XLIM[0] + 0.02, 0.985, "  paper's limit", color="#B00020",
                    fontsize=6.5, ha="left", va="top", transform=ax.get_xaxis_transform())
        # The Assistant end. Everything is measured relative to it, and without
        # the line the reader has to find x=0 by eye.
        ax.axvline(0.0, color="#666666", lw=0.8, ls="-", alpha=0.5, zorder=0)
    ax.set_ylabel("fraction of responses")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    if show_legend and stats is not None:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9, ncol=1)


def _suptitle(fig, arm: str, missing: bool, n_roles: Optional[int]) -> None:
    """Title carrying EXPLORATORY, and the blocked flag when the arm is absent.

    A missing arm is the one failure mode that can be mistaken for a scientific
    finding — an empty panel reads as "the intervention did nothing". Saying so
    in the title, not the caption, is the cheap insurance against that.
    """
    if missing:
        head = "EXPLORATORY - %s - ARM MISSING / BLOCKED (no data)" % arm
        color = "#B00020"
    else:
        head = "EXPLORATORY - %s - judged response category vs alpha" % arm
        color = "black"
    if n_roles:
        head += "\nroles = %d; error bars = bootstrap 95%% CI over roles " \
                "(%d resamples)" % (n_roles, N_BOOT)
    fig.suptitle(head, fontsize=12, color=color)


def _save(fig, path: Path, bottom: float = 0.035) -> None:
    """`bottom` reserves room for the footnote; the paper-axis one is two lines
    under an x-label that is itself two lines, and the default rect clips it."""
    fig.tight_layout(rect=(0, bottom, 1, 0.93))
    fig.savefig(str(path), dpi=300)
    plt.close(fig)
    LOG.info("wrote %s (%d bytes)", path, path.stat().st_size)


# Panel label per arm. Both current arms are targetless and travel along the
# Assistant Axis, so each is a single panel; the near/far split belonged to the
# target-directed arms that the new plan removed.
_ARM_LABEL = {
    "linear_axis": "linear — straight along the Assistant Axis",
    "manifold_axis": "manifold — along the fitted curve",
    "arm1_axis": "no target",            # legacy runs
}


def _panel_specs(arm: str) -> List[Tuple[Optional[str], str]]:
    """(target_distance, panel label) pairs — one panel per targetless arm."""
    if arm in _ARM_LABEL:
        return [(None, _ARM_LABEL[arm])]
    return [("near", 'target_distance = "near"'),
            ("far", 'target_distance = "far"')]


def _build(df: pd.DataFrame, arm: str) -> Tuple[List[Optional[Dict]],
                                                Optional[Dict], bool, int]:
    """Compute everything a figure needs before any axis is touched."""
    neg = _negctl_reference(df)
    missing = df[df["arm"] == arm].empty
    stats = []
    n_roles = 0
    for td, _label in _panel_specs(arm):
        rows = _with_unsteered(df, _arm_frame(df, arm, td))
        s = None if missing else _curve_stats(rows)
        stats.append(s)
        if s is not None and s["n_roles"]:
            n_roles = max(n_roles, int(max(s["n_roles"])))
    return stats, neg, missing, n_roles


def _render(df: pd.DataFrame, arm: str, out: Path,
            ylim: Optional[Tuple[float, float]] = None,
            paper_axis: bool = False,
            show_negctl: bool = True,
            xlim: Optional[Tuple[float, float]] = None,
            built: Optional[Tuple] = None) -> Optional[Tuple[float, float]]:
    """Draw and save one arm's figure; return the y-limits it used.

    Returning the limits is how fig03 inherits fig02's scale without either
    function knowing about the other.

    `built` is this arm's `_build` result when the caller already has one. That
    call runs a 2,000-replicate role bootstrap per alpha per panel, and
    `make_figures` needs it once for the shared y-limits before drawing; without
    this every bootstrap ran twice.
    """
    stats, neg, missing, n_roles = built if built is not None else _build(df, arm)
    already_signed = _alphas_are_signed(df)
    if not show_negctl:
        # On the paper's frame the control is worse than useless: it was run at
        # alpha 0.5/1.5/3.0, so two of its three points sit off-axis and
        # matplotlib joins the survivor to them with a diagonal that crosses the
        # whole panel and reads as a trend. Drop it rather than draw a line the
        # data does not support.
        neg = None
    specs = _panel_specs(arm)
    width = 6.4 if len(specs) == 1 else 11.5
    fig, axes = plt.subplots(1, len(specs), figsize=(width, 4.8), squeeze=False)
    axes = list(axes[0])

    if ylim is None:
        top = max([_panel_ymax(s, neg) for s in stats] + [0.0])
        ylim = (0.0, min(1.0, top * 1.12 + 0.03))

    for ax, s, (_td, label) in zip(axes, stats, specs):
        title = label if not missing else "%s - NO DATA" % label
        _draw_panel(ax, s, neg, title, show_legend=(ax is axes[0]),
                    paper_axis=paper_axis, xlim=xlim,
                    already_signed=already_signed)
        ax.set_ylim(*ylim)

    _suptitle(fig, arm, missing, n_roles)
    # Figure-level, not axes-level: an in-panel note lands on top of whichever
    # curve happens to be low at that x and hides the data it annotates.
    note = "categories and reading groups are judge.py's; alpha=0 is the " \
           "shared unsteered baseline"
    if paper_axis:
        # Say what is off the frame — at BOTH ENDS. The old test was
        # `abs(alpha) > -lo`, which only ever caught doses past the left edge:
        # the sweep's +0.25 point sits outside the right edge of PAPER_XLIM
        # (0.06) and was silently clipped off every paper-axis figure while the
        # note claimed nothing was missing.
        lo, hi = (xlim or PAPER_XLIM)
        off = sorted({float(a) for a in df["alpha"].dropna().unique()
                      if not (lo <= sgn_for_note(float(a), paper_axis,
                                                 already_signed) <= hi)})
        note = ("x-axis as arXiv:2601.10387 Fig. 4: signed fraction of the avg. "
                "residual norm; negative = away from the Assistant")
        if off:
            note += ("\nnot shown, outside the plotted range [%g, %g]: alpha %s"
                     % (lo, hi, ", ".join("%g" % a for a in off)))
    if neg is not None:
        note += ("  |  shaded bands = negctl (random direction), "
                 "95%% CI over %d roles, resolved by alpha" % neg["n_roles"])
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7,
             color="#444444", wrap=True)
    _save(fig, out, bottom=0.12 if paper_axis else 0.035)
    return ylim


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def make_figures(df: pd.DataFrame, outdir: Path, layer: int = 19,
                 paper_axis: bool = False, show_negctl: bool = True,
                 xlim: Optional[Tuple[float, float]] = None) -> List[Path]:
    """Produce the three figures. Never raises.

    This is called at the tail of pipelines that have already spent GPU hours
    producing numbers. A broken axis label must not destroy that run, so every
    figure is attempted independently and failures are logged and skipped.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    n_unjudged = int(df["judge_score"].isna().sum()) if "judge_score" in df else 0
    if n_unjudged:
        LOG.warning("%d/%d rows have judge_score=None and are excluded from "
                    "both numerator and denominator", n_unjudged, len(df))

    # Draw whichever arms the data actually holds, so this serves both the new
    # two-arm plan and the legacy three-arm runs without a flag.
    present = set(df["arm"].dropna().unique()) if "arm" in df else set()
    new_plan = [("linear_axis", "fig01_linear_axis_L%d.png" % layer),
                ("manifold_axis", "fig02_manifold_axis_L%d.png" % layer)]
    old_plan = [("arm1_axis", "fig01_arm1_axis_L%d.png" % layer),
                ("arm2_linear", "fig02_arm2_linear_L%d.png" % layer),
                ("arm3_manifold", "fig03_arm3_manifold_L%d.png" % layer)]
    plan = new_plan if present & {"linear_axis", "manifold_axis"} else old_plan

    # arms 2 and 3 must share a y-scale; compute it from both before drawing
    # either, so neither figure depends on the order they were rendered in.
    shared_ylim = None
    built: Dict[str, Tuple] = {}
    try:
        tops = []
        for arm, _name in plan:
            built[arm] = _build(df, arm)
            stats, neg, _missing, _n = built[arm]
            # The band only constrains the scale when it is actually drawn;
            # including it under --no-negctl left every curve squashed against
            # the bottom of a panel framed for something not on it.
            neg = neg if show_negctl else None
            tops.append(max([_panel_ymax(s, neg) for s in stats] + [0.0]))
        shared_ylim = (0.0, min(1.0, max(tops) * 1.12 + 0.03))
    except Exception:  # noqa: BLE001 - fall back to per-figure autoscale
        LOG.exception("could not compute the shared y-limits; the arm figures "
                      "will NOT be directly comparable")

    for arm, name in plan:
        path = outdir / name
        try:
            _render(df, arm, path,
                    ylim=None if arm == "arm1_axis" else shared_ylim,
                    paper_axis=paper_axis, show_negctl=show_negctl, xlim=xlim,
                    built=built.get(arm))
            written.append(path)
        except Exception:  # noqa: BLE001 - a figure must not kill the caller
            LOG.exception("failed to draw %s; continuing", name)
            plt.close("all")
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Figure-4 analogues (EXPLORATORY) for the steering arms.")
    p.add_argument("--judged", required=True, type=Path,
                   help="parquet of judged responses")
    p.add_argument("--outdir", required=True, type=Path,
                   help="directory the three PNGs are written to")
    p.add_argument("--layer", type=int, default=19,
                   help="layer index, used only in the filenames")
    p.add_argument("--full-range", action="store_true",
                   help="with --paper-axis, show every dose we swept (-3..0) "
                        "instead of only the -1..0 the paper covers; the "
                        "paper's limit is marked")
    p.add_argument("--no-negctl", action="store_true",
                   help="omit the negative-control band (its alphas are 0.5/1.5/"
                        "3.0, so on the paper axis two of three sit off-frame)")
    p.add_argument("--paper-axis", action="store_true",
                   help="plot x as the paper's signed fraction of the average "
                        "residual norm (negative = away from the Assistant), "
                        "framed to the -1..0 range Fig. 4 actually sweeps")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        df = pd.read_parquet(args.judged)
    except Exception:  # noqa: BLE001
        LOG.exception("could not read %s", args.judged)
        return 1

    written = make_figures(df, args.outdir, layer=args.layer,
                           paper_axis=args.paper_axis,
                           show_negctl=not args.no_negctl,
                           xlim=FULL_XLIM if args.full_range else None)
    LOG.info("%d figures written", len(written))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
