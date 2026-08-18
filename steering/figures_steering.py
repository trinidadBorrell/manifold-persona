"""Figure-4 analogues for the steering arms (arXiv:2601.10387, Fig. 4).

WHY THIS EXISTS
---------------
The paper's Figure 4 reads a steering intervention by asking a single question:
as the steering strength grows, where does the response *land*? Not "is it more
persona-like on a scalar", which hides collapse and degradation inside one
number, but "what fraction of responses is now on-target, off-target,
theatrical, or broken". This module draws that same reading for our three arms
so the comparison to the paper is a comparison of like with like.

Three figures, and deliberately no more:

    fig01  arm1_axis        one panel   - a global direction, no target
    fig02  arm2_linear      two panels  - straight line to a target (near|far)
    fig03  arm3_manifold    two panels  - curved path to the SAME target

fig03 is drawn on the y-limits computed from arm2 AND arm3 together. The whole
point of arms 2 and 3 is that they aim at the same place by different routes, so
the figures have to be physically superimposable; a per-figure autoscale would
let a difference in axis range masquerade as a difference in behaviour.

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


def _panel_ymax(stats: Optional[Dict], neg: Optional[Dict]) -> float:
    """Largest value a panel needs to show, error bars and band included."""
    vals = [0.0]
    if stats is not None:
        vals.append(float(np.nanmax(stats["mean"])))
        if np.isfinite(stats["hi"]).any():
            vals.append(float(np.nanmax(stats["hi"])))
    if neg is not None:
        vals.append(float(np.nanmax(neg["mean"])))
        if np.isfinite(neg["hi"]).any():
            vals.append(float(np.nanmax(neg["hi"])))
    return max(vals)


def _draw_panel(ax, stats: Optional[Dict], neg: Optional[Dict],
                title: str, show_legend: bool) -> None:
    """One Figure-4 panel: seven category curves plus the control band."""
    if neg is not None:
        nx = neg["alpha"]
        for j, cat in enumerate(CATEGORIES):
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
        x = stats["alpha"]
        for j, cat in enumerate(CATEGORIES):
            y = stats["mean"][:, j]
            lo, hi = stats["lo"][:, j], stats["hi"][:, j]
            err = np.vstack([np.where(np.isfinite(lo), y - lo, 0.0),
                             np.where(np.isfinite(hi), hi - y, 0.0)])
            err = np.clip(err, 0.0, None)
            label = cat
            if _GROUP_OF[cat]:
                label = "%s (%s)" % (cat, _GROUP_OF[cat])
            ax.errorbar(x, y, yerr=err, marker="o", ms=4, lw=1.6, capsize=2.5,
                        color=_PALETTE[cat], label=label, zorder=3)

    ax.set_xlabel("steering strength alpha")
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


def _save(fig, path: Path) -> None:
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(str(path), dpi=300)
    plt.close(fig)
    LOG.info("wrote %s (%d bytes)", path, path.stat().st_size)


def _panel_specs(arm: str) -> List[Tuple[Optional[str], str]]:
    """(target_distance, panel label) pairs — one panel for the untargeted arm."""
    if arm == "arm1_axis":
        return [(None, "no target")]
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
            ylim: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
    """Draw and save one arm's figure; return the y-limits it used.

    Returning the limits is how fig03 inherits fig02's scale without either
    function knowing about the other.
    """
    stats, neg, missing, n_roles = _build(df, arm)
    specs = _panel_specs(arm)
    width = 6.4 if len(specs) == 1 else 11.5
    fig, axes = plt.subplots(1, len(specs), figsize=(width, 4.8), squeeze=False)
    axes = list(axes[0])

    if ylim is None:
        top = max([_panel_ymax(s, neg) for s in stats] + [0.0])
        ylim = (0.0, min(1.0, top * 1.12 + 0.03))

    for ax, s, (_td, label) in zip(axes, stats, specs):
        title = label if not missing else "%s - NO DATA" % label
        _draw_panel(ax, s, neg, title, show_legend=(ax is axes[0]))
        ax.set_ylim(*ylim)

    _suptitle(fig, arm, missing, n_roles)
    # Figure-level, not axes-level: an in-panel note lands on top of whichever
    # curve happens to be low at that x and hides the data it annotates.
    note = "categories and reading groups are judge.py's; alpha=0 is the " \
           "shared unsteered baseline"
    if neg is not None:
        note += ("  |  shaded bands = negctl (random direction), "
                 "95%% CI over %d roles, resolved by alpha" % neg["n_roles"])
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7.5,
             color="#444444")
    _save(fig, out)
    return ylim


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def make_figures(df: pd.DataFrame, outdir: Path, layer: int = 19) -> List[Path]:
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

    plan = [("arm1_axis", "fig01_arm1_axis_L%d.png" % layer),
            ("arm2_linear", "fig02_arm2_linear_L%d.png" % layer),
            ("arm3_manifold", "fig03_arm3_manifold_L%d.png" % layer)]

    # arms 2 and 3 must share a y-scale; compute it from both before drawing
    # either, so neither figure depends on the order they were rendered in.
    shared_ylim = None
    try:
        tops = []
        for arm in ("arm2_linear", "arm3_manifold"):
            stats, neg, _missing, _n = _build(df, arm)
            tops.append(max([_panel_ymax(s, neg) for s in stats] + [0.0]))
        shared_ylim = (0.0, min(1.0, max(tops) * 1.12 + 0.03))
    except Exception:  # noqa: BLE001 - fall back to per-figure autoscale
        LOG.exception("could not compute the shared arm2/arm3 y-limits; "
                      "fig02 and fig03 will NOT be directly comparable")

    for arm, name in plan:
        path = outdir / name
        try:
            _render(df, arm, path,
                    ylim=None if arm == "arm1_axis" else shared_ylim)
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
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        df = pd.read_parquet(args.judged)
    except Exception:  # noqa: BLE001
        LOG.exception("could not read %s", args.judged)
        return 1

    written = make_figures(df, args.outdir, layer=args.layer)
    LOG.info("%d/3 figures written", len(written))
    return 0 if len(written) == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
