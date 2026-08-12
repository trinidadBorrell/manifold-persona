"""Per-family figures: the same evidence, split by what kind of thing it measures.

WHY THIS EXISTS BESIDE `figures.py`
-----------------------------------
The global ladder is now 36 rows tall. At that size it is a reference table, not
a figure: nobody reads 36 rows and forms a view. Worse, it flattens a real
distinction — five intrinsic-dimension estimators agreeing is ONE finding with
five votes, while a curvature metric and a density metric agreeing is two
independent lines of evidence. A flat list cannot show that and a family split
can.

Each family gets three figures:

  ladder.png            that family's rows only, all predictors, all rungs.
  scatter_<pred>.png    the family's metrics against one closeness measure.
  distributions.png     ONLY for families whose metrics summarise a per-point or
                        per-edge distribution. See below — this is the one that
                        answers something the ladder structurally cannot.

THE POINT OF `distributions.png`
--------------------------------
`knn_dist_mean`, `orc_mean`, `frc_mean` and their `_cv`/`_sd` partners are
summaries of 200 points (or ~1300 edges) per role. A summary throws away the
shape of that distribution, and the discarded part can be where the finding is:
a metric whose MEAN is flat against `axis_proj` while its SPREAD widens with it
is a real result that no scalar column can express. So each per-point metric is
also drawn as a p10-p90 band per role against closeness, plus a pooled hexbin of
every (role, point) pair.

Usage:
    .venv/bin/python exploratory/per_persona/figures_families.py --outdir <run>
"""
from __future__ import annotations

import argparse
import json
import textwrap
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import families as F
from common import C_REAL, C_DESIGN, C_GAUSS, C_INSTR
from figures import (RUNGS, RUNG_COLORS, RUNG_OFFSET, PRED_TITLES, YLABELS,
                     _offsets, defensive, grid_shape)
from metrics import PANEL_COLS
from stats_utils import linfit, fmt_p
from study_ladder import EXTRA_METRICS

DPI = 200
REFERENCE_PRED = "axis_proj"


def _preds_for(spec, lad) -> list:
    """Which closeness measures this family is drawn against.

    Every family gets all four unless it declares a ``predictors`` list. Family
    07 does, because its two metrics are also two of the predictors and the
    self-pairs carry no information.
    """
    avail = list(dict.fromkeys(lad.predictor))
    want = spec.get("predictors")
    return [p for p in avail if want is None or p in want]


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path.relative_to(path.parents[2]))


def _headline(spec, width=95):
    """One line for a figure title: the family's question, nothing more.

    The longer `reading` is deliberately NOT drawn. On a one-row family it
    wrapped to seven lines and dwarfed the plots, and a figure is the wrong
    place for a paragraph — `md/FAMILY-REPORTS.md` is the companion text.
    """
    return textwrap.fill(spec["question"], width)


@defensive
def family_ladder(lad, spec, out: Path):
    """One family's rows, every predictor, every rung."""
    metrics = [m for m in spec["metrics"] if m in set(lad.metric)]
    if not metrics:
        return
    preds = _preds_for(spec, lad)
    if not preds:
        return
    offs, ci_y = _offsets(RUNGS)
    sub = lad[(lad.predictor == preds[0]) & lad.metric.isin(metrics)
              & (~lad.degenerate.fillna(False))]
    order = sub.reindex(sub.r_ctrl_all.abs().sort_values().index)["metric"].tolist()
    if not order:
        return
    fig, axes = plt.subplots(1, len(preds), sharey=True,
                             figsize=(5.2 * len(preds), 0.55 * len(order) + 4.6))
    for ax, pred in zip(np.atleast_1d(axes), preds):
        s = lad[lad.predictor == pred].set_index("metric").reindex(order)
        thr = float(s["shuffle_max_abs_r_p95"].dropna().iloc[0]) if \
            s["shuffle_max_abs_r_p95"].notna().any() else np.nan
        y = np.arange(len(order))
        if np.isfinite(thr):
            ax.axvspan(-thr, thr, color="0.90", zorder=0,
                       label="axis-shuffle null")
        ax.axvline(0, color="k", lw=0.8, zorder=1)
        for j, (col, lab) in enumerate(RUNGS):
            ax.scatter(s[col], y + offs[j], s=30, color=RUNG_COLORS[j], zorder=3,
                       label=lab if pred == preds[0] else None)
        ax.hlines(y + ci_y, s["ci_lo_ctrl_all"], s["ci_hi_ctrl_all"],
                  color=RUNG_COLORS[-1], lw=1.8, zorder=2)
        for v in (-0.30, 0.30):
            ax.axvline(v, color=C_DESIGN, ls=":", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9)
        ax.set_xlim(-1, 1); ax.set_xlabel("correlation with predictor", fontsize=9)
        ax.set_title(PRED_TITLES.get(pred, pred), fontsize=8.5, linespacing=1.3)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    # No suptitle (user request, 2026-08-05) — the folder names the family and
    # the panel titles name the predictor. See md/FAMILY-REPORTS.md for the
    # family question and reading; a figure is the wrong place for prose.
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, out / "ladder.png")


@defensive
def family_scatter(panel, lad, spec, pred: str, out: Path):
    """The family's metrics as scatter against one closeness measure."""
    metrics = [m for m in spec["metrics"] if m in panel.columns
               and panel[m].std() > 0]
    if not metrics:
        return
    d = panel[panel.role != "default"]
    dflt = panel[panel.role == "default"]
    # A 4-metric family belongs on ONE row: they are four estimates of the same
    # quantity and the point is to compare them side by side.
    nrow, ncol = grid_shape(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.0 * nrow),
                             squeeze=False)
    for ax, m in zip(axes.ravel(), metrics):
        row = lad[(lad.predictor == pred) & (lad.metric == m)]
        ax.scatter(d[pred], d[m], s=11, alpha=.45, color=C_REAL, linewidths=0)
        f = linfit(d[pred], d[m])
        sl, ic = f["slope"], f["intercept"]
        xs = np.linspace(d[pred].min(), d[pred].max(), 50)
        ax.plot(xs, sl * xs + ic, color=C_DESIGN, lw=2)
        if len(dflt):
            dx, dy = float(dflt[pred].iloc[0]), float(dflt[m].iloc[0])
            if not (xs[0] <= dx <= xs[-1]):
                ex = np.linspace(min(xs[0], dx), max(xs[-1], dx), 50)
                ax.plot(ex, sl * ex + ic, ls=":", lw=1.2, color=C_DESIGN)
            ax.scatter([dx], [dy], s=110, marker="*", color="#111111", zorder=6)
        r = row.iloc[0] if len(row) else None
        # BH q across every test in the controlled rung, not a raw p — this is
        # one panel of 31 metrics x 4 predictors.
        ax.set_title(f"{m}\nraw r={r.r_raw:+.3f}   controlled={r.r_ctrl_all:+.3f}"
                     f"   ({fmt_p(r.get('q_global_ctrl_all'), 'q')})"
                     if r is not None else m, fontsize=9)
        ax.set_ylabel(YLABELS.get(m, m), fontsize=8)
        ax.set_xlabel(f"{pred}  (high = Assistant-like)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(f"{spec['title']} vs {pred}\n{_headline(spec)}", fontsize=11,
                 linespacing=1.4)
    # Reserve less for the title on short figures, or a one-row family ends up
    # mostly whitespace.
    fig.tight_layout(rect=[0, 0, 1, 0.99 - 0.10 / nrow])
    _save(fig, out / f"scatter_{pred}.png")


@defensive
def family_distributions(panel, pw_dir: Path, spec, key: str, out: Path):
    """What the scalar summaries threw away.

    Left: one vertical p10-p90 band per role against `axis_proj`, median dotted.
    Right: every (role, point) pair pooled into a hexbin. The band plot shows
    per-role spread; the hexbin shows whether the pooled cloud has structure the
    per-role summary hides.
    """
    wanted = F.POINTWISE.get(key, {})
    available = {n: pw_dir / f"{n}.npz" for n in wanted if (pw_dir / f"{n}.npz").exists()}
    if not available:
        print(f"  [dist] {key}: no pointwise arrays saved, skipped")
        return
    d = panel[panel.role != "default"].sort_values(REFERENCE_PRED)
    fig, axes = plt.subplots(len(available), 2, squeeze=False,
                             figsize=(13, 4.4 * len(available)))
    for i, (name, path) in enumerate(sorted(available.items())):
        with np.load(path) as z:
            vals = {r: z[r] for r in z.files}
        x, lo, mid, hi, pooled_x, pooled_y = [], [], [], [], [], []
        for _, row in d.iterrows():
            v = vals.get(row.role)
            if v is None or len(v) == 0:
                continue
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            x.append(row[REFERENCE_PRED])
            lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
            mid.append(np.median(v))
            pooled_x.append(np.full(v.size, row[REFERENCE_PRED])); pooled_y.append(v)
        x, lo, mid, hi = map(np.asarray, (x, lo, mid, hi))

        ax = axes[i][0]
        ax.vlines(x, lo, hi, color=C_GAUSS, lw=1.0, alpha=.75,
                  label="p10–p90 within role")
        ax.scatter(x, mid, s=13, color=C_REAL, zorder=3, label="role median")
        if len(x) > 1:
            # Two separate questions, two separate tests: does the typical value
            # move with closeness, and does the within-role SPREAD move with it?
            # Neither is in the ladder, so these are raw p-values (4 per family).
            fm = linfit(x, mid)
            fs_ = linfit(x, hi - lo)
            xs = np.linspace(x.min(), x.max(), 40)
            ax.plot(xs, fm["slope"] * xs + fm["intercept"], color=C_DESIGN, lw=2)
            ax.set_title(f"{name} — {wanted[name]}\n"
                         f"median vs {REFERENCE_PRED}: r={fm['r']:+.3f} "
                         f"({fmt_p(fm['p'])})   |   SPREAD (p90−p10): "
                         f"r={fs_['r']:+.3f} ({fmt_p(fs_['p'])})   n={fm['n']}",
                         fontsize=9)
        ax.set_xlabel(f"{REFERENCE_PRED}  (high = Assistant-like)", fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.legend(fontsize=7.5)

        ax2 = axes[i][1]
        hb = ax2.hexbin(np.concatenate(pooled_x), np.concatenate(pooled_y),
                        gridsize=45, cmap="viridis", bins="log", linewidths=0)
        plt.colorbar(hb, ax=ax2, label="points per bin (log)")
        ax2.set_xlabel(f"{REFERENCE_PRED}", fontsize=9)
        ax2.set_ylabel(f"{name} — every point of every role", fontsize=9)
        ax2.set_title("pooled: does the per-role summary hide structure?",
                      fontsize=9)
    fig.suptitle(f"{spec['title']} — the distributions behind the summaries\n"
                 "a flat median with a widening spread is a real finding the "
                 "ladder cannot show", fontsize=11, linespacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out / "distributions.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label-layer", type=int, default=19)
    args = ap.parse_args()
    run, L = Path(args.outdir), args.label_layer
    D = run / "data"

    panel = pd.read_csv(D / f"per_role_panel_L{L}.csv")
    lad = pd.read_csv(D / f"ladder_L{L}.csv")
    pw_dir = D / "pointwise"

    # Loud on purpose — see families.check_coverage. The ladder's metric set is
    # PANEL_COLS plus the closeness measures that ride as rows, so coverage is
    # checked against exactly that, not against PANEL_COLS alone.
    all_metrics = list(PANEL_COLS) + list(EXTRA_METRICS)
    F.check_coverage(all_metrics)
    print(f"{len(all_metrics)} metrics across {len(F.FAMILIES)} families\n")

    index = {}
    for key, spec in F.ordered():
        out = run / "figures" / "families" / F.folder(key)
        preds = _preds_for(spec, lad)
        print(f"{F.folder(key)}  ({len(spec['metrics'])} metrics, "
              f"{len(preds)} predictor(s))")
        family_ladder(lad, spec, out)
        for pred in preds:
            family_scatter(panel, lad, spec, pred, out)
        if key in F.POINTWISE:
            family_distributions(panel, pw_dir, spec, key, out)
        index[key] = {"folder": F.folder(key), "title": spec["title"],
                      "question": spec["question"], "reading": spec["reading"],
                      "metrics": spec["metrics"], "predictors": preds}

    json.dump({"exploratory": True, "families": index},
              open(D / f"families_index_L{L}.json", "w"), indent=2)
    print("\nfamily figures done")


if __name__ == "__main__":
    main()
