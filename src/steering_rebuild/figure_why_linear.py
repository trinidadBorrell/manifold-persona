"""Why the straight arm won: both arms are straight pushes, only one points right.

The headline of the axis experiment is that the manifold arm is strictly worse
(REPORT_axis_steering.md). This figure is the explanation, in one page:

  A  schematic -- NOT DATA. The mechanism. The fitted curve only covers the
     stretch of axis positions the roles actually occupy; past its last knot a
     cubic spline continues as a straight line along its end tangent. So the
     manifold arm, once it leaves that stretch, is also a fixed direction --
     just one tilted off the Assistant Axis.
  B  measured. What fraction of token coordinates fell outside the knot range.
  C  measured. How much of each arm's realised move lay along the axis (fig08).
  D  measured. What that cost: role adoption and degeneration, both arms.

Everything but A is read from the run's own artefacts; nothing is recomputed
from the cloud and no GPU is needed.

    python -m steering_rebuild.figure_why_linear \
        --judged <run>/steer/data/judged_L19.parquet \
        --delta-stats <run>/steer/data/delta_stats_L19.parquet \
        --divergence <run>/figures/fig08_arm_divergence.json \
        --outdir <run>/figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

LIN, MAN = "linear_axis", "manifold_axis"
C_LIN, C_MAN = "#4C72B0", "#C44E52"
ADOPT = ("human_role", "nonhuman_role")


def role_mean(judged: pd.DataFrame, cats) -> pd.Series:
    """Fraction in `cats`, averaged over roles (a role's 20 rollouts are not
    independent -- they share its prompts, so the role is the unit)."""
    hit = judged.judge_score.isin(cats)
    per_role = hit.groupby([judged.arm, judged.alpha, judged.role]).mean()
    return per_role.groupby(level=[0, 1]).mean()


def panel_a(ax, cos_arms: float, span: float, u_max: float) -> None:
    """Schematic. Hand-drawn geometry, real numbers in the annotations."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 400)
    cx, cy = 10.0 * t, 2.6 * np.sin(np.pi * 0.62 * t)          # the role band

    jt = rng.uniform(0, 1, 260)
    jx = 10.0 * jt + rng.normal(0, 0.28, jt.size)
    jy = 2.6 * np.sin(np.pi * 0.62 * jt) + rng.normal(0, 0.42, jt.size)
    ax.scatter(jx, jy, s=9, c="#B9C2CC", alpha=0.75, lw=0, zorder=1)

    ax.plot(cx, cy, color="#5A6570", lw=2.6, zorder=2)
    ax.fill_betweenx([-2.4, 8.6], 0, 10, color="#7F8C99", alpha=0.07, lw=0, zorder=0)
    ax.plot([10, 10], [-2.4, 8.6], color="#5A6570", lw=1.1, ls=":", zorder=2)
    ax.text(5.0, -2.25, "the fitted curve exists only here — the %.0f units of\n"
                        "axis that the 275 roles actually occupy" % span,
            ha="center", va="bottom", fontsize=8.8, color="#3D4750")

    ax.text(-0.3, -0.55, "Assistant\nend", ha="left", va="top",
            fontsize=9.2, color="#3D4750")
    ax.text(10.3, -0.55, "far end\n(monsters)", ha="left", va="top",
            fontsize=9.2, color="#3D4750")

    p0 = np.array([1.6, 2.6 * np.sin(np.pi * 0.62 * 0.16)])
    ax.scatter(*p0, s=52, c="#222", zorder=5)
    ax.text(p0[0] - 0.25, p0[1] - 0.35, "start", fontsize=9.2, ha="right", va="top")

    chord = np.array([10.0, cy[-1] - cy[0]])
    chord = chord / np.linalg.norm(chord)
    ax.annotate("", xy=p0 + 12.2 * chord, xytext=p0, zorder=6,
                arrowprops=dict(arrowstyle="-|>,head_width=0.30,head_length=0.62",
                                color=C_LIN, lw=3.0))
    ax.text(7.0, 1.15, "linear: one straight push, along the\nmeasured Assistant Axis",
            color=C_LIN, fontsize=9.6, ha="center", va="top", weight="bold")

    th = np.arccos(cos_arms)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    off = R @ chord
    k = int(np.searchsorted(t, 0.16))
    ax.plot(cx[k:], cy[k:], color=C_MAN, lw=3.0, zorder=6)
    tip = np.array([cx[-1], cy[-1]])
    ax.annotate("", xy=tip + 4.3 * off, xytext=tip, zorder=6,
                arrowprops=dict(arrowstyle="-|>,head_width=0.30,head_length=0.62",
                                color=C_MAN, lw=3.0, ls=(0, (5, 2))))
    ax.text(6.9, 8.3,
            "manifold: on the curve while the fit lasts, then a\n"
            "straight line along its end tangent — %.0f$\\degree$ off\n"
            "the axis (measured cos = %.2f)" % (np.degrees(th), cos_arms),
            color=C_MAN, fontsize=9.6, ha="left", va="top", weight="bold")

    ax.annotate("and the push went far past the end:\ntoken coordinates reached %s,\n"
                "about %.0f x the width of the fit" % (f"{u_max:,.0f}", u_max / span),
                xy=tuple(tip + 2.2 * off), xytext=(-0.9, 6.5), fontsize=8.8, color="#7A2E33",
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#7A2E33", lw=1.1))

    ax.set_xlim(-1.4, 16.4); ax.set_ylim(-3.0, 9.0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("A — the mechanism   (SCHEMATIC, not data)", fontsize=11, loc="left")


def panel_b(ax, ds: pd.DataFrame) -> dict:
    g = ds[ds.arm == MAN].groupby("alpha").frac_out_of_knots
    xs = np.array(sorted(g.mean().index))
    mu, lo, hi = (100 * g.mean()[xs].values, 100 * g.min()[xs].values,
                  100 * g.max()[xs].values)
    ax.plot(xs, mu, "-o", color=C_MAN, lw=2.4, ms=6)
    ax.fill_between(xs, lo, hi, color=C_MAN, alpha=0.16, lw=0)
    ax.axhline(50, color="#888", lw=1.0, ls="--")
    ax.text(-0.98, 52, "half the tokens", fontsize=8.6, color="#666")
    ax.set_ylim(0, 100)
    ax.set_title("B — how often the manifold arm was off the end of its own fit",
                 fontsize=11, loc="left")
    ax.set_xlabel("alpha (fraction of avg. residual norm)")
    ax.set_ylabel("% of token coordinates outside the knots")
    ax.grid(alpha=0.25, lw=0.6)
    return {str(a): float(v / 100) for a, v in zip(xs, mu)}


def panel_c(ax, div: dict) -> None:
    for arm, col, lab in ((LIN, C_LIN, "linear"), (MAN, C_MAN, "manifold")):
        d = div["axis_alignment_mean"][arm]
        xs = np.array(sorted(float(k) for k in d))
        ax.plot(xs, [d[str(x)] for x in xs], "-o", color=col, lw=2.4, ms=6, label=lab)
    ax.axhline(0, color="#888", lw=0.9)
    ax.set_title("C — how much of the realised move went along the axis",
                 fontsize=11, loc="left")
    ax.set_xlabel("alpha (fraction of avg. residual norm)")
    ax.set_ylabel(r"cos$\,($act$(\alpha)-$act$(0),\ -\hat a)$")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9.5, loc="lower left")


def panel_d(ax, judged: pd.DataFrame) -> dict:
    bad, good = role_mean(judged, ["nonsensical"]), role_mean(judged, ADOPT)
    out = {}
    for arm, col, lab in ((LIN, C_LIN, "linear"), (MAN, C_MAN, "manifold")):
        xs = np.array(sorted(bad.loc[arm].index))
        ax.plot(xs, 100 * bad.loc[arm][xs].values, "-o", color=col, lw=2.4, ms=5.5,
                label="%s — broke down" % lab)
        ax.plot(xs, 100 * good.loc[arm][xs].values, "--s", color=col, lw=2.0, ms=5,
                alpha=0.85, label="%s — adopted the role" % lab)
        out[arm] = {"nonsensical": {str(a): float(bad.loc[arm][a]) for a in xs},
                    "adopted": {str(a): float(good.loc[arm][a]) for a in xs}}
    ax.set_ylim(0, 100)
    ax.set_title("D — what the off-axis push cost", fontsize=11, loc="left")
    ax.set_xlabel("alpha (fraction of avg. residual norm)")
    ax.set_ylabel("% of responses")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8.8, loc="upper center", ncol=2, framealpha=0.9)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--delta-stats", required=True)
    ap.add_argument("--divergence", required=True)
    ap.add_argument("--curvature", default=None,
                    help="curvature_check.json, for the measured angle between arms")
    ap.add_argument("--geometry", default=None,
                    help="geometry_report.json, for the axis span")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    judged = pd.read_parquet(a.judged)
    judged = judged[judged.arm.isin([LIN, MAN])]
    ds = pd.read_parquet(a.delta_stats)
    div = json.load(open(a.divergence))
    cos_arms = 0.703
    if a.curvature:
        cos_arms = float(np.mean([r["mean"] for r in
                                  json.load(open(a.curvature))["arm_cosines"]
                                  if r["alpha"] < 0]))
    span = float(json.load(open(a.geometry))["axis_span"]) if a.geometry else 34.42

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.4))
    panel_a(axes[0, 0], cos_arms, span, float(ds.u_observed_max.max()))
    ook = panel_b(axes[0, 1], ds)
    panel_c(axes[1, 0], div)
    outcome = panel_d(axes[1, 1], judged)

    fig.suptitle("Both arms push in a straight line. Only one of them points along the axis.",
                 fontsize=14.5, y=0.985)
    fig.text(0.5, 0.012,
             "Qwen3-8B, layer 19, 50 roles x 20 rollouts per dose. B and D are means over roles "
             "(B's band is the min-max across roles); C is from fig08, measured on replayed "
             "activations.\nA is a drawing of the mechanism, not a projection of the data. "
             "Judge labels in D are UNVALIDATED (no human agreement study).",
             ha="center", fontsize=8.8, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    png = out / "fig09_why_linear_wins_L19.png"
    fig.savefig(str(png), dpi=300)
    plt.close(fig)
    json.dump({"cos_between_arms": cos_arms, "angle_deg": float(np.degrees(np.arccos(cos_arms))),
               "axis_span": span, "u_observed_max": float(ds.u_observed_max.max()),
               "frac_out_of_knots_by_alpha": ook,
               "frac_out_of_knots_overall_mean": float(ds[ds.arm == MAN].frac_out_of_knots.mean()),
               "frac_out_of_knots_overall_max": float(ds[ds.arm == MAN].frac_out_of_knots.max()),
               "outcome_by_alpha": outcome},
              open(out / "fig09_why_linear_wins.json", "w"), indent=2)
    print("wrote", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
