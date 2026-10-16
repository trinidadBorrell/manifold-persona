"""Judged behaviour vs latent position: % of answers labelled the target against the steered
state's Euclidean distance to the target centroid mu_B. One folder per layer:

  output/behavior-latent/layer_<L>/target_vs_distance.png              4 panels (one per arm), all questions pooled
  output/behavior-latent/layer_<L>/target_vs_distance_by_question.png  5 x 4 grid (question x arm)
  output/behavior-latent/layer_<L>/points.csv                          the plotted numbers

A point is one (route, alpha): x = % of its answers the subagent judge labelled the target
(25 answers pooled, 5 per question), y = mean distance of their steered states to mu_B (layer output
with steering on, averaged over the answer's tokens). Dots only; colour = route.
Only alpha >= 0. Forward routes only; the unsteered cell is drawn as alpha 0 of linear and manifold.

    python -m ace_steering.fig_behavior_latent
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ace_steering.common import OUT_ROOT, QUESTIONS  # noqa: E402
from ace_steering.grid import geom, route_personas  # noqa: E402
from ace_steering.judge import key_of  # noqa: E402
from ace_steering.sweep_page import FWD, labels, rows_and_acts  # noqa: E402

OUT = OUT_ROOT.parent / "behavior-latent"
ARMS = ["linear", "manifold", "ace"]                     # ACE manifold removed (user, 2026-10-02)
NAME = {"linear": "Linear", "manifold": "Manifold", "ace": "ACE", "ace_manifold": "ACE manifold"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]     # fixed order, one per route
MARK = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#15181d", "#4c525c", "#e3e6ea"


def table():
    d = rows_and_acts(OUT_ROOT)
    lab = labels(OUT_ROOT)
    d["label"] = [lab.get(key_of(r), (None,))[0] for _, r in d.iterrows()]
    d["is_target"] = [l == route_personas(r, int(L))[-1] for r, L, l in zip(d.route, d.layer, d.label)]
    d["dist"] = np.nan
    for (route, layer), idx in d.groupby(["route", "layer"]).groups.items():
        cB = geom(int(layer)).route(route)[3]
        H = np.stack(d.loc[idx, "h"].to_numpy())
        d.loc[idx, "dist"] = np.linalg.norm(H - cB, axis=1)
    # the unsteered cell is alpha 0 of both linear and manifold
    none = d[d.arm == "none"]
    d = pd.concat([d[d.arm != "none"], none.assign(arm="linear"), none.assign(arm="manifold")], ignore_index=True)
    return d.drop(columns=["h"])


def points(d, by_question):
    d = d[d.alpha >= 0]                                   # alpha 0 onwards (user, 2026-10-01)
    keys = ["layer", "route", "arm", "alpha"] + (["q_idx"] if by_question else [])
    g = d.groupby(keys)
    return g.agg(pct_target=("is_target", lambda s: 100 * s.mean()), dist=("dist", "mean"),
                 n=("dist", "count")).reset_index()


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9cdd3")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlim(-4, 104)


def draw(ax, P):
    for i, route in enumerate(FWD):
        r = P[P.route == route].sort_values("alpha")
        if r.empty:
            continue
        ax.scatter(r.pct_target, r.dist, s=22, color=COL[i], marker="o", edgecolor="white", linewidth=0.5,
                   alpha=0.9, zorder=3, label=route.replace(">", " → "))


def figure(P, layer, by_question, path):
    rows = len(QUESTIONS) if by_question else 1
    fig, axes = plt.subplots(rows, len(ARMS), figsize=(11.5, 3.2 * rows + 0.9), sharex=True, sharey=True, squeeze=False)
    for qi in range(rows):
        for j, arm in enumerate(ARMS):
            ax = axes[qi, j]
            style(ax)
            sub = P[P.arm == arm]
            if by_question:
                sub = sub[sub.q_idx == qi]
            draw(ax, sub)
            if qi == 0:
                ax.set_title(NAME[arm], fontsize=11, color=INK, loc="left", fontweight="bold")
            if j == 0:
                ax.set_ylabel(("“%s”\n" % QUESTIONS[qi] if by_question else "") + "distance to μ_target", fontsize=9, color=INK2)
            if qi == rows - 1:
                ax.set_xlabel("answers labelled the target (%)", fontsize=9, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:                                  # a panel may lack a route; take the fullest legend
        hh, ll = ax.get_legend_handles_labels()
        if len(ll) > len(l):
            h, l = hh, ll
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    each = "5 answers per point (one question, 5 system prompts)" if by_question else "25 answers per point"
    fig.suptitle("Layer %d · judged target share vs distance of the steered state to μ_target · α ≥ 0 · %s"
                 % (layer, each), fontsize=10, color=INK2, y=1.0 - 0.45 / fig.get_figheight() - 0.02)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    d = table()
    for by_q in (False, True):
        P = points(d, by_q)
        for layer, PL in P.groupby("layer"):
            folder = OUT / ("layer_%d" % layer)
            folder.mkdir(parents=True, exist_ok=True)
            name = "target_vs_distance" + ("_by_question" if by_q else "")
            figure(PL, int(layer), by_q, folder / (name + ".png"))
            PL.round(4).to_csv(folder / ("points" + ("_by_question" if by_q else "") + ".csv"), index=False)
    (OUT / "README.md").write_text(__doc__.strip() + "\n\nJudge: Claude Code subagents (claude-sonnet-5-5), route_judge_v3 prompt.\n"
                                   "Source run: output/ace-steering-29_09/ (step 2 + step 1 sample 0 at layer 19).\n")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
