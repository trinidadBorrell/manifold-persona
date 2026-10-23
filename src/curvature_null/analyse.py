"""Figures, tables and the decision rule for curvature_null (plan: output/20-oct-AI-pilled/PLAN.md).

    python -m curvature_null.analyse --dir $PM/CURV --out output/20-oct-AI-pilled
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REAL, GAUSS, SHUF = "#2a2a2a", "#2a78d6", "#eb6834"
NAME = {"real": "real", "gauss": "Gaussian null", "shuffle": "shuffled-coordinates null"}
R2_MIN = 0.10


def rule(bends, graph):
    """bends: {real: [10], shuffle: [[10]...]}; graph: {real, gauss, shuffle} (N = 276)."""
    real, null = np.array(bends["real"]), np.array(bends["shuffle"])
    c = bool(((real >= R2_MIN) & (real > null.max(0))).any())
    t = graph["real"]["trend"]
    tn = max(x["trend"] for k in ("gauss", "shuffle") for x in graph[k])
    a = bool(t > 1.0 and t > tn)
    return dict(bend_pass=c, trend_pass=a, max_bend=float(real.max()), max_bend_null=float(null.max()),
                pc_max_bend=int(real.argmax() + 1), trend=t, trend_null_max=tn,
                verdict="curved" if c and a else ("bent but not like a smooth surface" if c else "looks flat"))


def band(ax, x, ys, color, label):
    ys = np.array(ys)
    ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.25, lw=0, label=label + " (min–max of draws)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="output/20-oct-AI-pilled")
    a = ap.parse_args(argv)
    D, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    means = {int(f.stem.split("_L")[1]): json.loads(f.read_text()) for f in sorted(D.glob("means_L*.json"))}
    ans = json.loads((D / "answers_L28.json").read_text()) if (D / "answers_L28.json").exists() else None
    ctl = json.loads((D / "control.json").read_text()) if (D / "control.json").exists() else {}
    S = dict(layers={}, controls={})

    # ---- controls ----
    for key, v in ctl.items():
        S["controls"][key] = rule(v["bends"], v["276"])
    # ---- layers ----
    for L, m in means.items():
        S["layers"][L] = rule(m["bends"], m["graph"]) | dict(
            p50=m["graph"]["real"]["p50"], p50_gauss=float(np.mean([x["p50"] for x in m["graph"]["gauss"]])),
            p50_shuffle=float(np.mean([x["p50"] for x in m["graph"]["shuffle"]])),
            twonn=m["graph"]["real"]["twonn"], twonn_gauss=float(np.mean([x["twonn"] for x in m["graph"]["gauss"]])),
            twonn_shuffle=float(np.mean([x["twonn"] for x in m["graph"]["shuffle"]])),
            check=m["check"], frechet_mean=m["frechet_mean_role"])
    if ans:
        S["answers_L28"] = dict(bends=ans["bends"], sweep={n: dict(real=v["real"],
                                gauss_trend=[x["trend"] for x in v["gauss"]], shuffle_trend=[x["trend"] for x in v["shuffle"]],
                                gauss_p50=[x["p50"] for x in v["gauss"]], shuffle_p50=[x["p50"] for x in v["shuffle"]])
                                for n, v in ans["sweep"].items()})
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))

    # ---- fig1: bends at L28 ----
    m = means.get(28)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(1, 11)
    for ax, src, title in ((axs[0], m["bends"] if m else None, "role means (N = 276)"),
                           (axs[1], ans["bends"] if ans else None, "per-answer points (N = 11,040)")):
        if src is None:
            continue
        k = len(src["real"])
        band(ax, x[:k], src["shuffle"], SHUF, NAME["shuffle"])
        ax.plot(x[:k], src["real"], "o-", color=REAL, lw=2, label="real")
        ax.axhline(R2_MIN, color="#999", ls=":", lw=1, label="rule floor R² = 0.10")
        ax.set(title="Layer 28, " + title, xlabel="PC k (predicted from PCs 1–10 except k)",
               ylabel="cross-validated R², quadratic fit", xticks=x)
        ax.legend(fontsize=8, frameon=False)
    keys = list(ctl)
    if keys:
        r = [S["controls"][k]["max_bend"] for k in keys]
        rn = [S["controls"][k]["max_bend_null"] for k in keys]
        y = np.arange(len(keys))
        axs[2].barh(y + 0.2, r, 0.4, color=REAL, label="control")
        axs[2].barh(y - 0.2, rn, 0.4, color=SHUF, label="its shuffled null (max of draws)")
        if m:
            axs[2].axvline(S["layers"][28]["max_bend"], color="#c22", ls="--", lw=1.5, label="real role means, L28")
        axs[2].set(yticks=y, yticklabels=[k.replace("_", ", ") for k in keys], xlabel="max over PC 1–10 of the R²",
                   title="Synthetic controls, N = 276")
        axs[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig1_bends.png", dpi=150)
    plt.close(fig)

    # ---- fig2: path ratio by distance decile ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    dx = np.arange(1, 11)
    panels = [("Layer 28 role means (N = 276)", m["graph"] if m else None),
              ("Layer 28 per-answer points (N = 11,040)", ans["sweep"]["11040"] if ans else None)]
    for ax, (title, g) in zip(axs[:2], panels):
        if g is None:
            continue
        for kind, c in (("gauss", GAUSS), ("shuffle", SHUF)):
            band(ax, dx, [x["by_decile"] for x in g[kind]], c, NAME[kind])
        ax.plot(dx, g["real"]["by_decile"], "o-", color=REAL, lw=2, label="real")
        ax.set(title=title, xlabel="decile of straight distance (1 = nearest pairs)",
               ylabel="median graph path / straight distance", xticks=dx)
        ax.legend(fontsize=8, frameon=False)
    for key, c in zip(keys, plt.cm.tab10.colors):
        bd = np.array(ctl[key]["276"]["real"]["by_decile"])
        axs[2].plot(dx, bd / bd[4], color=c, lw=1.5, ls="-" if key.startswith("sphere") else "--",
                    label=key.replace("_", ", "))
    if m:
        bd = np.array(m["graph"]["real"]["by_decile"])
        axs[2].plot(dx, bd / bd[4], "o-", color=REAL, lw=2.5, label="real role means, L28")
    axs[2].set(title="Controls (N = 276), each divided by its 5th decile", xlabel="decile of straight distance",
               ylabel="path ratio / path ratio at decile 5", xticks=dx)
    axs[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_path_ratio_by_distance.png", dpi=150)
    plt.close(fig)

    # ---- fig3: against N ----
    if ans:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        ns = sorted(int(n) for n in ans["sweep"])
        for ax, key, lab in ((axs[0], "p50", "median graph path / straight distance"),
                             (axs[1], "trend", "trend: path ratio, decile 10 / decile 5")):
            for kind, c in (("gauss", GAUSS), ("shuffle", SHUF)):
                v = [[x[key] for x in ans["sweep"][str(n)][kind]] for n in ns]
                ax.fill_between(ns, [min(t) for t in v], [max(t) for t in v], color=c, alpha=0.25, lw=0,
                                label=NAME[kind] + " (min–max)")
            ax.plot(ns, [ans["sweep"][str(n)]["real"][key] for n in ns], "o-", color=REAL, lw=2, label="real")
            for ck, c in zip(keys, plt.cm.tab10.colors):
                ax.plot(ns, [ctl[ck][str(n)]["real"][key] for n in ns], color=c, lw=1, alpha=0.8,
                        ls="-" if ck.startswith("sphere") else "--", label=ck.replace("_", ", "))
            ax.set(xscale="log", xlabel="number of per-answer points N", ylabel=lab, title="Layer 28: " + lab)
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "fig3_against_N.png", dpi=150)
        plt.close(fig)

    # ---- fig4: PGA vs PCA match, real vs nulls ----
    if m:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        K = len(m["pga"]["real"]["max_rho"])
        kk = np.arange(1, K + 1)
        for kind, c in (("gauss", GAUSS), ("shuffle", SHUF)):
            band(ax, kk, [x["max_rho"] for x in m["pga"][kind]], c, NAME[kind])
        ax.plot(kk, m["pga"]["real"]["max_rho"], "o-", color=REAL, ms=4, lw=1.5, label="real")
        ax.set(xlabel="PGA component k", ylabel="max |Spearman ρ| with any PCA 1–50",
               title="Layer 28 role means: does PGA k copy a PCA direction?", ylim=(0, 1.02))
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "fig4_pga_vs_pca_on_nulls.png", dpi=150)
        plt.close(fig)

    # ---- fig5: layers ----
    if means:
        Ls = sorted(means)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        axs[0].plot(Ls, [S["layers"][L]["max_bend"] for L in Ls], "o-", color=REAL, label="real")
        axs[0].plot(Ls, [S["layers"][L]["max_bend_null"] for L in Ls], "s--", color=SHUF, label="shuffled null, max of draws")
        axs[0].axhline(R2_MIN, color="#999", ls=":", label="rule floor 0.10")
        axs[0].set(xlabel="layer", ylabel="max over PC 1–10 of the quadratic R²", title="Bend R², role means")
        axs[1].plot(Ls, [S["layers"][L]["trend"] for L in Ls], "o-", color=REAL, label="real")
        axs[1].plot(Ls, [S["layers"][L]["trend_null_max"] for L in Ls], "s--", color=GAUSS, label="nulls, max of draws")
        axs[1].axhline(1, color="#999", ls=":")
        axs[1].set(xlabel="layer", ylabel="path ratio, decile 10 / decile 5", title="Distance trend, role means")
        for key, c, lab in (("p50", REAL, "real"), ("p50_gauss", GAUSS, NAME["gauss"]), ("p50_shuffle", SHUF, NAME["shuffle"])):
            axs[2].plot(Ls, [S["layers"][L][key] for L in Ls], "o-", color=c, label=lab)
        axs[2].set(xlabel="layer", ylabel="median graph path / straight distance", title="Path ratio, role means")
        for ax in axs:
            ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "fig5_layers.png", dpi=150)
        plt.close(fig)

    # ---- tables ----
    T = ["| set | max bend R² (PC) | shuffled null max | trend | null max | verdict |", "|---|---|---|---|---|---|"]
    for k, r in S["controls"].items():
        T.append("| control %s | %.2f (PC %d) | %.2f | %.3f | %.3f | %s |" % (k.replace("_", ", "), r["max_bend"], r["pc_max_bend"],
                 r["max_bend_null"], r["trend"], r["trend_null_max"], r["verdict"]))
    for L, r in sorted(S["layers"].items()):
        T.append("| role means L%d | %.2f (PC %d) | %.2f | %.3f | %.3f | %s |" % (L, r["max_bend"], r["pc_max_bend"],
                 r["max_bend_null"], r["trend"], r["trend_null_max"], r["verdict"]))
    T += ["", "| layer | path ratio p50: real / Gaussian / shuffled | TwoNN: real / Gaussian / shuffled | check p50 (knn_graph) | check TwoNN |",
          "|---|---|---|---|---|"]
    for L, r in sorted(S["layers"].items()):
        T.append("| %d | %.2f / %.2f / %.2f | %.1f / %.1f / %.1f | %.2f | %.1f |" % (L, r["p50"], r["p50_gauss"], r["p50_shuffle"],
                 r["twonn"], r["twonn_gauss"], r["twonn_shuffle"], r["check"]["p50_knn_graph"], r["check"]["twonn_components"]))
    (out / "tables.md").write_text("\n".join(T) + "\n")
    print("\n".join(T))


if __name__ == "__main__":
    main()
