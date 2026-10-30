"""Part A figures and tables from `$PM/PGALIN/pgalin_<view>_L<L>.json` (one file per layer).

    python -m pga_linearity.analyse --dir $PM/PGALIN --out output/29-oct/partA [--layers 19,25,32]

Writes tables.md, summary.json and
  F12_subspace_overlap.png        mean cos^2 of the principal angles between span(PCA 1..k) and span(PGA 1..k), real vs nulls
  F13_direction_cosines.png       |cos(PGA k, PCA j)|, k, j <= 20, one panel per layer
  F14_geodesic_linearity.png      the four A4 measures per PGA direction, real vs nulls
  F15_principal_geodesics.png     PGA 1-3 polylines drawn in PCA 1-3 coordinates (primary layer)
  F16_reconstruction_gap.png      R^2 in span(PGA 1..k) and span(PCA 1..k), real vs nulls
  F17_logmap_fidelity.png         cos(log-map direction, true displacement from the Frechet mean) per persona, real vs nulls
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PM = Path("/data/project/eeg_foundation/data/manifold_persona/personamanifold")
C_REAL, C_GAUSS, C_SHUF = "#2a78d6", "#eb6834", "#1baf7a"     # fixed categorical order: real, Gaussian null, shuffled null
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 150})


def load(d, view, layers):
    return {L: json.loads((Path(d) / f"pgalin_{view}_L{L}.json").read_text()) for L in layers}




def stack(draws, key, field, sub=None):
    rows = []
    for d in draws:
        vals = []
        for x in d[key]:
            v = x[sub][field] if sub else x[field]
            vals.append(np.nan if v is None else v)
        rows.append(vals)
    return np.array(rows, dtype=float)


def pct(A):
    return np.nanpercentile(A, 5, 0), np.nanpercentile(A, 50, 0), np.nanpercentile(A, 95, 0)


def real_series(r, key, field, sub=None):
    return np.array([np.nan if (x[sub][field] if sub else x[field]) is None else (x[sub][field] if sub else x[field])
                     for x in r["real"][key]], dtype=float)


def draw_band(ax, k, A, color, label):
    lo, md, hi = pct(A)
    ax.fill_between(k, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(k, md, color=color, linewidth=1.2, linestyle="--", label=label)


# ---------------------------------------------------------------------------------------------

def fig_overlap(R, out):
    layers = list(R)
    fig, axes = plt.subplots(1, len(layers), figsize=(3.3 * len(layers), 3.0), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), layers):
        r = R[L]
        k = np.arange(1, len(r["real"]["A1"]) + 1)
        draw_band(ax, k, stack(r["null"]["gauss"], "A1", "mean_cos2"), C_GAUSS, "Gaussian null, 5–95%")
        draw_band(ax, k, stack(r["null"]["shuffle"], "A1", "mean_cos2"), C_SHUF, "shuffled null, 5–95%")
        ax.plot(k, real_series(r, "A1", "mean_cos2"), color=C_REAL, linewidth=2, marker="o", markersize=3, label="real personas")
        ax.set_title(f"layer {L}", color=INK)
        ax.set_xlabel("k (number of directions)")
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("mean cos² of principal angles\nspan(PCA 1..k) vs span(PGA 1..k)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "F12_subspace_overlap.png")
    plt.close(fig)


def fig_cosines(R, out, kmax=20):
    layers = list(R)
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.4))
    for ax, L in zip(np.atleast_1d(axes), layers):
        f = np.load(Path(R[L]["_frame"]))
        V = np.abs(f["V"][:kmax, :kmax])           # rows PCA j, cols PGA k
        im = ax.imshow(V.T, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"layer {L}", color=INK)
        ax.set_xlabel("PCA direction j")
        ax.set_ylabel("PGA direction k")
        ax.set_xticks(np.arange(0, kmax, 5)); ax.set_xticklabels(np.arange(1, kmax + 1, 5))
        ax.set_yticks(np.arange(0, kmax, 5)); ax.set_yticklabels(np.arange(1, kmax + 1, 5))
        ax.grid(False)
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), fraction=0.02, pad=0.02, label="|cos(PGA k, PCA j)|")
    fig.savefig(out / "F13_direction_cosines.png", bbox_inches="tight")
    plt.close(fig)


A4_FIELDS = [("line_r2", "straight-line R² of the polyline's points\n(PC1 share)"),
             ("arc_over_chord", "arc length / chord"),
             ("max_dev_over_chord", "max distance from chord / chord"),
             ("turning_deg", "mean turning angle between segments (°)")]


def fig_geodesic(R, out, sub="polyline"):
    layers = list(R)
    fig, axes = plt.subplots(len(A4_FIELDS), len(layers), figsize=(3.3 * len(layers), 2.3 * len(A4_FIELDS)), sharex=True)
    axes = np.array(axes).reshape(len(A4_FIELDS), len(layers))
    for j, L in enumerate(layers):
        r = R[L]
        k = np.arange(1, len(r["real"]["A4"]) + 1)
        for i, (field, lab) in enumerate(A4_FIELDS):
            ax = axes[i, j]
            draw_band(ax, k, stack(r["null"]["gauss"], "A4", field, sub), C_GAUSS, "Gaussian null, 5–95%")
            draw_band(ax, k, stack(r["null"]["shuffle"], "A4", field, sub), C_SHUF, "shuffled null, 5–95%")
            ax.plot(k, real_series(r, "A4", field, sub), color=C_REAL, linewidth=2, marker="o", markersize=3, label="real personas")
            if i == 0:
                ax.set_title(f"layer {L}", color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == len(A4_FIELDS) - 1:
                ax.set_xlabel("PGA direction k")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle({"polyline": "graph polyline through the Fréchet mean between the two cone ends",
                  "cone_ordered": "personas inside the 30° cone, ordered by their PGA coordinate"}[sub], fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / f"F14_geodesic_linearity_{sub}.png")
    plt.close(fig)


def fig_polylines(R, out, L):
    f = np.load(Path(R[L]["_frame"]), allow_pickle=True)
    Z, roles = f["Z"], f["roles"]
    r = R[L]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    pairs = [(0, 1), (0, 2), (1, 2)]
    cols = [C_REAL, C_GAUSS, C_SHUF]
    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(Z[:, a], Z[:, b], s=6, color="#c3c2b7", linewidths=0, zorder=1)
        for c, rec in zip(cols, r["real"]["A4"][:3]):
            idx = [list(roles).index(n) for n in rec["path_roles"]]
            P = Z[idx]
            ax.plot(P[:, a], P[:, b], color=c, linewidth=1.6, marker="o", markersize=3.5, zorder=3,
                    label=f"PGA {rec['k']}: {rec['ends']['minus']} → {rec['ends']['plus']}")
            ax.plot([P[0, a], P[-1, a]], [P[0, b], P[-1, b]], color=c, linewidth=0.9, linestyle=":", zorder=2)
        mu = list(roles).index(r["real"]["frechet_mean"])
        ax.scatter(Z[mu, a], Z[mu, b], s=40, facecolors="none", edgecolors=INK, zorder=4)
        ax.set_xlabel(f"PCA {a + 1}"); ax.set_ylabel(f"PCA {b + 1}")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"layer {L}: the principal geodesic polylines (solid) and their chords (dotted); circle = Fréchet mean ({r['real']['frechet_mean']})",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "F15_principal_geodesics.png")
    plt.close(fig)


def fig_gap(R, out):
    layers = list(R)
    fig, axes = plt.subplots(1, len(layers), figsize=(3.3 * len(layers), 3.0), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), layers):
        r = R[L]
        k = np.arange(1, len(r["real"]["A5"]) + 1)
        ax.plot(k, real_series(r, "A5", "r2_pca"), color=INK, linewidth=1.4, linestyle="-", label="span(PCA 1..k), real")
        draw_band(ax, k, stack(r["null"]["gauss"], "A5", "r2_pga"), C_GAUSS, "span(PGA 1..k), Gaussian null 5–95%")
        draw_band(ax, k, stack(r["null"]["shuffle"], "A5", "r2_pga"), C_SHUF, "span(PGA 1..k), shuffled null 5–95%")
        ax.plot(k, real_series(r, "A5", "r2_pga"), color=C_REAL, linewidth=2, marker="o", markersize=3, label="span(PGA 1..k), real")
        ax.set_title(f"layer {L}", color=INK); ax.set_xlabel("k (number of directions)")
    np.atleast_1d(axes)[0].set_ylabel("R² of the 276 persona vectors\nreconstructed in the span")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "F16_reconstruction_gap.png")
    plt.close(fig)


def fig_fidelity(R, out):
    layers = list(R)
    fig, axes = plt.subplots(1, len(layers), figsize=(3.3 * len(layers), 3.0), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), layers):
        r = R[L]
        real = np.array([c for c in r["real"]["A7"]["per_persona_cos"] if c is not None])
        for kind, col, lab in (("gauss", C_GAUSS, "Gaussian null"), ("shuffle", C_SHUF, "shuffled null")):
            med = np.array([d["A7"]["median"] for d in r["null"][kind]])
            f05 = np.array([d["A7"]["frac_below_0_5"] for d in r["null"][kind]])
            ax.axvspan(np.percentile(med, 5), np.percentile(med, 95), color=col, alpha=0.18, linewidth=0,
                       label=f"{lab}: median over personas, 5–95% of draws")
        xs = np.sort(real)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=C_REAL, linewidth=2, label="real personas (ECDF)")
        ax.axvline(np.median(real), color=C_REAL, linewidth=1, linestyle="--")
        ax.set_title(f"layer {L}: {r['real']['A7']['frac_below_0_5']:.0%} of personas below 0.5", color=INK)
        ax.set_xlabel("cos(log-map direction, true displacement)" if L == layers[len(layers) // 2] else "")
        ax.set_xlim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("share of personas (ECDF)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.suptitle("cos between each persona's log-map direction (first two hops from the Fréchet mean) and its true displacement", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "F17_logmap_fidelity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------

def fmt(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def null_range(draws, key, field, i, sub=None):
    A = stack(draws, key, field, sub)[:, i]
    return f"{np.nanpercentile(A, 5):.2f}–{np.nanpercentile(A, 95):.2f}"


def tables(R, out):
    lines = ["# Part A — PGA vs PCA directions (axis level)\n",
             "Real personas against the 5th–95th percentile of 20 Gaussian and 20 shuffled null draws with the same per-axis variances. "
             "View `last`, 276 Assistant Axis personas of Qwen3-8B.\n"]
    summ = {}
    for L, r in R.items():
        s = summ[str(L)] = dict(frechet_mean=r["real"]["frechet_mean"], graph_k=r["real"]["graph_k"])
        lines.append(f"\n## Layer {L}  (Fréchet mean: {r['real']['frechet_mean']}, graph k = {r['real']['graph_k']})\n")
        lines.append("### A1 — mean cos² of the principal angles between span(PCA 1..k) and span(PGA 1..k)\n")
        lines.append("| k | real | Gaussian null | shuffled null |\n|---|---|---|---|")
        for i in (0, 1, 2, 4, 9, 19):
            if i < len(r["real"]["A1"]):
                lines.append(f"| {i + 1} | {fmt(r['real']['A1'][i]['mean_cos2'])} | {null_range(r['null']['gauss'], 'A1', 'mean_cos2', i)} | {null_range(r['null']['shuffle'], 'A1', 'mean_cos2', i)} |")
        s["A1_mean_cos2"] = {str(x["k"]): x["mean_cos2"] for x in r["real"]["A1"]}
        lines.append("\n### A2, A3 — each PGA direction against the PCA directions\n")
        lines.append("| PGA k | var. share | closest PCA j (\\|cos\\|) | share inside PCA 1..10 / 1..20 / 1..50 | null share in PCA 1..10 (Gauss) | max \\|Spearman ρ\\| with a PCA score (j) | null ρ (Gauss) |\n|---|---|---|---|---|---|---|")
        for i in range(10):
            a2, a3 = r["real"]["A2"][i], r["real"]["A3"][i]
            g10 = np.array([d["A2"][i]["share_in_pca"]["10"] for d in r["null"]["gauss"]])
            lines.append(f"| {i + 1} | {r['real']['pga_var_share'][i]:.3f} | {a2['best_pca']} ({a2['max_cos']:.2f}) | "
                         f"{a2['share_in_pca']['10']:.2f} / {a2['share_in_pca']['20']:.2f} / {a2['share_in_pca']['50']:.2f} | "
                         f"{np.percentile(g10, 5):.2f}–{np.percentile(g10, 95):.2f} | {a3['max_rho']:.2f} ({a3['best_pca']}) | "
                         f"{null_range(r['null']['gauss'], 'A3', 'max_rho', i)} |")
        s["A2"] = r["real"]["A2"][:10]; s["A3"] = r["real"]["A3"][:10]
        for sub, title in (("polyline", "graph polyline through the Fréchet mean between the two cone ends"),
                           ("cone_ordered", "personas inside the 30° cone ordered by their PGA coordinate")):
            lines.append(f"\n### A4 — linearity of the principal geodesic: {title}\n")
            lines.append("| PGA k | ends (−, +) | n cone | n points | straight-line R² (null) | arc / chord (null) | max dev / chord (null) | turning ° (null) |\n|---|---|---|---|---|---|---|---|")
            for i, rec in enumerate(r["real"]["A4"][:5]):
                m = rec[sub]
                lines.append(f"| {rec['k']} | {rec['ends']['minus']}, {rec['ends']['plus']} | {rec['n_cone']} | {m['n']} | "
                             f"{fmt(m['line_r2'])} ({null_range(r['null']['gauss'], 'A4', 'line_r2', i, sub)}) | "
                             f"{fmt(m['arc_over_chord'])} ({null_range(r['null']['gauss'], 'A4', 'arc_over_chord', i, sub)}) | "
                             f"{fmt(m['max_dev_over_chord'])} ({null_range(r['null']['gauss'], 'A4', 'max_dev_over_chord', i, sub)}) | "
                             f"{fmt(m['turning_deg'], 0)} ({null_range(r['null']['gauss'], 'A4', 'turning_deg', i, sub)}) |")
        s["A4"] = r["real"]["A4"][:5]
        lines.append("\n### A5 — R² of the persona vectors reconstructed in the span\n")
        lines.append("| k | span(PCA 1..k) | span(PGA 1..k) | gap | gap on Gaussian null | gap on shuffled null |\n|---|---|---|---|---|---|")
        for i in (0, 1, 2, 4, 9, 19):
            if i < len(r["real"]["A5"]):
                x = r["real"]["A5"][i]
                lines.append(f"| {i + 1} | {x['r2_pca']:.3f} | {x['r2_pga']:.3f} | {x['gap']:.3f} | {null_range(r['null']['gauss'], 'A5', 'gap', i)} | {null_range(r['null']['shuffle'], 'A5', 'gap', i)} |")
        s["A5"] = r["real"]["A5"]
        a7 = r["real"]["A7"]
        lines.append("\n### A7 — log-map fidelity: cos(log-map direction, true displacement from the Fréchet mean), per persona\n")
        lines.append("| | real | Gaussian null (5–95% of draws) | shuffled null |\n|---|---|---|---|")
        for key, lab in (("median", "median over personas"), ("p10", "10th percentile"), ("frac_below_0_5", "share of personas with cos < 0.5"),
                         ("frac_negative", "share with cos < 0"), ("hops_mean", "mean path hops from the Fréchet mean"),
                         ("spearman_abs_pga1_vs_cos", "Spearman: |PGA 1 coordinate| vs cos")):
            g = np.array([d["A7"][key] for d in r["null"]["gauss"]]); sh = np.array([d["A7"][key] for d in r["null"]["shuffle"]])
            lines.append(f"| {lab} | {a7[key]:.2f} | {np.percentile(g, 5):.2f}–{np.percentile(g, 95):.2f} | {np.percentile(sh, 5):.2f}–{np.percentile(sh, 95):.2f} |")
        lines.append("- mean cos by path hops: " + ", ".join(f"{h} hops {v:.2f}" for h, v in a7["mean_cos_by_hops"].items()))
        lines.append("- lowest fidelity: " + ", ".join(f"{x['role']} ({x['cos']:.2f}, {x['hops']} hops)" for x in a7["lowest"][:8]))
        s["A7"] = {k: v for k, v in a7.items() if k not in ("per_persona_cos", "per_persona_hops")}
        h = r["halves"]
        lines.append("\n### A6 — question halves A and B\n")
        lines.append(f"- principal cosines between the top-3 PGA directions of the two halves: {', '.join(f'{c:.2f}' for c in h['pga_top3_principal_cos_A_vs_B'])}; |cos| of PGA 1: {h['pga1_cos_A_vs_B']:.2f}")
        lines.append(f"- A1 mean cos² at k = 3: half A {h['A']['A1'][2]['mean_cos2']:.2f}, half B {h['B']['A1'][2]['mean_cos2']:.2f}, all questions {r['real']['A1'][2]['mean_cos2']:.2f}")
        lines.append(f"- A4 straight-line R² of the PGA 1 polyline: half A {fmt(h['A']['A4'][0]['polyline']['line_r2'])}, half B {fmt(h['B']['A4'][0]['polyline']['line_r2'])}, all {fmt(r['real']['A4'][0]['polyline']['line_r2'])}")
        s["A6"] = dict(top3_principal_cos=h["pga_top3_principal_cos_A_vs_B"], pga1_cos=h["pga1_cos_A_vs_B"])
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PM / "PGALIN"))
    ap.add_argument("--out", default="output/29-oct/partA")
    ap.add_argument("--view", default="last")
    ap.add_argument("--layers", default="19,25,32")
    ap.add_argument("--primary", type=int, default=25)
    a = ap.parse_args(argv)
    layers = [int(x) for x in a.layers.split(",")]
    R = load(a.dir, a.view, layers)
    for L in layers:
        R[L]["_frame"] = str(Path(a.dir) / f"frame_{a.view}_L{L}.npz")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_overlap(R, out)
    fig_cosines(R, out)
    fig_geodesic(R, out, "polyline")
    fig_geodesic(R, out, "cone_ordered")
    fig_polylines(R, out, a.primary)
    fig_gap(R, out)
    fig_fidelity(R, out)
    tables(R, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
