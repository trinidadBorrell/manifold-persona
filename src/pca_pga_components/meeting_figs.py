"""Part 1 figures and tables shown at the 16-10 meeting, from the component counts.

  fig1_components_by_layer      components for 90% of the variance, PCA vs Fletcher PGA, by layer
  fig2_variance_explained_best_layer, fig3_one_vs_several_layers
  table_direction_ends          personas at the - / + end of PCA and PGA directions 1-3 (best layer)
  table_overlap_pca_pga, overlap_pca_pga_ends

Input: $PM/P2full/components_{last,mean}.json (pm_geometry.components). Static versions of the
"How Many Persona Components" page (pca_pga_components.pages count).

    python -m pca_pga_components.meeting_figs [--p2 $PM/P2full] [--out output/11-oct]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from route_judge_sweep.build import PM

_ap = argparse.ArgumentParser()
_ap.add_argument("--p2", default=str(PM / "P2full"))
_ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "output" / "11-oct"))
_a = _ap.parse_args()
HERE = Path(_a.out)
HERE.mkdir(parents=True, exist_ok=True)
D = {"views": {}}
for _v in ("last", "mean"):
    _s = json.loads((Path(_a.p2) / ("components_%s.json" % _v)).read_text())
    D["layers"], D["views"][_v] = _s["layers"], _s["views"][_v]
VIEWS = [("last", "Last answer token (PersonaManifold)"), ("mean", "Mean over answer tokens")]
COL = {"pca": "#6b7f99", "pga": "#0f7b74"}
SERIES = [("pca", "PCA (straight lines)", ":"),
          ("pga", "Principal geodesic analysis (Fletcher)", "-")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Fig 1: components for 90% of the variance, by layer
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
L = D["layers"]
for ax, (v, title) in zip(axes, VIEWS):
    V = D["views"][v]
    for k, name, ls in SERIES:
        ax.plot(L, [V["per_layer"][str(l)]["n90"][k] for l in L], ls, marker="o", ms=4, color=COL[k], label=name)
    ax.set(title=title, xlabel="Layer (hidden_states index)", xticks=L[::2])
    ax.set_ylim(bottom=0)
axes[0].set_ylabel("Components for 90% of the variance")
axes[0].legend(fontsize=8, loc="upper left")
save(fig, "fig1_components_by_layer")

# Fig 2: variance explained vs number of components, at the most stable layer
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (v, title) in zip(axes, VIEWS):
    V = D["views"][v]; B = V["per_layer"][str(V["best_layer"])]
    for k, name, ls in SERIES:
        y = 100 * np.array(B[f"{k}_explained"])[:30]
        ax.plot(np.arange(1, len(y) + 1), y, ls, marker="o", ms=3, color=COL[k], label=name)
    ax.axhline(90, color="grey", ls=":", lw=1)
    ax.set(title=f"{title}, layer {V['best_layer']}", xlim=(0.5, 30.5), ylim=(0, 101),
           xlabel="Number of components")
axes[0].set_ylabel("Variance explained (%)")
axes[0].legend(fontsize=8, loc="lower right")
save(fig, "fig2_variance_explained_best_layer")

# Fig 3: one layer vs several layers
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (v, title) in zip(axes, VIEWS):
    V = D["views"][v]; b0, b1 = V["band"][0], V["band"][-1]
    vers = [("M0", f"One layer\n(L{V['best_layer']})"), ("M1", f"Several layers,\ngeodesic (L{b0}–{b1})"),
            ("M1_all_layers", "All layers 12–32,\ngeodesic"), ("M2", f"Several layers,\nPCA (L{b0}–{b1})")]
    pga = [V[m]["n90"]["pga"] if isinstance(V[m]["n90"], dict) else V[m]["n90"] for m, _ in vers]
    stab = [np.mean(V[m]["stability"].get("pga_top3_principal_cos") or V[m]["stability"]["pca_top3_principal_cos"]) for m, _ in vers]
    x = np.arange(len(vers)); w = 0.5
    b1_ = ax.bar(x, pga, w, color=COL["pga"], label="Components for 90% (Fletcher; PCA for the last bar)")
    ax.bar_label(b1_, fontsize=8)
    ax.set(title=title, xticks=x, xticklabels=[n for _, n in vers], ylabel="Number needed for 90%")
    ax.tick_params(axis="x", labelsize=8)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(x, stab, "D", color="black", ms=8, label="Top-3 directions from two question halves (mean cosine)")
    ax2.set(ylim=(0, 1.02), ylabel="Split-half agreement (cosine)")
    ax2.spines["right"].set_visible(True)
h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
save(fig, "fig3_one_vs_several_layers")

# Table: personas at the - and + ends of PCA and Fletcher (PGA) directions 1-3, at the fig2 layer
out = ["# Personas at the ends of the first three directions",
       "",
       "Each persona's score along a direction is its signed projection on it. The − end lists the 6 personas "
       "with the most negative scores, the + end the 6 with the most positive. Which end is − and which is + is arbitrary.",
       ""]
for v, title in VIEWS:
    V = D["views"][v]; B = V["per_layer"][str(V["best_layer"])]
    out += [f"## {title}, layer {V['best_layer']}", "",
            "| Method | Direction | Share of variance | − end | + end |", "|---|---|---|---|---|"]
    for key, name in (("pca_ends", "PCA"), ("pga_ends", "PGA (Fletcher)")):
        for e in B[key][:3]:
            out.append(f"| {name} | {e['component']} | {100 * e['var_share']:.0f}% | "
                       f"{', '.join(e['minus'])} | {', '.join(e['plus'])} |")
    out.append("")
(HERE / "table_direction_ends.md").write_text("\n".join(out))

# Same table as a PNG/PDF: one table per persona-vector view, stacked
import textwrap
fig, axes = plt.subplots(2, 1, figsize=(10, 10.5))
for ax, (v, title) in zip(axes, VIEWS):
    V = D["views"][v]; B = V["per_layer"][str(V["best_layer"])]
    rows = [[name, str(e["component"]), f"{100 * e['var_share']:.0f}%",
             textwrap.fill(", ".join(e["minus"]), 30), textwrap.fill(", ".join(e["plus"]), 30)]
            for key, name in (("pca_ends", "PCA"), ("pga_ends", "PGA\n(Fletcher)")) for e in B[key][:3]]
    ax.axis("off")
    ax.set_title(f"{title}, layer {V['best_layer']}", fontsize=11, loc="left")
    t = ax.table(cellText=rows, colLabels=["Method", "Direction", "Share of\nvariance", "− end", "+ end"],
                 colWidths=[0.15, 0.1, 0.11, 0.32, 0.32], loc="upper center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 3.4)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#eeeeee")
        elif rows[r - 1][0] == "PCA":
            cell.set_facecolor("#f4f6f9")
        else:
            cell.set_facecolor("#eef6f5")
fig.suptitle("Personas at the ends of the first three directions\n(6 most negative / 6 most positive scores; sign is arbitrary)",
             fontsize=11, x=0.02, ha="left")
fig.tight_layout()
save(fig, "table_direction_ends")

# Shared personas between PCA ends and PGA ends (directions 1-3, 6 personas per end)
fig, axes = plt.subplots(2, 1, figsize=(10, 15))
for ax, (v, title) in zip(axes, VIEWS):
    V = D["views"][v]; B = V["per_layer"][str(V["best_layer"])]
    ends = lambda key, n: [(f"{n} {e['component']} {s}", set(e[k])) for e in B[key][:3] for k, s in (("minus", "−"), ("plus", "+"))]
    rows, cols = ends("pca_ends", "PCA"), ends("pga_ends", "PGA")
    M = np.array([[len(a & b) for _, b in cols] for _, a in rows])
    ax.imshow(M, cmap="Greens", vmin=0, vmax=6)
    for i, (_, a) in enumerate(rows):
        for j, (_, b) in enumerate(cols):
            s = sorted(a & b)
            if s:
                ax.text(j, i, f"{len(s)}\n" + "\n".join(s), ha="center", va="center", fontsize=7.5,
                        color="white" if len(s) >= 3 else "black")
    ax.set_xticks(range(6), [c for c, _ in cols]); ax.set_yticks(range(6), [r for r, _ in rows])
    ax.xaxis.tick_top(); ax.grid(False)
    ax.set_xticks(np.arange(-.5, 6), minor=True); ax.set_yticks(np.arange(-.5, 6), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    ax.set_title(f"{title}, layer {V['best_layer']}", fontsize=11, loc="left", pad=28)
    ax.set_ylabel("PCA direction and end"); ax.set_xlabel("PGA (Fletcher) direction and end")
fig.suptitle("Personas shared between PCA and PGA direction ends (out of 6 per end)", fontsize=12, x=0.02, ha="left")
fig.tight_layout()
save(fig, "overlap_pca_pga_ends")

# Table of PCA-end / PGA-end overlaps for the same direction number (all four end pairings, zeros included),
# last-token view at layer 28, fewest to most shared
V = D["views"]["last"]; B = V["per_layer"][str(V["best_layer"])]
rows = []
for a in B["pca_ends"][:3]:
    for ka, sa in (("minus", "−"), ("plus", "+")):
        b = B["pga_ends"][a["component"] - 1]
        for kb, sb in (("minus", "−"), ("plus", "+")):
            sh = sorted(set(a[ka]) & set(b[kb]))
            rows.append([f"PCA {a['component']} {sa}", f"PGA {b['component']} {sb}", len(sh), ", ".join(sh) or "—"])
rows.sort(key=lambda r: (r[2], r[0][4], r[0], r[1]))
title = f"Personas shared between PCA and PGA (Fletcher) direction ends — last answer token, layer {V['best_layer']}"
md = [f"# {title}", "", "Same direction number only (PCA k vs PGA k), all four end pairings. Out of the 6 personas listed per end. Sorted from fewest to most.", "",
      "| PCA end | PGA end | Shared (of 6) | Personas |", "|---|---|---|---|"]
md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows]
(HERE / "table_overlap_pca_pga.md").write_text("\n".join(md) + "\n")

fig, ax = plt.subplots(figsize=(7.5, 0.24 * (len(rows) + 1) + 0.5))
ax.axis("off")
t = ax.table(cellText=[[r[0], r[1], str(r[2]), r[3]] for r in rows], colLabels=["PCA end", "PGA end", "Shared (of 6)", "Personas"],
             colWidths=[0.13, 0.13, 0.15, 0.59], loc="upper center", cellLoc="left")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.25)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_text_props(weight="bold"); cell.set_facecolor("#eeeeee")
    else:
        cell.set_facecolor(plt.cm.Greens(0.08 + 0.12 * rows[r - 1][2]))
ax.set_title(f"Personas shared between PCA k and PGA (Fletcher) k direction ends\nlast answer token, layer {V['best_layer']}, fewest to most",
             fontsize=10, loc="left")
fig.tight_layout()
save(fig, "table_overlap_pca_pga")
