"""Figures for plan 2026-07-21-role-manifold-reconstruction.

All matplotlib PNG at 300 dpi; fig02 also emits an interactive plotly HTML.
Every function is defensive — a plotting failure logs and returns, never kills
the run that already produced the numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from manifold_persona.runlog import save_fig  # noqa: E402


def _save(fig, path: Path):
    save_fig(fig, path)          # dpi=300


def fig_posctrl(pc: dict, out: Path):
    try:
        cloud = pc["cloud"]; centers = pc["centers"]
        fig = plt.figure(figsize=(11, 4.5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(cloud.raw[:, 0], cloud.raw[:, 1], cloud.raw[:, 2], s=3,
                   alpha=0.15, c="#888")
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], c="#d1495b", lw=2)
        ax.set_title(f"Positive control: synthetic arc\nR²={pc['r2']:.3f} "
                     f"(null {pc['stats']['null_median']:.3f}, "
                     f"rel.red {pc['stats']['rel_reduction']:.2f})")
        ax.set_xlabel("dim0"); ax.set_ylabel("dim1"); ax.set_zlabel("dim2")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(pc["null"], bins=20, color="#bbb", label="null R²")
        ax2.axvline(pc["r2"], color="#d1495b", lw=2, label="real R²")
        ax2.set_xlabel("R²"); ax2.set_ylabel("count"); ax2.legend()
        ax2.set_title("control must clear its null")
        _save(fig, out / "figPC_positive_control.png")
    except Exception as e:  # noqa: BLE001
        print("figPC failed:", e)


def fig01_null_vs_real(report: dict, nulls: dict, out: Path):
    try:
        names, reals, nulldists = [], [], []
        # decider
        names.append("C_role"); reals.append(report["decider"]["r2"])
        nulldists.append(np.array(report["null_decider"]))
        for pct, d in report.get("tau", {}).items():
            key = f"C_tau{pct}"
            if key in nulls:
                names.append(key); reals.append(d["r2"]); nulldists.append(nulls[key])
        if "C_role_last" in nulls and "prompt_last" in report:
            names.append("C_role[last]"); reals.append(report["prompt_last"]["r2"])
            nulldists.append(nulls["C_role_last"])
        fig, ax = plt.subplots(figsize=(9, 5))
        parts = ax.violinplot(nulldists, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#cfd8dc"); pc.set_alpha(0.8)
        for i, r in enumerate(reals):
            c = "#2e7d32" if i == 0 else "#1565c0"
            ax.scatter(i + 1, r, color=c, zorder=3, s=60)
            ax.annotate(f"{r:.2f}", (i + 1, r), textcoords="offset points",
                        xytext=(8, 0), fontsize=9)
        ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, rotation=20)
        ax.set_ylabel("manifold R²")
        ax.set_title("Real R² (points) vs role-shuffle null (violins)\n"
                     "decider = C_role (green); others exploratory (blue)")
        _save(fig, out / "fig01_null_vs_real.png")
    except Exception as e:  # noqa: BLE001
        print("fig01 failed:", e)


def fig02_fig03_manifold(cloud, comp_tau2, mani, out: Path):
    try:
        rm = cloud.role_means
        is_def = np.array([r == "default" for r in cloud.role_names])
        # decode a grid on the surface for overlay (in PCA-3 view)
        gg = np.linspace(0, 1, 30)
        # 2D figure (PC1-2)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(rm[:, 0], rm[:, 1], c=comp_tau2, cmap="tab20", s=20)
        ax.scatter(rm[is_def, 0], rm[is_def, 1], marker="*", s=280,
                   edgecolor="k", facecolor="gold", label="default (Assistant)", zorder=5)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("Role means in PC1–2, coloured by cosine component (τ50)")
        ax.legend()
        _save(fig, out / "fig03_roles_pc12.png")

        # 3D figure (PC1-3) with role means
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(rm[:, 0], rm[:, 1], rm[:, 2], c=comp_tau2, cmap="tab20", s=18)
        ax.scatter(rm[is_def, 0], rm[is_def, 1], rm[is_def, 2], marker="*",
                   s=280, edgecolor="k", facecolor="gold", zorder=5)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title("Role means in PC1–3, coloured by cosine component (τ50)")
        _save(fig, out / "fig02_roles_pc123.png")

        # interactive plotly HTML
        try:
            import plotly.graph_objects as go
            fig3 = go.Figure(data=[go.Scatter3d(
                x=rm[:, 0], y=rm[:, 1], z=rm[:, 2], mode="markers",
                marker=dict(size=3, color=comp_tau2, colorscale="Turbo"),
                text=cloud.role_names)])
            fig3.update_layout(title="Role means (PC1–3), coloured by τ50 component",
                               scene=dict(xaxis_title="PC1", yaxis_title="PC2",
                                          zaxis_title="PC3"))
            fig3.write_html(str(out / "fig02_roles_pc123.html"))
            print("wrote", out / "fig02_roles_pc123.html")
        except Exception as e:  # noqa: BLE001
            print("fig02 html skipped:", e)
    except Exception as e:  # noqa: BLE001
        print("fig02/03 failed:", e)


def fig04_manifolds_vs_tau(tau_results: dict, out: Path):
    try:
        pcts = sorted(tau_results)
        taus = [tau_results[p]["tau"] for p in pcts]
        nman = [tau_results[p]["n_manifolds"] for p in pcts]
        largest = [max(tau_results[p]["component_sizes"]) for p in pcts]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = range(len(pcts))
        ax.bar([i - 0.2 for i in x], nman, width=0.4, label="# manifolds (≥3 roles)",
               color="#1565c0")
        ax.bar([i + 0.2 for i in x], largest, width=0.4,
               label="largest component size", color="#ef9a9a")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"τ{p}\n={tau_results[p]['tau']:.2f}" for p in pcts])
        ax.set_ylabel("count"); ax.legend()
        ax.set_title("Manifold count and largest component vs cosine threshold")
        _save(fig, out / "fig04_manifolds_vs_tau.png")
    except Exception as e:  # noqa: BLE001
        print("fig04 failed:", e)


def fig05_spline_vs_plane(report: dict, rows, out: Path):
    try:
        labels = ["C_role\nspline", "PCA-plane\n(k=3)", "C_role\nnull median"]
        vals = [report["decider"]["r2"], report["plane_r2"],
                report["decider"]["null_median"]]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(labels, vals, color=["#2e7d32", "#8d6e63", "#bbb"])
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=10)
        ax.set_ylabel("R²")
        ax.set_title("Curved spline vs flat plane vs chance (context)")
        _save(fig, out / "fig05_spline_vs_plane.png")
    except Exception as e:  # noqa: BLE001
        print("fig05 failed:", e)
