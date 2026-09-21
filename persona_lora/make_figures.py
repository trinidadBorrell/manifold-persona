"""Figures for the lab notebook.

fig1  in-role % with 95% Wilson CIs, every arm, against the two reference lines
fig2  held-out KL vs in-role % -- the rank inversion
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical palette (light mode): blue / orange / aqua / violet
C = {"steering": "#eb6834", "conditional": "#2a78d6", "per-persona": "#1baf7a",
     "baseline": "#6b6a66"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
INROLE = ("human_role", "nonhuman_role", "weird_role")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load(scratch: Path):
    items, res = {}, {}
    for tag, f in [("core", "core_items.jsonl"), ("rest", "rest_items.jsonl"),
                   ("pp", "pp_items.jsonl"), ("ml", "midlate_items.jsonl")]:
        p = scratch / f
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                items[(tag, r["id"])] = r
    for tag, pat in [("core", "judge_core_*.jsonl"), ("rest", "judge_rest_*.jsonl"),
                     ("pp", "judge_pp_*.jsonl"), ("ml", "judge_midlate_*.jsonl")]:
        for f in glob.glob(str(scratch / pat)):
            for l in open(f):
                r = json.loads(l)
                res[(tag, r["id"])] = r["score"]
    by = collections.defaultdict(collections.Counter)
    for k, sc in res.items():
        it = items.get(k)
        if it is None or it["role"] == "default":
            continue
        by[it["cfg"]][sc] += 1
    return by


def family(cfg):
    if cfg.startswith("steer"):
        return "steering"
    if cfg.startswith("cond"):
        return "conditional"
    if cfg.startswith("pp_"):
        return "per-persona"
    return "baseline"


def pretty(cfg):
    return (cfg.replace("_withprompt", " + prompt").replace("_alone", " alone")
               .replace("cond_", "cond ").replace("pp_", "per-persona ")
               .replace("steer_band_a", "steering α=").replace("prompted_n4", "prompt only")
               .replace("base_n4", "base (no prompt)"))


def fig1(by, out: Path):
    rows = []
    for cfg, c in by.items():
        n = sum(c.values())
        k = sum(c[x] for x in INROLE)
        lo, hi = wilson(k, n)
        rows.append((100 * k / n, lo, hi, cfg, k, n))
    rows.sort()
    ref_prompt = next(r[0] for r in rows if r[3] == "prompted_n4")
    ref_steer = next(r[0] for r in rows if r[3] == "steer_band_a4")

    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(rows) + 1.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ys = range(len(rows))
    for y, (pct, lo, hi, cfg, k, n) in zip(ys, rows):
        col = C[family(cfg)]
        ax.barh(y, pct, height=0.62, color=col, zorder=3,
                edgecolor=SURFACE, linewidth=1.2)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.6, alpha=.55, zorder=4,
                solid_capstyle="butt")
        ax.text(hi + 1.2, y, f"{pct:.0f}%", va="center", ha="left",
                fontsize=8.5, color=INK)
    ax.axvline(ref_prompt, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.axvline(ref_steer, color=C["steering"], lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.text(ref_prompt, len(rows) - .2, " prompt only", fontsize=8, color=MUTED,
            ha="left", va="bottom")
    ax.text(ref_steer, len(rows) - .2, " steering α=4", fontsize=8,
            color=C["steering"], ha="left", va="bottom")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([pretty(r[3]) for r in rows], fontsize=8.5, color=INK)
    ax.set_xlabel("responses judged in-role (%)   ·   95% Wilson CI   ·   n = 140 per arm",
                  fontsize=9, color=MUTED)
    ax.set_xlim(0, 78)
    ax.set_ylim(-0.8, len(rows) - 0.1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d4")
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ecebe7", lw=.8, zorder=0)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=C[k], lw=7, label=k)
               for k in ("steering", "conditional", "per-persona", "baseline")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
              labelcolor=INK, ncol=2)
    ax.set_title("No adapter beats the steering baseline",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)
    return rows


def fig2(by, kl: dict, out: Path):
    pts = []
    for cfg, c in by.items():
        n = sum(c.values())
        k = sum(c[x] for x in INROLE)
        base = cfg.replace("_withprompt", "").replace("_alone", "")
        if base in kl and cfg.endswith("withprompt"):
            pts.append((kl[base], 100 * k / n, cfg))
    if len(pts) < 3:
        print("fig2: not enough matched points")
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for x, y, cfg in pts:
        col = C[family(cfg)]
        ax.scatter(x * 1000, y, s=95, color=col, zorder=3,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.annotate(pretty(cfg).replace(" + prompt", ""), (x * 1000, y),
                    textcoords="offset points", xytext=(9, -3),
                    fontsize=8, color=INK)
    # rank correlation
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    rx = sorted(range(len(xs)), key=lambda i: xs[i])
    ry = sorted(range(len(ys)), key=lambda i: ys[i])
    rank_x = {i: r for r, i in enumerate(rx)}
    rank_y = {i: r for r, i in enumerate(ry)}
    n = len(pts)
    d2 = sum((rank_x[i] - rank_y[i]) ** 2 for i in range(n))
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    ax.set_xlabel("held-out KL to the prompted model  (×10⁻³)  ·  lower = better by KL",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("responses judged in-role (%)  ·  higher = better", fontsize=9,
                  color=MUTED)
    ax.set_title("KL barely predicts staying in character\n"
                 f"Spearman ρ = {rho:+.2f} (n = {n}); a good proxy would sit near −1",
                 fontsize=11.5, color=INK, loc="left", pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d9d8d4")
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(color="#ecebe7", lw=.8, zorder=0)
    ax.set_axisbelow(True)
    lo_x, hi_x = min(xs) * 1000, max(xs) * 1000
    ax.set_xlim(lo_x - 1.2, hi_x + 5.0)
    handles = [plt.Line2D([], [], marker="o", ls="", color=C[k], label=k, ms=8)
               for k in ("conditional", "per-persona")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out, f"(rho={rho:+.2f}, n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--kl", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    scratch = Path(args.scratch)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    by = load(scratch)
    print(f"{len(by)} configurations judged")
    fig1(by, out / "fig1_inrole_vs_baselines.png")
    fig2(by, json.loads(Path(args.kl).read_text()), out / "fig2_kl_vs_judge.png")


if __name__ == "__main__":
    main()
