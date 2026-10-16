"""PNG figures of a paper-measures page (Findings 1-3 and the Table 3 layout), from the page's own data.

    python -m steering.personamanifold.pm_measures_figs --page page.html --out figs/ [--label S]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PGA, PCA, PLUS, MINUS, START, INK = "#0f7b74", "#6b7f99", "#c8562a", "#2f6db5", "#5b6b6d", "#142021"
NAME = {"pga": "Fletcher direction", "pca": "PCA component"}


def load(page):
    s = Path(page).read_text()
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', s, re.S).group(1).replace("<\\/", "</"))


def grid(D, a, key, rows, ylabel, ylim, title, out, deg=False):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, d in zip(axs.flat, D["dirs"]):
        col = PGA if d["method"] == "pga" else PCA
        R = [r for r in rows if r["method"] == d["method"] and r["component"] == d["component"] and (r.get("alpha") in (None, a))]
        for j, s in enumerate(D["starts"]):
            rs = sorted([r for r in R if r["start"] == s], key=lambda r: r["x"])
            ax.plot([r["x"] for r in rs], [r[key] for r in rs], color=START, lw=0.9, ls=[":", "--", "-."][j % 3], alpha=0.8, label=s)
        rp = sorted([r for r in R if r["start"] == "__all__"], key=lambda r: r["x"])
        lo = [r.get(key + "_lo", r.get("lo")) for r in rp]
        hi = [r.get(key + "_hi", r.get("hi")) for r in rp]
        if lo and lo[0] is not None:
            ax.fill_between([r["x"] for r in rp], lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot([r["x"] for r in rp], [r[key] for r in rp], color=col, lw=2.6, marker="o", ms=4, label="all starts")
        if deg:
            ax2 = ax.twinx()
            ax2.plot([r["x"] for r in rp], [100 * r["degenerate"] for r in rp], color=PLUS, ls="--", lw=1.4)
            ax2.set_ylim(0, 105)
            ax2.tick_params(colors=PLUS, labelsize=8)
            if d is D["dirs"][2] or d is D["dirs"][5]:
                ax2.set_ylabel("judged degenerate (%)", color=PLUS, fontsize=9)
        ax.set_title("%s %d" % (NAME[d["method"]], d["component"]), fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_xlim(-1.1, 1.1)
        ax.grid(alpha=0.25)
        ax.set_xlabel("step (− towards − end · + towards + end)", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    axs.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="α")
    a = ap.parse_args()
    D, out = load(a.page), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = a.label
    for S in D["order"]:
        ic = [r for r in D["ic"] if r["alpha"] == S]
        grid(D, S, "ic", ic, "IC: mean pairwise cosine", (0.15, 0.6),
             "Finding 1 · interpolation coherence, all 20 answers of a step (%s = %s); dashed: judged degenerate" % (L, S),
             out / ("finding1_ic_all_%s%s.png" % (L, S)), deg=True)
        grid(D, S, "icq", ic, "IC within question", (0.3, 0.85),
             "Finding 1 · interpolation coherence, answers to the same question (%s = %s); dashed: judged degenerate" % (L, S),
             out / ("finding1_ic_within_question_%s%s.png" % (L, S)), deg=True)
        grid(D, S, "adh", D["ma"], "1 − distance to nearest persona / r95", (-1.2 if any(r.get("adh", 1) < 0.3 for r in D["ma"]) else 0.7, 1.02),
             "Finding 3 · manifold adherence per step (%s = %s)" % (L, S), out / ("finding3_ma_%s%s.png" % (L, S)))
    # Finding 2: TS bars, one panel per strength
    fig, axs = plt.subplots(1, len(D["order"]), figsize=(13, 4.2), sharey=True)
    for ax, S in zip(axs, D["order"]):
        names = ["%s %d" % (NAME[d["method"]], d["component"]) for d in D["dirs"]]
        for h, (off, col, lab) in enumerate([(-0.2, MINUS, "0 → − end"), (0.2, PLUS, "0 → + end")]):
            half = -1 if h == 0 else 1
            rs = [next(r for r in D["ts"] if r["alpha"] == S and r["start"] == "__all__" and r["method"] == d["method"]
                       and r["component"] == d["component"] and r["half"] == half) for d in D["dirs"]]
            xs = [i + off for i in range(len(rs))]
            ys = [100 * r["ts"] for r in rs]
            err = [[100 * max(r["ts"] - r["lo"], 0) for r in rs], [100 * max(r["hi"] - r["ts"], 0) for r in rs]]
            ax.bar(xs, ys, 0.38, color=col, label=lab, yerr=err, error_kw=dict(ecolor="#7b8a8c", lw=1))
        ax.axhline(50, color="#7b8a8c", ls=":", lw=1.2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" direction", "").replace(" component", "") for n in names], fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_title("%s = %s" % (L, S))
        ax.grid(axis="y", alpha=0.25)
    axs[0].set_ylabel("TS: steps moving towards the end value (%)")
    axs[0].legend(fontsize=8)
    fig.suptitle("Finding 2 · transition smoothness per half path (bars: 95% bootstrap; dotted: 50%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "finding2_ts.png", dpi=150)
    plt.close(fig)
    # Table 3 layout
    rows = []
    for m in ("pca", "pga"):
        for k in (1, 2, 3):
            d = next(x for x in D["dirs"] if x["method"] == m and x["component"] == k)
            row = ["%s %d" % (NAME[m], k)]
            for S in D["order"]:
                t = [r for r in D["ts"] if r["alpha"] == S and r["method"] == m and r["component"] == k and r["start"] == "__all__"]
                mean = lambda f: sum(r[f] for r in t) / len(t)            # noqa: E731
                ma = d["ma"].get(S) or d["ma"].get("all") or d["ma"]
                row += ["%.2f" % mean("ic_path"), "%.2f" % mean("icq_path"), "%.2f" % mean("ts"), "%.2f" % ((ma["-1"]["pooled"] + ma["1"]["pooled"]) / 2)]
            rows.append(row)
    hdr = ["Direction"] + ["%s %s=%s" % (c, L, S) for S in D["order"] for c in ("IC", "IC within q.", "TS", "MA")]
    (out / "table3.csv").write_text("\n".join(",".join(r) for r in [hdr] + rows) + "\n")
    fig, ax = plt.subplots(figsize=(14, 2.8))
    ax.axis("off")
    tb = ax.table(cellText=rows + [["Paper, Qwen2.5-7B: Geodesic", "0.69", "–", "0.65", "0.58"] + [""] * 4,
                                   ["Paper, Qwen2.5-7B: Euclidean-Linear", "0.55", "–", "0.49", "0.38"] + [""] * 4],
                  colLabels=hdr, loc="center", cellLoc="center", colWidths=[0.2] + [0.1] * (len(hdr) - 1))
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.4)
    fig.suptitle("As in the paper's Table 3 (mean over both halves and the three starts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "table3.png", dpi=150)
    plt.close(fig)
    print("figures ->", out, sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
