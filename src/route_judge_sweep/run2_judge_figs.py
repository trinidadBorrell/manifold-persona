"""Run 2 (qwen3_route.py, judged in $PM/JSG_L{L}_S{S}): one figure per judge, one panel per route,
share of answers per label group along s, the four pairs pooled -> output/11-oct/run2_L{L}_S{S}_{judge}.png

    python -m route_judge_sweep.run2_judge_figs --layer 15 --strength 4
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from giovanni_repro.qwen3_pin import PM  # noqa: E402

O = Path(__file__).resolve().parents[2] / "output" / "11-oct"
ROUTES = ["LIN", "PMG", "MAN4", "FL"]
JUDGES = {"J1": "original control", "J2": "revised control", "J3": "Assistant Axis, original", "J4": "Assistant Axis, revised",
          "J6": "open vocabulary", "J7": "minimal, 275 names", "J8": "revised control, 275 roles", "J10": "Ranking Judge"}
GROUPS = {   # label group -> colour; persona judges
    "target": "#c8562a", "persona on the route": "#2f6db5", "assistant": "#0f7b74", "other persona": "#9a9a9a",
    "other / none": "#c9b458", "collapsed": "#7a1f1f"}
AA = {"assistant": "#0f7b74", "nonhuman_role": "#c8562a", "human_role": "#e39a5a", "weird_role": "#8a6fc9",
      "ambiguous": "#9a9a9a", "other": "#c9b458", "nonsensical": "#7a1f1f"}
AA_NAME = {"nonhuman_role": "non-human role", "human_role": "human role", "weird_role": "weird role"}


def group(label, target, onroute, contains=False):
    lab = str(label).strip().lower().replace(" ", "_")
    if lab == target or (contains and target in lab.split("_")):
        return "target"
    if lab in ("assistant", "default"):
        return "assistant"
    if lab in onroute:
        return "persona on the route"
    if lab in ("collapsed", "non_sensical", "nonsensical"):
        return "collapsed"
    if lab in ("other", "none", "not_stated", "nan", ""):
        return "other / none"
    return "other persona"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--strength", default="4")
    a = ap.parse_args()
    D = PM / ("JSG_L%d_S%s" % (a.layer, a.strength))
    ans = pd.read_csv(D / "answers.csv", low_memory=False)
    ans = ans[ans.route != "ref"][["key", "pair", "route", "s", "target", "collapsed"]]
    spec = json.loads((PM / "ST15" / "cells.json").read_text())
    onroute = {}
    for pair in ans.pair.unique():
        rt = spec["pairs"]["%s|L15" % pair]["routes"]
        rt = rt if isinstance(rt, dict) else ast.literal_eval(rt)
        for r in ROUTES:
            onroute[(pair, r)] = set(rt[r][1:-1])
    lab = pd.read_csv(D / "labels.csv", low_memory=False)[["judge", "key", "label"]]
    j10 = pd.read_csv(D / "J10" / "labels.csv")
    j10 = pd.DataFrame(dict(judge="J10", key=j10.key, label=[json.loads(r)[0] for r in j10["rank"]],
                            r5=[r for r in j10["rank"]], nonsense=j10.nonsense))
    lab = pd.concat([lab, j10], ignore_index=True)
    O.mkdir(parents=True, exist_ok=True)
    xs = sorted(ans.s.unique())
    for J, desc in JUDGES.items():
        dj = lab[lab.judge == J].merge(ans, on="key")
        if dj.empty:
            print("no labels for", J)
            continue
        aa = J in ("J3", "J4")
        dj["g"] = dj.label.astype(str) if aa else [group(l, t, onroute[(p, r)], J == "J6") for l, t, p, r in zip(dj.label, dj.target, dj.pair, dj.route)]
        cols = AA if aa else GROUPS
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
        for ax, r in zip(axs, ROUTES):
            d = dj[dj.route == r]
            sh = d.groupby("s").g.value_counts(normalize=True).unstack(fill_value=0).reindex(xs, fill_value=0)
            for g, c in cols.items():
                if g in sh:
                    ax.plot(xs, 100 * sh[g], "-o", color=c, ms=4, lw=2, label=AA_NAME.get(g, g))
            if J == "J10":   # the share with the target among its first five names
                r5 = d.groupby("s").apply(lambda x: pd.Series([t in json.loads(k)[:5] for t, k in zip(x.target, x.r5)]).mean())
                ax.plot(xs, 100 * r5.reindex(xs), "--", color=cols["target"], lw=1.6, label="target in its top 5")
            ax.set_title(r, fontsize=11)
            ax.set_xlabel("s (position on the route)")
            ax.set_xticks(xs)
            ax.set_ylim(-3, 103)
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("% of answers")
        hl = {}
        for ax in axs:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                hl.setdefault(ll, hh)
        order = [AA_NAME.get(g, g) for g in cols] + ["target in its top 5"]
        l = [x for x in order if x in hl]
        h = [hl[x] for x in l]
        fig.legend(h, l, loc="lower center", ncol=min(len(l), 8), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))
        first = {"J10": " (first-ranked name)", "J6": " (a name containing the target's name counts as the target)"}.get(J, "")
        fig.suptitle("%s · %s%s · Qwen3-8B layer %d, S = %s · 4 pairs, 40 answers per point"
                     % (J, desc, first, a.layer, a.strength), fontsize=11)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        fn = O / ("run2_L%d_S%s_%s.png" % (a.layer, a.strength, J))
        fig.savefig(fn, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(fn.name, len(dj))


if __name__ == "__main__":
    main()
