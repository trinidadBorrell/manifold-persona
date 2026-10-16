"""Summary tables for the results report (output/behavior-latent/RESULTS.md).

Prints and saves output/behavior-latent/followups/*.csv:
  A   % target by source (assistant / validator / summarizer) x target x arm, L19 and L25, alpha 1 and 1.5
      (assistant rows: step 2, subagent judge)
  B   % of answers labelled the destination (assistant / grader / examiner / scheduler) on the reverse routes
  D   % target and % collapsed per floor and percentile p; mean distance to mu_B at each capped layer
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ace_steering.common import OUT_ROOT, STEP3_ROUTES, STEP5_LAYERS
from ace_steering.grid import route_personas
from ace_steering.judge import key_of

OUT = OUT_ROOT.parent / "behavior-latent" / "followups"
FWD = ["assistant>bard", "assistant>vampire", "assistant>leviathan"]
B_ROUTES = ["bard>assistant", "vampire>assistant", "leviathan>assistant", "tree>grader", "criminal>examiner", "void>scheduler"]


def labels(*tags):
    lab = {}
    for t in tags:
        for line in (OUT_ROOT / "judge" / ("judgements_%s_subagent.jsonl" % t)).read_text().splitlines():
            d = json.loads(line)
            lab[d["key"]] = d.get("persona_kind")
    from ace_steering.relabel_ai import overlay
    for k in overlay():
        if lab.get(k) == "other":
            lab[k] = "assistant"
    return lab


def frame(gen, routes, layers, lab, layer_for_options=None):
    d = pd.read_csv(OUT_ROOT / gen / "generations.csv",
                    usecols=["cell_id", "prompt_idx", "q_idx", "sample", "route", "layer", "arm", "alpha"])
    d = d[d.route.isin(routes) & d.layer.isin(layers)].copy()
    d["label"] = [lab.get(key_of(r)) for _, r in d.iterrows()]
    L_opt = (lambda L: layer_for_options) if layer_for_options else (lambda L: int(L))
    d["target"] = [l == route_personas(r, L_opt(L))[-1] for r, L, l in zip(d.route, d.layer, d.label)]
    d["collapsed"] = d.label == "collapsed"
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    # ---- A: source x target ----
    a = frame("step3", STEP3_ROUTES, [19, 25], labels("A"))
    s2 = frame("step2", FWD, [25], labels("step2"))
    s1 = frame("step1", FWD, [19], labels("step1"))
    s1 = s1[s1["sample"] == 0]
    d = pd.concat([a, s1, s2], ignore_index=True)
    d["source"], d["dest"] = zip(*[r.split(">") for r in d.route])
    d = d[d.arm != "none"]
    A = (d[d.alpha.isin([1.0, 1.5])].groupby(["dest", "source", "layer", "arm", "alpha"]).target.mean() * 100).round(0)
    A.to_csv(OUT / "A_target_by_source.csv")
    best = A.groupby(["dest", "source", "layer"]).max().unstack(["layer", "source"])
    print("\nA: best % target over arms and alpha in {1, 1.5}\n", best.to_string())
    best.to_csv(OUT / "A_best_target.csv")
    coll = (d[d.alpha == 1.5].groupby(["source", "layer", "arm"]).collapsed.mean() * 100).round(0).unstack("arm")
    print("\nA: % collapsed at alpha 1.5\n", coll.to_string())

    # ---- B: reverse routes ----
    b = frame("step2", B_ROUTES, [19, 25], labels("B"))
    b = b[b.arm != "none"]
    B = (b[b.alpha.isin([1.0, 1.5])].groupby(["route", "layer", "arm", "alpha"]).target.mean() * 100).round(0)
    B.to_csv(OUT / "B_target.csv")
    print("\nB: best % reaching the destination over arms and alpha in {1, 1.5}\n",
          B.groupby(["route", "layer"]).max().unstack("layer").to_string())
    base = b.copy()
    base0 = frame("step2", B_ROUTES, [19, 25], labels("B"))
    u = (base0[base0.arm == "none"].groupby(["route", "layer"]).target.mean() * 100).round(0).unstack("layer")
    print("\nB: unsteered % already labelled the destination\n", u.to_string())

    # ---- D: capping floors ----
    fwd6 = FWD + ["grader>tree", "examiner>criminal", "scheduler>void"]
    dd = frame("step5", fwd6, [19], labels("D"), layer_for_options=19)
    D = dd.groupby(["route", "arm", "alpha"]).agg(target=("target", "mean"), collapsed=("collapsed", "mean")).mul(100).round(0)
    D.to_csv(OUT / "D_target_collapsed.csv")
    print("\nD: % target / % collapsed by floor and percentile p (alpha column = p)\n",
          D.unstack(["arm"]).to_string())
    X = np.load(OUT_ROOT / "step5" / "steer_acts.npy", mmap_mode="r")
    g = pd.read_csv(OUT_ROOT / "step5" / "generations.csv", usecols=["route", "arm", "alpha"])
    from ace_steering.grid import geom
    rows = []
    for j, L in enumerate(STEP5_LAYERS):
        for (route, arm, p), idx in g.groupby(["route", "arm", "alpha"]).groups.items():
            cB = geom(L).route(route)[3]
            H = np.asarray(X[np.asarray(idx), j], dtype=np.float64)
            rows.append(dict(route=route, layer=L, arm=arm, p=p, dist=float(np.nanmean(np.linalg.norm(H - cB, axis=1)))))
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT / "D_distance.csv", index=False)
    print("\nD: mean distance to mu_B (mean over routes)\n",
          dist.groupby(["layer", "arm", "p"]).dist.mean().round(1).unstack(["arm", "p"]).to_string())


if __name__ == "__main__":
    main()
