"""The layer-sweep page with a question filter: every figure and the examples can be restricted to
one of the five questions (or all five pooled).

Same measures as sweep_page.py. Per question a point has 5 answers (one per system prompt), so the
judge shares move in 20% steps and the bootstrap bands are wide.

Compact data (keeps the page under the size limit):
  metrics  [route, layer, q, arm, alpha, dist, dist_lo, dist_hi, cos, cos_lo, cos_hi, n]   q = -1: all five
  shares   [route, layer, q, arm, alpha, n, {option: [share, lo, hi]}]                       non-zero options only
  examples [route, layer, arm, alpha, q, prompt, label, group, intensity, evidence, text]    20 per (route, layer, question)

    python -m ace_steering.sweep_q_page
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ace_steering.common import OUT_ROOT
from ace_steering.grid import geom, route_personas
from ace_steering.judge import key_of
from ace_steering.page import _no_nan
from ace_steering.sweep_page import FWD, group, labels, rows_and_acts

HERE = Path(__file__).resolve().parent
PER_QUESTION = 20
N_BOOT = 1000


def fill_template(data, title, layers, layer_label="Layer",
                  judge_note="Manifold steering is only defined from 0 (source) to 1 (target).",
                  ex_note="20 answers per question for the route and layer selected above (100 for all five)"):
    tpl = (HERE / "sweep_q_template.html").read_text()
    for k, v in (("__TITLE__", title), ("__LAYERS__", layers), ("__LAYERLABEL__", layer_label),
                 ("__JUDGENOTE__", judge_note), ("__EXNOTE__", ex_note)):
        tpl = tpl.replace(k, v)
    return tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def main():
    root = OUT_ROOT
    d = rows_and_acts(root)
    lab = labels(root)
    d["key"] = [key_of(r) for _, r in d.iterrows()]
    d["label"] = [lab.get(k, (None,))[0] for k in d.key]
    d["dist"] = np.nan
    d["cos"] = np.nan
    for (route, layer), idx in d.groupby(["route", "layer"]).groups.items():
        cB = geom(int(layer)).route(route)[3]
        H = np.stack(d.loc[idx, "h"].to_numpy())
        d.loc[idx, "dist"] = np.linalg.norm(H - cB, axis=1)
        d.loc[idx, "cos"] = (H @ cB) / (np.linalg.norm(H, axis=1) * np.linalg.norm(cB))
    d = d.drop(columns=["h"])
    personas = {"%s|L%d" % (r, L): route_personas(r, int(L)) for r, L in {(r, int(L)) for r, L in zip(d.route, d.layer)}}

    rng = np.random.default_rng(0)
    metrics, shares = [], []
    for q in [-1, 0, 1, 2, 3, 4]:
        dq = d if q < 0 else d[d.q_idx == q]
        for (route, layer, arm, al), g in dq.groupby(["route", "layer", "arm", "alpha"]):
            arms = ["linear", "manifold"] if arm == "none" else [arm]
            dist, cos = g.dist.dropna(), g.cos.dropna()
            m = [r4(dist.mean()), r4(dist.quantile(.25)), r4(dist.quantile(.75)),
                 r4(cos.mean()), r4(cos.quantile(.25)), r4(cos.quantile(.75)), int(len(dist))]
            lab_arr = g.label.fillna("unjudged").to_numpy()
            n = len(lab_arr)
            boot = rng.integers(0, n, size=(N_BOOT, n))
            opt = {}
            for o in np.unique(lab_arr):
                hit = (lab_arr == o).astype(float)
                b = hit[boot].mean(1)
                opt[str(o)] = [r4(hit.mean()), r4(np.percentile(b, 2.5)), r4(np.percentile(b, 97.5))]
            for a in arms:
                metrics.append([route, int(layer), q, a, float(al)] + m)
                shares.append([route, int(layer), q, a, float(al), n, opt])

    ex = []
    for (route, layer, q), g in d[d.label.notna()].groupby(["route", "layer", "q_idx"]):
        pick = [grp.index[rng.integers(len(grp))] for _, grp in g.groupby("label")]   # every label once
        rest = g.drop(pick)
        by_arm = {a: list(rng.permutation(ix)) for a, ix in rest.groupby("arm").groups.items()}
        arms = sorted(by_arm)
        k = 0
        while len(pick) < PER_QUESTION and any(by_arm.values()):
            a = arms[k % len(arms)]
            k += 1
            if by_arm[a]:
                pick.append(by_arm[a].pop())
        P = personas["%s|L%d" % (route, int(layer))]
        for j in pick:
            r = d.loc[j]
            lb, inten, ev = lab.get(r.key, (None, None, ""))
            t = r.response if isinstance(r.response, str) else ""
            ex.append([route, int(layer), r.arm, float(r.alpha), int(q), int(r.prompt_idx), lb, group(P, lb), inten, ev,
                       t if len(t) <= 3000 else t[:3000] + " … [cut; %d characters in full]" % len(t)])

    data = _no_nan(dict(metrics=metrics, shares=shares, personas=personas, examples=ex,
                        routes=FWD, layers=sorted(int(x) for x in d.layer.unique())))
    out = root / "step2_subagent" / "layer_sweep_by_question.html"
    out.write_text(fill_template(data, title="Manifold and Linear Steering: Layer Sweep by Question",
                                 layers="hidden_states 15–32"))
    print("metrics %d shares %d examples %d -> %s (%.1f MB)" % (len(metrics), len(shares), len(ex), out, out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
