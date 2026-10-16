"""Add the PersonaManifold arms to the layer-sweep artifact (plan 2026-10-08, 1.6).

Reads the live page (its embedded data and script), appends for the two new arms:
  metrics   distance / cosine of the steered state (layer output with the hook on, mean over the
            answer's tokens) to the target persona's average mu_B -- the same measure as the other arms
  shares    judge shares per t, with each arm's own option list (personas_arm)
  examples  up to 30 answers per route and layer, every judge label at least once
and patches the script: two new panels, per-arm judge options, "no system prompt" in the examples,
and a table for the strength check at t = 1 (alpha 0.5 / 1 / 2).

    python -m pm_addition_round.sweep_add --live LIVE.html --s1 S1 --g1 G1 --judge J --out page.html
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ace_steering.common import REP4_COLLAPSED
from ace_steering.grid import geom
from pm_addition_round.judge_batches import pm_key

N_BOOT = 1000
ARMS_NEW = ("pm_geodesic", "pm_linear")


def group_of(o, P):
    return "source" if o == P[0] else "target" if o == P[-1] else "mid" if o in P[1:-1] else o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--g1", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    html = Path(a.live).read_text()
    m = re.search(r'(<script id="data" type="application/json">)(.*?)(</script>)', html, re.S)
    D = json.loads(m.group(2))

    S1 = Path(a.s1)
    d = pd.read_csv(S1 / "generations.csv")
    X = np.load(S1 / "steer_acts.npy", mmap_mode="r")
    d["row"] = np.arange(len(d))
    lab = {}
    for line in (Path(a.judge) / "judgements.jsonl").read_text().splitlines():
        r = json.loads(line)
        lab[r["key"]] = (r["persona_kind"], r.get("intensity"), r.get("evidence") or "")
    # an aliased pm_linear cell carries its source cell's labels
    d["src_cell"] = np.where(d.aliased_from.notna(), d.cell_id.str.replace("pm_linear", "pm_geodesic"), d.cell_id)
    d["key"] = [pm_key(dict(r, cell_id=r["src_cell"])) for _, r in d.iterrows()]
    d["label"] = [("collapsed" if dr > REP4_COLLAPSED else lab.get(k, (None,))[0]) for k, dr in zip(d.key, d.deg_rep4)]

    opts = {}
    for rec in json.loads((Path(a.g1) / "summary.json").read_text()):
        for r, v in rec["routes"].items():
            opts["%s|L%d" % (r, rec["layer"])] = list(v["path_personas"])
    D["personas_arm"] = {"%s|%s" % (k, arm): v for k, v in opts.items() for arm in ARMS_NEW}

    main_cells = d.alpha == 1.0
    # ---- distance / cosine ----
    for (route, L), idx in d[main_cells].groupby(["route", "layer"]).groups.items():
        cB = geom(int(L)).route(route)[3]
        H = np.asarray(X[d.loc[idx, "row"].to_numpy()], dtype=np.float64)
        d.loc[idx, "dist"] = np.linalg.norm(H - cB, axis=1)
        d.loc[idx, "cos"] = (H @ cB) / (np.linalg.norm(H, axis=1) * np.linalg.norm(cB))
    rng = np.random.default_rng(0)
    for (route, L, arm, t), g in d[main_cells].groupby(["route", "layer", "arm", "t"]):
      for arm in ([arm, "pm_linear"] if t in (0.0, 1.0) else [arm]):     # shared endpoint cells
        D["metrics"].append(dict(route=route, layer=int(L), arm=arm, alpha=float(t), n=int(g.dist.notna().sum()),
                                 dist=round(float(g.dist.mean()), 3), dist_lo=round(float(g.dist.quantile(.25)), 3),
                                 dist_hi=round(float(g.dist.quantile(.75)), 3), cos=round(float(g.cos.mean()), 4),
                                 cos_lo=round(float(g.cos.quantile(.25)), 4), cos_hi=round(float(g.cos.quantile(.75)), 4)))
    # ---- shares (the t = 0 / 1 cells exist only under pm_geodesic: shown for both arms) ----
    dose = []
    for (route, L), dr in d.groupby(["route", "layer"]):
        P = opts["%s|L%d" % (route, L)]
        O = P + ([] if "assistant" in P else ["assistant"]) + ["other", "collapsed"]
        cells = [(arm, float(t), g) for (arm, t), g in dr[dr.alpha == 1.0].groupby(["arm", "t"])]
        cells += [("pm_linear", float(t), g) for (arm, t), g in dr[(dr.alpha == 1.0) & dr.t.isin([0.0, 1.0])].groupby(["arm", "t"])]
        for arm, t, g in cells:
            labs = g.label.fillna("unjudged").to_numpy()
            boot = rng.integers(0, len(labs), size=(N_BOOT, len(labs)))
            for o in O:
                hit = (labs == o).astype(float)
                b = hit[boot].mean(1)
                D["shares"].append(dict(route=route, layer=int(L), arm=arm, alpha=t, option=o, group=group_of(o, P),
                                        position=P.index(o) if o in P else None, n=len(labs), share=float(hit.mean()),
                                        lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5))))
        for al, g in dr[dr.t == 1.0].groupby("alpha"):
            labs = g.label.fillna("unjudged")
            dose.append(dict(route=route, layer=int(L), alpha=float(al), n=len(labs),
                             target=float((labs == P[-1]).mean()), source=float((labs == P[0]).mean()),
                             other=float(labs.isin(["other", "collapsed"]).mean())))
    D["pm_dose"] = dose
    # ---- examples ----
    for (route, L), dl in d[main_cells & d.label.notna()].groupby(["route", "layer"]):
        P = opts["%s|L%d" % (route, L)]
        pick = [g.index[rng.integers(len(g))] for _, g in dl.groupby("label")]
        rest = dl.drop(pick)
        pick += list(rest.index[rng.permutation(len(rest))[:max(0, 30 - len(pick))]])
        for j in pick:
            r = d.loc[j]
            lb, inten, ev = lab.get(r.key, (r.label, None, ""))
            t = r.response if isinstance(r.response, str) else ""
            D["examples"].append([route, int(L), r.arm, float(r.t), int(r.q_idx), -1, lb, group_of(lb, P), inten, ev,
                                  t if len(t) <= 3000 else t[:3000] + " … [cut]"])

    js = html
    reps = [
        ('.style.setProperty("--cols", ARMS.length)', '.style.setProperty("--cols", Math.min(ARMS.length, 3))'),
        ('const ARMS = ["linear", "manifold", "ace", "ace_manifold"]',
         'const ARMS = ["linear", "manifold", "ace", "ace_manifold", "pm_geodesic", "pm_linear"]'),
        ('ace_manifold: "ACE manifold", none: "Unsteered" }',
         'ace_manifold: "ACE manifold", pm_geodesic: "PersonaManifold geodesic", pm_linear: "PersonaManifold linear", none: "Unsteered" }'),
        ('ace_manifold: "h′ = h + U Uᵀ (T(α) − h)" }',
         'ace_manifold: "h′ = h + U Uᵀ (T(α) − h)", pm_geodesic: "no system prompt · h += P⁺γ(t)/|P⁺γ(t)| · s(t), γ = graph geodesic, x-axis t", pm_linear: "no system prompt · same, γ = straight line x₁→x₂, x-axis t" }'),
        ('    ARMS.forEach(a => {\n      const ar = rows.filter(r => r.arm === a);',
         '    ARMS.forEach(a => {\n      const PA = (D.personas_arm || {})[key() + "|" + a];\n      const S = PA ? series(PA) : series(P);\n'
         '      if (PA) { const sub = document.getElementById("share_" + a).previousElementSibling; sub.textContent = "judge options: " + PA.join(" → "); }\n'
         '      const ar = rows.filter(r => r.arm === a);'),
        ('<span>system prompt ${r[5] + 1}</span>', '<span>${r[5] < 0 ? "no system prompt" : "system prompt " + (r[5] + 1)}</span>'),
        ('    drawDist(); drawShares(); drawExamples();', '    drawDist(); drawShares(); drawExamples(); drawDose();'),
        ('  function render() {',
         '  function drawDose() {\n    const el = document.getElementById("pmDose"); if (!el || !D.pm_dose) return;\n'
         '    const rows = D.pm_dose.filter(r => r.route === st.route && r.layer === st.layer);\n'
         '    el.innerHTML = "<table><thead><tr><th>strength α</th><th>answers</th><th>target %</th><th>source %</th><th>other or collapsed %</th></tr></thead><tbody>" +\n'
         '      rows.map(r => `<tr><td>${r.alpha}</td><td>${r.n}</td><td>${(100 * r.target).toFixed(0)}</td><td>${(100 * r.source).toFixed(0)}</td><td>${(100 * r.other).toFixed(0)}</td></tr>`).join("") + "</tbody></table>";\n  }\n  function render() {'),
        ('<details class="panel"><summary>Table view for this route and layer</summary>',
         '<details class="panel"><summary>PersonaManifold geodesic at t = 1: strength α in eq. 6</summary><div class="tbl" id="pmDose"></div></details>\n    '
         '<details class="panel"><summary>Table view for this route and layer</summary>'),
    ]
    for old, new in reps:
        if old not in js:
            raise SystemExit("live page changed; cannot patch: %r" % old[:60])
        js = js.replace(old, new, 1)
    js = js.replace('<div class="grid2" id="distGrid">',
                    '<p class="lede">PersonaManifold arms (arXiv 2609.34571): the model gets no system prompt; the persona comes only from the added vector, so t = 0 already injects the source persona. Their judge options are the personas on their own graph path.</p>\n    <div class="grid2" id="distGrid">', 1)
    payload = json.dumps(D, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    js = js.replace(m.group(2), payload, 1)
    Path(a.out).write_text(js)
    print("page -> %s (%.1f MB), +%d metrics" % (a.out, len(js) / 1e6, int(main_cells.sum())))


if __name__ == "__main__":
    main()
