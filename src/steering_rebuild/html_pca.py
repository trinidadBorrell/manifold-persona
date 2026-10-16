"""Interactive PCA of the steering trajectories — one self-contained HTML per arm.

Same coordinates as `figures_pca.py` (PCA fitted on the role vectors, trajectories
projected in), so the static figure and these are the same picture. Plotly is
loaded from a CDN; everything else is inlined, so the file works from disk.

Hovering is the point: the static figure can show WHERE a response landed, but
not WHAT it said. Here every trajectory point carries its dose, the judge's
verdict distribution, the nearest role vectors in FULL space, and an example
response at that dose.

    python -m steering_rebuild.html_pca --acts a.npy --meta a.meta.parquet \\
        --judged judged.parquet --cloud <dir> --labels labels.parquet --outdir figures/
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from steering_rebuild.figures_pca import (load_geometry_bits, project, nearest_roles,
                                 intended_path)
# 3-D needs a third component; figures_pca's helper returns only two, so the
# basis is recomputed here rather than changing the static figure's basis.

ARM_COLOR = {"linear_axis": "#4C72B0", "manifold_axis": "#C44E52",
             "linear_target": "#4C72B0", "manifold_target": "#C44E52",
             "linear_journey": "#4C72B0", "manifold_journey": "#C44E52"}


def _wrap(t, n=90):
    words, lines, cur = str(t).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines[:8])


def basis3(g):
    """PCA basis with THREE components, fitted on the role vectors only.

    Same construction as the static figure (centre on the role vectors, SVD,
    take the leading components, project trajectories in afterwards) — just one
    component wider, so the interactive view can show the part of the geometry a
    2-D shadow has to discard.
    """
    C = np.asarray(g.centroids, dtype=np.float64)
    mu = C.mean(0)
    U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return mu, Vt[:3], var[:3]


def build(acts, meta, judged, g, mu, basis, var, arm, roles, out: Path,
          target=None):
    arms = [arm] if isinstance(arm, str) else list(arm)
    C = np.asarray(g.centroids)
    C2 = project(C, mu, basis)
    traces = []

    # --- SANITY-CHECK REFERENCES ----------------------------------------------
    # Without these the viewer has no way to tell whether a trajectory is
    # following the Assistant Axis or merely wandering: every trace is a curve on
    # screen, including the one that is a perfectly straight push in 4096-D.
    #
    #   1. `default` -- the Assistant baseline itself, the axis's own origin.
    #   2. the AXIS as a straight line through it, along +-a_hat. Anything that
    #      tracks the axis should lie near this line; anything that does not,
    #      does not.
    #   3. the fitted MANIFOLD curve S(u). Drawn beside the straight axis, it
    #      answers by eye whether "the curve is basically a line" -- the question
    #      lambda = 316 raised and that the numbers answered (38% chord
    #      deviation, 87.8 deg of tangent turn).
    from steering_core import interventions as _IV
    a_hat = _IV.unit(g.axis_unit)
    n_bar = float(g.n_bar)
    proj_c = C @ a_hat
    lo, hi = float(proj_c.min()), float(proj_c.max())
    pad = 0.15 * (hi - lo)

    i_def = g.roles.index("default") if "default" in g.roles else None
    if i_def is not None:
        d2 = C2[i_def]
        traces.append({"x": [float(d2[0])], "y": [float(d2[1])], "z": [float(d2[2])],
                       "text": ["default (the Assistant baseline)"],
                       "mode": "markers", "type": "scatter3d", "name": "Assistant (default)",
                       "hoverinfo": "text",
                       "marker": {"size": 9, "color": "#111111", "symbol": "diamond"}})
        # The axis line is anchored AT `default`, because that is what the axis is
        # defined from: a_hat = normalize(mean(default) - mean(fully role vectors)).
        origin = C[i_def] - (float(C[i_def] @ a_hat)) * a_hat
        ts = np.linspace(lo - pad, hi + pad, 60)
        L2 = project(origin[None, :] + ts[:, None] * a_hat[None, :], mu, basis)
        traces.append({"x": L2[:, 0].round(3).tolist(), "y": L2[:, 1].round(3).tolist(),
                       "z": L2[:, 2].round(3).tolist(),
                       "mode": "lines", "type": "scatter3d", "name": "Assistant Axis",
                       "hoverinfo": "skip",
                       "line": {"width": 6, "color": "#111111", "dash": "dash"}})

    us = np.linspace(float(g.spline.x[0]), float(g.spline.x[-1]), 220)
    S2 = project(g.spline.evaluate(us), mu, basis)
    traces.append({"x": S2[:, 0].round(3).tolist(), "y": S2[:, 1].round(3).tolist(),
                   "z": S2[:, 2].round(3).tolist(),
                   "mode": "lines", "type": "scatter3d", "name": "fitted manifold S(u)",
                   "hoverinfo": "skip",
                   "line": {"width": 6, "color": "#2E7D32"}})

    # the role cloud, hoverable by name
    traces.append({"x": C2[:, 0].round(3).tolist(), "y": C2[:, 1].round(3).tolist(),
                   "z": C2[:, 2].round(3).tolist(),
                   "text": list(g.roles), "mode": "markers", "type": "scatter3d",
                   "name": "role vectors", "hoverinfo": "text",
                   "marker": {"size": 2, "color": "#BBBBBB", "opacity": 0.55}})

    if target is not None and target in g.roles:
        t2 = C2[g.roles.index(target)]
        traces.append({"x": [float(t2[0])], "y": [float(t2[1])], "z": [float(t2[2])],
                       "text": ["target: %s" % target], "mode": "markers",
                       "type": "scatter3d", "name": "target: %s" % target,
                       "hoverinfo": "text",
                       "marker": {"size": 11, "color": "#F0A202", "symbol": "diamond-open",
                                  "line": {"width": 3}}})

    c_target = (np.asarray(C[g.roles.index(target)], dtype=np.float64)
                if target is not None and target in g.roles else None)

    for arm, role in [(a, r) for a in arms for r in roles]:
        base = meta[(meta.role == role) & (meta.arm == "unsteered")]
        rows = meta[(meta.role == role) & (meta.arm == arm)]
        alphas = sorted(set(rows.alpha) | ({0.0} if len(base) else set()), reverse=True)
        xs, ys, zs, hov = [], [], [], []
        for a in alphas:
            idx = base.index if (np.isclose(a, 0.0) and len(base)) else \
                rows.index[np.isclose(rows.alpha, a)]
            idx = np.asarray(idx)
            if not len(idx):
                continue
            V = acts[idx]
            V = V[np.linalg.norm(V, axis=1) > 0]
            if not len(V):
                continue
            m = V.mean(0)
            xy = project(m[None, :], mu, basis)[0]
            jj = judged.iloc[idx]
            # The axis run judged with D.1.3's perspective categories
            # (`judge_score`); the role-to-role run asks a different question and
            # records WHICH ROLE the text reads as (`persona`). Same hover slot,
            # so take whichever column this run actually has.
            vcol = "judge_score" if "judge_score" in jj.columns else "persona"
            vc = jj[vcol].value_counts(normalize=True)
            verdicts = ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in vc.head(3).items())
            ex = jj[jj.response.notna()]
            ex = ex.iloc[0].response if len(ex) else ""
            xs.append(round(float(xy[0]), 3)); ys.append(round(float(xy[1]), 3))
            zs.append(round(float(xy[2]), 3))
            # `%` binds tighter than `+`, so the label has to be spliced into
            # the format string BEFORE the substitution, not beside it.
            lbl = "f (fraction of the journey)" if "journey" in arm else "alpha"
            fmt = ("<b>%s</b> &nbsp; " + lbl + " = %+.3f<br>judge: %s<br>"
                   "nearest roles: %s<br><i>%s</i>")
            hov.append(fmt
                       % (html.escape("%s / %s" % (role, arm)), a,
                          html.escape(verdicts),
                          html.escape(", ".join(nearest_roles(m, C, g.roles, 4))),
                          _wrap(html.escape(str(ex)[:600]))))
        # THE INTENDED PATH, for contrast. This is h + alpha*N_bar*a_hat computed
        # from the role's own unsteered mean -- i.e. exactly where the push puts
        # the activation, before the model turns it into words. It is straight by
        # construction. The measured trace beside it is not, and the gap between
        # them is the whole point: pushing the state by X does not move the
        # LANGUAGE by X, because tokens are discrete and a slightly larger push
        # rewrites the sentence rather than nudging it.
        if len(base):
            h0 = acts[np.asarray(base.index)]
            h0 = h0[np.linalg.norm(h0, axis=1) > 0]
            if len(h0):
                h0 = h0.mean(0).astype(np.float64)
                al = np.array(sorted(set(rows.alpha) | {0.0}), dtype=float)
                P = project(intended_path(arm, h0, al, g, c_target), mu, basis)
                traces.append({
                    "x": P[:, 0].round(3).tolist(), "y": P[:, 1].round(3).tolist(),
                    "z": P[:, 2].round(3).tolist(),
                    "text": ["%s / %s intended, %s %.3f" % (role, arm,
                             "f" if "journey" in arm else "alpha", a) for a in al],
                    "mode": "lines+markers", "type": "scatter3d",
                    # The ARM has to be in the name: with both arms in one file
                    # the legend showed two entries both called
                    # "doctor (intended push)" and neither could be identified.
                    "name": "%s / %s (intended push)" % (role, arm),
                    "hoverinfo": "text",
                    "visible": "legendonly",
                    "line": {"width": 3, "color": ARM_COLOR.get(arm, "#333"), "dash": "dot"},
                    "marker": {"size": 3, "color": ARM_COLOR.get(arm, "#333"),
                               "symbol": "diamond"}})

        traces.append({"x": xs, "y": ys, "z": zs, "text": hov,
                       "mode": "lines+markers", "type": "scatter3d",
                       "name": "%s / %s" % (role, arm), "hoverinfo": "text",
                       "line": {"width": 5, "color": ARM_COLOR.get(arm, "#333"),
                                "dash": "dash" if "manifold" in arm else "solid"},
                       "marker": {"size": 4, "color": ARM_COLOR.get(arm, "#333")}})

    if target is not None:
        title = ("Journey steering toward %s — where each persona lands "
                 "(Qwen3-8B, layer 19)" % target)
    else:
        title = ("%s steering — where each persona lands (Qwen3-8B, layer 19)"
                 % ("Linear" if "linear" in arms[0] else "Manifold"))
    sub = ("PCA fitted on the 275 role vectors — PC1 %.1f%%, PC2 %.1f%%, PC3 %.1f%% "
           "(%.1f%% together). Drag to rotate, scroll to zoom. Hover a point for its dose, "
           "the judge's verdicts and an example response. Black dashed = the Assistant "
           "Axis, black diamond = the `default` Assistant baseline, green = the fitted "
           "manifold. A trajectory that follows the axis should hug the dashed line. "
           "Turn on an `intended push` trace in the legend to see where the "
           "intervention AIMED, against where the words actually landed."
           % (100 * var[0], 100 * var[1], 100 * var[2], 100 * var[:3].sum()))
    doc = """<!doctype html><meta charset="utf-8"><title>%s</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>body{font:14px system-ui,sans-serif;margin:0;background:#fff;color:#111}
h1{font-size:18px;margin:14px 18px 2px}p{margin:0 18px 10px;color:#555;font-size:12.5px}
#p{width:100vw;height:calc(100vh - 78px)}</style>
<h1>%s</h1><p>%s</p><div id="p"></div><script>
Plotly.newPlot('p', %s, {hovermode:'closest',
  scene:{xaxis:{title:'PC1'}, yaxis:{title:'PC2'}, zaxis:{title:'PC3'},
         aspectmode:'data',
         camera:{eye:{x:1.6, y:1.5, z:0.9}}},
  legend:{orientation:'h'}, margin:{l:0,r:0,t:0,b:0}},
  {responsive:true, displaylogo:false});
</script>""" % (html.escape(title), html.escape(title), html.escape(sub),
                json.dumps(traces))
    out.write_text(doc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--judged", required=True); ap.add_argument("--cloud", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--arms", default="linear_axis,manifold_axis")
    ap.add_argument("--target", default=None,
                    help="keep only rows steered toward this role, mark it, and put "
                         "every arm in ONE file so the routes can be compared")
    a = ap.parse_args()

    acts = np.load(a.acts); meta = pd.read_parquet(a.meta)
    judged = pd.read_parquet(a.judged).reset_index(drop=True)
    if len(judged) != len(meta):
        raise SystemExit("judged (%d) and acts meta (%d) are different lengths — they "
                         "must be row-aligned" % (len(judged), len(meta)))
    roles = [r.strip() for r in a.roles.split(",") if r.strip()]
    g, _, _, _ = load_geometry_bits(a.cloud, a.labels)
    mu, basis, var = basis3(g)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    arms = [x.strip() for x in a.arms.split(",")]
    if a.target:
        if "target_role" not in meta.columns:
            raise SystemExit("--target given but meta has no target_role column")
        # Filter WITHOUT reindexing: meta's index indexes `acts` and `judged`.
        meta = meta[(meta.target_role == a.target) | (meta.arm == "unsteered")]
        # Name the file after the ARM FAMILY, not the literal string "journey":
        # the contrast run writes the same targets and would otherwise overwrite
        # the journey files in place.
        fam = "journey" if any("journey" in x for x in arms) else \
              "contrast" if any("contrast" in x for x in arms) else "arms"
        p = build(acts, meta, judged, g, mu, basis, var, arms, roles,
                  out / ("interactive_%s_to_%s_L19.html" % (fam, a.target)),
                  target=a.target)
        print("wrote", p)
        return 0
    for arm in arms:
        p = build(acts, meta, judged, g, mu, basis, var, arm, roles,
                  out / ("interactive_%s_L19.html" % arm))
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
