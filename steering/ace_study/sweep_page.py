"""Layer-sweep page: distance and cosine to the target, judge shares, and 100 example answers per route and layer.

Forward routes only, subagent judge. The steered state is the layer's output with the hook on,
averaged over the answer's tokens (steer_acts.npy). Reference for both measures: the target
persona's centroid mu_B at that layer.

    python -m steering.ace_study.sweep_page
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from steering.ace_study.common import OUT_ROOT, STEP1_LAYER, STEP1_ROUTES
from steering.ace_study.grid import geom, route_personas
from steering.ace_study.judge import key_of

HERE = Path(__file__).resolve().parent
FWD = ["assistant>bard", "assistant>vampire", "assistant>leviathan",
       "grader>tree", "examiner>criminal", "scheduler>void"]
PER_LAYER = 100


def rows_and_acts(root):
    """Forward-route answers of step 2, plus step 1's sample 0 for its routes at layer 19."""
    frames = []
    for sub, extra in (("step2", None), ("step1", "s0")):
        d = pd.read_csv(root / sub / "generations.csv")
        X = np.load(root / sub / "steer_acts.npy", mmap_mode="r")
        d["row"] = np.arange(len(d))
        m = d.route.isin(FWD)
        if extra:
            m &= (d["sample"] == 0) & (d.layer == STEP1_LAYER) & d.route.isin(STEP1_ROUTES)
        m &= d.arm != "ace_manifold"                     # removed from all results (user, 2026-10-02)
        d = d[m].copy()
        d["h"] = list(np.asarray(X[d.row.to_numpy()], dtype=np.float64))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def labels(root):
    lab = {}
    for f in ("judgements_step1_subagent.jsonl", "judgements_step2_subagent.jsonl"):
        for line in (root / "judge" / f).read_text().splitlines():
            d = json.loads(line)
            lab[d["key"]] = (d.get("persona_kind"), d.get("intensity"), d.get("evidence") or "")
    from steering.ace_study.relabel_ai import apply
    return apply(lab)


def group(personas, lab):
    if lab == personas[0]:
        return "source"
    if lab == personas[-1]:
        return "target"
    if lab in personas:
        return "mid"
    return lab or "unjudged"


def main():
    root = OUT_ROOT
    d = rows_and_acts(root)
    lab = labels(root)
    d["key"] = [key_of(r) for _, r in d.iterrows()]
    d["label"] = [lab.get(k, (None,))[0] for k in d.key]
    print("answers", len(d), "unlabelled", int(d.label.isna().sum()))

    # ---- distance and cosine to the target centroid, per answer ----
    cos, dist = [], []
    for (route, layer), idx in d.groupby(["route", "layer"]).groups.items():
        cB = geom(int(layer)).route(route)[3]
        H = np.stack(d.loc[idx, "h"].to_numpy())
        ok = np.isfinite(H).all(1)
        dd = np.full(len(H), np.nan)
        cc = np.full(len(H), np.nan)
        dd[ok] = np.linalg.norm(H[ok] - cB, axis=1)
        cc[ok] = (H[ok] @ cB) / (np.linalg.norm(H[ok], axis=1) * np.linalg.norm(cB))
        d.loc[idx, "dist"], d.loc[idx, "cos"] = dd, cc
    metrics = []
    for (route, layer, arm, al), g in d.groupby(["route", "layer", "arm", "alpha"]):
        arms = ["linear", "manifold"] if arm == "none" else [arm]   # unsteered = both at alpha 0
        for a in arms:
            metrics.append(dict(route=route, layer=int(layer), arm=a, alpha=float(al), n=int(g.dist.notna().sum()),
                                dist=round(float(g.dist.mean()), 3),
                                dist_lo=round(float(g.dist.quantile(0.25)), 3), dist_hi=round(float(g.dist.quantile(0.75)), 3),
                                cos=round(float(g.cos.mean()), 4),
                                cos_lo=round(float(g.cos.quantile(0.25)), 4), cos_hi=round(float(g.cos.quantile(0.75)), 4)))

    # ---- judge shares: the same table the layer-sweep page uses ----
    from steering.ace_study.readout import shares as share_table
    opts = {"%s|L%d" % (r, L): route_personas(r, int(L)) for r, L in {(r, int(L)) for r, L in zip(d.route, d.layer)}}
    sh = share_table(d, opts)
    shares = sh.round(4).astype(object).where(sh.notna(), None).to_dict(orient="records")
    personas = {"%s|L%d" % (r, L): route_personas(r, int(L)) for r, L in {(r, int(L)) for r, L in zip(d.route, d.layer)}}

    # ---- 100 examples per layer, every (route, judge label) present at least once ----
    rng = np.random.default_rng(0)
    ex = []
    for (route, layer), dl in d.groupby(["route", "layer"]):
        dl = dl[dl.label.notna()]
        pick = []
        for _, g in dl.groupby("label"):                      # one of every label on this route and layer
            pick.append(g.index[rng.integers(len(g))])
        rest = dl.drop(pick)
        need = max(0, PER_LAYER - len(pick))
        if need:                                              # fill evenly over arms
            per_arm = rest.groupby("arm").groups
            arms = sorted(per_arm)
            k = 0
            while need and any(len(per_arm[a]) for a in arms):
                a = arms[k % len(arms)]
                k += 1
                if len(per_arm[a]):
                    j = per_arm[a][rng.integers(len(per_arm[a]))]
                    per_arm[a] = per_arm[a].drop(j)
                    pick.append(j)
                    need -= 1
        for j in pick:
            r = d.loc[j]
            P = personas["%s|L%d" % (r.route, int(r.layer))]
            lb, inten, ev = lab.get(r.key, (None, None, ""))
            t = r.response if isinstance(r.response, str) else ""
            ex.append([r.route, int(r.layer), r.arm, float(r.alpha), int(r.q_idx), int(r.prompt_idx),
                       lb, group(P, lb), inten, ev, t if len(t) <= 3000 else t[:3000] + " … [cut; %d characters in full]" % len(t)])
        print("%s L%d: %d examples, %d labels" % (route, layer, len(pick), dl.label.nunique()))

    from steering.ace_study.page import _no_nan
    data = dict(metrics=metrics, shares=shares, personas=personas, examples=ex,
                routes=FWD, layers=sorted(int(x) for x in d.layer.unique()))
    data = _no_nan(data)
    tpl = (HERE / "sweep_template.html").read_text()
    out = root / "step2_subagent" / "layer_sweep.html"
    out.write_text(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")))
    print("page -> %s (%.1f MB)" % (out, out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
