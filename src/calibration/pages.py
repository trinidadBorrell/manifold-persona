"""Page data + HTML for the three artifacts of plan 2026-10-09.

    python -m calibration.pages C  --dir CAL --notes notes_C.json  --out what_alpha_means.html
    python -m calibration.pages E1 --dir ST  --notes notes_E1.json --cal CAL --out steering_stability.html
    python -m calibration.pages E2 --dir DIR --notes notes_E2.json --cal CAL --out principal_direction.html

Each reads DIR/rows_measured.csv (calibration.analyse) and DIR/cells.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibration.analyse import boot, classify

HERE = Path(__file__).resolve().parent
ROUTE_ORDER = ["LIN", "PMG", "MAN3", "MAN4", "MAN5", "FL", "FG", "PCA"]


def clean(o):
    if isinstance(o, float):
        return None if not np.isfinite(o) else round(o, 4)
    if isinstance(o, (np.floating,)):
        return clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def write(tpl, data, out):
    payload = json.dumps(clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    Path(out).write_text((HERE / tpl).read_text().replace("__DATA__", payload))
    print("page -> %s (%.2f MB)" % (out, Path(out).stat().st_size / 1e6))


def rows(d):
    if "," in str(d):                               # one experiment run per layer (ST19,ST15): merge
        parts = [rows(x) for x in str(d).split(",")]
        spec = dict(parts[0][0])
        spec["layers"] = [L for sp, _ in parts for L in sp["layers"]]
        spec["pairs"] = {k: v for sp, _ in parts for k, v in sp.get("pairs", {}).items()}
        return spec, pd.concat([g for _, g in parts], ignore_index=True)
    spec = json.loads((Path(d) / "cells.json").read_text())
    g = pd.read_csv(Path(d) / "rows_measured.csv", low_memory=False)
    if "cls" not in g:
        g["cls"] = [classify(spec, r) for _, r in g.iterrows()]
    return spec, g


# -------------------------------------------------------------------------------------------------
# C
# -------------------------------------------------------------------------------------------------

def alpha_star(spec, g):
    """Smallest alpha whose start-only answers are judged the start persona >= 70%, collapse <= 10%."""
    out = {}
    s0 = g[(g.route == "start") & (g.cond == "none")]
    for (L, st), gg in s0.groupby(["layer", "start"]):
        ok = [a for a, ga in sorted(gg.groupby("alpha"), key=lambda x: x[0])
              if (ga.label == st).mean() >= 0.7 and ga.collapsed.mean() <= 0.10]
        out["%s|L%d" % (st, L)] = float(ok[0]) if ok else None
    return out


def page_C(a):
    spec, g = rows(a.dir)
    rng = np.random.default_rng(0)
    g = g[g.route != "ref"]
    met = ["frac_B", "dist_B_rel", "cos_B", "cos_dB"]
    curves = []
    for (L, route, cond, al, s), gg in g.groupby(["layer", "route", "cond", "alpha", "s"]):
        rec = dict(layer=int(L), route=route, cond=cond, alpha=float(al), s=float(s), n=len(gg))
        for m in met:
            for ro in ("L", "R"):
                rec["%s_%s" % (m, ro)] = boot(gg["%s_%s" % (m, ro)].dropna(), rng)
        for c in ("target", "start", "on-route", "off-route", "collapsed"):
            rec[c] = boot((gg.cls == c).to_numpy(), rng)
        curves.append(rec)
    window = []
    s1 = g[(g.s == 1.0) & (g.cond == "none")]
    for (L, st, al), gg in s1.groupby(["layer", "start", "alpha"]):
        window.append(dict(layer=int(L), start=st, alpha=float(al), target=float((gg.cls == "target").mean()),
                           collapsed=float(gg.collapsed.mean()), n=len(gg)))
    W = pd.DataFrame(window)
    W["ok"] = (W.target >= 0.3) & (W.collapsed <= 0.2)
    width = W.groupby("layer").apply(lambda x: x.groupby("start").ok.sum().mean()).to_dict()
    tgt = W.groupby("layer").target.mean().to_dict()
    best = max(width, key=lambda L: (width[L], tgt[L]))
    star = alpha_star(spec, g)
    notes = json.loads(Path(a.notes).read_text())
    data = dict(notes=notes["text"], facts=notes["facts"], layers=spec["layers"], alphas=spec["alphas"],
                starts=spec["starts"], routes=[r for r in ROUTE_ORDER if r in set(g.route)], curves=curves,
                window=window, alpha_star=star, best_layer=int(best), window_width={int(k): float(v) for k, v in width.items()})
    (Path(a.dir) / "calibration.json").write_text(json.dumps(clean(dict(alpha_star=star, best_layer=int(best),
                                                                         window_width=width))))
    write("cal_template.html", data, a.out)


# -------------------------------------------------------------------------------------------------
# E1
# -------------------------------------------------------------------------------------------------

def spearman(x, y):
    if len(x) < 3:
        return None
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def page_E1(a):
    spec, g = rows(a.dir)
    rng = np.random.default_rng(0)
    g = g[g.route != "ref"].copy()
    g["set"] = [spec["pairs"]["%s|L%d" % (p, L)]["set"] for p, L in zip(g.pair, g.layer)]
    cal = json.loads((Path(a.cal) / "calibration.json").read_text())
    star = cal["alpha_star"]
    g["is_star"] = [al == (star.get("%s|L%d" % (st, L)) or 1.0) for al, st, L in zip(g.alpha, g.start, g.layer)]
    pos = []
    for _, r in g.iterrows():
        nodes = spec["pairs"]["%s|L%d" % (r.pair, r.layer)]["routes"].get(r.route) or []
        lab = r.label
        pos.append(nodes.index(lab) / (len(nodes) - 1) if isinstance(lab, str) and lab in nodes and r.route != "start" else np.nan)
    g["route_pos"] = pos
    starts0 = g[g.route == "start"]
    order, comp, offc = [], [], []
    for view in ("1", "2", "4", "star"):
        gv = g[g.is_star] if view == "star" else g[g.alpha == float(view)]
        s0v = starts0[starts0.is_star] if view == "star" else starts0[starts0.alpha == float(view)]
        for (L, st_, route), gg in gv[gv.route != "start"].groupby(["layer", "set", "route"]):
            on = gg.dropna(subset=["route_pos"])
            b = []
            for _ in range(200):
                i = rng.integers(0, len(on), len(on)) if len(on) else []
                b.append(spearman(on.s.to_numpy()[i], on.route_pos.to_numpy()[i]) if len(on) > 2 else np.nan)
            b = np.array([x for x in b if x is not None and np.isfinite(x)])
            order.append(dict(view=view, layer=int(L), set=st_, route=route, n_on=len(on),
                              order=spearman(on.s.to_numpy(), on.route_pos.to_numpy()),
                              order_lo=float(np.percentile(b, 2.5)) if len(b) else None,
                              order_hi=float(np.percentile(b, 97.5)) if len(b) else None,
                              target_s1=boot((gg[gg.s == 1.0].cls == "target").to_numpy(), rng),
                              off=boot((gg.cls == "off-route").to_numpy(), rng),
                              collapsed=boot(gg.collapsed.to_numpy(), rng)))
            g0 = s0v[(s0v.layer == L) & (s0v.set == st_)]
            for s, gs in [(0.0, g0)] + list(gg.groupby("s")):
                rec = dict(view=view, layer=int(L), set=st_, route=route, s=float(s), n=len(gs))
                for c in ("start", "on-route", "target", "off-route", "assistant", "other", "collapsed"):
                    rec[c] = float((gs.cls == c).mean()) if len(gs) else None
                rec["frac_B"] = float(gs.frac_B_R.mean()) if len(gs) else None
                comp.append(rec)
    for (L, st_, route, al), gg in g[g.route != "start"].groupby(["layer", "set", "route", "alpha"]):
        offc.append(dict(layer=int(L), set=st_, route=route, alpha=float(al), off=boot((gg.cls == "off-route").to_numpy(), rng),
                         collapsed=boot(gg.collapsed.to_numpy(), rng), target=boot((gg[gg.s == 1.0].cls == "target").to_numpy(), rng)))
    notes = json.loads(Path(a.notes).read_text())
    write("st_template.html", dict(notes=notes["text"], facts=notes["facts"], layers=spec["layers"],
                                   routes=[r for r in ROUTE_ORDER if r in set(g.route)], order=order, comp=comp,
                                   off=offc, alpha_star=star), a.out)


# -------------------------------------------------------------------------------------------------
# E2
# -------------------------------------------------------------------------------------------------

def page_E2(a):
    spec, g = rows(a.dir)
    rng = np.random.default_rng(0)
    cal = json.loads((Path(a.cal) / "calibration.json").read_text())
    star = cal["alpha_star"]
    g = g[g.route != "ref"].copy()
    g["direction"] = [spec["pairs"]["%s|L%d" % (p, L)]["direction"] for p, L in zip(g.pair, g.layer)]
    side = []
    for _, r in g.iterrows():
        rt = spec["pairs"]["%s|L%d" % (r.pair, r.layer)]["routes"]
        mi = {n for k, v in rt.items() if k.endswith("m") for n in v[1:]}
        pl = {n for k, v in rt.items() if k.endswith("p") for n in v[1:]}
        lab = r.label
        side.append(0 if not isinstance(lab, str) or lab in mi & pl else (-1 if lab in mi else 1 if lab in pl else 0))
    g["side"] = side
    g["is_star"] = [al == (star.get("%s|L%d" % (st, L)) or 1.0) for al, st, L in zip(g.alpha, g.start, g.layer)]
    al_f = Path(a.dir) / "pca_vs_fletcher.json"
    align = json.loads(al_f.read_text())["cos"] if al_f.exists() else {}
    lean, cons, trait = [], [], []
    for view in ("1", "2", "4", "star"):
        gv = g[g.is_star] if view == "star" else g[g.alpha == float(view)]
        for k, gk in gv.groupby("direction"):
            base = gk[gk.route == "start"]
            b0 = (base.side > 0).mean() - (base.side < 0).mean() if len(base) else 0.0
            for route in ("FG", "FL", "PCA"):
                gr = pd.concat([gk[gk.route == route], base])
                for s, gs in gr.groupby("s"):
                    v = gs.side.to_numpy()
                    bb = rng.integers(0, len(v), size=(N := 500, len(v)))
                    bl = (v[bb] > 0).mean(1) - (v[bb] < 0).mean(1) - b0
                    lean.append(dict(view=view, direction=int(k), route=route, s=float(s), n=len(v),
                                     lean=float((v > 0).mean() - (v < 0).mean() - b0),
                                     lo=float(np.percentile(bl, 2.5)), hi=float(np.percentile(bl, 97.5)),
                                     collapsed=float(gs.collapsed.mean())))
                hi = gr[gr.s >= 0.75].groupby(["start", "q_idx", "para"]).side.mean()
                lo = gr[gr.s <= -0.75].groupby(["start", "q_idx", "para"]).side.mean()
                both = hi.index.intersection(lo.index)
                moved = [i for i in both if hi[i] != lo[i]]
                cons.append(dict(view=view, direction=int(k), route=route, n_series=len(both), n_moved=len(moved),
                                 consistency=float(np.mean([hi[i] > lo[i] for i in moved])) if moved else None))
                c = align.get(str(k)) if route == "PCA" else 1.0
                if "trait" in gr and c is not None and abs(c) >= 0.9:
                    # the hypotheses describe Fletcher's ends: PCA axis k is compared only when it is the
                    # same axis (|cos| >= 0.9), with its sign turned to Fletcher's
                    for sgn in (-1, 1):
                        gs = gr[gr.s == sgn * np.sign(c)]
                        want = "plus" if sgn > 0 else "minus"
                        trait.append(dict(view=view, direction=int(k), route=route, end=sgn, n=len(gs),
                                          expected=boot((gs.trait == want).to_numpy(), rng),
                                          opposite=boot((gs.trait == ("minus" if sgn > 0 else "plus")).to_numpy(), rng),
                                          baseline=float((base.trait == want).mean()) if len(base) else None))
    hyp = json.loads(Path(a.hyp).read_text())["directions"]["L%d" % spec["layers"][0]] if a.hyp else {}
    notes = json.loads(Path(a.notes).read_text())
    write("dir_template.html", dict(notes=notes["text"], facts=notes["facts"], hyp=hyp, lean=lean, cons=cons,
                                    trait=trait, routes=["FG", "FL", "PCA"]), a.out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["C", "E1", "E2"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--cal", default=None)
    ap.add_argument("--hyp", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"C": page_C, "E1": page_E1, "E2": page_E2}[a.exp](a)


if __name__ == "__main__":
    main()
