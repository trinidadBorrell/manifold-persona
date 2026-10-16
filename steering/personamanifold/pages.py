"""Build the two Part 2 pages (plan 2026-10-08).

  count  How Many Persona Components -- from P2/components_{last,mean}.json
  steer  Persona Component Steering  -- from PC/{cells.json, generations.csv} and the PC judge

Notes (claims, captions, hypotheses) are written by hand after reading the numbers and passed as
a JSON file, so the page states only what the numbers show.

    python -m steering.personamanifold.pages count --p2 P2 --notes notes_count.json --out count.html
    python -m steering.personamanifold.pages steer --pc PC --judge JPC --notes notes_steer.json --out steer.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from steering.ace_study.common import REP4_COLLAPSED
from steering.personamanifold.judge_batches import pm_key

HERE = Path(__file__).resolve().parent


def _write(tpl, data, out):
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    Path(out).write_text((HERE / tpl).read_text().replace("__DATA__", payload))
    print("page -> %s (%.2f MB)" % (out, Path(out).stat().st_size / 1e6))


def _clean(o):
    if isinstance(o, float):
        return None if not np.isfinite(o) else round(o, 4)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def count(a):
    P2 = Path(a.p2)
    views = {}
    for v in ("last", "mean"):
        f = P2 / ("components_%s.json" % v)
        if f.exists():
            s = json.loads(f.read_text())
            views[v] = s["views"][v]
            layers = s["layers"]
            for L, rec in views[v]["per_layer"].items():
                for k in ("isomap_residual", "pca_residual"):
                    rec.pop(k, None)
            views[v].pop("cka", None)
            views[v]["per_layer"] = {int(k): r for k, r in views[v]["per_layer"].items()}
    notes = json.loads(Path(a.notes).read_text())
    data = dict(notes=notes["text"], facts=notes["facts"], method=notes["method"], layers=layers, views=views)
    _write("count_template.html", _clean(data), a.out)


def steer_grid(pc, judge, notes):
    PC = Path(pc)
    spec = json.loads((PC / "cells.json").read_text())
    d = pd.read_csv(PC / "generations.csv")
    lab = {}
    for line in (Path(judge) / "judgements.jsonl").read_text().splitlines():
        r = json.loads(line)
        lab[r["key"]] = r["persona_kind"]
    d["k0"] = [pm_key(r) for _, r in d.iterrows()]
    side_of = {}
    for c in spec["cells"]:
        if c["sign"]:
            for p in c["path"][1:]:
                key = (c["start"], c["method"], c["component"], p)
                side_of[key] = c["sign"] if side_of.get(key, c["sign"]) == c["sign"] else 0
    rng = np.random.default_rng(0)
    dirs, lean, examples = [], [], []
    for method in ("pga", "pca"):
        for e in spec["ends"][method]:
            k = e["component"]
            rows = []
            for start in spec["starts"]:
                stem = "%s_%s_c%d" % (start, method, k)
                m = (d.start == start) & (((d.method == method) & (d.component == k)) | (d.method == "none"))
                g = d[m].copy()
                g["label"] = [("collapsed" if dr > REP4_COLLAPSED else lab.get(stem + "#" + kk))
                              for kk, dr in zip(g.k0, g.deg_rep4)]
                g["side"] = [side_of.get((start, method, k, l), 0) if l else 0 for l in g.label]
                g["x"] = g.sign * g.step
                rows.append(g)
            G = pd.concat(rows)
            G = G[G.label.notna()]
            # lean = (% judged a + side persona) - (% judged a - side persona), shown as the CHANGE from
            # the same start's unsteered answers (step 0): personas next to the start are judged often
            # without any steering, so the raw share mixes resemblance with movement
            lv = lambda v: float((v > 0).mean() - (v < 0).mean())
            base = {st_: lv(g.side.to_numpy()) for st_, g in G[G.x == 0].groupby("start")}
            for (start, x), g in G.groupby(["start", "x"]):
                lean.append(dict(method=method, component=k, start=start, x=float(x),
                                 lean=lv(g.side.to_numpy()) - base.get(start, 0.0), n=len(g)))
            b0 = lv(G[G.x == 0].side.to_numpy())
            for x, g in G.groupby("x"):
                s = g.side.to_numpy()
                b = rng.integers(0, len(s), size=(1000, len(s)))
                bl = (s[b] > 0).mean(1) - (s[b] < 0).mean(1) - b0
                lean.append(dict(method=method, component=k, start="__all__", x=float(x),
                                 lean=lv(s) - b0, n=len(s),
                                 lo=float(np.percentile(bl, 2.5)), hi=float(np.percentile(bl, 97.5))))
            hi = G[G.x >= 0.75].groupby(["start", "q_idx", "prompt_idx"]).side.mean()
            lo = G[G.x <= -0.75].groupby(["start", "q_idx", "prompt_idx"]).side.mean()
            both = hi.index.intersection(lo.index)
            moved = [i for i in both if hi[i] != lo[i]]          # series whose lean differs between the two ends
            cons = float(np.mean([hi[i] > lo[i] for i in moved])) if moved else None
            dirs.append(dict(method=method, component=k, var_share=e["var_share"], minus=e["minus"][:5], plus=e["plus"][:5],
                             hypothesis=notes["hypotheses"].get("%s%d" % (method, k), ""), consistency=cons,
                             n_series=int(len(both)), n_moved=len(moved)))
            for x in (-1.0, -0.5, 0.0, 0.5, 1.0):
                gx = G[G.x == x]
                for j in gx.index[rng.permutation(len(gx))[:4]]:
                    r = G.loc[j]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    t = r.response if isinstance(r.response, str) else ""
                    examples.append(dict(method=method, component=k, start=r.start, x=float(x), label=r.label,
                                         side=int(r.side), question=r.question, text=t[:2500]))
    return spec, dict(dirs=dirs, lean=lean, examples=examples)


def steer(a):
    notes = json.loads(Path(a.notes).read_text())
    grids, spec = {}, None
    for pc, judge, lab in zip(a.pc, a.judge, a.strength):
        spec, grids[lab] = steer_grid(pc, judge, notes)
    data = dict(notes=notes["text"], facts=notes["facts"], starts=spec["starts"], grids=grids, order=list(a.strength))
    _write("steer_template.html", _clean(data), a.out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["count", "steer"])
    ap.add_argument("--p2")
    ap.add_argument("--pc", action="append")
    ap.add_argument("--judge", action="append")
    ap.add_argument("--strength", action="append", help="label per --pc grid, e.g. 1 and 4")
    ap.add_argument("--notes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    count(a) if a.what == "count" else steer(a)


if __name__ == "__main__":
    main()
