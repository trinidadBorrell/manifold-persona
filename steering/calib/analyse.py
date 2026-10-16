"""Per-row measures and page data for C / E1 / E2 (plan 2026-10-09).

Latent measures (last answer token; at the steered layer L and at L + 4):
  references per (pair, layer, question, paraphrase), averaged over their 2 samples:
      h0 = no system prompt, hA = start prompted, hB = target prompted (C, E1)
  frac_B   = <h - h0, d> / |d|^2, d = hB - h0      0 = neutral, 1 = at prompted B, > 1 = past B
  dist_B   = |h - hB|                               (and dist_B_rel = dist_B / |d|)
  cos_B    = cos(h, hB);  cos_dB = cos(h - h0, d)
  near     = nearest of the 276 persona states (E1b, same layer), near_rel = its distance / median
             nearest-neighbour distance between personas
Behaviour: the judge label (collapsed rows are labelled "collapsed", never judged).

    python -m steering.calib.analyse --dir CAL --judge J_CAL --out CAL/page_data.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from steering.calib.judge import key

N_BOOT = 500


def load(d, judge, trait=None):
    d = Path(d)
    spec = json.loads((d / "cells.json").read_text())
    g = pd.read_csv(d / "generations.csv")
    S = np.load(d / "states.npz")
    cell = pd.DataFrame(spec["cells"]).set_index("cell_id")
    for c in ("pair", "start", "target", "route", "cond", "s"):
        g[c] = cell.loc[g.cell_id, c].to_numpy()
    g["key"] = [key(r) for _, r in g.iterrows()]
    lab = {}
    for line in (Path(judge) / "judgements.jsonl").read_text().splitlines():
        r = json.loads(line)
        lab[r["key"]] = r["persona_kind"]
    g["label"] = [("collapsed" if c else lab.get(k)) for k, c in zip(g.key, g.collapsed)]
    if trait:
        tl = {}
        for line in (Path(trait) / "judgements.jsonl").read_text().splitlines():
            r = json.loads(line)
            tl[r["key"]] = r["persona_kind"]
        g["trait"] = [("collapsed" if c else tl.get(k)) for k, c in zip(g.key, g.collapsed)]
    return spec, g, S


def latent(spec, g, S, d):
    """Adds the latent measures to g (both readouts)."""
    for ro, arr in (("L", "last_L"), ("R", "last_R")):
        H = S[arr].astype(np.float64)
        ref = {}
        for (pair, L, q, p, cond), idx in g[g.route == "ref"].groupby(["pair", "layer", "q_idx", "para", "cond"]).groups.items():
            ref[(pair, L, q, p, cond)] = H[np.asarray(list(idx))].mean(0)
        pers = {}
        for L in spec["layers"]:
            z = np.load(Path(d) / ("personas_L%d.npz" % L))
            X = z["X_abs_L" if ro == "L" else "X_abs_R"].astype(np.float64)
            sq = (X ** 2).sum(1)
            nn = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0))
            np.fill_diagonal(nn, np.inf)
            pers[L] = (X, z["roles"].astype(str), np.median(nn.min(1)))
        out = {k: np.full(len(g), np.nan) for k in ("frac_B", "dist_B", "dist_B_rel", "cos_B", "cos_dB", "near_rel")}
        near = np.empty(len(g), dtype=object)
        for i, (pair, L, q, p) in enumerate(zip(g.pair, g.layer, g.q_idx, g.para)):
            h = H[i]
            X, roles, med = pers[L]
            dd = np.sqrt(((X - h) ** 2).sum(1))
            j = int(np.argmin(dd))
            near[i], out["near_rel"][i] = roles[j], dd[j] / med
            h0, hB = ref.get((pair, L, q, p, "ref_neutral")), ref.get((pair, L, q, p, "ref_B"))
            if h0 is None or hB is None:
                continue
            dv = hB - h0
            out["frac_B"][i] = (h - h0) @ dv / (dv @ dv)
            out["dist_B"][i] = np.linalg.norm(h - hB)
            out["dist_B_rel"][i] = out["dist_B"][i] / np.linalg.norm(dv)
            out["cos_B"][i] = h @ hB / (np.linalg.norm(h) * np.linalg.norm(hB))
            out["cos_dB"][i] = (h - h0) @ dv / (np.linalg.norm(h - h0) * np.linalg.norm(dv) + 1e-12)
        for k, v in out.items():
            g["%s_%s" % (k, ro)] = v
        g["near_%s" % ro] = near
    return g


def boot(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (None, None, None)
    b = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return (float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def route_pos(spec, pair, L, route, label):
    """Position (0..1) of the judged persona on this route; None if not on it."""
    nodes = spec["pairs"]["%s|L%d" % (pair, L)]["routes"].get(route)
    if not nodes or label not in nodes:
        return None
    return nodes.index(label) / (len(nodes) - 1)


def classify(spec, r):
    """start / target / on-route / off-route / assistant / other / collapsed."""
    lab = r["label"]
    if lab is None or (isinstance(lab, float) and np.isnan(lab)):
        return "unjudged"
    if lab in ("collapsed", "other"):
        return lab
    if lab == r["start"]:
        return "start"
    if lab == r["target"]:
        return "target"
    nodes = spec["pairs"]["%s|L%d" % (r["pair"], r["layer"])]["routes"].get(r["route"]) or []
    if lab in nodes:
        return "on-route"
    if lab == "assistant":
        return "assistant"
    return "off-route"


def summarise(spec, g):
    """Generic per-cell summary used by all three pages."""
    rng = np.random.default_rng(0)
    g = g.copy()
    g["cls"] = [classify(spec, r) for _, r in g.iterrows()]
    rows = []
    keys = ["layer", "route", "cond", "alpha", "s"]
    for kk, gg in g[g.route != "ref"].groupby(keys):
        rec = dict(zip(keys, [int(kk[0]), kk[1], kk[2], float(kk[3]), float(kk[4])]), n=len(gg))
        for c in ("start", "target", "on-route", "off-route", "assistant", "other", "collapsed"):
            rec[c] = boot((gg.cls == c).to_numpy(), rng)
        for m in ("frac_B", "dist_B_rel", "cos_B", "cos_dB", "near_rel"):
            for ro in ("L", "R"):
                col = "%s_%s" % (m, ro)
                if col in gg:
                    rec[col] = boot(gg[col].dropna().to_numpy(), rng)
        rows.append(rec)
    return g, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--trait", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec, g, S = load(a.dir, a.judge, a.trait)
    g = latent(spec, g, S, a.dir)
    g, rows = summarise(spec, g)
    g.drop(columns=["response"]).to_csv(Path(a.dir) / "rows_measured.csv", index=False)
    Path(a.out).write_text(json.dumps(dict(exp=spec["exp"], summary=rows), default=float))
    print("%d rows measured, %d summary cells -> %s" % (len(g), len(rows), a.out))


if __name__ == "__main__":
    main()
