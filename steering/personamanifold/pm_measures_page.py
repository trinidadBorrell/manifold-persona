"""Page: the component-steering round (PC, PC4) scored with the PersonaManifold paper's three path
measures (IC, TS, MA; arXiv 2609.34571 §4.1). Same layout and tokens as the Persona Component Steering
page; its hypotheses (directions, ends) are read from that page's data.

    python -m steering.personamanifold.pm_measures_page --live <saved component page html> --out page.html
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from steering.personamanifold.paper_measures import BF_SYS, OUT, TRAITS, gens

HERE = Path(__file__).resolve().parent
N_BOOT = 300
rng = np.random.default_rng(0)


def ic_of(E, idx=None):
    """Mean pairwise cosine over pairs of different answers (a bootstrap draw can repeat an answer:
    pairs of an answer with its own copy are left out, else they add cosine 1 and bias IC up)."""
    n = len(E)
    if n < 2:
        return float("nan")
    S = E @ E.T
    idx = np.arange(n) if idx is None else np.asarray(idx)
    m = idx[:, None] != idx[None, :]
    return float(S[m].mean())


def icq_of(E, q, idx=None):
    """IC within question: for each question, the mean pairwise cosine among its answers (its two wordings x
    two samples); then the mean over questions. Copies from a bootstrap draw are left out as in ic_of."""
    q = np.asarray(q)
    idx = np.arange(len(E)) if idx is None else np.asarray(idx)
    vals = [ic_of(E[q == k], idx[q == k]) for k in np.unique(q)]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def ts_of(means):
    """means: (5 steps t = 0, .25, .5, .75, 1) x 5 traits -> TS over the 4 moves."""
    tot = np.sign(means[-1] - means[0])
    d = np.sign(np.diff(means, axis=0))
    return float((d == tot[None, :]).mean())


def clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else round(o, 4)
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default=None)
    a = ap.parse_args(argv)
    live = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',
                                Path(a.live).read_text(), re.S).group(1))
    dirs = [{k: d[k] for k in ("method", "component", "var_share", "minus", "plus", "hypothesis")}
            for d in live["grids"]["1"]["dirs"]]
    starts = live["starts"]

    g = gens()
    E = np.load(OUT / "minilm.npy")
    bf = pd.read_csv(OUT / "bigfive" / "bigfive.csv").set_index("key")
    g = g.join(bf[TRAITS + ["degenerate", "evidence"]], on="key")
    g["x"] = g["sign"] * g["step"]
    g["i"] = np.arange(len(g))
    ma = pd.read_csv(OUT / "ma.csv")
    mmeta = json.loads((OUT / "ma_meta.json").read_text())

    ic_rows, ts_rows, tr_rows, ma_rows, examples = [], [], [], [], []
    for alpha in ("1", "4"):
        ga = g[g.alpha == alpha]
        base = {s: ga[(ga.start == s) & (ga.method == "none")] for s in starts}
        for d in dirs:
            m, k = d["method"], d["component"]
            gd = ga[(ga.method == m) & (ga.component == k)]
            cells = {}                        # (start, x) -> rows
            for s in starts:
                cells[(s, 0.0)] = base[s]
                for x, gx in gd[gd.start == s].groupby("x"):
                    cells[(s, float(x))] = gx
            xs = sorted({x for _, x in cells})
            # IC per start and pooled (mean of starts, bootstrap answers within each cell)
            for x in xs:
                vals, valq = [], []
                for s in starts:
                    idx = cells[(s, x)].i.to_numpy()
                    qq = cells[(s, x)].q_idx.to_numpy()
                    v, vq = ic_of(E[idx]), icq_of(E[idx], qq)
                    vals.append(v)
                    valq.append(vq)
                    ic_rows.append(dict(alpha=alpha, method=m, component=k, start=s, x=x, ic=v, icq=vq, n=len(idx)))
                bs, bq = [], []
                for _ in range(N_BOOT):
                    vv, vq = [], []
                    for s in starts:
                        c = cells[(s, x)]
                        j = rng.integers(0, len(c), len(c))
                        b, qb = c.i.to_numpy()[j], c.q_idx.to_numpy()[j]
                        vv.append(ic_of(E[b], b))
                        vq.append(icq_of(E[b], qb, b))
                    bs.append(np.mean(vv))
                    bq.append(np.nanmean(vq))
                deg = float(pd.concat([cells[(t, x)] for t in starts]).degenerate.mean())
                ic_rows.append(dict(alpha=alpha, method=m, component=k, start="__all__", x=x, ic=float(np.mean(vals)), degenerate=deg,
                                    lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                    icq=float(np.mean(valq)), icq_lo=float(np.percentile(bq, 2.5)), icq_hi=float(np.percentile(bq, 97.5))))
            # Big Five means per (start, x) and TS per (start, half)
            for x in xs:
                for s in ["__all__"] + starts:
                    rr = pd.concat([cells[(t, x)] for t in starts]) if s == "__all__" else cells[(s, x)]
                    tr_rows.append(dict(alpha=alpha, method=m, component=k, start=s, x=x,
                                        **{t: float(rr[t].mean()) for t in TRAITS},
                                        degenerate=float(rr.degenerate.mean()), n=len(rr)))
            for half in (-1, 1):
                path = [0.0] + [half * t for t in (0.25, 0.5, 0.75, 1.0)]
                per = []
                for s in starts:
                    M = np.stack([cells[(s, x)][TRAITS].mean().to_numpy() for x in path])
                    v = ts_of(M)
                    per.append(v)
                    ts_rows.append(dict(alpha=alpha, method=m, component=k, half=half, start=s, ts=v))
                bs = []
                for _ in range(N_BOOT):
                    vv = []
                    for s in starts:
                        M = np.stack([cells[(s, x)][TRAITS].to_numpy()[rng.integers(0, len(cells[(s, x)]), len(cells[(s, x)]))].mean(0)
                                      for x in path])
                        vv.append(ts_of(M))
                    bs.append(np.mean(vv))
                sel = [r for r in ic_rows if r["alpha"] == alpha and r["method"] == m and r["component"] == k
                       and r["start"] == "__all__" and r["x"] in path[1:]]
                ic_half = np.mean([r["ic"] for r in sel])
                icq_half = np.mean([r["icq"] for r in sel])
                ts_rows.append(dict(alpha=alpha, method=m, component=k, half=half, start="__all__", ts=float(np.mean(per)),
                                    lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), ic_path=float(ic_half),
                                    icq_path=float(icq_half)))
            # examples: every answer of every (start, x), in question order
            for (s, x), rr in cells.items():
                for r in rr.sort_values(["q_idx", "prompt_idx", "sample"]).itertuples():
                    examples.append(dict(alpha=alpha, method=m, component=k, start=s, x=x, question=r.question,
                                         q=int(r.q_idx), para=int(r.prompt_idx), sample=int(r.sample), n_tok=int(r.n_new),
                                         text=str(r.response).replace("\ufffd", "?"),  # broken bytes in the model output
                                         degenerate=bool(r.degenerate) if r.degenerate == r.degenerate else None,
                                         evidence=r.evidence if isinstance(r.evidence, str) else "",
                                         bf=[None if r._asdict()[t] != r._asdict()[t] else int(r._asdict()[t]) for t in TRAITS]))
    alphas = sorted(ma.alpha.astype(str).unique()) if "alpha" in ma.columns else [None]   # clamp: MA per strength
    for d in dirs:
        d["ma"] = {}
        for al in alphas:
            dm = ma[(ma.method == d["method"]) & (ma.component == d["component"])]
            if al is not None:
                dm = dm[dm.alpha.astype(str) == al]
            for x, gx in dm.groupby("x"):
                for s, gs in gx.groupby("start"):
                    ma_rows.append(dict(alpha=al, method=d["method"], component=d["component"], start=s, x=float(x),
                                        d=float(gs.d_over_r95.mean()), adh=1 - float(gs.d_over_r95.mean()), nn=gs.nn.iloc[0]))
                ma_rows.append(dict(alpha=al, method=d["method"], component=d["component"], start="__all__", x=float(x),
                                    d=float(gx.d_over_r95.mean()), adh=1 - float(gx.d_over_r95.mean()),
                                    adh_lo=1 - float(gx.d_over_r95.max()), adh_hi=1 - float(gx.d_over_r95.min())))
            for half in (-1, 1):
                hx = dm[(np.sign(dm.x) == half)]
                per = {s: 1 - float(v.d_over_r95.mean()) for s, v in hx.groupby("start")}
                d["ma"].setdefault(al or "all", {})[str(half)] = dict(pooled=float(np.mean(list(per.values()))), per=per)

    out = Path(a.out)
    (out.parent / "data").mkdir(exist_ok=True)
    (out.parent / "data" / "examples.json").write_text(json.dumps(clean(examples), ensure_ascii=False, allow_nan=False))
    qs = (g[g.alpha == "1"].groupby(["q_idx", "prompt_idx"]).question.first().reset_index()
          .rename(columns={"q_idx": "q", "prompt_idx": "para"}).to_dict("records"))
    gen = dict(n_new_max=int(g.n_new.max()), hit_max=float(g.hit_max.mean()), temperature=float(g.temperature.iloc[0]),
               top_p=float(g.top_p.iloc[0]), top_k=int(g.top_k.iloc[0]))
    data = dict(dirs=dirs, starts=starts, traits=TRAITS, ic=ic_rows, ts=ts_rows, traitcurves=tr_rows, ma=ma_rows,
                questions=qs, gen=gen, ma_meta=mmeta, order=["1", "4"],
                notes=json.loads(Path(a.notes).read_text()) if a.notes else {},
                judge=dict(system=BF_SYS, instructions=(HERE / "BIGFIVE_AGENT.md").read_text()))
    s = json.dumps(clean(data), ensure_ascii=False, allow_nan=False)
    tpl = (HERE / "pm_measures_template.html").read_text()
    nt = data["notes"]
    if nt.get("page_title"):          # a copy of the page under another name (the clamp round)
        tpl = tpl.replace("<title>Persona Component Paper Measures</title>", "<title>%s</title>" % nt["page_title"])
        tpl = re.sub(r'<meta name="description" content="[^"]*">', '<meta name="description" content="%s">' % nt["page_description"], tpl, 1)
    Path(a.out).write_text(tpl.replace("__DATA__", s.replace("</", "<\\/")))
    print("page -> %s (%.2f MB)" % (a.out, Path(a.out).stat().st_size / 1e6))


if __name__ == "__main__":
    main()
