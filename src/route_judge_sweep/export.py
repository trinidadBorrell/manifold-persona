"""Artifact data for the judge prompt sweep: label tables, per-pair answer files and the §4 metrics.

    python -m route_judge_sweep.export --out <dir> [--draw 0]

Uses whatever labels exist (JS/labels.csv from collect.py) plus J1 (the existing control labels).
Writes <dir>/data/index.json (answers, label codes per judge, metrics) and <dir>/data/pair_<model>_<pair>.json
(question, response, every judge's label / evidence / analysis, the draw-0 prompts of batch 0).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from route_judge_sweep import prompts as P
from route_judge_sweep.build import MODELS, PAIRS, PM, ROUTES

import os
JS = Path(os.environ.get("JSWEEP_DIR", PM / "JS"))       # the clamp copy uses $PM/JSC
J1_DIRS = {"8b": PM / "J_ST15", "14b": PM / "qwen3_14b" / "J_ST21"}
JUDGES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10"]
if os.environ.get("EXPORT_JUDGES"):          # e.g. "J2,J8,J10": only the judges that are complete
    JUDGES = [j for j in JUDGES if j in os.environ["EXPORT_JUDGES"].split(",")]
DESC = {"J1": "Original control (existing labels)", **{J: s["desc"] for J, s in P.JUDGES.items()},
        "J10": "Giovanni's ranking judge, exact port (top-1 of 10)"}
NONE = {"other", "none", "not_stated", "unlisted"}
N_BOOT = 300


def norm(x):
    return str(x).strip().lower().replace(" ", "_") if isinstance(x, str) else None


def clean(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.generic):
        return x.item()
    return x


def load_answers():
    a = pd.read_csv(JS / "answers.csv", low_memory=False)
    a["ref"] = ""
    r = a.route == "ref"
    a.loc[r & (a.sys_role == a.start), "ref"] = "A"
    a.loc[r & (a.sys_role == a.target), "ref"] = "B"
    a.loc[r & a.sys_role.isna(), "ref"] = "N"
    a["collapsed"] = a.collapsed.astype(str) == "True"
    return a.reset_index(drop=True)


def load_j1(a):
    rows = []
    for m, d in J1_DIRS.items():
        keys = set(a.key[a.model == m])
        for line in open(d / "judgements.jsonl"):
            x = json.loads(line)
            if x["key"] in keys:
                rows.append(dict(model=m, key=x["key"], judge="J1", draw=0, label=norm(x.get("persona_kind")),
                                 intensity=x.get("intensity"), evidence=x.get("evidence")))
    j = pd.DataFrame(rows)
    # collapsed answers were not sent to the control judge: they count as collapsed
    got = set(zip(j.model, j.key))
    extra = [dict(model=r.model, key=r.key, judge="J1", draw=0, label="collapsed",
                  analysis="Not judged: the 4-gram rule flagged it as collapsed.")
             for r in a.itertuples() if (r.model, r.key) not in got]
    return pd.concat([j, pd.DataFrame(extra)], ignore_index=True)


def load_j10():
    """J10 verdicts (src/route_judge_sweep/j10.py): label = the judge's first-ranked persona."""
    f = JS / "J10" / "labels.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    rk = d["rank"].map(json.loads)
    return pd.DataFrame(dict(model=d.model, key=d.key, judge="J10", draw=0, label=rk.map(lambda r: r[0]),
                             rank10=d["rank"], analysis=d.why, nonsense=d.nonsense))


def recall(a, lab, pinfo, m, p, r):
    """J10 recall@k (Giovanni's score_rank.py): share of answers whose target is in the first k names."""
    d = a[(a.model == m) & ((a.pair == p) if p != "*" else True)]
    routes = ROUTES if r == "*" else [r]
    sets = dict(s0=d[d.route == "start"], s1=d[d.route.isin(routes) & np.isclose(d.s, 1.0)],
                B=d[d.ref == "B"], A=d[d.ref == "A"])
    out = {}
    for name, x in sets.items():
        rk = [json.loads(lab.rank10.get(i)) for i in x.index if isinstance(lab.rank10.get(i), str)]
        tg = [pinfo["%s|%s" % (m, pr)]["target"] for i, pr in zip(x.index, x.pair) if isinstance(lab.rank10.get(i), str)]
        for k in (1, 5, 10):
            out["%s_r%d" % (name, k)] = round(float(np.mean([t in q[:k] for t, q in zip(tg, rk)])), 4) if rk else None
        out[name + "_n"] = len(rk)
    return out


def pair_info(a):
    out = {}
    for m, (d, L) in MODELS.items():
        spec = json.loads((d / "cells.json").read_text())
        for pair in PAIRS:
            p = spec["pairs"]["%s|L%d" % (pair, L)]
            routes = p["routes"] if isinstance(p["routes"], dict) else ast.literal_eval(p["routes"])
            out["%s|%s" % (m, pair)] = dict(model=m, layer=L, pair=pair, start=p["start"], target=p["target"],
                                            routes={r: routes[r] for r in ROUTES},
                                            options=P.pair_options(p["options"]))
    return out


def roles_of(pi):
    """label -> role in the pair: start / target / on-route / assistant."""
    on = {x for r in pi["routes"].values() for x in r[1:-1]}
    return pi["start"], pi["target"], on


def metrics(a, lab, pinfo, J, sel_model, sel_pair, sel_route):
    """Shares and order for one judge on one selection (draw 0). J9 uses probabilities."""
    m = (a.model == sel_model)
    if sel_pair != "*":
        m &= a.pair == sel_pair
    d = a[m]
    if d.empty:
        return None
    L = lab.reindex(d.index)
    have = L.label.notna()
    d, L = d[have], L[have]
    if d.empty:
        return None
    pi = {p: pinfo["%s|%s" % (sel_model, p)] for p in d.pair.unique()}

    def p_of(rows, which):  # probability or indicator that each answer is judged as `which` (start/target)
        name = rows.pair.map(lambda p: pi[p][which])
        if J == "J9":
            return np.array([json.loads(r).get(n, 0.0) if isinstance(r, str) else 0.0
                             for r, n in zip(L.loc[rows.index, "ranking"], name)])
        return (L.loc[rows.index, "label"].values == name.values).astype(float)

    routes = ROUTES if sel_route == "*" else [sel_route]
    steer = d[d.route.isin(routes)]
    s0 = d[d.route == "start"]
    s1 = steer[np.isclose(steer.s, 1.0)]
    refA, refB = d[d.ref == "A"], d[d.ref == "B"]
    # position of the judged persona on the answer's route (0 = start, 1 = target), only answers on the route
    pos_rows = []
    for r in pd.concat([s0.assign(rts=[routes] * len(s0)), steer.assign(rts=[[x] for x in steer.route])]).itertuples():
        lb = L.label.get(r.Index)
        for rt in r.rts:
            path = pi[r.pair]["routes"][rt]
            if lb in path:
                pos_rows.append((r.q_idx, r.s, path.index(lb) / (len(path) - 1)))
    pos = pd.DataFrame(pos_rows, columns=["q", "s", "pos"])

    Q = int(a.q_idx.max()) + 1
    sets = {k: (df.q_idx.values.astype(int), p_of(df, w)) for k, df, w in
            [("t0", s0, "target"), ("t1", s1, "target"), ("st0", s0, "start"), ("st1", s1, "start"),
             ("bB", refB, "target"), ("aB", refA, "target")]}
    pq, ps, pv = pos.q.values.astype(int), pos.s.values, pos.pos.values

    def stat(wq):  # wq = how many times each question is drawn
        out = {}
        for k, (q, x) in sets.items():
            w = wq[q]
            out[k] = float((w * x).sum() / w.sum()) if len(q) and w.sum() > 0 else float("nan")
        rep = wq[pq].astype(int)
        x, y = np.repeat(ps, rep), np.repeat(pv, rep)
        out["rho"] = (float(spearmanr(x, y)[0]) if len(x) > 4 and len(set(x)) > 1 and len(set(y)) > 1
                      else float("nan"))
        out["rise"] = out["t1"] - out["t0"]
        out["drop"] = out["st0"] - out["st1"]
        return out

    qs = np.array(sorted(d.q_idx.unique()), int)
    est = stat(np.isin(np.arange(Q), qs).astype(float))
    rng = np.random.default_rng(0)
    boots = [stat(np.bincount(rng.choice(qs, len(qs)), minlength=Q).astype(float)) for _ in range(N_BOOT)]
    res = dict(n=int(len(d)), n_on=int(len(pos)))
    for k, v in est.items():
        bs = np.array([b[k] for b in boots], float)
        bs = bs[~np.isnan(bs)]
        res[k] = None if math.isnan(v) else round(v, 4)
        if k in ("rise", "drop", "rho") and len(bs):
            res[k + "_ci"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    return res


def kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    po = float((x == y).mean())
    cats = set(x) | set(y)
    pe = sum(float((x == c).mean()) * float((y == c).mean()) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draw", type=int, default=0)
    o = ap.parse_args(argv)
    out = Path(o.out) / "data"
    out.mkdir(parents=True, exist_ok=True)

    a = load_answers()
    idx = {(m, k): i for i, (m, k) in enumerate(zip(a.model, a.key))}
    allab = pd.read_csv(JS / "labels.csv", low_memory=False)
    j1 = pd.DataFrame() if (allab.judge == "J1").any() or "J1" not in JUDGES else load_j1(a)      # JSC carries J1 in labels.csv
    allab = pd.concat([allab, j1, load_j10()], ignore_index=True)
    allab["i"] = [idx.get((m, k)) for m, k in zip(allab.model, allab.key)]
    allab = allab[allab.i.notna()]
    allab["i"] = allab.i.astype(int)
    lab = allab[allab.draw == o.draw]
    pinfo = pair_info(a)

    # label codes per judge (draw o.draw), aligned to answer index
    judges = {}
    for J in JUDGES:
        dj = lab[lab.judge == J].set_index("i")
        vocab = sorted(dj.label.dropna().unique())
        code = {v: c for c, v in enumerate(vocab)}
        codes = [code.get(dj.label.get(i), -1) if i in dj.index else -1 for i in range(len(a))]
        jd = dict(desc=DESC[J], vocab=vocab, codes=codes, n=int((np.array(codes) >= 0).sum()))
        if J == "J9":
            probs = []
            for i in range(len(a)):
                r = dj.ranking.get(i) if i in dj.index else None
                pr = json.loads(r) if isinstance(r, str) else {}
                for k in pr:
                    if k not in code:
                        code[k] = len(vocab); vocab.append(k)
                probs.append({str(code[k]): round(v, 3) for k, v in pr.items()})
            jd["probs"] = probs
        if J == "J10":                     # the first five ranked names of every answer
            top5 = []
            for i in range(len(a)):
                r = dj.rank10.get(i) if i in dj.index else None
                rk = json.loads(r)[:5] if isinstance(r, str) else []
                for k in rk:
                    if k not in code:
                        code[k] = len(vocab); vocab.append(k)
                top5.append([code[k] for k in rk])
            jd["top5"] = top5
        judges[J] = jd

    # metrics for every selection
    met = []
    for J in JUDGES:
        L = lab[lab.judge == J].set_index("i")[["label", "ranking"]] if J == "J9" else \
            lab[lab.judge == J].set_index("i")[["label"]]
        for m in MODELS:
            for p in ["*"] + PAIRS:
                for r in ["*"] + ROUTES:
                    res = metrics(a, L, pinfo, J, m, p, r) if J not in ("J3", "J4") else None
                    if res and J == "J10":
                        res.update(recall(a, lab[lab.judge == J].set_index("i"), pinfo, m, p, r))
                    if res:
                        met.append(dict(judge=J, model=m, pair=p, route=r, **res))
        print("metrics", J, flush=True)

    # agreement between draws on answers judged in two draws
    agree = []
    for J in JUDGES:
        dj = allab[allab.judge == J]
        piv = dj.pivot_table(index="i", columns="draw", values="label", aggfunc="first")
        if 0 in piv and 1 in piv:
            both = piv[[0, 1]].dropna()
            if len(both):
                po, k = kappa(both[0], both[1])
                agree.append(dict(judge=J, n=int(len(both)), agree=round(po, 3), kappa=round(k, 3)))

    ans = dict(fields=["model", "pair", "route", "s", "ref", "q", "collapsed"],
               rows=[[r.model, r.pair, r.route, round(float(r.s), 2), r.ref, int(r.q_idx), bool(r.collapsed)]
                     for r in a.itertuples()])
    n_jobs = len(json.loads((JS / "manifest.json").read_text())["jobs"])
    done = allab[allab.judge != "J1"].name.nunique()
    index = dict(judges=judges, answers=ans, pairs=pinfo, metrics=met, agree=agree, draw=o.draw,
                 progress=dict(jobs_done=int(done), jobs_total=n_jobs, labels=int((allab.judge != "J1").sum())),
                 instructions=(P._REPO / "src/route_judge_sweep/JUDGE_SWEEP_AGENT.md").read_text(),
                 instructions_j10=(P._REPO / "src/route_judge_sweep/J10_AGENT.md").read_text())
    s = json.dumps(index, ensure_ascii=False, allow_nan=False)
    (out / "index.json").write_text(s)
    print("index.json", len(s) // 1000, "kB")

    # per-pair answer files
    keep = ["evidence", "analysis", "intensity", "self_id", "description", "secondary", "ranking", "mixed",
            "rank10", "nonsense"]
    for (m, pair), g in a.groupby(["model", "pair"]):
        rows = []
        for i, r in g.iterrows():
            jl = {}
            for x in allab[allab.i == i].itertuples():
                e = {k: clean(getattr(x, k, None)) for k in keep}
                e = {k: (str(v) if k == "intensity" else v) for k, v in e.items() if v is not None}
                e["label"] = clean(x.label)
                jl.setdefault(x.judge, {})[str(x.draw)] = e
            rows.append(dict(i=int(i), question=r.question, response=r.response, sys_role=clean(r.sys_role), j=jl))
        stem = "%s_%s" % (m, pair.replace(".R.", "-"))
        prompts = {}
        for J in JUDGES[1:]:
            f = JS / "sys" / ("%s_%s_b0_d%d.txt" % (J, stem, 0))
            if f.exists():
                prompts[J] = f.read_text()
        f10 = JS / "J10" / "chunk_000.txt"
        if f10.exists():
            t = f10.read_text()
            prompts["J10"] = t[:t.index("\n1. ")] + "\n1. <answer 1, whitespace collapsed, first 500 characters>\n... (15 answers per chunk)"
        s = json.dumps(dict(rows=rows, prompts=prompts), ensure_ascii=False, allow_nan=False)
        (out / ("pair_%s.json" % stem)).write_text(s)
        print(stem, len(s) // 1000, "kB")


if __name__ == "__main__":
    main()
