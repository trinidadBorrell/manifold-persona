"""Data for the single "steering story" page (output/steering-story-25_09/story_data.json).

Every point is a STEERED state (mean layer-L activation over response tokens with the hook live).
No text footprint anywhere. Cosine = centred cosine to the target centroid (both vectors minus the
mean of that layer's 275 centroids). PCA-3 is fitted per layer on that layer's centroids.

Sections
  1a  Giovanni "pin" vs our "add", identical setup   output/pin-vs-add-25_09
  1b  our add, Study A linear arm, 4 routes           output/steering-fix-24_09/studyA
  2   same Study A linear answers read at L19 and L32 output/steering-l19-to-l32-25_09
  3a  start = per-question mean position (Study B)   output/steering-fix-24_09/studyB
  3b  start = each answer's own position             output/steering-per-answer-25_09
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from steering.followups.common import Geom, degeneracy
from steering.run_steering import INTROSPECTIVE_QUESTIONS

O = Path("output")
OUT = O / "steering-story-25_09"
L32_GEOM = "/data/project/eeg_foundation/data/manifold_persona/geom_l25l32/geom_cache_L32.npz"


def collapsed(t):
    """Stricter than rep4>10 (user, 2026-09-25): also a 20+ same-character run (e.g. 妾妾妾, no
    spaces) or >40% repeated 4-word sequences (short loops like "I am the thousandth of a
    thousandth" x6). A collapsed answer counts as collapsed, never as target."""
    import re
    from collections import Counter
    t = str(t)
    if re.search(r"(.)\1{19,}", t):
        return True
    w = re.findall(r"\S+", t)
    if len(w) < 12:
        return False
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    c = Counter(g)
    return max(c.values()) > 10 or sum(v - 1 for v in c.values()) / len(g) > 0.40


def shares(g, study, A, B, J, canary_tab):
    """Per-alpha judge shares from the raw judge labels, with the stricter collapse rule applied
    first (a collapsed answer is never counted as target or source). Canary from the figC table."""
    keys = ["%s|%s|%d|%d" % (study, x.cell_id, int(x.unit_index), int(x["sample"])) for _, x in g.iterrows()]
    pk = np.array([J.get(k, "collapsed") for k in keys])
    c = np.array([collapsed(t) for t in g.response.fillna("")]) | (pk == "collapsed")
    al = g.alpha.to_numpy()

    def f(a):
        k = al == a
        return dict(tgt=float(((pk[k] == B) & ~c[k]).mean()), src=float(((pk[k] == A) & ~c[k]).mean()),
                    coll=float(c[k].mean()), canary=float(canary_tab.loc[a].canary_acc))
    return f


def r(x, n=3):
    return [round(float(v), n) for v in x]


class Space:
    def __init__(self, G):
        self.G, self.mu = G, G.C.mean(0)
        _, s, Vt = np.linalg.svd(G.C - self.mu, full_matrices=False)
        self.V, self.evr = Vt[:3], (s ** 2 / (s ** 2).sum())[:3]

    def p(self, X):
        return ((np.atleast_2d(X) - self.mu) @ self.V.T)

    def cos(self, X, c):
        Xc, cc = np.atleast_2d(X) - self.mu, c - self.mu
        return Xc @ cc / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(cc))


def series(S, X, groups, alphas, cB, extra=None):
    """Per alpha: mean PCA point, mean/sd centred cos to target, mean distance to target."""
    out = []
    for a in alphas:
        m = groups == a
        if not m.any():
            continue
        Xa = X[m].astype(np.float64)
        Xa = Xa[np.isfinite(Xa).all(1)]
        cs = S.cos(Xa, cB)
        dd = np.linalg.norm(Xa - cB, axis=1)
        row = dict(a=float(a), pc=r(S.p(Xa.mean(0))[0]), cos=float(cs.mean()), cos_sd=float(cs.std()),
                   d=float(dd.mean()), d_sd=float(dd.std()), n=int(len(Xa)))
        if extra:
            row.update(extra(a))
        out.append(row)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G19, G32 = Geom(), Geom(L32_GEOM)
    S19, S32 = Space(G19), Space(G32)
    D = dict(names=G19.names, cent19=[r(z, 2) for z in S19.p(G19.C)], cent32=[r(z, 2) for z in S32.p(G32.C)],
             evr19=r(S19.evr), evr32=r(S32.evr), s1a={}, s1b={}, s2={}, s3a={}, s3b={})

    def chord(S, G, A, B, amax, start=None):
        cA, cB = G.c(A), G.c(B)
        s0 = cA if start is None else start
        ts = np.linspace(0, amax, 21)
        return r(S.p(s0[None] + ts[:, None] * (cB - cA if start is None else cB - start)[None]).ravel().tolist())

    # ---------- 1a: pin vs add (Giovanni) ----------
    J = {}
    for l in open(OUT / "judge_pin_vs_add.jsonl"):
        x = json.loads(l)
        if x.get("error") is None:
            J[x["key"]] = x["persona_kind"]
    for route in ["assistant>vampire", "assistant>bard"]:
        A, B, cA, cB = G19.route(route)
        slug = route.replace(">", "-")
        d = pd.read_csv(O / "pin-vs-add-25_09" / f"responses_{slug}.csv")
        X = np.load(O / "pin-vs-add-25_09" / f"acts_post_{slug}.npy")
        d["pk"] = ["%s" % J.get("%s|%s|%g|%d|%d" % (x.route, x.method, x.strength, x.q, x["sample"])) for _, x in d.iterrows()]
        d["coll"] = [collapsed(t) for t in d.response.fillna("")]
        R = dict(src=G19.ix[A], tgt=G19.ix[B], path=chord(S19, G19, A, B, 2.0), methods={})
        for meth in ["pin", "add"]:
            m = (d.method == meth).to_numpy()
            dm = d[m]

            def ex(a, dm=dm):
                g = dm[dm.strength == a]
                return dict(tgt=float(((g.pk == B) & ~g.coll).mean()), src=float(((g.pk == A) & ~g.coll).mean()),
                            coll=float(g.coll.mean()))
            R["methods"][meth] = series(S19, X[m], dm.strength.to_numpy(), sorted(dm.strength.unique()), cB, ex)
        R["examples"] = [dict(method=x.method, a=float(x.strength), q=int(x.q), text=str(x.response)[:400],
                              pk="collapsed" if collapsed(x.response) else x.pk)
                         for _, x in d[(d.q == 0) & (d["sample"] == 0)].iterrows()]
        # Theoretical steered state if the model's own activation stayed at the unsteered mean h0:
        #   add: h0 + a*(B-A)          -> position along A->B = p0 + a
        #   pin: h0 + (s - p0)*(B-A)   -> position along A->B = s
        dAB = cB - cA
        h0 = X[(d.method == "add").to_numpy() & (d.strength == 0).to_numpy()].astype(np.float64).mean(0)
        p0 = float((h0 - cA) @ dAB / (dAB @ dAB))
        ss = sorted(d.strength.unique())
        R["p0"] = p0
        R["theory"] = {"add": [[float(a)] + r(S19.p(h0 + a * dAB)[0]) for a in ss],
                       "pin": [[float(a)] + r(S19.p(h0 + (a - p0) * dAB)[0]) for a in ss]}
        D["s1a"][route] = R

    # ---------- 1b / 2: Study A linear ----------
    ga = pd.read_csv(O / "steering-fix-24_09/studyA/generations.csv")
    JA = {}
    for l in open(O / "steering-fix-24_09/judge/judgements.jsonl"):
        x = json.loads(l)
        if x.get("error") is None:
            JA[x["key"]] = x["persona_kind"]
    XA = np.load(O / "steering-fix-24_09/studyA/steer_acts.npy")
    l32 = json.load(open(O / "steering-l19-to-l32-25_09/artifact_data.json"))
    rows32 = pd.read_csv(O / "steering-l19-to-l32-25_09/rows.csv")
    X19r, X32r = np.load(O / "steering-l19-to-l32-25_09/steer_L19.npy"), np.load(O / "steering-l19-to-l32-25_09/steer_L32.npy")
    for route in ["validator>vampire", "validator>bard", "assistant>vampire", "assistant>bard"]:
        A, B, cA, cB = G19.route(route)
        jc = pd.read_csv(O / f"steering-fix-24_09/figures/figC-judge-{route.replace('>', '-')}.csv")
        jl = jc[(jc.study == "A") & (jc.arm == "linear")].set_index("alpha")

        m = ((ga.route == route) & (ga.arm == "linear") & (ga.kind == "identity")).to_numpy()
        ex = shares(ga[m], "A", A, B, JA, jl)
        D["s1b"][route] = dict(src=G19.ix[A], tgt=G19.ix[B], path=chord(S19, G19, A, B, 4.0),
                               steps=series(S19, XA[m], ga.alpha.to_numpy()[m], sorted(ga.alpha[m].unique()), cB, ex),
                               examples=[dict(a=float(x.alpha), q=int(x.q_idx), text=str(x.response)[:400],
                                              pk="collapsed" if collapsed(x.response) else
                                              JA.get("A|%s|%d|%d" % (x.cell_id, int(x.unit_index), int(x["sample"])), "?"))
                                         for _, x in ga[m & (ga.q_idx == 0).to_numpy() & (ga["sample"] == 0).to_numpy()].iterrows()])
        h0b = XA[m & (ga.alpha == 0).to_numpy()].astype(np.float64).mean(0)
        D["s1b"][route]["theory"] = [[float(a)] + r(S19.p(h0b + a * (cB - cA))[0]) for a in sorted(ga.alpha[m].unique())]
        D["s1b"][route]["p0"] = float((h0b - cA) @ (cB - cA) / ((cB - cA) @ (cB - cA)))
        # section 2: L19 and L32 from the replay (same answers)
        mm = ((rows32.route == route) & (rows32.arm == "linear")).to_numpy()
        al = rows32.alpha.to_numpy()[mm]
        A32, B32 = G32.c(A), G32.c(B)
        D["s2"][route] = dict(src=G19.ix[A], tgt=G19.ix[B],
                              path19=chord(S19, G19, A, B, 4.0),
                              path32=r(S32.p(A32[None] + np.linspace(0, 4, 21)[:, None] * (B32 - A32)[None]).ravel().tolist()),
                              L19=series(S19, X19r[mm], al, sorted(set(al)), cB),
                              L32=series(S32, X32r[mm], al, sorted(set(al)), B32),
                              chord19=float(np.linalg.norm(cB - cA)), chord32=float(np.linalg.norm(B32 - A32)))

        # ---------- 3a: Study B linear_B (start = per-question mean) ----------
        gb = pd.read_csv(O / "steering-fix-24_09/studyB/generations.csv")
        XB = np.load(O / "steering-fix-24_09/studyB/steer_acts.npy")
        jb = jc[(jc.study == "B") & (jc.arm == "linear_B")].set_index("alpha")
        h0q = np.load(O / f"steering-fix-24_09/studyA/h0q_{route.replace('>', '-')}.npy")[:5].mean(0)

        mb = ((gb.route == route) & (gb.arm == "linear_B") & (gb.kind == "identity")).to_numpy()
        exb = shares(gb[mb], "B", A, B, JA, jb)
        D["s3a"][route] = dict(src=G19.ix[A], tgt=G19.ix[B], start=r(S19.p(h0q)[0]),
                               path=chord(S19, G19, A, B, 1.5, start=h0q),
                               steps=series(S19, XB[mb], gb.alpha.to_numpy()[mb], sorted(gb.alpha[mb].unique()), cB, exb))

    # ---------- 3b: per-answer start (validator>vampire) ----------
    pa = pd.read_csv(O / "steering-per-answer-25_09/results_rows.csv")
    XP = np.load(O / "steering-per-answer-25_09/steer_acts.npy")
    A, B, cA, cB = G19.route("validator>vampire")
    m = (pa.arm.isin(["unsteered", "add_linear"])).to_numpy()
    pp = pa[m]
    al = np.where(pp.arm == "unsteered", 0.0, pp.alpha)
    h0s = [np.load(O / f"steering-per-answer-25_09/h0_q{q}_s{s}.npy") for q in range(2) for s in range(3)]
    h0m = np.mean(h0s, axis=0)

    pc = np.array([collapsed(t) for t in pp.response.fillna("")])

    def exp(a):
        k = al == a
        g, c = pp[k], pc[k]
        return dict(tgt=float((g.tgt.to_numpy() & ~c).mean()), src=float((g.src.to_numpy() & ~c).mean()),
                    coll=float(c.mean()))
    D["s3b"]["validator>vampire"] = dict(src=G19.ix[A], tgt=G19.ix[B], start=r(S19.p(h0m)[0]),
                                         starts=[r(S19.p(h)[0]) for h in h0s],
                                         path=chord(S19, G19, A, B, 1.5, start=h0m),
                                         steps=series(S19, XP[m], al, sorted(set(al)), cB, exp))
    D["questions"] = INTROSPECTIVE_QUESTIONS
    (OUT / "story_data.json").write_text(json.dumps(D))
    print("wrote", OUT / "story_data.json", "%.0f KB" % ((OUT / "story_data.json").stat().st_size / 1e3))
    for k in ("s1a", "s1b", "s2", "s3a", "s3b"):
        print(k, list(D[k].keys()))


if __name__ == "__main__":
    main()
