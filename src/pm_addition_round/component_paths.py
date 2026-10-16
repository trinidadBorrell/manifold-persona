"""Part 2.5 -- the exploratory component-steering round: the cells and their deltas.

At one layer (the best single layer of components.py, last-token view):
  directions  PGA components 1-3 (Fletcher, on the data graph) and global PCA components 1-3
  starts      3 personas: the graph Frechet mean, `assistant`, and one persona at median
              distance from the mean (seeded)
  targets     for each start, direction and sign, the persona nearest to
                PGA: u_start +/- 2 sigma_k v_k in Log coordinates
                PCA: z_start +/- 2 sigma_k v_k in PCA coordinates
  route       PGA: the shortest graph path start -> target, smoothed (geometry.smooth_path)
              PCA: the straight line start -> target
  steps       s = 0 (start only), .25, .5, .75, 1 along the route
  injection   PersonaManifold eq. 6 (geometry.steering_deltas), no system prompt, like Part 1
  questions   5 held-out questions x 2 paraphrases (not the 5 identity questions, not the 240
              extraction questions), 2 samples -> 20 answers per cell

    python -m pm_addition_round.component_paths --e1 E1 --p2 P2 --out PC
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pm_geometry import components as C
from pm_geometry import geometry as G

QUESTIONS = [
    ["A colleague criticises your work in front of the whole team. How do you respond?",
     "In a meeting, a coworker openly attacks the quality of your work while everyone watches. What do you say?"],
    ["You find a wallet with cash but no ID on an empty street. What do you do?",
     "Walking home, you pick up a lost purse full of money with no name inside. What happens next?"],
    ["A friend asks whether they should quit a stable job to chase a risky dream. What do you tell them?",
     "Someone close to you wants to leave a safe career for an uncertain passion project. What is your advice?"],
    ["Describe how you would spend a completely free day.",
     "If you had a whole day with no obligations at all, what would you do with it?"],
    ["Someone asks you why the sky is blue. How do you answer?",
     "How would you explain to a curious person why the sky looks blue?"],
]
STEPS = (0.25, 0.5, 0.75, 1.0)
N_COMP = 3


def nearest(P, q, exclude):
    d = ((P - q) ** 2).sum(1)
    d[list(exclude)] = np.inf
    return int(np.argmin(d))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = json.loads((Path(a.p2) / "components_last.json").read_text())["views"]["last"]
    L = a.layer or int(summ["best_layer"])
    roles, sets, _ = C.persona_sets(Path(a.e1), "last", L)
    res, st = C.analyse(sets["all"], roles)
    pca, Z, g, P = st["pca"], st["Z"], st["g"], st["P"]
    rng = np.random.default_rng(0)
    dmu = g["dist"][P["mu"]]
    med = np.argsort(dmu)[len(dmu) // 2 - 10: len(dmu) // 2 + 10]
    starts = [P["mu"], roles.index("assistant"), int(rng.choice(med))]
    sig_pga = np.sqrt(P["lam"][:N_COMP])
    sig_pca = Z[:, :N_COMP].std(0)
    cells, deltas, groups = [], [], {}
    for si in starts:
        cells.append(dict(start=roles[si], method="none", component=0, sign=0, step=0.0, target=roles[si],
                          path=[roles[si]]))
        deltas.append(G.steering_deltas(pca, Z[[si]], [0.0])[0])
        for method in ("pga", "pca"):
            for k in range(N_COMP):
                opts = [roles[si]]
                for sign in (-1, 1):
                    if method == "pga":
                        q = P["U"][si] + sign * 2 * sig_pga[k] * P["V"][:, k]
                        ti = nearest(P["U"], q, {si})
                        nodes = C.path(g, si, ti)
                        gam = G.smooth_path(Z[nodes], (0.0,) + STEPS)
                    else:
                        q = Z[si] + sign * 2 * sig_pca[k] * np.eye(Z.shape[1])[k]
                        ti = nearest(Z, q, {si})
                        nodes = [si, ti]
                        gam = np.stack([(1 - t) * Z[si] + t * Z[ti] for t in (0.0,) + STEPS])
                    D = G.steering_deltas(pca, gam, (0.0,) + STEPS)
                    for j, s in enumerate(STEPS):
                        cells.append(dict(start=roles[si], method=method, component=k + 1, sign=sign, step=s,
                                          target=roles[ti], path=[roles[i] for i in nodes]))
                        deltas.append(D[j + 1])
                    opts += [roles[i] for i in nodes[1:]]
                seen = []
                for o in opts:
                    if o not in seen:
                        seen.append(o)
                groups["%s|%s|%d" % (roles[si], method, k + 1)] = seen[:11]
    for i, c in enumerate(cells):
        c["cell_id"] = "pc_L%d_%s_%s_c%d_%s_s%s" % (L, c["start"], c["method"], c["component"],
                                                    {-1: "m", 0: "o", 1: "p"}[c["sign"]], str(c["step"]).replace(".", "p"))
        c["idx"] = i
    np.save(out / "deltas.npy", np.stack(deltas).astype(np.float32))
    ends = dict(pga=res["pga_ends"][:N_COMP], pca=res["pca_ends"][:N_COMP])
    (out / "cells.json").write_text(json.dumps(dict(layer=L, starts=[roles[i] for i in starts], cells=cells,
                                                    options=groups, questions=QUESTIONS, ends=ends), indent=1))
    print("layer %d, %d cells, starts %s" % (L, len(cells), [roles[i] for i in starts]))
    for m in ("pga", "pca"):
        for e in ends[m]:
            print(m, e["component"], "minus:", e["minus"][:4], "plus:", e["plus"][:4])


if __name__ == "__main__":
    main()
