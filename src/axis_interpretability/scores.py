"""Step 0: every role's score on PCA 1-K and PGA 1-K (default K = 3) (Qwen3-8B, E1 view `last`, layer 28).

Same pipeline as pm_geometry.components.analyse (PCA99 -> kNN graph -> Fletcher PGA), without BSA.
Checks the 6 roles at each end against P2full/layers/last_L28.json.

    python -m axis_interpretability.scores --e1 $PM/E1 --p2 $PM/P2full --out $PM/AXI_L28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from pm_geometry import geometry as G
from pm_geometry.components import knn_graph, persona_sets, pga

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="last")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args(argv)
    K = a.k
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roles, sets, _ = persona_sets(Path(a.e1), a.view, a.layer)
    X = sets["all"]
    pca = G.PCA99.fit(X)
    Z = pca.to_pca(X)
    P = pga(Z, knn_graph(Z))
    # cosine of each role with each direction (Assistant Axis s2.2): PCA uses the mean-centred role
    # vector, PGA the log-map vector at the Frechet mean (zero for the mean itself)
    Xc = X - pca.mean
    pca_cos = (Xc @ pca.V[:K].T) / np.linalg.norm(Xc, axis=1, keepdims=True)
    un = np.linalg.norm(P["U"], axis=1, keepdims=True)
    pga_cos = np.divide(P["coords"][:, :K], un, out=np.zeros((len(Z), K)), where=un > 0)
    np.savez(out / "scores.npz", roles=np.array(roles), pca_score=Z[:, :K], pga_score=P["coords"][:, :K],
             pca_cos=pca_cos, pga_cos=pga_cos,
             # the no-system-prompt state is the origin of the persona vectors (h - h0); not a PGA graph node
             pca_noprompt=pca.to_pca(np.zeros((1, X.shape[1])))[0, :K], frechet_mean=roles[P["mu"]],
             pca_var_share=pca.var_ratio[:K] / pca.var_ratio.sum(), pga_var_share=P["lam"][:K] / P["lam"].sum())

    ref = json.loads((Path(a.p2) / "layers" / ("%s_L%d.json" % (a.view, a.layer))).read_text())
    check = {}
    for m, S in (("pca", Z), ("pga", P["coords"])):
        for k in range(min(K, len(ref["%s_ends" % m]))):     # P2full keeps the ends of directions 1-5
            minus = [roles[i] for i in np.argsort(S[:, k])[:6]]
            plus = [roles[i] for i in np.argsort(S[:, k])[::-1][:6]]
            r = ref["%s_ends" % m][k]
            check["%s%d" % (m, k + 1)] = minus == r["minus"] and plus == r["plus"]
    rho = spearmanr(Z[:, :K], P["coords"][:, :K])[0][:K, K:].tolist()
    rep = dict(ends_match_p2full=check, spearman_pca_i_pga_j=rho, frechet_mean=roles[P["mu"]])
    (out / "scores_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if not all(check.values()):
        raise SystemExit("ends differ from P2full")


if __name__ == "__main__":
    main()
