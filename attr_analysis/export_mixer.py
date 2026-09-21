"""Mixer data: every persona we have, oracle fitted per layer."""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import LAYERS, load_all  # noqa: E402

K, SHRINK = 24, 0.25
means, pts, personas = load_all()

out = {"k": K, "layers": [str(L) for L in LAYERS], "personas": personas,
       "names": [p for p in personas if p != "default"], "L": {}}
for L in LAYERS:
    a = means[L]["default"]
    M = np.stack([means[L][p] - a for p in personas])
    _, _, vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    E, V3 = vt[:K].T, vt[:3].T
    G = {}
    for p in personas:
        Y = (pts[L][p] - a) @ E
        S = np.cov(Y.T)
        S = (1 - SHRINK) * S + SHRINK * np.trace(S) / K * np.eye(K)
        G[p] = {"mu": np.round(Y.mean(0), 4).tolist(),
                "Si": np.round(np.linalg.inv(S), 5).tolist(),
                "logdet": float(np.linalg.slogdet(S)[1])}
    out["L"][str(L)] = {
        "dirs": {p: np.round((means[L][p] - a) @ E, 4).tolist()
                 for p in personas if p != "default"},
        "gauss": G,
        "view": {p: np.round((means[L][p] - a) @ V3, 3).tolist() for p in personas},
        "E3": np.round(E.T @ V3, 5).tolist(),
    }
json.dump(out, open("attr_analysis/mixer_data.json", "w"))
print(f"{len(out['names'])} personas x {len(LAYERS)} layers -> mixer_data.json")
