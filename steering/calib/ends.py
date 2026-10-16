"""Print the personas at both ends of the top Fletcher directions at one layer (for writing the E2
hypotheses before any E2 answer exists).

    python -m steering.calib.ends --e1b E1b --layer 28
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from steering.calib.cells import layer_geom


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1b", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args(argv)
    g, _ = layer_geom(a.e1b, a.layer)
    lam = g.P["lam"]
    out = {}
    for k in range(3):
        c = g.P["coords"][:, k]
        o = np.argsort(c)
        out[k + 1] = dict(var_share=float(lam[k] / lam.sum()), minus=[g.roles[i] for i in o[:a.n]],
                          plus=[g.roles[i] for i in o[::-1][:a.n]])
        print("direction %d (%.0f%%)\n  - end: %s\n  + end: %s" % (k + 1, 100 * out[k + 1]["var_share"],
                                                                 ", ".join(out[k + 1]["minus"]), ", ".join(out[k + 1]["plus"])))
    print(json.dumps(dict(K=g.K, mean_persona=g.roles[g.P["mu"]], d_star=g.idim["d_star"])))


if __name__ == "__main__":
    main()
