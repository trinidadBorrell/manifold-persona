"""Additive and pinned steering as geometry, per layer.

Everything is a vector from the assistant, projected into the top three directions
of the five persona directions, so all layers share one frame.

  target     d_L                     where the persona's own centroid sits
  add step   alpha * d_L             what additive adds at this layer
  add total  sum of alpha * d_l      what has been added by the end of this layer
  pin        s * d_L                 where pin puts the state along the direction
"""
import json
import sys

import numpy as np

sys.path.insert(0, "attr_analysis")
from loader import load_five

T = ["ghost", "pirate", "vampire", "musician", "spy"]
LAYERS = (26, 27, 28, 29, 30)
ADD = [0.25, 0.5, 0.75, 1.0]
PIN = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
BEST_ADD = {"ghost": 1.0, "pirate": 0.25, "vampire": 1.0, "musician": 1.0, "spy": 1.0}
BEST_PIN = {"ghost": 5.0, "pirate": 2.0, "vampire": 3.0, "musician": 4.0, "spy": 5.0}

means, _p, _a = load_five()
out = {"personas": T, "layers": [str(L) for L in LAYERS],
       "add_alphas": ADD, "pin_scales": PIN,
       "best_add": BEST_ADD, "best_pin": BEST_PIN, "P": {}}

for t in T:
    D = {L: means[L][t] - means[L]["default"] for L in LAYERS}
    M = np.stack([D[L] for L in LAYERS])
    _, _, vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    E = vt[:3].T
    pr = lambda v: [round(float(x), 3) for x in (v @ E)]
    per = {}
    for L in LAYERS:
        cum = {a: pr(sum(a * D[l] for l in LAYERS if l <= L)) for a in ADD}
        per[str(L)] = {
            "target": pr(D[L]),
            "norm": round(float(np.linalg.norm(D[L])), 1),
            "add_step": {str(a): pr(a * D[L]) for a in ADD},
            "add_total": {str(a): cum[a] for a in ADD},
            "pin": {str(s): pr(s * D[L]) for s in PIN},
            "add_total_norm": {str(a): round(float(np.linalg.norm(
                sum(a * D[l] for l in LAYERS if l <= L))), 1) for a in ADD},
            "pin_norm": {str(s): round(float(s * np.linalg.norm(D[L])), 1) for s in PIN},
        }
    out["P"][t] = per

json.dump(out, open("attr_analysis/steer_viz.json", "w"))
v = out["P"]["ghost"]["30"]
print("wrote steer_viz.json")
print(f"  ghost L30: target {v['norm']}, add total at alpha 1 = "
      f"{v['add_total_norm']['1.0']}, pin s=5 = {v['pin_norm']['5.0']}")
