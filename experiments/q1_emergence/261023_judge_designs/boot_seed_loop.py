"""How often the ex2 rule-1 lower bounds clear 20% across 50 bootstrap seeds (28/10/2026, exploratory).

Same cube, macro and crossed bootstrap as score_naming.py; only the bootstrap seed changes. The permutation p is
0.0001 in every saved run, so rule 1 turns on the lower bound alone.

    python boot_seed_loop.py   # from 261008_base_persona_map; writes ../261023_judge_designs/results/boot_seed_loop.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import score_naming as S  # noqa: E402

key, by_run, guesses = S.load()
mask = {tuple(x) for x in json.loads(Path("../261023_judge_designs/results/masks/loop_nonsense.json").read_text())}
out = {}
for run in ("run_main", "run_rep"):
    G, G1, F, donor, ids = S.cube(key, by_run, guesses, run, "ex2")
    inv = np.vectorize(lambda x: tuple(x) in mask, otypes=[bool])(ids)
    comp = np.broadcast_to(~inv.any(2, keepdims=True), inv.shape)
    clean = ~(F["copied"] | F["selfname"])
    keeps = dict(all=np.ones_like(inv), clean=clean, cc=comp & ~inv, cc_clean=comp & ~inv & clean)
    for j in G:
        H = S.target_hits(G[j])
        for name, keep in keeps.items():
            lbs = [S.boot(H, keep, np.random.default_rng(s))[0] for s in range(50)]
            out[f"{run}/{j}/{name}"] = dict(point=S.macro(H, keep), lb_min=min(lbs), lb_median=float(np.median(lbs)),
                                            lb_max=max(lbs), share_lb_ge_020=float(np.mean(np.array(lbs) >= 0.20)))
            print(f"{run} {j:6} {name:8} point {out[f'{run}/{j}/{name}']['point']:.3f}  lb {min(lbs):.3f}-{max(lbs):.3f}"
                  f"  median {np.median(lbs):.3f}  share>=0.20 {np.mean(np.array(lbs) >= 0.20):.2f}")
Path("../261023_judge_designs/results/boot_seed_loop.json").write_text(json.dumps(out, indent=1))
