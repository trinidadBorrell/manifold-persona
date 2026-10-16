"""Component steering with Giovanni's clamp on the PCA / PGA axes (cells from
pca_pga_steering.clamp_paths). Same as jobs_condor/pc_steer.py (addition round) except the hook: h <- h + (goal - h.u) u.

    python -m pca_pga_steering.steer_gpu --stage gen --pcc PCC --strength 4 --num-shards 34 --shard-index 0
    python -m pca_pga_steering.steer_gpu --stage collect --pcc PCC --strength 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ace_steering import steer_gpu as AS  # noqa: E402
from ace_steering.common import REP4_COLLAPSED, TEMPERATURE, TOP_K, TOP_P, seed_for  # noqa: E402
from steering_followups.common import degeneracy  # noqa: E402

N_SAMP = 2
MAX_NEW = 384


def stage_gen(a):
    spec = json.loads((Path(a.pcc) / "cells.json").read_text())
    axes = np.load(Path(a.pcc) / "axes.npy").astype(np.float64)
    L, S = spec["layer"], str(float(a.strength))
    out = Path(a.pcc) / ("S%s" % S.replace(".0", "")) / "cells"
    out.mkdir(parents=True, exist_ok=True)
    todo = spec["cells"][a.shard_index::a.num_shards]
    us = [(p, q, None, spec["questions"][q][p]) for _s in range(N_SAMP) for q in range(len(spec["questions"])) for p in range(2)]
    tok, model = AS.load_model(AS.MODEL)
    for i, c in enumerate(todo):
        stem = out / c["cell_id"]
        if AS.cell_done(stem):
            continue
        t0 = time.time()
        u = axes[c["axis"]]
        goal = float(c["goals"][S])
        res = dict(kind="pin", U=u[:, None], target=goal * u)        # his clamp along one axis
        seed = seed_for(c["start"], L, 0)                              # the PC round's seeds
        rows, acts, mag = AS.run_batch(tok, model, c["start"], us, res, L, seed, MAX_NEW)
        for j, r in enumerate(rows):
            dg = degeneracy(r["response"])
            r.update(cell_id=c["cell_id"], layer=L, start=c["start"], method=c["method"],
                     component=c["component"], sign=c["sign"], step=c["step"], strength=float(a.strength),
                     goal=goal, a=c["a"], d=c["d"], sample=j // (len(us) // N_SAMP), seed=seed,
                     temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, mean_push=mag,
                     collapsed=bool(dg["rep4"] > REP4_COLLAPSED), **{f"deg_{k}": v for k, v in dg.items()})
        np.save(str(stem) + ".steer.npy", np.stack(acts))
        AS.write_table(pd.DataFrame(rows), stem)
        print("[%d/%d] %s %.0fs push %.1f | %r" % (i + 1, len(todo), c["cell_id"], time.time() - t0,
                                                  mag or 0, rows[0]["response"][:80]), flush=True)


def stage_collect(a):
    spec = json.loads((Path(a.pcc) / "cells.json").read_text())
    S = str(float(a.strength)).replace(".0", "")
    d = Path(a.pcc) / ("S%s" % S)
    frames = []
    for c in spec["cells"]:
        stem = d / "cells" / c["cell_id"]
        if not AS.cell_done(stem):
            raise SystemExit("missing %s" % c["cell_id"])
        frames.append(AS.read_table(str(stem)))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(d / "generations.csv", index=False)
    print("%d rows -> %s" % (len(df), d / "generations.csv"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "collect"], required=True)
    ap.add_argument("--pcc", required=True)
    ap.add_argument("--strength", type=float, required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    a = ap.parse_args(argv)
    stage_gen(a) if a.stage == "gen" else stage_collect(a)


if __name__ == "__main__":
    main()
