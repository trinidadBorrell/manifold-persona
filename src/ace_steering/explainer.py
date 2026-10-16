"""Build the explanation page for the four steering arms -> <root>/steering_arms.html.

A toy (drawn in the browser from the formulas) plus step 1's measured numbers:
where each arm put the steered state along A -> B, how far it ended from the
target centroid, and how often answers collapsed.

    python -m ace_steering.explainer
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ace_steering.common import OUT_ROOT

HERE = Path(__file__).resolve().parent


def main():
    d = pd.read_csv(OUT_ROOT / "step1" / "generations.csv")
    none = d[d.arm == "none"]
    rows = []
    for arm in ("linear", "manifold", "ace", "ace_manifold"):
        g = d[d.arm == arm]
        if g.empty:
            continue
        if arm in ("linear", "manifold"):
            g = pd.concat([g, none.assign(arm=arm)])        # unsteered = these arms at alpha 0
        for al, h in g.groupby("alpha"):
            rows.append(dict(arm=arm, alpha=float(al), pos=round(float(h.chord_t.mean()), 3),
                             dtgt=round(float(h.d_tgt.mean()), 2),
                             coll=round(float(h.collapsed.mean()), 3), n=int(len(h))))
    data = dict(rows=rows, unsteered_pos=round(float(none.chord_t.mean()), 3),
                unsteered_dsrc=round(float(none.d_src.mean()), 1))
    tpl = (HERE / "explainer_template.html").read_text()
    out = OUT_ROOT / "steering_arms.html"
    out.write_text(tpl.replace("__DATA__", json.dumps(data, allow_nan=False).replace("</", "<\\/")))
    print("explainer -> %s (%d rows; arms %s)" % (out, len(rows), sorted({r["arm"] for r in rows})))


if __name__ == "__main__":
    main()
