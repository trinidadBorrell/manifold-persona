"""Propose step 2's ten routes; writes <root>/routes_step2_candidates.json.

Five routes run from assistant-like roles toward non-assistant ones, and five are
the SAME pairs reversed. The first three are step 1's (assistant>bard, >vampire,
>leviathan). The other two are chosen by rule:

  source   top 10% of the Assistant Axis projection at L19, not `assistant`
  target   bottom 10%, not bard / vampire / leviathan, not already a target
  both     present at every sweep layer, and the manifold_v1 route has 8 knots at
           every sweep layer (no empty chunk)

Candidates are ranked by how often the pair's endpoints keep their axis order
across layers, then by chord length at L19 (longer = cleaner separation). The
user confirms the pick; this script only proposes. Once confirmed, copy the
chosen list to routes_step2.json.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from steering.ace_study.common import K, KNOT_RULE, OUT_ROOT, STEP1_ROUTES, SWEEP_LAYERS, geom_path
from steering.followups.common import Geom
from steering.followups.selection import build_manifold


def main():
    G = {L: Geom(geom_path(L)) for L in SWEEP_LAYERS}
    ap = {L: dict(zip(G[L].names, np.load(G[L].path, allow_pickle=True)["axis_proj"]))
          for L in SWEEP_LAYERS}
    base = G[19]
    names = [n for n in base.names if all(n in G[L].ix for L in SWEEP_LAYERS)]
    a19 = np.array([ap[19][n] for n in names])
    hi, lo = np.percentile(a19, 90), np.percentile(a19, 10)
    used = {r.split(">")[1] for r in STEP1_ROUTES}
    sources = [n for n, v in zip(names, a19) if v >= hi and n not in ("assistant",)]
    targets = [n for n, v in zip(names, a19) if v <= lo and n not in used]
    cands = []
    for s, t in itertools.product(sources, targets):
        ok, order = True, 0
        for L in SWEEP_LAYERS:
            _, _, cA, cB = G[L].route("%s>%s" % (s, t))
            n_f = build_manifold(KNOT_RULE, cA, cB, G[L].C, K).n_centroids
            n_b = build_manifold(KNOT_RULE, cB, cA, G[L].C, K).n_centroids
            if n_f != K or n_b != K:
                ok = False
                break
            order += int(ap[L][s] > ap[L][t])
        if ok:
            cands.append(dict(route="%s>%s" % (s, t), axis_order_layers=order,
                              chord_L19=float(np.linalg.norm(base.c(t) - base.c(s))),
                              axis_L19=[float(ap[19][s]), float(ap[19][t])]))
    cands.sort(key=lambda c: (-c["axis_order_layers"], -c["chord_L19"]))
    # at most one route per source and per target, so the two picks are unrelated
    pick, seen = [], set()
    for c in cands:
        s, t = c["route"].split(">")
        if s in seen or t in seen:
            continue
        pick.append(c["route"])
        seen |= {s, t}
        if len(pick) == 2:
            break
    fwd = STEP1_ROUTES + pick
    routes = fwd + [">".join(r.split(">")[::-1]) for r in fwd]
    out = dict(routes=routes, rule=__doc__.strip().splitlines()[0], candidates=cands[:20])
    (OUT_ROOT / "routes_step2_candidates.json").write_text(json.dumps(out, indent=1))
    print("sources (top 10%%): %s" % sources)
    print("targets (bottom 10%%): %s" % targets)
    for c in cands[:10]:
        print("  %-24s axis order kept at %d/7 layers  chord L19 %.1f" % (c["route"], c["axis_order_layers"], c["chord_L19"]))
    print("proposed step-2 routes:", routes)


if __name__ == "__main__":
    main()
