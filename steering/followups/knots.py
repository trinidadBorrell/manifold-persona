"""Exp3 -- waypoint rules compared (CPU).

Plan: plans/2026-09-24-steering-followups.md (Design, exp3; mentor point a).

For each route, each knot rule (distance, density, nearest) and each k in
{4, 8, 16}: which personas were picked, how much the pick overlaps the nearest
rule's (Jaccard), how close the knots sit to the segment, how long the route is
relative to the chord (detour), how much of that is spline overshoot, and
whether the curve actually passes through its knots (knot error; success
criterion < 1e-8).

TWO FAMILIES OF ROUTES.
  Study A  P0 = c_A (source centroid) -> P1 = c_B.
  Study B  P0 = h0 (measured unsteered footprint) -> P1 = c_B.
Study B in the real run is per question (h0(q)); here it is the route-mean h0,
which is what exists before exp4. For the assistant routes no h0 exists yet, so
their Study B rows use the source centroid and are flagged
h0_source = 'centroid_fallback' -- identical to Study A by construction, and
listed only so the table has every cell the plan names.

Usage:
    .venv/bin/python -m steering.followups.knots [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from steering.followups.calibrate import default_h0_specs, parse_h0_specs
from steering.followups.common import GEOM_DEFAULT, OUT_ROOT, ROUTES, Geom, load_h0
from steering.followups.selection import (KNOT_RULES, build_manifold, endpoint_exclusions,
                                          mean_segment_distance)
from steering.manifold_paths import chord_coords

KS = (4, 8, 16)
KNOT_ERR_MAX = 1e-8


def jaccard(a, b):
    a, b = set(a), set(b)
    return float(len(a & b) / len(a | b)) if (a | b) else 1.0


def route_rows(G, route, study, P0, P1, h0_source):
    ex = endpoint_exclusions(G.C, P0, P1)
    rows = []
    for k in KS:
        picks = {}
        for rule in KNOT_RULES:
            p = build_manifold(rule, P0, P1, G.C, k, exclude_idx=ex)
            idx = [int(i) for i in p.centroid_idx]
            u, _ = chord_coords(G.C[idx], P0, P1) if idx else (np.zeros(0), None)
            picks[rule] = idx
            rows.append(dict(
                study=study, route=route, rule=rule, k=k, n_knots=len(idx),
                knots=" ".join(G.names[i] for i in idx),
                mean_dist_segment=mean_segment_distance(G.C, idx, P0, P1),
                detour=p.detour_ratio, polyline=p.polyline_ratio, overshoot=p.overshoot,
                knot_error=p.knot_error(),
                n_outside=int(((u < 0) | (u > 1)).sum()),
                chord_len=float(np.linalg.norm(P1 - P0)), h0_source=h0_source))
        for r in rows[-len(KNOT_RULES):]:
            r["jaccard_vs_nearest"] = jaccard(picks[r["rule"]], picks["nearest"])
    return rows


def to_markdown(df):
    cols = ["study", "route", "k", "rule", "n_knots", "jaccard_vs_nearest",
            "mean_dist_segment", "detour", "overshoot", "knot_error", "n_outside"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        cells = []
        for c, v in zip(cols, r.values):
            if c == "knot_error":
                cells.append("%.1e" % v)
            elif isinstance(v, float):
                cells.append("%.3f" % v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geom", default=str(GEOM_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_ROOT / "exp3_knots"))
    ap.add_argument("--h0", action="append", default=None,
                    help="as in calibrate.py: ROUTE=file.npy[,file.csv] or ROUTE=@name")
    a = ap.parse_args(argv)

    G = Geom(a.geom)
    specs = default_h0_specs()
    specs.update(parse_h0_specs(a.h0))
    rows = []
    for route in ROUTES:
        _, _, cA, cB = G.route(route)
        rows += route_rows(G, route, "A", cA, cB, "n/a")
    for route in ROUTES:
        _, _, cA, cB = G.route(route)
        if route in specs:
            name, npy, csv = specs[route]
            h0, _ = load_h0(npy, csv)
        else:
            h0, name = cA, "centroid_fallback"
        rows += route_rows(G, route, "B", h0, cB, name)
    df = pd.DataFrame(rows)
    ok = bool((df.knot_error < KNOT_ERR_MAX).all())
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "exp3_knots.csv", index=False)
    names = df[["study", "route", "k", "rule", "knots"]]
    md = ["# Exp3 -- waypoint rules (hidden state 19)", "",
          "geometry: `%s`. success criterion knot_error < %g: **%s**" % (a.geom, KNOT_ERR_MAX,
                                                                        "PASS" if ok else "FAIL"),
          "", to_markdown(df), "", "## Picked personas (ascending chord coordinate)", "",
          "| study | route | k | rule | knots |", "|---|---|---|---|---|"]
    md += ["| %s | %s | %d | %s | %s |" % tuple(r) for r in names.itertuples(index=False)]
    (out / "exp3_knots.md").write_text("\n".join(md) + "\n")
    print(to_markdown(df))
    print("\nknot_error < %g on all %d cells: %s" % (KNOT_ERR_MAX, len(df), ok))
    print("wrote %s/exp3_knots.{csv,md}" % out)
    if not ok:
        raise SystemExit(1)
    return df


if __name__ == "__main__":
    main()
