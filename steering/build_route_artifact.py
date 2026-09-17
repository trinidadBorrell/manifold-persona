"""Rebuild the route-results artifact's data blob from a route-judged parquet.

Plan: plans/2026-09-08-steering-rebuild.md.

The page's layout, PCA geometry (`const G`) and drawing code are stable; only
`const R` -- the judged position mix per route, arm and alpha -- changes when
the judge is re-run. This script rewrites that one line in place, plus the
handful of prose spans whose wording is tied to the option list, and leaves
everything else byte-for-byte.

POSITIONS. A verdict is mapped to where it sits on the route, not kept as a
persona name: `A` (the prompted source), `waypoint`, `B` (the steering target),
or `other`. A is tested before the waypoints because a route may thread a
centroid that is also its own endpoint, and the endpoint reading is the one
that means something.
"""
from __future__ import annotations

import argparse
import json
import re

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from steering import route_judge as RJ

POSITIONS = ["A", "waypoint", "B", "other"]


def position(label, a, waypoints, b):
    if label == a:
        return "A"
    if label == b:
        return "B"
    if label in waypoints:
        return "waypoint"
    return "other"


def frac(label, a, waypoints, b):
    """Arc fraction of a positional label, or None for `other`."""
    seq = [a] + list(waypoints) + [b]
    if label in seq:
        return seq.index(label) / (len(seq) - 1)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parquet", default="route_judged_v2_L19.parquet")
    ap.add_argument("--html", required=True, help="artifact to rewrite in place")
    ap.add_argument("--audit", default="",
                    help="per-alpha audited counts [A, waypoint, B, other] per route "
                         "and arm; drawn alongside the judge's labels where present")
    ap.add_argument("--max-example-chars", type=int, default=900)
    args = ap.parse_args()

    data = Path(args.data)
    df = pd.read_parquet(data / args.parquet)
    manifest = json.loads((data / "path_manifest.json").read_text())
    wps = {(r["A"], r["B"]): list(r["manifold"]["centroid_roles"])
           for r in manifest["routes"] if r["kind"] == "pair"}

    alphas = sorted(df["alpha"].unique().tolist())
    routes = {}
    for (a, b), sub in df.groupby(["path_a", "path_b"]):
        w = wps[(a, b)]
        sub = sub.copy()
        sub["pos"] = [position(s, a, w, b) for s in sub["route_score"]]
        sub["frac"] = [frac(s, a, w, b) for s in sub["route_score"]]
        arms, examples = {}, []
        for arm, asub in sub.groupby("arm"):
            cells = {}
            for al, cell in asub.groupby("alpha"):
                n = len(cell)
                mix = {p: round(float((cell["pos"] == p).sum()) / n, 4)
                       for p in POSITIONS}
                on = cell["frac"].dropna()
                cells[("%g" % al)] = {
                    "mix": mix,
                    "mean_frac": round(float(on.mean()), 4) if len(on) else None,
                    "n": int(n)}
                # One verbatim response per distinct verdict, so the reader can
                # check the judge rather than take the percentages on trust.
                for lbl, g in cell.groupby("route_score"):
                    # Prefer a row that still has the judge's reasoning: a
                    # parser bug dropped the analysis text (not the verdict) on
                    # some rows, and an example without it is worth less.
                    with_an = g[g["route_analysis"].fillna("").str.len() > 0]
                    r0 = (with_an if len(with_an) else g).iloc[0]
                    examples.append({
                        "arm": arm, "alpha": float(al), "label": str(lbl),
                        "pos": position(lbl, a, w, b),
                        "question": r0["question"],
                        "response": r0["response"][:args.max_example_chars],
                        "analysis": (r0["route_analysis"] or "")[:400]})
            arms[arm] = cells
        # The instrument itself, verbatim, so a reader can check what the
        # judge was asked rather than trust a paraphrase of it. Built by the
        # same call the run used, from the same waypoints.
        demo = sub[sub["alpha"] == max(alphas)].iloc[0]
        routes["%s->%s" % (a, b)] = {
            "A": a, "B": b, "seq": [a] + w + [b],
            "arms": arms, "examples": examples,
            "prompt": {
                "system": RJ.build_system(a, w, b),
                "user": RJ.USER_TEMPLATE.format(question=demo["question"],
                                                response=demo["response"]),
                "verdict": str(demo["route_score"]),
                "model": RJ.JUDGE_MODEL}}

    # The audited reading, where one exists. It is NOT a correction applied to
    # the judge's numbers -- both are kept, and the page draws whichever the
    # reader selects, so the disagreement stays visible instead of being
    # silently resolved in one direction.
    if args.audit:
        audit = json.loads(Path(args.audit).read_text())
        for key, arms_in in audit.items():
            if key not in routes:
                raise SystemExit("audit names unknown route: %s" % key)
            out = {}
            for arm, cells in arms_in.items():
                out[arm] = {}
                for al, counts in cells.items():
                    n = sum(counts)
                    out[arm][al] = {
                        "mix": dict(zip(POSITIONS, [round(c / n, 4) for c in counts])),
                        "n": n}
            routes[key]["audit_arms"] = out

    R = {"alphas": alphas, "positions": POSITIONS, "routes": routes,
         "ambiguous_routes": [], "n_total": int(len(df)),
         "n_clean": int(df["route_score"].notna().sum())}

    html = Path(args.html).read_text()
    def put(pattern, text):
        nonlocal html
        html, n = re.subn(pattern, lambda _m: text, html, count=1, flags=re.M)
        if n != 1:
            raise SystemExit("pattern not found in html: %s" % pattern)

    put(r"^const R = .*$", "const R = " + json.dumps(R) + ";")
    put(r"^const POS=.*$", "const POS=" + json.dumps(POSITIONS) + ";")
    put(r"^const PCOL=.*$",
        'const PCOL={A:"--src",waypoint:"--way",B:"--tgt",other:"--other"};')
    Path(args.html).write_text(html)

    print("routes: %d   rows: %d   scored: %d" % (len(routes), R["n_total"], R["n_clean"]))
    for k, rt in routes.items():
        for arm, cells in rt["arms"].items():
            avg = {p: sum(c["mix"][p] for c in cells.values()) / len(cells)
                   for p in POSITIONS}
            print("  %-22s %-14s " % (k, arm.replace("_pair", "")) +
                  "  ".join("%s %5.1f%%" % (p, 100 * avg[p]) for p in POSITIONS))


if __name__ == "__main__":
    main()
