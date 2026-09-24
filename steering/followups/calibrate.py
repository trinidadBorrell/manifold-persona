"""Exp1 -- alpha calibration per route and direction (CPU).

Plan: plans/2026-09-24-steering-followups.md (Design, exp1; mentor point b).

WHY. Every pair-steering run so far swept alpha in [0, 1] of the chord P1-P0
and treated alpha = 1 as "arrived". It is only arrival if the model STARTED at
P0 = c_A. It does not: unsteered, it sits well off the chord. The affine
steering paper states the dose at which a direction reaches each centroid, so
this does the same: for a direction v and the measured start h0,

    alpha_neg = (c_A - h0) . v / ||v||^2      the alpha that reaches c_A's level
    alpha_pos = (c_B - h0) . v / ||v||^2      ... and c_B's

i.e. the alphas at which h0 + alpha*v has the same projection on v as each
centroid. Three directions:

    chord         v = P1 - P0 = c_B - c_A    (what every earlier run used)
    target        v = c_B - h0              (alpha_pos = 1 by construction)
    axis          v = Assistant Axis, unit  (alpha in activation units, since
                                             ||v|| = 1 -- NOT in chords)

plus three route descriptors: cos(P1-P0, c_B-h0), the perpendicular distance of
h0 from the chord's line, and ||c_B-h0|| / ||P1-P0|| (how many chords away the
target really is).

h0 SOURCES. The unsteered footprint exists only for the validator source
prompt (the dose and located runs). It is valid for BOTH validator routes,
because the unsteered state depends on the source prompt and the question, not
the target. For the assistant routes nothing exists before exp4's alpha=0
cells; lacking an h0 the row falls back to the source centroid and is FLAGGED
h0_source = 'centroid_fallback'. On that fallback alpha_neg = 0 on every
direction and the chord and target directions coincide, so those rows are
placeholders, not measurements. Re-run with `--h0 ROUTE=<run>/data/h0_<slug>.npy`
after exp4 stage A0 to replace them.

Usage:
    .venv/bin/python -m steering.followups.calibrate [--out-dir <dir>]
    .venv/bin/python -m steering.followups.calibrate [--out-dir <dir>] \
        --h0 'assistant>vampire=<run>/data/h0_assistant-vampire.npy'
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from steering.followups.common import GEOM_DEFAULT, OUT_ROOT, H0_SOURCES, ROUTES, Geom, load_h0
from steering.manifold_paths import chord_coords


def calibrate_route(G: Geom, route: str, h0: np.ndarray, h0_source: str, h0_n: int):
    """One row per direction for this route."""
    A, B, cA, cB = G.route(route)
    chord = cB - cA
    tgt = cB - h0
    u, r = chord_coords(h0[None, :], cA, cB)
    L = float(np.linalg.norm(chord))
    ntgt = float(np.linalg.norm(tgt))
    cos_ct = float(chord @ tgt / (L * ntgt)) if ntgt > 0 else float("nan")
    dirs = [("chord", chord), ("target", tgt)]
    if G.axis_unit is not None:
        dirs.append(("axis", G.axis_unit))
    rows = []
    for name, v in dirs:
        vv = float(v @ v)
        if vv <= 0:
            a_neg = a_pos = float("nan")
        else:
            a_neg = float((cA - h0) @ v / vv)
            a_pos = float((cB - h0) @ v / vv)
        rows.append(dict(
            route=route, source=A, target=B, direction=name,
            v_norm=float(np.sqrt(vv)), alpha_neg=a_neg, alpha_pos=a_pos,
            # push, in activation units, that alpha_pos asks for along v
            push_pos=abs(a_pos) * float(np.sqrt(vv)) if np.isfinite(a_pos) else float("nan"),
            chord_len=L, h0_t=float(u[0]), h0_perp=float(r[0]),
            d_h0_cA=float(np.linalg.norm(h0 - cA)), d_h0_cB=ntgt,
            cos_chord_target=cos_ct, target_over_chord=ntgt / L,
            h0_source=h0_source, h0_n=h0_n))
    return rows


def parse_h0_specs(specs):
    """'ROUTE=path.npy[,path.csv]' or 'ROUTE=@name' (a key of H0_SOURCES)."""
    out = {}
    for s in specs or []:
        route, _, rhs = s.partition("=")
        if route not in ROUTES:
            raise SystemExit("--h0 route %r not in %s" % (route, ROUTES))
        if rhs.startswith("@"):
            npy, csv = H0_SOURCES[rhs[1:]]
            out[route] = (rhs[1:], npy, csv)
        else:
            npy, _, csv = rhs.partition(",")
            out[route] = (Path(npy).name, Path(npy), Path(csv) if csv else None)
    return out


def default_h0_specs():
    """Validator routes: the dose run (the file the plan names). Assistant: none."""
    npy, csv = H0_SOURCES["dose_L19"]
    return {r: ("dose_L19", npy, csv) for r in ROUTES if r.startswith("validator>")}


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["route", "direction", "alpha_neg", "alpha_pos", "push_pos", "cos_chord_target",
            "h0_perp", "target_over_chord", "h0_t", "h0_source"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(("%.3f" % v) if isinstance(v, float) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geom", default=str(GEOM_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_ROOT / "exp1_calibration"))
    ap.add_argument("--h0", action="append", default=None,
                    help="ROUTE=file.npy[,file.csv] or ROUTE=@dose_L19|@located_L19_temp07; "
                         "repeatable; overrides the default for that route")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    a = ap.parse_args(argv)

    G = Geom(a.geom)
    specs = default_h0_specs()
    specs.update(parse_h0_specs(a.h0))
    rows = []
    for route in ROUTES:
        if route in specs:
            name, npy, csv = specs[route]
            h0, n = load_h0(npy, csv)
            src = name
        else:
            h0, n, src = G.c(route.split(">")[0]), 0, "centroid_fallback"
        rows += calibrate_route(G, route, h0, src, n)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number")
    if not np.isfinite(num.to_numpy()).all():
        raise SystemExit("non-finite calibration value -- exp1 success criterion fails")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = "exp1_calibration" + (("_" + a.tag) if a.tag else "")
    df.to_csv(out / (stem + ".csv"), index=False)
    md = ["# Exp1 -- alpha calibration (hidden state 19)", "",
          "geometry: `%s`" % a.geom, "",
          "alpha is in units of the direction v; `axis` has ||v|| = 1, so its alphas are "
          "activation units. `push_pos` = |alpha_pos|*||v||. Rows flagged "
          "`centroid_fallback` use h0 = c_A and are placeholders.", "", to_markdown(df), ""]
    (out / (stem + ".md")).write_text("\n".join(md))
    print(to_markdown(df))
    print("\nwrote %s/%s.{csv,md}" % (out, stem))
    return df


if __name__ == "__main__":
    main()
