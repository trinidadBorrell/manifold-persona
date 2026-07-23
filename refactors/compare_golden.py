"""Compare a manifold.run output dir against the step-0 golden.

Usage:  .venv/bin/python refactors/compare_golden.py output/manifold_h1-2/<stamp>

Exit 0 = identical under the plan's golden policy, exit 1 = differs.
Policy (refactors/2026-07-23-selfcontained-one-command.md, "Safety net"):
  metrics.csv           byte-identical
  manifolds_C_role.npz  bit-exact (array_equal)
  REPORT.md             identical after normalizing the timestamp
  manifest.json         identical after dropping started/finished/elapsed_sec/run_id
  figures               filename set only, NOT bytes (matplotlib is not byte-reproducible)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

GOLDEN = Path("refactors/baselines/2026-07-23-selfcontained-one-command/golden")
VOLATILE = {"started", "finished", "elapsed_sec", "run_id"}
STAMP = re.compile(r"2026-\d\d-\d\dT\d\d-\d\d")


def main(cand: Path) -> int:
    fails = []

    a = (GOLDEN / "data" / "metrics.csv").read_bytes()
    b = (cand / "data" / "metrics.csv").read_bytes()
    print(f"metrics.csv           {'IDENTICAL' if a == b else 'DIFFER'}")
    if a != b:
        fails.append("metrics.csv")

    za = np.load(GOLDEN / "data" / "manifolds_C_role.npz")
    zb = np.load(cand / "data" / "manifolds_C_role.npz")
    if sorted(za.files) != sorted(zb.files):
        fails.append("npz keys")
        print("manifolds_C_role.npz  KEY SET DIFFERS")
    else:
        bad = [k for k in za.files if not np.array_equal(za[k], zb[k])]
        print(f"manifolds_C_role.npz  {'ALL %d BIT-EXACT' % len(za.files) if not bad else 'DIFFER: %s' % bad}")
        if bad:
            fails.append(f"npz {bad}")

    ra = STAMP.sub("<stamp>", (GOLDEN / "REPORT.md").read_text())
    rb = STAMP.sub("<stamp>", (cand / "REPORT.md").read_text())
    print(f"REPORT.md             {'IDENTICAL' if ra == rb else 'DIFFER'}")
    if ra != rb:
        fails.append("REPORT.md")

    ma = {k: v for k, v in json.loads((GOLDEN / "manifest.json").read_text()).items() if k not in VOLATILE}
    mb = {k: v for k, v in json.loads((cand / "manifest.json").read_text()).items() if k not in VOLATILE}
    print(f"manifest.json         {'IDENTICAL' if ma == mb else 'DIFFER'}")
    if ma != mb:
        for k in sorted(set(ma) | set(mb)):
            if ma.get(k) != mb.get(k):
                print(f"    {k}: {ma.get(k)!r} -> {mb.get(k)!r}")
        fails.append("manifest.json")

    fa = sorted(p.name for p in (GOLDEN / "figures").iterdir())
    fb = sorted(p.name for p in (cand / "figures").iterdir())
    print(f"figure set ({len(fb)})       {'IDENTICAL' if fa == fb else 'DIFFER'}")
    if fa != fb:
        print(f"    only in golden: {sorted(set(fa) - set(fb))}")
        print(f"    only in cand:   {sorted(set(fb) - set(fa))}")
        fails.append("figures")

    print("\nGOLDEN MATCH" if not fails else f"\nGOLDEN MISMATCH: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
