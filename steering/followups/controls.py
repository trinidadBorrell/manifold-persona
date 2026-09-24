"""Compute the plan's positive/negative controls from the judged CSVs (audit 2026-09-24).

Plan, Controls:
  positive  target_vector at alpha=1: target share >= 0.20 on >= 1 route with canary >= 0.8
  negative  every arm at alpha=0: target share <= 0.05
Writes <run>/controls.csv and exits non-zero if either fails.

    CONDOR_GUARD_OK=1 .venv/bin/python -m steering.followups.controls
"""
import glob
import sys
from pathlib import Path

import pandas as pd

from steering.followups.common import OUT_ROOT


def main():
    root = Path(OUT_ROOT)
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(root / "figures" / "figC-judge-*.csv"))])
    pos = df[(df.study == "A") & (df.arm == "target_vector") & (df.alpha == 1.0)]
    pos_ok = bool(((pos.share_target >= 0.20) & (pos.canary_acc >= 0.8)).any())
    neg = df[df.alpha == 0.0]
    neg_ok = bool((neg.share_target <= 0.05).all())
    out = pd.DataFrame([
        dict(control="positive", rule="target_vector a=1: target>=0.20 & canary>=0.8 on >=1 route",
             value=";".join("%s=%.2f/canary %.2f" % (r.route, r.share_target, r.canary_acc)
                            for r in pos.itertuples()), passed=pos_ok),
        dict(control="negative", rule="all arms a=0: target<=0.05",
             value="max %.2f over %d cells" % (neg.share_target.max(), len(neg)), passed=neg_ok)])
    out.to_csv(root / "controls.csv", index=False)
    print(out.to_string(index=False))
    sys.exit(0 if (pos_ok and neg_ok) else 1)


if __name__ == "__main__":
    main()
