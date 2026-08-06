"""Run the geometry-vs-axis study into one dated folder.

Same orchestration contract as `run_id_stage.py` and the other exploratory
stages: one fixed run dir, shared via `MP_RUN_DIR` and passed explicitly as
`--outdir`, with each script's stdout tee'd to `logs/`.

The order is not arbitrary:

  1. calib_estimators   the hard gate — planted manifolds of known dimension.
                        Exits non-zero on failure, which stops everything below,
                        so no result can quote an estimator that cannot measure
                        a dimension at this sample size.
  2. study_panel        the per-role table every later script reads.
  3. study_design_null  needs the real panel to compare its draws against.
  4. study_ladder       needs the design null for its DESIGN-EXPLAINED flags.
  5. study_regression   \
  6. study_families      >  independent of each other; panel is all they need.
  7. confound_variance  /
  8. figures            needs every data file above.

`confound_sysprompt.py` is deliberately NOT run here: it is post hoc, it writes
to its own run dir, and it takes a *completed* run as input.

Usage:
    MP_ROLE_DIR=data/embeddings_roles_resp40 \\
      .venv/bin/python exploratory/per_persona/run_geometry.py --layer 0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import FIGURES_DIR, timestamp

HERE = Path(__file__).resolve().parent
PY = sys.executable

# (script, wants). "cloud" = the script loads the raw per-example cloud and so
# takes --view/--layer; "n_null" = it also takes the null-draw count. Everything
# else works from the files an earlier step wrote and takes neither.
STEPS = [
    ("calib_estimators.py", {"cloud"}),
    ("study_panel.py", {"cloud"}),
    ("study_design_null.py", {"cloud", "n_null"}),
    ("study_ladder.py", set()),
    ("study_regression.py", set()),
    ("study_families.py", {"cloud"}),
    ("confound_variance.py", set()),
    ("figures.py", set()),
]


def run(script: str, extra: list, run_dir: Path, env: dict):
    print(f"\n=== {script} {' '.join(extra)} ===", flush=True)
    log = run_dir / "logs" / f"{Path(script).stem}.log"
    cmd = [PY, str(HERE / script), "--outdir", str(run_dir), *extra]
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        fh.write(proc.stdout)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SystemExit(f"{script} exited {proc.returncode} — see {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg")
    ap.add_argument("--layer", type=int, default=None,
                    help="layer INDEX in the stored cloud (resp40 keeps one "
                         "layer, so 0)")
    ap.add_argument("--label-layer", type=int, default=19,
                    help="layer number written into output FILENAMES (the real "
                         "depth behind resp40's primary_layer=0)")
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    run_dir = Path(args.outdir) if args.outdir else \
        FIGURES_DIR / (args.stamp or timestamp())
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MP_RUN_DIR"] = str(run_dir)
    print(f"Results -> {run_dir}")

    cloud_args = ["--view", args.view]
    if args.layer is not None:
        cloud_args += ["--layer", str(args.layer)]
    for script, wants in STEPS:
        extra = ["--label-layer", str(args.label_layer)]
        if "cloud" in wants:
            extra += cloud_args
        if "n_null" in wants:
            extra += ["--n-null", str(args.n_null)]
        run(script, extra, run_dir, env)

    print(f"\nDone. Data in {run_dir}/data, figures in {run_dir}/figures")


if __name__ == "__main__":
    main()
