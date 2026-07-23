"""Run the assistant-axis exploratory stage into one dated folder + REPORT.md."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import timestamp, FIGURES_DIR

HERE = Path(__file__).resolve().parent
PY = sys.executable
CLUSTER_COL = "kmeans_pca95"


def run(script, extra, env):
    print(f"\n=== {script} {' '.join(extra)} ===")
    subprocess.run([PY, str(HERE / script), "--view", env["_VIEW"]] + extra, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    stamp = args.stamp or timestamp()
    run_dir = FIGURES_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env["MP_RUN_DIR"] = str(run_dir); env["_VIEW"] = args.view
    print(f"Results -> {run_dir}")

    run("01_intrinsic_dimension.py", [], env)
    run("02_clustering.py", [], env)
    run("03_umap_axis.py", ["--cluster_col", CLUSTER_COL], env)
    run("04_role_families.py", ["--cluster_col", CLUSTER_COL], env)
    run("05_role_map.py", [], env)
    run("06_cluster_maps.py", [], env)
    run("make_report.py", ["--cluster_col", CLUSTER_COL], env)
    print(f"\nDone. See {run_dir}/REPORT.md")


if __name__ == "__main__":
    main()
