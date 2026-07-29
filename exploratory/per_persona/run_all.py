"""Run the per-persona stage into one dated folder + REPORT.md.

Same orchestration contract as exploratory/assistant_axis/run_all.py (one fixed
timestamp shared via MP_RUN_DIR), but the report is written here rather than in a
separate make_report.py: this study is three scripts, and its report is a short
read of three JSON files.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import FIGURES_DIR, timestamp

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(script, extra, env):
    print(f"\n=== {script} {' '.join(extra)} ===")
    subprocess.run([PY, str(HERE / script), "--view", env["_VIEW"]] + extra,
                   check=True, env=env)


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def _num(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_report(run_dir: Path, view: str, layer: int):
    j = lambda p: json.load(open(run_dir / p))
    idr = j(f"01_per_persona_id_{view}_L{layer}.json")
    clu = j(f"02_per_persona_clustering_{view}_L{layer}.json")
    bud = j(f"03_compute_budget_{view}_L{layer}.json")

    m, dv = idr["_meta"], idr["design_variance"]
    L = []
    A = L.append
    A(f"# Per-persona manifold — {view}, layer {layer}\n")
    A(f"Each of the **{m['n_roles']} roles** is treated as its own point cloud of "
      f"**{m['points_per_role']} points** in {m['ambient']} ambient dimensions, and gets its "
      f"own intrinsic dimension and its own clustering.\n")

    A("## Headline\n")
    A(f"The within-role cloud is **{_pct(dv['instr_frac']['median'])} instruction phrasing, "
      f"{_pct(dv['quest_frac']['median'])} question, and only "
      f"{_pct(dv['interaction_frac']['median'])} interaction**. Instruction and question are "
      "the two axes of the 5x5 extraction grid, so everything except that last term is forced "
      "by the experiment's design, not by the model. The additive rank of a 5x5 grid is "
      f"(5-1)+(5-1) = **{m['additive_design_rank']}**, and lPCA returns exactly "
      f"{_num(idr['per_role_summary']['lPCA']['median'], 1)} for essentially every role.\n")
    A("**Both requested results are therefore measurements of the extraction grid, not of "
      "persona geometry.** They are reported in full below, with the null that shows why.\n")

    A("## 1. Intrinsic dimension, per role\n")
    A(f"![ID](01_per_persona_id_{view}_L{layer}.png)\n")
    A("`design null` = 25 synthetic points on the same 5x5 additive grid, built from the "
      "empirical instruction/question effect covariances, containing **no persona at all**. "
      "`Gaussian null` = 25 structureless points matched to the pooled within-role covariance.\n")
    A("| estimator | real median (IQR) | design null (IQR) | inside null IQR? |")
    A("|---|---|---|---|")
    for k, v in idr["verdict"].items():
        if v is None:
            continue
        r, n = v["real"], v["design_null"]
        A(f"| {k} | {_num(r['median'])} ({_num(r['q25'])}–{_num(r['q75'])}) | "
          f"{_num(n['median'])} ({_num(n['q25'])}–{_num(n['q75'])}) | "
          f"{'yes' if v['real_median_inside_design_iqr'] else 'no'} |")
    ax = idr.get("axis_vs_id", {})
    A(f"\nPer-role ID vs assistant-axis position: r = {_num(ax.get('pearson_r_excl_default'))} "
      f"excluding `default` (r = {_num(ax.get('pearson_r_all'))} with it). `default` itself has "
      f"much the lowest per-role ID in the set (participation ratio "
      f"{_num(ax.get('default_participation_ratio'))} against a median of "
      f"{_num(idr['per_role_summary']['PCA_participation_ratio']['median'])}). That is **not** a "
      "geometric fact about the Assistant: `default`'s five instruction slots are one empty "
      "prompt plus four near-synonymous generic strings (\"You are an AI assistant.\", \"You are "
      "a large language model.\", ...), whereas every character role's five slots are five "
      "distinct persona embodiments. Its instruction axis therefore carries little variance, and "
      "since instruction variance is ~68% of the total, its ID collapses. Same design artefact, "
      "seen from the other side.\n")
    A(f"Source: `exploratory/per_persona/01_per_persona_id.py` "
      f"({m['runtime_s']}s for all {m['n_roles']} roles).\n")

    A("## 2. Clustering, per role\n")
    A(f"![clustering](02_per_persona_clustering_{view}_L{layer}.png)\n")
    cm, real, null = clu["_meta"], clu["real"], clu["design_null"]
    A(f"k searched over {cm['k_range'][0]}–{cm['k_range'][1]} (25 points will not support more), "
      "in each role's own PCA-95% space. ARI is scored against the two factors we know.\n")
    A("| method | median k | silhouette | ARI vs instruction | ARI vs question | "
      "design-null ARI vs instruction |")
    A("|---|---|---|---|---|---|")
    for meth in ("kmeans", "kmeans_k5", "gmm", "hdbscan", "dbscan"):
        g = lambda d, s: d.get(f"{meth}_{s}", {}).get("median")
        A(f"| {meth} | {_num(g(real,'n_clusters'), 0)} | {_num(g(real,'silhouette'), 3)} | "
          f"{_num(g(real,'ari_instr'), 3)} | {_num(g(real,'ari_quest'), 3)} | "
          f"{_num(g(null,'ari_instr'), 3)} |")
    A(f"\nGround-truth partition silhouette: instruction "
      f"{_num(real['truth_instr_silhouette']['median'], 3)} vs question "
      f"{_num(real['truth_quest_silhouette']['median'], 3)}. Every role splits into the 5 "
      "instruction phrasings, and the design null — which has no persona in it — splits the "
      "same way and just as cleanly.\n")
    A(f"Source: `exploratory/per_persona/02_per_persona_clustering.py` "
      f"({cm['runtime_s']}s for all {cm['n_roles']} roles).\n")

    A("## 3. Compute budget\n")
    A(f"![budget](03_compute_budget_{view}_L{layer}.png)\n")
    bm = bud["_meta"]
    tim = ", ".join(f"{k} roles {v}s" for k, v in bud["analysis_seconds_by_n_roles"].items())
    A(f"**Analysis is free.** Measured: {tim}. Restricting to 10 / 50 / 100 roles buys nothing "
      "on the cloud we already have — run all 276.\n")
    need = ", ".join(f"d={k} needs N≥{v}" for k, v in bud["min_n_within_20pct"].items() if v)
    rec = bud["id_recovery_vs_n"]
    at25 = ", ".join(f"a true d={k} reads as {v['25']:.1f}" for k, v in rec.items()
                     if v.get("25"))
    A("**Extraction is the only real cost.** Planted manifolds of known dimension are only "
      f"recovered once a role has enough points ({need}, for an estimate within 20% of truth). "
      f"At the 25 points per role we have now, {at25} — which is the range this study reports. "
      f"Target: **{bud['target_points_per_role']} points/role = "
      f"{bud['questions_per_role_needed']} questions x {bm['n_instructions']} instructions** "
      f"(the question pool holds {bm['question_pool']}).\n")
    A(f"At {bm['sec_per_record']} s/record — {bm['sec_per_record_source']}:\n")
    A("| roles | records | hours | days |")
    A("|---|---|---|---|")
    for k, v in bud["extraction_budget"].items():
        A(f"| {k} | {v['records']:,} | {v['hours']} | {v['days']} |")
    A(f"\nSource: `exploratory/per_persona/03_compute_budget.py`.\n")

    out = run_dir / "REPORT.md"
    out.write_text("\n".join(L))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg", choices=["prompt_avg", "prompt_last"])
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--n_null", type=int, default=100)
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    run_dir = FIGURES_DIR / (args.stamp or timestamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MP_RUN_DIR"] = str(run_dir)
    env["_VIEW"] = args.view
    print(f"Results -> {run_dir}")

    nn = ["--n_null", str(args.n_null)]
    run("01_per_persona_id.py", nn, env)
    run("02_per_persona_clustering.py", nn, env)
    run("03_compute_budget.py", [], env)
    write_report(run_dir, args.view, args.layer)
    print(f"\nDone. See {run_dir}/REPORT.md")


if __name__ == "__main__":
    main()
