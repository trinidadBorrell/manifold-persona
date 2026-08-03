"""Run the per-role ID stage into one dated folder + REPORT.md.

This is the EARLIER of the two studies in this directory: 276 per-role intrinsic
dimensions and 276 clusterings, plus the compute budget. The later
geometry-vs-axis study has its own driver, `run_geometry.py`.

Same orchestration contract as exploratory/assistant_axis/run_all.py (one fixed
timestamp shared via MP_RUN_DIR), but the report is written here rather than in a
separate make_report.py: this study is three scripts, and its report is a short
read of three JSON files.

The `01_`/`02_`/`03_` prefixes on the OUTPUT filenames are deliberately kept even
though the scripts were renamed — two published run folders
(`figures/resp40/`, `figures/29-Jul-2026-1255/`) carry those names, and
`id_vs_axis.py --rundir` reads `01_per_role_id_*.csv` out of them.
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
    n_i, n_q = m.get("grid", [None, None])
    grid = f"{n_i}x{n_q}"
    inter = dv["interaction_frac"]["median"]
    L = []
    A = L.append
    A(f"# Per-persona manifold — {view}, layer {layer}\n")
    A(f"Each of the **{m['n_roles']} roles** is treated as its own point cloud of "
      f"**{m['points_per_role']} points** ({grid}: instruction phrasings x questions) in "
      f"{m['ambient']} ambient dimensions, and gets its own intrinsic dimension and its own "
      "clustering.\n")

    A("## Headline\n")
    A(f"The within-role cloud is **{_pct(dv['instr_frac']['median'])} instruction phrasing, "
      f"{_pct(dv['quest_frac']['median'])} question, and {_pct(inter)} interaction**. "
      f"Instruction and question are the two axes of the {grid} extraction grid, so everything "
      "except that last term is forced by the experiment's design, not by the model. The "
      f"additive rank of a {grid} grid is ({n_i}-1)+({n_q}-1) = "
      f"**{m['additive_design_rank']}**; observed lPCA is "
      f"{_num(idr['per_role_summary']['lPCA']['median'], 1)}.\n")
    if inter is not None and inter < 0.05:
        A("Interaction — the only term a grid does **not** force — is under 5%. On this cloud "
          "the per-role ID and clustering are therefore measurements of the extraction grid, "
          "not of persona geometry. Reported in full below, with the null that shows why.\n")
    else:
        A(f"Interaction — the only term a grid does **not** force — carries {_pct(inter)} of "
          "the within-role variance, so there is genuine role-specific structure here beyond "
          "the design. The design null below is what separates the two.\n")

    A("## 1. Intrinsic dimension, per role\n")
    A(f"![ID](01_per_persona_id_{view}_L{layer}.png)\n")
    A(f"`design null` = {m['points_per_role']} synthetic points on the same {grid} additive "
      "grid, built from the empirical instruction/question effect covariances, containing "
      f"**no persona at all**. `Gaussian null` = {m['points_per_role']} structureless points "
      "matched to the pooled within-role covariance.\n")
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
    d_pr = ax.get("default_participation_ratio")
    med_pr = idr["per_role_summary"]["PCA_participation_ratio"]["median"]
    A(f"\nPer-role ID vs assistant-axis position: r = {_num(ax.get('pearson_r_excl_default'))} "
      f"excluding `default` (r = {_num(ax.get('pearson_r_all'))} with it). `default`'s own ID is "
      f"{_num(d_pr)} against a median of {_num(med_pr)}.")
    if d_pr is not None and med_pr and d_pr < 0.6 * med_pr:
        A("\nThat gap is **not** a geometric fact about the Assistant. `default`'s instruction "
          "slots are one empty prompt plus near-synonymous generic strings (\"You are an AI "
          "assistant.\", \"You are a large language model.\", ...), whereas every character "
          "role's slots are distinct persona embodiments. Its instruction axis therefore "
          "carries little variance, and instruction variance dominates the total — so its ID "
          "collapses. Same design effect, seen from the other side.\n")
    else:
        A("\n")
    A(f"Source: `exploratory/per_persona/id_per_role.py` "
      f"({m['runtime_s']}s for all {m['n_roles']} roles).\n")

    A("## 2. Clustering, per role\n")
    A(f"![clustering](02_per_persona_clustering_{view}_L{layer}.png)\n")
    cm, real, null = clu["_meta"], clu["real"], clu["design_null"]
    A(f"k searched over {cm['k_range'][0]}–{cm['k_range'][1]} — spanning both ground-truth "
      f"counts ({n_i} instructions, {n_q} questions) — in each role's own PCA-95% space. "
      "ARI is scored against the two factors we know.\n")
    A("| method | median k | silhouette | ARI vs instruction | ARI vs question | "
      "design-null ARI vs instruction |")
    A("|---|---|---|---|---|---|")
    meths = ["kmeans"] + [f"kmeans_k{k}" for k in sorted({n_i, n_q})] + \
            ["gmm", "hdbscan", "dbscan"]
    for meth in meths:
        g = lambda d, s: d.get(f"{meth}_{s}", {}).get("median")
        A(f"| {meth} | {_num(g(real,'n_clusters'), 0)} | {_num(g(real,'silhouette'), 3)} | "
          f"{_num(g(real,'ari_instr'), 3)} | {_num(g(real,'ari_quest'), 3)} | "
          f"{_num(g(null,'ari_instr'), 3)} |")
    ti = real["truth_instr_silhouette"]["median"]
    tq = real["truth_quest_silhouette"]["median"]
    A(f"\nGround-truth partition silhouette: instruction {_num(ti, 3)} vs question "
      f"{_num(tq, 3)}. Compare each method's ARI column against the design-null column: where "
      "they match, the split is the grid; where the real ARI departs from it, it is not.\n")
    A(f"Source: `exploratory/per_persona/clustering_per_role.py` "
      f"({cm['runtime_s']}s for all {cm['n_roles']} roles).\n")

    A("## 3. Compute budget\n")
    A(f"![budget](03_compute_budget_{view}_L{layer}.png)\n")
    bm = bud["_meta"]
    tim = ", ".join(f"{k} roles {v}s" for k, v in bud["analysis_seconds_by_n_roles"].items())
    A(f"**Analysis is free.** Measured: {tim}. Restricting to 10 / 50 / 100 roles buys nothing "
      f"on the cloud we already have — run all {m['n_roles']}.\n")
    need = ", ".join(f"d={k} needs N≥{v}" for k, v in bud["min_n_within_20pct"].items() if v)
    rec = bud["id_recovery_vs_n"]
    # Quote the recovery curve at the grid point nearest this cloud's actual
    # size, not at a hardcoded N — the two clouds differ 8x.
    n_now = m["points_per_role"]
    here = ", ".join(
        f"a true d={k} reads as {v[min(v, key=lambda s: abs(int(s) - n_now))]:.1f}"
        for k, v in rec.items() if v)
    n_shown = min(next(iter(rec.values())), key=lambda s: abs(int(s) - n_now))
    A("**Extraction is the only real cost.** Planted manifolds of known dimension are only "
      f"recovered once a role has enough points ({need}, for an estimate within 20% of truth). "
      f"At {n_now} points per role (curve sampled at N={n_shown}), {here}. "
      f"Target: **{bud['target_points_per_role']} points/role = "
      f"{bud['questions_per_role_needed']} questions x {bm['n_instructions']} instructions** "
      f"(the question pool holds {bm['question_pool']}).\n")
    A(f"At {bm['sec_per_record']} s/record — {bm['sec_per_record_source']}:\n")
    A("| roles | records | hours | days |")
    A("|---|---|---|---|")
    for k, v in bud["extraction_budget"].items():
        A(f"| {k} | {v['records']:,} | {v['hours']} | {v['days']} |")
    A(f"\nSource: `exploratory/per_persona/compute_budget.py`.\n")

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
    run("id_per_role.py", nn, env)
    run("clustering_per_role.py", nn, env)
    run("compute_budget.py", [], env)
    write_report(run_dir, args.view, args.layer)
    print(f"\nDone. See {run_dir}/REPORT.md")


if __name__ == "__main__":
    main()
