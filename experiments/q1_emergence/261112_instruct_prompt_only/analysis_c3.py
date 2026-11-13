"""Prompt-only control analysis on CPU (rules: THRESHOLDS.md): M1 (C3 map split-half r), M2 (reply map vs C3 map,
paired D2 - M2), M3 (role decoding), each with pod A's own functions, halves, draws and seeds.

  .venv/bin/python -I analysis_c3.py --c3 run_c3_full --poda_run ../261029_instruct_persona_map/run_full --out results_full
  (--n_boot 200 --n_perm 199 for a quick engineering check; those numbers are not results)
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POD_A = HERE.parent / "261029_instruct_persona_map/analysis.py"
STATES_SAVED = [8, 9, 16, 17, 23, 24]


def pod_a():
    spec = importlib.util.spec_from_file_location("_poda_analysis", POD_A)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify(folder):
    for line in (folder / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        if pod_a().sha256(folder / name) != h:
            raise SystemExit(f"{folder / name} does not match its checksum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", required=True, type=Path)
    ap.add_argument("--poda_run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=9999)
    ap.add_argument("--n_halves", type=int, default=50)
    a = ap.parse_args()
    A = pod_a()
    M = A.M
    verify(a.c3)
    manifest = json.loads((a.poda_run / "manifest.json").read_text())
    rows_r = [json.loads(line) for line in (a.poda_run / "rows.jsonl").open()]
    roles = [r for r in manifest["roles"] if r != "default"]  # pod A's role list and order
    questions = sorted({r["question"] for r in rows_r})
    acts_r = np.load(a.poda_run / "reply_mean_acts.npy", mmap_mode="r")
    keys = json.loads((a.c3 / "c3_keys.json").read_text())
    acts_c = np.load(a.c3 / "c3_acts.npy")
    if len(keys) != 4200:
        raise SystemExit(f"{len(keys)} C3 prompts; the analysis needs the full 4,200")
    rows_c = [dict(role=k[0], format=k[1], wording=k[2], question=k[3], replicate=0, valid=True) for k in keys]
    cfg = dict(n_boot=a.n_boot, n_perm=a.n_perm, n_halves=a.n_halves)
    report = dict(rules="THRESHOLDS.md v1.0", config=cfg, roles=roles, n_questions=len(questions), states={})
    for state in (17, 9, 23):
        Xr = np.asarray(acts_r[:, state], dtype=np.float64)
        Xc = np.asarray(acts_c[:, STATES_SAVED.index(state)], dtype=np.float64)
        m1 = A.map_statistics(Xc, rows_c, roles, questions, **cfg)
        halves = A.make_halves(questions, n=a.n_halves)
        sides = {k: M.build_sides(*A.cell_means(X, rows, roles, questions), halves)
                 for k, X, rows in (("reply", Xr, rows_r), ("c3", Xc, rows_c))}
        rr, W = M.make_draws(len(roles), len(questions), n=a.n_boot, seed=A.SEED)
        panel = A.map_panel(sides, rr, W, [("reply", "reply"), ("reply", "c3")], strict=True)
        d2, m2 = panel[("reply", "reply")]["r"], panel[("reply", "c3")]["r"]
        diff = A.summary(d2 - m2)
        decode = {}
        for name, X, rows in (("c3", Xc, rows_c), ("reply", Xr, rows_r)):
            P = A.prompt_index(rows, roles, questions)
            F = A.prompt_features(X, P)
            res = A.decode_features(P, lambda tr, te, F=F: (F[tr], F[te]), A.decode_splits(P, n=a.n_halves))
            decode[name] = dict(accuracy=A.summary(A.accuracy_draws(res["cell"], rr, W)),
                                mean_fold_accuracy=res["mean_fold_accuracy"])
        m1_pass = bool(m1["raw_r"]["lo"] is not None and m1["raw_r"]["lo"] >= .5 and m1["p"] is not None and m1["p"] < .01)
        reading = ("the replies add role structure beyond the prompt" if diff["lo"] is not None and diff["lo"] > 0 else
                   "no evidence that the replies add structure beyond the prompt (not proof that the maps are the same)")
        report["states"][str(state)] = dict(
            M1_c3_map={k: m1[k] for k in ("raw_r", "p", "null_p99", "precision_met")},
            M1_prompt_alone_gives_a_role_map=m1_pass,
            D2_reply_map=A.summary(d2), M2_reply_vs_c3=A.summary(m2), D2_minus_M2=diff, M2_reading=reading,
            M3_decoding=decode)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "c3_results.json").write_text(json.dumps(A.clean(report), indent=1))
    s = report["states"]["17"]
    print(json.dumps(dict(M1=s["M1_c3_map"]["raw_r"], M1_pass=s["M1_prompt_alone_gives_a_role_map"],
                          D2=s["D2_reply_map"], M2=s["M2_reply_vs_c3"], D2_minus_M2=s["D2_minus_M2"],
                          reading=s["M2_reading"],
                          decode={k: v["accuracy"] for k, v in s["M3_decoding"].items()}), indent=1))


if __name__ == "__main__":
    main()
