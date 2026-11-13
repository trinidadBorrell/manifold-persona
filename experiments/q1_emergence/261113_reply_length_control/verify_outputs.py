"""Check the saved length-control outputs from the files (Codex review 13/11/2026, findings 1 and 2). Reads only.

Per stage:
- every output file matches SHA256SUMS.txt; the inputs, model and code match the manifest;
- keys.json covers every row of rows.jsonl in order;
- each cut array is float16 [rows, 4 states, width]; every row is all finite or all NaN; the finite rows are exactly
  the eligible rows (valid, at least N kept tokens); the count and role shares equal support.json;
- values: on rows with exactly N kept tokens the cut span is the whole reply, so the first-N state must reproduce
  the stored reply mean at all four states (pod A's C0 gate); first16 and first32 differ on rows longer than 32;
- the smoke and full manifests share one fingerprint (every field except the row count, limit, results, time and the
  GPU card form, which is recorded).

  python -I verify_outputs.py --stage sft --run /out/own_replies/sft_full --full /out/length_control/full/sft \
      --smoke /out/length_control/smoke/sft --out /out/length_control/verify/sft.json
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_length_control", HERE / "length_control.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
RUN_ONLY = ("n_rows", "limit", "check_passed", "available", "seconds")
OUT_FILES = {"first16_acts.npy", "first32_acts.npy", "keys.json", "support.json", "check.json", "manifest.json"}


def fingerprint_diff(a, b):
    """Manifest fields that differ between two runs, ignoring the per-run fields and the GPU card form: Modal gives
    an A100-80GB as PCIe or SXM4, and the launcher cannot choose. Library versions must match."""
    def strip(m):
        m = {k: v for k, v in m.items() if k not in RUN_ONLY}
        if isinstance(m.get("versions"), dict):
            m["versions"] = {k: v for k, v in m["versions"].items() if k != "device"}
        return m
    a, b = strip(a), strip(b)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def smoke_gate(smoke_man, code, input_sums, entry):
    """Before a full run: what the smoke ran with must equal what the full run will use. Returns the differing parts."""
    bad = []
    if not smoke_man.get("check_passed"):
        bad.append("smoke check not passed")
    if smoke_man["code"] != code:
        bad.append("code")
    if smoke_man["inputs"]["sha256"] != {k: input_sums.get(k) for k in smoke_man["inputs"]["sha256"]}:
        bad.append("inputs")
    if (smoke_man["model"], smoke_man["revision"]) != (entry["repo"], entry["revision"]):
        bad.append("model or revision")
    return bad


def read_sums(path):
    return dict(reversed(line.split("  ", 1)) for line in Path(path).read_text().splitlines())


def row_states(x, chunk=2000):
    """Per row: are all states finite; are all states NaN."""
    fin, nan = np.zeros(len(x), bool), np.zeros(len(x), bool)
    for i in range(0, len(x), chunk):
        a = np.asarray(x[i:i + chunk]).reshape(min(chunk, len(x) - i), -1)
        fin[i:i + chunk], nan[i:i + chunk] = np.isfinite(a).all(1), np.isnan(a).all(1)
    return fin, nan


def n_equal_rows(x, y, rows, state, chunk=2000):
    """Rows where two cut arrays hold the same vector at one state."""
    n = 0
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        n += int((np.asarray(x[r, state]) == np.asarray(y[r, state])).all(1).sum())
    return n


def verify_stage(stage, run, full, smoke):
    R = L.rows_module()
    run, full, smoke = Path(run), Path(full), Path(smoke)
    fails = []

    def need(ok, msg):
        if not ok:
            fails.append(msg)
        return bool(ok)

    man = json.loads((full / "manifest.json").read_text())
    sums = read_sums(full / "SHA256SUMS.txt")
    need(set(sums) == OUT_FILES, f"SHA256SUMS.txt lists {sorted(sums)}")
    bad = [n for n, h in sums.items() if L.sha256(full / n) != h]
    need(not bad, f"output files differ from SHA256SUMS.txt: {bad}")
    run_sums = read_sums(run / "SHA256SUMS.txt")
    bad = [n for n, h in man["inputs"]["sha256"].items() if run_sums.get(n) != h or L.sha256(run / n) != h]
    need(not bad, f"inputs differ from the manifest: {bad}")
    entry = L.STAGES[stage]
    need((man["stage"], man["model"], man["revision"]) == (stage, entry["repo"], entry["revision"]),
         "stage, model or revision differs from stages.json")
    code_now = dict(length_control_py=L.sha256(HERE / "length_control.py"), rows_py=L.sha256(L.ROWS_PY))
    need(man["code"] == code_now, "the run's code hashes differ from the code here")
    need(man["check_passed"], "C0 check not passed")
    need((man["cuts"], man["states"]) == (L.CUTS, L.STATES), "cuts or states differ")

    rows = [json.loads(line) for line in (run / "rows.jsonl").open()]
    keys = json.loads((full / "keys.json").read_text())
    need(man["limit"] == 0 and keys["row_index"] == list(range(len(rows))), "the full run does not cover every row in order")
    rows = [rows[i] for i in keys["row_index"]]
    need(keys["keys"] == [R.row_key(r) for r in rows], "keys.json differs from rows.jsonl")
    stored = np.load(run / "reply_mean_acts.npy", mmap_mode="r")
    need(stored.shape[0] == len(rows), "stored states and rows differ in length")
    valid = np.array([bool(r["valid"]) for r in rows])
    kept = np.array([len(r["kept_ids"]) for r in rows])
    sup, sup_now = json.loads((full / "support.json").read_text()), L.support(rows)

    rec = dict(stage=stage, n_rows=len(rows), cuts={})
    X = {}
    for c, n in L.CUTS.items():
        x = np.load(full / f"{c}_acts.npy", mmap_mode="r")
        cr = rec["cuts"][c] = dict(dtype=str(x.dtype), shape=list(x.shape))
        if not need(x.dtype == np.float16 and x.shape == (len(rows), len(L.STATES), stored.shape[2]),
                    f"{c}: dtype {x.dtype} or shape {x.shape} is wrong"):
            continue
        X[c] = x
        fin, nan = row_states(x)
        eligible = valid & (kept >= n)
        cr.update(n_finite=int(fin.sum()), n_eligible=int(eligible.sum()), n_partly_finite=int((~fin & ~nan).sum()),
                  n_finite_not_eligible=int((fin & ~eligible).sum()), n_eligible_not_finite=int((eligible & ~fin).sum()))
        need(cr["n_partly_finite"] == 0, f"{c}: {cr['n_partly_finite']} rows are partly finite")
        mask_ok = need((fin == eligible).all(), f"{c}: finite rows differ from eligible rows")
        need(sup[c]["n_finite_rows"] == cr["n_finite"], f"{c}: finite count differs from support.json")
        need(sup[c]["available"] == sup_now[c]["available"] and sup[c]["lowest"][0] == sup_now[c]["lowest"][0]
             and all(abs(sup[c]["share"][k] - v) < 1e-9 for k, v in sup_now[c]["share"].items())
             and set(sup[c]["share"]) == set(sup_now[c]["share"]), f"{c}: support.json differs from a recount")
        eq = np.flatnonzero(valid & (kept == n))
        cr["n_exact_length_rows"] = len(eq)
        if need(len(eq) >= 2, f"{c}: fewer than 2 rows with exactly {n} kept tokens") and mask_ok:
            ref = np.asarray(stored[eq], np.float64)[:, L.STATES]
            got = np.asarray(x[eq], np.float64)
            cells = {f"L{s}": R.c0_stats(ref[:, k], got[:, k]) for k, s in enumerate(L.STATES)}
            cr["exact_length_check"] = cells
            need(all(v["pass"] for v in cells.values()), f"{c}: exact-length rows do not reproduce the stored states")
    longer = np.flatnonzero(valid & (kept > max(L.CUTS.values())))
    if len(X) == len(L.CUTS):
        k17 = L.STATES.index(17)
        rec["n_rows_first16_equals_first32"] = n_equal_rows(X["first16"], X["first32"], longer, k17)
        need(rec["n_rows_first16_equals_first32"] == 0, "first16 equals first32 on rows longer than 32")

    smoke_man = json.loads((smoke / "manifest.json").read_text())
    need(L.sha256(smoke / "manifest.json") == read_sums(smoke / "SHA256SUMS.txt")["manifest.json"],
         "smoke manifest differs from its SHA256SUMS.txt")
    rec["smoke_fingerprint_diff"] = fingerprint_diff(smoke_man, man)
    rec["devices"] = dict(smoke=smoke_man.get("versions", {}).get("device"), full=man.get("versions", {}).get("device"))
    need(smoke_man["check_passed"] and smoke_man["limit"] == 64 and not rec["smoke_fingerprint_diff"],
         f"smoke run differs from the full run: {rec['smoke_fingerprint_diff']}")
    rec.update(passed=not fails, failures=fails)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["instruct", "sft", "dpo"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = verify_stage(a.stage, a.run, a.full, a.smoke)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(stage=a.stage, passed=rec["passed"], failures=rec["failures"]), indent=1), flush=True)
    if not rec["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
