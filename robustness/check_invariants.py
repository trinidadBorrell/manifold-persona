"""Invariant checks: recompute pinned result numbers, compare, report.

This is a smoke alarm for published results. It does not block anything.
A FAIL is a question, not a verdict: it can be a bug OR a discovery.

How it works:
  1. `invariants.json` holds the trusted numbers (written by --pin).
  2. This script re-runs the canonical pipeline scripts into a scratch
     dir, extracts the same numbers, and compares.
  3. It prints one PASS/FAIL row per pin, then exits 0 (all pass) or 1.

It never duplicates pipeline math. Every number comes from the same
script that produced the published value.

Tiers (each row reports its depth honestly):
  raw      recomputed from the activation clouds with current code
  cached   final statistic recomputed from a committed baseline input
           (robustness/baseline/), so it tests the aggregation code only
  artifact read from a committed baseline artifact; tests nothing about
           current code — upgraded to raw by --full

Usage:
  .venv/bin/python robustness/check_invariants.py          # fast, ~3 min
  .venv/bin/python robustness/check_invariants.py --full   # + panel and
                                                           # calibration
                                                           # reruns (slow)
  .venv/bin/python robustness/check_invariants.py --pin    # accept the
                                                           # current
                                                           # numbers as
                                                           # the new pins

To update pins after an intentional change: run with --pin, read the
`git diff robustness/invariants.json`, and say why in the commit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PP = REPO / "exploratory" / "per_persona"
BASELINE = Path(__file__).resolve().parent / "baseline"
PINS_FILE = Path(__file__).resolve().parent / "invariants.json"
PYTHON = str(REPO / ".venv" / "bin" / "python")
L = 19
# The panel, ladder and calibration all live on the response cloud.
RESP_ENV = {"MP_ROLE_DIR": "data/embeddings_roles_resp_40q"}

LADDER_METRICS = ["MLE", "TwoNN", "PCA_participation_ratio",
                  "PCA_dim_90pct", "H1_total_persistence"]

# Exact pins: measured value must sit within tol of the pinned value.
# The tol here is a FALLBACK. When invariants.json carries a "tol" block
# (written by calibrate_tolerances.py from measured spread), that wins.
EXACT = [
    ("prompt.instr_frac",        0.02),
    ("prompt.quest_frac",        0.02),
    ("prompt.interaction_frac",  0.005),
    ("response.instr_frac",      0.02),
    ("response.quest_frac",      0.03),
    ("response.interaction_frac", 0.03),
    ("fold_change",              6.0),
    ("pr_median_real",           1.0),
    ("pr_median_null",           1.0),
    ("mle_worst_calib_error",    0.05),
] + [(f"r.{m}", 0.05) for m in LADDER_METRICS]

# Sign pins: only the direction is asserted. These survive legitimate
# pivots that move the magnitude.
SIGN = [f"r.{m}" for m in LADDER_METRICS]

# Structural pins: fixed comparisons that define the published story.
# They hold no pinned number, so --pin never touches them.
COMPARE = [
    ("pr_median_real > pr_median_null + 1.0",
     lambda v: v["pr_median_real"] > v["pr_median_null"] + 1.0),
    ("fold_change > 10",
     lambda v: v["fold_change"] > 10.0),
    # MLE fails its own calibration gate on this machine (GATE_OVERRIDE).
    # If this pin ever fails, calibration got FIXED — good news, but it
    # reopens the MLE-trust question. Investigate, do not just re-pin.
    ("mle_worst_calib_error > 0.20  (MLE stays untrusted)",
     lambda v: v["mle_worst_calib_error"] > 0.20),
]


def run(cmd: list[str], ok_codes=(0,), extra_env: dict | None = None) -> None:
    env = os.environ.copy()
    env.pop("MP_ROLE_DIR", None)  # canonical data dirs only
    env.update(extra_env or {})
    print(f"  $ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=REPO, env=env,
                       capture_output=True, text=True)
    if p.returncode not in ok_codes:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
        raise SystemExit(f"step failed (exit {p.returncode}): {cmd}")


def measure(full: bool) -> tuple[dict, dict]:
    """Recompute every pinned number. Returns (values, depth-per-value)."""
    values: dict[str, float] = {}
    depth: dict[str, str] = {}
    scratch = Path(tempfile.mkdtemp(prefix="invariants_"))
    (scratch / "data").mkdir()
    (scratch / "logs").mkdir()

    # -- variance splits + fold-change: always raw ------------------------
    print("\n[1/4] variance splits (raw recompute from both clouds)")
    # The prompt cloud predates the attention-sink fix and is known
    # contaminated. That is the POINT of the prompt_5x5 pin: it is the
    # contrast cloud whose within-role variance is ~99% extraction grid.
    # So we opt past the clean-cloud guard for this step only.
    run([PYTHON, str(PP / "confound_variance.py"),
         "--outdir", str(scratch), "--label-layer", str(L)],
        extra_env={"MP_ALLOW_UNCLEAN": "1"})
    pv = json.load(open(scratch / "data" / f"prompt_vs_response_L{L}.json"))
    for label, key in (("prompt_5x5", "prompt"), ("response_5x40", "response")):
        c = pv["clouds"][label]
        if "error" in c:
            raise SystemExit(f"cloud {label} unavailable: {c['error']}")
        for t in ("instr_frac", "quest_frac", "interaction_frac"):
            values[f"{key}.{t}"] = c[t]["median"]
            depth[f"{key}.{t}"] = "raw"
    values["fold_change"] = pv["interaction_fold_change"]
    depth["fold_change"] = "raw"

    # -- per-role panel: baseline copy, or full rerun ----------------------
    print("\n[2/4] per-role panel")
    if full:
        run([PYTHON, str(PP / "study_panel.py"), "--outdir", str(scratch),
             "--view", "prompt_avg", "--label-layer", str(L)],
            extra_env=RESP_ENV)
        panel_depth = "raw"
    else:
        shutil.copy(BASELINE / f"per_role_panel_L{L}.csv", scratch / "data")
        panel_depth = "cached"
        print("  using committed baseline panel (run with --full to rerun)")
    shutil.copy(BASELINE / f"design_null_L{L}.json", scratch / "data")

    panel = pd.read_csv(scratch / "data" / f"per_role_panel_L{L}.csv")
    null = json.load(open(scratch / "data" / f"design_null_L{L}.json"))
    values["pr_median_real"] = float(
        panel["PCA_participation_ratio"].median())
    depth["pr_median_real"] = panel_depth
    values["pr_median_null"] = float(
        null["design_null"]["PCA_participation_ratio"]["median"])
    depth["pr_median_null"] = "artifact"  # 100-draw null is never rerun here

    # -- correlation ladder: aggregation recompute -------------------------
    print("\n[3/4] correlation ladder (recomputed from the panel)")
    run([PYTHON, str(PP / "study_ladder.py"), "--outdir", str(scratch),
         "--label-layer", str(L)])
    lad = pd.read_csv(scratch / "data" / f"ladder_L{L}.csv")
    lad = lad[lad.predictor == "axis_proj"].set_index("metric")
    for m in LADDER_METRICS:
        values[f"r.{m}"] = float(lad.loc[m, "r_ctrl_all"])
        depth[f"r.{m}"] = panel_depth

    # -- MLE calibration gate ----------------------------------------------
    print("\n[4/4] MLE calibration gate")
    if full:
        # exit 2 is the gate itself failing — expected, still writes JSON
        run([PYTHON, str(PP / "calib_estimators.py"), "--outdir", str(scratch),
             "--view", "prompt_avg", "--label-layer", str(L)],
            ok_codes=(0, 2), extra_env=RESP_ENV)
        calib = json.load(open(scratch / "data" / f"calibration_L{L}.json"))
        depth["mle_worst_calib_error"] = "raw"
    else:
        calib = json.load(open(BASELINE / f"calibration_L{L}.json"))
        depth["mle_worst_calib_error"] = "artifact"
        print("  using committed baseline calibration (run --full to rerun)")
    values["mle_worst_calib_error"] = float(
        calib["verdict"]["worst_relative_error_d_le_10"]["MLE"])

    shutil.rmtree(scratch, ignore_errors=True)
    return values, depth


def check(values: dict, depth: dict) -> int:
    if not PINS_FILE.exists():
        print("No invariants.json yet. Run with --pin first.")
        return 2
    pins = json.loads(PINS_FILE.read_text())
    exp = pins["values"]
    tols = pins.get("tol", {})
    rows, n_fail = [], 0

    for key, fallback in EXACT:
        tol = tols.get(key, fallback)
        got, want = values[key], exp[key]
        ok = abs(got - want) <= tol
        n_fail += not ok
        rows.append(("exact", key, f"{want:.4g}", f"{got:.4g}",
                     f"±{tol:g}", depth[key], ok))
    for key in SIGN:
        got = values[key]
        want = pins["signs"][key]
        ok = (got > 0) == (want > 0)
        n_fail += not ok
        rows.append(("sign", key, "+" if want > 0 else "−",
                     "+" if got > 0 else "−", "", depth[key], ok))
    for desc, fn in COMPARE:
        ok = bool(fn(values))
        n_fail += not ok
        rows.append(("struct", desc, "", "", "", "", ok))

    w = max(len(r[1]) for r in rows)
    print(f"\n{'kind':7s} {'invariant':{w}s} {'pinned':>10s} {'now':>10s} "
          f"{'tol':>8s} {'depth':8s} verdict")
    for kind, name, want, got, tol, dep, ok in rows:
        print(f"{kind:7s} {name:{w}s} {want:>10s} {got:>10s} "
              f"{tol:>8s} {dep:8s} {'PASS' if ok else '** FAIL **'}")

    print(f"\npinned {pins['pinned_at']} at {pins['git_commit'][:9]} "
          f"(tier: {pins['tier']})")
    if n_fail:
        print(f"\n{n_fail} FAIL. A fail is a question, not a verdict.")
        print("  Bug?       Compare your diff against the source script "
              "of the failing pin.")
        print("  Discovery? If the change is intentional, rerun with "
              "--pin, then commit the")
        print("             invariants.json diff and say why.")
    else:
        print("\nAll invariants hold.")
    return 1 if n_fail else 0


def pin(values: dict, full: bool) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    out = {
        "pinned_at": date.today().isoformat(),
        "git_commit": commit,
        "tier": "full" if full else "fast",
        "values": {k: round(v, 6) for k, v in values.items()},
        "signs": {k: (1 if values[k] > 0 else -1) for k in SIGN},
    }
    if PINS_FILE.exists():  # keep calibrated tolerances across re-pins
        old = json.loads(PINS_FILE.read_text())
        for keep in ("tol", "tol_provenance"):
            if keep in old:
                out[keep] = old[keep]
    PINS_FILE.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nPinned {len(out['values'])} values -> {PINS_FILE}")
    print("Review with: git diff robustness/invariants.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true",
                    help="also rerun the panel and the calibration gate "
                         "(slow: ripser over 276 roles)")
    ap.add_argument("--pin", action="store_true",
                    help="write the measured values as the new pins")
    args = ap.parse_args()

    if not args.pin and not PINS_FILE.exists():
        print("No invariants.json yet. Run with --pin first.")
        raise SystemExit(2)
    values, depth = measure(args.full)
    if args.pin:
        pin(values, args.full)
    else:
        raise SystemExit(check(values, depth))


if __name__ == "__main__":
    main()
