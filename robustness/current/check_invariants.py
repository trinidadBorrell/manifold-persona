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
           (robustness/current/baseline/), so it tests the aggregation code only
  artifact read from a committed baseline artifact; tests nothing about
           current code — upgraded to raw by --full

AN `artifact` PIN IS REPORTED AS "NOT VERIFIED", NEVER AS PASS. Comparing a
number read out of the baseline against a pin derived from that same baseline is
`abs(x - round(x, 6)) <= tol`, which holds no matter what the code does. The
default fast mode leaves several pins in that state, so its summary line says
how many were verified and how many were not; --full recomputes them.

Usage:
  .venv/bin/python robustness/current/check_invariants.py          # fast, ~3 min
  .venv/bin/python robustness/current/check_invariants.py --full   # + panel and
                                                           # calibration
                                                           # reruns (slow)
  .venv/bin/python robustness/current/check_invariants.py --pin    # accept the
                                                           # current
                                                           # numbers as
                                                           # the new pins

To update pins after an intentional change: run with --pin, read the
`git diff robustness/current/invariants.json`, and say why in the commit.
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

REPO = Path(__file__).resolve().parents[2]
PP = REPO / "exploratory" / "per_persona"
BASELINE = Path(__file__).resolve().parent / "baseline"
PINS_FILE = Path(__file__).resolve().parent / "invariants.json"
PYTHON = str(REPO / ".venv" / "bin" / "python")
L = 19
# The panel, ladder and calibration all live on the response cloud.
RESP_ENV = {"MP_ROLE_DIR": "data/embeddings_roles_resp_40q"}

# The current panel drops PCA_dim_90pct and lPCA (see metrics.py
# DROPPED_FROM_PANEL), so the ladder no longer outputs them. Their old
# pins live on in robustness/legacy/.
LADDER_METRICS = ["MLE", "TwoNN", "PCA_participation_ratio",
                  "H1_total_persistence"]

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
# Each carries the values it reads, so a comparison evaluated entirely on
# baseline constants can be reported as unverified rather than as PASS.
COMPARE = [
    ("pr_median_real > pr_median_null + 1.0",
     lambda v: v["pr_median_real"] > v["pr_median_null"] + 1.0,
     ("pr_median_real", "pr_median_null")),
    ("fold_change > 10",
     lambda v: v["fold_change"] > 10.0,
     ("fold_change",)),
    # MLE fails its own calibration gate on this machine (GATE_OVERRIDE).
    # If this pin ever fails, calibration got FIXED — good news, but it
    # reopens the MLE-trust question. Investigate, do not just re-pin.
    ("mle_worst_calib_error > 0.20  (MLE stays untrusted)",
     lambda v: v["mle_worst_calib_error"] > 0.20,
     ("mle_worst_calib_error",)),
]

# A pin at this depth was READ OUT OF THE BASELINE, not recomputed. Comparing it
# against a number derived from that same baseline is `abs(x - round(x, 6)) <=
# tol`: it cannot fail, whatever the code does. Those rows are reported as NOT
# VERIFIED rather than PASS, and the run's summary line says how many there
# were — the fast mode used to print "All invariants hold." while half its pins
# were structurally incapable of failing, so a regression that broke
# study_panel.py or calib_estimators.py outright still exited 0.
VACUOUS_DEPTHS = {"artifact"}


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
    if not (BASELINE / f"design_null_L{L}.json").exists():
        raise SystemExit(
            f"no baseline at {BASELINE}. Bootstrap the current tier first:\n"
            f"  run study_panel.py, study_design_null.py and "
            f"calib_estimators.py once,\n  copy their outputs into "
            f"{BASELINE}/, then run --full --pin.")
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
    # In fast mode this is the median of a COPIED baseline CSV — no current code
    # produced it, so it is an artifact read, not a cached recompute. (The
    # ladder pins below are genuinely "cached": study_ladder.py really does run
    # over that panel, so its aggregation code is under test.)
    depth["pr_median_real"] = "raw" if panel_depth == "raw" else "artifact"
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
    rows, n_fail, n_vacuous = [], 0, 0

    def record(kind, name, want, got, tol, dep, ok, vacuous):
        nonlocal n_fail, n_vacuous
        # A vacuous comparison that nevertheless MISMATCHES is not silence, it
        # is evidence: the baseline artifact and invariants.json were written
        # from different runs. Counted and named, never folded into the "we did
        # not check this" bucket.
        if vacuous and ok:
            n_vacuous += 1
        elif not ok:
            n_fail += 1
        rows.append((kind, name, want, got, tol, dep, ok, vacuous))

    for key, fallback in EXACT:
        tol = tols.get(key, fallback)
        got, want = values[key], exp[key]
        ok = abs(got - want) <= tol
        record("exact", key, f"{want:.4g}", f"{got:.4g}", f"±{tol:g}",
               depth[key], ok, depth[key] in VACUOUS_DEPTHS)
    for key in SIGN:
        got = values[key]
        want = pins["signs"][key]
        # `(nan > 0) == (want > 0)` is True whenever the pinned sign is
        # non-positive, so a measurement that came back NaN passed silently.
        ok = bool(pd.notna(got)) and (got > 0) == (want > 0)
        record("sign", key, "+" if want > 0 else "−",
               "+" if pd.notna(got) and got > 0 else ("nan" if pd.isna(got) else "−"),
               "", depth[key], ok, depth[key] in VACUOUS_DEPTHS)
    for desc, fn, inputs in COMPARE:
        ok = bool(fn(values))
        deps = {depth[k] for k in inputs}
        # Vacuous only when EVERY input was read out of the baseline; a
        # comparison with one live operand can still fail, and should.
        record("struct", desc, "", "", "", "/".join(sorted(deps)), ok,
               deps <= VACUOUS_DEPTHS)

    w = max(len(r[1]) for r in rows)
    print(f"\n{'kind':7s} {'invariant':{w}s} {'pinned':>10s} {'now':>10s} "
          f"{'tol':>8s} {'depth':16s} verdict")
    # BASELINE DRIFT: an `artifact`-depth row whose comparison could not fail,
    # and did. It means baseline/ was regenerated without a matching --pin.
    for kind, name, want, got, tol, dep, ok, vac in rows:
        if ok:
            verdict = "NOT VERIFIED" if vac else "PASS"
        else:
            verdict = "** BASELINE DRIFT **" if vac else "** FAIL **"
        print(f"{kind:7s} {name:{w}s} {want:>10s} {got:>10s} "
              f"{tol:>8s} {dep:16s} {verdict}")

    n_checked = len(rows) - n_vacuous
    print(f"\npinned {pins['pinned_at']} at {pins['git_commit'][:9]} "
          f"(tier: {pins['tier']})")

    now_env, pinned_env = _numeric_env(), pins.get("env")
    if pinned_env:
        moved = {k: (pinned_env[k], now_env[k]) for k in now_env
                 if k in pinned_env and pinned_env[k] != now_env[k]}
        if moved:
            print("\nNUMERIC ENVIRONMENT CHANGED since these pins were written:")
            for k, (was, now) in sorted(moved.items()):
                print(f"  {k}: {was} -> {now}")
            print("  A moved value with a moved library is a different "
                  "investigation from a moved value without one. sklearn in "
                  "particular changed PCA's svd_solver='auto' heuristic, which "
                  "can shift plane_r2 and curv_gain with no code change.")
    else:
        print("\nNOTE: these pins predate environment recording; rerun --pin to "
              "capture numpy/scipy/sklearn versions alongside them.")
    if n_vacuous:
        print(f"\n{n_vacuous} pin(s) NOT VERIFIED: the value was read out of "
              f"robustness/current/baseline/ rather than recomputed, so the "
              f"comparison is of that baseline against itself and cannot fail. "
              f"Run with --full to recompute them from the clouds.")
    if n_fail:
        print(f"\n{n_fail} of {n_checked} verified invariants FAIL. A fail is "
              f"a question, not a verdict.")
        print("  Bug?       Compare your diff against the source script "
              "of the failing pin.")
        print("  Discovery? If the change is intentional, rerun with "
              "--pin, then commit the")
        print("             invariants.json diff and say why.")
    else:
        print(f"\nAll {n_checked} verified invariants hold"
              + (f" ({n_vacuous} not verified)." if n_vacuous else "."))
    return 1 if n_fail else 0


def _numeric_env() -> dict:
    """The libraries whose version can move a published number on its own.

    sklearn is the live one: `requirements.txt` allows >= 1.6 (hdbscan needs
    it), and the `svd_solver="auto"` heuristic PCA uses changed across that
    boundary, so `plane_r2` and `curv_gain` can shift with no code change at
    all. Pinning the versions does not stop that, but it stops it happening
    invisibly — a moved number with a moved sklearn is a different
    investigation from a moved number without one.
    """
    from manifold_persona.provenance import lib_versions
    return lib_versions()


def pin(values: dict, full: bool) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    out = {
        "pinned_at": date.today().isoformat(),
        "git_commit": commit,
        "tier": "full" if full else "fast",
        "env": _numeric_env(),
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
    print("Review with: git diff robustness/current/invariants.json")


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
