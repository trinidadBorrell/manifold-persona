# Robustness checks

Two tiers. Both must pass before a result is quoted.

## legacy/ — old design (frozen)

Pins measured on the pre-integration panel (pinned 2026-08-07 at
`acef6c0`). That panel code is not on this branch, so the checker
validates the pinned numbers against the committed baseline artifacts
only. It reruns nothing.

```
.venv/bin/python robustness/legacy/check_artifacts.py
```

- Covers: PR medians, MLE calibration error, the five ladder
  correlations, their signs, and two structural pins.
- Skips: variance fractions and fold_change (they need raw clouds).
- Full legacy rerun: check out `feature/invariant-checks` and run its
  `robustness/check_invariants.py`.
- Do not edit `legacy/invariants.json` or `legacy/baseline/`. They are
  the frozen record.

## current/ — new design (Trinidad panel)

Same machinery as before, adapted to the current panel: the ladder
drops `PCA_dim_90pct` and `lPCA` (see `metrics.py DROPPED_FROM_PANEL`).

Not yet pinned. Blocked on the team decision for the MLE neighbourhood
size (k=10 vs k=20, see docs/notes/calibration-history.md). To
bootstrap after that call:

1. Run `study_panel.py`, `study_design_null.py`, `calib_estimators.py`
   once; copy their outputs into `current/baseline/`.
2. `.venv/bin/python robustness/current/check_invariants.py --full --pin`
3. Derive tolerances: `robustness/current/calibrate_tolerances.py`.
4. Commit `current/invariants.json` and `current/baseline/`.

Then routine use:

```
.venv/bin/python robustness/current/check_invariants.py        # fast
.venv/bin/python robustness/current/check_invariants.py --full # + reruns
```

A FAIL is a question, not a verdict: find the diff that moved the
number, then either fix the bug or re-pin with a written reason.
