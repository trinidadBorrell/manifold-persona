# Robustness invariants

This is a smoke alarm for published results. It is not a gate.
Run it after a merge. It recomputes the trusted numbers with the
current code and tells you if any moved.

## Run it

```bash
.venv/bin/python robustness/check_invariants.py          # fast, ~3 min
.venv/bin/python robustness/check_invariants.py --full   # + panel and calibration reruns (slow)
```

Exit 0: all invariants hold. Exit 1: at least one moved.

## When a pin fails

A FAIL is a question, not a verdict. Two answers are possible:

1. **Bug.** Your change broke a computation. Compare your diff
   against the source script of the failing pin.
2. **Discovery.** The change is intentional. Accept the new numbers:

```bash
.venv/bin/python robustness/check_invariants.py --pin
git diff robustness/invariants.json   # review what moved
```

Commit the diff and say why in the message. Old pins stay in git
history. Nothing is lost.

## Where tolerances come from

Tolerances are measured, not guessed. `calibrate_tolerances.py`
derives them from two perturbations:

- **Role bootstrap** (2000 draws): how much each statistic wobbles
  when the 276 roles are resampled. A drift smaller than that wobble
  is not a meaningful change.
- **Seed sweep**: the calibration gate plants seeded synthetic
  manifolds. Different seeds give different planted clouds. The
  spread across seeds sets that pin's tolerance.

Rule: `tol = max(2 * SD, floor)`. The evidence sits in
`invariants.json` under `tol_provenance`. Rerun after big data or
pipeline changes:

```bash
.venv/bin/python robustness/calibrate_tolerances.py \
    --calib-glob '<sweep-dir>/calib_seed_*/data/calibration_L19.json'
```

## What each kind means

| Kind   | Rule                                        |
|--------|---------------------------------------------|
| exact  | value must stay within tolerance of the pin  |
| sign   | only the direction must hold                 |
| struct | a fixed comparison that defines the finding (never re-pinned) |

Sign pins survive pivots that move magnitudes. That is by design:
research numbers may move, but they must move the predicted way.

## Depth column

Each row reports how deep the recompute went:

| Depth    | Meaning                                                    |
|----------|------------------------------------------------------------|
| raw      | recomputed from the activation clouds with current code    |
| cached   | statistic recomputed from `baseline/` inputs (tests aggregation code only) |
| artifact | read from a committed baseline file; `--full` upgrades it to raw |

`baseline/` holds the reference inputs from the 2026-08-07
`robust-geometry` run (see `exploratory/per_persona/figures/robust-geometry/GATE_OVERRIDE.md`).
The 100-draw design null is never rerun here; it is too slow.

## Notes

- The prompt cloud is known contaminated (attention sink). The
  `prompt.*` pins load it on purpose: it is the contrast cloud.
  The checker sets `MP_ALLOW_UNCLEAN=1` for that step only.
- The `mle_worst_calib_error > 0.20` pin asserts a FAILURE we
  measured. If it ever fails, calibration got fixed on your
  machine. Investigate — do not just re-pin. This is the open
  dispute with Trinidad's recorded 17%.
