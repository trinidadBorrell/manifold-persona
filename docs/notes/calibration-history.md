# Calibration gate: recorded numbers and provenance

Moved out of `exploratory/per_persona/calib_estimators.py` comments (2026-08-12).

Trinidad's recorded calibration (PCA-50 working space, her data copy):

- PCA_dim_90pct: worst 20%, bias +0.11, spread 1.00 (exact at d=3, 8)
- MLE: worst 17%, bias −0.34, spread 0.31 (smoothest, but under-reads)
- PCA_participation_ratio: worst 18%, bias +0.22, spread 1.40

History: the gated list previously held MLE alone, chosen from the RAW 2048-dim
diagnostic, and was not revisited after the panel moved to PCA-50 (amendment A2).
Denoising rescues the two PCA-threshold measures; dim_90pct is the most accurate
estimator in the panel. No run was mis-gated by the correction (2026-07-31).

The three failures are unchanged in every working space: lPCA 67%, TwoNN 77%,
PCA_dim_95pct 120%. They stay in the panel but must never be quoted as a
dimension.

Her evidence files (`CALIBRATION.md`, `CALIBRATION-RESULTS.md`) never reached
the repo (the old .gitignore bug); requested 2026-08-07, and again 2026-08-12
with her run's `calibrated to real data: ... per-dim noise` line. Our
measurement on the current canonical 40q cloud: MLE worst error 38%
(seed-robust; see `output/calib_sensitivity_2026-08-07`,
`robustness/baseline/calibration_L19.json`). Root cause of the 17-vs-38 gap:
data provenance (her calibration cloud's noise floor), not code — the gate is
noise-conditional.
