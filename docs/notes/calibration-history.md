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
the repo (the old .gitignore bug); requested 2026-08-07 and 2026-08-12.

RESOLVED (2026-08-12): the 17-vs-38 gap is the MLE neighbourhood size.
Before the idim repair (587e7f5, 2026-08-03), `skdim.id.MLE(K=...)` was
silently ignored (skdim 0.3.6 reads the constructor K only when
`neighborhood_based=False`), so MLE ran at the default `n_neighbors=20`.
The repair passes `n_neighbors=min(10, n-2)=10` to `fit()`. On the canonical
40q cloud (byte-identical to her HF copy), same seeds and noise:

- k=20 -> worst relative error 0.171 (her recorded 17%)
- k=10 -> worst relative error 0.382 (our 38%; the d=3 -> 4.15 signature)

Not data, not environment. The earlier noise-floor hypothesis moved the number
too but was not the historical cause. The k change likely also explains the
MLE-vs-axis correlation gap (her -0.701 vs our -0.66).

Open team decision: k=20 calibrates at n=200 (passes the 0.20 gate), k=10
does not. The repair targeted the small-n sweep (n=10-25) where k=20 is
degenerate. Options: per-use-case neighbourhood, or keep k=10 and drop MLE
from the gated set.
