# curvature_null (Oct 20)

**Question.** Is persona space curved, or does it only look curved? The 20-10 interpretability result (PGA is never easier to read than PCA) only matters if the space is curved. The evidence so far is a path ratio of 1.74 and a TwoNN of 9.1, both from 276 points in 178 dimensions. A flat 8-D ball gives 1.47.

**Data.** Qwen3-8B, `$PM/E1`, view `last`:
- role means (276) at layers 12–32;
- per-answer points (11,040: answer minus the no-prompt answer to the same question) at layer 28.

**Plan.** `output/20-oct-AI-pilled/PLAN.md`. The decision rule was fixed before any real data was seen.

| Test | What |
|---|---|
| C, primary | Cross-validated R² of a ridge quadratic fit that predicts PC k from PCs 1–10 except k. A linear fit gives 0 by construction, so R² above the shuffled null is a bend. |
| A | kNN graph (k = 10) shortest path / straight distance, overall and per distance decile. Trend = the ratio in decile 10 / the ratio in decile 5. It grows with distance on a curved surface (arc / chord). |
| B | A at N = 276, 1,000, 3,000 and 11,040 per-answer points |
| D | Max \|Spearman ρ\| of PGA k against PCA 1–50, real against the nulls |
| E | TwoNN, real against the nulls |

The nulls are refit to each sample, with the same N and the same PCA variances. The Gaussian null is flat. The shuffled null permutes each PCA coordinate on its own. The controls are sphere caps of dimension 2 or 8, and the same caps flattened, with 1% or 10% noise.

## Files
| File | Role |
|---|---|
| `run.py` | Condor job: `--mode means --layer L`, `--mode answers --layer 28`, `--mode control` → `$PM/CURV/*.json` |
| `analyse.py` | Decision rule, `tables.md`, `summary.json`, `fig1`–`fig5` into `output/20-oct-AI-pilled` |

## Commands
```bash
condor_submit jobs_condor/curv_null.submit       # 23 CPU jobs: control, answers L28, means L12-32
python -m curvature_null.analyse --dir $PM/CURV --out output/20-oct-AI-pilled
```

## Result
Full write-up: `output/20-oct-AI-pilled/RESULTS.md`.
- **Controls:** all 8 are classified correctly (4 spheres "curved", 4 flat sets "looks flat").
- **Role means, layers 19–32: curved.** At L28 the bend R² reaches 0.69 against a null of 0.01, and the distance trend is 1.076 against 1.053. Layers 12–18 are bent but fail the trend.
- **The old path ratio of 1.74 is below the flat nulls** (1.81–1.85), so it was not evidence of curvature. TwoNN is 9.1 against 20.7 for the nulls.
- **PGA from about direction 10 on matches PCA exactly as it does on flat nulls,** so those directions differ from PCA by graph noise. PGA 2 is the only direction outside the null range.
