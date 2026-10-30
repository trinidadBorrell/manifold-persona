# pga_linearity — how linear are the PGA directions, and how much do they differ from PCA? (Oct 29)

**Status:** run 2026-10-29 (Part A of `plans/2026-10-29-geometry-predicts-position-and-misalignment.md`, local)  ·  **Data:** `$PM/E1`, view `last`, layers 19, 25, 32  ·  **Outputs:** `$PM/PGALIN/`, `output/29-oct/partA/`

## Question
At the axis level: are Fletcher's principal geodesic directions (PGA, `pm_geometry.components.pga`: k = 10 kNN graph, graph Fréchet mean, 2-hop log map) the PCA directions in disguise, and are the principal geodesics themselves straight? The 20-10 studies showed PGA is not more interpretable than PCA and that PGA directions from about 10 on match PCA as on flat nulls; this measures the overlap and the linearity directly.

## Method
All in the exact PCA coordinates of the 276 persona vectors (SVD of the centred matrix, all 275 components), so PCA directions are the coordinate axes and PGA directions are the columns of V. Nulls: 20 Gaussian and 20 shuffled clouds with the same per-axis variances (`curvature_null.run.null`), graph and PGA refit per draw.

| # | Measure |
|---|---|
| A1 | principal angles between span(PCA 1..k) and span(PGA 1..k), k = 1…20: mean cos², Grassmann distance |
| A2 | per PGA direction: closest PCA axis and \|cos\|; share of the direction inside span(PCA 1..10 / 20 / 50) |
| A3 | per PGA direction: max \|Spearman ρ\| between persona scores on PGA k and on PCA j, j ≤ 50 |
| A4 | the principal geodesic as a graph polyline (shortest paths from the Fréchet mean to the two most extreme personas inside a 30° cone around ±u_k): straight-line R² (PC1 share of its points), arc / chord, max distance from the chord / chord, mean turning angle. Second variant: the cone personas ordered by coordinate (uninformative when the cone is empty: it falls back to all personas) |
| A5 | R² of the 276 vectors reconstructed in span(PGA 1..k) against span(PCA 1..k) |
| A6 | A1 and A4 on the two question-half persona sets |
| A7 | log-map fidelity: cos between each persona's log-map vector (direction of the first two hops) and its true displacement from the Fréchet mean; against the null, and against the persona's \|PGA 1 coordinate\| |

## Files
| File | What |
|---|---|
| `run.py` | One layer per condor job → `$PM/PGALIN/pgalin_last_L<L>.json`, `frame_last_L<L>.npz` |
| `analyse.py` | `tables.md`, `summary.json`, F12–F17 into `output/29-oct/partA/` |

## Commands
```bash
condor_submit jobs_condor/pgalin.submit                       # 3 CPU jobs (x86, repo venv), minutes
python -m pga_linearity.analyse --dir $PM/PGALIN --out output/29-oct/partA
```

## Result
See `output/29-oct/partA/tables.md` (numbers) and the figures.
- **PGA 1 is PCA 1** at every layer (|cos| 0.84–0.91, Spearman 0.92–0.97; the k = 1 overlap is above the null band at layers 25 and 32).
- **From k = 2 on, the PGA span is no closer to the PCA span than on a flat cloud**: mean cos² of the principal angles is inside the null band for k = 2…20 at all three layers. PGA 2 is the one direction with low linear content (share inside PCA 1–10: 0.58–0.60 at layers 19 and 25, below the null band).
- **The PGA spans lose more variance than flat nulls do**: the R² gap to span(PCA 1..k) is 0.06–0.12 and above the null band for k ≥ 5 at all layers.
- **The principal geodesics are not straight.** At layer 25 the PGA 1 and 2 polylines have straight-line R² 0.50 / 0.65 (null 0.64–0.82), arc / chord 4.5 / 4.7 (null 1.8–2.6) and bend farther from their chord than any null draw; at layers 19 and 32 they are inside the null band.
- **The log map itself is faithful.** cos between a persona's log-map direction and its true displacement from the Fréchet mean is 1 by construction for personas 1–2 hops away (most of them) and 0.7–0.8 at 3–5 hops; the real distribution is inside the null range (median 0.86–1.00, 10th percentile 0.69–0.71). The real graph needs slightly more hops (mean 2.2–2.7 against ≤ 2.5 on nulls). So the bending is in the geodesic polylines, not in the per-persona coordinates.
