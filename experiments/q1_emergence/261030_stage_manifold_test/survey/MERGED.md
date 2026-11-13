# Merged survey: what the field assumes about curved representations (05/11/2026)

Sources: `S0_test_papers.md`, `S1.md` to `S6.md`, `../LITERATURE.md`. Protocol: `PROTOCOL.md`. This merge is mine. Paper-level detail, section numbers and URLs are in the strand files.

## Coverage

| Strand | Field | Counted reads | Stop rule | Saturated |
|---|---|---|---|---|
| S0 | Three curvature-test papers | 3 full | Fixed list | n/a |
| S1 | Statistics of the manifold hypothesis | 24 full, 1 main text | Cap | No |
| S2 | Neuroscience | 19 full, 6 main text | Cap | No |
| S3 | Deep networks | 25 full, 3 abstract (not counted) | Cap | No |
| S4 | LLM concepts | 26 full | Cap | No |
| S5 | Model validation | 25 full | Cap | No |
| S6 | Across training | 25 full, 1 abstract (not counted) | Cap | No |

- About 150 reads, plus the 8 papers in `LITERATURE.md`. I did not remove duplicates between strands.
- No strand reached the five-clean-reads rule, so no strand is saturated at paper level.
- At the level of the 14 groups below, my reading is that late papers mostly refined a group and did not open a new one. I built the groups after seeing the papers, so this check is weak.
- "Full" means the extracted text. Figures were not seen, and some table cells and derivations were skipped.
- Most papers were read by agents. I read PersonaManifold, Wu 2018 and Jensen 2020 myself. I spot-checked three agent claims against the source text and they matched: Schulte and Rügamer (ID of LLM embeddings is zero in a strict sense), Dabholkar and Barak (high co-smoothing allows extraneous dynamics), and Pellegrino et al. (sliceTCA for mixed covariability classes). I have not checked the rest.
- Blocked: several supplements, Brahma 2016, Cai 2021, SigClust original. Rate limits blocked some citation steps.
- Correction to `S6.md` N3: it says we have no repeats. We do: 5 wordings × 5 replicates per cell, nested, not 25 independent draws (`S2.md` N3).
- `S0_test_papers.md` includes small simulations that an agent ran unasked, with no saved script. They are not evidence here.

## The 14 assumption groups

Status: **Challenged** = at least one paper gives evidence against it, often in another domain. It does not mean refuted, and it does not mean most researchers reject it. **False here** = fails for our data by design. The last column is my inference.

The protocol searched for critiques on purpose, so the count of challenged rows reflects the search and not the balance of the field. The survey did not measure how many researchers hold each assumption.

| # | Assumption | Status | Main sources | For this test |
|---|---|---|---|---|
| A1 | A true manifold exists. | Challenged | Schulte and Rügamer: strict (Hausdorff) dimension is zero for finite token inputs; specific to LLMs, not vision. Fefferman: a test decides fit in a class. Whiteley: low-dimensional structure follows from generic latent models. | Claim better prediction than rivals, not that a manifold exists. |
| A2 | One connected surface with one dimension. | Challenged | Robinson, Brown, Park 2025, Shafran, Dooms | Roles may be polytopes, spokes or local pieces. |
| A3 | A continuous latent lies behind the points. | Untested for roles | Validated only with a measured latent (head angle). Gupta, Engels, Nava | No role coordinate exists. |
| A4 | Smooth, low curvature, and the right scale is known. | Challenged | Aamari (reach), Gurnee (kinks), Cacioli | Smoothness is a GP prior, so a smooth fit is no finding. |
| A5 | Dense, even, independent sampling. | False here | Almost all theory. Kaufman, Lim, Whiteley, Song and Müller | Roles and questions are the units. Effective n is near 20 for role claims. Crossed bootstrap needed (Sellam; Schütt). |
| A6 | Noise is small, additive, and separate from structure. | Challenged | Genovese, Binnie, Altan, Valeriani, Song and Müller | Noise moves shape measures toward flat or inflates dimension. Estimate noise per checkpoint from wordings. |
| A7 | Euclidean distance on raw activations is right. | Challenged | Quemy, Timkey, Schulte, Klabunde | Rogue dimensions and norms can decide distances. Unit RMS is not enough. Add a standardised arm. |
| A8 | One pooled vector at one layer stands for the model. | Challenged | Valeriani, Cheng, Bigoulaeva (OLMo-2 instruction vectors, not personas), PersonaManifold limit, July E8 | Report states 9, 17 and 23. Mean pooling is a choice. |
| A9 | Flat, clump and curved are separate and complete. | Challenged | Whiteley: a mixture is the same model with a finite latent. Pellegrino and De and Chaudhuri: low-rank interaction. Cohen: additive baseline. Brown, Shafran: local pieces. | Rivals: additive, low-rank role × question, local-subspace mixture, curved. |
| A10 | A better held-out fit shows the right structure. | Challenged | Shinn (PCA on smooth time series), Dabholkar and Barak, Gerber (principal curves), Lebedev (regression to arbitrary curves), Fu and Perry (column-split CV on correlated Gaussian noise), Hénaff (fit-then-measure bias), Owen and Perry (risk, not truth) | None of these tests a GPLVM on activations, so transfer is my inference. Run every null through the whole pipeline. Add a target that should fail. |
| A11 | The fitted latents and kernels can be read directly. | Challenged | GPLVM latents are not identifiable. Titsias, Lawrence, Diaconis (horseshoe), El Karoui (kernels act near-linear under his high-dimensional models, with p/n fixed; may not apply after PCA to 50 dimensions), ARD depends on the prior | Compare invariants. Keep all models in one framework: a linear-kernel GPLVM equals PCA. |
| A12 | Geometry comes from the model, not the input. | Challenged | Karkada, Saxe, Cai, Cui, Dujmović, Elsayed and Cunningham | Text floor on the same cells. |
| A13 | One measure compares across checkpoints. | Challenged | Gröger, Josse, Klabunde, Quemy, Davari, Murphy. Scores move with rank and d/n (Gröger: O(d/n) width confounder). | Calibrate per checkpoint. Inference from `S6.md`, no paper shows it: a row-pairing null will reject for checkpoints that read the same text, so it gives only a chance floor. Permute whole roles or questions. Use generative nulls. |
| A14 | Stage change is monotonic, general across text, and caused by the objective. | Challenged | Ansuini, Lee, Yin (non-monotonic). Ruppik (RoBERTa, local ID): fine-tuning lowered ID on its own dataset, standardized difference 1.19, against 0.08 and 0.1 on others. Wang §5: public Base–Instruct pairs confound objective, data and duration. | Do not test a trend alone. Second text needed. Say "stage", not "objective". Update size as a cause is an open question, not a finding. |

## Four meanings of "curved"

| Meaning | Plain sense | Method | Caution |
|---|---|---|---|
| Nonlinear | The cloud leaves a flat subspace | GPLVM vs PCA (July) | Also true of clumps and low-rank interaction |
| Extrinsic bend | A surface bends in activation space | Quadratic fit | A flat 1-D gradient with saturating similarity also bends (Diaconis) |
| Intrinsic | Distances inside the surface are non-Euclidean | Ollivier–Ricci, Song and Müller | Needs dense sampling. Reads a bent curve as flat. |
| Path | Turning angle along an ordered sequence | Hosseini, Hénaff | Needs an order; role cells have none |

## Gaps found by the survey

- No paper reached measures cloud curvature across base, SFT, DPO and RL. Other shape measures do cross stages: spectra (Li et al.), local ID (Ruppik), linear charts (Wang), task layers (Bigoulaeva).
- No paper reached gives an equivalence test or minimum detectable change for a representation measure.
- No interval for curved-minus-flat held-out error was found.
- No paper reached covers a crossed role × question design for shape.
- No test found separates curved, clumped and noisy at once. SMDS compares cluster and curved hypotheses, but it needs distance labels.

## Effect on the plan (inference)

- The claim narrows: "structure beyond a flat and additive description, in prediction", not "a curved manifold".
- The rival list grows from three to five, in one fitting framework.
- A stage claim needs a margin fixed before data, with power from role-level units.
- Pod C's stable map suggests stage effects may be small. Lu's same-role cosine above 0.99 (Gemma 2 27B, base reading instruct rollouts) points the same way, but raw cosines may reflect a shared mean, so it is weak support. A null result is possible and must be interpretable.

## Audit against source text (06/11/2026)

I checked these claims in the extracted paper text myself. I did not use a subagent, as this is a check of my own merge.

| Claim | Result |
|---|---|
| Schulte: strict dimension zero | Confirmed. Limited to token inputs; scope added. |
| Dabholkar: high co-smoothing allows extraneous structure | Confirmed. |
| Pellegrino: mixed covariability classes | Confirmed that the paper covers this. Low-rank detail not checked. |
| Whiteley: mixture is the same model with finite latent | Confirmed. |
| Lebedev: arbitrary-curve fit generalizes to held-out conditions | Confirmed. |
| Shinn: unseen data explained about as well as training data | Confirmed for PCA on smooth series. Scope added. |
| Gerber: test coordinates adapt to the fit | Confirmed for principal curves. Scope added. |
| Fu and Perry: k≥2 when correlation exceeds 0.5 | Confirmed, for Gabriel column-split CV. Scope added. |
| Ruppik: 1.19 vs 0.08 and 0.1 | Confirmed. RoBERTa, local ID. Scope added. |
| Lu: same-role cosine above 0.99 | Confirmed. Reading weakened. |
| Gröger: width confounder O(d/n) | Confirmed. |
| Diaconis: horseshoes from a gradient | Confirmed in outline. |
| El Karoui: near-linear in high dimension | Confirmed under his models. May not apply to us; scope added. |
| Altan: high noise inflates dimension | Confirmed in outline. The 200% figure was not checked. |

Overclaims corrected:

1. I wrote that the core assumptions are "mostly challenged". That reflects a search aimed at critiques. Withdrawn.
2. I wrote "no paper measures cloud shape across stages". Spectra, local ID and linear charts do cross stages. The gap is curvature.
3. I stated "row-pairing nulls always reject" as fact. It is an agent's inference.
4. I credited update size to Wang. Wang states a confound; update size is an open question from `S3.md`.
5. "Challenged" read as refuted. Many challenges come from other domains.

Not checked: every other paper-level claim in the strand files, including Hénaff, Karkada beyond the first read, and all other agent readings.
