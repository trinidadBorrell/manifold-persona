# Literature check for the stage manifold test (05/11/2026)

Eight papers read in full as extracted PDF text. Figures were not inspected. I read PersonaManifold, Wu 2018 and Jensen 2020. Two agents read the other five and reported to me; I did not read those five myself. "Inference" marks a point that a paper does not state.

## Summary

- None of the eight papers measures curvature across LLM training stages.
- None gives a flat-vs-curved test with a clump control and a noise control.
- Exploratory reading only. This is not a full review and is not an external claim.

## Papers

| Paper | Setup | Relevant result | Limit for this test |
|---|---|---|---|
| Xu et al. 2026, PersonaManifold, [arXiv 2609.34571](https://arxiv.org/abs/2609.34571), NeurIPS 2026 | Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, Instruct only. About 2K personas, 40 questions, one mean vector per persona, last response token, one layer (14 or 16), PCA to 99% variance. | Intrinsic dimension 15–23. Ollivier–Ricci curvature differs by persona type (Fig. 2). Geodesic distance beats Euclidean on behaviour triplets by +5.1 to +6.1 points (Table 2). | No stages, no base model. No flat, Gaussian or clump null for curvature. Synthetic checks (App. M) test geodesic recovery on curved shapes only. Layer chosen by the same TCR metric, on 200 held-out personas. |
| Li et al. 2025, Tracing Representation Geometry, [arXiv 2509.23024](https://arxiv.org/abs/2509.23024), NeurIPS 2025 | OLMo-2 1B/7B and Pythia pretraining. Llama-3.1-Tülu-3.1-8B post-training. Last token, last layer. | RankMe up in SFT and DPO, down in RLVR (§3.4, Fig. 6A). Pretraining is not monotonic. | Linear spectra only. No effect sizes for stages in the text. No OLMo-2-7B stage data. |
| Karkada et al. 2026, Symmetry in language statistics, [arXiv 2602.15029](https://arxiv.org/abs/2602.15029), ICML 2026 | Word embeddings from co-occurrence. Gemma 2 2B for comparison. | Statistics alone give curved shapes (Prop. 1, 3). Shapes survive removing direct month-month statistics (Fig. 4). | No checkpoints. Roles have no known coordinate. Inference: spectrum and curve are linked in this theory, so claim "structure beyond second moments". |
| Jensen et al. 2020, Manifold GPLVM, [arXiv 2006.07429](https://arxiv.org/abs/2006.07429) | Synthetic tori, SO(3), S3; fly and mouse head direction. | Latents placed with half the neurons, score on the held-out neurons at held-out conditions (§3.1). Correct manifold wins on 30/30 synthetic sets. | Baselines are Euclidean GPLVMs, not PCA. No shuffle null. |
| Wu et al. 2018, Latent odor manifold, [NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/17b3c7061788dbe82de5abe9f6fe22b3-Abstract.html) | 22 odors, about 500 neurons, 10 repeats. | Co-smoothing (§4): latent from observed neurons, score on unobserved neurons. Uses repeats to model noise. | No PCA score in the co-smoothing test; PCA appears only in the 2D plot. |
| Tiblias et al. 2026, SMDS, [arXiv 2510.01025](https://arxiv.org/abs/2510.01025), TMLR | Labelled tasks, at most 500 examples per task. Three base/instruct pairs. | A cluster hypothesis competes directly with curved ones (Table 9). Instruction tuning did not substantially change the best shape (D.8). Date score rises for Llama and Qwen, falls for Gemma. | Needs distance labels, so it cannot run on roles. Bootstrap iterations act as replicates (inference). |
| Sarfati et al. 2026, Shape of Beliefs, [arXiv 2602.02315](https://arxiv.org/abs/2602.02315) | Llama-3.2-1B, numeric sequences. OLMo-2-7B appears in Fig. 9, qualitative only. | Failure of a flat additive model read as curvature (App. E.4). | No curvature statistic or null. No training stages. |
| Bhalla et al. 2026, SAEs and concept manifolds, [arXiv 2604.28119](https://arxiv.org/abs/2604.28119) | Llama-3.1-8B, layer 19, last token. | Local flat pieces can mimic a curve (Fig. 5). "Manifold" is partly convention (fn. 1). | No clump control on real data. No training stages. |

## Points for the design

| Point | Source | Status |
|---|---|---|
| Place a held-out point with one part of the data, score it on another part. | Jensen §3.1, Wu §4 | Owner decision pending |
| Compare with a clump model as well as a flat model. Twenty equal clumps need 19 dimensions. | SMDS Table 1, D.4 | To propose |
| Hold out whole roles and questions, not random cells, so a clump model gets no free role mean. | SMDS gap (inference) | To propose |
| Covariance-matched Gaussian surrogate per checkpoint. Same spectrum, no nonlinear structure. | Karkada, inference | To propose |
| Text floor on the same cells: state 0 and text features. | Karkada | To propose |
| Report RankMe and alpha-ReQ per checkpoint on the same cells. Do not assume a monotonic rise. | Li et al. | To propose |
| Second text source. E3 `ex2` is already in pod C. | Li et al. §3.4, inference | To propose |
| Rescale each checkpoint to unit RMS. Sweep rank. | Bhalla | To propose |
| Bootstrap over roles and questions, paired across checkpoints. Fix layer and token first. | SMDS, inference | To propose |
| Check for additivity: question-mean removal can leave interaction that looks curved. | Sarfati App. E.4, inference | To propose |
| Token choice: PersonaManifold uses the last token. July E8 found no curved advantage at the last response token, on Qwen2.5-3B-Instruct. Pod C stored mean-pooled replies. | PersonaManifold §3.1; E8 report | Note in scope |

Needs a GPU, so out of scope for this design: a random-init OLMo-2 floor; the numeric positive control from Sarfati Fig. 9.

## Assumptions the field makes (05/11/2026, draft)

Drawn from the eight papers above and three partly read papers: Robinson et al. 2025 ([arXiv 2504.01002](https://arxiv.org/abs/2504.01002), abstract to Algorithm 1), Gilbert and O'Neill ([arXiv 2309.13478](https://arxiv.org/abs/2309.13478), abstract, introduction, assumption lines), Song and Müller ([arXiv 2505.09844](https://arxiv.org/abs/2505.09844), abstract, introduction, assumption lines). This is a sample, not a survey of "most researchers". The last column is my reading.

| ID | Assumption | Who states or uses it | Challenge | For this test |
|---|---|---|---|---|
| S1 | A concept is a direction (linear representation). | Persona vector work, Assistant Axis | PersonaManifold, Karkada | The rival to beat |
| S2 | Data lie near a low-dimensional manifold. | Gilbert and O'Neill; Robinson says it is often tacit | Robinson rejects it for token embeddings | Assumption, not a finding |
| S3 | One connected manifold with one dimension. | Implicit in global fits, including GPLVM | Robinson: dimension varies by point | Clumps and patches test this |
| S4 | Smooth, low curvature, positive reach, so local patches look flat. | Gilbert and O'Neill (C2), Robinson (reach) | Any finite sample fits some manifold (Robinson) | A fit alone is not evidence |
| S5 | Sampling is dense relative to curvature. | PersonaManifold kNN graphs; Gilbert and O'Neill on small samples | 20 roles | Weak here |
| S6 | Points are random draws with positive density on the manifold. | Song and Müller A0, Robinson | Li et al. call i.i.d. unrealistic | False here: 20 selected roles, crossed with questions |
| S7 | Noise is small and additive around the surface. | Jensen (independent), Wu (structured) | Wu: noise is correlated | Measure it per checkpoint |
| S8 | Euclidean or cosine distance in the residual stream is the right metric. | Nearly all | Scale and whitening change results | Rescale; state the metric |
| S9 | One pooled vector per item keeps the structure. | PersonaManifold, July E8 | PersonaManifold limit; E8 last-token result | Mean pooling is a choice |
| S10 | A continuous latent lies behind the points. | Jensen, Wu, Karkada, Sarfati | Roles have no known coordinate | Untested |
| S11 | The geometry comes from the model, not only the input statistics. | Often implicit | Karkada | Needs a text floor |
| S12 | Geometry matters for behaviour. | PersonaManifold, Bhalla | Not tested in pod C | Out of scope; say so |
| S13 | Checkpoints can be compared with one measure. | Li et al. | Scale, noise, rank differ by stage | Project axiom A4; needs calibration |

"Curved" has three meanings in these papers:

| Meaning | Example | Method |
|---|---|---|
| Nonlinear: the cloud leaves a flat subspace | July E8, Karkada, Sarfati | GPLVM vs PCA |
| Extrinsic bend: a surface bends in activation space | Circle or cylinder; intrinsic curvature can be zero | Quadratic fit (Gilbert and O'Neill) |
| Intrinsic: distances within the surface are non-Euclidean | PersonaManifold (Ollivier–Ricci), Song and Müller | Graph geodesics, dispersion test |

## Team work that overlaps (06/11/2026)

Read from `origin/causal-study` at `c611214` and vault page `Reference/Team code - Trini and Gio.md`. README text only; Trinidad's outputs are not in git, so her numbers are not checked here.

| Work | Owner | Overlap with this test |
|---|---|---|
| `src/curvature_null` (20 Oct): ridge quadratic bend test, kNN path trend, flat Gaussian and shuffled nulls, sphere-cap controls. Qwen3-8B, 276 role means, last token. README: curved at layers 19–32. | Trinidad | Direct. It answers "is role space curved" on one Instruct model. Reuse the test. |
| `src/pga_linearity` (29 Oct): PGA vs PCA with the same nulls. | Trinidad | Partial. Curved-method detail on the same cloud. |
| PersonaManifold measures (`1f86ee9`), close read of the paper (T-064). | Trinidad | Known to the team since 2 Oct, when the team called it a scoop. |
| Paper split (17 Sep): "Is there a manifold in personas?" is Part 3. | Trinidad | The existence question is hers. Part 2, where the space emerges, is Brendon's. |
| Persona basis, LoRA personas, judges (`origin/persona-definition`, last commit 16 Oct). | Giovanni | No shape-across-stages work found. His LoRA code is off-repo, so this is unseen. |
| 1B stage study, 13–14 Aug (T-012). | Brendon | Same idea with Trinidad's metrics. Failed the audit; Q1 results before 5 Oct are invalid. |

Not found in git: any curvature measure across training stages. Off-repo work is unknown (open question from T-082).
