# S0: three test papers read in full (05/11/2026)

**The one-line answer:** None of the three papers gives a ready test for "more curved by stage". Only Song and Müller is a flat-vs-curved test, and its p-value is not valid on a cloud like ours.

Exploratory notes, not an external claim. "Inference" marks my reading. A paper does not state it. Paraphrase is the default; the file has one direct quote.

---

## Read status

| Paper | Read level | Last line reached | Extraction limits |
|---|---|---|---|
| Robinson, Dey, Chiang, arXiv 2504.01002v3 (NeurIPS 2025) | full, appendices A.1–A.10 and Tables 4–10 | 1558 of 1558 (page 30) | Figures 1–8 not visible. Non-Latin tokens are blank in Tables 5–10. |
| Gilbert, O'Neill, arXiv 2309.13478v2 | full, Appendix A and references | 1687 of 1687 | Figures 1–9 not visible. Most results exist only in those figures. |
| Song, Müller, arXiv 2505.09844v2 | full, Supplement S.1–S.7, proofs, references | 4687 of 4687 (page 57) | Figures not visible. Some formulas are scrambled; I rebuilt them from context and from the proofs. |

**What I could not read:**

- Any figure. Power curves (S-M Figs. S.3, S.4) and all CA-PCA curves are known only from the text.
- Robinson's code repository (`kb1dds/stratified_estimator`) and ZIP supplement. The text gives no values for vmin, vmax or W.
- External proofs that the papers rely on: Dubey and Müller 2019, Lin and Müller 2021 Lemma S.7, Aaron and Bodart 2018, Robinson et al. 2025 Thm 1, Lim et al. [23].

---

## Summary

| Question | Robinson 2025 | Gilbert and O'Neill | Song and Müller 2025 |
|---|---|---|---|
| Type | Hypothesis test per point | Estimator. No test, no null, no p-value. | Hypothesis test on the whole cloud |
| Output | p-value per point | Mean local dimension | ρ̂, T, confidence interval |
| What it asks | Does local dimension change with radius? | What is d, if curvature bends local PCA? | Is the sign of intrinsic curvature non-zero? |
| Smooth curved surface | Stays in the null | Gives d. Reports no curvature. | Rejects, with sign |
| Flat | Stays in the null | Gives d | Does not reject. This does not prove flat. |
| Bent curve or rolled sheet (extrinsic bend) | Stays in the null (inference) | Gives d | Reads as flat (§5.1 Example 1) |
| Clusters | Not tested. Likely rejects. No label. | Not tested. Two manifolds only (§4.4). | Not tested. Excluded by assumption. |
| Noise | Not tested on synthetic data | Small noise only. Random images: same as PCA (§4.3). | Pushes ρ̂ toward 0 in low dimension (§4.2, S.1.4) |
| Largest setting with known truth | n = 3000 in R³ | n = 20,000 in R²⁵; n = 200 in R¹²⁴⁴¹⁶ (airplane, d = 1) | n = 1000, p = 100, intrinsic dimension 2 |
| Runs on 800 cells? (inference) | Yes, but it tests the wrong thing | Yes, as a description only | Yes, but use no p-value from the formula |

No paper compares geometry across checkpoints. No paper has a clump control.

---

## 1. Robinson, Dey, Chiang: Token Embeddings Violate the Manifold Hypothesis

### Test statistic and null (§1.1 Table 1, §4 Algorithm 1)

For each point:

1. Sort distances to all other points. Row index = count v (Monte Carlo volume). Entry = radius r(v).
2. Keep rows vmin to vmax.
3. Local slope s(v) of log v against log r, by a 3-point centred difference (`numpy.gradient`). The slope is the dimension estimate.
4. Run a two-sample t-test on adjacent sliding windows of size W along v.
   - Manifold test: p-value for a constant slope.
   - Fiber bundle test: p-value for a slope that falls as v rises.
5. Apply Holm-Bonferroni to each set of p-values. α = 10⁻³ (A.6).

| Test | H0 at point ψ, ball radius r < τ | H1 |
|---|---|---|
| Manifold | One unique dimension at ψ | No unique dimension at ψ |
| Fiber bundle | Dimension does not rise as r rises | Dimension rises at some r |

The basis is Theorem 1 (§3.1). For a fiber bundle, log volume against log radius is piecewise linear, and slopes can only fall as r rises. The result is asymptotic for small r (A.5).

The paper does not state the t-test variant. It does not state how many windows enter each point's p-value, or how they merge. §5.2 says "Bonferroni"; Algorithm 1 says Holm-Bonferroni.

Failure to reject carries no information. In the paper's words: "Failing to reject the null is uninformative" (abstract). A rejection also gives no alternative and no singularity type (§7).

### Assumptions

| Assumption | Where | Status | ID |
|---|---|---|---|
| Positive reach; only slope changes below reach count | Table 1, §7 | Stated | S4, N1 |
| Compact, finite-dimensional, finite volume, with boundary allowed | Thm 1 | Stated | S2, N8 |
| Connected, so one dimension | §1.1, §2 | Stated | S3 |
| Smooth one-to-one embedding, full-rank Jacobian | §2, fn. 3 | Stated | S4 |
| Sampling density = normalized volume form | §3.1, A.5 | Stated | S6, N2 |
| Counts converge to volume. Authors call this a strong assumption. | §3.1 | Stated | S5, N2 |
| Inherited Euclidean distance | §2 | Stated | S8 |
| Small radius = noise, large radius = signal; fiber dimension is the same everywhere (homoscedastic) | A.4 | Stated, as one possible reading | S7 |
| At most one slope change inside vmin–vmax, ideally | A.6 | Stated | N1 |
| vmin, vmax and reach come from inspection of a sample of the same data | §5.2, A.6, A.10 | Stated, but consequence tacit | N1 |
| One global vmin, vmax for all points, though reach and density vary | §5.2 | Tacit | N1 |
| Adjacent slopes are valid t-test samples | §4 | Tacit. Inference: they share points and come from sorted distances, so they are correlated. | N3 |
| The point set is fixed; no resampling needed | §4 | Stated | N10 |
| Claim is about the input embedding matrix, not activations | A.2 | Stated | S9 (scope) |

Thm 2 (singularities reach the output) needs a generic transformer and a dimension bound (§3.2). Table 3 uses the median estimates.

### Conditions where it was shown to work

| Data | n | Ambient dim | Noise | Result | Section |
|---|---|---|---|---|---|
| Unit sphere | not stated | 3 | none stated | No rejection; p near 1 | §5.1 |
| Cusp surface z⁸ + x² + y² = 0 | 1200, of which 41 on cusp and boundary | 3 | none stated | Cusp found, p < 5×10⁻⁸. Boundary missed; larger vmax finds it but loses specificity. Fiber test: no rejection. | §5.1 |
| Lollipop | 3000 | 3 | none stated | 75 and 58 points near the two singularities, p < 10⁻⁵. Fiber test rejects at 2 points. | §5.1 |
| GPT2, Llemma7B, Mistral7B, Pythia6.9B tokens | 50,257; 32,016; 32,016; 50,254 | 768 or 4096 | no truth known | 66, 33, 40, 54 manifold rejections | §5.2 |

The paper reports no false-positive rate and no power. There is no synthetic case above dimension 3 and no noisy synthetic case. vmin, vmax and W are not given for any run. Each token run took about 12 hours, mostly the distance matrix (A.3).

**Caution on the headline counts (inference).** The rejection counts are close to n × α: 66 vs 50, 33 vs 32, 40 vs 32, 54 vs 50. A.1 itself says about 0.1% of tokens, and α is 0.1%. Uncorrected uniform p-values would give counts this size. The paper says the listed p-values are corrected (Table 2 caption, §5.2), and p near 1 on the sphere fits that. The text alone cannot settle this. Check the code before anyone cites the counts. Also, Table 8 repeats token IDs (19929 three times, 3903 twice), and Table 4 gives a different ID for one of those tokens.

### What it can and cannot separate

| Truth | Behaviour | Source |
|---|---|---|
| Smooth curved manifold | In the null. Curvature is not the target; §2 says curvature does not extend to singular spaces and uses reach in its place. | §2, §5.1 |
| Flat | In the null (inference; no flat synthetic case) | — |
| Cusp, pinch, dimension change | Rejects | §5.1, A.7 |
| Boundary | Rejects the manifold test only with larger vmax. Fiber test stays in the null. | §5.1 |
| Clusters | Not tested. Inference: a gap gives a slope fall, then a rise, so the fiber test rejects. It gives no "cluster" label. | §7 |
| Noise | Homoscedastic noise fits the fiber null. Heteroscedastic noise rejects it. So a rejection can mean noise, not shape. | A.4 |
| High local curvature | A rejection means singular or high curvature. The test cannot tell these apart. | §5.2, §6 |

### Could it run on our 800 cells? (inference)

**It runs, but it does not answer our question.** The 800 × 800 distance matrix takes seconds.

- It tests irregularity, not curvature. A cloud that gets smoothly more curved stays in the null at every stage.
- The design itself puts steps in the volume curve. Each cell has 39 role mates and 19 question mates. Expect slope changes near v = 19 and v = 39 at every cell, with or without a shape.
- With 800 cells, vmin–vmax is short, so W is small. No calibration exists at n = 800, D = 4096.
- Use: at most an irregularity screen, compared with surrogates. Do not use it as the main test.

---

## 2. Gilbert and O'Neill: CA-PCA

### Test statistic and null

**There is no test statistic and no null.** The paper says "test" for the estimator. It reports means and standard deviations of d̂ (§4), with no p-value or interval.

Estimator (§3.2 eq. 3.10, 3.12, Algorithm 3.1), for a base point x:

1. Take the k + 1 nearest neighbours. Set r = (‖x − x_k‖ + ‖x − x_{k+1}‖)/2.
2. λ̂ = eigenvalues of the covariance of x₁…x_k, divided by r².
3. For each d, let T_d = Σ_{j>d} λ̂_j. Set λ_i^(d) = λ̂_i + (3d+4)/(d(d+4)) · T_d for i ≤ d.
4. d̂ = argmin_d [ ‖(1/(d+2) − λ_i^(d))_{i≤d}‖₂ + 2 T_d ].
5. Repeat for N random base points. Return the mean of d̂.

Basis (§3.1 Prop. 3.1), for a quadratic graph in a unit ball, with A and B built from the eigenvalues of the quadratic forms:

- Tr Σ₁ = d/(d+2) − 2d·A/((d+2)²(d+4)) − B/(d+2)² + O(Λ⁴)
- Tr Σ₂ = 2A/(d+2)² + O(Λ⁴)

The flat baseline is eq. 2.1: d eigenvalues equal to 1/(d+2), the rest 0. The method does not fit a quadratic. It moves tail eigenvalues back onto the top d, using a calibration.

### Assumptions

| Assumption | Where | Status | ID |
|---|---|---|---|
| Data on or near a lower-dimensional manifold | §1 | Stated | S2 |
| C² smoothness, locally a graph | §1, §3.1 | Stated | S4 |
| Uniform density on the manifold; a smooth density counts as locally constant | §3.3, §4.2 | Stated | S6, N2 |
| Curvature small at neighbourhood scale; O(Λ⁴) dropped | §3.2 | Stated | S4, S5 |
| A = B: eigenvalues of the quadratic forms have random signs | §3.2 | Stated, called a best guess | N7 |
| All top eigenvalues equal | §3.2 | Stated, called somewhat reasonable | N7 |
| Factor 2 in (3.10) and in the regularizer (3.12): chosen for simplicity, not derived | §3.2 | Stated | N7 |
| Top d sample eigenvalues come from Σ₁, the rest from Σ₂ | §3.2 | Stated | S5 |
| No boundary (preferred in experiments) | §4.1 | Stated | N4 |
| One d for the cloud; output is a mean | Alg. 3.1 | Tacit. §4.4 shows the mean hides a mix. | S3 |
| Tail eigenvalues are curvature, not noise | §3.2 | Tacit. Theory has no noise term. | S7 |
| Sample size suits the scale: k > d and k much less than n | §3.2, §4.6 | Partly stated | S5 |
| Euclidean distance selects neighbours | §3.2 | Tacit | S8 |
| Independent sample points | §3.2, §4 | Tacit | S6 |

### Conditions where it was shown to work

| Data | True d | n | D | Noise | Reported result | Section |
|---|---|---|---|---|---|---|
| Klein bottle | 2 | 400 | 4 | none | Converges faster than PCA | §4.2 |
| S⁷ | 7 | 5000 | 8 | none | Faster than PCA | §4.2 |
| Curve (4.1) | 1 | 100 | 8 | none | Rises with k, slower than PCA | §4.2 |
| SO(5) | 10 (inference; number not printed) | 20,000 | 25 | none | Slightly better than PCA at k ≤ 50 | §4.2 |
| SO(3) ⊕ T³ | 6 | 20,000 | 15 | none | Does not converge to 6 | §4.2 |
| Airplane, 1 angle | 1 | 200 | 124,416 | none | Stays near 1–2; PCA blows up after k = 13 | §4.3 |
| Airplane, SO(3) | 3 | 10,000 | 124,416 | none | Not accurate; better than PCA | §4.3 |
| Isomap faces | about 3, with boundary | 698 | 4096 | real images | Closer to 3 than PCA and LB | §4.3 |
| Random images | no manifold | 698 | 4096 | all noise | Same as PCA | §4.3 |
| Circle ∪ S³ | 1 and 3 | 200 + 200 | 4 | none | Mean about 2; large SD shows the mix | §4.4 |
| Klein + error | 2 | 400 | 4 | uniform ±ε per coordinate, ε to 1.4 (a = 10, b = 5), k = 20, 200 trials | Robust to a limit, then reads 4 | §4.5 |
| SO(5) + error | 10 | 20,000 | 25 | ε = 0.1 | Slightly worse, still close | §4.5 |
| Surface (4.2) | 2 | 400, k = 15, N = 200 | 4 | none | Good to about a = 6; PCA fails after a = 4–5 | §4.6 |
| Klein, n varies | 2 | 80–560, k = 20 | 4 | none | Worse than PCA at lowest n; better from n = 200 | §4.6 |

Suggested k: 10–25 for low d, 30–50 for higher d (§5). N = 200 unless stated. §4.6 gives u, v in [0,1] in eq. 4.2 and [−1,1]² for sampling; the paper does not reconcile these.

### What it can and cannot separate

| Truth | Behaviour |
|---|---|
| Curved vs flat | No output for this. It reports d only. PCA and CA-PCA also differ on a flat surface at a = 0, and the paper cannot explain why (§4.6). So the PCA − CA-PCA gap is not a clean curvature signal (inference). |
| Clusters | Not tested. Inference: with k inside a clump, it reports clump dimension. |
| Mix of dimensions | Partly, through a large SD, and the paper only says it may help (§4.4). |
| Noise | Small error is tolerated (§4.5). Large error thickens the shape to full dimension. Random images give the PCA answer (§4.3). |
| Boundary | Reads low for PCA by construction (§4.1). CA-PCA does better in two cases, cause unknown (§4.3). |

### Could it run on our 800 cells? (inference)

**It runs, as a description only.** It costs seconds. d̂ cannot exceed k − 1 (§4.1).

- If d is about 20, n/d is 40. The nearest paper case with d = 10 had n/d = 2000. The method was not shown at our ratio.
- k must exceed d but stays near one role block (39 role mates). k = 39–50 is about 5–6% of cells. The paper's d = 2 case needed k ≤ 10% of n.
- Tr Σ₂ could give a per-point A estimate, once d is fixed. This is an extrinsic bend read-out, which the paper does not propose. Noise in 4096 dimensions adds to the same tail, so it needs a noise floor. We have no repeat measurements on fixed text.
- A plateau in d̂ as k grows is the useful signal. Pure noise has no plateau (see My checks).

---

## 3. Song and Müller: Inference for Dispersion and Curvature of Random Objects

Warning on labels: the paper's condition (S1) is strict positivity. It is not draft assumption S1. Below it is "(S1-SM)".

### Test statistic and null (§4.1, §4.2)

- Fréchet variance V̂_F = (1/n) Σ d²(μ̂, X_i) (eq. 4). Metric variance V̂_M = (1/(n(n−1))) Σ_{i<j} d²(X_i, X_j) (eq. 5).
- Metric curvature ρ̂ = V̂_F / V̂_M − 1 (eq. 9).
- T_n = √n ρ̂ / σ̂, with σ̂² = âᵀ Σ̂ â and â = (−V̂_F/V̂_M², 1/V̂_M)ᵀ (eq. 10, 11). Σ̂ comes from eq. 6.
- H0: ρ = 0 (flat). H1: ρ ≠ 0, or one-sided. Reject when |T_n| > z_{α/2}. T_n tends to N(0,1) under H0 (Thm 4).
- Equivalent: the joint region for (V_M, V_F) with χ²₂ (eq. 7) must cross the line V_M = V_F.
- Intrinsic version T_{I,n} (eq. 13, Thm 5): replace d with d_I, estimated by graph shortest paths. Algorithm 1 joins X_i, X_j when d ≤ r, then runs Dijkstra.
- **Key detail:** the intrinsic Fréchet mean is an argmin over sample points ξ ∈ X_n, so it is a medoid (§4.2). The ambient version searches all of M (eq. 4).

Direction of the logic: flat gives ρ = 0; strictly negative gives ρ < 0; strictly positive gives ρ > 0 (Prop. 4). The converse is not proved. ρ̂ = 0 does not prove flat, and the paper says so (§5.1, §5.2). Power tends to 1 only if √n ρ_n → ∞ (§4.1).

**The ambient test is empty for our data (inference from §1).** With Euclidean d and a free mean, V_F = V_M, and ρ̂ = −1/n for any shape. I confirmed this numerically. Only the intrinsic version carries information.

### Assumptions

| Assumption | Where | Status | ID |
|---|---|---|---|
| X₁…X_n i.i.d. | §1, §2, proofs in S.4 | Stated | S6 |
| P strictly positive on M, or on A with P(A) = 1: (S1-SM), (A0) | §3.2, §4.2 | Stated | S6 |
| Separable, complete, geodesic space; unique geodesics | §1, §3.2, Thm 3 | Stated | N5 |
| A simply connected | §1, §4.2 | Stated | N5, S3 |
| Unique Fréchet mean: (F1), (A2) | §2, §4.2 | Stated | N5 |
| Curvature bounded with one sign (Hadamard for Thm 2, non-negative for Thm 3) | §3.2 | Stated | N6 |
| Total boundedness and entropy: (F0), (F2), (A1), (A3) | §2, §4.2 | Stated | N8 |
| Moments and non-degeneracy: (M0)–(M3), (A4) | §2, §4.2 | Stated | N8 |
| Metric d is given and suitable; choice is an open problem | §7, S.6 | Stated | S8 |
| Compact k-dimensional manifold, K-geodesically smooth: (B1) | S.3 | Stated, for the distance rate | S4, N9 |
| P is ε-standard; a density bounded below is enough: (B2) | S.3 | Stated | S5, S6 |
| Ball size r_n in the rate of Thm S.1 | S.3 | Stated; r is never reported in any run | N9 |
| Graph distances can replace d_I in Thm 5 | §4.2, S.3 | Tacit. See below. | N9 |
| Medoid replaces the Fréchet mean with no effect on the limit | §4.2, proof of Thm 5 | Tacit | N11 |
| Noise small; noise can break the assumptions | S.5 | Stated | S7 |
| ρ can be compared across data sets | — | Not claimed. S.7 says its value depends on P. | S13 |

**Two theory gaps (inference).**

- Thm 5 uses the true d_I. Thm S.1 bounds the plug-in error by max{(log n/n)^(2/(3k)), n^(−1/2)}. The first term falls below n^(−1/2) only for k = 1. So no theorem gives N(0,1) for the computed T when k ≥ 2. At n = 800 the first term is about 0.20 for k = 2 and 0.85 for k = 20 (constants unknown).
- The medoid adds about n^(−2/k) to V̂_F in flat space. √n times this tends to 0 only for k < 4. So the bias does not vanish on the √n scale for k ≥ 4, and it is always positive.

**Shapes that break conditions (inference).** A full circle, sphere or torus fails (F1). It also fails (M2), because E[d²|X] is constant. Mixed-sign curvature fails the one-sign bound, and ρ can cancel to 0.

### Conditions where it was shown to work

| Data | Intrinsic dim | n | Ambient | Noise | Result | Section |
|---|---|---|---|---|---|---|
| Hemisphere + truncated Gaussian | 2 | not stated | R³ | σ = 1/32 to 1/2 | Positive for small σ; interval reaches 0 as σ rises | §4.2 Fig. 2 |
| Gaussians by rotation in W₂(R²) | 1 | 100 | W₂ | none | Compatible with flat | §5.1 |
| Hemisphere B₁, hyperboloid B₂, plane B₃ | 2 | 1000 | R³ | σ = 0.1 | 95% CIs (0.080, 0.125), (−0.095, −0.075), (−0.014, 0.042) | §5.2 Table 1 |
| SPD 3×3, six metrics (ambient test) | known | 100 | SPD | none | Sign matches theory | S.1.1 |
| Hemisphere (ambient geodesic) | 2 | 50 | S² | none | Positive at 99% | S.1.2 |
| D_κ cap vs flat D₀ | 2 | 200, 500, 1000; 500 runs | R³ | none | Size near 0.05 at κ = 0; power rises with κ, n | S.1.3 |
| Cap E_p in R^p | 2 | 200, 500, 1000 | p = 3 to 100 | σ = 0.05, 0.1, 0.2, or 3/(10√p) | Power falls with p at fixed low noise; rises at fixed SNR. **Power only: no flat case.** | S.1.4 |
| Gait | not stated | 52 per group | 2×2 SPD | real | Healthy: compatible with flat. Orthopedic: negative at α = 0.01. Stable over 3 metrics. | §6.1, S.6 |
| Energy | not stated | 120 | S²₊ | real | Close to flat | §6.2 |
| MNIST digits | not stated | 5421–6742 | R⁷⁸⁴ | real | All ten digits positive, no truth known | S.2.1 |

S.2.2 (temperature) reports metric variances only. It has no curvature test. The flat null was checked for size only at k = 2, in R³, with no noise. Exact size and power numbers exist only in figures.

All ten MNIST digits read positive. This may be medoid bias, not curvature (inference; see My checks).

### What it can and cannot separate

| Truth | Behaviour | Source |
|---|---|---|
| Positive vs negative vs flat 2-D surface | Yes, at n ≥ 200–1000, σ ≤ 0.1 | §5.2, S.1.3 |
| Size of curvature | No. Sign only. ρ depends on P. S.7 gives a relation only for special cases, with a fixed mean. | S.7 |
| Bent curve | Reads flat: ρ_I = 0 for a 1-D set | §5.1 |
| Rolled sheet, cylinder patch | Reads flat. The test is for intrinsic curvature only (inference). | §4.2, §7 |
| Clusters | Not tested; excluded by simple connection. Graph may split, and d_I is then infinite (inference). | — |
| Non-convex flat region or tree-like support | Can read negative, from the shape of the support and not from a bent surface (inference). | — |
| Noise | Low dimension: pulls toward flat (§4.2). High ambient dimension: untested for size, and see My checks. | §4.2, S.1.4 |

### Could it run on our 800 cells? (inference)

**It runs, but the formula p-value is not valid here.**

- Not i.i.d. Cells are 20 selected roles × 40 questions. σ̂ treats them as 800 independent draws, so T is too large. The population is not defined either (S6).
- The medoid bias is large at our likely dimension, and it mimics positive curvature.
- The answer depends on r, and the paper never reports r.
- A bent 1-D arc of roles would read flat. So the test would miss the "nonlinear" sense of curved.
- A change in ρ̂ across stages can come from a change in P or in spread, not curvature (S.7, S13).
- The paper has no two-sample test for ρ. Stages are paired on the same cells, so the joint CLT does not apply directly.

**Possible safe use:** ρ̂_I as a descriptive number, per checkpoint. Calibrate it against same-spectrum surrogates and clump surrogates at the same r. Use a bootstrap over roles and questions, paired across checkpoints. Sweep r.

---

## New assumption IDs

| ID | Assumption | Papers | Our case (inference) |
|---|---|---|---|
| N1 | Scale window: the analyst picks a radius or neighbour range below reach, from the same data. | Robinson | Fix the range before the run, or calibrate on surrogates |
| N2 | Sampling density is uniform on the surface, or locally constant. Stronger than S6. | Robinson, Gilbert and O'Neill | False. Roles were selected. |
| N3 | Adjacent local slopes act as independent samples. | Robinson | Unverified. Inflates p-values if false. |
| N4 | No boundary. | Gilbert and O'Neill; Robinson rejects at a boundary with larger vmax; S-M Cor. S.1 assumes none, yet its hemisphere and square have one | Likely false |
| N5 | Support is connected and simply connected; geodesics and Fréchet mean are unique. | Song and Müller | Untested. Clumps break it. |
| N6 | One curvature sign across the support. | Song and Müller | Unknown |
| N7 | Quadratic forms have random-sign eigenvalues, equal top eigenvalues, and heuristic factors of 2. | Gilbert and O'Neill | Unknown |
| N8 | Regularity: compact or totally bounded, entropy bound, finite moments, non-degenerate variance. | Song and Müller, Robinson | Mostly harmless. (M2) can nearly fail for a symmetric cloud. |
| N9 | Graph shortest paths stand in for intrinsic distance at a good r, with no short cuts. | Song and Müller | Weak at high k and n = 800 |
| N10 | The point set is the object itself; no sampling error. | Robinson | Conflicts with our wish for confidence intervals |
| N11 | A sample point can replace the Fréchet mean. | Song and Müller | False at high k |

---

## Corrections for the draft in ../LITERATURE.md

Not applied. This task writes one file only.

| Draft text | Finding |
|---|---|
| "Curved" table: extrinsic bend, method "Quadratic fit (Gilbert and O'Neill)" | CA-PCA does not fit a quadratic or measure curvature. It estimates d. |
| "Curved" table: intrinsic, "dispersion test" | Correct, but sign only, and it reads bent curves as flat. |
| S6: "positive density" for Robinson | Too weak. Robinson models density as the volume form (N2). |
| S4: "(C2)" | Means C² smoothness, not a labelled assumption. |
| S2, S3, S4 lines on Robinson | Confirmed: §1, Table 2, §1.1. |
| Summary: no flat-vs-curved test with clump and noise control | Still holds for clumps. Song and Müller has a noise study, but only in low dimension or with power only. |

---

## My checks (not from the papers)

Quick simulations, run in memory on 05/11/2026 with `.venv/bin/python`. Seeds fixed, scripts not saved, six replicates per row. So they are hints, not results. n = 800 in all cases.

**Song and Müller intrinsic test.** I used eq. 9, 13 and the medoid from §4.2. r is a multiple of the smallest radius that connects the graph. Cells show mean ρ̂ and rejections at 5% out of 6.

| Cloud | r × 1 | r × 1.25 | r × 2 |
|---|---|---|---|
| Hemisphere (positive control) | +0.139, 6/6 | +0.147, 6/6 | +0.145, 6/6 |
| Flat uniform, k = 2 | +0.019, 3/6 | +0.006, 4/6 | +0.002, 0/6 |
| Flat Gaussian, k = 10 | −0.009, 5/6 | −0.002, 4/6 | +0.133, 6/6 |
| Flat Gaussian, k = 20 | −0.037, 4/6 | +0.045, 4/6 | +0.274, 6/6 |
| Pure noise, D = 4096 | −0.044, 3/6 | +0.930, 6/6 | +0.930, 6/6 |
| 20 equal clumps | +0.109, 4/6 | +0.915, 6/6 | +0.915, 6/6 |
| Additive role + question means in 256 dims, noise 0.1 (no bend) | −0.052, 6/6 | +0.816, 6/6 | +0.754, 6/6 |

Reading (inference): T is not N(0,1) on flat data at k = 10 and 20 with any r tried. k = 2 also over-rejects at the two small r. Flat, noise, clump and additive clouds can all give a larger ρ̂ than the true hemisphere. The additive row is closest to our crossed design, and it has no bend. The sign of a false rejection changes with r. Ambient check: Euclidean d with a free mean gave ρ̂ = −0.00125 = −1/n.

**CA-PCA.** Mean d̂ over 100 base points. My implementation of eq. 2.2, 3.10, 3.12.

| Cloud | k = 20 | k = 39 | k = 50 | k = 100 |
|---|---|---|---|---|
| Pure noise, D = 4096 | 19.0 | 38.0 | 49.0 | 99.0 |
| Flat Gaussian, d = 20 | 16.7 | 21.3 | 24.8 | 34.4 |
| Sphere S¹⁹ | 15.6 | 19.3 | 19.7 | 20.0 |
| Sphere S² | 2.0 | 2.0 | 2.0 | 2.0 |
| Additive role + question means in 256 dims, noise 0.1 | 19.0 | 38.0 | 49.0 | 98.5 |

Reading (inference): noise gives d̂ = k − 1, with no plateau. A curved d = 20 cloud plateaus; a flat Gaussian keeps rising. The additive crossed cloud, with no bend, looks like noise to CA-PCA.

---

## Open items

| Item | Needed |
|---|---|
| Robinson: is 66, 33, 40, 54 near n × α a real result or a correction issue? | Read `kb1dds/stratified_estimator`. |
| Robinson: vmin, vmax, W values | Same repository |
| S-M: exact size and power values | Figures S.3, S.4 from the PDF |
| CA-PCA: values on random images and Isomap faces | Figures 5c, 6c from the PDF |
| Medoid bias: rerun with saved script, more replicates | Owner decision; belongs in the design step |
