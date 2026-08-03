# Curveball Steering: The Right Direction To Steer Isn't Always Linear

Raval, Song, Wu, Harrasse, Phillips, Barez, Abdullah. arXiv:2603.09313v3 (22 Mar 2026).

**Pages read:** 1–8 of 38 (§1 through §6 part 1). §6 remainder, §7+, and appendices
A–E are **not** read; anything from them is marked `[not read]`.

Why it's here: this is the closest existing answer to "is there a better space to
steer in than a global linear direction?" — and unlike Manifold Steering it operates
on **behavioral/personality attributes**, which is much closer to personas.

---

## Claim 1 — activation space is measurably non-Euclidean (§2.1)

Method: train an **ensemble of VAEs** on activation vectors from a fixed layer for a
concept-specific dataset. Each decoder induces a **pullback Riemannian metric** on
the latent space, `g(z) = J_μ(z)ᵀJ_μ(z) + J_σ(z)ᵀJ_σ(z)`, aggregated over the
ensemble by Monte Carlo averaging (which inflates distances where decoders
disagree, i.e. in high epistemic uncertainty).

Then compute a **distortion ratio** over 500 i.i.d. activation pairs:

    R_distortion = E[ d_geo(z1,z2) / d_Euc(z1,z2) ]

A locally-Euclidean space would concentrate at R = 1. Fig. 2(b) reports mean
distortion ratios far above 1 and **concept-dependent**:

| concept | mean R (Fig. 2b) |
|---|---|
| self-awareness (general) | 31.52 |
| wealth-seeking | 46.46 |
| corrigible-more | 24.10 |
| power-seeking | 38.83 |

Two conclusions the authors draw: (i) activation space is non-Euclidean, and
(ii) *different concepts have systematically different amounts of distortion*, so
the geometry is concept-specific, not a single global property of the model.

## Claim 2 — Curveball steering (§3)

Uses **polynomial kernel PCA** (pKPCA), `k(x,y) = (x·y + γ)^p` with degree **p ∈ {2,3}**.
Choice justified two ways: a polynomial kernel captures **global** structure (an RBF
kernel would preserve local structure instead), and it keeps the parametric model
low-degree so it generalizes.

Algorithm 1:
1. Center activations, run KernelPCA → `Z` (top `k` eigenvectors define `φ: R^d → R^m`).
2. Class means `z_0, z_1` in kernel space; steering direction `ẑ = (z_1 − z_0)/‖z_1 − z_0‖`.
3. Per generated token: take the **last-token activation** `a_curr`, project
   `a' = φ⁻¹(φ(a_curr))`, keep the **residual** `r = a_curr − a'`, steer
   `a_target = φ(a_curr) + α·ẑ` in kernel space, invert, then **add the residual back**:
   `a_steered = φ⁻¹(a_target) + r`.

The inverse `φ⁻¹` has no closed form; they use **kernel-weighted pre-image
reconstruction** `[Appendix C, not read]`.

Preserving the orthogonal residual is the point: the manifold component gets steered,
the off-manifold component is carried through untouched. With **p = 1** the method
reduces exactly to standard linear PCA steering, so it is a strict generalization.

They also state four desiderata any nonlinear steering method must satisfy (§2.2) —
projectable, data-respecting, **functional** (`φ` must apply to *new* points, which
rules out t-SNE/UMAP/ISOMAP/Laplacian Eigenmaps), and approximately invertible.
This is a genuinely useful checklist if this repo tries its own nonlinear map.

## Claim 3 — curvature predicts when it helps (§4)

Synthetic manifolds: two classes on patches of an `m`-dim hypersphere of radius
`r = 10/κ`, randomly mapped into `R^512`, Gaussian noise σ = 0.01, curvature
`κ ∈ {0.1, 1.0, 5.0, 10.0, 20}`, steering strength `α ∈ [0, 20]`.

Result: at **κ < 2** linear and Curveball perform similarly. Beyond **κ ≈ 8** linear
steering "exhibits catastrophic degradation due to pushing the datapoints
off-manifold"; at **κ > 10** Curveball shows ~3× lower tangent-space deviation.
Curveball wins on target distance in **72.9%** of (κ, α) conditions.

## Claim 4 — it works on real behavioral traits (§5)

Models: **llama-3.2-1B-Instruct** (layer 10) and **phi-3.5-mini-Instruct** (layer 22).
Datasets: Anthropic "Advanced AI Risk" model-written evals extended with an
LLM-generated pipeline to **8k–10k conversations** per attribute.

Table 1 — Δp(behavior) on binary choice, and ΔJudge score (0–100) on open-ended:

| concept | Llama linear | Llama Curveball | Phi linear | Phi Curveball |
|---|---|---|---|---|
| self-awareness | 14% | **24%** | 0.6% | **25.4%** |
| wealth-seeking | 15% | **28%** | 2.3% | **6.7%** |
| power-seeking | 16% | **47%** | 2.9% | **14.9%** |
| corrigible | **21%** | 17% | 2.1% | **93.4%** |
| humorous | **54.9** | 28.2 | **85** | 75 |
| rudeness | **85.7** | 26.1 | 61.0 | **100** |
| excitement | **41.4** | 37.9 | 90.0 | 90.0 |
| sadness | 15.4 | **19.5** | 85.0 | **100** |

Honest reading: Curveball wins clearly on the **binary behavioral** concepts
(3 of 4, and by a lot on Phi), but on the **open-ended linguistic traits** it is
mixed and is *worse* on Llama for humorous/rudeness/excitement. The authors
acknowledge this: "not all behavioral features benefit equally from nonlinear
methods… open-ended responses may contain many additional dimensions of information
that impact the final geometry."

## Claim 5 — why it works (§6)

1. **Different regions of the activation manifold want different steering vectors.**
   k-means the negative-label activations of `corrigible-more`, compute a linear
   steering vector per cluster: those per-cluster vectors have cosine similarity
   ~0.28–0.66 with the global vector and cluster away from it (Fig. 5A/C). The
   global linear vector is a compromise.
2. **Curveball's magnitude is adaptive** — a uniform step in KPCA space maps back to
   a *varying* magnitude in ambient space (Fig. 5B), unlike a fixed-α linear vector.
3. Spearman correlation between Curveball vector magnitude and paired-class
   distance varies by concept (Table 2): self-awareness 0.54, power-seeking 0.43,
   wealth-seeking 0.38, corrigible-more **−0.42**, humor ~0.
