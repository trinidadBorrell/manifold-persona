# Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior

Wurgaft, Rager, Kowal, Shyam, Feucht, Bhalla, Haklay, Bigelow, Sarfati, McGrath,
Lewis, Merullo, Goodman, Fel, Geiger, Lubana. arXiv:2605.05115v1 (6 May 2026).
Goodfire + Stanford, UCL, Northeastern, Harvard, Technion IIT.
Code: <https://github.com/goodfire-ai/causalab/tree/main/manifold_steering>

**Pages read:** 1–11 of 11 (§1 through §4). Appendices A–C are cited throughout and
are **not** in the downloaded PDF. §5+ / conclusion are not in this PDF either.

---

## ⚠️ Read this first

**This is not a persona paper.** It is a general paper about the geometry of
activation space, and every experiment uses **structured conceptual domains**, not
characters or traits:

- **cyclic**: days of the week, months of the year ("What is four days after Monday?")
- **sequential**: letters, ages ("What letter comes four after M?", "Alice is 7,
  Bob is 5 years older. How old is Bob?")
- **in-context-learned graphs**: 5×5 grid and 9×9 cylinder, from Park et al. 2025b's ICLR task
- a **video world model** with physical dynamics (mountain car)

The word "persona" does not appear in the parts of the paper I read. It is the
right *methodological* reference for this repo — it is the paper that says "steering
is choosing a geometry, not a direction" — but citing it as evidence about persona
manifolds specifically would be wrong. That link is a hypothesis this repo would
have to test, not something the paper establishes.

Default model: **Llama 3.1 8B**, activations from **layer 28**.

---

## Setup (§2.1)

Two spaces:
- **activation space** `A = R^n`
- **behavior space** `Y = Δ^|Z|`, the open probability simplex over a conceptual
  domain `Z` (plus an "other" class for off-concept probability mass) — i.e. the
  model's output distribution restricted to the tokens in `Z`.

For a class of queries sharing an answer, they average hidden activations and
output distributions into **activation centroids** and **behavior centroids**.

## Fitting the manifolds (§2.2)

- **Activation manifold `M_h`**: reduce activations to **64 dimensions via PCA**,
  compute concept centroids, fit **cubic splines** through the centroids.
- **Behavior manifold `M_y`**: same, but first map each centroid from the simplex
  into **Hellinger space** via `p ↦ √p`, which linearizes the geometry (Hellinger
  distance becomes ordinary Euclidean distance), fit splines there, then square
  back to recover valid distributions.
- For the 2-D in-context tasks (§4) they use **thin plate splines** instead of
  cubic splines — the 2-D analogue.

## Result 1 — the two manifolds are approximately isometric (§2.3)

Both manifolds recapitulate the conceptual structure: weekdays and months form a
**loop**; letters and ages form an **open curve**. The circle in *behavior* space is
described as a novel finding, arising because sharply-peaked output distributions
put the remaining mass on the target's neighbors.

Correlation between geodesic distance on `M_h` and geodesic distance on `M_y`
(cumulative Euclidean on the activation side, cumulative Hellinger on the behavior side):

| task | manifold-geodesic r | straight-line r |
|---|---|---|
| weekdays | 0.99 | 0.89 |
| months | 0.89 | 0.53 |
| letters | 0.999 | 0.71 |
| ages | 0.999 | 0.36 |

## Result 2 — manifold steering produces natural behavior (§3.1–3.2)

Two interpolation strategies between endpoint activations `h*_0`, `h*_1`:

    π_lin(t) = (1−t)·h*_0 + t·h*_1                      (linear steering)
    π_m(t)   = s((1−t)·u_0 + t·u_1),  u_i = s^{-1}(h*_i)  (manifold steering)

where `s : R^k → A` is a parameterization of `M_h`. Linear interpolates in `A`;
manifold interpolates in the manifold's **intrinsic coordinates** and maps back, so
it stays on `M_h`.

K = 50 intervention points, 16 prompts sampled from the task's input distribution.
Manifold steering shifts probability mass **smoothly through adjacent concepts**
(Mon→Tue→Wed→Thu); linear steering **"teleports"** mass between non-adjacent
concepts, and sometimes puts more mass on unrelated tokens than on any concept
near the path midpoint.

Quantified by a cumulative **Bhattacharyya energy** to the nearest point on `M_y`,
`E_BC(γ) = ∫ d_BC(γ(t), M_y) dt` (lower = more natural):

| task | manifold | linear |
|---|---|---|
| weekdays | 0.34 ± 0.03 | 0.93 ± 0.11 |
| months | 0.36 ± 0.01 | 1.09 ± 0.06 |
| letters | 2.42 ± 0.07 | 6.95 ± 0.27 |
| ages | 5.21 ± 0.09 | 13.49 ± 0.29 |

Average improvement **2.8×**, all comparisons p < 0.001.

## Result 3 — the reverse direction (§3.3)

**Pullback**: take a geodesic on the *behavior* manifold, then optimize (L-BFGS,
within the first 32 dims of the 64-D PCA subspace) for an activation path that
*induces* it. The recovered activation path resembles the manifold-steering path,
scored by an intrinsic R²:

| task | R² pullback | R² linear |
|---|---|---|
| weekdays | 0.77 ± 0.03 | 0.42 ± 0.07 |
| months | 0.75 ± 0.04 | 0.32 ± 0.05 |
| letters | 0.78 ± 0.04 | 0.23 ± 0.05 |
| ages | 0.47 ± 0.05 | 0.24 ± 0.01 |

## The framing that matters for this repo (§3.4)

Steering = choosing a **Riemannian metric** `G` on activation space and taking its
geodesic. `L_G(π) = ∫ √(π̇ᵀ G(π) π̇) dt`. Definition 1 gives three:

- `G_I = I_n` — **flat / linear steering**. Encodes no knowledge of activations or outputs.
- `G_E(h) = (α·e^{−E(h)} + β)^{−1} I_n` — **density geometry / manifold steering**, where
  `E(h) ∝ −log p(h)`. Cheap to move where activations are dense, expensive off-manifold.
- `G_F(h) = J_F(h)ᵀ g_y(F(h)) J_F(h) + ε I_n` — **pullback**, `F : A → Y` the
  activations→behavior map, `g_y` a metric on `M_y` (here the induced Hellinger metric).

Their claim: `G_E` (from internal activations) and `G_F` (from outputs) are derived
from different sources yet converge on approximately the same paths, so `M_h` and
`M_y` are "two images of the same conceptual geometry, related by an approximate
Riemannian isometry."

> "This recasts the core problem of steering from finding the right *direction* to
> finding the right *geometry*." (abstract)

## Result 4 — factored control (§4)

On the 2-D ICLR grid/cylinder tasks, the fitted manifold has a **2-D intrinsic
coordinate system**, and steering along one coordinate moves that dimension with
**minimal off-target impact** on the other. Linear steering keeps teleporting.
Geodesic correlations across the two manifolds: **r = 0.99** for both 5×5 grid and
9×9 cylinder; linear-path correlations drop to 0.90 (grid) and 0.81 (cylinder).
