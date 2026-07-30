# About the definition of a persona

## 1. The literal definition

A persona **is** the system prompt. Literally. That string is the object; everything
else in this document is an attempt to represent it in a form we can compare,
measure, and do geometry on.

## 2. What the system prompt does

What it does is shift a distribution: from `p(response)` to `p(response | persona)`.

To quantify that, note this is mathematically the same as a shift between two
expectations over questions `q ∈ Q`:

```
p(r)           =  E_{q~Q} [ p(r | q) ]
p(r | persona) =  E_{q~Q} [ p(r | persona, q) ]
```

We could work directly in the space of probability distributions and compute a KL
divergence or a Wasserstein barycenter / optimal transport map to characterise this
shift. But that is much more complicated and weird to define — Wasserstein needs a
ground metric on the vocabulary that we do not have, and KL barycenters collapse to
things dominated by content rather than character.

## 3. Instead: work with the generator

In ML we say we have the distribution if we have the generator. Define

```
g : Q → Δ(R),        g(q) ~ p(response | q)
```

Then we are looking for the shift `f` such that

```
f(g, persona) = g_persona,      g_persona(q) ~ p(response | persona, q)
```

So quantifying how the responses change given a system prompt is **the same thing**
as quantifying how to modify the model at inference time to steer toward that
persona without actually putting the system prompt in.

## 4. How the field estimates it — and where the problems start

To estimate `f` (call the estimate `f̂`), take a subset of good-quality questions,
generate, and average the residual stream to see how things change once the system
prompt is added. That is what Persona Vectors and the Assistant Axis do.

Two problems.

### Problem 1 — an average is not the proper centroid in this space

If the vectors were normalized, the centroid should be computed **on the sphere**,
not in the ambient space. They are not normalized — so we still need to decide how
to average properly, probably in a more cosine/angular geometry.

**What we found when we actually looked.** The residual stream is not normalized at
all (mean ‖h‖ runs 0.44 → 91 across layers; these are pre-norm architectures, so
RMSNorm is applied to a *copy* entering each sublayer and the stream itself is an
unnormalized running sum). But chasing "is the mean the right centroid?" turned up
something worse than a geometry error: the first token is an **attention sink** with
norm ~112× the median, and because it is identical in every prompt while the mean
divides by the sequence length `T`, the whole average was dominated by a constant
over `T`. **PC1 of our point cloud was `1/prompt_length` at r = 0.998.** So the mean
was not merely the wrong centroid — it was measuring the wrong quantity. Details and
numbers in [`diagnostics/README.md`](../../diagnostics/README.md).

### Problem 2 — the estimate is nowhere near the expectation

Much more important: we are averaging ~256 questions in an infinitely dimensional
space. We are way off the actual expected value. No amount of care about *how* we
average fixes the fact that `f̂ ≠ f`, and it never will be.

## 5. The exact `f`

We would like the exact shift `f`, not an estimate `f̂`.

That exact `f` is either **prepending the system prompt**, or — numerically
pointwise identical — **using its KV cache**. Because attention is causally masked,
the prefix's keys and values at every layer and head depend only on the prompt:

```
{ K_P^(l,h), V_P^(l,h) }        for all layers l, heads h
size:  2 · L · |P| · d_model
```

This is an exact representation of the persona, capturing 100% of its information
and giving 100% faithful steering when used at inference time. No questions, no
averaging, no approximation.

### The real problem: non-fixed dimensionality

We cannot compare personas in this form. The representation is not a `d_model`
vector — its size depends on the system prompt length `|P|`. Two personas with
different prompt lengths live in different spaces.

(There is a second, subtler issue: the KV cache is only defined **up to permutation
of its slots**, since the attention readout `Σᵢ softmaxᵢ(q·kᵢ)·vᵢ` is a sum. So the
natural object is a *set* of key–value pairs, not a matrix, and its coordinates are
not individually meaningful.)

## 6. Why `f` is not a fixed shift in the weights

This is the key structural fact, and it is what makes a single averaged vector
hopeless rather than merely imprecise.

For one attention head, with query `q = q(x)`, prefix `(K_P, V_P)` and context
`(K_C, V_C)`, the following is an exact identity:

```
Attn(q, [K_P;K_C], [V_P;V_C])  =  λ(x)·softmax(q K_Pᵀ)V_P  +  (1−λ(x))·softmax(q K_Cᵀ)V_C

                     Σ_{i∈P} exp(q·kᵢ)
       λ(x)  =  ──────────────────────────────────
                Σ_{i∈P} exp(q·kᵢ) + Σ_{j∈C} exp(q·kⱼ)
```

so the shift the persona applies to the residual stream is

```
Δh(x)  =  λ(x) · [ softmax(q K_Pᵀ)V_P  −  softmax(q K_Cᵀ)V_C ]
```

`K_P` and `V_P` are constants, but **`λ(x)` and the softmax weights depend on the
input**. Different queries put different attention mass on the prefix, and read
different rows of `V_P`. So `f` is exact but it shifts the model **differently for
every input** — it is a shift *function*, not a fixed `ΔW`.

Three consequences:

1. **A single averaged residual-stream vector is not just imprecise, it is the wrong
   object class.** It has no input-dependence at all. It is a constant approximation
   to a function. That is why it "would never be good enough" — more questions
   shrink the estimation error but not the approximation error.
2. **No static `ΔW` can be exact.** A weight modification applies one fixed map at
   every position; it has nowhere to store the set `(K_P, V_P)` that the
   query-conditional readout requires. The exception is precise: drop the softmax
   and in *linear* attention the prefix becomes an exact additive low-rank weight
   update, independent of `x`. That is the regime where "prompting ≡ weight update"
   results hold. With softmax it is an approximation.
3. **Everything the model can express through the prefix lives in `conv(V_P)`**, the
   convex hull of the stored value vectors, because the softmax weights form a
   probability distribution. This bounds what any compression of the prefix can
   reach.

**How big is the gap?** That is exactly what Experiment 1 measures.

## 7. Two ways forward

We already know from the lossy representation that persona space is low-dimensional
and that it is useful for steering. So:

### A. A better lossy representation: LoRA, with fixed and controlled dimension

Keep it lossy but with much more controlled dimensionality and far more faithful
than a single vector. Train it by **distillation**, which lets us estimate against
thousands of synthetic inputs instead of 256 questions:

```
min_θ  E_{x~D} [ KL( p(· | persona, x) ‖ p_θ(· | x) ) ]
```

Note the expectation is now over the **loss**, not over the representation. In
principle the minimiser matches the conditional at every `x`; the expectation only
decides how questions are weighted when we cannot match all of them.

**Open question from §6:** if the true shift really is strongly input-dependent, a
plain LoRA (a fixed linear map, no gate) may be the wrong class too, and we should
consider a **LoRA conditioned by a gate on `x`** — something of the form
`Δh(x) = λ_φ(x)·B A h`, which reintroduces the gate that a prefix has for free.
Experiment 1 tells us whether this is necessary.

Prior art worth knowing: **ReFT / LoReFT** already does low-rank learned
interventions on hidden states (this is "steering with a learned operator instead of
a fixed vector"), and hypernetworks that emit a LoRA from a text description exist.
Neither has been done *for personas* and then asked about the geometry of the
resulting adapter space. That is our question.

### B. Stay in KV-cache space and do the geometry there

The more mathematically rigorous route: keep the exact object and fix the
comparability problem directly.

- **Soft prefixes of fixed length `k`.** Exactly `2·L·k·d_model` parameters, the
  same shape for every persona, and free-form (not restricted to being the KV of
  real tokens), so strictly richer than a `k`-token hard prompt. Exact when
  `k ≥ |P|`, lossy below — and the loss is measurable.
- **`conv(V_P)`** as the object of study: compare personas by their reachable sets
  rather than by a point. Comparing sets of different cardinality is an optimal
  transport problem, which is at least well-posed.
- **Functional view.** The KV cache defines a map `x ↦ Δh(x)` that shifts the weight
  space. The domain is infinite-dimensional, so work in the **image** of that map,
  which need not be.

## Experiment 1 — how far is the true shift from a fixed vector?

The cheapest decisive test, and the one that chooses between §7A-with-gate and
§7A-without.

For a persona, run each question **with** and **without** the system prompt, and
take the difference in the residual stream at the *shared* question-token positions
(so positions align and content cancels):

```
Δh_ℓ(q, t)  =  h_ℓ^{sys}(q, t)  −  h_ℓ^{no-sys}(q, t)
```

Stack over questions and positions into a matrix and look at its singular spectrum.

- **Approximately rank 1** → `Δh(x) ≈ λ(x)·v`: one fixed direction with variable
  gain. Then a persona *is* a direction, and constant-α steering is wrong only in
  the gain — the fix is a gate, not a new geometry.
- **High rank** → the gated-constant picture is dead, the direction itself moves
  with the input, and §7B is the honest route.

The fraction of variance in the top singular direction is a direct, quantitative
answer to "how much is the actual difference" between `f̂` and `f`. No generation, no
training, ~40 lines.
