# Methods: the two calculations behind the per-persona numbers

Two quantities carry most of the interpretation in this study, and both are easy
to misread. This documents exactly what they are, in the notation of the code
that produces them.

Throughout: a **role's cloud** is the set of points belonging to one persona —
`n_i` instruction phrasings × `n_q` questions, one point per cell, each a vector
in `h = 2048` dimensions. Two clouds have been run: 5×5 = 25 points/role (prompt
tokens) and 5×40 = 200 points/role (response tokens).

---

## 1. Within-role variance fractions

**Code:** `common.py::design_fractions` · **Reported by:** `01`, middle panel

### The question it answers

A role's points differ from each other for three possible reasons: the
instruction phrasing changed, the question changed, or the *combination* did
something neither factor predicts on its own. Only the third is persona
geometry. The first two are the experiment's grid, and would appear even if the
model had no per-persona structure at all.

### The calculation

Take one role's cloud `X` of shape `[n, h]` where `n = n_i · n_q`. Centre it:

```
Xc = X − mean(X)                     # subtract the role's own mean vector
SS_total = Σ ‖Xc‖²                   # total sum of squares, summed over all h dims
```

Form the two **marginal means** — for each level of a factor, the average of the
points at that level:

```
A_i = mean over the n_q points that share instruction i      # i = 1..n_i
B_j = mean over the n_i points that share question j         # j = 1..n_q
```

`A_i` is the instruction effect: how instruction `i` displaces the cloud on
average, having averaged the questions away. `B_j` is the same for questions.
Both are vectors in `h` dimensions, and both sets sum to zero because `Xc` is
centred.

Fit the **additive model** — grand mean plus the two effects, no interaction —
and call what it misses the residual:

```
fit_ij   = A_i + B_j
resid_ij = Xc_ij − fit_ij
```

The three fractions are each term's share of the total:

```
instr_frac       = Σ ‖A_i‖²   / SS_total
quest_frac       = Σ ‖B_j‖²   / SS_total
interaction_frac = Σ ‖resid‖² / SS_total
```

(the sums run over all `n` points, so `A_i` is counted once per point at level
`i`). These are Frobenius sums over all 2048 dimensions at once — each effect is
a *vector*, not a per-feature quantity.

### Why the three sum to exactly 1

Because the grid is **balanced**: exactly one point per (instruction, question)
cell. That makes the instruction and question effects orthogonal, so there is no
shared variance to argue about and no attribution ambiguity. Verified on role
`alien`:

```
instr 0.760 + quest 0.233 + interaction 0.007 = 1.0000000000
⟨A[instr], B[quest]⟩ = −3.6e−15                       (orthogonal)
rank(A[instr] + B[quest]) = 8 = (5−1) + (5−1)         (the additive design rank)
rank(Xc) = 24
```

`grid_shape()` asserts the balance holds for every role before any of this runs.

### Why interaction is the number that matters

A two-factor grid **forces** rank `(n_i − 1) + (n_q − 1)` before the model
contributes anything — 8 for the 5×5 cloud, 43 for the 5×40. Any intrinsic
dimension measured on gridded points reports that structure unless interaction
carries real variance. So:

| | prompt 5×5 | response 5×40 |
|---|---|---|
| instruction | 68.4% | 1.8% |
| question | 31.0% | 80.3% |
| **interaction** | **0.6%** | **17.7%** |
| verdict | the grid | real role-specific structure |

At 0.6% a persona-free design null reproduces every result. At 17.7% it does
not. This single number is why the two clouds get opposite conclusions.

### What it is not

- Not a per-feature variance. It is one scalar per term, pooled over 2048 dims.
- Not a significance test. It is a decomposition of observed variance; the
  design null in `01` is what supplies the reference.
- Not comparable across clouds without the grid shape. A bigger `n_q` gives the
  question factor more levels to spend variance on.

---

## 2. Intrinsic dimension vs closeness to the Assistant

**Code:** `04_id_vs_axis.py` · **Reported by:** `04`

### The two "closeness" measures

Both are computed on **role mean vectors** — one point per role, its cloud
averaged, in the full 2048-dim space.

**`axis_proj`** — signed projection on the Assistant Axis
(`manifold_persona/common.py::assistant_axis`):

```
axis = mean(points of role "default") − mean(all role points)
axis = axis / ‖axis‖                                  # unit vector
axis_proj(role) = mean(role) · axis
```

High = Assistant-like. This is the paper's definition (arXiv:2601.10387 §2.1.3).

**`dist_default`** — plain Euclidean distance from the Assistant's mean:

```
dist_default(role) = ‖ mean(role) − mean(default) ‖
```

The intent was that `axis_proj` measures position *along* one direction while
`dist_default` catches being unlike the Assistant in *any* direction.

> **They turned out to be redundant.** Their correlation vectors across the six
> estimators agree at **r = 0.999** on both clouds — mirror images. They are one
> finding stated twice, not two independent confirmations. Reported anyway,
> labelled, so nobody counts them as replication.

Note both measure where a persona *sits*, while ID measures the *shape* of its
cloud. And "closer to the Assistant Axis" here means closer to the default
Assistant's mean point — not distance to the axis line itself.

`default` is excluded from every fit: its distance from itself is 0 and it
defines the axis.

### Partial correlation, and the confound it removes

**The problem.** A role whose cloud is simply more spread out scores higher ID
from nearly every estimator — measured here at up to **r = −0.78** between ID
and log within-role variance. Spread also shifts a role along the axis. So a raw
ID↔axis correlation can be entirely spread acting on both ends, with no direct
relationship at all.

**The fix** (`04_id_vs_axis.py::partial_corr`). Let `z = log(SS_total)` per role:

```
1.  regress ID on z         →  keep residual  ID⊥      (part of ID that z can't explain)
2.  regress axis_proj on z  →  keep residual  axis⊥
3.  partial r = corr(ID⊥, axis⊥)
```

Read it as: **among roles with the same cloud size, does ID still track axis
position?** Significance uses `t = r·√((n−k−2)/(1−r²))` with `k = 1` control.

Where raw and partial disagree, the raw one was the confound talking. That
happens: participation ratio is raw −0.32 but partial **+0.13** — sign flip. The
−0.32 originally reported by `01` was mostly scale, not axis position.

### Multiple testing

Six estimators are tested per predictor, so p-values get **Benjamini–Hochberg**
FDR correction within each predictor (`bh_fdr`). Spearman is reported alongside
Pearson because several ID distributions are skewed.

### r, not r²

Every correlation reported is Pearson **r ∈ [−1, +1]** (from
`scipy.stats.pearsonr`). Negative values in the tables confirm this — r² cannot
be negative. Response cloud, vs `axis_proj`:

| estimator | r | r² | partial r | partial r² |
|---|---|---|---|---|
| MLE | −0.906 | 0.821 | −0.805 | 0.648 |
| dim 95% | −0.900 | 0.810 | −0.773 | 0.598 |
| dim 90% | −0.897 | 0.805 | −0.765 | 0.585 |
| TwoNN | −0.765 | 0.586 | −0.567 | 0.321 |
| lPCA | +0.183 | 0.034 | +0.444 | 0.197 |
| participation ratio | −0.318 | 0.101 | +0.134 | 0.018 |

So MLE: axis position accounts for ~82% of the variance in per-role ID, ~65%
after removing cloud scale.

### Reading the result

Four of six estimators agree strongly and survive the control: **more
Assistant-like roles have lower-dimensional within-role manifolds.** `lPCA`
dissents (+0.44) — unexplained; it is a rank-threshold count rather than a
neighbour- or spread-based measure, but that is a description, not a reason.
Participation ratio is null once scale is removed.

The prompt cloud reverses the sign on all six (partial r **+0.08 to +0.64**,
significant for five — `lPCA` is the exception at q = 0.20). Its ID is 99.4%
grid (see §1), so that correlation is about grid saturation, not geometry —
contrast, not a competing result.

One loose end, recorded rather than resolved: `default` has the *highest*
`axis_proj` yet the *lowest* ID on the prompt cloud, against that cloud's own
positive trend. It is excluded from all fits, so it moves no number here, but
the trend does not extrapolate to it.
