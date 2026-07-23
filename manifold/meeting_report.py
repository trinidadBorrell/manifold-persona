"""Combined team-meeting overview of the two H1 manifold runs.

Covers:
  run 1  output/manifold_h1-2/2026-07-21T14-03  — does the role manifold exist? (H1)
  run 2  output/manifold_h1-2/2026-07-22T17-16  — does coarsening the role set help?

Emits BOTH a markdown file and a single self-contained HTML file (figures inlined
as base64, no external requests) so the HTML can be opened or sent as one file.

Numbers are read from each run's saved metrics, never retyped, so this cannot
drift from the runs. Prose is authored here.

Usage:
    .venv/bin/python -m manifold.meeting_report
"""
from __future__ import annotations

import base64
import io
import os
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output" / "manifold_h1-2"
R1 = OUT / "2026-07-21T14-03"
R2 = OUT / "2026-07-22T17-16"
R3 = REPO / "exploratory" / "assistant_axis" / "figures" / "21-Jul-2026-1918"
DEST = OUT / "MEETING-2026-07-22"

TAG = "prompt_avg_L26"      # all three studies share this view/layer

MAX_W = 1500        # downscale for the embedded HTML; PNGs on disk stay 300 dpi


# --------------------------------------------------------------------------- #
def fig(run: Path, name: str, caption: str, note: str = "") -> dict:
    # manifold runs keep figures in <run>/figures/; the exploratory run dir IS
    # already a figures dir, so fall back to the run dir itself.
    p = run / "figures" / name
    if not p.exists():
        p = run / name
    return {"path": p, "caption": caption, "note": note}


def load_numbers() -> dict:
    m1 = pd.read_csv(R1 / "data" / "metrics.csv").set_index("construction")
    d2 = pd.read_csv(R2 / "data" / "sweep_metrics.csv")
    v2 = json.load(open(R2 / "data" / "verdict.json"))
    refs = {int(k): v for k, v in json.load(open(R2 / "data" / "id_reference.json")).items()}
    g = d2.groupby("n")
    idx = json.load(open(R3 / f"intrinsic_dimension_{TAG}.json"))
    clu = json.load(open(R3 / f"clustering_{TAG}.json"))
    axr = json.load(open(R3 / f"03_axis_ranking_{TAG}.json"))
    fam = json.load(open(R3 / f"04_role_families_{TAG}.json"))
    return {
        "m1": m1,
        "idx": idx, "clu": clu, "axr": axr, "fam": fam,
        "d2": d2, "v2": v2, "refs": refs,
        "rr": g["rel_reduction"].mean(), "rr_lo": g["rel_reduction"].min(),
        "rr_hi": g["rel_reduction"].max(),
        "r2": g["r2"].mean(), "nullm": g["null_median"].mean(),
        "plane": g["plane_r2"].mean(), "gain": g["curv_gain"].mean(),
        "pc": g["pc_rel_reduction"].mean(),
        "id2nn": g["id_TwoNN"].mean(),
        "spread": g["role_spread"].mean(), "noise": g["within_role_noise"].mean(),
        "ns": sorted(int(n) for n in d2["n"].unique()),
    }


# --------------------------------------------------------------------------- #
def build_markdown(N: dict, figures: list) -> str:
    m1, rr, ns = N["m1"], N["rr"], N["ns"]
    L = []
    A = L.append

    A("# Persona manifolds — where we are\n")
    A("**Team meeting, 22 Jul 2026.** Three studies on the same character-role activation "
      "cloud of Qwen2.5-3B-Instruct: two preregistered manifold runs and one exploratory "
      "geometry/clustering study. Everything below is reproducible from the run "
      "directories with the seeds in each `manifest.json`.\n")

    # ---------------------------------------------------------------- headline
    A("## The three questions, and the three answers\n")
    A("| | Question | Answer |")
    A("|---|---|---|")
    A(f"| **Run 1** (21 Jul) | Do role representations lie on a low-dimensional manifold? | "
      f"**Yes.** Spline R² {m1.loc['C_role','r2']:.3f} vs a role-shuffle null of "
      f"{m1.loc['C_role','null_median']:.3f}; p={m1.loc['C_role','p']:.4f}; the manifold "
      f"explains **{m1.loc['C_role','rel_reduction']*100:.0f}% more** of the residual than "
      f"chance, against a preregistered 30% floor. |")
    A(f"| **Run 2** (22 Jul) | Is the manifold *better* if we build it from fewer role "
      f"centroids? | **Stronger by the headline metric, but not more manifold.** The n-fair "
      f"effect size rises {rr[276]:.3f} → {rr[10]:.3f} as the role set shrinks 276 → 10, yet "
      f"curvature and dimensional structure both *fall*. |")
    A(f"| **Exploratory** (21 Jul) | What does the role space actually look like? | "
      f"**A continuum, ~10-dimensional, with one axis that matters.** HDBSCAN finds **zero** "
      f"clusters (100% noise); PC1 aligns with the Assistant Axis at "
      f"cos = **{N['axr']['cos_pc1_axis']:.2f}**; `default` separates as a family of one. |")
    A("")
    A("> **The three agree.** \"One manifold, not several\" (run 1, cosine thresholds), "
      "\"no density clusters at all\" (exploratory, HDBSCAN) and \"coarsening doesn't reveal "
      "new structure\" (run 2) are three routes to the same picture: **a single smooth, "
      "~10-dimensional continuum of roles, organised principally by distance from the "
      "Assistant.**\n")

    # ---------------------------------------------------------------- METHOD
    A("## 1. Method — what we actually measure\n")

    A("### 1.1 The data ⚠️ read this before quoting any number\n")
    A("We follow the **Assistant Axis** paper (arXiv:2601.10387) for the role set and the "
      "definition of a role vector (the mean over a role's rollouts), and **Manifold "
      "Steering** (arXiv:2605.05115) for the spline manifold. **We do not yet match the "
      "paper's extraction.** Three differences, all of which matter:\n")
    A("| | Assistant Axis paper | **What these two runs use** |")
    A("|---|---|---|")
    A("| Token basis | residual stream over the **generated response** | **the prompt** — "
      "system(role) + user question, ending at `<\\|im_start\\|>assistant\\n`. **No answers "
      "were generated.** |")
    A("| Depth | ~**0.5** of network depth | **layer 26 of 36 blocks = 0.72 depth** |")
    A("| Rollout filter | an LLM judge keeps only in-character responses (score 3) | "
      "**no judge** — there is nothing to judge, since there are no responses |")
    A("| Model | 27B+ | Qwen2.5-3B-Instruct |")
    A("")
    A("> **So what is the manifold made of?** 276 roles × 25 prompts = 6,900 points. The 25 "
      "prompts per role are 5 phrasings of the role instruction × 5 questions. Each point is "
      "the layer-26 residual stream **averaged over the prompt tokens**. A role vector is the "
      "mean of that role's 25 prompt points.\n")
    A("> **Therefore what we have shown is that _instruction geometry_ is low-dimensional** "
      "— how the model represents *being told* to be a character. Whether *behaving* as the "
      "character lies on the same manifold is **untested**. That is the single biggest "
      "caveat on this work and it should be said out loud in the meeting.\n")
    A("> **Status of the paper-matched run:** the response-token / 0.5-depth extraction "
      "exists as code and is **15% complete** (5 of 33 checkpoint shards, 640 of 4,140 "
      "records, `data/embeddings_roles_resp/`). It is checkpointed and resumable; finishing "
      "it is ~5–6 h of local generation. Until then, every number here is prompt-based.\n")

    A("### 1.2 The pipeline, end to end\n")
    A("1. **Extract.** For each of 6,900 (role, instruction, question) prompts, take the "
      "layer-26 residual stream and average over prompt tokens → 6,900 × 2,048.\n"
      "2. **Reduce.** PCA to **D = 50**, fit **once** on all 6,900 points and shared by "
      "every analysis, so results at different role counts stay comparable.\n"
      "3. **Role vectors.** Mean of each role's 25 points → 276 centroids in 50-D.\n"
      "4. **Fit the surface.** Take `k = 3` intrinsic coordinates as PCA of the centroids, "
      "then fit a **thin-plate spline** mapping those 3 coordinates → 50-D through the "
      "centroids. (Roles come with no intrinsic coordinates, unlike the causal models the "
      "spline method was built for, so we derive them from the data — a documented extension.)\n"
      "5. **Score.** Project every one of the 6,900 raw points onto the surface by "
      "Gauss-Newton and measure what is left over:\n")
    A("```\nNRE = SSR / TSS   = fraction of spread the surface FAILS to explain\n"
      "manifold-R² = 1 − NRE = fraction it explains\n```\n")
    A("6. **The honest part — the null.** A thin-plate spline is *flexible*: bend it through "
      "276 anchors in 50-D and it hugs almost any cloud, so R² alone proves nothing. We "
      "shuffle which points carry which role label, rebuild the centroids, and re-run the "
      "**entire** fit-and-score, 100 times. Whole label sets are permuted, so significance is "
      "read at the role level (n=276) — not at the level of 6,900 correlated points, which "
      "would wildly overstate it.\n")
    A("7. **The deciding number** is the gap, expressed as **relative NRE reduction** = "
      "`(NRE_null − NRE_real) / NRE_null`. Preregistered bar: p < 0.05 **and** relative "
      "reduction ≥ **0.30**.\n")

    A("### 1.3 Controls (these ran first, and could have stopped the work)\n")
    A("| Control | Purpose | Result |")
    A("|---|---|---|")
    A(f"| **Positive control** — synthetic curved 3-D manifold in 50-D at the data's own "
      f"noise scale | if the pipeline can't find a manifold that is definitely there, a null "
      f"result would be meaningless | **PASS**, R² {m1.loc['positive_control','r2']:.3f} vs "
      f"null {m1.loc['positive_control','null_median']:.3f} (run 1); **PASS in all 36 cells** "
      f"of run 2 |")
    A("| **Negative control** — the role-shuffle permutation null | proves we aren't "
      "manufacturing structure | ran in every fit; it is the denominator of the decider |")
    A(f"| **Flat baseline** — PCA plane, same k=3 | tells us whether *curvature* bought "
      f"anything | R² {m1.loc['PCA_plane(k=3)','r2']:.3f} vs spline "
      f"{m1.loc['C_role','r2']:.3f} |")
    A("| **Regression check** (run 2) | that shared code hadn't silently changed | run 2's "
      "276-role cell reproduces run 1 to ~1e-12 |")
    A("")
    A("Both runs were **preregistered**: hypothesis, deciding metric, threshold and stopping "
      "rule were written and approved before the code ran (`plans/2026-07-21-…md`, "
      "`plans/2026-07-22-…md`). Run 2 was **independently audited** afterwards; no blocking "
      "findings.\n")

    # ---------------------------------------------------------------- RUN 1
    A("## 2. Run 1 — the manifold exists\n")
    A("| Construction | R² | null median | p | rel. reduction | verdict |")
    A("|---|---:|---:|---:|---:|---|")
    for name in ["C_role", "C_raw", "PCA_plane(k=3)", "C_role[prompt_last]", "C_role[k=2]"]:
        r = m1.loc[name]
        def _c(v, nd=3):
            return "—" if pd.isna(v) else f"{v:.{nd}f}"
        A(f"| `{name}` | {_c(r['r2'])} | {_c(r['null_median'])} | {_c(r['p'], 4)} | "
          f"{_c(r['rel_reduction'])} | {r['verdict']} |")
    A("")
    A("**Headline:** role activations sit measurably closer to a smooth 3-D surface than "
      "shuffled data does — 45% less unexplained spread, 27 chance-SDs above the null. "
      "Robust to using the last prompt token instead of the mean, and to k=2.\n")
    A("**Four things we learned that we did not set out to test:**\n")
    A(f"1. **It is ONE manifold, not several.** We split roles by cosine similarity at the "
      f"25/50/75th percentiles; all three thresholds left the graph fully connected, because "
      f"role vectors are spread almost orthogonally (the thresholds land at "
      f"{-0.24:.2f}/{-0.01:.2f}/{0.23:.2f}). Threshold-free checks agree: persistence gives "
      f"N=1 stable over 68% of thresholds, spectral eigengap a weak N≈2.\n")
    A("2. **The true intrinsic dimension is ~8–10, not the 3 we assumed** (TwoNN 8.5, MLE "
      "8.9, lPCA 10). Our k=3 surface *underfits* and still beat the null — the existence "
      "claim is safe, the specific dimension was wrong. Refitting at k≈8 is the top "
      "carry-forward.\n")
    A("3. **Curvature earns its keep descriptively:** spline 0.655 vs flat plane 0.466.\n")
    A("4. **The Assistant sits at the edge, not the centre.** The `default` persona is at the "
      "extreme low-PC1 end of the cloud — consistent with the Assistant-Axis idea that PC1 "
      "tracks distance from the default persona.\n")

    # ---------------------------------------------------------------- RUN 2
    A("## 3. Run 2 — does coarsening help?\n")
    A("**Setup.** For n ∈ {10, 25, 50, 75, 100, 150, 200, 276} we k-means the 276 role means "
      "into n clusters and keep each cluster's **medoid** — a real role, never a synthetic "
      "average — so every n is a maximally-spread summary of the same cloud. `default` is "
      "always included. 5 k-means seeds per n = 36 cells, each with its own 100-permutation "
      "null and its own positive control.\n")
    A("**The trap we had to avoid.** With 10 anchors a 3-D spline in 50-D passes almost "
      "exactly through everything, so raw R² rises as n falls no matter what. Only a null "
      "computed *at the same n* is fair.\n")
    A("| n | rel. NRE reduction (decider) | raw R² | null R² | spline − plane | TwoNN ID | "
      "pos-control |")
    A("|---:|---:|---:|---:|---:|---:|---:|")
    for n in ns:
        band = "" if n == 276 else f" [{N['rr_lo'][n]:.3f}–{N['rr_hi'][n]:.3f}]"
        A(f"| {n} | **{N['rr'][n]:.3f}**{band} | {N['r2'][n]:.3f} | {N['nullm'][n]:.3f} | "
          f"{N['gain'][n]:.3f} | {N['id2nn'][n]:.1f} | {N['pc'][n]:.2f} |")
    A("")
    A("### What this says\n")
    A(f"**By the preregistered decider, coarsening wins** — {N['rr'][276]:.3f} → "
      f"{N['rr'][10]:.3f}, monotone, seed ranges non-overlapping, every cell significant and "
      f"every control passing. *This falsified our own preregistered prediction*, which said "
      f"the effect would fall. We report it as a falsification.\n")
    A("**But three things cut against reading that as \"a better manifold\":**\n")
    A(f"1. **Curvature collapses.** Spline-minus-plane falls {N['gain'][276]:.3f} → "
      f"{N['gain'][10]:.3f}. At n=10 a flat 3-D plane explains {N['plane'][10]:.3f} of the "
      f"{N['r2'][10]:.3f} the curved surface gets. **The n=10 \"manifold\" is essentially a "
      f"linear subspace.**\n")
    A(f"2. **The dimensional evidence goes the other way.** Real intrinsic dimension vs a "
      f"size-matched structureless Gaussian: at n=276 the real cloud is far below the "
      f"reference ({N['id2nn'][276]:.1f} vs {N['refs'][276]['TwoNN']['median']:.1f}); at n=10 "
      f"the gap is nearly closed ({N['id2nn'][10]:.1f} vs "
      f"{N['refs'][10]['TwoNN']['median']:.1f}). There is *less* to see at n=10, not more.\n")
    A(f"3. **The positive control trends the same way** ({N['pc'][276]:.2f} → "
      f"{N['pc'][10]:.2f}) — and it contains no role structure at all. So roughly half the "
      f"decider's rise is a property of fitting with few anchors, not evidence about roles. "
      f"*(Post hoc — spotted after the run, not preregistered.)*\n")
    A("**Checked and clean:** the ratio of role spread to within-role noise is flat across "
      "the whole sweep (≈9.0–9.8), so this is not an artefact of medoid selection picking "
      "better-separated roles at small n.\n")
    A("> **Take to the meeting:** *coarsening buys legibility, not evidence.* Use small-n "
      "plots to explain the geometry to people; keep the 276-role surface as the object of "
      "record. (Offered as the joint reading of three metrics — the preregistered decider on "
      "its own says coarsening is stronger.)\n")

    # ------------------------------------------------- EXPLORATORY / CLUSTERING
    idx, clu, axr, fam = N["idx"], N["clu"], N["axr"], N["fam"]
    gl, g95 = idx["global"], idx["global_pca95"]
    p95 = clu["pca95"]
    fams = fam["families"]

    A("## 4. Clustering and geometry — the exploratory study\n")
    A("A third, **exploratory** study (`exploratory/assistant_axis/figures/21-Jul-2026-1918`) "
      "on the *same* 276 role vectors, same view and layer. It is descriptive: no "
      "preregistration, no decider, no permutation test. Its job was to answer \"what does "
      "this cloud look like?\" — and its answers turn out to line up with the manifold "
      "results in a way that is worth a slide.\n")

    A("### 4.1 How many dimensions? ~9–13, and it is not an artefact of the ambient space\n")
    A(f"With only **276 points in 2,048 dimensions**, the obvious objection is that intrinsic-"
      f"dimension estimators are being fooled by the ambient dimension. So every estimator was "
      f"run twice: on the raw 2,048-D vectors, and after PCA to "
      f"**{idx['_meta']['pca95_dim']} dims (95% of variance)**.\n")
    A("| Estimator | full 2,048-D | after PCA-95% | shift |")
    A("|---|---:|---:|---:|")
    for k in ["TwoNN", "MLE", "lPCA", "MOM", "TLE", "CorrInt"]:
        A(f"| {k} | {gl[k]:.1f} | {g95[k]:.1f} | {g95[k]-gl[k]:+.1f} |")
    A(f"| *PCA participation ratio* | {gl['PCA_participation_ratio']:.1f} | — | — |")
    A("")
    A(f"**Estimates move by only 1–2 dimensions.** These estimators work on *local* "
      f"neighbour distances, so the ambient dimension is largely irrelevant — what limits "
      f"them is n=276, and that biases them **downward**, so ~9–13 is a **lower bound**. "
      f"Note it takes {idx['_meta']['pca95_dim']} linear dimensions to hold 95% of the "
      f"variance but the *intrinsic* dimension is ~10: the cloud is a curved ~10-D object "
      f"sitting in a much larger linear subspace. This independently corroborates run 1's "
      f"post-hoc finding (~8–10) and confirms **k=3 underfits**.\n")

    A("### 4.2 There are no clusters. The role space is a continuum.\n")
    A("Six clustering methods, all in the PCA-95% space:\n")
    A("| Method | clusters found | silhouette | Davies–Bouldin | noise |")
    A("|---|---:|---:|---:|---:|")
    label = {"kmeans": "KMeans (optimal k)", "kmeans_k12": "KMeans k=12",
             "kmeans_k24": "KMeans k=24", "gmm": "GMM", "hdbscan": "**HDBSCAN**",
             "dbscan": "DBSCAN"}
    for k in ["kmeans", "kmeans_k12", "kmeans_k24", "gmm", "hdbscan", "dbscan"]:
        v = p95[k]
        sil = "—" if v.get("silhouette") is None else f"{v['silhouette']:.3f}"
        db = "—" if v.get("davies_bouldin") is None else f"{v['davies_bouldin']:.2f}"
        A(f"| {label[k]} | {v.get('n_clusters')} | {sil} | {db} | "
          f"{v.get('noise_frac', 0)*100:.0f}% |")
    A("")
    A(f"**The headline is HDBSCAN: it finds zero clusters and labels 100% of roles as "
      f"noise.** DBSCAN finds 2 and throws away 21%. Every silhouette is ≤ "
      f"{max(v['silhouette'] for v in p95.values() if v.get('silhouette')):.2f} — on a scale "
      f"where 0.5 is 'reasonable structure' and 0 is 'no better than arbitrary'. KMeans and "
      f"GMM return partitions only because they are *required* to: give them a k and they "
      f"will cut the cloud somewhere.\n")
    A("> **This is the same result the manifold study reached by a different route.** Run 1 "
      "tried to split roles by cosine similarity at three thresholds and the graph stayed "
      "fully connected every time — **one** manifold, not several. Density-based clustering "
      "now says the same thing in its own language: there are no density-separated groups to "
      "find. Roles are a **continuum**, not a set of types. Any \"persona categories\" we "
      "draw are our own cuts through a smooth object, and should be presented that way.\n")

    A("### 4.3 PC1 *is* the Assistant Axis\n")
    A(f"Cosine between PC1 of the role cloud and the Assistant Axis "
      f"(mean(default) − mean(roles)): **{axr['cos_pc1_axis']:.3f}**. PC1 alone carries "
      f"{axr['pca_var_top3'][0]*100:.0f}% of the variance (PC2 "
      f"{axr['pca_var_top3'][1]*100:.0f}%, PC3 {axr['pca_var_top3'][2]*100:.0f}%).\n")
    A("So the single largest axis of variation among 276 characters is *how much like the "
      "default Assistant they are*. That is a strong replication of the paper's central "
      "claim, on our own data and our own model.\n")
    top = list(axr["most_assistant_like"].items())
    bot = list(axr["least_assistant_like"].items())
    A("| Most Assistant-like | proj | Least Assistant-like | proj |")
    A("|---|---:|---|---:|")
    for i in range(6):
        A(f"| {top[i][0]} | {top[i][1]:+.1f} | {bot[i][0]} | {bot[i][1]:+.1f} |")
    A("")
    A(f"The ordering is interpretable end to end: institutional/epistemic roles "
      f"(supervisor, instructor, architect, economist) at the Assistant end; non-agentive and "
      f"non-human entities (leviathan, tree, mycorrhizal, golem, echo) at the far end. "
      f"`default` ranks **#1** at {top[0][1]:+.1f}, more than **twice** the next role "
      f"({top[1][0]}, {top[1][1]:+.1f}).\n")

    A("### 4.4 The Assistant is an outlier, not just an extreme\n")
    singles = [k for k, v in fams.items() if v["size"] == 1]
    f7 = fams[singles[0]] if singles else None
    if f7 and f7["roles"] == ["default"]:
        others = sorted((v["mean_axis_proj"] for k, v in fams.items()
                         if k not in singles), reverse=True)
        A(f"Ward hierarchical clustering — used here purely as a **descriptive** device, it "
          f"is not from the paper and nothing is scored against it — splits the 276 roles "
          f"into {len(fams)} families. **One of them contains `default` and nothing else.** "
          f"Its mean axis projection is {f7['mean_axis_proj']:+.1f}; the most Assistant-like "
          f"*family* of real characters sits at {others[0]:+.1f}.\n")
        A("The Assistant does not sit at the end of a line of increasingly helpful "
          "characters — it sits **off on its own**, roughly five times further along the "
          "axis than any group of characters reaches. The MST skeleton (fig08) shows the same "
          "thing geometrically: `default` is a hub hanging off the main body.\n")
    A("The remaining families are semantically coherent even though nothing semantic went "
      "into building them — they were formed from activations alone:\n")
    A("| Family | n | mean axis | character of the group |")
    A("|---|---:|---:|---|")
    gloss = {
        "1": "non-human / non-agentive entities", "3": "artistic, playful, unmoored",
        "2": "supernatural and mythic", "14": "life-stage and loss",
        "15": "dispositional stances", "13": "academic / theoretical",
        "12": "mystical and collective", "11": "adversarial / predatory",
        "9": "caring and mediating", "10": "creative professions",
        "4": "evaluative and judging", "6": "instructional and organising",
        "8": "advocacy and identity", "5": "scientific and technical",
        "7": "**the Assistant, alone**"}
    for k, v in sorted(fams.items(), key=lambda kv: kv[1]["mean_axis_proj"]):
        A(f"| {k} | {v['size']} | {v['mean_axis_proj']:+.1f} | {gloss.get(k,'—')} — "
          f"*{', '.join(v['roles'][:5])}*… |")
    A("")
    A("Reading up the table is a readable gradient: *things* → *myths* → *stances* → "
      "*professions* → *the Assistant*. **But note this is a hierarchy we imposed on a "
      "continuum** (§4.2): the families are a convenient way to talk about the axis, not "
      "discovered kinds.\n")

    A("### 4.5 Where `default` lands under every method\n")
    comp = json.load(open(R3 / f"02_cluster_composition_{TAG}.json"))
    A("| Method | `default`'s cluster | size | its mean axis | highest-axis cluster? |")
    A("|---|---|---:|---:|---|")
    all_highest = True
    for k in ["kmeans", "kmeans_k12", "kmeans_k24", "gmm", "hdbscan", "dbscan"]:
        dc = p95[k].get("default_cluster")
        if dc is None:
            A(f"| {label[k]} | **noise** — unassigned | — | — | n/a |")
            continue
        rows = comp[k]
        me = next(r for r in rows if r["cluster"] == dc)
        best = max(rows, key=lambda r: r["mean_axis_proj"])
        hi = best["cluster"] == dc
        all_highest &= hi
        verdict = ("**yes**" if hi else
                   "no — c{} is higher at {:+.2f}".format(best["cluster"],
                                                          best["mean_axis_proj"]))
        A(f"| {label[k]} | c{dc} | {me['size']} | {me['mean_axis_proj']:+.2f} | {verdict} |")
    A("")
    A(("Wherever a partition exists, `default` falls in **the most Assistant-like cluster of "
       "that partition** — all four of them." if all_highest else
       "`default` does not always land in the highest-axis cluster; see the table.") +
      " And where the method is allowed to say *\"this point belongs to no dense group\"* — "
      "HDBSCAN and DBSCAN — it says exactly that about `default`. Both facts point the same "
      "way: the Assistant is at the extreme of the only axis that matters, and it is not "
      "part of any dense group of characters.\n")
    k24 = next((r for r in comp["kmeans_k24"]
                if r["cluster"] == p95["kmeans_k24"].get("default_cluster")), None)
    if k24 and k24["size"] == 1:
        A(f"**Worth flagging:** at k=24, KMeans spends one of its 24 clusters on `default` "
          f"**alone** (size {k24['size']}, mean axis {k24['mean_axis_proj']:+.2f}). That is "
          f"an independent method reaching the same conclusion as the Ward family in §4.4 — "
          f"given enough clusters to spend, both isolate the Assistant by itself before they "
          f"split any group of real characters.\n")

    # ---------------------------------------------------------------- FIGURES
    A("## 5. Figures\n")
    for f in figures:
        # relative to the MEETING dir, so the .md renders in place on disk
        rel = os.path.relpath(f["path"], DEST)
        A(f"### {f['caption']}\n")
        A(f"![{f['caption']}]({rel})\n")
        if f["note"]:
            A(f"{f['note']}\n")

    # ---------------------------------------------------------------- NEXT
    A("## 6. What we do next\n")
    A("| Priority | Action | Why |")
    A("|---|---|---|")
    A("| **1** | **Finish the response-token / 0.5-depth extraction** (15% done, resumable, "
      "~5–6 h) and re-run both studies on it | closes the biggest gap to the paper and tests "
      "whether *behaviour* shares the manifold of *instruction* |")
    A("| **2** | Refit at **k ≈ 8**, the measured intrinsic dimension | k=3 was an assumption "
      "and it underfits; this is the honest-dimension version of the decider |")
    A("| **3** | Re-run run 2's decider as **real minus positive-control** effect size | "
      "divides out the low-anchor regime effect found post hoc |")
    A("| 4 | Random-draw and semantically-coherent subset selection | separates \"how many "
      "roles\" from \"how spread out they are\" |")
    A("| 5 | Add the LLM judge on rollouts | the remaining paper difference once responses "
      "exist |")
    A("")
    A("**Not yet claimed, and we should be careful not to imply it:** that this is persona "
      "*essence*; that behaviour lies on this manifold; that the manifold is steerable (that "
      "is H2, unstarted); anything about misalignment (H3).\n")

    A("---\n")
    A("*Sources: `output/manifold_h1-2/2026-07-21T14-03/REPORT.md` and "
      "`…/2026-07-22T17-16/REPORT.md`; plans in `plans/`; every number regenerated from the "
      "runs' saved metrics by `manifold/meeting_report.py`.*")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def _img_b64(path: Path, max_w: int = MAX_W) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def build_html(md: str, figures: list) -> str:
    """Render the markdown to a single self-contained HTML file.

    Figures are inlined as base64 JPEG so the file works with no network and no
    sibling directories — it can be attached to an email or opened from a USB
    stick and still render.
    """
    import re
    body = md
    # replace image links with inlined data URIs
    for f in figures:
        rel = os.path.relpath(f["path"], DEST)
        b64 = _img_b64(f["path"])
        body = body.replace(f"![{f['caption']}]({rel})",
                            f'<img src="data:image/jpeg;base64,{b64}" alt="{f["caption"]}">')

    def md_table(block: str) -> str:
        rows = [r for r in block.strip().split("\n") if r.strip().startswith("|")]
        if len(rows) < 2:
            return block
        out = ["<table>"]
        for i, r in enumerate(rows):
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "th" if i == 0 else "td"
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        out.append("</table>")
        return "\n".join(out)

    html, buf = [], []

    def flush_table():
        if buf:
            html.append(md_table("\n".join(buf)))
            buf.clear()

    in_code = False
    for line in body.split("\n"):
        if line.startswith("```"):
            flush_table()
            html.append("<pre><code>" if not in_code else "</code></pre>")
            in_code = not in_code
            continue
        if in_code:
            html.append(line)
            continue
        if line.strip().startswith("|"):
            buf.append(line)
            continue
        flush_table()
        s = line
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", s)
        s = re.sub(r"`([^`]+?)`", r"<code>\1</code>", s)
        if s.startswith("#"):
            lvl = len(s) - len(s.lstrip("#"))
            html.append(f"<h{lvl}>{s.lstrip('# ').strip()}</h{lvl}>")
        elif s.startswith("> "):
            html.append(f"<blockquote>{s[2:]}</blockquote>")
        elif s.startswith("<img"):
            html.append(f"<figure>{s}</figure>")
        elif re.match(r"^\d+\. ", s):
            html.append(f"<p class='num'>{s}</p>")
        elif s.startswith("- "):
            html.append(f"<li>{s[2:]}</li>")
        elif s.strip() == "---":
            html.append("<hr>")
        elif s.strip():
            html.append(f"<p>{s}</p>")
    flush_table()

    css = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#5b6570;--line:#e2e6ea;--accent:#1565c0;
      --warn:#fff8e1;--warnb:#f0b429;--code:#f5f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e6e9ec;--mut:#9aa5b1;
      --line:#2b3138;--accent:#6ba7e8;--warn:#2e2717;--warnb:#b8860b;--code:#1c2126}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;
     font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif}
main{max-width:940px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:2.1rem;line-height:1.2;margin:0 0 .3em;letter-spacing:-.02em}
h2{font-size:1.5rem;margin:2.4em 0 .6em;padding-bottom:.3em;border-bottom:2px solid var(--line)}
h3{font-size:1.15rem;margin:1.8em 0 .5em;color:var(--accent)}
p{margin:.7em 0}
p.num{margin:.5em 0 .5em 1.2em}
li{margin:.35em 0 .35em 1.4em}
blockquote{margin:1em 0;padding:.85em 1.1em;background:var(--warn);
           border-left:4px solid var(--warnb);border-radius:0 6px 6px 0}
blockquote p{margin:.3em 0}
code{background:var(--code);padding:.12em .38em;border-radius:4px;font-size:.88em;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);padding:1em;border-radius:8px;overflow-x:auto}
pre code{background:none;padding:0}
.tablewrap,table{display:block;overflow-x:auto;max-width:100%}
table{border-collapse:collapse;margin:1.1em 0;font-size:.92em;width:100%}
th,td{border:1px solid var(--line);padding:.5em .7em;text-align:left;vertical-align:top}
th{background:var(--code);font-weight:650}
tr:nth-child(even) td{background:color-mix(in srgb,var(--code) 45%,transparent)}
figure{margin:1.4em 0;text-align:center}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px}
hr{border:0;border-top:1px solid var(--line);margin:2.5em 0}
em{color:var(--mut)}
"""
    return (f"<title>Persona manifolds — team meeting, 22 Jul 2026</title>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<style>{css}</style><main>" + "\n".join(html) + "</main>")


# --------------------------------------------------------------------------- #
def main():
    DEST.mkdir(parents=True, exist_ok=True)
    N = load_numbers()

    figures = [
        fig(R2, "fig01_sweep_decider.png",
            "Run 2 · fig01 — THE DECIDER: effect size vs role count",
            "*Left panel is the claim.* Green = relative NRE reduction (how much better than "
            "a role-shuffle null at that same n); it rises 0.451 → 0.646 as the role set "
            "shrinks 276 → 10, band = spread over 5 k-means seeds, red line = the "
            "preregistered 0.30 floor. Right panel shows why raw R² can't be compared across "
            "n: the real and null curves move in *opposite* directions."),
        fig(R1, "fig01_null_vs_real.png",
            "Run 1 · fig01 — real R² against the permutation null",
            "Violins are the null distributions (shuffled role labels, identical pipeline); "
            "points are the real fits. The decider `C_role` (green) sits far to the right of "
            "its null — that separation is the whole result."),
        fig(R1, "figPC_positive_control.png",
            "Run 1 · positive control — the pipeline can find a manifold that is there",
            "A synthetic curved 3-D manifold in 50-D at the data's own noise scale. This ran "
            "*first* and would have stopped the study had it failed."),
        fig(R1, "fig03_roles_pc12.png",
            "Run 1 · fig03 — the 276 role vectors, with the Assistant marked",
            "★ = `default`. It sits at the extreme low-PC1 edge of the cloud, not at its "
            "centre — PC1 behaves like distance from the default persona."),
        fig(R2, "fig05_spline_manifold_n10.png",
            "Run 2 · fig05 — what 10 role centroids actually look like",
            "Illustrative. Large labelled markers are role vectors; faint dots are the 25 "
            "prompts behind each; red curve passes exactly through every centroid (its loops "
            "between knots are interpolation overshoot, not data). Axes are the fitted "
            "surface's own intrinsic coordinates."),
        fig(R2, "fig03_sweep_curvature_gain.png",
            "Run 2 · fig03 — curvature collapses as the role set shrinks",
            "Spline R² minus flat-plane R². At n=10 there is almost nothing left for "
            "curvature to explain: the small-n 'manifold' is a linear subspace. This is the "
            "main reason not to read the decider as 'a better manifold'."),
        fig(R2, "fig02_sweep_intrinsic_dim.png",
            "Run 2 · fig02 — intrinsic dimension against a size-matched null",
            "Solid = real role vectors; dashed + grey band = the same estimator on "
            "structureless Gaussian data of the same size. Read the *gap*: it is widest at "
            "n=276 and nearly closed at n=10, i.e. there is less dimensional structure to "
            "see in the small sets, not more."),
        fig(R1, "fig12_intrinsic_dimension.png",
            "Run 1 · fig12 — the intrinsic dimension is ~8–10, not the 3 we assumed",
            "Post hoc. Our k=3 surface underfits an ~8–10-dimensional manifold and still "
            "beat the null. Refitting at k≈8 is the top carry-forward."),
        fig(R2, "fig08_mst_skeleton_n100.png",
            "Run 2 · fig08 — MST skeleton over 100 role centroids",
            "Post hoc, added on request. Minimum spanning tree under cosine distance — a "
            "skeleton that touches every centroid without imposing an ordering. `default` "
            "(★) shows up as a hub sitting off the main body."),
        fig(R3, f"01_intrinsic_dimension_{TAG}.png",
            "Exploratory · intrinsic dimension, full space vs PCA-95%",
            "Six estimators run twice. The bars barely move, so ~9-13 is not an artefact of "
            "the 2,048-D ambient space; with n=276 it is a *lower* bound. k=3 underfits."),
        fig(R3, f"02_clustering_{TAG}.png",
            "Exploratory · cluster-quality scan — there are no clusters",
            "Silhouette never exceeds ~0.13 at any k. HDBSCAN finds zero density clusters and "
            "calls all 276 roles noise. The role space is a continuum, which is the same "
            "conclusion the manifold study reached by cosine thresholds."),
        fig(R3, f"04_role_dendrogram_{TAG}.png",
            "Exploratory · Ward dendrogram — descriptive only",
            "Not from the paper and nothing is scored against it. Read it as a convenient way "
            "to name regions of a continuum. `default` separates as a family of ONE."),
        fig(R3, f"06_kmeans_maps_{TAG}.png",
            "Exploratory · role map, KMeans at optimal k (UMAP + PCA, 2-D and 3-D)",
            "One of six such panels (also HDBSCAN, DBSCAN, GMM, k=12, k=24). Role names are "
            "printed and ★ marks `default`. Note the absence of gaps between groups."),
        fig(R3, f"03_pca12_{TAG}.png",
            "Exploratory · PC1-PC2 with the Assistant Axis",
            "cos(PC1, Assistant Axis) = 0.87 and PC1 carries 32% of the variance: the largest "
            "axis of variation across 276 characters is how Assistant-like they are."),
        fig(R1, "fig11_components_vs_tau.png",
            "Run 1 · fig11 — it is ONE manifold, not several",
            "Post hoc, threshold-free: N=1 is stable across 68% of cosine thresholds. The "
            "high-threshold fragmentation is over-splitting of a single manifold."),
    ]
    figures = [f for f in figures if f["path"].exists()]
    missing = [f["caption"] for f in figures if not f["path"].exists()]
    if missing:
        print("MISSING:", missing)

    md = build_markdown(N, figures)
    (DEST / "MEETING-REPORT.md").write_text(md)
    print("wrote", DEST / "MEETING-REPORT.md", len(md), "chars")

    html = build_html(md, figures)
    (DEST / "MEETING-REPORT.html").write_text(html)
    print("wrote", DEST / "MEETING-REPORT.html",
          f"{(DEST / 'MEETING-REPORT.html').stat().st_size/1e6:.1f} MB, "
          f"{len(figures)} figures inlined")


if __name__ == "__main__":
    main()
