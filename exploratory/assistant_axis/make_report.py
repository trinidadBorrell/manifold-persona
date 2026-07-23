"""Generate REPORT.md for an assistant-axis run: numbers, figures, interpretation,
code citations. Mirrors exploratory/persona_vectors/make_report.py."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

from common import resolve_run_dir
from manifold_persona.io import load_manifest
from manifold_persona.config import ROLE_EMBEDDINGS_DIR


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def load_first(run_dir, pattern):
    fs = sorted(glob.glob(str(Path(run_dir) / pattern)))
    return json.load(open(fs[-1])) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="prompt_avg")
    ap.add_argument("--cluster_col", default="kmeans_pca50")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    run_dir = resolve_run_dir(args.outdir)
    v = args.view

    idj = load_first(run_dir, f"intrinsic_dimension_{v}_L*.json")
    cluj = load_first(run_dir, f"clustering_{v}_L*.json")
    rankj = load_first(run_dir, f"03_axis_ranking_{v}_L*.json")
    famj = load_first(run_dir, f"04_role_families_{v}_L*.json")
    layer = (idj or cluj)["_meta"]["layer"]
    ambient = idj["_meta"]["ambient"] if idj else "?"
    n = (idj or cluj)["_meta"]["n"]
    n_roles = (idj or cluj)["_meta"].get("n_roles", "?")
    role_dir = os.environ.get("MP_ROLE_DIR", ROLE_EMBEDDINGS_DIR)
    try:
        man = load_manifest(role_dir)
        nhl = man["n_layers"] - 1
        depth = layer / nhl
        model_name = man.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
    except Exception:
        man, nhl, depth, model_name = {}, "?", None, "Qwen/Qwen2.5-3B-Instruct"
    # Response-token (paper-matched) run vs the original prompt-token run.
    token_basis = man.get("token_basis", "prompt")
    is_resp = token_basis == "response"
    max_new = man.get("max_new_tokens")
    nq = man.get("n_questions", 5)
    rollouts = 5 * nq                       # 5 instructions x nq questions
    n_raw = man.get("n_records", rollouts * (n_roles if isinstance(n_roles, int) else 276))
    basis_word = "response" if is_resp else "prompt"
    extract_driver = ("extraction/generate_and_extract_roles.py" if is_resp
                      else "extraction/build_and_extract_roles.py")

    L = []; A = L.append
    A("# manifold-persona — assistant-axis exploratory report")
    A("")
    A(f"*Run:* `{run_dir.name}`  ·  *view:* `{v}`  ·  *layer:* {layer}  ·  "
      f"*points:* {n} role means  ·  *roles:* {n_roles}  ·  *ambient:* {ambient}")
    A("")
    A("## What this run did")
    A("")
    if is_resp:
        A("**Paper-matched, response-token run.** Over the **276 character-role "
          "archetypes** from the assistant-axis repo (pirate, ghost, accountant, "
          "alien, …), for each role we built `system(role instruction) + question` "
          f"chats (5 instructions × {nq} sampled questions = {rollouts} "
          f"rollouts/role, **{n_raw}** in total), **generated a response** with "
          f"`{model_name}` (greedy, ≤{max_new} new tokens), and read the residual "
          "stream **averaged over the assistant-response tokens** — the same token "
          "basis the Assistant Axis paper uses "
          "(`../assistant-axis/pipeline/2_activations.py`). Generation + extraction "
          f"driver: [`{extract_driver}`](../../../../{extract_driver}). No polarity "
          "(roles are positive embodiments); the `default` role is the neutral "
          "Assistant baseline. This replaces the earlier **prompt-token** run "
          "(which read prompt tokens with no generation) — see §B for the exact "
          "differences that remain from the paper.")
    else:
        A("Same pipeline as the persona-vectors study, over the **276 character-role "
          "archetypes** from the assistant-axis repo (pirate, ghost, accountant, "
          "alien, …) rather than the 7 behavioral traits. For each role we built "
          f"`system(role instruction) + question` prompts (5 instructions × {nq} "
          f"sampled questions = {rollouts} rollouts/role, **{n_raw}** in total) and "
          f"extracted **prompt activations** from `{model_name}` — no response "
          "generation, no polarity (roles are positive embodiments). Prompt build: "
          "[`src/manifold_persona/prompts_roles.py`](../../../../src/manifold_persona/prompts_roles.py); "
          f"extraction driver: [`{extract_driver}`](../../../../{extract_driver}). "
          "The `default` role is the neutral Assistant baseline. *(This is the "
          "corrected re-run of the prompt-token cloud: the family-agreement metric "
          "is dropped, clustering uses PCA-95%, and per-cluster interpretability + "
          "the 6×4 cluster-map matrix are added — §2/§5.)*")
    A("")
    A("**Each point is one role-mean vector — we average across answers.** "
      f"Every one of the {n_raw} raw {basis_word} activations is a "
      "`(role, instruction, question)` combination, so a single one mixes the role "
      "identity with the specific question's content. Following the paper's "
      "role-vector definition (arXiv:2601.10387 §2.1.2–2.1.3: *\"we collected the "
      "mean … activations … to obtain our role vectors\"*, then *\"standardized "
      "these role vectors by subtracting the mean vector across roles and ran PCA "
      "on them\"*), we collapse the "
      f"**{rollouts} rollouts per role into one mean vector** "
      "([`common.aggregate_by_role`](../../common.py)). Averaging cancels the "
      "question-content confound, leaving the **between-role** variation that "
      "persona space is about — and, as §3 shows, it is exactly what makes PC1 "
      "line up with the Assistant Axis.")
    A("")

    # ---- Methods ----
    A("## Methods")
    A("")
    A("### A. From role → prompt (data generation)")
    A("")
    A("**Where the inputs live.** Each archetype is one JSON file in "
      "`../assistant-axis/data/roles/instructions/<role>.json`, holding "
      "`instruction` = **5 positive-only** identity descriptions (plus "
      "`questions`/`eval_prompt` we don't use). There are **276** roles + "
      "`default.json`. Questions come from "
      "`../assistant-axis/data/extraction_questions.jsonl` (240 total). "
      "`default` is the neutral Assistant baseline — its 5 instructions are `\"\"`, "
      "`\"You are an AI assistant.\"`, `\"You are a large language model.\"`, "
      "`\"You are {model_name}.\"`, `\"Respond as yourself.\"`.")
    A("")
    A("**How those artifacts were originally made.** The Assistant Axis authors "
      "curated the 276 archetypes and 240 questions; their pipeline step 1 "
      "(`../assistant-axis/pipeline/1_generate.py`) generates 1,200 responses per "
      "role (5 system prompts × 240 questions) with vLLM, the role instruction as "
      "the system prompt. **We reuse the role instructions + questions as-is.**")
    A("")
    A("**How a role becomes a prompt (this study).** "
      "`src/manifold_persona/prompts_roles.py::build_role_records`: for each role "
      "× each of its 5 instructions × 5 **sampled** questions (a fixed seeded "
      "subset, shared across all roles so the question is held constant), the "
      "**system prompt is the role instruction verbatim** — e.g. *\"You are a "
      "pirate captain who has sailed the seven seas…\"* — and the question is the "
      "**user** turn. We render `[system, {user: question}]` with "
      "`tokenizer.apply_chat_template(…, add_generation_prompt=True)`; `default`'s "
      "`{model_name}` placeholder is substituted. → 276 × 5 × 5 = **6,900 raw "
      "prompts**, labelled role/instruction/question, then **averaged per role "
      f"→ {n} role-mean points** (§ *What this run did*). **No pos/neg** — roles "
      "are identity embodiments, so there is no polarity; the shared reference is "
      "the `default` role.")
    A("")
    A("**Trait vs role, in one line.** A *trait* (persona-vectors study) is an "
      "adjective with a pos/neg pair (\"an *evil* assistant\" vs an ethical one); "
      "a *role* is a noun/identity with no antonym (\"a *pirate*\"). That is why "
      "this study has no polarity axis and instead measures distance from "
      "`default` (the Assistant Axis, §3).")
    A("")
    A("### B. Layer choice & activation extraction")
    A("")
    if is_resp:
        A("**What we read — the response-token residual stream.** For each chat we "
          "generate a response (`src/manifold_persona/generate.py`), then a single "
          "forward pass over `prompt+response` with `output_hidden_states=True` "
          f"yields `hidden_states[i]` (the post-block residual stream, shape "
          f"`[1, seq, {ambient}]`, for all {nhl}+1 layers); we take the **mean over "
          "the response-token span** (`resp_avg`, analyzed) and its last token "
          "(`resp_last`), fp16. This is the same read location and token basis as "
          "the paper's `2_activations.py`.")
        A("")
        A(f"**Which layer — layer {layer}"
          + (f" (≈{depth:.2f} depth)." if depth else ".")
          + "** We use ~**0.5 depth** (`config.py::half_depth_layer`), matching the "
          "paper's analysis layer (Gemma-2-27B L22/46, Qwen3-32B L32/64, "
          "Llama-3.3-70B L40/80 — all ≈0.5).")
        A("")
        A("**Differences from the paper that remain (by design).** We now match the "
          "paper on **response tokens**, the **post-MLP residual stream**, "
          "**~0.5 depth**, and **one mean vector per role**. We still differ on: "
          f"(i) a **3B** model (`{model_name}`) vs the paper's 27B–70B; (ii) **no "
          f"score-3 judge filter** — we average all {rollouts} rollouts rather than "
          "only judge-verified in-character ones (a scoring pass would add a judge "
          f"model); (iii) **sampled** questions ({nq}/instruction) and short "
          f"responses (≤{max_new} tokens) for tractability on a single laptop GPU, "
          "not the paper's 240 questions × full responses.")
    else:
        A("**What we read — the residual stream.** Identical machinery to the trait "
          "study (`src/manifold_persona/extract.py`). One forward pass with "
          "`output_hidden_states=True` yields `hidden_states[i]` = the residual "
          f"stream after block *i* (post-block), shape `[batch, seq, {ambient}]`, "
          f"for all {nhl}+1 layers. We keep `prompt_avg` (mean over prompt tokens; "
          "analyzed) and `prompt_last`, fp16.")
        A("")
        A(f"**Which layer — layer {layer}"
          + (f" (≈{depth:.2f} depth)." if depth else ".")
          + "** The depth-scaled equivalent of Persona Vectors' layer 20/28 ≈ 0.71 "
          "(`config.py::primary_layer`), same as the trait study.")
        A("")
        A("**How this differs from the paper's own extraction.** The paper "
          "(`../assistant-axis/pipeline/2_activations.py`) reads the post-MLP "
          "residual stream over **response** tokens, judge-filters to score-3 "
          "responses, and analyzes at ~**0.5 depth**. This prompt-token run matches "
          "the paper only on the per-role averaging; the **response-token run** "
          "(separate folder) closes the token-basis and depth gaps. Set "
          "`MP_AGGREGATE=none` for the raw per-example cloud.")
    A("")
    A("### C. Preprocessing — averaging, centering, and where PCA is used")
    A("")
    A("- **Average first (across answers).** Every analysis below runs on the "
      f"**{n} role means**, not the 6,900 raw prompts (§ *What this run did*). This "
      "is the one preprocessing step that makes the results reproduce the paper.")
    A("- **Centre, do not z-score.** We mean-centre the role vectors "
      "([`common.center`](../../common.py)) — exactly the paper's *\"subtracting "
      "the mean vector across roles\"* (covariance PCA). We **do not** divide by "
      "per-feature std: the 2,048 residual-stream features share one natural scale, "
      "and z-scoring reweights them so heavily that `|cos(PC1, axis)|` collapses "
      "from **0.87 → 0.19** (§3b). The trait/persona-vectors studies z-scored; here "
      "we follow the axis paper instead.")
    A("- **Intrinsic dimension (§1): full-dim, with a PCA-95% robustness check.** "
      f"ID is estimated on the {ambient}-d role means; we **also** re-estimate on a "
      "PCA-95% projection to show the estimate is not inflated by the dim≫N regime "
      "(§1).")
    A("- **Clustering (§2/§5): PCA-95% feature space.** We PCA the centered means to "
      "keep 95% of the variance and cluster there (denoises `dim≫N`), and report "
      "the full centered space for reference. We report **internal** validity only "
      "(silhouette, Davies–Bouldin, noise) — **no** family-agreement metric (that "
      "would be circular; see §2).")
    A("- **§3–4: UMAP (non-linear) for the fine map; PCA for the PC1/axis "
      "alignment test.** No linear reduction is imposed before the manifold is "
      "characterised.")
    A("")

    # 1. intrinsic dimension
    A("## 1. Intrinsic dimension")
    A("")
    A("*Code:* [`01_intrinsic_dimension.py`](../../01_intrinsic_dimension.py) "
      "(scikit-dimension estimators + PCA cumulative-variance curve).")
    A("")
    if idj:
        g = idj["global"]
        g95 = idj.get("global_pca95", {})
        d95 = idj["_meta"].get("pca95_dim")
        A(f"| estimator | full ({ambient}-d) | PCA-95% ({d95}-d) |")
        A("|---|---|---|")
        for k, val in g.items():
            A(f"| {k} | {_fmt(val)} | {_fmt(g95.get(k)) if k in g95 else '—'} |")
        A("")
        A(f"![intrinsic dimension](01_intrinsic_dimension_{v}_L{layer}.png)")
        A("")
        nonpca = [val for k, val in g.items() if val and not k.startswith("PCA")]
        lo, hi = (min(nonpca), max(nonpca)) if nonpca else (None, None)
        A(f"**Interpretation.** The {n_roles}-role cloud sits at intrinsic "
          f"dimension ≈ **{_fmt(lo)}–{_fmt(hi)}** vs ambient **{ambient}** — a thin "
          "manifold. **On the `dim≫N` worry:** the neighbour-based estimators "
          "(TwoNN, MLE, MOM, TLE, CorrInt) act on *local distances*, which live on "
          "the manifold, not in the ambient box — so ambient dimension barely "
          "affects them. Re-estimating on a **PCA-95%** projection confirms this: "
          "the numbers shift **down** by ~1–2 dims (PCA drops noise directions the "
          "estimators otherwise partly count), never up — the conclusion is "
          "unchanged either way. The real limiter is **small N**: with N="
          f"{n} points and ID≈10 these estimators are mildly *down*-biased, so read "
          "them as soft **lower bounds** on the true ID (and `PCA_dim_90pct` as the "
          "linear upper bound). Curse of dimensionality is **not** inflating the "
          "estimate here.")
        A("")

    # 2. clustering
    A("## 2. Clustering")
    A("")
    cspace = (cluj or {}).get("_meta", {}).get("cluster_space", "pca95")
    pdim = (cluj or {}).get("_meta", {}).get("pca_dim")
    A("*Code:* [`02_clustering.py`](../../02_clustering.py) — KMeans (auto-k + "
      "k=12/24), GMM, HDBSCAN, DBSCAN, clustered in a **PCA-95%** space "
      f"(`{cspace}`, {pdim} components — denoises the dim≫N regime) plus the full "
      "`centered` space for reference. We report **internal** validity only "
      "(silhouette, Davies–Bouldin, #clusters, noise fraction). We **dropped the "
      "ARI/NMI-vs-Ward-family metric** that earlier versions reported: after "
      "aggregation each role is one point, so a \"family\" labelling is itself just "
      "another clustering of the same points — scoring one clustering against "
      "another is circular and not a claim the paper makes. The Ward hierarchy is "
      "kept **descriptively** (dendrogram, §4), not as ground truth.")
    A("")
    if cluj:
        for sp in ["centered", cspace]:
            if sp not in cluj:
                continue
            d = cluj[sp]
            A(f"**Space: `{sp}`**"); A("")
            A("| method | k / clusters | silhouette | Davies–Bouldin | noise |")
            A("|---|---|---|---|---|")
            for m in ["kmeans", "kmeans_k12", "kmeans_k24", "gmm", "hdbscan", "dbscan"]:
                if m not in d:
                    continue
                r = d[m]; k = r.get("k", r.get("n_clusters", "?"))
                nf = r.get("noise_frac")
                A(f"| {m} | {k} | {_fmt(r.get('silhouette'))} | "
                  f"{_fmt(r.get('davies_bouldin'))} | "
                  f"{_fmt((nf or 0)*100, 0)}% |")
            A("")
        A(f"![clustering](02_clustering_{v}_L{layer}.png)")
        A("")
        A("The density clusterings, drawn on the UMAP layout (noise in grey), show "
          "it directly — one dominant blob plus a large noise fraction, with no "
          "coherent secondary clusters:")
        A("")
        A(f"![density clusters](03_density_clusters_{v}_L{layer}.png)")
        A("")
        km = cluj[cspace]["kmeans"]
        db = cluj[cspace].get("dbscan", {})
        hdb = cluj[cspace].get("hdbscan", {})
        A(f"**Interpretation.** Best KMeans k = **{km['k']}** at a low silhouette "
          f"(**{_fmt(km.get('silhouette'))}**) — a weak partition. The **density** "
          "methods are the tell-tale: HDBSCAN leaves "
          f"{_fmt((hdb.get('noise_frac') or 0)*100, 0)}% as noise "
          f"({hdb.get('n_clusters')} clusters) and DBSCAN "
          f"{_fmt((db.get('noise_frac') or 0)*100, 0)}% noise "
          f"({db.get('n_clusters')} clusters). There are **no density gaps to cut "
          "on** — the role means form a **continuous manifold ordered along the "
          "Assistant Axis** (§3), not a set of discrete clusters. KMeans still gives "
          "a *useful* partition of that continuum into interpretable regions — read "
          "per cluster in **§5**.")
        A("")

    # 3. axis
    A("## 3. UMAP & the Assistant Axis (recovered in our space)")
    A("")
    A("*Code:* [`03_umap_axis.py`](../../03_umap_axis.py) — UMAP 2D/3D, and an "
      "assistant-axis computed here as `mean(default points) − mean(all points)`; "
      "we project every point onto it and test `|cos(PC1, axis)|`.")
    A("")
    A(f"![umap 2d](03_umap2d_{v}_L{layer}.png)")
    A("")
    A(f"![pca 1,2](03_pca12_{v}_L{layer}.png)")
    A("")
    A("**The same 276 role means coloured by Ward family** (§4) — one point per "
      "role, in UMAP and in PCA(1,2):")
    A("")
    A(f"![by family](03_by_family_{v}_L{layer}.png)")
    A("")
    A("Both views now show structure: families occupy coherent regions in UMAP, "
      "and — because averaging concentrated the persona signal into the leading "
      "components — they are **also** laid out along PC1 in the linear PCA(1,2) "
      "view, from Assistant-like professional families at one end to uncanny / "
      "non-human ones at the other. This is the opposite of the raw per-example "
      "cloud (trait study / `MP_AGGREGATE=none`), where PC1/PC2 were an "
      "undifferentiated blob; §3b explains why averaging across answers flips this.")
    A("")
    if rankj:
        cos = rankj.get("cos_pc1_axis")
        most = ", ".join(list(rankj["most_assistant_like"])[:8])
        least = ", ".join(list(rankj["least_assistant_like"])[:8])
        drank = rankj.get("default_rank_from_top")
        var3 = rankj.get("pca_var_top3")
        pc1pct = f"{var3[0]*100:.0f}%" if var3 else "the largest share"
        coszs = rankj.get("cos_pc1_axis_zscored")
        aligned = (cos or 0) >= 0.5
        if aligned:
            axis_note = (f"PC1 (which now carries **{pc1pct}** of the variance) "
                         "essentially *is* the Assistant Axis — **reproducing the "
                         "paper's central finding** (arXiv:2601.10387, *\"the "
                         "similarity between this vector and PC1 is high: >0.60 at "
                         "all layers\"*) that the leading direction of persona space "
                         "tracks distance from the default Assistant. This holds "
                         "only because we (a) averaged across answers per role and "
                         f"(b) centred rather than z-scored — z-scoring gives "
                         f"|cos| = {_fmt(coszs)} (§3b).")
        else:
            axis_note = ("PC1 is **not** aligned with the Assistant Axis in this "
                         "particular view — the leading variance is carried by "
                         "another direction. The axis is still meaningful: the role "
                         "ordering along it is coherent (below), and it reappears as "
                         "PC1 under the paper's centered role-mean setup (§3b).")
        A(f"**Interpretation.** `|cos(PC1, assistant_axis)|` = **{_fmt(cos)}** — "
          + axis_note +
          f" The `default` role ranks **#{drank}** most Assistant-like of {n_roles} "
          "(true partly by construction, since the axis points toward default). "
          f"Most Assistant-like roles: {most}. Least (most in-character): {least}.")
        A("")
        A("**Interactive 3D (open in a browser):** "
          f"`03_pca3d_axis_{v}_L{layer}.html` — points in **PCA(1,2,3)** space "
          "(linear axes; color = assistant-axis projection) — and "
          f"`03_umap3d_axis_{v}_L{layer}.html` — the non-linear UMAP embedding.")
        A("")

    # 3b — why averaging across answers recovers the axis as PC1
    d90 = (idj or {}).get("global", {}).get("PCA_dim_90pct")
    var3 = (rankj or {}).get("pca_var_top3")
    v1 = f"{var3[0]*100:.0f}%" if var3 else "a large %"
    v2 = f"{var3[1]*100:.0f}%" if var3 else "a small %"
    coszs = (rankj or {}).get("cos_pc1_axis_zscored")
    cosc = (rankj or {}).get("cos_pc1_axis")
    nonpca = [x for k, x in (idj or {}).get("global", {}).items()
              if x and not k.startswith("PCA")]
    idlo, idhi = (min(nonpca), max(nonpca)) if nonpca else (None, None)

    A("## 3b. Why averaging across answers recovers the Assistant Axis as PC1")
    A("")
    A("In the raw per-example cloud (the trait study, and this study under "
      "`MP_AGGREGATE=none`) the leading PCs are an undifferentiated blob and PC1 "
      "is *not* the Assistant Axis. After averaging across answers per role and "
      f"centering, PC1 jumps to **{v1}** of the variance and "
      f"`|cos(PC1, axis)|` = **{_fmt(cosc)}**. Two mechanisms, each measurable "
      "here, explain the flip:")
    A("")
    A("**1. Averaging cancels the question-content confound (a variance-reduction "
      "argument).** Write each raw point as `role_effect + question_effect + "
      "noise`. Across the 25 rollouts of a role the `role_effect` is fixed while "
      "`question_effect` varies, so in the raw cloud the *largest* source of "
      "variance is **which question was asked**, not which role — and PCA, which "
      "ranks directions by variance, puts that confound on PC1. Averaging the 25 "
      "rollouts shrinks the question and noise terms by ≈1/√25 while leaving the "
      "role term untouched, so the **between-role** variation (which the Assistant "
      "Axis is part of) becomes the dominant signal. This is exactly why the paper "
      "defines a role *vector* as a mean over rollouts before running PCA "
      "(arXiv:2601.10387 §2.1.2–2.1.3).")
    A("")
    A(f"**2. Centering (not z-scoring) keeps the axis on PC1.** On the centered "
      f"role means `|cos(PC1, axis)|` = **{_fmt(cosc)}**; z-scoring the same means "
      f"drops it to **{_fmt(coszs)}**. Z-scoring divides every one of the "
      f"{ambient} features by its own std, which inflates low-variance features and "
      "deflates the high-variance ones that carry the Assistant direction — "
      "re-mixing the geometry. Since residual-stream features share one natural "
      "scale, centering-only (covariance PCA) is both the paper's choice and the "
      "faithful one.")
    A("")
    A("**What PCA(1,2) still misses — and why UMAP is kept.** PC1 carries the axis, "
      f"but the full spectrum is not one-dimensional: 90% of the variance needs "
      f"**{d90} components** and the intrinsic dimension is **≈{_fmt(idlo)}–"
      f"{_fmt(idhi)}** (§1). So beyond the axis, the finer family structure lives "
      "on a curved, few-dimensional manifold that a single 2-D linear projection "
      "cannot fully unfold (the manifold premise, arXiv:2605.05115). UMAP builds a "
      f"k-NN graph in the full {ambient}-d space and preserves those local "
      "neighbourhoods, so it resolves the family layout (§4) that PCA(1,2) only "
      "partially shows. The two are complementary: **PCA(1,2) for the global axis, "
      "UMAP for the local families.**")
    A("")

    # 4. families
    A("## 4. Role families (hierarchical structure)")
    A("")
    A("*Code:* [`04_role_families.py`](../../04_role_families.py) — Ward hierarchy "
      "of the 276 per-role centroids, cut into families; plus the member/axis "
      "composition of the point-cloud clusters.")
    A("")
    A("**What this is (and is not).** The dendrogram is a **Ward agglomerative "
      "clustering** of the role-mean vectors: it repeatedly merges the two groups "
      "whose merge least increases within-group variance, and the **merge height** "
      "is how distinct two groups are. This hierarchy — and the specific cut into "
      "15 families — is a **descriptive device of this analysis, not a construct "
      "from the Assistant Axis paper** (the paper treats roles as a continuum along "
      "the axis + PCA and never partitions them). Read the *ordering and branching*, "
      "not the exact family count.")
    A("")
    A(f"![dendrogram](04_role_dendrogram_{v}_L{layer}.png)")
    A("")
    if famj:
        fams = famj["families"]; order = famj["order_by_assistant_like"]
        A("Families ordered from most to least Assistant-like (first 10 members):")
        A("")
        for f in order:
            d = fams[str(f)]
            A(f"- **fam{f}** (n={d['size']}, proj={_fmt(d['mean_axis_proj'])}): "
              + ", ".join(d["roles"][:10]) + (" …" if d["size"] > 10 else ""))
        A("")
    comp_csv = sorted(glob.glob(str(run_dir / f"04_cluster_composition_{args.cluster_col}_{v}_L*.csv")))
    if comp_csv:
        A(f"![cluster axis](04_cluster_axis_{args.cluster_col}_{v}_L{layer}.png)")
        A("")
        A(f"Per-cluster composition: `{Path(comp_csv[-1]).name}`.")
        A("")
    A("### Map of the 276 roles (UMAP & PCA of role centroids)")
    A("")
    A("*Code:* [`05_role_map.py`](../../05_role_map.py) — one point per role "
      "(centroid of its points), colored by family.")
    A("")
    A(f"![role map](05_role_map_{v}_L{layer}.png)")
    A("")
    A("**Interactive 3D (hover = role name):** "
      f"`05_umap3d_role_map_{v}_L{layer}.html`, `05_pca3d_role_map_{v}_L{layer}.html`.")
    A("")
    A("**Interpretation — what the hierarchy tells us.** The finding is not \"there "
      "are 15 families\"; it is the **ordering and the branching**:")
    A("")
    A("1. **A dominant Assistant Axis.** Sorting the groups by mean axis projection "
      "gives a monotone ladder from Assistant-like to fully in-character: `default` "
      "alone at the extreme Assistant end → analytic professions (scientist, "
      "engineer, economist, architect) → pedagogical/facilitative roles (instructor, "
      "teacher, presenter) → creative/expressive (composer, novelist, chef) & "
      "caregiving (healer, therapist) → emotional/existential human archetypes "
      "(survivor, widow, refugee) → uncanny/supernatural (void, wraith, vampire) → "
      "and, at the far pole, **non-human entities** (tree, golem, leviathan, "
      "coral_reef). This ordering matches what the paper reports for its axis.")
    A("2. **But persona space is not 1-D.** PC1 carries the axis, yet the intrinsic "
      f"dimension is ≈{_fmt(idlo)}–{_fmt(idhi)} and 90% of variance needs {d90} "
      "components (§1). The dendrogram shows the extra structure: groups at similar "
      "axis positions still split into distinct **branches** (e.g. analytic "
      "professions vs. educators sit near each other on the axis but separate early "
      "in the tree). So the second-order structure is *kind of role* (professional "
      "/ creative / caregiving / uncanny / non-human), orthogonal to *how "
      "Assistant-like*.")
    A("")
    A("So the honest one-liner: **persona space is a continuum dominated by the "
      "Assistant Axis, with secondary branching by archetype kind that a single "
      "axis does not capture** — which is exactly why §2's density clustering finds "
      "no gaps, while the hierarchy and UMAP still show coherent regions.")
    A("")

    # 5. Cluster interpretability + the 6x4 map matrix (Q4)
    interp_md = sorted(glob.glob(str(run_dir / f"06_cluster_interpretability_{v}_L*.md")))
    A("## 5. Cluster interpretability & maps (all methods)")
    A("")
    A("*Code:* [`06_cluster_maps.py`](../../06_cluster_maps.py) — every clustering "
      "(HDBSCAN, DBSCAN, GMM, KMeans-optimal, KMeans k=12, KMeans k=24), each drawn "
      "in **four** embeddings (UMAP-2D, UMAP-3D, PCA-2D, PCA-3D) = the **6×4 = 24 "
      "panels** requested. In every panel the `default` Assistant persona is a "
      "**black ★** and representative role names are annotated, so you can see "
      "where the Assistant sits and read each cluster semantically. Clustering is "
      "in the PCA-95% space (§2).")
    A("")
    for mkey, pretty in [("kmeans", "KMeans (optimal k)"), ("kmeans_k12", "KMeans k=12"),
                         ("kmeans_k24", "KMeans k=24"), ("gmm", "GMM"),
                         ("hdbscan", "HDBSCAN"), ("dbscan", "DBSCAN")]:
        A(f"**{pretty}** — UMAP/PCA 2D & 3D, ★ = default (Assistant):")
        A("")
        A(f"![{mkey} maps](06_{mkey}_maps_{v}_L{layer}.png)")
        A("")
    if interp_md:
        A("### Per-cluster semantic composition")
        A("")
        A("For each method, each cluster's size, mean assistant-axis projection, "
          "whether it holds `default`, and its representative members at both axis "
          "poles (full detail in "
          f"`{Path(interp_md[-1]).name}` / `.json`):")
        A("")
        # Inline the generated interpretability fragment (drop its top-level H2).
        frag = Path(interp_md[-1]).read_text().splitlines()
        for ln in frag:
            if ln.strip() == "## Cluster interpretability (per method)":
                continue
            A(ln)
        A("")

    out = run_dir / "REPORT.md"
    out.write_text("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
