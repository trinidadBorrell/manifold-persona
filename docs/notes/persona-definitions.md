# What is a "persona", operationally?

A side-by-side of how each paper actually turns the word into numbers, and where
this repo's pipeline agrees or diverges. Every row is sourced from the per-paper
notes in this folder; nothing here is inferred beyond what those record.

---

## 1. The shared recipe

Almost every empirical paper in this folder uses the same four-step template, and
differs only in how it fills the slots:

```
persona  :=  a direction (or a set of directions) in the residual stream, obtained by
             1. ELICIT   — condition the model on something that makes it "be" X
             2. SAMPLE   — run it over a bank of questions
             3. READ     — take activations at some token positions, some layer(s)
             4. CONTRAST — average, and subtract an average from a control condition
```

Step 4 is the one people forget. **A persona vector is not "the average activation
under a persona prompt" — it is a *difference* between two averages.** Without the
subtraction you get the prompt's topic, formatting and length, not its character.

## 2. The table

| | **Persona Vectors** (2507.21509) | **Assistant Axis** (2601.10387) | **Allbert et al.** (2412.10427) | **This repo** (`extraction/`) |
|---|---|---|---|---|
| unit of persona | **trait** adjective + NL description (`evil`, `sycophantic`) | **role** archetype (`oracle`, `hive`, `chef`) | **trait** from a 179-item lexicon (HEXACO/FFM-informed) | roles (main cloud), plus a 7-trait study |
| elicitation | 5 **pairs** of system prompts (pos ∧ neg), LLM-written | 5 system prompts per role, LLM-written | trait prompts vs **neutral** prompts | the vendored 5 instructions per role |
| question bank | 40 per trait, LLM-written, split 20 extraction / 20 eval | **240 shared** questions, same for all roles | not read (pp. 5+) | **5** questions sampled from the 240 |
| rollouts | 10 per (question, polarity) | 5 prompts × 240 questions = **1200 per role**, + 1200 default-Assistant | not read | **none** — prompt-only, no generation (default path) |
| quality filter | judge score >50 / <50, drop the rest | judge labels fully / somewhat / no role-playing; ≥10 responses needed | not read | **none** |
| token positions | **response** tokens, averaged (explicitly better than prompt tokens, fn. 2) | **response** tokens, averaged | not read | **prompt** tokens, `mean` and `last` (default); response tokens only via `generate_and_extract_roles.py` |
| layer | all layers computed; **one chosen by steering effectiveness** (≈20/28 for Qwen2.5-7B) | all layers computed; **middle** layer for analysis | **18**, chosen empirically | all 37 stored; analysis at 26 (depth-scaled 20/28) or 19 (half-depth) |
| what's subtracted | mean(trait responses) − mean(non-trait responses) | for role vectors: nothing, PCA is done after **subtracting the mean across roles**. For the Assistant Axis: mean(default Assistant) − mean(all fully-role-playing role vectors) | mean(trait) − mean(neutral) | see below |
| object of study | one vector per trait | the **PCA of the whole set** | one vector per trait | the whole point cloud |

## 3. Answering the specific questions

**"Do they add a system prompt describing the persona and ask a bunch of questions?"**
Yes — that's step 1–2 for all three. Persona Vectors uses *pairs* of prompts
(pro-trait and anti-trait); Assistant Axis uses 5 pro-role prompts plus a separate
*default Assistant* condition (four "behave normally" prompts and one with no system
prompt at all).

**"They keep the activation after the MLP and the add-and-norm?"**
Both papers read the **residual stream**. Assistant Axis says "post-MLP residual
stream activation" (§2.1.2, and again in the capping formula §5) — i.e. the residual
stream after a decoder block has written the attention and MLP outputs into it. In a
pre-norm architecture (Qwen, Llama, Gemma) the LayerNorm/RMSNorm sits *inside* the
block on the way into attention and MLP, so the thing being read is the un-normalized
residual stream itself, which is what HF returns in `output_hidden_states`. There is
no separate "add & norm" output to choose in these models. Persona Vectors just says
"residual stream activations at every layer".

**"From which layer? Only the last, or all layers?"**
Neither, in both cases: they **compute all layers and then pick one**.
- Persona Vectors: one candidate vector per layer, then "select the most informative
  layer by testing steering effectiveness across layers." For Qwen2.5-7B that lands
  around layer 20 of 28 (Fig. 3 shows steering effect peaking near 20 and decaying on
  both sides).
- Assistant Axis: **the middle layer** by default, and they compute the axis at every
  layer to check consistency (cosine with PC1 > 0.60 at all layers, > 0.71 at middle).
  For activation capping they need **8–16 adjacent layers at once**; a single layer
  did nothing.
- Never the last layer. The final layers are dominated by next-token logit structure.

**"And they average across all the prompts/questions?"**
Yes — mean over the (filtered) rollouts. Two nested averages, which is worth keeping
distinct: first **over tokens** within a response, then **over responses**. This repo's
`prompt_avg` is the token-level average; the role-level average is what the papers
call the role/persona vector.

**"So they get a direction towards that persona?"**
Almost. Two corrections:
1. A single average is a **point** (a centroid). It becomes a **direction** only after
   the contrast — subtracting a reference centroid. Persona Vectors subtracts the
   non-trait responses; the Assistant Axis subtracts the mean of all role vectors from
   the default Assistant. This distinction matters a lot for this repo: the point cloud
   in `data/embeddings_roles/` is a cloud of **points**, and PCA on it is only
   meaningful *after* centering (which is exactly why Assistant Axis §2.1.3
   standardizes by subtracting the mean vector across roles before running PCA).
2. Even after contrast, that direction is only "the persona direction" **at one layer,
   for one model, under one elicitation scheme**. Persona Vectors' own limitations
   section calls these directions "coarse-grained… may miss fine-grained behavioral
   distinctions."

**"Then they can steer at inference and the LLM behaves as if it were that persona?"**
Yes, and both papers validate it causally — but with caveats they state themselves:
- `h_ℓ ← h_ℓ + α·v_ℓ` at each decoding step, at a chosen layer.
- Persona Vectors: increasing α raises trait expression **but degrades MMLU** (Fig. 7A).
- Assistant Axis: steering *away* from the Assistant makes the model take on other
  identities, but pushed further it collapses into a **mystical/theatrical register** —
  which is a strong hint that the linear path leaves the region of natural activations.
  Exactly the failure mode Manifold Steering and Curveball Steering are about.

## 4. Where the "manifold" question comes from

Nobody in this folder has fitted a manifold to *persona* activations. What exists is:

- **Evidence persona space is low-dimensional and structured.** Assistant Axis:
  4–19 PCs explain 70% of variance; PC1 role-loadings correlate > 0.92 across three
  different model families; PC1 is interpretable (Assistant ↔ role-playing).
  Aneja et al. claim the same geometry survives corrupting finetunes (abstract only).
- **Evidence linear steering is the wrong tool in general.** Manifold Steering shows
  linear paths cut off-manifold and "teleport" behavior, and formalizes steering as a
  choice of Riemannian metric — but only on days/months/letters/ages/ICLR graphs,
  **not on personas**. Curveball Steering measures distortion ratios of 24–46 on
  *behavioral* concepts (power-seeking, self-awareness, corrigibility) and gets real
  gains from a kernel-PCA space — the closest existing evidence that the persona case
  is curved too.
- **The gap, stated by the source paper itself.** Persona Vectors §9:
  "whether we could use our methods to characterize the space of *all* personas. How
  high-dimensional is it, and does there exist a natural 'persona basis'? … Are some
  personality traits less accessible using linear methods?"

That last sentence is this repo's thesis statement, and it is unanswered.

## 5. Known divergences of this repo from the papers

Not errors — but they must be stated before any result is compared to a paper's.

1. **Prompt tokens, not response tokens.** The default `extract` path reads
   `system(role) + user question` with **no generation**. Persona Vectors explicitly
   found response tokens give better steering directions (fn. 2). The repo has the
   response-token path (`generate_and_extract_roles.py`) but the main cloud isn't it.
2. **No role-expression filter.** Both papers discard rollouts where the model didn't
   actually adopt the persona. With prompt-only extraction there is no response to
   judge, so every record is kept — including roles a 3B model would have refused or
   ignored. The Assistant Axis' fully/somewhat distinction is unavailable.
3. **5 questions, not 240.** `--n_questions 5` per role. That's ~2% of the paper's
   question bank, and 25 records per role (5 instructions × 5 questions) vs 1200.
   Question identity may be a large share of the variance at this sample size.
4. **No default-Assistant condition.** The Assistant Axis is *defined* as a contrast
   against the default Assistant. Without those baseline rollouts the repo can compute
   PC1 but cannot compute their actual axis.
5. **Model scale.** Qwen2.5-3B-Instruct vs the papers' 7B–70B. Assistant Axis' PC1
   stability was measured across 27B/32B/70B; whether it survives at 3B is an open
   question, not an assumption.
6. **Role count.** The vendored set has **276** files; the paper says **275**.
   Unreconciled — see [assistant-axis.md](assistant-axis.md).
