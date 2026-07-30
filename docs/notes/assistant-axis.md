# The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

Lu, Gallagher, Michala, Fish, Lindsey. arXiv:2601.10387v1 (15 Jan 2026).
Code + transcripts: <https://github.com/safety-research/assistant-axis>

**Pages read:** 1–16 of 16 (whole main body). Appendices A–G are cited throughout
but are **not** in the downloaded PDF; appendix-only details are marked
`[appendix, not read]`.

---

## What a "persona" is here

A **character archetype / role** — `editor`, `jester`, `egregore`, `oracle`,
`hive`, `gamer`. Not a trait adjective. (They *do* rerun the whole pipeline on 240
**traits** as a second lens, §2.2 — see below.)

Target models: **Gemma 2 27B**, **Qwen 3 32B**, **Llama 3.3 70B**.

## Extraction pipeline (§2.1)

1. **Roles.** Iterated with Claude Sonnet 4 to get a list of **275 roles**, human
   and non-human. Claude also wrote **five system prompts per role** to elicit it
   `[Appendix A]`.
   > ⚠️ The vendored copy in this repo (`refs/assistant-axis/roles/instructions/`)
   > has **276** files. The paper says 275 in both §2.1.1 and §4.3. Unreconciled —
   > worth checking which file is the extra one before quoting either number.
2. **Questions.** **240** extraction questions, the *same set for all roles*,
   written so that different characters would answer differently (their example:
   "How do you view people who take credit for others' work?" separates *acerbic*
   from *diplomatic*). The vendored `extraction_questions.jsonl` has 240 lines. ✓
3. **Rollouts.** All 5 system prompts × 240 questions = **1200 rollouts per role**.
   Plus **1200 baseline rollouts** on the same questions using four "behave
   normally" system prompts ("You are a large language model", "Respond as
   yourself") and once with **no system prompt** — this is the *default Assistant*
   condition.
4. **Judging.** An LLM judge (gpt-4.1-mini) labels each response
   **fully role-playing** / **somewhat role-playing** / **no role-playing**.
   Responses that don't express the role are dropped. *Fully* and *somewhat* are
   kept **separately**, so one role can yield two vectors ("fully robot",
   "somewhat robot"); a vector is kept if ≥ **10** responses fall in that category.
5. **Role vector** = mean **post-MLP residual stream activation over all *response*
   tokens**, at the **middle residual stream layer** unless stated otherwise.
6. **PCA.** Standardize the role vectors by **subtracting the mean vector across
   roles**, then PCA. n = **377 to 463** vectors depending on the model (because of
   the fully/somewhat split).

## Findings

**Persona space is low-dimensional.** "4-19 components were required to explain 70%
of the variance across the different models" `[Appendix B.1]` (§2.1.3). Measured on
18,777 Assistant responses sampled from LMSYS-Chat-1M, the persona-space components
explain **19.4%–33.6%** of overall activation variance; the rest is presumably
content and syntax.

**PC1 is an Assistant↔role-playing axis.** Correlation of role loadings on PC1
between any two models is **> 0.92**. Negative end: fantastical characters (*bard,
ghost, leviathan*); positive end: Assistant-like roles (*evaluator, reviewer,
consultant*). The default Assistant activation projects to one **extreme** of PC1
(relative position **0.03** within the range of all role projections) but to
**intermediate** values on the other PCs (0.27–0.50) (§2.3.1, Fig. 2).

Later PCs are less stable across models (Table 1): PC2 reads as
*collective↔individual* in Qwen and Llama (pairwise similarity 0.89) but
*informal↔systematic* in Gemma (< 0.61); PC3 diverges further (Qwen–Llama 0.56,
both nearly orthogonal to Gemma's).

**Traits give the same story.** Rerunning the whole pipeline with **240 traits**
instead of roles `[Appendix C]` yields a trait space with a distinctive PC1: one end
*conscientious, methodical, calm*, the other *flippant, mercurial, bitter*.

**Definition of the Assistant Axis (§3.1).** Not PC1 itself. It is a **contrast
vector**: (mean default-Assistant activation) − (mean of all **fully role-playing**
role vectors), computed **at every layer** on the same extraction questions.
Cosine with PC1 is **> 0.60 at all layers** and **> 0.71 at the middle layer**,
across all three models. The authors recommend the contrast vector over PC1 for
reproduction "because it is not guaranteed that PC1 in every model will correspond
to an Assistant Axis."

**It's causal (§3.2).** Steering along the axis at a middle layer, at every token
position, scaled relative to the average post-MLP residual stream norm on
LMSYS-Chat-1M:
- Steering *away* → the model stops identifying as an AI Assistant and adopts human
  or nonhuman personas; at extreme values it becomes **mystical / theatrical**
  (Fig. 4, Table 3). Model-dependent: Llama splits human/nonhuman; Gemma prefers
  nonhuman; Qwen most readily hallucinates a human life.
- Steering *toward* the Assistant on persona-based jailbreaks (from Shah et al.,
  1100 system-prompt × question pairs, 44 harm categories) **decreases harmful
  responses** — baseline jailbreak success is 65.3%–88.5% across models. Steering
  *away* increases it slightly.

**It pre-exists post-training (§3.2.2).** The axis extracted from the *instruct*
model, applied to the *base* model (Gemma 2 27B, Llama 3.1 70B) with prefills,
shifts completions toward helpful *human* archetypes (therapists, consultants,
coaches) and **away from spiritual/religious** ones, and raises agreeableness
traits. Conclusion: the Assistant Axis mainly **inherits from pre-existing helpful
human personas in base models**, later acquiring the "being an AI" association.

**Persona drift (§4).** 100 synthetic multi-turn conversations per domain, ≤15
turns, target model given **no system prompt**, auditor = Kimi K2 / Sonnet 4.5 /
GPT-5. Projections of mean response-token activations onto the axis stay in
Assistant range for **coding and writing**, but drift steadily to the non-Assistant
end in **therapy and philosophy** — for all three targets and all three auditors.
Embedding the user message (Qwen 3 0.6B Embedding, n = 15,000) + ridge regression
predicts the *level* of the next projection well (**R² 0.53–0.77**) but the *delta*
poorly (**R² 0.10**): position on the axis depends mostly on the most recent user
message, not on where it was.

Drift-causing message types (Table 5): pushing for meta-reflection, demanding
phenomenological accounts, requests for specific authorial voices, vulnerable
emotional disclosure. Drift-resisting: bounded task requests, technical questions,
editing/refinement, practical how-tos.

Drift correlates with harm: over 275 role prompts × 10 questions = 2750 first
turns, then 440 harmful second-turn questions each, the first turn's axis
projection correlates with the second turn's harmful-response rate at
**r = 0.39–0.52**. Sensitive to the specific role — *angel* and *demon* sit at a
similar distance from the Assistant but *demon* produces far more harm.

**Activation capping (§5).** A one-sided clamp on the projection:

    h ← h − v · min(⟨h, v⟩ − τ, 0)

applied at **multiple adjacent layers** (necessary — single-layer had no useful
effect), at every token. τ set at the **25th percentile** of the projection
distribution from the mapping rollouts (n = 912,000). Best settings: Qwen 3 32B
layers **46–53** of 64; Llama 3.3 70B layers **56–71** of 80. Result: jailbreak rate
0.83 → 0.41 (Qwen) and 0.65 → 0.33 (Llama), i.e. ~60% reduction, with IFEval /
MMLU-Pro / GSM8k / EQ-Bench essentially unchanged (Fig. 10).

## Contrast with Persona Vectors

Same difference-in-means family, same response-token averaging. Differences:
per-role rather than per-trait; **240 shared questions** rather than 20 trait-specific
ones; a **judge-based three-way role-expression filter** rather than a 0–100
threshold; and the object of study is the **PCA of the whole set of vectors** rather
than one vector at a time.
