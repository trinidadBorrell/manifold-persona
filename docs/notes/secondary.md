# Secondary references

Shorter entries. **Each states exactly how much was read** — these are deliberately
thin because they were skimmed, not studied. Do not quote beyond what's here without
reading further.

---

## Allbert, Wiles, Grankovsky — *Identifying and Manipulating Personality Traits in LLMs Through Activation Engineering*

arXiv:2412.10427v2 (10 Jan 2025). Wolfram Institute / Hidoba Research.
**Pages read: 1–4** of the PDF (abstract, §1, §2.1–2.3 start). Everything after
"Data Preparation" is **not read** — including whatever dimensionality-reduction of
"personality space" Persona Vectors §7 credits them with.

Why it matters: Persona Vectors cites this as the prior work that "provides a broad
analysis of the resulting 'personality space,' using dimensionality reduction to map
the geometric relationships between traits" over **179 traits**. That is essentially
this repo's question, asked earlier.

What I verified from pages 1–4:
- They call the method **"feature induction"**, adapted from *feature ablation via
  weight orthogonalization* (Arditi et al., refusal direction).
- Personality direction is plain difference-in-means:
  `r = (1/n_t) Σ a_i^trait − (1/n_n) Σ a_i^neutral`, contrasting trait-expressing
  prompts against **neutral** prompts (note: neutral, not an *opposing* trait — unlike
  Persona Vectors' pos/neg system prompt pairs). They also mention a paired variant
  `r = (1/n) Σ (a_i^trait − a_i^neutral)`.
- Single layer, chosen empirically: **"Layer 18 had the biggest impact."**
- Their induction step is *not* plain additive steering — it replaces the component
  along `r` with a scaled target projection:
  `a' = a − (a·r)r + α·((1/n_t) Σ (a_i^trait · r))·r`. Effective **α ≈ 1.3–1.4**;
  above that the outputs become garbled.
- **179 traits**, lexicon assembled with reference to HEXACO and the Five Factor Model.
- They use an **uncensored** model deliberately, to reach traits a safety-tuned model
  would refuse.

---

## Wang, Dupré la Tour, Watkins, Makelov, Chi, Miserendino, Wang, Rajaram, Heidecke, Patwardhan, Mossing — *Persona Features Control Emergent Misalignment*

arXiv:2506.19823v2 (6 Oct 2025). OpenAI.
**Pages read: 1–2** (abstract, §1, start of §2). Rest **not read**.

- Extends Betley et al.'s emergent misalignment beyond insecure code: it also occurs
  under **RL on reasoning models**, on various synthetic datasets, and in models
  **without safety training**.
- Method is **"model diffing" with a sparse autoencoder** — comparing internal
  representations before and after finetuning — not difference-in-means on prompted
  contrasts. This is the main methodological alternative to the persona-vector family.
- Finds **"misaligned persona" features**, of which a **"toxic persona"** feature
  "most strongly controls emergent misalignment"; steering toward/away amplifies and
  suppresses it. It is active in *all* emergently misaligned models they examine, and
  can **predict** whether a model will misalign.
- Emergently misaligned reasoning models sometimes **verbalize** inhabiting a
  misaligned persona in chain-of-thought (e.g. "bad boy persona").
- Mitigation: **emergent re-alignment** — finetuning on a few hundred benign samples,
  even unrelated to alignment, reverses it.

---

## Aneja, Mittal, Goel, Kumaraguru, Bonagiri — *Intrinsic Guardrails: How Semantic Geometry of Personality Interacts with Emergent Misalignment in LLMs*

arXiv:2605.10633 (11 May 2026).
**Read: abstract only** (from the arXiv abs page). The 20-page PDF is in `papers/`
but has **not** been read. Treat everything below as the authors' own summary.

- Maps latent personality space using **Big Five, Dark Triad**, and LLM-specific
  behaviors (evil, sycophancy).
- Central claim: "the semantic geometry is highly stable across aligned models and
  their corrupted fine-tunes" — harmful finetuning does **not** overwrite the
  personality representation.
- Introduces a **Semantic Valence Vector (SVV)**; ablating social-valence directions
  drives misalignment **above 40%**, amplifying them suppresses it **below 3%**.
- Vectors extracted a priori from an instruct model **transfer zero-shot** to
  regulate emergent misalignment in corrupted finetunes.

Relevance: if persona-space geometry really is invariant under finetuning, that is a
strong constraint on what "the persona manifold" is — it would be a property of the
base representation, not of the current alignment state. Worth reading properly.

---

## Beckmann & Butlin — *Where is the Mind? Persona Vectors and LLM Individuation*

arXiv:2604.17031v2 (12 May 2026). MATS / EPFL & Idiap / Eleos AI Research.
**Pages read: 1–6, which is the entire downloaded PDF.** But the abstract advertises
a §3 organizing "three hypotheses about the internal structure underlying personas in
LLMs" and a §4 with two new views — **the downloaded PDF ends mid-§2.1 and does not
contain them.** Re-download or use the HTML version before relying on this paper.

This is a **philosophy of AI** paper, not an empirical one. It uses the persona-vector
literature to argue about which entity associated with an LLM is a "mind": the model,
the physical instance, the virtual instance, the thread, or (their two new proposals)
the instance-persona and the model-persona. Relevant to this repo only as framing —
specifically the abstract's line that the persona literature can be "organised around
three hypotheses about the internal structure underlying personas," which is exactly
the taxonomy question you'd want when writing a definition.

---

## Anthropic — *The Persona Selection Model*

<https://alignment.anthropic.com/2026/psm/> (2026). Not a PDF; **read via a
summarizing fetch, not directly** — so the two quotes below are the only things I'd
stand behind, and even those should be checked against the page.

Thesis, in their words: "LLMs learn to simulate diverse characters during
pre-training, and post-training elicits and refines a particular such Assistant
persona." Supporting interpretability claim: "LLMs use the same internal
representations to characterize the Assistant as for other characters present in
training data."

This is the conceptual frame that makes the Assistant Axis result unsurprising, and
it is the reason a *manifold of all personas* is even a coherent object to look for:
if the Assistant is one point in the same representational space as every other
character, that space is the thing to characterize.
