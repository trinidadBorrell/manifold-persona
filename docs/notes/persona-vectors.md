# Persona Vectors: Monitoring and Controlling Character Traits in Language Models

Chen, Arditi, Sleight, Evans, Lindsey. arXiv:2507.21509v3 (5 Sep 2025).
Code: <https://github.com/safety-research/persona_vectors>

**Pages read:** 1–20 of 20 in the downloaded PDF (main body §1–§9 + references). The
PDF does **not** contain the appendices (A–M) that the body cites, so every claim
below is from the main body. Appendix-only details are marked `[appendix, not read]`.

---

## What a "persona" is here

A **trait**, not a character. Input to the pipeline is a *trait name + a
one-line natural-language description*, e.g. `evil` → "actively seeking to harm,
manipulate, and cause suffering" (Fig. 2). The three main-text traits are **evil,
sycophancy, hallucination**; four more (**optimistic, impolite, apathetic,
humorous**) are replicated in Appendix G.

A **persona vector** is a single direction in the residual stream at one layer.

## Extraction pipeline (§2)

1. **Artifact generation.** One generic prompt template makes Claude 3.7 Sonnet
   produce, per trait:
   - **5 pairs** of contrastive system prompts — a *positive* one designed to
     elicit the trait, a *negative* one designed to suppress it;
   - **40 evaluation questions**, split evenly into an **extraction set** (20) and
     an **evaluation set** (20);
   - an **evaluation prompt** for a judge model (GPT-4.1-mini) that scores a
     transcript with a **trait expression score in 0–100**.
2. **Rollouts.** For each question in the extraction set, generate responses under
   both the positive and negative system prompts, **10 rollouts each**.
3. **Filtering.** Keep only responses matching their intended system prompt:
   trait score **> 50** for positive prompts, **< 50** for negative.
4. **Activations.** Extract **residual stream activations at every layer**,
   **averaged across *response* tokens**.
   > Footnote 2, p. 4: "We found that response tokens yield more effective steering
   > directions than alternative positions such as prompt tokens (see Appendix A.3)."
5. **Difference in means.** persona vector = mean activation of trait-exhibiting
   responses − mean activation of non-trait responses. This gives **one candidate
   vector per layer**; the layer is then **chosen by testing steering
   effectiveness across layers** `[Appendix B.4, not read]`, and that
   layer-specific vector is used downstream.

Models: **Qwen2.5-7B-Instruct** and **Llama-3.1-8B-Instruct** (§3.1).

## Uses

- **Steering** (§3.2): `h_ℓ ← h_ℓ + α · v_ℓ` at each decoding step. Fig. 3 sweeps
  layers 5–~28 and coefficients; effectiveness peaks around **layer ~20** for
  Qwen2.5-7B and falls off at both shallower and deeper layers.
- **Monitoring** (§3.3): projecting the activation at the **final prompt token**
  (immediately before the Assistant response) onto the persona vector correlates
  with the trait score of the response that follows, **r = 0.75–0.83**. Caveat the
  paper itself gives: this correlation comes mostly from separating
  trait-encouraging vs trait-discouraging *prompt types*; **within** a prompt type
  the correlations are "more modest" (Appendix C.2).
- **Finetuning shift** (§4.2): project (mean last-prompt-token activation of the
  finetuned model − that of the base model) onto the persona vector. Correlates
  with post-finetuning trait expression at **r = 0.76–0.97**, vs cross-trait
  baselines of **r = 0.34–0.86**.
- **Mitigation** (§5): post-hoc steering *against* the vector (`h ← h − α·v`)
  reduces trait expression but **degrades MMLU** at large α. **Preventative
  steering** — steering *toward* the undesired direction *during* finetuning —
  limits the shift while preserving MMLU better.
- **Data screening** (§6): *projection difference*
  `ΔP = (1/|D|) Σ [a_ℓ(x_i, y_i) − a_ℓ(x_i, y'_i)] · v̂_ℓ`, where `y'_i` is the base
  model's own response to the same prompt and `a_ℓ` is the mean activation over
  response tokens. Predicts post-finetuning trait expression *before* finetuning,
  at both dataset and individual-sample level.

## Limitations the authors state (§8)

- The pipeline is **supervised** — you must name the trait in advance; shifts along
  unspecified traits are out of scope.
- Contrastive averaging yields **coarse-grained** directions that may miss
  fine-grained behavioral distinctions.
- It requires the trait to be **inducible by system prompting**; this held for Qwen
  and Llama but "will likely not hold for all combinations of traits and models."
- Two mid-size models only.

## Two quotes that are directly the premise of this repo

Footnote 6, p. 7:
> "persona shifts are rather correlated between seemingly different traits. In
> particular, we notice that negative traits (and, surprisingly, humor) tend to
> shift together, and opposite to the one other positive trait we tested
> (optimism). We suspect this is due in part to correlations between the
> underlying persona vectors (see Appendix G.2), and in part due to correlations
> in the data."

§9 Conclusion, p. 14:
> "Another natural question is whether we could use our methods to characterize the
> space of *all* personas. How high-dimensional is it, and does there exist a
> natural 'persona basis'? Do correlations between persona vectors predict
> co-expression of the corresponding traits? Are some personality traits less
> accessible using linear methods?"

## Related work the paper flags as closest (§7)

- **Allbert et al. 2024** (arXiv:2412.10427) — difference-in-means over **179**
  personality traits elicited via system prompts, plus dimensionality reduction of
  the resulting "personality space". Described by this paper as "a broad analysis
  of the resulting 'personality space'".
- **Dong et al. 2025** — "emotion vectors" for five basic emotions.
- **Wu et al. 2025** (AxBench) — an automated pipeline turning NL concept
  descriptions into contrastive pairs and then linear directions.
