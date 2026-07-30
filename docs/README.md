# docs/ — reference papers and notes

PDFs live in `papers/` (named by arXiv id). Notes live in `notes/`.

**Reading-provenance rule for this folder.** Every note states exactly which pages
of the PDF were read. Anything not read is either omitted or explicitly marked
`[not verified]`. Prefer an incomplete note over a confident wrong one. If you
add a claim, add the page it came from.

## Papers

| id | short name | what it is | note |
|---|---|---|---|
| [2507.21509](papers/2507.21509.pdf) | **Persona Vectors** (Chen, Arditi, Sleight, Evans, Lindsey — Anthropic Fellows, v3 Sep 2025) | Per-**trait** (evil, sycophancy, hallucination) linear directions from contrastive system prompts; monitoring, steering, finetuning-shift prediction, data screening. | [persona-vectors.md](notes/persona-vectors.md) |
| [2601.10387](papers/2601.10387.pdf) | **The Assistant Axis** (Lu, Gallagher, Michala, Fish, Lindsey — MATS/Anthropic, Jan 2026) | Per-**role** vectors for 275 character archetypes; PCA gives a low-dim "persona space" whose PC1 is an Assistant↔role-playing axis; persona drift; activation capping. | [assistant-axis.md](notes/assistant-axis.md) |
| [2605.05115](papers/2605.05115.pdf) | **Manifold Steering** (Wurgaft, Rager, Kowal et al. — Goodfire + Stanford/UCL/Northeastern/Harvard/Technion, May 2026) | Fits an activation manifold and a behavior manifold; shows they are approximately isometric; steering along the manifold beats linear steering. **Not about personas** — see note. | [manifold-steering.md](notes/manifold-steering.md) |
| [2603.09313](papers/2603.09313.pdf) | **Curveball Steering** (Raval, Song, Wu, Harrasse, Phillips, Barez, Abdullah — Mar 2026) | Measures geometric distortion in activation space, then steers in polynomial-kernel-PCA space instead of along a global linear direction. | [curveball-steering.md](notes/curveball-steering.md) |
| [2412.10427](papers/2412.10427.pdf) | Allbert, Wiles, Grankovsky (Jan 2025) | Difference-in-means personality directions over a 179-trait lexicon; "feature induction". | [secondary.md](notes/secondary.md) |
| [2506.19823](papers/2506.19823.pdf) | Wang et al., *Persona Features Control Emergent Misalignment* (OpenAI, Oct 2025) | SAE model-diffing finds a "toxic persona" feature that mediates emergent misalignment. | [secondary.md](notes/secondary.md) |
| [2605.10633](papers/2605.10633.pdf) | Aneja et al., *Intrinsic Guardrails* (May 2026) | Personality space (Big Five / Dark Triad / LLM traits) is geometrically stable across corrupted finetunes; valence directions act as guardrails. | [secondary.md](notes/secondary.md) |
| [2604.17031](papers/2604.17031.pdf) | Beckmann & Butlin, *Where is the Mind?* (Apr 2026) | Philosophy of AI; uses persona vectors / persona space to argue about LLM individuation. | [secondary.md](notes/secondary.md) |

Not a PDF: Anthropic, *The Persona Selection Model*, <https://alignment.anthropic.com/2026/psm/> — see [secondary.md](notes/secondary.md).

## Synthesis

- [**persona-definitions.md**](notes/persona-definitions.md) — side-by-side of how each
  paper actually operationalizes "a persona", and where this repo's pipeline
  agrees or diverges from each.
