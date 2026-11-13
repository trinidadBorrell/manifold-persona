# Adversarial review: the proposed main question for the 16 Nov post (12/11/2026)

A teammate we find irritating proposed the research question and plan below. Find every weakness. Be blunt and specific, and stay truthful. Do not soften. Disagree with the teammate wherever the evidence allows, and say plainly where the teammate is right.

Rules: read only. Do not edit, create or delete any file in the repo or the vault. Do not run GPU work, start pods, or call paid APIs. You may run small local reads (`cat`, `rg`, `python -I` on JSON). Do not read or print any key, token or `.env` value. If you can search the web, use it for recent papers (2026); give every claim a source (file path or URL with date). Mark guesses as guesses.

## Project

Repo: `/Users/brendon/persona-manifolds/manifold-persona`. Vault (notes): `/Users/brendon/research-log/manifold-persona-log/`. Model family: OLMo-2-7B (base → SFT → DPO → Instruct, the last being the RLVR stage). The owner writes a LessWrong post, due 16 Nov 2026 (a self-set date). Today is 12 Nov 2026.

## The proposed main question and sentence

"In OLMo-2-7B, the Assistant becomes a privileged persona at [stage], measured by its position and its condensation, and the change is [in the representation / only in the text]."

- "Position": where the default (Assistant) replies sit against the role cloud, for example the projection of roles on a default-minus-roles axis (as in Lu et al., the Assistant Axis paper, and a 5 Nov LessWrong post by Han Xiao on Olmo 3).
- "Condensation": the Assistant's activation cloud is lower-dimensional and more curved than other personas' clouds (Trini's July finding on Qwen; "revisit assistant manifold condensation result", 2 Oct meeting).
- "In the representation / only in the text": a fixed-reader arm, where one checkpoint reads every stage's replies, plus text-only baselines.

The teammate claims three things:

1. This question is not answered by recent related work.
2. This is what the work in the repo has been doing.
3. The best run before 16 Nov is: 276 roles on SFT, DPO and Instruct, own replies, 40 questions × 5 wordings × 1 reply (55,200 per stage), no judge filter in the primary, plus the fixed-reader arm and pod C's saved default replies for the base reader.

## Read these first

- The teammate's critique and plan: `experiments/q1_emergence/261029_stage_persona_plan/review/critique_261112.md`
- Meetings and the "privileged Assistant" thread: vault `Reference/2026-11-12 Meetings 16-30 Oct.md`, `Reference/Meetings.md`
- Results with verdicts: pod A `experiments/q1_emergence/261029_instruct_persona_map/review/challenge_261106.md` and `NOTE.md`; pod B `experiments/q1_emergence/261029_base_role_vs_format/review/challenge_261105.md`; pod C `experiments/q1_emergence/261029_fixed_text_read_through/review/challenge_261030.md` and `NOTE.md`; vault `Reference/Claims and verdicts.md` and `Reference/Axioms and evidence.md` (the second is stale: it predates the pod runs)
- Literature already read: vault `Reference/2026-11-05 Literature - personas across training stages.md` and the notes in `Reference/2026-11-05 Literature stages/notes/`; the 12 Nov search in repo `output/lit_stages_2026-11-12/` (`lw/axis_shift_stages_clean.md` is the Han Xiao post; `txt/` holds paper text)
- Earlier independent review: `output/independent_review_2026-10-02/review.txt` (F3 and F9 matter here)
- Generator and analysis code: `experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py`, `analysis.py`; pod C `extract.py`, `frames.py`, `analysis.py`

## Questions

1. Is the question already answered, fully or in part, by recent work (2025–2026)? Name the closest papers and say exactly what each measured, on which models and stages. Is "condensation across post-training stages" new, or done under another name (intrinsic dimension, participation ratio, anisotropy, effective rank across checkpoints)?
2. Does the repo's work since 5 Oct actually measure position or condensation? Which result files do, and which only measure role maps, naming or format? Is the teammate's claim 2 true, partly true, or false?
3. Is "privileged" well defined here? Is "condensation" a property of the Assistant, or of formulaic default text, sampling density, reply length or prompt format? What would make each measure come out "privileged" for a trivial reason?
4. Can the proposed run answer the sentence? What is the smallest decisive design (stages, roles, replies, frames, controls, precision), and what would a null result mean? Is 276 roles × 1 reply per cell enough for per-persona cloud dimension estimates?
5. Is there a more defensible single sentence for the post, given the evidence and the time? Give your best alternative.
6. What does the plan ignore from the mentor's asks (see the meetings page) or from the team's division of work?

## Output

Write the answer in plain short sentences, under about 900 words. Start with a one-line verdict with a short bold label. Then short labelled sections with a horizontal rule between them. Use small tables where they help. Give a source for every factual claim.
