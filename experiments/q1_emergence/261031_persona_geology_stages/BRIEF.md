# Brief for Codex: Persona Geology across OLMo-2 training stages (31/10/2026)

Owner: Brendon. Written by Claude from Brendon's request on 31/10/2026:
"Give this chat to codex agent and have it run the experiment through training stages",
using a ChatGPT chat and the Persona Geology post to fill in what the chat missed.

## 1. The question

Ratnam & Africa, "Persona Geology: Underlying Structure in Character Training Adapters" (Oct 2026):
- Post: https://www.mindscapes.world/writing/persona-geology/ (also on LessWrong:
  https://www.lesswrong.com/posts/iErBZ5fomL4tyyHy6/persona-geology-underlying-structure-in-character-training)
- They trained 134 trait LoRAs on ONE model (Qwen3.5-4B) and found a 5-factor map
  (Warmth, Competence, Timidity, Arousal, Imagination).

The ChatGPT chat (https://chatgpt.com/share/6ae52010-c278-83e9-a59e-097223b27d93) proposed the extension:
"run the same trait-adapter experiments at each checkpoint, compare the resulting maps, and check behaviour.
That would test when the structure appears, which relationships survive, and which training stage reorganises them."
The post only compares two character-training stages (OCT stage 1 DPO vs stage 2 introspection SFT). It never
tests base -> SFT -> DPO -> RLVR. That gap is this experiment.

**Question:** Does the trait-adapter factor map exist at each OLMo-2 post-training stage (base, SFT, DPO,
Instruct/RLVR), and does it stay the same, beyond what the training data alone forces?

This is Q1 (emergence) in `research/PLAN.md`. Pod C (`../261029_fixed_text_read_through/`) found the role map
flat from base to RL, with a wording bias (0.027) larger than every stage difference. Read its NOTE.md first.

## 2. What the ChatGPT chat missed (from the full post)

Method facts you must use or knowingly change (record every change in NOTE.md):
- Training: adapted Open Character Training (OCT, https://arxiv.org/abs/2511.01689) with Persona Cartography
  changes (Baines et al., https://arxiv.org/abs/2607.07916). Every adapter: the same 445 prompts,
  13 AdamW DPO steps, a small SFT term, a KL penalty, LoRA rank 64, ONE shared random LoRA init. LR not given.
- Data: per trait, Claude Sonnet 4.6 writes a 120-200 word constitution; GLM-4.5-Air writes a chosen and a
  rejected answer. Traits: 100 Goldberg unipolar Big Five markers (20 per factor, both poles) + 34 Condon words.
- Released: https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35 (adapters, factor adapters,
  stage-two transcripts) and data pages under https://www.mindscapes.world/persona/ . FIRST check whether the
  chosen/rejected training pairs are released. If yes, reuse them: no generation cost, same data at every stage.
- Map: Frobenius inner products between LoRA updates (B·A products) -> 134x134 similarity -> principal axis
  factoring + oblimin; factor count by parallel analysis (a judgement call; they chose 5). Tucker congruence
  for matching, 0.85 line.
- Controls they used: scrambled zoo (chosen/rejected swapped on half of pairs -> no factors); permuted zoo
  (adapter trained on another trait's data -> factors match names only at chance); second seed (40 traits).
- **Key catch 1:** contrast-pair text embeddings predict 94% of held-out adapter positions (adapters: 98%).
  So the same data at every stage could give the same map BY CONSTRUCTION. Your stage comparison must
  report what each stage adds beyond the data-embedding prediction, not only map similarity.
- **Key catch 2:** raw weight directions are not stable: same trait, two seeds, cosine +0.018 (random level);
  stage 1 vs stage 2 weight cosine 0.000, but activation-shift cosine 0.53. Output (B) subspace overlap 46% vs
  14% unrelated; activation-weighted cosine 0.66 vs 0.09. Different base checkpoints have different weights,
  so compare maps by (a) similarity-matrix structure / factor congruence and (b) activation shifts on fixed
  inputs. Do not compare raw weight vectors across checkpoints as the main measure.
- Behaviour: Claude Sonnet 4.5 judge rates Big Five 1-7 on 24 scenarios (repeat reliability 0.79-0.90);
  steering along merged adapters; the map predicts behaviour inside its range only. Judge prompts not published.
- Emergent misalignment link (optional, later): 2,000 bad-medical-advice rows -> 13/79 misaligned answers
  vs 0/79 for good advice; the post calls the direction "carelessness" and says a contrastive control is needed.
- The post is heavily agent-written and blog level; its wiki lists withdrawn claims
  (https://www.mindscapes.world/persona/wiki/). Check claims you rely on there.

## 3. Project rules that apply (read them, do not skip)

- `AGENTS.md`, `.claude/CLAUDE.md` (Traps section), `.claude/skills/research-cycle/SKILL.md`.
- Base models: plain text, one `<|endoftext|>` at the start, no chat template. DPO on the base checkpoint
  needs a plain-text format; decide it in the design and state it. Do not pass `--fewshot` for trained stages.
- `hidden_states[0]` is the embedding output; map layer labels with `io.hidden_state_index`.
- Thomas (supervisor, meeting 30/10): do trait work "in one stage first, and then we can maybe extend".
  So the first run is ONE stage (OLMo-2 Instruct), then the sweep.
- Heavy work on a pod; pod outputs go straight to the private HF dataset
  `brend0nc/manifold-persona-experiment-artifacts`; record revisions; check SHA256SUMS.
- Vault logging per `AGENTS.md`; run the log linter after each vault edit.
- Never commit. Never edit `.claude/` or `CLAUDE.md`.

## 4. What to do, in order

1. **Design (research-cycle, de-risk mode).** Write `NOTE.md` and `THRESHOLDS.md` in this folder: question,
   expected result, baseline, controls (scrambled + permuted + second seed + data-embedding prediction),
   measures, budget, stop rules. Fix pass rules before any run. Checkpoints: reuse the OLMo-2-1124-7B
   base/SFT/DPO/Instruct pins in `../261029_fixed_text_read_through/checkpoints.json`. If you propose the 1B
   family for the de-risk step, say so plainly with the reason; do not switch silently.
2. **Data.** Find out if the post's pairs are released. Fetch them into the artifact path if yes.
3. **Code + local tests.** Training script (one adapter per call, fixed shared LoRA init, fixed seed),
   map code (Frobenius similarity, factoring, parallel analysis, congruence), activation-shift comparison,
   controls. Tests on CPU with a tiny model. Render and save exact training inputs for 5 traits per stage
   and read them before any GPU time.
4. **De-risk run (one stage, OLMo-2 Instruct):** a small zoo (for example the 20 Goldberg markers needed for
   both poles of each factor at a few per pole, plus the scrambled control). Pass rule: real zoo shows factor
   structure that the scrambled zoo does not.
5. **Stage sweep** (base, SFT, DPO, Instruct) only after step 4 passes and Brendon says go.

## 5. Stop gates: stop and report to Brendon, then wait

- Before ANY paid API call (data generation or judging). Give the cost estimate. Judge runs need his OK.
- Before any GPU use. You have no pod. Ask Brendon for a pod address. Do NOT use the pod at
  213.173.102.5; other sessions own it.
- After step 4, with the result and raw examples, before the sweep.
- If the released data is missing and must be generated.

## 6. Report format to Brendon

Plain, short sentences. One-line answer first with a bold label. Small chunks with bold headers.
Small tables for status. End with a **Next step** chunk when he must decide.
