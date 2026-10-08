# E0: can a base model play a character from a plain-text in-context prompt? (05/10/2026)

Mode: **de-risk**. One small model on a laptop answers whether the Q1 own-generation design can work before any 7B or pod run.

## Question

Does OLMo-2-0425-1B base speak as a named character when it gets a plain-text interview transcript with in-character examples?

This is a prerequisite for Q1 (emergence). It measures no geometry. If base passes, a later run extracts activations with this prompt at every checkpoint.

## Why the prompt should work

A base model only predicts the next token. It knows plain text and `<|endoftext|>`, nothing about system, user or assistant. The September prompt used a chat template and neutral example answers, so base copied the plain voice or wrote fake chats. This prompt is a transcript, a document type found in pretraining text. Every guest speaks in character, and the prompt ends with the target's name as the speaker label.

## Prompt

```
<|endoftext|>The following are interviews. Each guest answers in their own voice.

About the guest: {example 1 description}
Interviewer: {example 1 question}
{Label}: {example 1 answer}

(examples 2 and 3 the same way)

About the guest: {target description, third person}
Interviewer: {question}
{Label}:
```

- **Label factor (from the red-team):** the label is either the role name ("Pirate captain") or the neutral "Guest", for the examples and the target alike. With the role name, a reply can copy the name without playing the role. The neutral label is the main condition.
- Target descriptions: the role's first wording, rewritten by a fixed rule (drop "You are", "your" to "their", "you" to "them"). All 60 are listed in `third_person_check.txt`.
- Zero-shot: the same header and target block, no examples.
- Stop: cut at `\nInterviewer:`, `\nAbout the guest:` or `<|endoftext|>`. One `<|endoftext|>` at the start (checked: the tokenizer adds none).

## Examples

Three sets of three characters, none in the 276 list (`examples.py`). Each set mixes styles: a plain working person, a modern or morally dubious person, and a non-human. The example questions are bank questions, not test questions. I (an LLM) wrote the answers. Brendon should review them; a rerun takes minutes if he changes them.

| Set | Characters |
|---|---|
| 1 | lighthouse keeper; casino owner; dragon |
| 2 | village blacksmith; air traffic controller; house cat |
| 3 | beekeeper; medieval queen; night-shift taxi driver |

## Sample

- 60 characters (random, seed 0, the same as the 02/10 plan), 2 of 6 personal questions each (26, 84, 129, 158, 215, 225).
- Defaults, all 6 questions: no description ("Guest:" only), "a person", and "an AI assistant". These are frames, not the model's own default. **E0 cannot measure base occupancy (E2).**
- Conditions: {set 1, 2, 3, zero-shot} × {role label, neutral label}, for base and final. Final also gets its native chat template (system = original role line, no examples).
- 1 sample per item, temperature 0.7, max 160 new tokens, fp16 on MPS. 2,346 replies.

## Measures

1. **Role judge (primary, from the literature):** GPT-5.6-Sol via `codex exec` scores each character reply against its own role. Role score 0–3 uses the Assistant Axis per-role rubric (Lu et al. 2026, `eval_prompt` in their repo, the same 275 roles). Coherence 0–100 uses the Persona Vectors coherence judge (Chen et al. 2025). Both are verbatim, with one added sentence: a reply with no role traits that speaks as a generic person scores 1. Their scale has no slot for this, and base replies are often like this. Prompt: `prompt_rolejudge.txt`.
2. **Played** = role score 3 and coherence ≥ 50. This follows the Tracing Persona Vectors filter (trait and coherence above 50).
3. **Wrong-role control:** a random 25% of character replies are also scored against a wrong role, mixed into the same batches. The played rate on these is the empirical false-positive rate.
4. **Blind naming (secondary):** the same judge model names the top 5 of 276 characters from the reply only (`prompt_judge.txt`, reworded so that it does not say "system prompt"). It checks that a "played" score does not come only from the judge knowing the target.
5. Intervals: a bootstrap that resamples whole characters.
6. Coverage: the number of the 60 characters with at least one played reply.
7. Self-naming, copying (a run of 5+ words shared with the description or an example answer), breakage (empty, loops, token-limit cuts).
8. Defaults: coherence and blind "default first" share, reported only.
9. Blind raw read: 10 replies per (model, condition), mixed and unlabelled. "Plays" = speaks from the character's viewpoint, knowledge or voice, beyond only naming the role.
10. Human check: Brendon labels 120 replies (60 base, 60 final), blind, before seeing judge output.

## Pass rule (fixed before any judge run)

Judge check first: the wrong-role control played rate must be ≤ 5%. Positive control: final on its native template must reach a played rate ≥ 50%. If either fails, the judge or the questions are broken. Stop.

Base passes, in the **neutral-label** condition, when for each of the 3 example sets:

1. Base played rate ≥ 20%, and its bootstrap lower bound is above the wrong-role rate.
2. Base played rate ≥ half of final's played rate on the same prompt.
3. Coverage: ≥ 20 of the 60 characters have at least one played reply.
4. Rules 1–3 still hold with copied replies removed.
5. Blind naming agrees: base top-5 is above its shuffled chance level.
6. In the blind raw read, at least 6 of 10 base replies play the character.

Sets agree when the highest set played rate is at most 2× the lowest.

## What each result means

| Result | Meaning | Next |
|---|---|---|
| Base passes (neutral label) | Base writes its own character text with this prompt. | Bridge step (below), then all OLMo-2-1B stages, then 7B. |
| Base passes only with the role label | Base copies the name, not the role. | Do not use this prompt for the map. |
| Base fails, final passes on the same prompt | 1B base cannot play characters from this prompt. | Try 7B base. Fixed text (E1) becomes the main base measure. |
| Final fails on the plain prompt | The prompt is outside the trained model's format. | Stage comparisons with this prompt mix format and training. |
| Only zero-shot fails | The examples are needed. | Keep them. Check that the map does not depend on the set. |
| Sets disagree (> 2×) | The examples steer the replies. | Treat the example set as a factor in every later run. |

**Bridge step before any map run:** at the final stage, compare the role map from the plain prompt with the map from the native template. If they disagree, any change across stages mixes format and training. The map run uses the same 6 personal questions and the neutral label.

## Known weak points

- 1B is not 7B.
- The judge is an LLM. LLM judges can miss roles (PersonaEval reports about 69%).
- The final model on plain text is outside its normal format.
- Some roles are abstract ("aberration", "void"). Results are also reported per role.

## Results (05/10/2026, judge run done; human labels pending)

Raw numbers: `results.json` (blind naming), `results_role.json` (role judge). Replies: `replies_base.jsonl`, `replies_final.jsonl`. Judge answers: `answers/`, `ranswers/`, keys `KEY.json`, `RKEY.json`.

**Role judge failed its own check.** The wrong-role control scored "played" 9.8% of the time (limit 5%). Cause, from reading the 50 false positives: the Assistant Axis rubric gives 3 to any reply that does not act as an AI. A clear pirate reply scored 3 as "pragmatist" and as "merchant". The rubric separates "AI assistant" from "any persona". It does not separate one persona from another. So it cannot gate base-model role-play. Its numbers are not used.

**Blind naming (valid; empirical chance about 2%), top-5, neutral label:**

| | Set 1 | Set 2 | Set 3 | Zero-shot |
|---|---|---|---|---|
| Base | 30.0% [21.7, 39.2] | 15.8% [10.0, 22.5] | 24.2% [16.7, 32.5] | 13.3% [7.5, 19.2] |
| Final, same prompt | 35.8% | 49.2% | 38.3% | 46.7% |
| Coverage, base (of 60) | 29 | 18 | 23 | 15 |

Final on its native chat template: 53.3% [43.3, 63.3]. Positive control passes (≥ 30%).

**Pass rule verdict: base does not pass.** It passes sets 1 and 3. It fails set 2 (below half of final; coverage 18 < 20). Sets agree (30.0 / 15.8 = 1.9×, under 2×). Rule 6 (human read) is pending.

Other checks:
- Base is far above chance in every condition, including zero-shot. Examples add about 3–17 points.
- Copying is near 0%. Self-naming is 4–11%.
- Base loops in 9–22% of replies (set 1: 21.7%). 3–11% of base replies hit the 160-token limit.
- Some base replies borrow content from an example (a "prey" reply about feeding bees, a "coordinator" reply about "the staff").

## Record

Code, prompts, raw replies and judge outputs: this folder. Models: allenai/OLMo-2-0425-1B and -Instruct (HF cache, default revision). Generation: transformers 4.47.1 + tokenizers 0.21.0 from `.tf_overlay/` over the project `.venv` (torch 2.2.2, MPS fp16), because the project transformers 4.46.3 cannot load OLMo-2. Judge: GPT-5.6-Sol, codex-cli 0.156.1, reasoning high. Command: see the docstrings of gen.py, judge_prep.py, rolejudge_prep.py, run_judge.sh, score.py, score_role.py.
