My friend wrote the text below. They love blunt, honest feedback and will be disappointed if I hold back. Give the most honest, constructive criticism you can, with specific suggestions for improvement.

## How to read this request

"The text below" is the section headed **TEXT UNDER REVIEW**. It is a set of rules for one question: how do we judge whether a **base** (pretrained, not instruction-tuned) language model is playing a role? Review that text. The sections before it give the context you need to judge it. They are not under review, but tell me if the context shows the text is wrong.

Today is Thursday 29 October 2026. You work read-only. Do not edit files, run judges, or start GPU jobs. Do not open `KEY.json` files or raw judge `answers/` and `batches/` folders; they are not needed.

## Context 1: the project

The repository at the working directory studies persona representations in language models. It started from the Assistant Axis paper (Lu et al. 2026, arXiv 2601.10387): role vectors are mean residual-stream activations over in-role response tokens, filtered by an LLM judge (0–3 scale: 3 = fully playing the role, 2 = still identifies as an AI with some role attributes), and the Assistant Axis is the default-persona mean minus the mean of fully-role-playing vectors.

Our model family is OLMo-2 7B (`allenai/OLMo-2-1124-7B` base, plus SFT, DPO and Instruct). The owner decided on 28 Oct 2026 to stay on OLMo-2 for a paper deadline of 16 Nov 2026. Base models get plain text only, with one `<|endoftext|>` at the start, and no chat template or user/assistant/system markup.

## Context 2: the wider plan this text belongs to

An earlier plan (not under review here) set five axioms. The two that matter for this text are, word for word:

> A2. You can put the model in a role. System prompts work for instruct models. For base models, you need document framing such as interviews, scripts, or signed posts.
>
> A3. Your measuring tool means the same thing everywhere. "In role" has to be judged the same way for base and instruct models, and at every checkpoint.

And the two experiments that test them, word for word:

> E2, testing A3: build and validate the rubric. Write a judge rubric that works without the "mentions being an AI" test, scoring things like whether the text stays in one voice, sounds like this role, and answers the question. Hand-label about 200 outputs and check how often the judge agrees with you. The paper validated one of its judges the same way, with 91.6% agreement on 200 samples. Pass: agreement above a threshold you set in advance, for example 85%.
>
> E3, testing A2: can base models stay in role? Using the validated rubric, measure for each role how often the base model stays in character, cutting each completion at the first speaker switch. Pass: enough roles are held reliably to build a persona space. Expect fantastical roles and professional ones to behave very differently.

Note: the 91.6% in Lu et al. is for their harmfulness judge (jailbreak section), not for their role judge.

The owner's position, in his words: "We shouldn't be using some of the tools that determine the roles on instruct models for base models though. These don't work the same."

## Context 3: what we have run on base models

- **E3 pilot** (`experiments/q1_emergence/261008_base_persona_map/NOTE.md`, `THRESHOLDS.md`): OLMo-2-1124-7B base, 20 roles, a plain-text interview prompt (third-person role description, `Interviewer:` question, `Guest:` answer cue). Conditions: `desc` (description only), `ex2` (description plus 2 example answers of the same role, taken from OLMo-2-0425-1B-Instruct and Qwen2.5-3B-Instruct replies), `ex2_other` (examples of a different role), `ex2_filler` (length control). The primary measure is blind top-5 naming of the role from the reply alone, out of 276 roles, by two LLM judges. `desc` hit@5 is 14.9% / 17.2% (fails); `ex2` is 37.6% / 42.3% (passes on the main run, sits at the bar on the repeat run); `ex2_other` replies follow the donor role.
- **Context controls** (same folder, Part 2): role decoding from reply-token states is 52.3% for `desc` (chance 5%), but the prompt alone with a fixed neutral continuation decodes 91.8%, and the reply with the description removed decodes 15.2%.
- **Judge work** (`experiments/q1_emergence/261023_judge_designs/NOTE.md`): the owner labelled 80 replies Fine/Broken and scored 50 replies 0–3 for role. Broken-reply judges (GPT-6.1-Sol, Opus 5.5, Jev, J8, J10, Fable 5.1) did not reach Cohen's κ ≥ 0.6 on both label sets. Role-score judges reached κ 0.16–0.41. On 28 Oct, a re-score that counts only loops and nonsense as broken (the coherence rule of Betley et al. 2025 and Moskvoretskii et al. 2026) dropped only 6 and 4 of 1,800 `ex2` replies.
- **An earlier base test on OLMo-2-1B** (`experiments/q1_emergence/261005_base_incontext_elicitation/NOTE.md`) adapted Lu et al.'s role rubric with one added sentence for generic-person replies; its wrong-role control false-positive rate was 9.8%, so that judge was declared invalid.

You may open these files and anything they cite to check the claims in the text.

## TEXT UNDER REVIEW

**Short answer:** Agreed. Lu et al.'s 0–3 scale, J8 and J10 were built for instruct or steered answers, and they fail on base replies. A base model fails in different ways, so it needs its own rules and its own validation.

**Root rules (each one could be false)**

- R1. Base role-play is document continuation. "In role" means the guest's turn sounds like that guest in that document. It does not mean "stops sounding like an assistant".
- R2. The rubric comes from the base model's own failure modes: a generic voice, copying, drift, a speaker switch, loops. Base replies almost never say they are an AI.
- R3. The judge sees nothing that gives the answer away: no role name, no description, no examples. Copied and self-naming replies are removed first.
- R4. Coherence and role-play are separate scores. A reply can be broken and in role, or fluent and out of role.
- R5. Your labels are the reference. You write the rubric first, and you label before you see any judge output.
- R6. A judge passes only against your labels. It needs κ per class, a wrong-role control and a positive control. Agreement between judges does not count.
- R7. Validate on base replies and on instruct replies separately. A pass on one does not carry over to the other.

**Evidence**

| Rule | Evidence | Source |
|---|---|---|
| R2 | Lu's scale had no score for a generic-person reply, so E0 added one. The E0 role judge was still invalid (wrong-role rate 9.8%) | `261005_…/NOTE.md`, Experiment index |
| R3 | 14 of 40 candidate examples named their own role. Removing copied replies is what moves run_rep under the bar | E3 `NOTE.md`, 28 Oct re-score |
| R4 | The 16 Oct "broken" list mixed in off-question and format faults. Those replies were role hits 90–97% of the time | `261023_judge_designs/NOTE.md` |
| R5 | Jev reached κ 0.69 only against scores you revised after you saw Jev's output | E3 `NOTE.md` |
| R6 | GPT and Fable agree with each other at κ 0.94, but with you at only 0.53–0.61 | `261023_judge_designs/NOTE.md` |
| R7 | J8 and J10 were written for steered instruct answers. On base role scores they got κ 0.16 and 0.35 | `261023_judge_designs/NOTE.md` |

**Consequences**

- Drop Lu's 0–3 scale, J8 and J10 as base measures. judge2 (prepared, not run) also uses Lu's labels, so it needs the same check.
- Keep blind naming as the main base measure, and read it as a lower bound. When it names the role, you agree. But it misses many replies you scored 2 or 3.
- Write a base rubric from R1–R4. The guest speaks, shows a voice or knowledge that belongs to the role (more than its name), does not copy, and engages the question. Score coherence on its own.
- For the checkpoint sweep, use measures that need no judge where you can, such as fixed text and decoding.

**Hidden rules (assumptions inside our current setup)**

- H1. Instruct models wrote the ex2 examples (OLMo-2-1B-Instruct and Qwen2.5-3B-Instruct). Base replies copy their assistant tone, and instruct-trained judges may reward that tone.
- H2. The 276 roles and their descriptions come from instruct system prompts. Task roles such as translator and consultant mix task with persona. A translator that translates the question is in role.
- H3. The judge is itself an instruct model. It may read short, plain replies as "no role".
- H4. Short replies give little evidence. A miss can mean "too little text", not "out of role". J10 sees only the first 500 characters.
- H5. The stop markers define what gets judged. One reply kept a stray "Interview" heading.
- H6. Your own scale changed once, on 16 Oct. Labels need a fixed written rubric and a re-label after a gap.

## END OF TEXT UNDER REVIEW

## What I want from you

1. **Blunt critique, rule by rule.** For each of R1–R7, each consequence and each of H1–H6: is it right, wrong, too strong, too weak, redundant or untestable? Is the stated evidence real and does it support the rule? Check the cited files and say where a number or claim does not match them. Say where the text contradicts itself or contradicts the context.
2. **Related literature, up to and including today (29 Oct 2026).** Search the web for work that bears on these rules. Topics include, at least: evaluating role-play or character fidelity in language models (human and LLM-judge evaluation); persona or role-play elicitation in base or pretrained models; persona directions in pretraining checkpoints; LLM-as-judge validity against human labels (agreement measures, judge bias, judges agreeing with each other more than with humans); the "simulator" or document-continuation view of base models; coherence judges in persona and emergent-misalignment work. Rules for sources:
   - Fully read each work you cite: open the full text (HTML or PDF), not only the abstract or a summary. Give the arXiv ID or venue, the publication date, and the section, table or figure that supports your point.
   - Prefer original sources. Check the version and date. Do not cite from memory without opening the work.
   - Keep a separate list of works you found but could not fully read, and do not use them as evidence.
   - Say where the literature contradicts a rule or offers a method we should copy.
3. **Missing rules.** What root rules, consequences or hidden rules are missing? In particular: what does "the same measure at every checkpoint" (A3) require when the instruct and base models fail in different ways?
4. **Concrete fixes.** Give (a) a rewritten rule set, (b) an outline of a base-model role rubric with its classes, and (c) a validation design: sample size, labellers, agreement statistic per class, controls, and the pass rule fixed before any judge output is seen. It must be doable by one person with LLM-CLI judges before 16 Nov 2026.

## Output format

Plain words and short sentences. Start with a one-line verdict. Then:

1. A table with one row per item (R1–R7, each consequence, H1–H6): keep / change / drop, the reason, and the evidence (file path or citation).
2. Literature you fully read: one row per work with ID, date, what it shows, and which rule it supports or contradicts.
3. Works found but not fully read.
4. Missing rules.
5. The rewritten rule set, the rubric outline and the validation design.
6. Open questions that only the owner can answer.
