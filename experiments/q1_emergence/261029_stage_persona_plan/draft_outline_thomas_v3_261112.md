# Draft outline for Thomas, v3 (12 Nov 2026)

Status: bullet outline for Thomas's comments, before any prose. Every number is checked against its result file (`review/proof_ledger_261112.md`, `../261112_free_checks/NOTE.md`, `owner_review/NOTE.md`). Agent checks only; no human review yet.

## Narrative

- I study when the Assistant persona becomes special during OLMo-2-7B training (base, SFT, DPO, Instruct after RLVR).
- My results so far are about measurement. Most of the role signal in OLMo-2's activations comes from the prompt and the reply text, and the base model plays roles only weakly from a plain description.
- On the same text, no OLMo-2 stage sets the default persona apart from the roles. Whether the default moves when each stage writes its own replies is the open question, and I plan that test next.
- Throughout, I pair a behaviour measure (what the model writes or predicts) with a structure measure (its internal states) on the same replies, because recent work shows the two can disagree across stages.

---

## Claims

### Claim 1: the role signal in OLMo-2 activations mostly comes from the prompt and the reply text

- **Claim:** In OLMo-2-7B, a decoder reads the role from the prompt alone about as well as from prompt and reply together, and reply text alone gives about the same role map as the model's states. Strength: narrow claim, one model family, 10 or 20 roles.
- **Context:** Lu et al. (2026, App. B.3) found that base Gemma 2 27B reading Instruct role replies gives nearly the same role components (cosines 0.93, 0.87, 0.83). Persona Geology (Ratnam and Africa, 2026) found that trait-adapter weights follow their training text (94% of held-out variance). Neither reports a text baseline for activations.
- **Evidence (base, pod B, 10 roles, chance 10%):** a decoder trained on three text formats finds the role in the fourth at 0.82 to 0.92 with the description and reply in context. With the full prompt and one fixed neutral continuation, and no reply, it finds it at 0.91 to 0.98. With the reply alone and the description removed, 0.43 to 0.47.
- **Evidence (base, E3, 20 roles, chance 5%):** with the description in context 52.3% [41.4, 61.7]; prompt alone 91.8% [86.0, 95.9]; reply alone 15.2% [10.0, 21.0].
- **Evidence (all stages, pod C, 20 roles):** base, SFT, DPO and Instruct read the same 20,000 Instruct role replies with the question but no role prompt. Base and Instruct role maps agree at r = 0.90 [0.77, 0.93]. A decoder that sees only the reply's words names the role at 0.80, against 0.76 for base states and 0.75 for Instruct states (chance 0.05). A character n-gram map of the same replies agrees with the Instruct map at r = 0.85, and base states beat the best text map by 0.05 [0.01, 0.10].
- **Evidence (Instruct, pod A, 20 roles):** the role map repeats across question halves, r = 0.90 [0.80, 0.94], but the role prompt is in context, so this may be a map of the prompts.
- **Limits:** the base controls are report-only (no null test against the main result) and use hidden state 16, not 17. There is no prompt-only control on the Instruct model. On fixed text, the stage-to-stage changes in the map (at most 0.003) are smaller than the bias an equal-map control shows (0.027), so I claim no stage change in the map.

### Claim 2: the base model plays roles from a plain description only weakly

- **Claim:** OLMo-2-7B base, given a plain-text role description, writes replies that a blind judge can name above chance but rarely; when I read fresh base replies, I judged 3 of 20 as playing the role. Strength: hedged; one judge setup and one reader (me).
- **Context:** Han Xiao (2026) names the base model as open, because base models need a different way to prompt them. Lu et al. steered base models along the Instruct Assistant axis and got helpful human personas, but did not ask the base model to play roles.
- **Evidence (behaviour):** blind judges put the right role in their top 5 of 276 names for 14.9% [9.5, 21.4] and 17.2% [11.6, 23.6] of description-only base replies, against 2.4% and 2.9% for shuffled labels (E3, 20 roles). On 20 fresh pod B replies, I said yes to 3; on 20 fresh Instruct replies, to 16.
- **Evidence (structure, paired):** the base states carry the role mostly through the description in context (claim 1). So structure shows the role in most base replies, and behaviour shows it in few.
- **Limits:** the judge's yes matched mine on 13 of 13 fresh chat replies but on only 1 of 4 fresh base replies, so base judge rates (pod B's 15.4%) are uncertain. With two same-role example answers, naming rises to 37.6% and 42.3%, but the replies follow the examples: with another role's examples, judges name the donor role (24.8% to 29.8%), not the described one (2.4% to 5.0%).

### Claim 3 (exploratory): on fixed text, no OLMo-2 stage sets the default apart

- **Claim:** When base, SFT, DPO and Instruct read the same Instruct replies without the role prompt, the default persona's replies sit inside the spread of the 20 roles, and their cloud is wider than a typical role's cloud, at every stage. Strength: exploratory; rules fixed before the numbers, but these states were seen in earlier analyses.
- **Context:** Lu et al. found the default Assistant at one extreme of the main role axis in three instruct models (0.03 from the edge on a 0 to 1 scale, with 275 roles). The "privileged Assistant" has five definitions in recent work; this check tests two structural ones: position on a roles-only axis, and a tighter cloud.
- **Contrary evidence:** in OLMo-2 I do not see the default at an extreme, with the role prompt in view (pod A, 1.07 role SDs) or hidden (pod C, 1.08). My statistic, role set (20 roles) and model differ from Lu's, so this is not a direct contradiction; it needs the own-reply test with more roles.
- **Evidence:** the default sits 1.08 role SDs from the role mean at every stage; text features alone give 1.00. The default cloud has about 12% more variance than the median role cloud (log-ratio 0.111 to 0.122, intervals above 0); in text features alone the excess is larger (0.58). Stage-to-stage changes are within 0.02. A second, separate build of the same rules gives the same numbers.
- **Limits:** every stage read Instruct-written text, not its own replies. The default replies were written under identity prompts. With 20 roles, a shift in the default's position smaller than about 0.85 role SDs would not show; 40 roles would detect about 0.6.

---

## Questions the post answers

| # | Question | Status | My evidence | Related work and how it relates |
|---|---|---|---|---|
| 1 | Does the base model play roles from a plain description? | Answered: weakly | Claim 2 | Han Xiao leaves the base model open; this fills that gap for OLMo-2 |
| 2 | How much of the role signal in the states is the prompt and the text? | Answered: most of it | Claim 1 | Supports and extends Lu et al. App. B.3 with a text baseline; same direction as Persona Geology, in activations rather than weights |
| 3 | Do all stages read the same text with the same role map? | Answered: yes | Claim 1 (pod C) | Repeats Lu et al. B.3 in OLMo-2 across ten checkpoints |
| 4 | On fixed text, is the default set apart at any stage? | Answered, exploratory: no | Claim 3 | No direct prior test found |
| 5 | When each stage writes its own replies, does the default move relative to the roles? | Open | Planned experiment 1 | Han Xiao's parallel work tests this on Olmo 3 7B (desirable roles move toward the Assistant after DPO, +0.19; gap with undesirable roles p = 0.066; RLVR changes little). I compare my OLMo-2 results with his |
| 6 | When do the Assistant's preferences appear, at the output and inside the model? | Open | Planned experiment 4 | Nguyen (OLMo-3.1-32B): at the output the bias jumps at DPO; inside, a base mid-layer bias (+0.31) is erased at SFT and rebuilt at DPO (+0.60). Small at 7 to 8B (Llama 3.1 8B +0.078) |
| 7 | Does each stage find the default's own replies more familiar than role replies? | Open | Planned experiment 3 | The plain stage trend is published for OLMo-2-7B stories (Nik Khah and Bahreini: base +0.43, SFT -0.12, DPO -0.57, final -0.58 nats per token). Asvin G. and Lindsey (OLMo-3-32B): absent at base, appears at SFT inside the Assistant turn, spreads at DPO and RLVR, absent at about 2B |
| 8 | Does RLVR change persona representations? | Partly answered | On fixed text, no change at RLVR (pod C); RLVR trained on grade-school math (`allenai/RLVR-GSM`) and none of my 40 questions is a math task | Han Xiao: RLVR changes little; Nik Khah and Bahreini: final close to DPO; Asvin G. and Lindsey: RLVR widens own-text familiarity in Olmo 3 |
| 9 | Is the default's reply cloud tighter than role clouds? | Answered on fixed text: no (wider); open for own replies | Claim 3 | Li et al.: overall representation spread rises at SFT and DPO and falls at RLVR (Tülu 3.1 8B); my ratio against the roles at the same stage cancels such overall changes |

---

## Planned experiments (not run)

| # | Experiment | How it would run | Answers | Cost (estimate) |
|---|---|---|---|---|
| 1 | Own-reply stage test | SFT and DPO write their own chat replies on the pod A grid (roles plus default, no example turns); default position and spread at each stage; text baseline; no judge filter, with judge labels on a sample for the paired behaviour measure. Role count 20 or 40 (check 1) | Q5, Q8, Q9 | About 2 A100 hours plus checks |
| 2 | Base default, redesigned | The base model writes default replies in the plain-text formats; every model reads them with no description in view, so default and roles are compared on the same footing | Q4 for base's own text | Minutes of GPU; code change first |
| 3 | Writer-by-reader familiarity | Base, SFT, DPO and Instruct read every stage's replies with the output layer kept, in the chat frame; compare default replies with role replies | Q7 | About 3 GPU hours |
| 4 | Coin-flip preference test per stage | The user-turn coin-flip items of Nguyen, scored from next-word probabilities at the output and inside the model (logit lens), at every stage | Q6 | Minutes per stage; no generation, no judge |

Code changes before any run: a default option and a model switch in the pod B generator, a model switch in the pod A generator, a pod C reader that accepts other writers, and the output layer kept for experiments 3 and 4. Three pods, one per model, about 5 hours in total; billed by the second.

---

## Method notes

- Base models get plain text with one start token; I checked all 24,000 pod B prompts. The Instruct prompts equal the model's own chat template; I checked all 21,000 pod A prompts.
- No run uses example turns in the prompt. On OLMo-2-1B Instruct, two neutral example turns cut blind role naming from 55.4% to 2.9% (276 replies each).
- Judge (GPT-6.1 Terra): on my 40 practice labels it agreed 31 of 40 times (77.5%) and its yes was right 16 of 17 times; the same labels chose the judge, so this is optimistic. On 40 fresh replies its yes was right 13 of 13 times on chat replies and 1 of 4 on base replies.

---

## Limits of the whole post

- One model family at one size (7B) and 10 or 20 roles.
- Fixed-text results do not show what each stage does with its own replies.
- Structure results alone do not show behaviour; the behaviour side rests on one judge and my own reading.
- No human review of the analyses yet.

---

## Questions for Thomas

- Is a measurement post (claims 1 to 3) the right scope for 16 Nov, with the stage tests as the follow-up?
- Which of the four planned experiments are necessary, and in what order?
- 20 or 40 roles for the own-reply stage test?

---

## Words used here

| Word | Meaning |
|---|---|
| Default | Replies with no role prompt: four identity wordings and one with no system message. |
| Role map | The table of distances between the role means, compared by rank correlation. |
| Fixed text | Every stage reads the same Instruct-written replies, so only the reader changes. |
| Role SD | The spread of the roles' positions on the main role axis. |
| Hit@5 | A blind judge lists 5 names from 276; a hit if the true role is among them. |
| Behaviour and structure | Behaviour is what the model writes or predicts; structure is its internal states. |

## References

- Lu et al. (2026). The Assistant Axis. arXiv 2601.10387.
- Han Xiao (2026). How the assistant axis shifts across post-training stages. LessWrong, 5 Nov.
- Ratnam and Africa (2026). Persona Geology. LessWrong / Mindscapes, 30 Oct.
- Asvin G. and Lindsey (2026). From Simulation to Enaction. arXiv 2605.25459.
- Nguyen (2026). Measuring the Assistant's Harmlessness Preferences on the User Turn. arXiv 2609.23935.
- Li et al. (2025). Tracing the Representation Geometry of Language Models from Pretraining to Post-training. arXiv 2509.23024.
- Nik Khah and Bahreini (2026). Is the Judge Its Twin? arXiv 2609.32196.
