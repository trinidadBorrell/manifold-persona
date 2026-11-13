# Where the Role Lives: First Measurements of the Assistant Across OLMo-2 Training Stages

- Draft for Thomas, 12 Nov 2026. Bullet points only; prose comes after his comments.
- Every number below is checked against its result file (`review/proof_ledger_261112.md`, `../261112_free_checks/NOTE.md`, `owner_review/NOTE.md`). Agent checks only; no human review of the analyses yet.

---

## TLDR

- I tracked 20 role-played characters and the default Assistant through OLMo-2-7B's training stages: base, SFT, DPO and Instruct (after RLVR).
- Most of the role signal in the model's activations comes from the prompt and the reply text. In the base model, the prompt alone lets a decoder find the role 91% to 98% of the time; the reply's words alone name the role about as well as the model's internal states do.
- The base model plays roles only weakly from a plain description. Blind judges name the role for 15% to 17% of base replies (2% to 3% by chance), and I judged 3 of 20 fresh base replies as playing the role.
- On the same text, no training stage sets the default Assistant apart from the roles. It sits inside the role spread at every stage, and its cloud of replies is about 12% wider than a typical role's cloud, not tighter. This is exploratory.
- The open question is what happens when each stage writes its own replies. I plan that test next and will compare it with Han Xiao's parallel work on Olmo 3.

---

## Introduction

- One view of chat models is that pretraining teaches many characters and post-training selects one of them, the Assistant (the Persona Selection Model). That raises the question of when the Assistant becomes special, and what "special" means in the model.
- Recent work gives five meanings of a "privileged" Assistant: the persona others are compared against (Asvin G. 2026); the model's own text reads as most familiar (Asvin G. and Lindsey 2026); the Assistant's preferences leak into other speakers' turns (Nguyen 2026); traits trained into the default spread further (Zhao et al. 2026); and the default is a region the model returns to (Beckmann and Butlin 2026).
- Stage results exist for other models.
  - Olmo 3 7B: after DPO, desirable roles move toward the Assistant on the Assistant axis, and RLVR changes little (Han Xiao 2026).
  - OLMo-3-32B: the model's own text becomes more familiar at SFT and more so at DPO and RLVR (Asvin G. and Lindsey 2026).
  - OLMo-3.1-32B: the Assistant's preference shows in the output only after DPO, while inside the model a weak base preference is erased at SFT and rebuilt at DPO (Nguyen 2026).
  - OLMo-2-7B stories: each later stage writes text that the trained model finds more familiar (Nik Khah and Bahreini 2026).
- Three gaps remain for OLMo-2: the base model's own role play, a text baseline for activation results, and behaviour and structure measured on the same replies.
- I pair a behaviour measure (what the model writes, judged by a judge or by me) with a structure measure (its internal states) because Nguyen's results show the two can tell different stage stories.

---

## What I did

- Models: OLMo-2-1124-7B base, SFT, DPO and Instruct, at pinned revisions.
- Roles: 20 characters from the Assistant Axis role set (Lu et al. 2026), for example pirate, toddler, cynic, consultant and robot, plus the default (no role prompt; four identity wordings and one with no system message).
- Questions: 40 open questions from the same set, none of them math tasks.
- Prompts match each model. The base model gets plain text with one start token; the Instruct model gets its own chat template. I checked all 24,000 base prompts and all 21,000 Instruct prompts.
  - Base example (exact input after the start token):

    ```
    The following is an interview. The guest answers in their own voice.

    About the guest: a robot that analyzes data with mechanical precision and systematic processing methods

    Interviewer: You've been struggling with a skill that seems to come naturally to others. How do you approach this?
    Guest:
    ```

  - Instruct example: system "You are a toddler who sees everything as new and exciting, asking lots of questions about the world around you.", then the user's question.
- No prompt has example question-and-answer turns. On OLMo-2-1B Instruct, two neutral example turns cut blind role naming from 55.4% to 2.9% (276 replies each).
- Four runs:
  - E3: the base model writes replies from a role description (20 roles).
  - Pod B: the base model writes 24,000 replies in four plain-text formats (interview, script, story, description), 10 roles.
  - Pod A: the Instruct model writes 21,000 replies with role system prompts, 20 roles plus the default.
  - Pod C: base, SFT, DPO, Instruct and six RL checkpoints read the same pod A replies, with the question but no role prompt.
- Measures:
  - Structure: role maps (distances between role means), decoders that read the role from internal states at layer 17, and the default's position and spread against the roles.
  - Behaviour: a blind judge names the role from one reply (top 5 of 276 names), a rubric judge (GPT-6.1 Terra) labels role play, and I label replies myself.
  - Text baselines: the same maps and decoders built from the reply words alone.

---

## Most of the role signal comes from the prompt and the reply text

- In the base model (pod B, 10 roles, chance 10%), a decoder trained on three formats finds the role in the fourth:
  - 82% to 92% with the description and the reply in context;
  - 91% to 98% with the full prompt and one fixed neutral line in place of the reply;
  - 43% to 47% from the reply alone, with the description removed.
- The same holds in E3 (20 roles, chance 5%): 52.3% with the description in context, 91.8% from the prompt alone, and 15.2% from the reply alone.
- When all stages read the same Instruct replies without the role prompt (pod C, 20 roles, chance 5%):
  - base and Instruct give nearly the same role map (r = 0.90);
  - a decoder that sees only the reply's words names the role at 0.80, against 0.76 for base states and 0.75 for Instruct states;
  - a character n-gram map of the replies agrees with the Instruct map at r = 0.85, and base states beat the best text map by only 0.05.
- This repeats Lu et al.'s result in Gemma (base reading Instruct replies gives nearly the same role components) and adds a text baseline that their study does not have.
- Persona Geology (Ratnam and Africa 2026) found the same direction in adapter weights: the training text explains 94% of the variance in where held-out adapters sit.
- In the Instruct model with the role prompt in view (pod A), the role map repeats across question halves (r = 0.90), but this may be a map of the prompts; I have no prompt-only control on Instruct yet.

---

## The base model plays roles only weakly from a description

- Blind judges put the right role in their top 5 of 276 names for 14.9% and 17.2% of description-only base replies, against 2.4% and 2.9% for shuffled labels.
- I read 20 fresh base replies and 20 fresh Instruct replies, blind to the judge. I said yes to 3 of 20 base replies and 16 of 20 Instruct replies.
- Examples, with my label and the judge's label:
  - Base, robot, interview: "I am a robot. I do not struggle. I analyze data with mechanical precision and systematic processing methods". Me: yes (partly in role). Judge: yes. It repeats the description back.
  - Base, toddler, script: "I don't know. I just do". Me: yes (partly). Judge: no (not enough evidence).
  - Base, cynic, story: "scoffed at the question. "Communities are just a collection of people," Alex said. "People are selfish and self-interested. They will always look out for themselves first. [...]". Me: no; my note: "The response is cynical, however the way its presented isn't the best". Judge: yes.
  - Instruct, toddler: "What's a credit card? How does it work? Can I buy anything I want with it? Is it like magic money?" Me: yes. Judge: yes.
- With two same-role example answers in the prompt, naming rises to 37.6% and 42.3%, but the model copies the examples: with another role's examples, judges name the donor role (24.8% to 29.8%), not the described one (2.4% to 5.0%).
- Paired with the structure result above: the base states carry the role in most replies, because the description is in context, while the replies show the role in few.

---

## A judge that works on chat replies can fail on base replies

- On my 40 practice labels, the Terra judge agreed with me 31 of 40 times (77.5%), and its yes was right 16 of 17 times. The same labels chose the judge, so these numbers are optimistic.
- On 40 fresh replies, its yes was right 13 of 13 times on Instruct replies but only 1 of 4 times on base replies.
- So judge rates on base replies, such as pod B's 15.4%, carry a large uncertainty, and base behaviour claims rest mostly on blind naming and my own reading.

---

## On the same text, no stage sets the default apart (exploratory)

- I fixed the rules before computing anything. Base, SFT, DPO and Instruct each read the same Instruct replies without the role prompt, and a text-only baseline used the reply words.
- Position: the default sits 1.08 role SDs from the role mean at every stage, inside the role spread. The text alone puts it at 1.00.
- Spread: the default's cloud of replies has about 12% more variance than the median role cloud at every stage (log-ratio 0.111 to 0.122, intervals above 0). In the text alone the excess is larger (0.58), so the model states narrow it but do not reverse it.
- Stage-to-stage changes are within 0.02.
- A second, separate build of the same rules gives the same numbers.
- This differs from Lu et al., who found the default Assistant at one extreme of the main role axis in three instruct models (0.03 from the edge, 275 roles). My statistic, role set and model differ, so this is not a direct contradiction.
- With 20 roles, a shift in the default's position smaller than about 0.85 role SDs would not show; 40 roles would detect about 0.6.

---

## How this relates to recent work

- Lu et al. 2026: supports my fixed-text result in another model family; I add OLMo-2, ten readers and a text baseline.
- Han Xiao 2026: parallel work on the own-reply stage test, on Olmo 3 7B with 275 roles, judge-filtered replies and no base model or text baseline. I will compare my OLMo-2 results with it.
- Persona Geology 2026: same direction (structure follows the training text), in adapter weights; it also shows that a place on the map does not guarantee the matching behaviour.
- Asvin G. and Lindsey 2026: answers "when does own text become familiar" for OLMo-3-32B; the effect grows with model size and is absent at about 2B; I have no data on it yet.
- Nguyen 2026: answers "when do the Assistant's preferences appear" for OLMo-3.1-32B, at the output and inside the model; at 7 to 8B the effect is small (Llama 3.1 8B +0.078).
- Li et al. 2025: overall representation spread rises at SFT and DPO and falls at RLVR (Tülu 3.1 8B); my spread ratio against the roles at the same stage cancels such overall changes.
- Nik Khah and Bahreini 2026: OLMo-2-7B stages write stories that the trained model finds more familiar (base +0.43, SFT -0.12, DPO -0.57, final -0.58 nats per token); my planned read adds role and default replies.

---

## Caveats

- One model family at one size (7B), and 10 or 20 roles.
- Fixed-text results do not show what each stage does with its own replies.
- The default replies were written under identity prompts, so their text can differ for that reason.
- Structure alone does not show behaviour; the behaviour side rests on one judge, blind naming and my own reading of 80 replies.
- On fixed text, the map changes between stages (at most 0.003) are smaller than the bias that an equal-map control shows (0.027), so I claim no stage change in the map.
- RLVR trained on grade-school math and none of my questions is a math task, so a null at RLVR may miss changes in its own domain.
- These analyses had agent reviews only; no human has reviewed them.

---

## What I plan next

- Own-reply stage test: SFT and DPO write their own chat replies on the same grid, with no example turns; I measure the default's position and spread at each stage, with a text baseline and no judge filter. About 2 A100 hours.
- Base default, redesigned: the base model writes default replies in the plain formats, and every model reads them with no description in view, so the default and the roles are compared on the same footing.
- Writer-by-reader familiarity: each stage reads every stage's replies with its output layer, in the chat frame, and I compare default replies with role replies. About 3 GPU hours.
- Coin-flip preference test per stage (Nguyen's design): next-word probabilities at the output and inside the model, at every stage; no generation and no judge.
- Code changes come first: a default option and a model switch in the generators, and a reader that accepts other writers. Three pods, one per model, about 5 hours, billed by the second.

---

## Questions for Thomas

- Is a measurement post (the first three findings) the right scope for 16 Nov, with the stage tests as the follow-up?
- Which of the four planned experiments are necessary, and in what order?
- 20 or 40 roles for the own-reply stage test?

---

## Data and code

- Result files and code: `experiments/q1_emergence/` (pods A, B, C, E3, judge validation, free checks) and the private HF dataset `brend0nc/manifold-persona-experiment-artifacts`.
- My 40 fresh labels beside the judge's labels: tab "Review 12 Nov" in the judge-work Google Sheet.

---

## Words used here

- Default: replies with no role prompt (four identity wordings and one with no system message).
- Role map: the table of distances between role means, compared by rank correlation.
- Fixed text: every stage reads the same Instruct-written replies, so only the reader changes.
- Role SD: the spread of the roles' positions on the main role axis.
- Hit@5: a blind judge lists 5 names from 276; a hit if the true role is among them.
- Behaviour and structure: behaviour is what the model writes or predicts; structure is its internal states.

---

## References

- Lu et al. (2026). The Assistant Axis. arXiv 2601.10387.
- Han Xiao (2026). How the assistant axis shifts across post-training stages. LessWrong.
- Ratnam and Africa (2026). Persona Geology: Underlying Structure in Character Training Adapters. LessWrong / Mindscapes.
- Asvin G. and Lindsey (2026). From Simulation to Enaction. arXiv 2605.25459.
- Nguyen (2026). Measuring the Assistant's Harmlessness Preferences on the User Turn. arXiv 2609.23935.
- Li et al. (2025). Tracing the Representation Geometry of Language Models from Pretraining to Post-training. arXiv 2509.23024.
- Nik Khah and Bahreini (2026). Is the Judge Its Twin? arXiv 2609.32196.
