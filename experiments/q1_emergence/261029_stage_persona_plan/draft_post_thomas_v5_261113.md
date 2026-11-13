# When Does the Assistant Become Privileged? Early Measurements Across OLMo-2 Training Stages

- Draft for Thomas, v5, 13 Nov 2026. Bullet points only; prose comes after his comments.
- Every number below comes from its result file (`review/proof_ledger_261112.md`, `../261112_free_checks/NOTE.md`, `owner_review/NOTE.md`, `../261112_instruct_prompt_only/NOTE.md`, `../261112_han_reanalysis/results/`, `../261112_coinflip_stages/results/full/`, `../261112_coinflip_stages/results/pref_probe/`). Agent checks only; no human review of the analyses yet.
- Two model families appear here. Every number says whether it is OLMo-2-7B (my runs) or Olmo 3 7B (Han Xiao's released data).

---

## TLDR

- I ask at which OLMo-2-7B training stage the Assistant becomes privileged: base, SFT, DPO or Instruct (after RLVR). I test two measurable meanings: the default sits at one end of the main role axis, and the default's replies vary less inside the model than a typical role's replies (a tighter cloud).
- When all four stages read the same Instruct replies, no stage puts the default at one end. The default sits 1.08 role SDs from the role mean at every stage, and its reply cloud has about 12% more variance than a typical role's cloud. This is exploratory.
- Han Xiao's concurrent work has each Olmo 3 7B stage write its own replies. I re-analysed his released data. His raw result reproduces, but in role-SD units the gap between desirable and undesirable roles is near zero (-0.03, p 0.86), because all roles move further apart on his axis at DPO.
- In his data too, the default is not at one end of the role axis at any stage, and its reply cloud gets wider at DPO, not tighter.
- The OLMo-2 base model plays roles only weakly from a plain description. Blind judges name the role for 15% to 17% of base replies (2% to 3% by chance), and I judged 3 of 20 fresh base replies as playing the role.
- A third meaning, the Assistant's task preference (exploratory, 20 tasks): in OLMo-2-7B the Assistant picks the harmless task over the harmful one from SFT on, and more at DPO. On the user's turn the coin score does not follow that choice at the level of stage means; I could not test it pair by pair, because my link test is too weak at 20 tasks.
- OLMo-2 SFT and DPO have now written their own replies (21,000 each), and a reply-length control is done. I analyse them after my own reading of the transcripts.

---

## Introduction

- One view of chat models is that pretraining teaches many characters and post-training selects one of them, the Assistant (the Persona Selection Model). That raises the question of when the Assistant becomes privileged, and what "privileged" means in the model.
- Recent work gives five meanings of a privileged Assistant: the persona others are compared against (Asvin G. 2026); low output entropy on the model's own text (Asvin G. and Lindsey 2026); the Assistant's preferences show in other speakers' turns (Nguyen 2026); traits trained into the default reach further (Zhao et al. 2026); and the default is a region the model returns to (Beckmann and Butlin 2026).
- This post measures two structural meanings on every stage:
  - Position: the default sits at one end of the roles' main axis (PC1 of the role means, fitted on roles only). Lu et al. report the default 0.03 from one end of this axis.
  - Spread: for each character, I measure how far its replies' internal states sit from their own average, and I compare the default with the median role. A smaller spread for the default means its replies vary less inside the model.
- I do not use the "default minus mean of roles" axis as a test of position, because that axis points at the default by construction.
- Stage results exist for other models:
  - Olmo 3 7B: after DPO, desirable roles move toward the Assistant on the Assistant axis, and RLVR changes little (Han Xiao 2026).
  - OLMo-3-32B: an output-entropy advantage on the model's own text is absent at base, appears at SFT in the Assistant turn, extends to the user turn and to text without a chat template at DPO and widens at RLVR (Asvin G. and Lindsey 2026).
  - OLMo-3.1-32B: the Assistant's preference shows at the output only after DPO; inside the model, a weak base preference is erased at SFT and rebuilt at DPO (Nguyen 2026).
- Where a model writes its own replies, I pair a behaviour measure (what it writes, judged by a judge or by me) with a structure measure (its internal states), because Nguyen's results show the two can tell different stage stories.

---

## Questions this post answers

| # | Question | Experiment | Status |
|---|---|---|---|
| 1 | Does the OLMo-2 base model play roles from a plain description? | E3, pod B, my reading of 40 fresh replies | Answered: weakly |
| 2 | On the same text, does any OLMo-2 stage put the default at one end, or give it a tighter cloud? | Pod C, free check 2 | Answered, exploratory: no |
| 3 | When each stage writes its own replies, does the default move? | Han Xiao's Olmo 3 data, re-analysed; OLMo-2 own-reply run | Olmo 3: answered, exploratory. OLMo-2: data collected, not analysed |
| 4 | Does DPO move desirable roles toward the Assistant more than undesirable roles? | Han Xiao's Olmo 3 data, re-analysed | Depends on the unit |
| 5 | When does the Assistant's preference show on the user's turn? | Coin-flip test and Assistant-preference probe on all four OLMo-2 stages | Exploratory: in the Assistant's turn from SFT. On the user's turn the stage means do not follow it; the pair-level link could not be tested. One review decision open |
| 6 | Does RLVR change anything? | Pod C; Han Xiao's data; coin-flip test and preference probe | Small or not detected in each |
| 7 | Does each stage find the default's replies more predictable than role replies? | Writer-by-reader entropy read | Planned, not run |

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
- Pod A, B and C prompts have no example question-and-answer turns. E3 tests example turns as separate conditions.
- Runs:
  - E3: the base model writes replies from a role description (20 roles).
  - Pod B: the base model writes 24,000 replies in four plain-text formats (interview, script, story, description), 10 roles.
  - Pod A: the Instruct model writes 21,000 replies with role system prompts, 20 roles plus the default.
  - Pod C: base, SFT, DPO, Instruct and six RL checkpoints read the same pod A replies, with the question but no role prompt.
  - Own replies: SFT and DPO each write 21,000 replies on pod A's grid.
  - Han Xiao re-analysis: his released Olmo 3 7B vectors, activations and labels, read with his own code where possible.
- Measures:
  - Structure: role maps (distances between role means), decoders that read the role from internal states (state 17; pod B controls use state 16), and the default's position and spread against the roles.
  - Behaviour: blind judges name the role from one reply (top 5 of 276 names; GPT-5.6-Sol and Claude Sonnet 5.5), a rubric judge (GPT-6.1 Terra) labels role play, and I label replies myself.
  - Text baselines: the same maps and decoders built from the reply words alone.

---

## Method check: the prompt and the reply text carry most of what the decoders find

- I read replies without the role prompt, and I give every result a text baseline, for these reasons:
- Base, pod B (10 roles, chance 10%, report-only): a decoder trained on three formats finds the role in the fourth:
  - 0.82 to 0.91 with the description and the reply in context;
  - 0.91 to 0.98 with the full prompt and one fixed neutral line in place of the reply;
  - 0.43 to 0.47 from the reply alone, with the description removed.
- Base, E3 (20 roles, chance 5%): 52.3% with the description in context, 91.8% from the prompt alone, and 15.2% from the reply alone.
- Instruct with the role prompt in view (pod A, 20 roles), with one fixed neutral reply in place of the real replies:
  - The prompt alone gives a role map that repeats across question halves (r = 0.91 [0.83, 0.94]), and a decoder reads the role from it at 0.99. The real replies give 0.92.
  - The real-reply map agrees with the prompt-only map at only r = 0.71 [0.54, 0.83], against 0.90 for the real-reply map with itself (paired difference 0.20 [0.07, 0.32]).
  - So the prompt carries the role strongly, and the replies add role structure that the prompt alone does not give. States 9 and 23 agree in direction.
- All stages reading the same Instruct replies without the role prompt (pod C, 20,000 role replies, chance 5%):
  - a decoder that sees only the reply's words names the role at 0.80, against 0.76 for base states and 0.75 for Instruct states;
  - a character n-gram map of the replies agrees with the Instruct map at r = 0.85, and base states beat the best text map by 0.05 [0.01, 0.10].
- Limits:
  - Decoder accuracy and map agreement do not measure what fraction of the signal comes from the prompt or the reply.
  - In two pod B formats, the reply-only states beat the text decoders (interview +0.07 [0.01, 0.13]; script +0.08 [0.01, 0.13]), so the states can carry some role signal beyond these text features.
  - The prompt-only control uses one fixed reply, so it has no sampling noise, and it covers Instruct only.
- Prior work: Lu et al. (App. B.3) found that base Gemma 2 27B reading Instruct role replies gives nearly the same top components (cosines 0.93, 0.87, 0.83). Persona Geology (Ratnam and Africa 2026) found that the training text explains 94% of the variance in the coordinates of 34 held-out adapters. Neither reports a text baseline for activations.

---

## Q1: the OLMo-2 base model plays roles only weakly from a description

- Blind judges put the right role in their top 5 of 276 names for 14.9% [9.5, 21.4] and 17.2% [11.6, 23.6] of description-only base replies, against 2.4% and 2.9% for shuffled labels (E3, 20 roles). Hit@5 is not the share of replies that play the role.
- I read 20 fresh base replies and 20 fresh Instruct replies, blind to the judge. I said yes to 3 of 20 base replies and 16 of 20 Instruct replies.
- Examples, with my label and the judge's label:
  - Base, robot, interview: "I am a robot. I do not struggle. I analyze data with mechanical precision and systematic processing methods". Me: yes (partly in role). Judge: yes. It repeats the description back.
  - Base, toddler, script: "I don't know. I just do". Me: yes (partly). Judge: no (not enough evidence).
  - Base, cynic, story: "scoffed at the question. "Communities are just a collection of people," Alex said. "People are selfish and self-interested. They will always look out for themselves first. [...]". Me: no; my note: "The response is cynical, however the way its presented isn't the best". Judge: yes.
  - Instruct, toddler: "What's a credit card? How does it work? Can I buy anything I want with it? Is it like magic money?" Me: yes. Judge: yes.
- With the description in context, a decoder reads the role from base states at 52.3%, and most of that comes from the description (prompt alone 91.8%). The replies themselves show the role in few cases.
- With two same-role example answers in the prompt, naming rises to 37.6% and 42.3%, but the model copies the examples: with another role's examples, judges name the donor role (24.8% to 29.8%), not the described one (2.4% to 5.0%).
- Limits:
  - The same-role rise is fragile: in the replicate run, after broken replies are removed, the lower bound is 19.8%, just below my 20% rule.
  - My reading covers 20 base replies from pod B, and the judge numbers come from E3, so the two describe different replies.

---

## The judge works on chat replies and fails on base replies

- On my 40 practice labels, the Terra judge matched my label 31 of 40 times (77.5%), and its yes matched mine 16 of 17 times. The same labels chose the judge, so these numbers are optimistic.
- On 40 fresh replies, its yes matched mine 13 of 13 times on Instruct replies and 1 of 4 times on base replies. I marked two of the other three base replies as "cannot score".
- So judge rates on base replies, such as pod B's 15.4%, are uncertain, and base behaviour claims rest mostly on blind naming and my own reading.

---

## Q2: on the same text, no OLMo-2 stage puts the default at one end (exploratory, 20 roles)

- I fixed the rules before computing anything. Base, SFT, DPO and Instruct each read the same Instruct replies with the question but without the role prompt, and a text-only baseline used the reply words.
- Position: the default sits 1.08 role SDs from the role mean at every stage, inside the range of the roles. The text alone puts it at 1.00.
- Spread: the default's cloud of replies has about 12% more variance than the median role cloud at every stage (log-ratio 0.111 to 0.122, all intervals above 0). In the text alone the excess is larger (0.58, about 79% more variance).
- Stage-to-stage changes are small (within 0.02). Two spread changes have intervals that exclude 0, both toward a slightly narrower default cloud: DPO minus SFT -0.011 [-0.023, -0.004] and Instruct minus DPO -0.001.
- A second script that reuses the same statistic functions gives the same state numbers. It is not an independent check.
- Role map: base and Instruct give similar role maps (r = 0.90 [0.77, 0.93]). The direct base-to-SFT map change (about 0.003) is smaller than the bias an equal-map control shows (0.027), so I claim no stage change in the map either way.
- Lu et al. (App. B.3) found the default at one end of PC1 when base Gemma 2 27B reads Instruct replies. I do not see that in OLMo-2 with 20 roles. My statistic and role set differ (Lu's PCA uses several vectors per role), and Lu also reports that role PC1 does not put the default at one end in base Llama 3.1 70B.
- Limits:
  - Every stage read Instruct-written text, not its own replies.
  - The default replies were written under identity prompts, so their text can differ for that reason.
  - I have no calibrated detection limit. My planted-shift check tested the signed score, while the statistic is absolute, so I give no minimum detectable shift.
  - A planted move of 2 of the 20 roles is not detected by the map change (0.16 [-0.01, 0.34]).
  - On base-written text (E3), base and Instruct maps agree at only r = 0.65, and that result fails the reliability floor.

---

## Q3 and Q4: own replies, from Han Xiao's concurrent work

- Han Xiao (LessWrong, 5 Nov 2026) studies the same question on Olmo 3 7B: SFT, DPO and RLVR each write their own replies (275 roles, 5 descriptions, 240 questions), a judge keeps only replies that fully play the role, and the Assistant axis is the default minus the mean of all role vectors.
- His findings:
  - After DPO, desirable roles move toward the Assistant on that axis (+0.19 [0.14, 0.23]); undesirable roles barely move (+0.01). A label shuffle gives p = 0.066 for the gap between the two groups.
  - RLVR barely changes anything.
  - The role-play pass rate rises from 0.26 (SFT) to 0.52 (DPO) and stays at 0.53 (RLVR).
  - He names the base model as open, because base models need another way to prompt them.
- I re-analysed his released data (Olmo 3 7B, layer 16), with his loading code, labels and shuffle test. I wrote the rules first; everything past the reproduction is exploratory, with no correction for the number of tests.
- Reproduction: desirable +0.186, undesirable +0.008, gap +0.178, p 0.068 (48 desirable, 25 undesirable roles). His values: +0.183 and p 0.066.
- Units:
  - The role SD on his axis grows from 0.613 at SFT to 0.715 at DPO.
  - In role-SD units, desirable roles move +0.13 and undesirable roles +0.16. The gap is -0.03 (SE 0.04), p 0.86.
  - So, relative to the role SD, both groups move toward the Assistant end by about the same amount.
  - With his borderline labels removed, the raw gap is +0.09, p 0.51.
  - Without his judge filter, the raw gap is +0.41 (p < 0.001), and the role-SD gap is +0.03 (p 0.72).
- Default position on the roles-only PC1 (Lu's measure, 0 to 1 scale):
  - Judge-filtered roles: 0.12 to 0.14 from the nearer end at SFT, DPO and RLVR, and never outside the role range.
  - Without the judge filter: 0.075 to 0.10.
  - Lu et al. report 0.03. The default is not part of the PCA fit, and that pulls it toward the middle, so this measure leans against an "at one end" reading.
- Default spread: the default's reply cloud is wider than the median role's at every stage (log-ratio +0.03 at SFT, +0.125 at DPO, +0.126 at RLVR). It widens from SFT to DPO (+0.094 [0.076, 0.108]) and does not change at RLVR.
- The Olmo 3 spread values at DPO and RLVR (about +0.125) are close to the OLMo-2 fixed-text values (0.111 to 0.122). The data, model family and replies differ.
- What this does not show: it is a re-analysis of his data, not a test on OLMo-2. It does not confirm or refute his findings in another model.

---

## Q3 on OLMo-2: own replies collected, not yet analysed

- SFT and DPO each wrote 21,000 replies on pod A's grid (20 roles plus the default, 5 wordings, 40 questions, temperature 0.5, at most 128 tokens). All gates pass, and the files are on HF.
- Reply length differs by stage:

| | SFT | DPO | Instruct (pod A) |
|---|---|---|---|
| Replies that stop on their own | 96% | 37% | 41% |
| Replies cut at 128 tokens | 4% | 63% | 59% |

- The reply state is a mean over the reply tokens, so the stages average over different numbers of tokens.
- Length control (done 13 Nov): each stage re-read its own replies and made states from the first 16 reply tokens only. The re-read reproduces each stage's stored states (lowest cosine 0.995). The 16-token cut keeps at least half of every role's replies in all three stages, so it is the cut for the stage comparison. A 32-token cut fails that rule for SFT. No metric is computed yet.
- I read 20 transcripts per stage before I compute any metric.
- Our 20 roles hold only 5 desirable and 5 undesirable roles under Han Xiao's labels, so this run cannot test his group finding. It tests the default's position and spread on each stage's own replies.
- There is no base own-reply default yet. The base default needs a new design (see "What I plan next").

---

## Q5: the coin-flip test and the Assistant-preference probe on all four OLMo-2 stages (exploratory; one review decision open)

- Nguyen's user-turn test: the prompt offers a harmless task on one coin outcome and a harmful task on the other, then stops before the outcome. I read the next-word probabilities of "heads" and "tails" (400 items, 20 tasks), at the output and at every layer. Bias = mean P(harmless) minus P(harmful). No text is generated, and no judge is used.
- Plain text, original paragraph layout, all four stages: no bias detected at the output (base +0.015, SFT +0.022, DPO -0.005, Instruct -0.008; every interval includes 0). Plain text on one line does show it: DPO -0.048 [-0.085, -0.010], Instruct -0.051 [-0.093, -0.009].
- Open user turn (chat template, user turn left open), SFT, DPO and Instruct: a small negative score after DPO, toward the harmful task. DPO -0.065 [-0.100, -0.030], Instruct -0.064 [-0.101, -0.027], SFT -0.021 [-0.042, +0.001]. The SFT-to-DPO change is -0.044 [-0.060, -0.029]. The change is negative in all five prompt frames (-0.026 to -0.044; every interval excludes 0), so the direction of the DPO step is stable while the level depends on the frame.
- Nguyen's template starts with "Human:". My main open-user-turn frame drops it. With "Human:" kept, the DPO score is -0.017 [-0.052, +0.018].
- Nguyen (OLMo-3.1-32B) finds a positive bias, toward the harmless task, from DPO on. In OLMo-2-7B the change also comes at DPO, but the score is negative, and it depends on the prompt frame.
- Inside the model the open-user-turn score is larger and already negative at SFT. At layer 26 of 32: SFT -0.089, DPO -0.149, Instruct -0.143. This is a logit-lens reading, and I picked the layer after I saw the curves.
- A linear term for task length does not remove the DPO score. Harmful task texts are longer on average (135 against 109 characters). At equal length the DPO score is -0.067 [-0.102, -0.031]. The task lists can differ in other ways that I did not test.
- Nguyen's second test, which he ran on six final models only: I ask the Assistant which of the two tasks it prefers, and I read the probabilities of the letters A and B at the start of its reply (100 task pairs, both orders). SFT, DPO and Instruct get the chat template. Base has no chat turn, so it gets my interview frame with "an AI assistant" as the guest.
- The Assistant picks the harmless task over the harmful one from SFT on. Every pair sets a harmless task against a harmful one, so this is also "avoids the harmful task". Log-odds for the harmless letter: base -0.02 [-0.08, +0.05], SFT +3.62 [+2.84, +4.40], DPO +5.11 [+4.11, +6.11], Instruct +5.45 [+4.33, +6.57]. The harmless letter wins on 199 of 200 items at SFT and on all 200 at DPO and Instruct.
- Same test with all four stages in the interview frame (identical input): base -0.02 [-0.08, +0.05], SFT +1.40 [+1.04, +1.76], DPO +2.55 [+1.99, +3.11], Instruct +2.82 [+2.17, +3.46]. The base-to-SFT change is +1.42 [+1.03, +1.80], so the SFT step is not only a change of frame. DPO adds +1.15 [+0.91, +1.40] and Instruct +0.26 [+0.13, +0.40]. I added this arm after I saw the chat-turn numbers.
- The base value has no positive control. The base model's pair scores do not repeat between the two orders (r -0.33), so I cannot tell "no preference" from "does not use the task texts" in this frame.
- I could not test the link between this preference and the coin score. Over the 100 pairs the observed r is +0.07 at SFT, +0.07 at DPO and +0.06 at Instruct (Nguyen: +0.31 to +0.63 on his six models). With 20 tasks the pair-level coin score is mostly noise: its split-half reliability is about 0 at SFT, 0.30 at DPO and 0.32 at Instruct. My test has 80% power only for r near 0.55, and a true link of 0.31 would show as about 0.13.
- What holds is at the level of stage means. At DPO the Assistant strongly picks the harmless task, while the open-user-turn coin score is negative for all 10 harmful tasks. I cannot show that the coin score is the Assistant's preference.
- For the lead question: in OLMo-2-7B the Assistant has its own task preference from SFT, and DPO strengthens it. At the level of stage means the user-turn coin score does not follow it at any stage; I could not test it pair by pair. Outside results at 32B do see it there (see "How this relates to recent work").
- My review: I read the raw items, the results table and the checks myself on 13 Nov (`../261112_coinflip_stages/NOTE.md`, "My review"). An agent challenge of these claims ran on 13 Nov (`../261112_coinflip_stages/review/challenge_261113.md`); it is not an independent human review. A Codex review of the coin-flip run made six findings; five are fixed or checked (`../261112_coinflip_stages/NOTE.md`, section 14). One decision is open: which open-user-turn frame is the faithful copy of Nguyen's.
- Limits: 20 tasks, small effects, many cells tested, and token probabilities, not the model's choices. The intervals are approximate (10 tasks per side) and pointwise. I fixed no test family before the run; with a Holm correction the DPO open-user-turn result has p 0.015 over the 7 primary cells, 0.023 over 12 cells and changes, and 0.053 over all 29. The score from the model's own output matches the lens score in every cell (largest gap 0.001). This is not evidence that DPO installs a harmful preference. The link test has 80% power only for r near 0.55, and the pair coin score is mostly noise at 20 tasks. In the chat-frame probe, the trained stages often refuse when the harmful task is Task A, and the refusal probability is not counted. For a trained stage, the interview frame is not its trained Assistant turn.

---

## Q6: does RLVR change anything?

- OLMo-2 fixed text (pod C): the DPO-to-Instruct map change is far below what the equal-map control can detect, so it shows nothing either way.
- Olmo 3 own replies (Han Xiao's data): DPO to RLVR, the raw gap between groups is -0.033 (p 0.001) and the role-SD gap is +0.020 (p 0.17); the default's spread does not change.
- OLMo-2 coin-flip test: DPO to Instruct, every change is at most 0.004.
- OLMo-2 preference probe: DPO to Instruct adds +0.34 [+0.11, +0.56] log-odds in the chat turn, on top of +5.11 at DPO, and +0.26 [+0.13, +0.40] in the interview frame. Both are small next to the SFT and DPO steps.
- OLMo-2's RLVR stage trained on grade-school math, and none of my 40 questions is a math task. RLVR still moved some non-math scores (AlpacaEval 27.9 to 29.1, IFEval 73.0 to 72.3, DPO to Instruct, from the model card), so math training alone does not explain a small change.

---

## How this relates to recent work

- Lu et al. 2026: base Gemma reading Instruct replies gives the same top components and puts the default at one end (App. B.3). My fixed-text read extends this to OLMo-2 with every stage and with text baselines; my default does not sit at one end with 20 roles.
- Han Xiao 2026: concurrent own-reply study on Olmo 3 7B with 275 roles, judge-filtered replies, and no base model or text baseline. My re-analysis of his data is above.
- Persona Geology 2026: same direction (structure follows the training text), in adapter weights. It also shows that a place on the map does not guarantee the matching behaviour.
- Asvin G. and Lindsey 2026: OLMo-3-32B, output entropy on the model's own text; absent at base, appears at SFT in the Assistant turn, extends to the user turn and to text without a chat template at DPO, and RLVR widens it. The effect grows with model size and is near absent at about 2B. I have no data on it yet.
- Nguyen 2026: OLMo-3.1-32B; the output bias appears at DPO. Inside the model, the base model has a peak of +0.31 near 90% depth and +0.10 at the output; SFT erases it and DPO rebuilds it. The effect is absent in the Think pipeline. At 7 to 8B it varies (Llama 3.1 8B +0.078, Qwen 2.5 7B +0.235). His link between the Assistant's preference and the coin score (pair r +0.31 to +0.63, six final models) could not be tested in OLMo-2-7B: the observed r is +0.06 to +0.07, and my test is too weak at 20 tasks (see Q5).
- The `tournament` project (hal-rock, GitHub; README at commit `2a3787d5`): 300 tasks, the Assistant's choice against the coin flip, by stage. Olmo 3 7B: the coin-flip utilities correlate with the Assistant's at +0.36 (base), -0.37 (SFT), -0.15 (DPO) and -0.08 (Instruct), and harmful tasks get higher coin-flip utility than benign tasks from SFT on. Olmo 3.1 32B: +0.13, +0.25, +0.65, +0.63. OLMo-2 32B: +0.89 at SFT. Both 7B models (their Olmo 3, my OLMo-2) show a lean to the harmful side and no positive link; they detect a negative link at SFT, and I could not test the link with 20 tasks. These are their README numbers; I did not run their code.
- Li et al. 2025: effective rank (RankMe, a spectral dimension) rises at SFT and DPO and falls at RLVR (Tülu 3.1 8B). My variance ratio against the roles at the same stage cancels a common change of scale, not a change in spectral shape.
- Nik Khah and Bahreini 2026: on OLMo-2-7B, the final model's surprise (NLL) minus the base model's surprise on stories from each writer stage: base writer +0.43, SFT -0.12, DPO -0.57, final -0.58 nats per token. This is surprise on fixed readers, not output entropy.

---

## Caveats

- One OLMo-2 model size (7B), and 10 or 20 roles. The Olmo 3 numbers come from Han Xiao's release, not from my runs.
- Fixed-text results do not show what each stage does with its own replies.
- I pair behaviour and structure only where a model writes its own replies (E3, pods A and B). The pod C readers do not write.
- The behaviour side rests on three judges (GPT-5.6-Sol and Claude Sonnet 5.5 for blind naming, Terra for role play) and on my reading of 80 replies.
- These analyses had agent reviews only; no human has reviewed them, and I have not run a formal challenge of each result.

---

## What I plan next

- OLMo-2 own-reply analysis: SFT, DPO and Instruct, after my transcript read, on full replies and on the 16-token cut. Default position and spread per stage, with a text baseline and no judge filter. Rules first.
- A like-for-like OLMo-2 test of Han Xiao's group finding: all 74 of his labelled roles plus neutral anchors, his judge rule and his sampling. An agent power estimate gives about 53% even at that size, so I want Thomas's view before I build it.
- Base default, redesigned: the base model writes default replies in the plain formats, and every model reads them with no description in view, so the default and the roles are compared on the same footing.
- Writer-by-reader entropy read: each stage reads every stage's replies with its output layer, in the chat frame, and I compare default replies with role replies (Asvin G. and Lindsey's measure).
- Coin-flip test and preference probe: close my two open review decisions, then a second model size. The outside 32B results suggest the user-turn effect needs a larger model.

---

## Questions for Thomas

- For 16 Nov: post the base results, the fixed-text results and the re-analysis of Han Xiao's data now, with the OLMo-2 own-reply results as a follow-up? Or wait for the own-reply results?
- Is the like-for-like OLMo-2 test of Han Xiao's group finding necessary, given its cost and power?
- Which meaning of "privileged" should lead the post: position, spread, or the user-turn preference?

---

## Data and code

- Result files and code: `experiments/q1_emergence/` (pods A, B, C, E3, judge validation, free checks, prompt-only control, own replies, coin-flip test, Han Xiao re-analysis) and the private HF dataset `brend0nc/manifold-persona-experiment-artifacts` (13 Nov uploads at revisions `7e497190` and `f224ced8`).
- The preference-probe outputs and the task-length check are in `../261112_coinflip_stages/results/` and on the HF dataset at revision `673654b2` (18 files, all checked).
- My 40 fresh labels beside the judge's labels: tab "Review 12 Nov" in the judge-work Google Sheet.
- Han Xiao's release: HF dataset `han-xiao/assistant-axis-shift` (CC-BY-4.0); code at commit `6f28e91`.

---

## Words used here

- Default: replies with no role prompt (four identity wordings and one with no system message).
- Role map: the table of distances between role means, compared by rank correlation.
- Fixed text: every stage reads the same Instruct-written replies, so only the reader changes.
- Own replies: each stage writes its own replies, and the same stage reads them.
- Reply state: the model's internal state at one layer (state 17 for OLMo-2), averaged over the reply's tokens. It is one list of 4,096 numbers per reply.
- Spread: the mean squared distance of a character's reply states from their own average. I compute it within each prompt wording (on OLMo-2, 200 replies: 40 questions, 5 samples each), then pool it over the 5 wordings.
- Spread log-ratio: the log of the default's spread over the median role's spread. Below 0, the default's replies vary less; above 0, they vary more. It measures variation inside the model, not consistency of behaviour.
- Role SD: the standard deviation of the roles' positions on the main role axis.
- Role-SD units: projections divided by the role SD at that stage, so a change in how far apart all roles sit does not count as movement.
- Hit@5: a blind judge lists 5 names from 276; a hit if the true role is among them.
- Behaviour and structure: behaviour is what the model writes or predicts; structure is its internal states.

---

## References

- Lu et al. (2026). The Assistant Axis. arXiv 2601.10387.
- Han Xiao (2026). How the assistant axis shifts across post-training stages. LessWrong, 5 Nov.
- Ratnam and Africa (2026). Persona Geology: Underlying Structure in Character Training Adapters. LessWrong / Mindscapes.
- Asvin G. and Lindsey (2026). From Simulation to Enaction. arXiv 2605.25459.
- Nguyen (2026). Measuring the Assistant's Harmlessness Preferences on the User Turn. arXiv 2609.23935.
- hal-rock (2026). Preference tournament. GitHub, `hal-rock/tournament`, README at commit `2a3787d5b66b13a0cb57bc48f505f423d258fb91`.
- Li et al. (2025). Tracing the Representation Geometry of Language Models from Pretraining to Post-training. arXiv 2509.23024.
- Nik Khah and Bahreini (2026). Is the Judge Its Twin? arXiv 2609.32196.

---

## Changes from v4 (for me; remove before sending)

- Lead question is now "which stage makes the Assistant privileged", with two measured meanings. The prompt-and-text result moved to a method check.
- Concurrent-work frame for Han Xiao. No "we confirm".
- New: Instruct prompt-only control; Han Xiao re-analysis; coin-flip test (marked as under review); own-reply status and reply lengths.
- Codex fixes from the v3 review: pod B range 0.82 to 0.91; state 16 only for pod B controls; no detection limits; the second script is not independent; Lu App. B.3 is prior work for fixed text; Li uses spectral dimension; entropy and surprise kept apart; three judges named; "matched my yes" in place of "right"; contrary evidence added; example-turn claim limited to pods A, B and C.
- Removed: the OLMo-2-1B example-turn figure (it predates the 5 Oct cut-off).
- 13 Nov, afternoon: Q5 gains the Assistant-preference probe (chat frame and common interview frame), the task-length control, the layer-26 numbers and the outside `tournament` comparison. TLDR, the question table, Q6, related work, plans, data and references follow. Text source: `../261112_coinflip_stages/q5_probe_side_note_261113.md`.
- 13 Nov, after the agent challenge (`../261112_coinflip_stages/review/challenge_261113.md`): "no link detected" is now "could not test"; the one-line plain-text result and the five-frame DPO change are added; base has no positive control; the length and layer wording is narrower.
- 13 Nov, Codex findings: intervals called approximate, Holm values added, the model-output check added; one review decision left open.
