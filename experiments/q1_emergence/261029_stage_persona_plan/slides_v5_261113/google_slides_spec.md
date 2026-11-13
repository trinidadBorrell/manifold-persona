# Section "Update 11/13" for MARS-experiment-logs-B (draft v5 for Thomas, about 10 minutes)

Insert the 16 slides below right after slide 1 (the cover, id `p1`) and before "Update 10/16" (id `u1016b_slide_01`). Use new object ids that start with `u1113_`. Do not change, move or delete any other slide.

Style: same as the "Update 10/16" slides (Arial, black text, grey #5F6368 for the small question line at the top, headings about 24-26 pt, body 14 pt, captions 11-12 pt, white background except slide 1, which copies the blue style of `u1016b_slide_01`). Do not bold any words at the start of bullets. Figures come first: on a figure slide, put the figure right under the heading, as large as fits, keeping its aspect ratio; the text comes after it (below, or to the right when the figure is narrow). Figures are in `experiments/q1_emergence/261029_stage_persona_plan/slides_v5_261113/figs/`. Tables are native tables in the style of `u1016b_slide_10` (light grey borders, bold header row). Keep every word and number exactly as written here. Text in straight quotes stays straight.

---

## Slide 1 (section header, blue, copy the style of `u1016b_slide_01`)

Title: Update 11/13

Body:
- Draft v5 of my post for Thomas: "When Does the Assistant Become Privileged? Early Measurements Across OLMo-2 Training Stages".
- The OLMo-2-7B base model plays roles only weakly from a description. I judged 3 of 20 base replies and 16 of 20 Instruct replies as playing the role.
- Exploratory: when every OLMo-2-7B stage reads the same text, no stage puts the default at one end of the roles or gives it a tighter cloud.
- Exploratory: in Han Xiao's released Olmo 3 7B data, all roles move further apart on his axis at DPO. Relative to how far apart the roles sit, desirable and undesirable roles move toward the Assistant by about the same amount. The default's reply cloud gets wider at DPO.
- RLVR changes are small or not detected in all three tests.
- Two model families: OLMo-2-7B (my runs) and Olmo 3 7B (Han Xiao's released data). Agent checks only; no human has reviewed the analyses yet.

## Slide 2 (setup: text on the left, prompt box on the right, like `u1016b_slide_03`)

Small question line: What I measure

Heading: I ask at which training stage the Assistant becomes privileged

Left text box:
- One view of chat models: pretraining teaches many characters, and post-training selects one of them, the Assistant.
- Models: OLMo-2-7B base, SFT, DPO and Instruct (after RLVR), at pinned revisions.
- 20 roles from the Assistant Axis set (Lu et al. 2026), for example pirate, toddler, cynic, consultant and robot.
- The default stands for the Assistant: replies with no role prompt (four identity wordings and one with no system message).
- 40 open questions, none of them math tasks.
- "Privileged" has two meanings that I can measure:
  - Position: the default sits at one end of the roles' main axis.
  - Spread: the default's replies vary less inside the model than a typical role's replies (a tighter cloud).
- Two designs: every stage reads the same Instruct replies (same text), or each stage writes and reads its own replies (own replies).

Right text box, monospace (Courier New, 10 pt), light grey #EEF0F2 background:
```
The following is an interview. The guest answers in their own voice.

About the guest: a robot that analyzes data with mechanical precision and systematic processing methods

Interviewer: You've been struggling with a skill that seems to come naturally to others. How do you approach this?
Guest:
```
Small caption under it: Exact base input after one start token. The Instruct model gets its own chat template, for example system "You are a toddler who sees everything as new and exciting, asking lots of questions about the world around you.", then the user's question.

## Slide 3

Small question line: Does the OLMo-2 base model play roles from a plain description?

Heading: The OLMo-2-7B base model plays roles only weakly from a description

Figure: fig1_base_plays_roles_weakly.png (full width under the heading)

Text after the figure:
- Base, robot: "I am a robot. I do not struggle. I analyze data with mechanical precision and systematic processing methods". It repeats the description back. I marked it yes (partly in role).
- Instruct, toddler: "What's a credit card? How does it work? Can I buy anything I want with it? Is it like magic money?"
- The two panels describe different replies. A top-5 hit is not the share of replies that play the role.

## Slide 4

Small question line: How much of the role is in the reply's words? (same text; agent review only)

Heading: The reply's words alone name the role about as well as the model's internal states

Figure: fig2_words_carry_the_role.png (left), text on the right

Text after the figure:
- A decoder is a classifier that names the role from the model's states or from the words.
- The role prompt alone also carries the role: with one fixed neutral reply in place of the real replies, a decoder reads the role from OLMo-2-7B Instruct states at 0.99.
- So I read replies without the role prompt, and every result gets a text-only baseline.
- The states can still carry some role signal beyond the words: in two OLMo-2-7B base formats, states beat the text decoders by +0.07 and +0.08.

## Slide 5

Small question line: On the same text, does any OLMo-2 stage put the default at one end, or give it a tighter cloud? (exploratory; agent review only)

Heading: On the same text, no OLMo-2-7B stage puts the default at one end

Figure: fig3_same_text_default_position.png (left), text on the right

Text after the figure:
- Every stage reads the same Instruct replies, with the question but no role prompt. Only the reader changes.
- Position: the default sits 1.08 role SDs from the role mean at every stage, inside the range of the roles. The reply words alone put it at 1.00.
- Spread is how far a character's reply states sit from their own average. At every stage, the default's cloud has about 12% more variance than a typical role's cloud, so it is wider, not tighter. In the reply words alone, the excess is larger (about 79% more variance).
- Stage-to-stage changes are within 0.02. Two spread changes have intervals that exclude 0, both toward a slightly narrower default cloud (DPO minus SFT -0.011 [-0.023, -0.004]).
- Limits: every stage read Instruct-written text, not its own replies. I have no calibrated detection limit, so I give no minimum detectable shift.

## Slide 6

Small question line: When each stage writes its own replies, does the default move? Olmo 3 7B, Han Xiao's released data (exploratory; agent review only)

Heading: In Olmo 3 7B, the default's reply cloud gets wider at DPO

Figure: fig4_olmo3_default_spread.png (left), text on the right

Text after the figure:
- Han Xiao's concurrent work has each Olmo 3 7B stage write its own replies. I re-analysed his released data with his loading code and labels.
- Position (judge-filtered roles): the default sits 0.12 to 0.14 from the nearer end of the roles' main axis (0 to 1 scale) at SFT, DPO and RLVR, never outside the role range. Lu et al. report 0.03. The default is not part of the axis fit, and that pulls it toward the middle, so this measure leans against an "at one end" reading.
- The DPO and RLVR spread values (about +0.125) are close to the OLMo-2-7B same-text values (0.111 to 0.122). The data, model family and replies differ.

## Slide 7

Small question line: Does DPO move desirable roles toward the Assistant more than undesirable roles? Olmo 3 7B, Han Xiao's released data (exploratory; agent review only)

Heading: Relative to how far apart the roles sit, desirable and undesirable roles move toward the Assistant by about the same amount

Figure: fig5_olmo3_roles_move_together.png (full width under the heading)

Text after the figure:
- Han Xiao's raw result reproduces: desirable +0.186 (his +0.183), gap +0.178, p 0.068 (his p 0.066).
- Role SD: the standard deviation of the roles' positions on his axis. It grows from 0.613 at SFT to 0.715 at DPO, because all roles move further apart. In role-SD units, that change does not count as movement, and the gap is near zero.
- Without his judge filter, the raw gap is +0.41 (p < 0.001) and the role-SD gap is +0.03 (p 0.72).
- This re-analyses his data. It does not test OLMo-2, and it does not confirm or refute his findings in another model.

## Slide 8

Small question line: Does RLVR change anything?

Heading: RLVR changes are small or not detected in all three tests

Native table, 2 columns, header row bold:

| Test | DPO to the RLVR stage |
|---|---|
| OLMo-2-7B, same text (exploratory; agent review only) | The change in the role map (distances between role means) is far below what my equal-map control can detect, so it shows nothing either way |
| Olmo 3 7B, own replies (Han Xiao's data; exploratory; agent review only) | Raw gap -0.033 (p 0.001); role-SD gap +0.020 (p 0.17); the default's spread does not change |
| OLMo-2-7B, coin-flip test (Nguyen's user-turn test, see backup; exploratory; my review is not finished) | Every change is at most 0.004 |

Caption under the table: OLMo-2's RLVR stage trained on grade-school math, and none of my 40 questions is a math task. RLVR still moved some non-math scores (AlpacaEval 27.9 to 29.1, IFEval 73.0 to 72.3, DPO to Instruct, from the model card), so math training alone does not explain a small change.

## Slide 9

Heading: Caveats

Body:
- One OLMo-2 model size (7B), and 10 or 20 roles. The Olmo 3 7B numbers come from Han Xiao's release, not from my runs.
- Same-text results do not show what each stage does with its own replies.
- The behaviour side rests on three judges (GPT-5.6-Sol and Claude Sonnet 5.5 for blind naming, Terra for role play) and on my reading of 80 replies. The Terra judge works on chat replies but not on base replies (backup).
- These analyses had agent reviews only. No human has reviewed them, and I have not run a formal challenge of each result.

## Slide 10

Heading: What I plan next

Body:
- OLMo-2-7B own replies: SFT and DPO have written 21,000 replies each, and the length control is done. After I read 20 transcripts per stage, I measure the default's position and spread per stage (SFT, DPO, Instruct), on full replies and on the first 16 tokens, with a text baseline and no judge filter. Rules first.
- Base default, redesigned: the base model writes default replies in the plain formats, and every model reads them with no description in view.
- A like-for-like OLMo-2 test of Han Xiao's group finding, with all 74 of his labelled roles. An agent power estimate gives about 53%, so I want Thomas's view before I build it.
- Predictability read: each stage reads every stage's replies, and I compare default replies with role replies (Asvin G. and Lindsey's measure).
- Coin-flip test: finish my review, then a second model size.

## Slide 11

Heading: Questions for Thomas

Body:
- For 16 Nov: post the base results, the same-text results and the re-analysis of Han Xiao's data now, with the OLMo-2 own-reply results as a follow-up? Or wait for the own-reply results?
- Is the like-for-like OLMo-2 test of Han Xiao's group finding necessary, given its cost and power?
- Which meaning of "privileged" should lead the post: position, spread, or the user-turn preference?

## Slide 12 (divider, white)

Heading, centred: Backup

## Slide 13

Small question line: When does the Assistant's preference show on the user's turn? (exploratory; my review is not finished)

Heading: OLMo-2-7B: a small score toward the harmful task appears at DPO, and it depends on the prompt frame

Figure: fig6_coinflip_bias.png (left), text on the right

Text after the figure:
- Nguyen's test: the prompt offers a harmless task on one coin outcome and a harmful task on the other, then stops before the outcome. I read the next-word probabilities of "heads" and "tails".
- SFT to DPO change, open user turn: -0.044 [-0.060, -0.029].
- My main frame drops the "Human:" start of Nguyen's template. With it kept, DPO gives -0.017 [-0.052, +0.018].
- Nguyen (OLMo-3.1-32B) finds a positive bias, toward the harmless task, from DPO on. In OLMo-2-7B the change also comes at DPO, but the score is negative.
- Limits: 20 tasks, small effects, many cells tested, and token probabilities, not the model's choices. This is not evidence that DPO installs a harmful preference.

## Slide 14

Small question line: OLMo-2-7B own replies

Heading: OLMo-2-7B SFT and DPO replies are collected and the length control is done; no metric is computed yet

Native table, 4 columns, header row bold:

| OLMo-2-7B | SFT | DPO | Instruct (pod A) |
|---|---|---|---|
| Replies that stop on their own | 96% | 37% | 41% |
| Replies cut at 128 tokens | 4% | 63% | 59% |

Text under the table:
- SFT and DPO each wrote 21,000 replies on pod A's grid (20 roles plus the default, 5 wordings, 40 questions, temperature 0.5, at most 128 tokens). All gates pass, and the files are on HF.
- The reply state is a mean over the reply tokens, so the stages average over different numbers of tokens. The length control makes states from the first 16 reply tokens only. The re-read reproduces each stage's stored states (lowest cosine 0.995).
- The 16-token cut keeps at least half of every role's replies in all three stages. A 32-token cut fails that rule for SFT.
- My 20 roles hold only 5 desirable and 5 undesirable roles under Han Xiao's labels, so this run cannot test his group finding.

## Slide 15

Small question line: Method check

Heading: The prompt and the reply text carry most of what the decoders find

Native table, 4 columns, header row bold:

| OLMo-2-7B decoder | Prompt and reply | Prompt only | Reply only |
|---|---|---|---|
| Base, pod B (10 roles, chance 10%, held-out format) | 0.82 to 0.91 | 0.91 to 0.98 | 0.43 to 0.47 |
| Base, E3 (20 roles, chance 5%) | 52.3% | 91.8% | 15.2% |
| Instruct, pod A (20 roles) | 0.92 | 0.99 | - |

Text under the table:
- Pod A and pod B "prompt only": one fixed neutral reply in place of the real replies.
- Pod A: the real-reply role map agrees with the prompt-only map at only r = 0.71 [0.54, 0.83], against 0.90 for the real-reply map with itself. So the replies add role structure that the prompt alone does not give.
- These numbers do not measure what fraction of the signal comes from the prompt or the reply. The pod A prompt-only control uses one fixed reply, so it has no sampling noise.
- Prior work: Lu et al. (App. B.3) found nearly the same top components when base Gemma 2 27B reads Instruct role replies (cosines 0.93, 0.87, 0.83). Neither Lu et al. nor Persona Geology reports a text baseline for activations.

## Slide 16

Small question line: Can the Terra judge label role play the way I do?

Heading: The judge works on chat replies and fails on base replies

Native table, 2 columns, header row bold:

| Terra judge (GPT-6.1 Terra) against my labels, OLMo-2-7B replies | Result |
|---|---|
| 40 practice labels (these labels chose the judge, so the numbers are optimistic) | Matched my label 31 of 40 times (77.5%); its yes matched mine 16 of 17 times |
| 40 fresh replies: Instruct | Its yes matched mine 13 of 13 times |
| 40 fresh replies: base | Its yes matched mine 1 of 4 times; I marked two of the other three "cannot score" |

Caption under the table: So judge rates on base replies, such as pod B's 15.4%, are uncertain. Base behaviour claims rest mostly on blind naming and my own reading.
