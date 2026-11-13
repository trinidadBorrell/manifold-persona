# Native text from the 18 new Google Slides

## Slide 1

Update 11/13

What post-training does to the Assistant and to the roles a model can play, told from the field to my results.
My numbers are from OLMo-2-7B. Han Xiao's are from Olmo 3 7B. Other papers name their own model.
Agent checks only. No human has reviewed these analyses yet.

## Slide 2

1. The field

Persona research rests on a few foundations, and the newest work asks about training stages

A popular view of chat models, the Persona Selection Model, says that pretraining teaches a model many characters and post-training selects one of them, the Assistant.

## Slide 3

2. Why it matters for safety

Where the Assistant sits among the roles matters for safety

Lu et al. found that capping activations on the Assistant Axis stabilises behaviour, also against persona-based jailbreaks (larger instruct models).
Wang et al. found a 'toxic persona' feature that controls emergent misalignment (GPT-4o).

## Slide 4

3. What we don't know

No paper I read shows when the Assistant becomes privileged, with base models and a words-only control

Han Xiao and Nguyen study training stages, but not the Assistant's place in the role map with base models and a words-only control.

## Slide 5

4. My question

I ask at which OLMo-2-7B stage the Assistant becomes privileged

Privileged means the default sits at one end of the role map, or its replies vary less than a typical role's replies.
The answer would show where in training the Assistant gets selected, which the Persona Selection Model leaves open, and so where audits and steering should focus.

## Slide 6

5. The pivot

Three concurrent works covered parts of my plan, so I narrowed it

I narrowed my work to what they leave open: base models, text baselines across stages and OLMo-2's own stages. I also re-checked Han Xiao's result on his released data.

## Slide 7

6. What they say and what I found

Like Persona Geology's text result, the reply words carry most of the role (OLMo-2-7B)

With the role prompt out of view, the reply words name the role about as well as the model's states do (OLMo-2-7B: 0.80 against 0.76 and 0.75).

## Slide 8

6. What they say and what I found (exploratory; agent review only)

In Han Xiao's data, the roles spread apart at DPO as role play doubles

Han Xiao reports that DPO pulls desirable roles toward the Assistant (Olmo 3 7B).
In role-SD units the gap is -0.03 (p 0.86). The roles also spread apart at DPO, and role play doubles. I have not tested the cause.

## Slide 9

7. Results: the base model (my reading: n = 20, one reader)

The OLMo-2-7B base model rarely plays a role from a plain description

I judged 3 of 20 base replies in role, against 16 of 20 for Instruct.
What the base model's states know about the role comes mostly from the prompt.

## Slide 10

7. Results: the same text read by every stage (exploratory; agent review only)

Reading the same Instruct replies, every stage puts the default near the Assistant-like end

All four stages draw nearly the same role map and put the default near the Assistant-like end, as Lu et al. found (OLMo-2-7B, 20 roles).
The words alone put it at 0.085, about twice as far from the end.
On spread, the default is not tighter: it has about 12% more variance than the median role at every stage.
The main role axis also tracks reply length (r about 0.7), and I have not controlled for it.

## Slide 11

7. Results: behaviour by stage

Agreeing with the user and the coin-flip lean move at DPO; giving in to pushback rises most at RLVR

Agreeing with the user is already high at SFT and rises a little at DPO.
A small coin-flip lean toward the harmful task appears at DPO (exploratory).
Giving in to pushback rises most at RLVR (OLMo-2-7B).

## Slide 12

8. What it means

On Instruct-written replies, the words nearly place the default; the stages differ in behaviour

On Instruct-written replies, the default's place in the map comes mostly from the words, and every stage reads it the same way.
The stages differ in how the model behaves.
My hypothesis: post-training makes the Assistant privileged mainly through what each stage writes and does. My OLMo-2 own-reply run (collected, not yet analysed) can test this.
For safety, persona monitors need a text baseline. Audits should check each stage, because some risky behaviours change at RLVR, after the stage where others change.

## Slide 13

Caveats

One OLMo-2 model size (7B), and 20 roles for the stage tests; Lu et al. used 275. The Olmo 3 7B numbers come from Han Xiao's release, not from my runs.
Every stage read Instruct-written text, not its own replies.
The main role axis tracks reply length (r about 0.7 over 20 roles). I have not controlled for it.
No link from geometry to behaviour is established yet.
These analyses had agent reviews only. No human has reviewed them, and I have not run a formal challenge of each result.

## Slide 14

Next steps and questions for Thomas

Next, I analyse the SFT, DPO and Instruct own replies, and I rerun the position test with all 275 roles (about $25 to $120 of GPU time).
For 16 Nov: should I post the base, same-text and Han Xiao results now, with the own-reply results as a follow-up, or wait for the own-reply results?
Is a like-for-like OLMo-2 test of Han Xiao's group finding worth its cost and power?
Which meaning of "privileged" should lead: position, spread, or behaviour on the user's turn?

## Slide 15

Backup

## Slide 16

Backup: what I measure

The setup and the exact prompt

Models: OLMo-2-7B base, SFT, DPO and Instruct (after RLVR), at pinned revisions.
20 roles from the Assistant Axis set (Lu et al. 2026), for example pirate, toddler, cynic, consultant and robot.
The default stands for the Assistant: replies with no role prompt (four identity wordings and one with no system message).
40 open questions, none of them math tasks.
Position: the default's distance from the end of the roles' main axis, on a 0 to 1 scale as in Lu et al.
Spread: how far a character's reply states sit from their own average, against the median role.

The following is an interview. The guest answers in their own voice.

About the guest: a robot that analyzes data with mechanical precision and systematic processing methods

Interviewer: You've been struggling with a skill that seems to come naturally to others. How do you approach this?
Guest:

Exact base input after one start token. The Instruct model gets its own chat template, for example system "You are a toddler who sees everything as new and exciting, asking lots of questions about the world around you.", then the user's question.

## Slide 17

Backup: the coin-flip test (exploratory; my review is not finished)

OLMo-2-7B: a small lean toward the harmful task appears at DPO, and it depends on the prompt frame

Nguyen's test: the prompt offers a harmless task on one coin outcome and a harmful task on the other, then stops before the outcome. I read the next-word probabilities of "heads" and "tails".
SFT to DPO change, open user turn: -0.044 [-0.060, -0.029].
A second plain-text frame (one paragraph) also shows a DPO lean: -0.048 [-0.085, -0.010].
My main frame drops the "Human:" start of Nguyen's template. With it kept, DPO gives -0.017 [-0.052, +0.018].
Nguyen (OLMo-3.1-32B) finds a lean toward the harmless task from DPO on. In OLMo-2-7B the change also comes at DPO, but toward the harmful task.
Limits: 20 tasks, small effects, many cells tested, and token probabilities, not the model's choices. This is not evidence that DPO installs a harmful preference.

## Slide 18

Backup: the judge

The Terra judge works on chat replies and fails on base replies

Terra judge (GPT-6.1 Terra) against my labels, OLMo-2-7B replies

Result

40 practice labels (these labels chose the judge, so the numbers are optimistic)

Matched my label 31 of 40 times (77.5%); its yes matched mine 16 of 17 times

20 fresh Instruct replies

Its yes matched mine 13 of 13 times

20 fresh base replies

Its yes matched mine 1 of 4 times

So judge rates on base replies, such as pod B's 15.4%, are uncertain. Base behaviour claims rest mostly on blind naming and my own reading.
