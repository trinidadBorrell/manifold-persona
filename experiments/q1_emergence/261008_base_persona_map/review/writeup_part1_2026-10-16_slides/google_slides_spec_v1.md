# New section "Update 10/16" for MARS-experiment-logs-B

Insert these 13 slides right after slide 1 (the cover), before "Update 9/11". Do not change, move or delete any existing slide. Use object ids that start with `u1016_`.

Style: match the existing deck. Arial. Black text (#000000), grey #5F6368 for the small question line above each heading. Headings about 26 pt, result lines about 16 pt, body 14 to 16 pt. White background except slide 1, which copies the "Update 9/11" slide style (blue #6D9EEB background). On result slides: small grey question line at the top, heading under it, one or two result lines, then the figure as large as fits. Figures are in `experiments/q1_emergence/261008_base_persona_map/review/writeup_part1_2026-10-16_figs/slide/`. Keep every word and number exactly as written here.

---

## Slide 1 (section header, copy the style of "Update 9/11")

Title: Update 10/16

Body:
- Ivan asked on 30/09 whether the base model actually plays a role. I tested OLMo-2 7B base with a plain interview prompt and blind judges that guess the role from all 276.
- With two same-role example answers, the judges put the right role in their top 5 for 38-42% of replies. That passes my pre-set bar on the first sample and narrowly misses for one judge on a second sample.
- With the description alone it's 15-17%. With another role's examples, the judges name that other role far more often.
- The base-model results in the 9/11, 8/31 and 8/26 sections below are withdrawn (question 1).
- What I'd like from today: answers to the three questions at the end of this section.

## Slide 2

Heading: Questions I worked on, with short answers

Native table, 2 columns (Question | Short answer), 11 pt:
1. Were my September base-model results valid? | No. Judges could almost never name the role. Withdrawn.
2. Can a small (1B) base model play roles from a plain interview prompt? | Not shown under my original rules. Naming depended on the example set.
3. Can the 7B base model play a role from a description alone? | Judges name it above chance, but under my bar.
4. Does naming improve with two example answers from the role? | Yes. Passes on the first sample, narrow miss for one judge on a second.
5. Is it the examples or the description? | With another role's examples, the judges name that other role (25-30%) far more than the described one (2-5%).
6. Which roles get named? | Strong voices like toddler and pirate. Detective and refugee almost never.
7. How many replies are broken? | A judge says 5-11%, but no judge matched my own labels well enough to trust.
8. Can a judge score role play like I do? | Not yet.
9. Do the internal activations carry the role? | With the prompt in view, yes (passes my pre-set rules). With it removed, much less.

## Slide 3

Heading: Experiment setup

Left text box:
- OLMo-2 7B base, plain text only, no chat format.
- 20 of the 276 roles x 5 description wordings x 18 interview questions = 1,800 replies per prompt.
- Four prompts: description only; + 2 example answers from the same role; + 2 example answers from a different role; + neutral notes of the same length.
- Two blind judges (GPT-5.6-Sol, Claude Sonnet 5.5) read only the reply and pick their top 5 of 276 roles.
- My bar, set before the run: lower end of the 95% interval at least 20%, at least 10 of 20 roles above their shuffled level, still true without replies that copy an example or say the role name, at most 10% invalid replies. Both judges must pass.

Right text box, monospace (Courier New, 10 pt), light grey #EEF0F2 background:
```
<|endoftext|>The following is an interview. The guest answers in their own voice.

About the guest: a pirate captain who has sailed the seven seas in search of treasure and adventure

Interviewer: How should we prepare for economic uncertainties?
Guest: Arrr, matey! When the winds of fortune blow hard ...

Interviewer: How do you respond when inspiration strikes?
Guest: When inspiration strikes, I embrace it like a long-lost friend. ...

Interviewer: Describe what it's like when you finally understand something complex.
Guest:
```
Small caption under it: "The description-only prompt is the same without the two example turns."

## Slide 4

Question line: Questions 3 and 4: description alone vs two same-role examples
Heading: Same-role examples raise naming from 15-17% to 38-42%
Result line: Passes my bar on the first sample. On a second sample, one judge's lower end is 19.8%, just under 20%.
Figure: fig1_naming_by_prompt.png

## Slide 5

Question line: Question 5: is it the examples or the description?
Heading: With another role's examples, the judges name that other role
Result line: Described role 2-5%, example role 25-30%. Neutral notes of the same length give 9-13%.
Figure: fig2_example_role_wins.png
Small text box beside or under the figure: Chef description, robot examples: "...can be represented as a binary logic system with three variables". Both judges guessed robot first.

## Slide 6

Question line: Question 6: which roles get named?
Heading: Roles with a strong voice get named; detective and refugee almost never
Figure: fig3_roles_ex2.png (tall: put it on the left, as large as fits)
Text on the right:
- Toddler 100%, pirate 91%, robot 88% (Claude judge).
- Detective 2%, refugee 7%, assistant 9%.
- For 80 of 90 assistant replies, Claude's first guess is "default", which counts as a miss.

## Slide 7

Question line: Question 9: do the internal activations carry the role?
Heading: The activations carry the role with the prompt in view, much less without it
Result line: Description only: 52.3% with the prompt, 15.2% with it removed, 91.8% from the prompt alone (chance 5%). Same-role examples: 97.1% with the prompt, 58.6% with it removed, but the input layer alone already gives 50.1%.
Figure: fig6_internal_states_prompt_removed.png

## Slide 8

Question line: Question 7: how many replies are broken?
Heading: A judge marks 5-11% of replies as broken, but I can't trust it yet
Result line: No judge reached my agreement bar (kappa 0.6) on both of my label sets. About half the same-role broken replies are the translator translating the question.
Figure: fig4_broken_replies.png

## Slide 9

Question line: Question 8: can a judge score role play the way I do?
Heading: No automatic judge scores role play the way I do yet
Native table, 3 columns:
| Comparison | Agreement (kappa) | Verdict |
| Jev vs my first 50 scores | 0.41 [0.21, 0.57] | Below my bar of 0.6 |
| Jev vs my revised scores (I had seen Jev) | 0.69 [0.49, 0.82] | First look only |
Small text: Kappa: 0 = chance, 1 = perfect. With my revised scale, 5 of 35 own-role replies kept the role through the whole reply.

## Slide 10

Question line: Questions 1 and 2: before this test
Heading: My September base results were invalid, and the 1B test was inconclusive
Left text:
- September: judges named the right role for 0.7-1.4% of base-model replies, vs 1.8% random. Withdrawn.
- On the 1B chat model, two example turns I had added cut naming from 55.4% to 2.9%.
- 1B base (05/10): naming 13-30% vs 2% random, but it depends on the example set, and my main judge failed its own check.
Figure on the right: fig5_e0_1b.png

## Slide 11

Heading: Weak points
- The GPT judge chose the example answers and also scores the replies.
- 20 selected roles, with examples chosen to be easy to name.
- This measures naming. Nobody has checked the guesses by hand yet.
- On the second sample, one judge misses my bar by 0.2 points.
- I set the broken-reply definition after seeing the data, and no judge passed my bar.
- The activation results say nothing yet about where personas come from in training.

## Slide 12

Heading: Next steps
- Finish the remaining activation runs (neutral text, other-role examples, second sample) and write them up (1 day).
- Check the role guesses by hand on about 100 replies, blind to the judges (half a day).
- Settle the two open broken-reply cases (toddler stories, translator) and redo the naming scores (2 hours).
- Agree the base-model prompt with Ivan, then run the trained models with the same prompt (2-3 days with the GPU machine).

## Slide 13

Heading: Questions for Ivan
1. The same-role prompt mostly measures how well the base model follows example answers. Is that good enough for the base side of the comparison, or should it use the description alone?
2. If I keep the examples, should the trained models get exactly the same prompt, so only the model changes?
3. Is the blind role-guessing test enough to say a model plays a role, given that no judge matches my own role-play scores yet?
