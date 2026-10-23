# Section "Update 10/16" for MARS-experiment-logs-B (version 2: continues the earlier story)

Replace the 13 slides with ids starting `u1016_` (deck positions 2-14) by the 13 slides below, in the same place: right after slide 1 (the cover) and before "Update 9/11". Use new object ids that start with `u1016b_`. Do not change, move or delete any other slide.

Style: same as before (Arial, black text, grey #5F6368 for the small question line, headings about 26 pt, result lines 16 pt regular, body 14-16 pt, white background except slide 1, which copies the "Update 9/11" blue style). Figures as large as fits, keeping aspect ratio. Figures are in `experiments/q1_emergence/261008_base_persona_map/review/writeup_part1_2026-10-16_figs/slide/`. Keep every word and number exactly as written here.

---

## Slide 1 (section header, blue, copy the style of "Update 9/11")

Title: Update 10/16

Body:
- Last time I had just run the 7B base test of Ivan's question: does the base model actually play the role? Here are the results.
- With two example answers from the role, judges put the right role in their top 5 for 38-42% of replies. That passes the bar I set before the run on the first sample and narrowly misses for one judge on a second sample. With the description alone it's 15-17%, under the bar.
- The replies follow the examples: with another role's examples, the judges name that other role.
- With a working prompt, the base model's activations do separate the roles. So my September claim that base models can't tell roles apart doesn't hold. Whether the role map forms at SFT needs the same prompt at every stage.
- What I'd like from today: answers to the three questions at the end of this section.

## Slide 2

Heading: Where the story was

Text (14-15 pt):
- 14-24/09 report: base models barely separate roles internally (1.7-4.1% decoding), and the role map forms at SFT.
- 30/09, Ivan: does the base model actually play the persona? Test it blind: a judge reads the reply and guesses the role from all 276.
- The audit said no. The base models got a chat format, and judges named the role for 0.7-1.4% of replies vs 1.8% random. The example turns I had added also cut role play in trained models (1B chat model: 55.4% with no turns, 2.9% with them).
- Fixes: a plain interview prompt with no chat markup, example answers in the prompt (Thomas's suggestion), Ivan's blind test, and a pass bar written down before the run.
- Last update: the 1B base model got 13-30% vs about 2% random, with one example set falling short. The 7B run had just been scored.

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

Question line: Ivan's question: does the 7B base model play the role?
Heading: Same-role examples raise naming from 15-17% to 38-42%
Result line: Passes my bar on the first sample. On a second sample, one judge's lower end is 19.8%, just under 20%.
Figure: fig1_naming_by_prompt.png

## Slide 5

Question line: Is it the examples or the description?
Heading: With another role's examples, the judges name that other role
Result line: Described role 2-5%, example role 25-30%. Neutral notes of the same length give 9-13%.
Figure: fig2_example_role_wins.png
Small text box beside or under the figure: Chef description, robot examples: "...can be represented as a binary logic system with three variables". Both judges guessed robot first.

## Slide 6

Question line: Which roles get named?
Heading: Roles with a strong voice get named; detective and refugee almost never
Figure: fig3_roles_ex2.png (tall: on the left, as large as fits)
Text on the right:
- Toddler 100%, pirate 91%, robot 88% (Claude judge).
- Detective 2%, refugee 7%, assistant 9%.
- For 80 of 90 assistant replies, Claude's first guess is "default", which counts as a miss.

## Slide 7

Question line: The September question again: can the base model tell roles apart internally?
Heading: With a working prompt, the base model's activations do separate the roles
Result line: Description only: 52.3% (chance 5%). With two same-role examples: 97.1%. Re-read without the prompt: 15.2% and 58.6%, while the prompt alone gives 91.8%, so part of this can come from the prompt itself.
Figure: fig6_internal_states_prompt_removed.png
Small grey note under the figure: "September used 276 roles and a chat prompt, so its numbers aren't directly comparable."

## Slide 8

Heading: What this changes in the earlier story

Native table, 2 columns (Earlier claim | Now), 13 pt:
| Earlier claim | Now |
| Base models barely separate roles internally (14-24/09) | Doesn't hold. With a plain prompt, the 7B base model's activations separate 20 roles at 52% vs 5% chance. |
| The role map forms at SFT (14-24/09) | Not shown yet. The trained-stage runs had the example turns too, so they need the plain prompt. |
| Base maps differ between model families (14-24/09) | Withdrawn: the base prompt was unusable. |
| Pushback flip rates across stages (Experiment 2) | Held up in the 01/10 audit. |

## Slide 9

Question line: Last update's next step: score replies myself and check the automatic judges
Heading: A judge marks 5-11% of replies as broken, but I can't trust it yet
Result line: No judge reached my agreement bar (kappa 0.6) on both of my label sets. About half the same-role broken replies are the translator translating the question.
Figure: fig4_broken_replies.png

## Slide 10

Question line: Can a judge score role play the way I do?
Heading: No automatic judge scores role play the way I do yet
Native table, 3 columns:
| Comparison | Agreement (kappa) | Verdict |
| Jev vs my first 50 scores | 0.41 [0.21, 0.57] | Below my bar of 0.6 |
| Jev vs my revised scores (I had seen Jev) | 0.69 [0.49, 0.82] | First look only |
Small text: Kappa: 0 = chance, 1 = perfect. With my revised scale, 5 of 35 own-role replies kept the role through the whole reply.

## Slide 11

Heading: Weak points
- The GPT judge chose the example answers and also scores the replies.
- 20 selected roles, with examples chosen to be easy to name.
- This measures naming. Nobody has checked the guesses by hand yet.
- On the second sample, one judge misses my bar by 0.2 points.
- I set the broken-reply definition after seeing the data, and no judge passed my bar.
- The activation results show the role can be read with the prompt in view. They don't yet say where personas come from in training.

## Slide 12

Heading: Next steps
- Run the trained stages with the same plain prompt, to retest whether the role map forms at SFT (2-3 days with the GPU machine).
- Check the role guesses by hand on about 100 replies, blind to the judges (half a day).
- Finish the remaining activation runs (neutral text, other-role examples, second sample) and write them up (1 day).
- Settle the two open broken-reply cases (toddler stories, translator) and redo the naming scores (2 hours).

## Slide 13

Heading: Questions for Ivan
1. The same-role prompt mostly measures how well the base model follows example answers. Is that good enough for the base side of the stage comparison, or should it use the description alone?
2. If I keep the examples, should the trained stages get exactly the same prompt, so only the model changes?
3. Is the blind role-guessing test enough to say a model plays a role, given that no judge matches my own role-play scores yet?
