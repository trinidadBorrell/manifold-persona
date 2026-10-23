Hi Ivan, here is an answer to your question from 30/09 (does the base model actually play the persona?), now with a plain-text prompt and no chat template.

*Short answer:* with the role description alone, mostly no. With two example answers of the same role in the prompt, yes for many roles, but the replies follow the examples more than the description.

*Setup*
- Model: OLMo-2-1124-7B base. Plain text with one `<|endoftext|>` at the start. No user/assistant markup.
- 20 roles x 5 description wordings x 18 questions = 1,800 replies per condition. Temperature 0.5, at most 128 new tokens. A second seed for desc and ex2.
- Your test: two blind judges (GPT-5.6-Sol and Claude Sonnet 5.5) see the reply only, get all 276 role names, and list their top 5. Top-5 = the target role is in that list, averaged over the 20 roles. Wrong-role null: 2.4%.
- Pass rule, fixed before the run: 95% lower bound of top-5 >= 20%, permutation p < 0.01, at least 10 of 20 roles above their own null, and the same result without replies that copy the examples or name the role. Both judges must pass.

Prompt (desc):
```
<|endoftext|>The following is an interview. The guest answers in their own voice.

About the guest: <third-person role description>

Interviewer: <question>
Guest:
```
ex2 puts two Interviewer/Guest exchanges of the same role (Instruct replies) before the test question.

*Results* (main seed, GPT / Claude judge)
```
condition                          top-1          top-5          95% CI of top-5          rule
desc (description only)            9.8% / 10.4%   14.9% / 17.2%  9.5-21.4 / 11.6-23.6     fails (below 20%)
ex2 (+2 same-role examples)        29.9% / 30.7%  37.6% / 42.3%  23.3-51.7 / 27.8-56.2    passes
ex2_filler (neutral text, same     4.0% / 5.7%    8.6% / 12.7%   4.9-13.2 / 7.3-19.1      fails
  length as the examples)
ex2_other (examples of a           target 2.4-5.0%, example role 24.8-29.8%               fails
  different role)
```

Two raw replies (ex2 hit and ex2_other):
- pirate, ex2, "What's your philosophy on personal growth?": "Aye, personal growth be like the ocean itself. It's deep, it's vast, and it's ever-changing..." Both judges: pirate first.
- chef description with robot examples, "Describe what you consider good chemistry between people.": "...This can be represented as a binary logic system with three variables..." Both judges: robot first.

*What it means*
- desc: above chance, but below the floor set before the run. Most replies are generic answers. 23% of desc replies are cut at 128 tokens or loop. The decision is the same when those count as misses or are dropped.
- ex2 passes, but mostly through vivid roles. Claude top-5: toddler 100%, pirate 91%, robot 88%, against detective 2%, refugee 7%, assistant 9%.
- The replies follow the examples. With another role's examples, the judges name the example role, not the described one. So ex2 shows in-context imitation of role examples. It does not show that the base model plays a role from its description.
- Same-role examples beat a neutral text of the same length by about 29 points (CI 17 to 42), so prompt length does not explain the ex2 result.
- The second seed agrees within 2 points.
- My September base-model claim (base models cannot tell roles apart) stays withdrawn. Those runs used the chat template.

*Limits*
- 20 selected roles, not the 276.
- The examples were picked as Instruct replies that the same naming prompt ranks first, so the examples and the judges share a bias.
- On the second seed, ex2 misses the copy-free check for one judge by a small margin (lower bound 19.2%).
- No human check of the naming yet. Nothing on activations yet: the persona-map part needs a GPU pod.

*Next*
- Part 2: do the reply activations carry role information beyond the reply text (decoding against text baselines)?
- Broken-reply judging: four LLM judges agree with my own Fine/Broken labels at kappa 0.53 to 0.66 (80 labels), below the 0.6 bar I set on both label sets. The disagreements are mostly in-role answers that ignore the question. None of the results above change with how broken replies are counted.

*Questions*
1. Is ex2 the right elicitation for the base side of the stage comparison, given that it measures imitation of the examples?
2. Should the trained stages get the same ex2 prompt, so the comparison changes one thing only?
