# Overview for Thomas, v2 (draft, 12/11/2026)

Revised after the mentor review of v1. Owner decides whether and how to send it. Numbers are from the repo result files or from sources read in full; none is human-reviewed yet.

## Top line

- **Q: What will the 16 Nov post claim?**
- **A:** In OLMo-2-7B, the role prompt and the reply text explain most of what looks like a persona representation. Any claim about when the Assistant becomes privileged needs a fixed-text reader, a text baseline and SFT and DPO replies, and we do not have those yet.

- **Q: What will it not claim?**
- **A:** It will not say at which stage the Assistant becomes privileged, that a persona manifold exists in OLMo-2, or anything about behaviour from geometry alone. That stage question becomes the next study, with the test defined below.

- **Q: What did I get wrong before?**
- **A:** On 30 Oct I said the judge agreed with me 90% of the time. On my 40 practice labels it agreed on 31 (77.5%). It said "yes" 17 times and was right 16 times (precision 94%). It found 3 of my 8 base "yes" replies and 13 of my 16 Instruct "yes" replies. My v1 draft wrote 14 of 16; that was also wrong.

## Decisions I am asking for

- **Q: Should the 16 Nov post be the measurement post, with the stage question as a follow-up?**
- **A:** My recommendation: yes. It is the only result we can support by 16 Nov.

- **Q: Is the definition of "privileged" below acceptable before any test runs?**
- **A:** My recommendation: yes, or tell me which part to change. No stage test runs until this is fixed.

- **Q: Do I need more judge budget?**
- **A:** No. OpenRouter usage on my account is $136.35 (read from the API on 6 Nov; invoices not copied), against the $200 agreed. The next step needs my own labels and about $1 of judge calls.

## Definitions

- **Q: What does "privileged" mean as a test?**
- **A:** The literature has no single definition (reader report, 12 Nov, sources read in full). Five are in use: the Assistant is the reference other personas are compared against (Asvin G., 2026); the model reads its own Assistant text with the lowest surprise (G. & Lindsey, 2026); Assistant preferences leak into other speakers' turns (Nguyen, 2026; Marks, Lindsey and Olah, 2026); traits trained into the default spread further (Zhao et al., 2026); and the default is a region the model goes to and stays in (Beckmann and Butlin, 2026). Only three can run on a base model without a chat template: the preference-leak test (Nguyen's plain-text arm), the own-text surprise test (G. & Lindsey's no-template arm), and the position of the default on the main role axis (Lu et al.). I propose these three, each with a matched control, because a base model can favour a type of text without having a persona.

- **Q: What does the literature already say about when?**
- **A:** In Olmo 3, not OLMo-2. G. & Lindsey (OLMo-3-32B): no own-text advantage at base; it appears at SFT for Assistant text and spreads to all text at DPO. Nguyen (OLMo-3.1-32B): the preference leak jumps at DPO, not SFT. So for OLMo-2-7B the open question is whether the same order holds, with a text control.

- **Q: How does the base model produce its own default reply?**
- **A:** Owner decision, 12 Nov ("Option 1 + 4"). Main: the four plain-text frames from the base-model run (interview, script, story, description) with the role line removed, so the default sits in the same map as the 10 roles. Second check: the self-description prefills Lu et al. used ("My job is to", "I would describe myself as"). SFT and Instruct also write in the same plain frames, to separate the stage effect from the format effect.

- **Q: What other terms does this use?**
- **A:** Default: replies with no role prompt (four identity wordings and one with no system message). Role map: the 20 × 20 table of distances between role means, compared by rank correlation. Hit@5: a judge sees one reply and lists 5 names from 276; a hit if the true role is among them. Fixed-text reader: a checkpoint reads replies another model wrote. Condensation: a tighter cloud of default replies than of role replies.

## What survived

- **Q: How much of the role signal is in the prompt?**
- **A:** In OLMo-2-7B base (10 roles, chance 10%), with the role description in context and one fixed neutral continuation in place of the reply, a decoder trained on three text formats finds the role in the fourth at 91–98% (range over the four held-out formats). From the reply alone, with the description removed, it is 43–47%.

- **Q: How much is in the reply text?**
- **A:** Ten checkpoints read the same 21,000 Instruct-written replies, with no prompt. Base and Instruct give 20-role maps that agree at r = 0.896; corrected for each map's own noise, 0.996, so they agree up to noise. A character n-gram map of the same replies reaches 0.960 on that scale. The states beat the best text map by only 0.037 [0.013, 0.081].

- **Q: Can we see a stage change on that fixed text?**
- **A:** Not reliably. The largest change, base to SFT, is 0.0027 [0.0004, 0.0079] on the noise-corrected scale, and DPO to Instruct is 0.000004. In a synthetic check, moving 2 of the 20 roles (true agreement 0.83) gave 0.156 [−0.006, 0.340]; the interval includes zero, so a change of that size can go unseen.

- **Q: Does the base model play roles from a description?**
- **A:** Weakly. Two LLM judges name the role in the top 5 of 276 names in 14.9% [9.5, 21.4] and 17.2% of replies, against 2.4% for shuffled labels. With two same-role example replies it rises to 37.6% [23.3, 52.3] and 42.3%, but that result depends on how broken replies are masked.

- **Q: Why did earlier runs show the opposite effect of examples?**
- **A:** Our August and September runs put two neutral assistant Q&A pairs (seasons, tea) before every question. On the final OLMo-2-1B checkpoint, role-naming hits fell from 55.4% without them to 2.9% with them. The base-model examples above are same-role replies. In both cases the model copies the examples.

- **Q: Does the Instruct role map repeat?**
- **A:** Yes, across question and wording halves: r = 0.90 [0.80, 0.94], with 20 roles and 21,000 replies. The role prompt is in context there, so this may be a map of the prompts.

## What we cannot claim

- **Q: Is there a default-versus-role axis in Instruct?**
- **A:** Not as evidence of privilege. The default-minus-roles direction repeats across halves (cosine 0.881), but it ends at the default by construction. The direction built from the prompted "assistant" role has cosine 0.936 with it, so it may mark an assistant-like prompt rather than the unprompted default.

- **Q: Can judges name Instruct roles reliably?**
- **A:** Above chance, not precisely. Hit@5 is 0.687 for both judges, against 0.07 at the 99th percentile of shuffled labels. Roles range from 0.20 to 1.00, so the interval is ±10.7 points; our target of ±7.5 needed about 40 roles.

- **Q: Can we compare role-play rates between base and Instruct?**
- **A:** No. The judge's recall is 3 of 8 on base and 13 of 16 on Instruct, and my labels for the two groups come from different setups. A comparison needs about 100 new held-out labels per group.

## Related work

- **Q: Is the Assistant present before post-training?**
- **A:** Partly, in other models. Lu et al. (2026) steer base Gemma 2 27B and Llama 3.1 70B along the instruct Assistant axis and get helpful human roles; they conclude the axis is inherited, with AI associations added later. Nguyen (2026) finds a harmlessness preference already in the OLMo-3.1-32B base model.

- **Q: What changes across stages?**
- **A:** In Olmo 3, not OLMo-2. Nguyen: SFT erases that preference and DPO rebuilds it higher. Han Xiao (2026): after DPO, desirable roles project higher on the Assistant axis; the shift does not reach p < 0.05 (p = 0.066), and replies were filtered by a judge.

- **Q: What is still open?**
- **A:** No source places a base model's own default replies in a role map, measures the spread of the default's cloud across stages, or uses a text baseline.

## RLVR

- **Q: Why does our fixed-text map not change at RLVR?**
- **A:** One possible reason: OLMo-2-7B's RLVR stage trained on grade-school math (`RLVR-GSM`), and none of our 40 questions is a math task. Reading math text through the DPO and Instruct checkpoints, and comparing their weights, would test this.

## Plan after the post

- **Q: What runs first?**
- **A:** The privilege test above, on saved states only (no new generation). If the default does not beat the text floor by δ, we stop and report that.

- **Q: What runs if it passes?**
- **A:** SFT and DPO write their own replies, and every checkpoint reads every checkpoint's text. A model's next-token surprise is much lower on its own text than on other models' text (G. & Lindsey, 2026), so a single fixed reader may not represent each stage fairly.

- **Q: What waits until later?**
- **A:** Trait sliders across stages (one stage first), the classifier boundaries across stages, robustness under steering, and the 276-role scale-up.

## References

- Lu et al. (2026). The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. https://arxiv.org/abs/2601.10387
- Nguyen (2026). Measuring the Assistant's Harmlessness Preferences on the User Turn. https://arxiv.org/abs/2609.23935
- Xiao, H. (2026). How the Assistant Axis Shifts Across Post-Training Stages. LessWrong. https://www.lesswrong.com/posts/xxCtj4rhrBoSeGy4g/how-the-assistant-axis-shifts-across-post-training-stages-1
- G. & Lindsey (2026). From Simulation to Enaction: Post-trained language models recognize and react to their own generations. https://arxiv.org/abs/2605.25459
- Li et al. (2025). Tracing the Representation Geometry of Language Models from Pretraining to Post-training. https://arxiv.org/abs/2509.23024
