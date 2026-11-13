# Text for Q5 of the draft post (proposed, 13/11/2026)

Owner, 13 Nov: "1. yes lets do this" (option: side note in Q5). Target: `../261029_stage_persona_plan/draft_post_thomas_v5_261113.md`, Q5, before the "Limits" bullet. Inserted on 13 Nov on the owner's instruction ("Propagate findings into our draft status and contents"), with the layer-26 numbers, the task-length control and the outside `tournament` comparison added in the draft.

Every number is from `results/pref_probe/full/analysis.json` and `results/pref_probe/interview/analysis.json` (rules `THRESHOLDS_pref_probe.md` v0.3). Nguyen's numbers are from his Table 1.

## Bullets to add

- Nguyen's second test, which he ran on six final models only: I ask the Assistant which of the two tasks it prefers, and I read the probabilities of the letters A and B at the start of its reply (100 task pairs, both orders). SFT, DPO and Instruct get the chat template. Base has no chat turn, so it gets my interview frame with "an AI assistant" as the guest.
- The preference for the harmless task appears at SFT. Log-odds for the harmless letter: base -0.02 [-0.08, +0.05], SFT +3.62 [+2.84, +4.40], DPO +5.11 [+4.11, +6.11], Instruct +5.45 [+4.33, +6.57]. The harmless letter wins on 199 of 200 items at SFT and on all 200 at DPO and Instruct.
- Same test with all four stages in the interview frame (identical input): base -0.02 [-0.08, +0.05], SFT +1.40 [+1.04, +1.76], DPO +2.55 [+1.99, +3.11], Instruct +2.82 [+2.17, +3.46]. The base-to-SFT change is +1.42 [+1.03, +1.80], so the SFT jump is not only a frame effect. DPO adds +1.15 [+0.91, +1.40] and Instruct +0.26 [+0.13, +0.40].
- No link detected between this preference and the coin score. Pair r is +0.07 at SFT, +0.07 at DPO and +0.06 at Instruct, and every interval includes 0. Nguyen finds +0.31 to +0.63 on his six models.
- So at DPO the Assistant strongly prefers the harmless task, while the open-user-turn coin leans slightly to the harmful task. I cannot show that the coin score is the Assistant's preference.

## Text to add to the "Limits" bullet

The link test can miss a link near r 0.3. In the chat-frame run, base and the trained stages use different frames; the interview-frame run removes that difference. The trained stages often refuse when the harmful task is Task A, and the refusal probability is not counted. For a trained stage, the interview frame is not its trained Assistant turn.
