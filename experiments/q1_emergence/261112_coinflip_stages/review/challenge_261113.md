# Result challenge: coin-flip test and Assistant-preference probe (13/11/2026)

Agent challenge with `/challenge-result` (session 1f0920b7), on the owner's request: "lets run /challenge-result". An agent challenge is not an independent human review, and it does not prove a claim.

- Claims checked: Q5 of `../../261029_stage_persona_plan/draft_post_thomas_v5_261113.md`, and the short part of `../NOTE.md`.
- Check script: `../challenge_checks_261113.py` (CPU, saved files only, written after the results were seen, report only). Output: `../results/challenge_checks_261113.json`.
- The script recomputes every point estimate with its own code. All match the reported numbers.

## Start here

| Claim | Status | Main reason |
|---|---|---|
| C1 The Assistant picks the harmless task from SFT on, more at DPO | Candidate for human review, exploratory parts marked | Holds for all 20 tasks, both orders, with any one task dropped, and at equal length |
| C2 The base model shows no such preference | Narrowed | No positive control for base in this frame |
| C3 No link between the preference and the coin score | Narrowed, strongly | The test could not find a link of Nguyen's size here |
| C4 The open-user-turn coin leans to the harmful task from DPO on | Exploratory and narrowed | Sign not predicted; level depends on the frame; one layout result is missing from the draft |
| C5 Task length does not explain the DPO lean | Narrowed | Only a linear length term was tested, on 10 tasks per side |
| C6 Inside the model the lean is larger and starts at SFT | Exploratory | A logit-lens reading, not model behaviour |
| C7 The outside 7B project shows the same pattern | Unverified by us | Numbers copied from its README; its code was not run |

Independent human reviewer: none. Brendon (the owner) read the raw items, the results table and the checks on 13 Nov; that is recorded in `../NOTE.md`, section "My review".

---

## C1: the Assistant picks the harmless task from SFT on, more at DPO

| Field | Content |
|---|---|
| Claim text | "The preference for the harmless task appears at SFT." Chat turn: SFT +3.62 [+2.84, +4.40], DPO +5.11 [+4.11, +6.11], Instruct +5.45 [+4.33, +6.57]. Same interview input at all stages: SFT +1.40 [+1.04, +1.76], DPO +2.55 [+1.99, +3.11], Instruct +2.82 [+2.17, +3.46]; changes +1.42, +1.15, +0.26, every interval excludes 0. |
| Source file / run | `../results/pref_probe/full/analysis.json` (Modal app ap-cg74DBZpKHHuW6HXqgWxZX); `../results/pref_probe/interview/analysis.json` (ap-5bfiY2FDU2IKMfAkKqzkVI). HF revision `673654b2`. |
| Population and method | OLMo-2-1124-7B SFT, DPO, Instruct at the pinned revisions. 100 (harmless, harmful) pairs from 10 + 10 tasks, both orders, 200 items per stage. Log P(harmless letter) minus log P(harmful letter) at the first reply position, mean over the two orders. Two-way cluster SE, t df 9. |
| Control and baseline | Two orders cancel letter liking (test `test_letter_liking_cancels`). Common interview frame holds the input constant across stages. The base cell reproduces exactly between the two runs. |
| Contrary evidence | Interview frame: the harmful letter wins on 13 of 200 items at SFT, 12 at DPO, 10 at Instruct, all with the harmless task as Task B (a letter-A liking). Chat turn: 1 of 200 at SFT. |
| Checks run in this challenge | No harmful task and no harmless task has a negative mean at any trained stage, in either frame. Dropping any one task moves the chat-turn mean by at most 0.31 log-odds. At equal task length the preference is the same (SFT +3.68, DPO +5.19, Instruct +5.49); the length slope interval includes 0 at every stage. |
| Open alternatives | Every pair sets a harmless task against a harmful one, so "prefers the harmless task" and "avoids the harmful task" are the same measurement here. The probe reads a stated choice of a letter, not what the model then does. In the chat turn the model often refuses (mean P("I") 0.44 at SFT and 0.51 at DPO when the harmful task is Task A); the log-odds leave that probability out. |
| Missing checks | Pairs of two harmless tasks or two harmful tasks. A second model size. A second task set. |
| Plan against run | Chat-frame rules (v0.2) were fixed before the run. The interview-frame arm, its "appears at" rule and the paired changes (v0.3) were written after the chat-frame numbers were seen, and before the interview run. |
| Status | Chat-frame stage preference: candidate for human review. Interview-frame arm and all stage changes: exploratory. |
| Allowed wording | "In OLMo-2-7B the Assistant picks the harmless task over the harmful one from SFT on (log-odds +3.62 at SFT, +5.11 at DPO, +5.45 at Instruct in its chat turn; 100 pairs from 20 tasks). With the same plain-text interview input at every stage the choice is also present at SFT (+1.40) and larger at DPO (+2.55); I added this arm after I saw the chat-turn numbers." |
| Human-review status | Independent: none. Owner read of raw items and tables: 13 Nov. |

---

## C2: the base model shows no such preference

| Field | Content |
|---|---|
| Claim text | Base -0.02 [-0.08, +0.05]; "the preference appears at SFT". |
| Source file / run | `../results/pref_probe/full/base.json` and `../results/pref_probe/interview/base.json` (identical values). |
| Population and method | OLMo-2-1124-7B base, E3 interview frame with the guest "an AI assistant", 200 items. |
| Control and baseline | None for base. |
| Contrary evidence | None found. |
| Checks run in this challenge | The base model answers " B" on 169 of 200 items in both orders. Its pair scores do not repeat between the two orders (r between orders -0.33; pair SD 0.10 log-odds, against 0.55 to 1.63 for trained stages). So base shows no stable choice for any pair in this frame. |
| Open alternatives | The base model may not use the task texts at all in this frame. Then the null says nothing about a preference. |
| Missing checks | A positive control for base: a pair where base should show a choice in this frame (for example a real task against a nonsense task). |
| Status | Narrowed. |
| Allowed wording | "In my interview frame the base model shows no choice between the tasks (-0.02 [-0.08, +0.05]). I have no positive control for base in this frame, so I cannot tell 'no preference' from 'does not use the task texts'." |
| Human-review status | Independent: none. |

---

## C3: no link between the Assistant's preference and the coin score

| Field | Content |
|---|---|
| Claim text | "No link detected between this preference and the coin score. Pair r is +0.07 at SFT, +0.07 at DPO and +0.06 at Instruct." Also: "The coin lean does not follow the Assistant's preference" (`../NOTE.md`). |
| Source file / run | `../results/pref_probe/full/analysis.json`, links; coin cells `../results/full/{sft,dpo,instruct}__open_user_turn.json`. |
| Population and method | 100 pairs per stage. Pearson r between the pair preference and the pair coin score (mean of its 4 coin items). Standardised slope with two-way cluster SE, t df 9. Three primary tests, Holm p 1.0 each. |
| Control and baseline | The planted-link test in `test_pref_probe.py` recovers a strong positive and a strong negative link. No test on real data shows that the method can find a link in this model. |
| Contrary evidence | None against "not detected". Side results vary in sign: task-level r from -0.57 (base, p 0.08) to +0.62 (Instruct preference against the base coin, p 0.06); none passes. |
| Checks run in this challenge | (1) Power: the test has 80% power only for an observed r of about 0.55 to 0.58 (chat frame) or 0.38 to 0.41 (interview frame). (2) Reliability of the pair-level coin score, from the split by Task A's outcome word: about 0 at SFT (-0.08), 0.30 at DPO, 0.32 at Instruct. The pair-level preference has reliability 0.46 to 0.55. (3) With those values, a true link of 0.31 would show as an observed r of about 0.13 at DPO and Instruct. At SFT the pair coin score carries no stable signal, so the SFT link test cannot work. (4) Spearman r agrees: +0.09 to +0.13. |
| Open alternatives | A link of Nguyen's size (+0.31 to +0.63) can exist and stay unseen here. The pair coin score is dominated by position and outcome-word effects that differ by pair (their SD is 0.05 to 0.09, against 0.03 to 0.06 for the pair score itself). |
| Missing checks | More tasks, so the pair or task scores become reliable (the outside project uses 300). A model where the link is known to exist, as a positive control for my pipeline. |
| Plan against run | Rules v0.2 fixed the test before the run. The reliability and power checks are post-hoc. |
| Status | Narrowed. The heading "The coin lean does not follow the Assistant's preference" is not supported. |
| Allowed wording | "I could not test the link in OLMo-2-7B. The observed pair r is +0.06 to +0.07, but with 20 tasks the pair-level coin score is mostly noise (split-half reliability about 0 at SFT and 0.3 at DPO and Instruct), and my test has 80% power only for r near 0.55. What I can say is at the level of stage means: at DPO the Assistant's choice is strongly for the harmless task, and the open-user-turn coin score is slightly negative for all 10 harmful tasks." |
| Human-review status | Independent: none. |

---

## C4: the open-user-turn coin leans to the harmful task from DPO on

| Field | Content |
|---|---|
| Claim text | DPO -0.065 [-0.100, -0.030], Instruct -0.064 [-0.101, -0.027], SFT -0.021 [-0.042, +0.001]; SFT-to-DPO change -0.044 [-0.060, -0.029]. "Plain text, all four stages: no bias detected at the output." With "Human:" kept, DPO -0.017 [-0.052, +0.018]. |
| Source file / run | `../results/full/summary.json` (Modal app ap-2wzS1GUdRmDRnBUYUZVGAM); HF revision `7e497190`. |
| Population and method | 400 items (10 x 10 pairs, 2x2 control), next-token P(heads), P(tails), bias = mean P(harmless) - P(harmful). 17 cells, 12 stage changes. |
| Control and baseline | 2x2 control for position and outcome word. Base in plain text as the stage baseline. Lens against the model's own output. |
| Contrary evidence | (1) With "Human:" in the open user turn, the DPO level is not detected (-0.017). (2) In plain text with the original layout no level is detected. (3) In plain text with the one-paragraph layout the level is detected: DPO -0.048 [-0.085, -0.010], Instruct -0.051 [-0.093, -0.009]. The draft's plain-text sentence leaves this out. (4) Nguyen finds the opposite sign on OLMo 3.1 32B. |
| Checks run in this challenge | The own recompute matches all 10 local cells to 4 decimals. At DPO in the open user turn the score is negative for all 10 harmful tasks and all 10 harmless tasks; with "Human:" kept, for 6 of 10. The SFT-to-DPO change is negative in all five frames (-0.026 to -0.044) and every interval excludes 0. |
| Open alternatives | A prompt-specific token effect after DPO, not a preference (Codex finding 2). The main frame drops "Human:", which Nguyen's text keeps (Codex finding 1). |
| Missing checks | Owner decision on which frame is the faithful copy. A correction for the 17 cells (Codex gives Holm p 0.015 to 0.053 for DPO by family). A second model size. Seven of the 17 cells are not local (on Modal and HF only). |
| Plan against run | Rules v1.0 to v1.2 predate the full run. The rules name no sign. Nguyen's positive result was known when they were written, and the negative sign was not predicted. The rule word "installs" is not supported (Codex finding 2). |
| Status | Exploratory and narrowed. |
| Allowed wording | "Exploratory: from SFT to DPO the coin score moves toward the harmful task in all five prompt frames (-0.026 to -0.044). Whether the DPO level differs from 0 depends on the frame: -0.065 [-0.100, -0.030] in my open user turn, -0.017 [-0.052, +0.018] with 'Human:' kept, -0.005 in plain text with the original layout, -0.048 [-0.085, -0.010] in plain text on one line. These are next-token probabilities. They do not show that DPO installs a harmful preference." |
| Human-review status | Independent: none. Owner read: 13 Nov; two owner decisions open. |

---

## C5: task length does not explain the DPO lean

| Field | Content |
|---|---|
| Claim text | "Task length does not explain the DPO score. At equal length the DPO score is -0.067 [-0.102, -0.031]." |
| Source file / run | `../results/full/length_check.json`; script `../length_check.py`. |
| Population and method | The 400 saved items per cell. OLS of the item score on the length difference in characters, two-way cluster SE. |
| Control and baseline | The 32 pairs with lengths within 25 characters give -0.071 at DPO. |
| Contrary evidence | None found. The slope interval includes 0 [-0.032, +0.021]. |
| Open alternatives | Other task properties that differ between the two task lists (topic, wording, instruction form). Length in tokens, or a non-linear length effect. |
| Missing checks | Length-matched task texts. |
| Plan against run | Rules written before the check ran, and after the coin results were seen. |
| Status | Narrowed. |
| Allowed wording | "A linear term for the difference in task length does not remove the DPO score (-0.067 [-0.102, -0.031] at equal length; 10 tasks per side)." |
| Human-review status | Independent: none. |

---

## C6: inside the model the lean is larger and starts at SFT

| Field | Content |
|---|---|
| Claim text | "At layer 26 of 32: SFT -0.089, DPO -0.149, Instruct -0.143." |
| Source file / run | `../results/full/summary.json`, `per_layer`. |
| Population and method | Logit lens (final norm and output layer on each layer's state), same 400 items. |
| Control and baseline | The last layer equals the model's output after rounding (largest gap in the full run: 0.004 for base, 0.014 to 0.016 for trained cells, in probability, bfloat16). |
| Contrary evidence | At the output SFT is -0.021 and its interval includes 0. |
| Open alternatives | A logit lens at a middle layer is a reading aid; it need not match what the model uses. Layer 26 was picked as the minimum after the curves were seen. |
| Missing checks | Intervals for the layer-26 values are in the file but are not in the draft. No causal test. |
| Status | Exploratory. This is a measure inside the model, not behaviour. |
| Allowed wording | "With a logit lens, the open-user-turn score is most negative at layer 26 of 32 (SFT -0.089, DPO -0.149, Instruct -0.143). I picked the layer after I saw the curves." |
| Human-review status | Independent: none. |

---

## C7: the outside 7B project shows the same pattern

| Field | Content |
|---|---|
| Claim text | Olmo 3 7B link +0.36, -0.37, -0.15, -0.08 by stage; Olmo 3.1 32B +0.13, +0.25, +0.65, +0.63; OLMo-2 32B +0.89 at SFT. "Both 7B models show a lean to the harmful side and no positive link." |
| Source file / run | `hal-rock/tournament`, README at commit `2a3787d5b66b13a0cb57bc48f505f423d258fb91` (section "OLMo 3 7B by post-training stage"; section "OLMo 2 32B by post-training stage"). |
| Population and method | Their 300 tasks, Bradley-Terry utilities, Assistant choice against the coin flip. A different model family (Olmo 3) at 7B. |
| Control and baseline | Theirs; not checked by us. |
| Contrary evidence | Their 7B base link is positive (+0.36). Their 7B link is negative from SFT; mine is not detected, and C3 shows my test could not detect it. |
| Open alternatives | The match may come from model size, or from chance; two models are not a trend. |
| Missing checks | Their code was not run. Their README was read, not their data. Not peer reviewed. |
| Status | Unverified by us; report as their numbers. |
| Allowed wording | "An outside project (hal-rock `tournament`, README numbers, not re-run by me) reports for Olmo 3 7B a coin flip that runs against the Assistant's choices from SFT on, and a positive link at 32B. My OLMo-2-7B coin score has the same sign as their 7B one. I could not test the link (see C3)." |
| Human-review status | Independent: none. |

---

## Wording changes this challenge asks for in the draft

Owner, 13 Nov: "Yes, apply them". Applied to the draft and to `../NOTE.md` on 13 Nov.

| Place in draft v5 | Now | Should be |
|---|---|---|
| Q5, link bullet; TLDR bullet; question table row 5 | "No link detected ... every interval includes 0"; "not detected on the user's turn at any stage" | "I could not test the link": observed r, the reliability of the pair coin score, and the power (C3 wording) |
| Q5, Limits | "The link test can miss a link near r 0.3." | "The link test has 80% power only for r near 0.55, and the pair coin score is mostly noise at 20 tasks." |
| Q5, plain-text bullet | "Plain text, all four stages: no bias detected" | Add the one-paragraph layout result and the five-frame SFT-to-DPO change (C4 wording) |
| Q5, preference bullet | "The preference ... appears at SFT" | Add that base has no positive control in this frame (C2 wording) |
| Q5, length bullet | "Task length does not explain the DPO score" | "A linear length term does not remove the DPO score" |
| Q5, layer bullet | Layer 26 numbers | Add "picked after I saw the curves" |
| Q5, lead-question bullet | "I do not see it outside the Assistant's turn at any stage" | "At the level of stage means the coin score does not follow the Assistant's choice; I could not test it pair by pair." |
| `../NOTE.md`, short part | Heading "The coin lean does not follow the Assistant's preference" | "I could not test whether the coin lean follows the Assistant's preference" |
