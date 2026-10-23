**Verdict:** Revise before sending. Five blockers. The main naming values match the final results, but several claims exceed the evidence.

---

**Blocker 1: The 1B verdict uses a changed rule**

Write-up lines 12, 26, 73, 84–85; figure 5 title.

Evidence: [Original plan](../../261005_base_incontext_elicitation/NOTE.md), “Pass rule,” requires a valid role-play judge, played-rate thresholds, played-role coverage, and a human check. [Role scores](../../261005_base_incontext_elicitation/results_role.json), `wrong_role_played = 0.098039`, fails the 5% judge limit; `base/set2_neutral.coverage = 27`, which exceeds 20. The reported 18 roles and 15.8% instead come from [naming scores](../../261005_base_incontext_elicitation/results.json), `base/set2_neutral.{coverage,top5}`; their `verdict` applies changed naming rules, and the required human check remains pending.

Replacement: “The 1B check did not establish a pass under my original rules. The role-play judge failed its check, and the required human check is still pending. Blind naming is exploratory: the second example set gave 15.8%, with at least one hit for 18 of 60 roles.”

Figure title: “The 1B model's naming rate depends on the example set.”

---

**Blocker 2: The new broken-reply labels have not passed the full rule**

Write-up line 143; figure 4 dashed line and legend.

Evidence: [Naming results](../results/naming_v2.json), `settings.masks = ["none", "frozen"]` and `primary_decisions.plays_ex2`, confirm a first-sample pass under the original definition; `run_main/ex2.results.frozen.gpt.invalid_rate = 0.080556`. The new labels are absent from those masks; [pass rules](../THRESHOLDS.md), Part 1 rule 4, still require at most 10% invalid replies and a complete-group check. [Broken-reply report](../results/broken/finish_261016/report.json), `owner_estimate.ex2`, allows rates above 10%, so “the decisions don't depend on this” is not established for the same-role prompt.

Replacement: “The first sample passes with 8.1% invalid replies under my original definition. I have not checked the full pass rule with the new broken-reply labels. The description-only prompt already fails when I score every reply.”

Figure replacement: Remove the dashed line, or label it “10% limit for the original invalid-reply definition.”

---

**Blocker 3: The headline hides the failed repeat and treats naming as role play**

Write-up lines 1, 3, 15, 27, 97, 184; figure 1 pass label.

Evidence: [Naming results](../results/naming_v2.json), `primary_decisions.plays_ex2`, describes the first sample; `seed_rule_agreement.ex2.frozen.plays = [true,false]`. The second sample's `run_rep/ex2.results.none.gpt.clean.ci95[0] = 0.198204` misses the 20% floor. Description-only naming remains above its shuffled baseline, and [pass rules](../THRESHOLDS.md), owner decision A, retain human labels as the validity check; write-up line 161 says that check is absent.

Replacement: “For 20 selected roles, two same-role examples give 37.6% and 42.3% top-five naming in the first sample. That sample passes my numerical naming rules, while the second sample fails one judge's rule. The description alone gives 14.9% and 17.2%, above its shuffled baseline. These scores measure naming; I have not yet checked them against human labels.”

Figure label: “Passes the numerical rules in the first sample.”

---

**Blocker 4: The audit does not identify the cause of the base-model failure**

Write-up lines 11, 20, 25, 69–70.

Evidence: [GPT audit](../../../../output/results_2026-09/audit_2026-10-01/audit_judges_gpt.json) and [Opus audit](../../../../output/results_2026-09/audit_2026-10-01/audit_judges_opus.json), `char.exp1|olmo|base.top5` and `char.exp1|tulu|base.top5`, measure naming under the tested prompts, without a matched base-model prompt change. [Later audit](../../../../output/results_2026-09/audit_2026-10-02/judge_gpt.json), `olmo1b|rlvr.top5 = 0.028986` versus `olmo1b|rlvr_zeroshot.top5 = 0.554348`, tests removing examples only on the 1B final model. Neither source establishes that base models never encountered chat markup, or that examples blocked role play in every trained model.

Replacement: “I withdrew these runs as evidence for base-model role play. They show low naming under the tested prompts; I have not isolated the cause in a matched prompt comparison. On the 1B final model, removing the two example turns raised naming from 2.9% to 55.4%.”

For line 20: “I used a plain interview prompt to test the base model without chat markup.”

---

**Blocker 5: Two selected examples were not named as their own role**

Write-up line 41; claim ledger rows S1 and P1.

Evidence: [Chosen examples](../pilot_examples_final.json), entries with `id = c3e183` and `id = 5d46cc`, have `role = assistant`, `rank = 1`, but `judge_top5[0] = default`; neither top-five list includes assistant. [Selection code](../auto_examples.py), line 155, accepts default for assistant, while [naming code](../score_naming.py), `target_hits` and `ROLES`, score the target name separately. Thus 38 of 40 examples meet the literal “own role first” claim; the exception also affects the interpretation of the assistant result.

Replacement: “Thirty-eight of 40 examples were ranked first for their role. For the two assistant examples, the judge picked ‘default’ first; I accepted that for selection. The final naming test counts ‘assistant’ and ‘default’ separately.”

---

**Major 1: The length control has contrary evidence**

Write-up lines 40, 99; figure 1 neutral-text label.

Evidence: [Naming results](../results/naming_v2.json), `contrasts_run_main.ex2_filler-desc.gpt`, gives −6.3 points, 95% interval [−11.3, −2.1]; `run_main/ex2_filler.results.frozen.gpt.invalid_rate = 0.475`. The neutral text lowers naming and has many invalid replies under the original rule; [pass rules](../THRESHOLDS.md), decision C, report approximately matched lengths, 234.1 versus 231.6 tokens.

Replacement: “Same-role examples give 28.9 and 29.6 points more naming than neutral text of about the same length. The neutral text also lowers naming compared with the description alone, so it may itself harm the replies.”

---

**Major 2: “Override” is stronger than the measured comparison**

Write-up lines 3, 103, 113–116, 184; figure 2 title.

Evidence: [Naming results](../results/naming_v2.json), `run_main/ex2_other/shift6`, `shift10`, and `shift14`, keys `donor_hit5` and `results.none.*.all_replies.macro_hit5`, support donor naming of 24.8–29.8% versus target naming of 2.4–5.0%. These rates do not establish that the description has no effect; in `shift10.results.none.gpt`, `all_replies.perm_p = 0.0057`, while `clean.perm_p = 0.1577` is weaker evidence, not proof that the effect disappeared.

Replacement: “With conflicting examples, the judges name the example role more often than the described role. This comparison does not measure how much influence the description retains.”

For line 116: “In one donor set, described-role naming is above its shuffled baseline before exclusions. The evidence is weaker after copied and self-naming replies are removed.”

Figure title: “The judges name the example role more often than the described role.”

---

**Major 3: The translator example has not been isolated as the cause**

Write-up line 142.

Evidence: [Broken-reply report](../results/broken/finish_261016/report.json), `rates.run_main/ex2.{final_broken,top_broken_roles}` and `rates.run_rep/ex2.{final_broken,top_broken_roles}`, confirms 53/105 and 50/89. [Chosen examples](../pilot_examples_final.json), translator entry for `question_idx = 194`, translates the question, but these files contain no test replacing that example while keeping the rest fixed.

Replacement: “Translator replies account for 53 of 105 broken calls in the first sample and 50 of 89 in the second. One translator example also translates the question. I have not tested whether replacing that example removes this behaviour.”

---

**Major 4: Sensitivity alone does not show that the population rates are low**

Write-up line 141.

Evidence: [Broken-reply report](../results/broken/finish_261016/report.json), `agreement.owner50.gpt`, gives 10 true positives, 7 missed cases, and 2 false positives; `fresh30_by_stratum` uses selected flag groups. `owner_estimate.desc` and `owner_estimate.ex2` cover 1,748 and 1,749 replies after exclusions, rather than all 1,800; their conservative intervals include the automatic point estimates.

Replacement: “GPT found 10 of 17 replies I called broken in the first 50. It also marked 2 of 33 other replies as broken. My separate estimates use 1,748 and 1,749 replies after exclusions; their intervals are wide. These checks leave the true broken rates uncertain.”

---

**Major 5: The naming and broken-reply populations differ**

Write-up lines 14, 35, 91, 107, 140; figure 4 caption.

Evidence: [Run notes](../NOTE.md), “Full pilot run,” records 15,120 main replies and 3,654 repeat replies, including extra questions and default prompts. [Naming results](../results/naming_v2.json), the three `run_main/ex2_other/shift*` entries, use 5,400 conflicting-example replies; [broken-reply report](../results/broken/finish_261016/report.json), `limits[1]` and `rates.run_main/ex2_other.n`, use only 1,800, with one fixed donor per group.

Replacement: “The naming analysis uses 1,800 replies per prompt type and donor set, restricted to the 18 shared questions. Conflicting examples have three donor sets, giving 5,400 replies. The broken-reply analysis uses only one fixed donor per group, giving 1,800 conflicting-example replies.”

---

**Major 6: The stated pass rule omits required checks**

Write-up line 43.

Evidence: [Pass rules](../THRESHOLDS.md), Part 1 rules 2 and 4, require each counted role to exceed its own null's 99th percentile and a complete-group sensitivity check. “Primary decisions and multiple tests” requires Holm correction across four decisions; [naming results](../results/naming_v2.json), `holm.adjusted`, currently gives 0.0004 for both naming tests, with the two pending information tests entered as 1.

Replacement: “Each counted role must exceed 99% of its own shuffled scores. The result must also hold in groups where all five wordings give valid replies. I apply a correction across four planned tests; the current corrected naming p-values are 0.0004, pending the final combined report.”

---

**Major 7: The figure 1 axis adds an untested meaning**

Write-up lines 87, 93; figure 1 vertical axis and title.

Evidence: [Figure code](writeup_part1_2026-10-16_figs/make_figs.py), figure 1 `set_ylabel`, says “higher = clearer”; [naming results](../results/naming_v2.json), `run_main/ex2.results.none.*.all_replies`, measures top-five identification, not answer clarity or quality. The description-only intervals extend above 20%, so “stays under 20%” must refer to the observed means.

Axis replacement: “Replies with the target role in the judge's top five (%)”.

Title replacement: “Observed naming rises from 14.9–17.2% to 37.6–42.3% with same-role examples.”

---

**Minor 1: Figure 4 labels overlap**

Write-up line 135; figure 4 last two horizontal-axis labels.

Evidence: [Rendered figure](writeup_part1_2026-10-16_figs/fig4_broken_replies.png) visibly overlaps “only, 2nd sample” and “examples, 2nd sample”; [figure code](writeup_part1_2026-10-16_figs/make_figs.py), `cells`, puts each long phrase on one line.

Replacement labels, with a line break at each slash: “Description only / second sample”; “Same-role examples / second sample”.

---

**Minor 2: The additional internal-state figure needs separate method labels**

Write-up lines 166–168; figure 6, which is present in the figure folder but not linked in this write-up.

Evidence: [Figure code](writeup_part1_2026-10-16_figs/make_figs.py), figure 6 `series` and caption, calls all features averages over reply tokens and calls the text classifier “no model.” [Saved results](../results/part2/part2_context_run_main.json), `{desc,ex2,ex2_filler}.{R,c2,c3,text_joint_tfidf_length_stop}.{acc_mean_50,ci95}`, match all 12 plotted values, but the prompt control reads a fixed sentence and the text baseline also uses a classifier.

Caption replacement: “The coloured bars use layer 17 states from the reply or the fixed control sentence. The grey bars use reply words, length, and stop information. All bars show held-out role classification.”

Legend replacement: “Reply words, length, and stop information”.

---

**Minor 3: Keep the prose changes small**

Write-up lines 3, 5, 13, 25–30, 114.

Evidence: The main negative contrast is at line 3; the short absolute claim is at line 114; numbered result references add repeated signposting. The [claim ledger](writeup_part1_2026-10-16_ledger.md), final S2 note, says Thomas's attribution is absent from the run notes; there are no em dashes, bold-first bullets, or JSON keys in the current narrative to remove.

Replacement for line 13: “I chose the example answers by a fixed rule before the base-model run.”

For line 5, use only the plain-word claim-ledger link; remove numbered result references from the summary during the planned rewrite.

---

**Checks completed**

| Check | Result |
|---|---|
| Main naming rates, intervals, nulls, coverage, first guesses, seed differences | Match `naming_v2.json`; no stale v1 values found in the write-up or figures 1–3 |
| Audit rates and intervals | Match the saved audit files; causal scope needs correction |
| 1B rates and figure 5 bars | Match neutral-label naming results; the original verdict uses another measure |
| Broken rates, counts, agreement scores, intervals | Match the report; definitions and populations need clearer limits |
| Original and revised role-score comparisons | Match both comparison files; the five revised scores of 3 all concern own-role replies |
| Assistant 80/90 claim and quoted examples | Confirmed through `judge/KEY.json`, judge answers, and the shared-question reply records |
| Figures 1–6 | Viewed; source values checked; label findings listed above |
| Files changed by this review | This review file only; no HTTP or model calls |

The write-up changed during this review. Line numbers refer to the original reviewed version, SHA-256 `cf69ef599ad429ffd06853a5eb52189189a386f3f9e55f861b2f45bc48e6f40f`.

---

**Next step**

Fix the five blockers before sending the write-up to Ivan.
