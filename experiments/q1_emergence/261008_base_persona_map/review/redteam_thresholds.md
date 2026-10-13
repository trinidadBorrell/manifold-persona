**Bottom line:** The plan can give false passes and false failures, so it needs major changes before the pilot.

---

**Risk summary**

| Rank band | Problems | Main risk |
|---|---:|---|
| Critical | 7 | The tests do not support the stated claims. |
| High | 11 | Bias, leakage, or weak controls can change the result. |
| Medium | 4 | Secondary analyses can mislead interpretation. |

---

**1. Critical — Part 2 does not establish a persona map**

Decoding and repeatable distances show stable role-linked information. Prompt wording, reply vocabulary, or length can create both results. The plan gives this information a stronger name than the tests support. Require cross-wording transfer, external trait prediction, and prompt-only negative controls.

---

**2. Critical — The statistical unit is wrong or undefined**

Replies share roles, role-specific wordings, and questions. Questions cross roles, while wordings nest within roles. A role-only bootstrap misses question and generation variation. Define the estimand, then use crossed role-question inference with wording nested within role.

---

**3. Critical — The wrong-role null is not a valid null distribution**

The procedure draws other labels, but it does not recompute a full clustered statistic. The stated null gives one share, so its “95th percentile” is undefined. It also gives common pilot labels more influence than rare candidate labels. Permute role mappings for whole role blocks, then recompute every pass statistic.

---

**4. Critical — The activation permutation breaks dependent blocks**

Reply-level shuffling within each question treats five wordings as independent. It can also assign one role’s related replies to several labels. Permute role labels as five-wording blocks within each question. Repeat the complete split, scaling, fitting, and scoring pipeline for every permutation.

---

**5. Critical — The judge is circular for `ex2`**

The same judge family selected every example and will score replies shaped by those examples. This choice directly favors `ex2`. Exact phrase removal does not remove style or paraphrase transfer. Use another judge family or independent human scoring as the primary `ex2` measure.

---

**6. Critical — The human check cannot validate the judge safely**

One owner and 100 validation replies give weak evidence. Pooled, enriched data hide condition errors and change class prevalence. The rule limits false positives but sets no sensitivity floor. Use two blind raters, adjudication, sample weights, and per-condition sensitivity, specificity, and predictive values.

---

**7. Critical — The Part 1 pass rule permits a weak result**

A 20% hit rate means four replies succeed and sixteen fail. Coverage of ten roles lets half the role set fail. That does not support the broad claim “plays the roles.” Narrow the claim, or set a human-justified macro rate and stronger role coverage.

---

**8. High — The substantive floors use point estimates only**

Point estimates alone can meet the 20%, 10%, and 0.70 floors. Their confidence intervals can fall far below those floors. Significance against a weak null does not repair this problem. Require simultaneous lower confidence bounds above every substantive floor.

---

**9. High — Held-out questions still leak description wording**

Train and test sets reuse all five role-specific descriptions. A classifier can memorize description traces that remain in reply states. Hold out both questions and description wordings. Add new description paraphrases that never occur in training.

---

**10. High — The text baseline is too weak**

TF-IDF logistic regression is a weak surface-text baseline. Higher activation accuracy does not prove information beyond text. Add semantic, style, length, and stop-reason baselines. Test incremental activation gain over text with nested cross-validation.

---

**11. High — `ex2_other` is not a length control**

It matches example count, not token length or content. The fixed, other-group donor adds semantic conflict and donor identity. One donor per target also confounds donor and target. Rotate several token-matched donors and include token-matched neutral fillers.

---

**12. High — The `ex2` contrast has the wrong interpretation**

`ex2 > ex2_other` compares aligned examples with conflicting examples. It does not isolate prompt length or description effects. The stated attribution therefore does not follow. Estimate `ex2-desc`, donor-control minus `desc`, and neutral-filler minus `desc` separately.

---

**13. High — One stochastic reply per prompt is not enough**

Each prompt has one sample. The seed depends on batch layout, and a batch-size change changes replies. Current intervals do not measure generation variation. Repeat independent seeds and cluster by prompt; at least replicate a fixed subset.

---

**14. High — The multiple-test rule is not operational**

Holm correction cannot act directly on four composite pass rules. The plan does not define one p-value per decision. It also omits key contrasts and optional layers from the test family. Define four joint primary p-values, apply Holm, and mark all other tests exploratory.

---

**15. High — Exclusions can select an easy subset**

Part 2 drops invalid replies, and Part 1 removes copies for one check. Unequal removal can leave only easy replies in one condition. A condition can then pass on a small selected subset. Set attrition limits, define denominators, and require balanced complete-cell sensitivity analyses.

---

**16. High — The primary question set is unclear and selected**

`desc` has 40 questions, while shared comparisons use 18. Those 18 depend on which questions the examples used. Cross-condition results can therefore reflect question choice. Use the same fixed 18 for all primary decisions, with 40 and 38 as sensitivity sets.

---

**17. High — The decoding pipeline is under-specified**

The plan does not define aggregation across 20 overlapping splits. It also omits regularization, tuning, class weighting, and TF-IDF fitting rules. Filtering can make nominal 5% chance incorrect. Freeze the pipeline and use macro accuracy, training-only preprocessing, and nested tuning.

---

**18. High — Direct naming and paraphrase leakage can pass**

A reply can name its role without sustained role behavior. Exact phrase rules miss paraphrases and transferred style. They can also flag common phrases as copies. Rerun scores after role-term masking, and calibrate fuzzy-copy checks against unrelated replies.

---

**19. Medium — Map reliability can be stable for the wrong reason**

Spearman-Brown assumes comparable halves, but the plan does not test this. Shared prompts can inflate stability across repeated random halves. The 190 distance edges also share roles. Report raw correlations, block confidence intervals, and text and prompt geometry baselines.

---

**20. Medium — The plan has no power check**

Twenty roles give little cluster-level power after Holm correction. One thousand permutations give only ten draws in the upper one percent. The plan has no minimum detectable effect calculation. Simulate clustered data, then use exact tests or at least 9,999 permutations.

---

**21. Medium — The judge-hit subset is circular**

The secondary subset uses the same judge as Part 1. Selection can sharpen geometry around that judge’s preferences. This result is not an independent check. Mark it exploratory, or select the subset with independent human labels.

---

**22. Medium — The geometry checks use unsafe summaries**

PC sign matching fails when close eigenvalues let PCs rotate. The Assistant Axis can reflect unmatched default prompts. Stability alone does not solve either problem. Report eigengaps and subspace angles, then match default prompt form and length.
