# Part 2 code review

**Answer:** Do not run this code on the real outputs yet. Nine findings are blockers.

---

**Scope**

I reviewed `part2_analysis.py` against the frozen Part 2 plan. I did not open `run_main/` or `run_rep/`.

I ran the supplied self-test and small synthetic tests. The supplied self-test exited with code 0.

| Status | Count |
|---|---:|
| Must fix before the real run | 9 |
| Other findings | 3 |
| Confirmed correct checks | 9 |

---

## Correctness bugs

### B1. Missing roles change the target statistic or stop the analysis

- **Severity:** High. **Blocker:** Yes.
- **File:** `part2_analysis.py:62`, `part2_analysis.py:85`, `part2_analysis.py:179`, `part2_analysis.py:287`.
- **Failing case:** A test fold with no replies for one role omits that role from `macro_acc`. A map half with no replies for one role gives an empty mean and can make SVD fail.
- **Fix:** Validate every split and half before fitting. Stop with a clear error if any required role has no train, test, or half data.

---

### B2. Non-finite map values can give a falsely small p-value

- **Severity:** Critical. **Blocker:** Yes.
- **File:** `part2_analysis.py:184`, `part2_analysis.py:207`, `part2_analysis.py:212`, `part2_analysis.py:265`, `part2_analysis.py:313`, `part2_analysis.py:320`.
- **Failing case:** Zero-length centred vectors give `NaN`. Then `null >= observed` or `bootstrap <= 0` is false, which can give the minimum p-value.
- **Fix:** Check all vectors, correlations, bootstrap values, and null values for finite data. Stop the affected analysis instead of writing `NaN` or counting it as a non-exceedance.

---

### B3. The assistant axis does not give the two default frames equal weight

- **Severity:** Medium. **Blocker:** Yes.
- **File:** `part2_analysis.py:250`, `part2_analysis.py:256`, `part2_analysis.py:330`.
- **Failing case:** If `default_ai` has eight valid replies and `default_llm` has nine, the code uses weights 8:9. If one frame is empty, it silently uses only the other frame.
- **Fix:** Keep the default-frame label. Compute one mean for each frame, require both means, then average the two means with equal weight.

---

### B4. The self-test cannot fail

- **Severity:** Medium. **Blocker:** Yes.
- **File:** `part2_analysis.py:397`.
- **Failing case:** The test only prints values. It returns code 0 even when signal does not pass or noise does pass.
- **Fix:** Add assertions for split isolation, chance accuracy, permutation results, map reliability, block movement, paired gain, and non-finite input failures.

---

## Deviations from the frozen plan

### D1. The decoding bootstrap does not average role-question cell accuracies

- **Severity:** High. **Blocker:** Yes.
- **File:** `part2_analysis.py:101`, `part2_analysis.py:111`.
- **Failing case:** `stat` sums hits and counts after question sampling. Questions held out more often get more weight, so questions are not equal inference units.
- **Fix:** First compute one accuracy for each role-question cell. On each draw, average sampled question cells within each sampled role, then average sampled roles.

---

### D2. The gain bootstrap does not repeat selection of the best text baseline

- **Severity:** High. **Blocker:** Yes.
- **File:** `part2_analysis.py:291`, `part2_analysis.py:295`, `part2_analysis.py:308`.
- **Failing case:** Held-out scores select one baseline once. Bootstrap draws then condition on that selection, even when another baseline is best on a draw.
- **Fix:** Fit paired activation-plus-text predictions for all three baselines. Repeat the frozen best-baseline selection inside each paired bootstrap draw.

---

### D3. The map bootstrap does not rerun the observed statistic

- **Severity:** Critical. **Blocker:** Yes.
- **File:** `part2_analysis.py:200`, `part2_analysis.py:226`.
- **Failing case:** The observed map value averages 50 half splits. Each bootstrap draw uses only one half split, so its interval targets another statistic.
- **Fix:** For each crossed draw, compute the same mean over 50 disjoint question-and-wording halves. Keep duplicate question draws in one half only.

---

### D4. The map null uses new half splits for every permutation

- **Severity:** High. **Blocker:** Yes.
- **File:** `part2_analysis.py:217`, `part2_analysis.py:221`, `part2_analysis.py:316`.
- **Failing case:** The observed value and each null value use different auxiliary random splits. The permutation changes labels and the statistic's random input.
- **Fix:** Generate the 50 half definitions once. Use those exact halves for the observed value and every map permutation.

---

### D5. Eigengaps are calculated for half A only

- **Severity:** High. **Blocker:** Yes.
- **File:** `part2_analysis.py:208`, `part2_analysis.py:212`.
- **Failing case:** `sb` is calculated but never used for eigengaps. A stable half A and unstable half B are reported as stable.
- **Fix:** Calculate and report the top-three eigengaps for both halves. Label each result clearly.

---

### D6. Donor decoding has no inference result

- **Severity:** Medium. **Blocker:** No, if it is clearly descriptive and exploratory.
- **File:** `part2_analysis.py:324`.
- **Failing case:** The code reports one donor accuracy without a crossed interval or null test. This does not meet the plan's inference rules for a donor decoding claim.
- **Fix:** Label it as a descriptive exploratory estimate. Otherwise add the planned crossed bootstrap and a donor-block permutation that matches its statistic.

---

### D7. The file writes an unplanned two-test Holm result

- **Severity:** Medium. **Blocker:** No for computation. It is a blocker for the final report.
- **File:** `part2_analysis.py:384`, `part2_analysis.py:392`.
- **Failing case:** `p_holm_over_part2` is not the frozen four-decision Holm adjustment. A reader can use this smaller adjustment by mistake.
- **Fix:** Write raw intersection-union p-values here. Apply Holm once, after all four Part 1 and Part 2 p-values are present.

---

### D8. The axis threshold is not recorded as a rule result

- **Severity:** Low. **Blocker:** No.
- **File:** `part2_analysis.py:328`, `part2_analysis.py:339`.
- **Failing case:** The output gives the cosine but no `axis_interpretable` value for the frozen 0.80 threshold.
- **Fix:** Add the explicit Boolean rule after the axis calculation. Keep it outside the primary Holm family.

---

## Checks that pass

| Check | Result |
|---|---|
| Question split | Train and test questions are disjoint. |
| Wording split | Train has three wordings. Test has the other two. |
| Standardization | The scaler fits training rows only. |
| TF-IDF | The vocabulary and weights fit training text only. |
| Sentence embeddings | The fixed model does not fit this data. |
| Complete-data macro accuracy | The code averages class recall across all present roles. |
| Decoding null | Observed and null values use the same first ten fixed splits. |
| Permutation blocks | All wordings and all three donors of a role move together within a question. |
| Paired gain draws | Combined and baseline cell tables use the same sampled roles and questions. |

---

## Other requested checks

**Map halves:** The ordinary and bootstrap half builders keep each question and wording in one half only. This part is correct.

**Principal angles:** The code uses the three singular values between the two top-three subspaces. This calculation is correct.

**Donor labels:** `example_role` is the correct donor target. It uses the same held-out question and wording splits.

**Intersection-union rule:** `max(decoding p, gain p)` has the correct form. D2 currently makes the gain p-value unreliable.

**Holm function:** The step-down adjustment code is correct. The input family is incomplete in this file.

**Empty texts:** Generator-marked empty replies are excluded. A malformed `valid=true` empty corpus can still make TF-IDF stop with an empty-vocabulary error.

---

## Runtime assessment

The decoding null schedules 399,960 logistic fits. The map null schedules about four million 50-half evaluations across four conditions and two feature types.

Synthetic 4,096-feature tests took about 0.17 to 0.20 seconds per logistic fit on one local thread. One 50-half map permutation took about 2.0 seconds.

Perfect 240-core scaling gives a lower bound near 20 minutes for these nulls. Real convergence, process startup, embedding work, and storage will increase this time.

There is no checkpoint inside a condition. Cap BLAS threads at one, run a synthetic node benchmark, and save null chunks before the 9,999-permutation run.

---

## Test evidence

The supplied self-test printed signal accuracy 0.700 and noise accuracy 0.051. It also printed signal map r 0.616 and noise map r -0.013.

A synthetic block test confirmed disjoint question and wording sets. It also confirmed that donor and wording rows stay in one permuted role block.

A missing-role map test failed with `LinAlgError: SVD did not converge`. A constant-map test returned `NaN`, and an empty TF-IDF corpus raised `ValueError`.
