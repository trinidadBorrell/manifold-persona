# Part 2 code re-review

**Answer:** Do not run this code on real outputs yet. Three blockers remain.

---

**Scope**

I reviewed `part2_analysis.py` and the old B1-B4 and D1-D8 findings.

I did not read files in `run_main/`, `run_rep/`, or `part2/`.

| Status | Count |
|---|---:|
| Fixed | 11 |
| Partly fixed | 1 |
| Not fixed | 0 |
| New bugs | 4 |
| Blockers left | 3 |

---

## Previous findings

| ID | Status | Blocker now |
|---|---|---|
| B1 | FIXED | No |
| B2 | FIXED | No |
| B3 | FIXED | No |
| B4 | PARTLY FIXED | Yes |
| D1 | FIXED | No |
| D2 | FIXED | No |
| D3 | FIXED | No |
| D4 | FIXED | No |
| D5 | FIXED | No |
| D6 | FIXED | No |
| D7 | FIXED | No |
| D8 | FIXED | No |

---

### B1. Missing roles

**FIXED.** Split checks now reject a missing role in train or test at `part2_analysis.py:87-90`.

Cell checks reject a globally missing role at `part2_analysis.py:140-142`.

Map halves reject missing role data at `part2_analysis.py:217-224`.

---

### B2. Non-finite map values

**FIXED.** `finite` rejects non-finite values at `part2_analysis.py:56-60`.

`cosdist` rejects zero vectors at `part2_analysis.py:227-232`.

Map statistics, nulls, bootstraps, and axis values use these checks at lines 253, 271, 279, 304, and 326.

---

### B3. Unequal default-frame weights

**FIXED.** The axis now calculates one mean for each default frame at `part2_analysis.py:319-323`.

It then gives the two frame means equal weight at `part2_analysis.py:324`.

It also stops if either frame has no replies.

---

### B4. Self-test cannot fail

**PARTLY FIXED.** The self-test now has real assertions at `part2_analysis.py:478-527`.

It tests splits, block movement, decoding, maps, non-finite data, and missing roles.

It still does not run or check the primary gain path at `part2_analysis.py:362-385`.

It also compares a 10-half map estimate with a 50-half bootstrap at lines 496-499.

This mismatch does not test the D3 production fix. Keep this finding as a blocker.

---

### D1. Decoding bootstrap cell weights

**FIXED.** `cell_acc` first calculates each role-question cell at `part2_analysis.py:131-142`.

`draw_stat` averages sampled questions within roles, then sampled roles, at `part2_analysis.py:149-151`.

---

### D2. Best-baseline selection

**FIXED.** The code fits all three baseline and combined models at `part2_analysis.py:362-375`.

Each draw selects its best baseline at `part2_analysis.py:378-381`.

The point estimate selects its best baseline at `part2_analysis.py:376` and `part2_analysis.py:384`.

---

### D3. Map bootstrap statistic

**FIXED in production.** Each map draw creates 50 halves at `part2_analysis.py:295-299`.

It then calculates the mean over those halves at `part2_analysis.py:303`.

Duplicate question draws use one multiplicity vector and one half assignment.

---

### D4. Fixed map halves for the null

**FIXED.** The code creates the 50 halves once at `part2_analysis.py:388`.

Observed and permuted map statistics receive the same halves at `part2_analysis.py:391-393`.

---

### D5. Eigengaps for both halves

**FIXED.** The code calculates half A and half B gaps at `part2_analysis.py:251-252`.

It writes both labeled results at `part2_analysis.py:396`.

---

### D6. Donor inference label

**FIXED.** The donor result now says exploratory and descriptive at `part2_analysis.py:400-403`.

It also states that no interval or null exists.

---

### D7. Unplanned Holm result

**FIXED.** The file now writes only the raw intersection-union value at `part2_analysis.py:414-420`.

The note assigns the four-test Holm step to the report.

---

### D8. Axis threshold result

**FIXED.** The output writes `axis_interpretable = cosine >= 0.80` at `part2_analysis.py:412`.

---

## Distinct-role map bootstrap

**Verdict:** The code correctly implements the declared modified bootstrap.

`np.unique` builds the distinct role set at `part2_analysis.py:300`.

The code builds and centres vectors only for that set at `part2_analysis.py:303`.

It removes pairs from two copies of the same role at `part2_analysis.py:301-303`.

Pairs between different roles retain multiplicity. A pair appears once for each pair of sampled copies.

Question multiplicity remains in `qmult` at `part2_analysis.py:293`.

All copies of one question receive the same half assignment at `part2_analysis.py:294-299`.

This is a modified bootstrap, not the ordinary plug-in bootstrap. The source describes this change clearly.

A direct reconstruction matched the first code draw exactly.

A 300-draw pure-noise test gave a 95% interval of `[-0.074, 0.092]`.

---

## New bugs

### N1. Old null files can silently control new results

- **Severity:** Critical. **Blocker:** Yes.
- **File:** `part2_analysis.py:170-177`, `part2_analysis.py:270-271`, `part2_analysis.py:392`.

`map_null` returns any existing file without checking its requested draw count.

A synthetic cache test requested 99 draws. The function silently returned three cached draws.

No cache validates the seed, input identity, halves, feature identity, or algorithm version.

This matters now because the rewritten map algorithm keeps the old file names.

Store and validate metadata. Reject every cache that does not match the full analysis identity.

---

### N2. The frozen 18-question and five-wording design is not enforced

- **Severity:** High. **Blocker:** Yes.
- **File:** `part2_analysis.py:79-90`, `part2_analysis.py:94-102`.

The builders use all unique values that remain after filtering.

They do not require 18 questions and five wordings.

A synthetic 17-question, four-wording input ran with an 8/9 question split and a 3/1 wording split.

That silently changes the frozen 9/9 and 3/2 analysis.

Validate the exact design before any fit or map calculation.

---

### N3. The embedding cache has no identity check

- **Severity:** Medium. **Blocker:** No for a confirmed fresh output directory.
- **File:** `part2_analysis.py:444-446`.

The code accepts an existing embedding file by name alone.

It does not check row count, row order, model revision, or input fingerprint.

A same-size stale file can silently attach embeddings to the wrong replies.

Store cache metadata and validate it before reuse.

---

### N4. Empty-text filtering is inconsistent

- **Severity:** Low. **Blocker:** No if generation marks every empty reply invalid.
- **File:** `part2_analysis.py:74-76`, `part2_analysis.py:348-350`, `part2_analysis.py:404-406`.

`subset` excludes a valid row with empty text at `part2_analysis.py:74-76`.

The invalid count does not count that row at `part2_analysis.py:348-350`.

The axis selection includes that row at `part2_analysis.py:404-406`.

This is not a blocker if generation always marks empty replies invalid.

---

## Test evidence

| Check | Result |
|---|---|
| Supplied self-test | Exit code 0; `SELFTEST PASSED` |
| Distinct-role reference | Exact match |
| Pure-noise map bootstrap | 95% interval `[-0.074, 0.092]` |
| Reduced design | Silently accepted |
| Wrong-size map cache | Silently accepted |

The self-test command used four jobs and one OpenMP thread, as requested.
