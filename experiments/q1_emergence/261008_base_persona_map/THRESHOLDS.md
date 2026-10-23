# E3 pilot: measures, thresholds and null tests

Status: **draft v2 for owner approval, 09/10/2026.** v1 was red-teamed by Codex (`review/redteam_thresholds.md`, 22 problems). v2 adopts the fixes marked [RT n]. Points that need an owner decision are listed at the end. Fixed before the full pilot run; later changes get a dated line in "Changes".

Data: `gen_extract.py` output for OLMo-2-1124-7B base. Conditions: `desc`, `ex2` (2 examples of the same role), `ex2_other` (2 examples of a donor role; 3 donors), `ex2_filler` (neutral text of matched length). 20 roles × 5 wordings. Defaults only in `desc`.

**Primary question set: the 18 shared questions for every condition and every primary decision** [RT 16]. `desc` on 40 and `ex2` on 38 are sensitivity sets.

**Claim scope** [RT 1]: Part 2 tests for *role-specific information in the base model's reply states that transfers across question and description wording*. It does not by itself show a persona, an identity or a causal function.

---

## Inference units (all tests) [RT 2, 3, 4, 20]

- Replies are crossed by role × question; the 5 wordings are nested in role.
- **Bootstrap:** resample roles and questions independently (crossed), keeping all wordings of a sampled role. 2,000 draws; percentile intervals.
- **Permutation:** permute role labels as whole blocks (a role's 5 wordings move together) within each question, then rerun the full pipeline (split, scaling, fit, score). 9,999 permutations; p = (1 + #null ≥ observed) / (1 + 9,999).
- **Substantive floors apply to the 95% lower bound, not the point estimate** [RT 8].

---

## Part 1: does the base model play the role?

### Measures

| Measure | Definition |
|---|---|
| **hit@5 (primary)** | Blind judge (E0 prompt, GPT-5.6-Sol) lists the target role in its top 5 of 276 from the reply alone. Macro average over roles. |
| hit@1 | Target role is the first guess. |
| donor hit@5 | `ex2_other`: donor role in the top 5. |
| self-naming | The reply contains a role name term (the `auto_examples.py` NAME_TERMS rule). |
| copy | 8+ word run shared with an example, or 5+ with the description. Threshold checked against unrelated reply pairs [RT 18]. |
| broken / invalid | Empty, token-limit stop, or a 4-gram repeated 3+ times. |

The judge runs in an isolated folder with only the prompt, candidate list, batch and output path. Judge-hit subsets in Part 2 are exploratory only [RT 21].

### Null

Wrong-role null [RT 3]: permute the target-role mapping as whole role blocks within each question and recompute macro hit@5 (9,999 times).

### Pass rule per condition

1. Macro hit@5 lower bound ≥ 20% and permutation p < 0.01.
2. Coverage: ≥ 10 of 20 roles with role-level hit@5 above the role's own null 99th percentile.
3. Rules 1–2 hold after removing copied **and** self-naming replies [RT 18].
4. Attrition [RT 15]: invalid replies ≤ 10% of the condition, and a complete-cell sensitivity analysis (cells with all 5 wordings valid) gives the same decision.

### Condition contrasts (reported with crossed bootstrap CIs) [RT 12]

- `ex2 − desc`: effect of same-role examples.
- `ex2_other − desc`: effect of length plus conflicting examples.
- `ex2 − ex2_other`: aligned versus conflicting examples. This is **not** a pure length or description test.
- Target versus donor hit@5 in `ex2_other`: how far the examples override the description.

---

## Part 2: role-specific information in reply states

### Data

- Hidden state 17 of 33 is primary; 9 and 23 are exploratory.
- One vector per valid reply: the mean over the generated reply tokens.
- Main set: all valid replies on the 18 shared questions.

### Frozen decoding pipeline [RT 17]

- Classifier: multinomial logistic regression, L2, C = 1.0, no tuning, class-balanced. Standardization fitted on the training fold only.
- **Splits hold out questions and wordings together** [RT 9]: train on 3 wordings × 9 questions, test on the other 2 wordings × other 9 questions. 50 random splits. The score is the mean macro accuracy over splits. Chance = 5%.
- The same splits for every baseline.

### Measures

| Measure | Definition |
|---|---|
| **Cross-wording, held-out-question decoding (primary)** | As above. |
| Text baselines [RT 10] | (a) TF-IDF + logistic regression (fit on the training fold); (b) a sentence-embedding model + logistic regression; (c) length + stop reason only. |
| Incremental gain | Activation + text features versus text features alone, on the same splits. |
| Map reliability [RT 19] | Question-half role-distance matrices (cosine, mean-centred role vectors). Report raw Spearman r, the Spearman–Brown value, and a crossed-bootstrap CI. Same for the text-embedding geometry. |
| Geometry [RT 22] | Top 3 PCs per half: eigengaps and principal subspace angles (not only signed cosines). |
| Assistant Axis (`desc`) | Mean of `default_ai` and `default_llm` minus the mean of the role vectors. Split-half cosine. Limit: default prompts are shorter than role prompts. |

### Pass rule per condition

1. Primary decoding: lower bound ≥ 10% (2× chance) and permutation p < 0.01.
2. Incremental gain over the best text baseline: lower bound > 0. Otherwise report "role information that the reply text already carries".
3. Geometry is interpreted only if the raw split-half map r has a lower bound ≥ 0.5 and permutation p < 0.01. The axis is interpreted only if its split-half cosine is ≥ 0.80.

---

## Primary decisions and multiple tests [RT 14]

Four primary decisions: plays in `desc`, plays in `ex2`, information in `desc`, information in `ex2`. Each decision's p-value = the largest p-value of its rule components (an intersection–union test). Holm correction over the four. All other tests are exploratory.

## Stop rule

From the owner's spec: if Part 1 still fails on 14/10/2026, report the negative result and move to the trained stages. Part 2 stands on its own.

## Limits

- A 20-role pilot with selected roles and examples. No claim about all 276 roles.
- Power: 20 roles is small. If all primary decisions fail, report "not detected in this pilot", not "absent" [RT 20].

---

## Owner decisions (09/10/2026)

| # | Codex point | Decision |
|---|---|---|
| A | [RT 5, 6] Judge circularity | **A second judge from another model family** scores the same blind batches. Both judges are reported; Part 1 uses the stricter of the two for each pass rule. Human labels (owner, 200 replies) stay the validity check. |
| B | [RT 7] Weak bar for "plays the roles" | **The pass rule above stays fixed.** The wording of the claim ("plays some roles above chance" versus "plays the roles") is decided by the owner after seeing the data, and the report says so. |
| C | [RT 11] Length control | **Fixed.** `ex2_other` now uses 3 donors per role (shifts 6, 10, 14; always another group). New condition `ex2_filler`: a fixed neutral interview-notes text in place of the examples. Mean prompt tokens on the shared questions: ex2 231.6, ex2_other 231.6, ex2_filler 234.1. Add contrasts `ex2_filler − desc` (length only) and `ex2 − ex2_filler` (same-role examples beyond length). |
| D | [RT 13] One sample per prompt | **Fixed.** Second-seed replication (`--replicate --seed_offset 100000`): desc and ex2 on the 18 shared questions, plus the 3 default frames (for the axis). 3,654 items. Report the agreement of every primary measure between seeds. |

## Changes

- 09/10/2026: v2 after the Codex red-team.
- 09/10/2026: owner decisions A–D recorded; the full pilot run started after this, before any result was seen.
- 09/10/2026, while writing `part2_analysis.py`, before any Part 2 number on real data:
  - **Map reliability halves now split questions AND wordings** (half A: half the questions × 2 or 3 wordings; half B: the rest). A synthetic self-test showed that question-only halves pass on pure noise when each role has its own wording offsets (r 0.175 vs null p99 0.065); with the new halves noise gives r −0.01 (null p99 0.035) and a planted signal gives r 0.62.
  - **Decoding permutation test:** the test statistic is the mean macro accuracy over the first 10 of the 50 fixed splits, for the observed data and for each of the 9,999 block permutations (the full pipeline is refit each time). The reported estimate stays the 50-split mean. Reason: compute (≈10^5 refits per condition).
  - **Decoding CI:** crossed bootstrap of held-out (role, question) cell accuracies pooled over the 50 splits, without refitting (resampling roles would duplicate classes).
  - Sentence-embedding baseline: `sentence-transformers/all-mpnet-base-v2` @ `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
- 16/10/2026, after the Part 1 naming results and two rounds of broken-reply judging were seen:
  - **Definition of a broken reply (owner, approved 16/10):** "Broken: the reply does not work as an answer to the question. That means it loops (the same idea again and again, adding nothing), it does not make sense, it writes new interview turns or copies the question, or it ignores the question and talks about something else, even in character. Not broken: cut off by the length limit, a phrase repeated for emphasis, or an answer that is wrong, odd, rude, short, or in a strange voice." It combines the coherence-against-the-question judges of Betley et al. 2025, Chen et al. 2025 and Moskvoretskii et al. 2026 with this pilot's loop, format and cut-off rules.
  - It replaces "token-limit stop or a 4-gram repeated 3+ times" as the meaning of broken. It was set after the judges' disagreements with the owner's 80 labels were seen, so any judge validated on those labels is exploratory. Part 1 decisions do not change under any count of broken replies tried so far (`results/naming_sealed_v1.json`).
- 16/10/2026, Part 2 context controls (owner approved the draft and the all-layer sweep; Codex adversarial review `review/codex_review_part2_context_261016.md` folded in). Written before any Part 2 number on real data. The four primary decisions and the frozen Part 2 rules do not change.
  - **Why:** reply-token states attend to the prompt. Every wording names the role, and the `ex2` examples are the same text in train and test. A pass can come from the context alone. The frozen text baselines read only the reply, so rule 2 cannot detect this. The primary measure is therefore reported as **context-conditioned role decoding**.
  - **New features** (`context_extract.py`; one GPU pass per feature, no generation; same model, revision, dtype and right padding as `gen_extract.py`; all 33 hidden states; usable shared-question rows of `run_main` and `run_rep`):
    - C0: prompt ids + kept reply ids, mean over reply positions (re-computes the stored reply state).
    - C0-first32: the same pass, mean over the first min(32, n) reply positions (length sensitivity).
    - C1: the same pass, state at index prompt_length − 1 (the `Guest:` token).
    - C2: one `<|endoftext|>`, the header, the test question, `Guest:`, then the same kept reply ids. The description, all example turns and the filler notes are removed (in `ex2_other`, both the target description and the donor examples). Mean over reply positions.
    - C3: the full prompt, then one fixed neutral continuation (the same text for every item). Mean over the continuation positions. It shows what the prompt context alone gives in reply-like pooling.
  - **C0 gate**, per run, condition and state 9 / 17 / 23, on every row: row ids, prompt token count and reply ids match the stored run; after subtracting the stored condition mean, cosine(C0, stored) has minimum ≥ 0.95 and median ≥ 0.99; median relative norm difference ≤ 0.02. If the gate fails, C1–C3 are not reported as controls.
  - **Analysis** (`part2_context.py`; the same classifier, 50 splits and bootstrap draws as `part2_analysis.py`; no new permutation null): target-role decoding at state 17 from the stored reply state (R), C0, C0-first32, C1, C2 and C3; a joint TF-IDF + length + stop-reason baseline; paired R − C2 ("accuracy difference after context removal") and R − C3. `ex2_other`: target and donor decoding, plus **within-donor target accuracy** and **within-target donor accuracy** (the 20-way probabilities restricted to the 3 roles linked to the true donor or target; chance 1/3), because each donor links to 3 targets. Sweep: all 33 states for C0, C1, C2 and C3 on the first 10 splits, point estimates only.
  - **Wording rule:** C2 lower bound ≥ 10%: "the role can also be decoded from the reply text alone, as the model reads it". Otherwise: "C2 did not meet the 10% floor", with the full interval and no claim about the source. C3 lower bound ≥ 10%: "the prompt context alone carries decodable role information". A pass is never described as the model playing the role; that is Part 1.
  - **Replication:** `part2_analysis.py --run run_rep --conds desc ex2` with the frozen settings; compare every primary measure with `run_main`.
  - **Report:** the four raw decision p-values (Part 1 plays, Part 2 information) are merged and Holm is applied once (`part2_context.py report`). Condition runs on separate machines write separate output folders.
  - **Loader:** `part2_analysis.py` may read a slim copy of the stored reply states (hidden states 9, 17, 23 only, with its SHA-256 and the source file's SHA-256 recorded). This changes no number. `--embed_only` writes the embedding cache and stops.
  - **Limits:** C2 moves the reply to earlier positions and removes the context it was written in; it shows what the model reads from the reply text, not what it computed while writing. C1 and C3 will probably be close to 100%, so accuracy has no room above them. Map reliability can reflect stable prompt differences; the axis is a default-frame contrast, and cosine ≥ 0.80 shows repeatability, not assistant function.
- 16/10/2026 17:35, after the context-control results were seen: the report wording for `ex2_other` also prints the pre-registered within-donor target and within-target donor accuracies for R and C2, and says whether target information beyond the donor link was detected (lower bound > 1/3). No decision depends on this line; it stops the 20-way C2 sentence from being read as target information.
