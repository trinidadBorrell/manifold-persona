**Verdict:** Fix two blockers before code: D2 can certify identical maps, and F3 can reject correct canary data.

---

**Source references**

Paths start at `experiments/q1_emergence/`. Each alias names one file; `C:71` means that file, line 71.

| Alias | File |
|---|---|
| D | `261029_stage_persona_plan/DESIGN.md` |
| A | `261029_instruct_persona_map/THRESHOLDS.md` |
| B | `261029_base_role_vs_format/THRESHOLDS.md` |
| C | `261029_fixed_text_read_through/THRESHOLDS.md` |
| N | `261029_instruct_persona_map/e3_noise_check.py` |
| NOTE | `261029_instruct_persona_map/NOTE.md` |
| JSON | `261029_instruct_persona_map/results/e3_noise_check.json` |
| E | `261008_base_persona_map/part2_analysis.py` |
| R1 | `261029_stage_persona_plan/review/codex_review_v2_r1_261029.md` |
| R2 | `261029_stage_persona_plan/review/codex_review_v2_r2_261029.md` |

---

**Blockers**

**B1. [OWNER] D2 can pass with identical complete role maps. Its warning does not repair its decision.**

**Evidence.** C:70–71 still declares a transition from a positive reliability-normalized difference.
C:73 acknowledges the identical-map counterexample, but C:74 only flags disagreement with question-only halves.
G5 records this bias without making it a failed check (C:47).
Thus the rule can certify the specific defect its own control detects.

**Concrete failing case.** Draw `S`, then `E`, as independent 20 × 32 standard-normal arrays with NumPy RNG seed 31.
Let `k = [-3, -3, 2, 2, 2]` index the five wordings.
Set base, DPO and final Instruct vectors to `S`; set SFT vectors to `S + 0.2*k[w]*E`.
Repeat each vector across all 40 questions and all five replicates.

Every complete role vector is `S`, because the five wording effects average to zero.
I used N:74–83's 50 wording halves and N:62–71's centred cosine-distance/Spearman map statistic.
Question resampling has no effect in this case.
The bootstrap used E:368–389's distinct-role centring and repeated-pair weights, with 2,000 role draws and RNG seed 2026.

| Quantity | Offline result |
|---|---:|
| Base and Instruct reliability | 1.000000 |
| SFT reliability | 0.789476 |
| SFT reliability 95% interval | [0.692072, 0.848780] |
| Base normalized agreement with Instruct | 1.000000 |
| SFT raw agreement with Instruct | 0.933319 |
| SFT normalized agreement with Instruct | 1.050414 |
| Base → SFT difference 95% interval | [0.031528, 0.076995] |
| Bootstrap decision p | 0 / 2,000 |
| Question-only difference | 0 |

Both reliability floors pass, and every bootstrap reliability is positive; the smallest is 0.609047.
No draw was removed.
The specified bootstrap p remains zero after Holm correction (C:36,71,78).
D2 therefore passes although no complete role map changed.

**Why this matters.** The revised report-only interpretation correctly removes the earlier false change claim (C:85–86; R2:68–97).
The same correction still drives a primary pass decision (C:68–74).
A limitation paragraph cannot make that decision valid.
This is a new D2 finding, rather than a repeat of the repaired report-only wording.

**Exact proposed fix.** The smallest defensible fix is to make D2 report-only and remove its three transitions from Holm.
Keep D1's decision, both half schemes, all raw values, and the current normalized trajectory.
Replace the pass sentence with: “No transition receives a pass from this statistic; checkpoint-specific wording sensitivity can cause positive differences.”
Keep G5's equal-map case as a required regression case that must never produce a transition pass.

This changes when D2 passes, so it needs the owner.
If a transition pass remains required, specify and calibrate a different test before implementation.
Requiring question-only confirmation would reject this example, but does not validate the normalization under other checkpoint-specific half effects.
Do not claim that this one sensitivity establishes a general noise correction (C:72–74).

---

**B2. [TEXT] F3's corrected class count still permits a false canary failure after unequal reply loss.**

**Evidence.** Pod A permits 95% valid replies per role (A:35).
Pod C takes all valid source rows (C:16), then requires neutral F3 decoding within 0.5 percentage points of chance (C:55).
The canary uses four non-default roles (D:166).
The text does not require a separate balanced F3 grid after source-row filtering (C:16–24,55).

**Concrete failing case.** Give every question a single neutral state, identical across roles, wordings and replicates.
Use state `-1` for questions 0–19 and `+1` for questions 20–39.
Make all replies invalid for role 0 on questions 0 and 1, and role 1 on questions 20 and 21.
The two other roles keep all replies; affected roles retain exactly 95%.

This meets A:35, and F3 state differences within each question are exactly zero.
I used the specified standardized, class-balanced logistic classifier (E:175–187).
Across 50 question-and-wording splits from RNG seed 0, mean macro accuracy was 26.1186%.
The allowed canary range is 24.5%–25.5%, so correct neutral inputs fail C:55.

A simpler split trains on even question IDs and wordings 0–2, then tests the remaining IDs and wordings.
It gives 26.3158% macro accuracy.
Selective missingness changes the question distribution within each role, without changing any F3 input for that question.
Changing `1/20` to `1/K` repairs R2's class-count defect, but does not repair this balance defect (R2:29–47; C:55).

**Exact proposed fix.** Build F3 independently on the complete frozen role × question × wording grid.
Include neutral rows even when the corresponding generated reply is invalid or unjudged.
Give every role the same question and wording weights in each train and test fold.
Keep the current `1/K` chance value and 0.5-point tolerance.

F3 requires neutral text and question IDs, so source reply validity need not remove its rows (C:20,22).
Add the 95%-valid counterexample above to the offline four-role acceptance test (D:55–56).
Assert balance before fitting, and report an invalid control grid separately from role leakage.
This changes the control population, not an owner threshold.

---

**Other issues**

**O1. [TEXT] The F4 support fix checks point estimates, but not bootstrap samples. Remaining B4 boundary case.**

**Evidence.** C:24 requires at least one common question in each fixed half before extraction.
C:32 then resamples questions with replacement.
C:52 defines what to do with non-positive reliability, but not a map missing its required rows before reliability exists.
The reused estimator raises when a role half has no rows (E:302–307).

**Concrete failing case.** Take a fixed half with questions 0–19 and wordings 0 and 1.
For each question 1–19, make one different role's replies too short for F4 in those two wordings only.
Keep all other replies long enough, including question 0 and the entire other half.
Each affected role retains 99% of its full rows, but question 0 is that half's only common question.

The coverage and point-support rules both pass (C:23–24).
A 40-question bootstrap omits question 0 with probability `(39/40)^40 = 0.363232`.
Such a draw has no supported question in that half.
An implementation can crash or silently condition its interval by discarding these draws.

**Exact proposed fix.** Extend the support check to every paired bootstrap draw before calculating any map statistic.
If any draw has no supported half, report that sensitivity's interval, normalized comparison and pass outcome as unavailable.
Do not delete draws, replace halves, or silently change the question population.
Keep F1 on the same F4 rows, and apply this rule to the identifiable subset too (C:16,24).

This supplies a defined unavailable outcome without changing the accepted 50% support value.
Add the sparse-common-question case to the existing offline half-support tests (D:56).
The point-support addition answers R2's original empty-half example; it does not cover this bootstrap boundary (R2:101–119).

---

**What you verified**

I compared the current rules with round 1 and round 2, including all eight round 2 findings (R1:16–327; R2:25–208).
Only new defects and remaining boundary cases appear above.
The plan is still pre-code, so I did not report the planned implementation work as a new defect (D:3,163–165).

| Check | Evidence and result |
|---|---|
| Saved noise artifacts | Both activation hashes matched N:36–43's checks. Each condition/run had a finite 20 × 5 × 18 × 4,096 grid under N:46–58. |
| Noise output | Every saved condition field and all 50 halves reproduced exactly. Computation: N:118–145; saved results: JSON:891–971. I imported functions without calling the file-writing `main()` at N:146–148. |
| Noise interpretation | The note limits its conclusions to the two base runs and removes the precision forecast (NOTE:7,32–39). |
| Rows and replicates | A has 21,000 source rows, including 20,000 non-default rows (A:15,18). B has 24,000 rows and three replicates (B:18). C inherits five replicates (C:16). |
| Judge panel | The three-replicate option has 12,600 source rows and 12,000 non-default rows. D1 uses the approved panel, including its precision target (A:87–88,111–113). |
| Layers and maps | All three files use hidden-state index 17, with embeddings at 0 (A:17; B:20; C:26). Map centring is explicit (A:58; B:70; C:34). |
| Nulls | Map tests permute one map or use a joint format null (A:59–62; B:70–71; C:54,64). Decoding has a separate within-question block null (B:62). |
| D2 false pass | The equal-complete-map construction above passes the stated reliability and transition rules (C:52,70–71). |
| Canary F3 | Correct neutral inputs exceeded the allowed range after permitted source-row loss (A:35; C:16,55). |
| Filtered support | The added common-question rule can still leave bootstrap halves undefined (C:24,32). |

The dimensions, replicate counts and primary layer conventions agree across the three threshold files and the design (D:61,87,101,118,138).
The synthetic D2 failure and F3 balance failure are the material exceptions to usable pass rules.
The F4 boundary affects a sensitivity, so it is listed separately.

I used small CPU checks with `.venv/bin/python -B`, saved local arrays and local JSON.
I used no network, GPU, paid model or model API.
I did not test checkpoint availability, package installation, GPU numerics or restore behavior.
The design already requires those engineering checks before pod allocation (D:150–152,165).
