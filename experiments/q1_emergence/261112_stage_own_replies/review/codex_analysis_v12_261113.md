**Fix first. I found rule mismatches and reporting errors; I did not establish corruption of the primary saved numbers.**

The permitted CPU tests passed: **11 passed, 2 warnings**. No Modal or GPU code ran, and I changed no files.

---

**1. Medium — Full-view filtering uses SFT’s validity flags for every stage**

`prep()` builds `meta` from SFT, including `valid`; `source()` then uses those flags for DPO and Instruct too ([analysis_v12.py:460](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:460), [analysis_v12.py:521](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:521)). The reference profile instead uses the intersection of all stages’ valid flags ([analysis_v12.py:458](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:458), [analysis_v12.py:485](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:485)).

This can discard valid replies from another stage, include its invalid replies, or make the reference and source disagree. **It does not affect this recorded run:** I independently counted 21,000 valid rows in each stage, consistent with [prep.json:37](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/results/analysis_v12/prep.json:37).

**Smallest fix:** save validity flags per stage; use each stage’s flags for full replies and Instruct’s flags for its reference.

---

**2. Medium — Masked Lu values use different weighting from the primary PC1**

The primary role means weight retained rows equally through `CR.means` ([check_role_count.py:83](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_free_checks/check_role_count.py:83)). Lu first averages rows within each question, then weights question means equally ([analysis_v12.py:388](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:388), [check3_lu_scale.py:25](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_free_checks/check3_lu_scale.py:25), [analysis_v12.py:393](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:393)).

With unequal retention across questions, these produce different role means and potentially different PC1s; the masked Lu result is therefore **not simply the primary PC1 expressed on another scale**. The balanced, unmasked full-view Lu values are unaffected.

**Smallest fix:** disclose the equal-question weighting; if the intended axis is the primary PC1, form its means with `CR.means` before calling Lu’s scaling helper.

---

**3. Medium — “Almost the same role map” is unsupported**

NOTE and the vault call DPO and Instruct almost the same map ([NOTE.md:130](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/NOTE.md:130), [2026-W46.md:173](/Users/brendon/research-log/manifold-persona-log/Weeks/2026-W46.md:173)). The cited correlation compares only **20 PC1 role scores**, not the complete role-distance maps ([analysis_v12.py:273](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:273), [analysis_v12.py:288](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:288)).

DPO’s reported PC1 share is only 0.327, so the correlation leaves most role variance untested ([dpo_full_s17.json:54](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/results/analysis_v12/sources/dpo_full_s17.json:54)). The within-stage D2 gates do not establish cross-stage map agreement.

**Smallest fix:** replace “almost the same role map” with “almost identical full-view PC1 role profiles.”

---

**4. Medium — The write-up turns a coarse power grid into a hard detection limit**

“Changes under 1.5 role SD cannot be seen” overstates P3 ([NOTE.md:133](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/NOTE.md:133), [2026-W46.md:162](/Users/brendon/research-log/manifold-persona-log/Weeks/2026-W46.md:162)). The saved rates are 3.4% at 1.0 and 94.7% at 1.5; intermediate sizes were not tested ([power.json:1067](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/results/analysis_v12/power.json:1067)).

The vault summary also omits qualifications retained in NOTE: first16 text fails P1, and SFT full spread loses its “wider” interval on the common mask ([NOTE.md:89](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/NOTE.md:89), [NOTE.md:124](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/NOTE.md:124)). That masked SFT interval is −0.031 to 0.077 ([results_v12.json:894](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/results/analysis_v12/results_v12.json:894)).

**Smallest fix:** say “1.5 was the smallest tested shift reaching 80% simulated rejection”; include both qualifications in the summary.

---

**5. Low — The power artifacts do not guarantee blinding**

Each source saves real point estimates, Lu positions and other results before `power()` runs ([analysis_v12.py:416](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:416), [analysis_v12.py:539](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:539)). Furthermore, `power.json` contains both signed and absolute position changes ([analysis_v12.py:584](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:584)).

Absolute changes depend on the real position: a shift crossing zero can reveal its magnitude, while the matching signed/absolute changes here reveal information about its sign. **I found no exposed real stage-difference magnitude in the median-centred P3 output itself**, and no evidence that anyone inspected the source estimates early.

**Smallest fix:** release only power flags and centred P3 results before measurement; withhold source estimates and absolute responses until afterward.

---

**6. Low — P3 reports an undocumented worst-direction rate**

For each positive grid magnitude, P3 simulates both signs, counts only rejection in the planted direction, and reports the **minimum** rate ([analysis_v12.py:566](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:566)). The frozen rule specifies the rejection rate under the stated change rule; it does not specify this minimum over directions ([THRESHOLDS.md:95](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/THRESHOLDS.md:95)).

This is a conservative sensitivity summary, but it is a different reported quantity, and it conditions on gates rather than simulating them. Median centring and preservation of paired full/first16 errors are otherwise consistent with the specified bootstrap approximation.

**Smallest fix:** report both directional rates; label their minimum explicitly and apply the actual change-reading rule.

---

**7. Low — Resume checks do not bind outputs to one complete run**

The launcher skips a source whenever its JSON exists ([modal_analysis_v12.py:142](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/modal_analysis_v12.py:142)). JSON is written before the non-atomic NPZ write, and later loading checks code hashes without binding that pair to input, mask, draw or runtime hashes ([analysis_v12.py:539](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:539), [analysis_v12.py:544](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:544)).

A missing or truncated NPZ will usually stop loading; a readable stale NPZ or changed input can escape these checks. **I found no demonstrated mixture in this run.**

**Smallest fix:** write arrays atomically, hash them, and resume only from a verified JSON/NPZ pair with a complete run fingerprint.

---

**8. Low — Gate enforcement and numerical draw accounting are incomplete**

The reading gate is applied to primary confirmatory results, while state-minus-text results are emitted without checking both sources’ gates ([analysis_v12.py:673](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:673), [analysis_v12.py:710](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:710)). Separately, empty-cloud draws are counted, but `interval()` silently removes any other non-finite values without adding them to the dropped count ([analysis_v12.py:340](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:340), [analysis_v12.py:118](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:118)).

All recorded primary and text map gates pass, all recorded profile correlations exceed 0.5, and recorded dropped shares are zero. Thus I found no demonstrated changed reading from these defects here.

**Smallest fix:** propagate gate flags into report-only comparisons; reject or count numerical failures and report paired draw counts.

---

**9. Low — Han’s per-role change output is incomplete**

The rule requests projection changes per role, but `change_per_role` contains only the ten labelled roles—or eight after exclusions ([THRESHOLDS.md:116](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/THRESHOLDS.md:116), [analysis_v12.py:640](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:640), [results_v12.json:2073](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/results/analysis_v12/results_v12.json:2073)). All-role raw projections are saved, so the missing changes remain recoverable.

Also, disappearance after normalization is **consistent with** a scale explanation; it does not establish that explanation as NOTE and the vault state ([NOTE.md:127](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/NOTE.md:127), [2026-W46.md:171](/Users/brendon/research-log/manifold-persona-log/Weeks/2026-W46.md:171)).

**Smallest fix:** emit changes for all 20 roles separately from group testing; soften the scale attribution.

---

**10. Low — The endpoint test includes equality**

The frozen criterion says **beyond** two role SDs; the implementation accepts exactly two through `>=` and `<=` ([THRESHOLDS.md:104](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/THRESHOLDS.md:104), [analysis_v12.py:603](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:603)). This does not change the saved readings.

**Smallest fix:** use strict comparisons and add boundary cases to the existing reading test.

---

**Checks with no additional finding**

| Check | Result |
|---|---|
| Primary position and spread formulas | **None found.** They call the imported statistics with the specified split estimate and within-wording log-ratio ([analysis_v12.py:315](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:315)). |
| Shared, paired draws | **None found.** One saved draw set supplies every source; differences subtract corresponding draws ([analysis_v12.py:491](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:491), [analysis_v12.py:700](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:700)). |
| Signed orientation | **None found.** Instruct’s assistant anchor, cross-source profile correlation and split/draw vector alignment implement the stated convention ([analysis_v12.py:280](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:280), [check_role_count.py:98](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_free_checks/check_role_count.py:98)). |
| Shift and scale algebra | **None found.** P1 is an affine-response engineering check, nearly automatic when axes and denominators are stable; it does **not** establish detection power ([analysis_v12.py:294](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:294)). |
| Row keys and common mask | **None found.** I independently checked all three ordered key lists against `keys.json` and recounted the 19,100-row common mask; minimum cloud/wording support is 71 ([prep.json:45](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/results/analysis_v12/prep.json:45)). |
| First16 token span | **None found.** Text decodes `kept_ids[:16]`; extraction pools precisely those 16 token positions ([analysis_v12.py:474](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:474), [length_control.py:45](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:45)). |
| Map mask and strict summary | **None found.** Masking through `valid` reaches pod A’s cell function; its strict summary rejects undefined map draws ([analysis_v12.py:369](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:369), [pod A analysis.py:105](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/analysis.py:105)). |
| Han labels, axis, units and permutation | **None found**, apart from the reporting omissions above: correct labels, default-facing unit axis, separate stage SD normalization, and exhaustive 252/70 splits ([analysis_v12.py:627](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:627)). |
| Test family and combined readings | **None found**, except equality at the endpoint: ten tests, 0.25th/99.75th percentiles, paired changes and both-view agreement ([analysis_v12.py:118](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:118), [analysis_v12.py:678](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12.py:678)). |
| NOTE’s primary numerical tables | **None found.** Their rounded points, intervals, test numbers and readings match `results_v12.json` ([NOTE.md:97](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/NOTE.md:97)). |

---

**Provenance and verification limits**

All six recorded analysis-code hashes match current files, and all **33 local JSON files** match the analysis upload receipt ([prep.json:3](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/results/analysis_v12/prep.json:3), [receipt:4](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_modal_gpu/hf_receipt_261113_analysis.json:4)). I found no logged failure or rerun explaining `instruct_full_m16_s17`; its 1,397.8 seconds alone is **not evidence of a corrupt result** ([power log:194](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/analysis_v12_power.log:194)).

The five absent input arrays and absent output draw arrays prevent independent reconstruction of the saved intervals, numerical draw failures and P3 rates. Receipt agreement proves file identity, not statistical correctness.

**Verdict: fix first.**