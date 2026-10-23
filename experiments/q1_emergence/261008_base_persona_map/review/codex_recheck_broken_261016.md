**Result:** One blocker remains. Blocker 2 is NOT FIXED. Automatic labels remain provisional.

---

**Blockers**

| Blocker | Status | Evidence |
|---|---|---|
| 1. Owner reasons | FIXED | [Code:270](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/finish_broken_261016.py:270) reads owner tags before GPT reasons. All 80 owner labels remain unchanged. |
| 2. Fresh30 agreement | NOT FIXED | Four strata and frame weights are present. Weighted confusion counts remain absent. |
| 3. Paired contrasts | FIXED | [Code:309](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/finish_broken_261016.py:309) shares 2,000 role/question draws across groups. All six contrast intervals reproduce independently. |

---

**Owner reasons**

The map is `loop→loop`, `nonsense→nonsense`, `echo→format`, and `babble→off_question`.
Rows 11420, 3815, 6185, and 9093 now use `loop`; 12269 uses `nonsense`.
The [reason audit](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/results/broken/finish_261016/report.json:1049) records 18 owner-tag reasons and five unresolved reasons; the reason audit remains incomplete.

---

**Paired contrasts**

Main ex2 minus desc is −1.78 percentage points; its conditional 95% interval is −7.95 to +5.89.

---

**Blocker 2: missing weighted confusion counts**

The [four stratum tables](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/results/broken/finish_261016/report.json:825) expose both missed unflagged ex2 positives.
The [plan:162](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/review/codex_advice_broken_261016.md:162) also requires weighted confusion counts; those counts remain absent.
Applying `sum(frame_N × cell_count / sample_n)` gives these estimates for the 1,749-reply ex2 frame.

| GPT ex2 basis | True positive | False negative | True negative | False positive |
|---|---:|---:|---:|---:|
| Sample: 15 replies | 1 | 3 | 11 | 0 |
| Frame estimate | 17.25 | 477.54 | 1,254.21 | 0 |

Estimated frame sensitivity is 3.49%; sample sensitivity is 25%.

---

**Other issues**

| Issue | Status | Evidence |
|---|---|---|
| Pure GPT comparator | FIXED | [Report:43](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/results/broken/finish_261016/report.json:43) adds fragment-adjusted counts. Main desc/ex2 counts are 130/102, versus final counts 137/105. |
| Provenance | NOT FIXED | Source hashes overwrite each other. Both judging keys and the current report-builder source lack hashes. |
| Fragment reporting | FIXED | All six rate changes reproduce. Directions total 39 Broken→Fine and one Fine→Broken. `run_rep/4588` changes from Fine to `loop`. |
| Report limits | FIXED | [Report:4](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/results/broken/finish_261016/report.json:4) states role/question selection, fixed donors, two seeds, four groups without owner evidence, and conditional intervals. |
| Reason presentation | NOT FIXED | Fine counts and shares are present. Each group totals 1,800. Required evidence excerpts remain absent; [code:343](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/finish_broken_261016.py:343) produces counts and shares only. |
| Role concentration | FIXED | Translator counts are present. Their updated reason split appears below. |
| Owner relevance boundary | NOT FIXED | These outputs record no boundary resolution for 12231 and 12269. Both retain the owner's Broken label. |

---

**Provenance evidence**

[Code:454](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/finish_broken_261016.py:454) uses `p.name` as the hash key.
The replication reply hash replaces the main reply hash; the round-3 manifest hash replaces the finish manifest hash.
The finish and round-3 judging keys are absent, as is the current builder hash.
The label hash matches, and current prompts rebuild every saved batch exactly; source-version notes are now present.

---

**Translator correction**

Translator replies contribute 53/105 main ex2 Broken calls and 50/89 replication ex2 calls: 50.5% and 56.2%.
Main now has 52 `format` reasons and one owner `loop` reason at [12974](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/results/broken/finish_261016/labels.jsonl:1558).
Replication has 50 `format` reasons.
My earlier statement that all main translator reasons were `format` no longer applies.

---

**Verification**

| Check | Result |
|---|---|
| Saved outputs | Both files match the rebuild in memory, byte for byte. |
| Coverage and labels | Exact 10,800 keys; six complete grids; 80 owner labels preserved; 782 Broken calls. |
| Fragments and intervals | All 51 suffixes checked; all six paired contrasts independently reproduced. |
| Saved judging inputs | All 435 batch prompts, 579 answer records, and 10,800 Jev request hashes pass. |

I used exact integer hypergeometric sums to avoid SciPy's temporary file writes.
I made no model calls or HTTP requests, and wrote only this review.

---

**Next step**

Add weighted confusion tables to close blocker 2.
Use distinct source paths for hashes, and add the required evidence excerpts.
