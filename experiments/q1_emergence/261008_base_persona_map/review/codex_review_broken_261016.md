**Verdict:** Partial finish. Keep the labels provisional; the promised reason and uncertainty report is incomplete. Three blockers remain.

---

**Blockers**

**1. The code discards stated owner reasons.**

[finish_broken_261016.py:264](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/finish_broken_261016.py:264) assigns `owner_broken_reason_unstated` when the owner says Broken and GPT says Fine.
For `run_main/12269`, the owner explicitly selected `nonsense`; see [owner_labels_fresh30_2026-10-16.json:93](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/review/attrition/owner_labels_fresh30_2026-10-16.json:93).
Five of the 12 unresolved rows have explicit `loop` or `nonsense` tags: 11420, 12269, 3815, 6185, and 9093.
Read and preserve owner reason tags before using the fallback; disclose any remaining reason audit as incomplete.

---

**2. Fresh-sample agreement hides the important misses.**

[finish_broken_261016.py:366](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/finish_broken_261016.py:366) reports pooled confusion counts, but omits condition/flag tables and weighted confusion counts.
GPT finds only 1/4 owner-Broken ex2 replies and misses both unflagged positives, `run_main/12231` and `run_main/12269`.
The unflagged stratum contains 1,611/1,749 frame replies, so the pooled 26/30 accuracy conceals an important weakness.
Add all four strata and their population weights; keep the original failed gate unchanged.

---

**3. The required paired contrasts are absent.**

[finish_broken_261016.py:297](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/finish_broken_261016.py:297) draws a separate bootstrap sample for each group; the report contains no contrast intervals.
For main ex2 minus desc, my 2,000 paired draws (seed 20261016) give −1.78 percentage points; conditional 95% interval: −7.95 to +5.89.
This does not establish that examples reduce Broken replies.
Add condition and seed contrasts with shared role/question draws, and state that classification error remains outside these intervals.

---

**Other issues**

| Item | Finding |
|---|---|
| Pure GPT comparator | Missing fragment-adjusted GPT counts: 130 desc and 102 ex2, versus final counts 137 and 105; raw GPT confounds fragment changes with owner substitutions ([code:314](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/finish_broken_261016.py:314)) |
| Provenance | Unexplained round-3 source-version differences; missing hashes for keys, reply files, owner inputs, and final outputs ([manifest:8](/Users/brendon/persona-manifolds/e3_broken_finish/manifest.json:8)) |
| Fragment reporting | Missing directions and rate changes: 39 Broken→Fine, one Fine→Broken; `run_rep/4588` becomes `loop`, a separate change from format removals |
| Report limits | Missing explicit limits: selected roles/questions, fixed donors, two seeds, four groups without owner evidence; conditional classification intervals |
| Reason presentation | Missing Fine counts, percentages, and evidence excerpts |

---

**Deviations from the plan**

| Deviation | Judgment |
|---|---|
| 16 workers | Accept. All 291 batches are valid; the judging method stayed fixed. |
| Separate derived batches | Accept. Raw replies and derived copies never share a batch. |
| GPT reasons or an unresolved fallback | Accept the refusal to invent reasons. Reject ignoring existing owner tags: blocker 1. |
| Recovered fresh frame | Accept. I reproduced the exact ordered 30-item draw from the page generator's rule. |

---

**The owner estimate and the judges**

The ex2 estimate, 28.3% with conservative 95% interval 2.6–74.5%, is correct for the 1,749-reply frame.
It excludes 51 replies; it does not estimate all 1,800 without an additional assumption.
On that same frame, raw Opus, GPT, and Jev rates are 4.7%, 5.6%, and 10.5%; all fall inside the owner interval.
This sample cannot determine which rate is close to the owner-defined population rate.

---

**Why the owner estimate is unstable**

Both unflagged positives are toddler answers to question 200, with different wordings; each changes the estimate by 13.2 percentage points.
In [run_main/replies.jsonl:12270](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/run_main/replies.jsonl:12270), item 12269 gives a coherent fort-building story about accepting feedback.
I cannot defend its `nonsense` tag from that text; the relevant dispute is whether its indirect answer meets the question.
Preserve the owner label, but resolve this boundary before treating the estimate as evidence that the judges greatly undercount failures.

---

**The translator role**

Translator replies contribute 53/105 final Broken calls in main ex2 and 50/89 in replication ex2; all use `format`.
[run_main/replies.jsonl:12957](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/run_main/replies.jsonl:12957) translates the emergency-savings question and repeats it in English, without answering it.
That fits the approved question-copy rule; the owner's Broken label on 12974 supports this treatment.
Report this role concentration: these calls measure answer failure, and do not establish unintelligible language or failed role enactment.

---

**What holds**

| Promised review check | Result |
|---|---|
| Definition | Pass: saved prompts preserve the owner rule and its exclusions. |
| Coverage | Pass: exact canonical keys; six complete 20 × 18 × 5 grids. |
| Provenance | Inputs verified; documentation incomplete as noted above. |
| Judge extension | Pass: same model, high effort, prompt, and schema; blind identifiers. |
| Combination | Pass: all 80 owner labels preserved; raw judge comparisons use identical text. |
| Fragment handling | Pass: all 51 suffixes verified; preceding text and raw calls preserved; six derived calls are Broken. |
| Agreement | Original gate reproduced: every judge and the majority fail. Reporting needs blocker 2. |
| Sampling design | Pass for the restricted frame; 8/7 strata preserved; known owner50 counts added separately. |
| Uncertainty | Individual crossed intervals and hypergeometric intervals reproduce. Contrasts need blocker 3. |
| Reasons | Totals reconcile to 782 Broken replies; owner reason recovery fails blocker 1. |
| Claim scope | Provisional status is clear; explicit limits need completion. |
| Completion | Local rebuild passes; full report remains incomplete. No validated exclusion mask was produced. |

---

**Verification**

I rebuilt `labels.jsonl` and `report.json` in memory; both matched byte for byte.
I checked all saved GPT/Opus batch inputs and outputs, and all 10,800 Jev request hashes against source text.
Recorded GPT tool reads contain only the generic skill file: four reads in the extension and four in round 3.
I made no model calls or HTTP requests, and wrote only this review.

---

**What the owner must decide — Next step**

1. Set the relevance boundary for related anecdotes, especially 12231 and 12269; retain all original labels in the audit.
2. Keep translation-only replies Broken under the current rule, or record a prospective definition change across every condition.
3. Accept a partial provisional release, or require the three local corrections before calling this finish complete.
