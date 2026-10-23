**Verdict:** A Part 2 pass cannot show that the base model plays the persona. The draft still overclaims.

---

**Blockers**

**1. The probe can identify prompt context.**  
[gen_extract.py:106](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/gen_extract.py:106), [part2_analysis.py:112](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/part2_analysis.py:112): splits separate questions and wordings, but retain role names and identical examples.  
**Failing case:** Generic replies can pass while reply-only text baselines fail.  
**Fix:** Report “context-conditioned role decoding”; add C1 and an identical neutral continuation under every prompt.  
Activation gain shows improvement over the tested baselines; it does not establish information beyond reply text.

**2. “Valid” does not mean a valid answer.**  
[gen_extract.py:271](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/gen_extract.py:271): validity means nonempty text; length cuts remain included.  
**Failing case:** Different reply lengths, stopping patterns, and extra speaker turns change the pooled states.  
Exact `\nInterviewer:` markers are already cut; variants can survive.  
**Fix:** Keep the frozen estimate, report these rates, and add matched-length and first-answer sensitivities with a joint text/length/stop baseline.

**3. C0 can pass after the role signal disappears.**  
[Draft:26](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/review/part2_context_controls_draft_2026-10-16.md:26): raw cosine alone is insufficient.  
**Failing case:** My synthetic test removed every role difference and still obtained cosine 0.99995.  
**Fix:** Check row identities, token boundaries, norms, and numerical error after centring, per condition and used layer.  
The percentile gate also permits a small group of bad rows.

**4. The C2 wording rule makes unsupported source claims.**  
[Draft:35](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/review/part2_context_controls_draft_2026-10-16.md:35), [draft:43](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/review/part2_context_controls_draft_2026-10-16.md:43).  
**Failing case:** A C2 interval of 8–30% misses the 10% floor but does not establish absence.  
**Fix:** Say “did not meet the C2 floor”; report the full interval.  
Rename “context share” to “accuracy difference after context removal”; paired bootstrap draws cannot make this a causal decomposition.

**5. Donor and target labels are linked.**  
[gen_extract.py:138](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/gen_extract.py:138), [draft:36](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/review/part2_context_controls_draft_2026-10-16.md:36).  
**Failing case:** Each donor identifies three possible targets; perfect donor recognition permits one-third target accuracy in the balanced design.  
**Fix:** Report a donor-only target baseline and target decoding within donor groups.  
Donor-minus-target accuracy alone cannot separate their contributions.

---

**Should fix**

**Specify C1 and C2 exactly.**  
Use one leading `<|endoftext|>`, the exact header, no added special tokens, and right padding with masks.  
C1 uses index `prompt_length − 1`; causal attention prevents reply leakage there.  
C2 keeps the test question, `Guest:`, and unchanged kept reply IDs; remove descriptions, all example exchanges, and filler notes.  
For `ex2_other`, remove both the target description and donor examples; pool only the new reply positions.

**Do not promise completion within one hour.**  
The local synthetic test took 4.99 seconds for ten fits and 0.78 seconds for one 50-half activation map.  
The 99,990 decoding fits alone imply about 26 minutes with perfect 32-worker scaling.  
Maps, serial baseline fits, embeddings, transfer, extraction, and worker overhead add time.  
Benchmark the actual pods; 80 GB specifies memory capacity, not GPU speed.

| Execution item | Required action |
|---|---|
| CPU pods | Use `--conds CONDITION --jobs 32`; cap OMP, MKL, and OpenBLAS threads at one. |
| Embeddings | Use `--device cpu`, or copy the completed GPU embedding cache. |
| Caches | Preserve code, fingerprint, full input rows, and row order. |
| Outputs | Use separate output folders; merge explicitly. |

[part2_analysis.py:490](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/part2_analysis.py:490) writes the same JSON filename for separate condition runs.  
It saves condition results only after the maps finish; interrupted permutations have no partial checkpoint.

**Finish the report calculation.**  
[score_naming.py:286](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/score_naming.py:286) still substitutes `1.0` for both Part 2 p-values.  
Merge the four raw decision p-values and apply Holm once; local decision flags are provisional.  
Run replication with `--conds desc ex2`; compare gain, maps, and axis measures too.  
The draft’s decoding-only agreement falls short of the frozen replication requirement; another seed still reuses the prompts and examples.

**Limit map and axis claims.**  
[part2_analysis.py:275](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/part2_analysis.py:275): reliable distances can reflect stable prompt differences.  
[Axis:350](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/part2_analysis.py:350): halves reuse wordings, and defaults have shorter descriptions.  
Call it a “default-frame contrast”; cosine ≥0.80 establishes repeatability, not assistant function.  
Decoding intervals condition on fitted predictions and omit refitting uncertainty.

**Next step: use the two hours on C0, C1, C2, corrected wording, the neutral-continuation control, and report assembly.**

---

**Can wait**

New example banks, description-preserving/example-removing C2, and ridge residuals on C1 can follow.  
Fit residualisation inside training folds; removing predictive signal does not prove its source.

Correct the literature wording: decoding without geometric restructuring does not establish unchanged behaviour. [System-prompt study](https://arxiv.org/html/2609.38205v1)

The self-test passed. I edited no files and opened no sealed data or results.

**Blockers: 5.**