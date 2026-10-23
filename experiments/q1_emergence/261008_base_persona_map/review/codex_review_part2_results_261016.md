**Verdict:** No confirmed scoring bug; the proposed claims about information sources and role-playing still exceed the evidence.

---

**Bugs**

| Check | Finding |
|---|---|
| Row mapping | Correct inverse mapping; expected 10,854 row positions match the gate hash; primary subsets exclude 54 defaults |
| Splits and draws | Same row order, seeds, splits, and paired bootstrap draws |
| Restricted decoding | Correct candidate labels and probability columns; hand-built donor-only test gives 1/3 |
| Gate | Recorded 12/12 passes; minimum centred cosine 0.998256; slim states exactly match full states |
| File checks | [Loader](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/part2_context.py:97) skips recorded feature-file hashes; stale arrays could pass metadata checks |
| Desc wording | Correct threshold decision; misleading display rounding |

Raw context arrays are absent locally; read-only access also blocked classifier refits. Thus, I checked the code and records, but did not independently reproduce every number.

Desc correctly fails: **0.0997973 < 0.10**. However, [formatting](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/part2_context.py:221) prints `0.100`; report **9.9797–21.0292%**, without rounding the lower bound into a pass.

---

**Allowed claims**

These values come from the [context results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/results/part2/part2_context_run_main.json).

| Condition | Supported interpretation |
|---|---|
| desc | Prompt context alone permits 91.8% decoding; context removal reduces R from 52.3% to C2 15.2% |
| ex2_filler | Prompt context alone permits 92.9% decoding; context removal reduces R from 45.9% to C2 11.5% |
| ex2 | Question-plus-reply C2 beats the tested joint baseline: 58.6% versus 44.3%; paired gain 14.3 points [9.8, 19.0] |
| ex2_other | C2 decodes donors at 53.4%; additional target decoding within donors does not meet the stated detection rule |

For ex2, a qualified claim is allowed: **C2 features support better prediction than this specified text baseline**. The sentence-embedding comparison remains pending.

For ex2_other, the 17:35 qualification is essential: linked donor labels can support target prediction without separate target information.

---

**Not allowed**

“Mostly from prompt context” is not established: accuracy losses cannot measure information shares, and C2 changes positions and reading context.

“Beyond reply text” overclaims: C2 retains the question and identical reply tokens; token features can outperform TF-IDF.

Ex2 C2 already reaches **50.1% at state 0**; the stated 0.13–0.16 sweep range describes desc, not every condition.

“Played donor, not target” also overclaims: R decodes targets within donors at **59.5% [54.3–64.2%]**; C2 non-detection cannot establish absence.

---

**Sentences for tonight**

1. “Desc and filler C2 miss the 10% floor; their intervals are 9.9797–21.0292% and 7.7–16.3%.”
2. “Ex2 C2 reaches 58.6%, versus 44.3% for joint text; token cues remain a possible explanation.”
3. “Conflicting examples give C2 donor accuracy of 53.4%; target accuracy within donor groups is 35.1% [32.3–37.8%].”
4. “Part 2 decisions remain pending; Part 1 ex2 passes in one seed and fails in the second.”