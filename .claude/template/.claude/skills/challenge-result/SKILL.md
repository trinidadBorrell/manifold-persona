---
name: challenge-result
description: Use before an external empirical claim, report, figure, or takeaway.
---

# Challenge each result

This review is an added safeguard. It is separate from the two-mode research workflow. A checklist, hook, or agent review does not prove a scientific claim true.

Make one row per claim. Use actual file paths and run IDs. Write `missing` for absent numbers, files, or checks. Do not invent values.

| Claim | Source file and run | Population and method | Control and baseline | Contrary evidence | Open alternatives | Missing checks | Status | Allowed wording | Human review |
|---|---|---|---|---|---|---|---|---|---|
| [claim text] | [actual path and ID] | [sample, selection, method, metric] | [control and baseline] | [failed or conflicting cases] | [other causes] | [gaps or none] | [status] | [bounded claim] | [name and date, or none] |

1. Open the cited source. Check each number, denominator, population, metric, and method against the proposed claim.
2. Trace the code path from raw cases to the reported outcome. Inspect inputs, outputs, and errors.
3. Check relevant leakage, matched comparisons, baselines, negative controls, and contrary evidence. State checks that do not apply and why.
4. Compare the reported metric and tests with the plan. Record changes made after results were seen and any multiple tests.
5. Test the interpretation against open alternatives. Separate measured signals from observed model behavior.

| Condition | Status | Allowed wording |
|---|---|---|
| Source or needed number is absent | `unverified` | State that the claim is unverified and name the missing source. Do not report the number. |
| A negative control fails or contrary evidence limits the claim | `narrowed` | State the observed result and scope. Keep the failed control visible. |
| A metric or test changed after results were seen | `exploratory` | Name the revised measure and keep the original test in the record. Do not call the result confirmatory. |
| No listed fault limits the checked claim | `candidate for human review` | State the exact scope, controls, and remaining checks. |

Record every condition that applies. A human must decide what an important result means. An agent critique is not independent human review. Record the human reviewer's name and date, or `none`; without that review, avoid definitive wording.
