---
name: challenge-result
description: Use when preparing any external empirical claim, report, figure, or takeaway from a research result.
---

# Challenge Each Result

Use this project safeguard before `presenting-results`. This challenge responds to this project's independent review. Hughes and Perez did not specify it. A file, hook, or agent review does not prove a scientific claim true.

## Claim ledger

Make one row for each external claim. Split claims when their source, population, or method differs. Use exact file paths and run IDs. Write `missing` for any absent number, file, or check. Do not invent values.

| Claim text | Source file / run | Population and method | Control and baseline | Contrary evidence | Open alternatives | Missing checks | Status | Allowed wording | Human-review status |
|---|---|---|---|---|---|---|---|---|---|
| [one claim] | [exact path; run ID] | [items; selection; method; metric] | [matched control; baseline] | [failed tests; conflicting results] | [other causes] | [specific gaps or none] | [status] | [claim with limits] | [reviewer and date, or none] |

## Check each claim

1. Open the cited source and identify the current run. Check the saved number, denominator, population, metric, and method against the report.
2. Trace the code path from raw cases through preprocessing to the outcome. Read raw inputs and outputs. Check the exact prompts and observed model behavior.
3. Check for preprocessing leakage. Match examples, splits, prompts, and outcomes across comparisons. Check baselines, negative controls, and contrary evidence.
4. Compare the reported metric and test family with the original plan. Record changes made after results were seen. Check multiple tests and adjusted results.
5. Test the interpretation against open alternatives. Separate measured geometry or activation from observed model behavior. Do not claim behavior from geometry alone.

## Set status and wording

| Condition | Status | Allowed wording |
|---|---|---|
| Source file or needed number is absent | `unverified` | "This claim is unverified: [exact source or number] is missing." |
| A negative control fails or contrary evidence limits the claim | `narrowed` | "[Observed result] under [method and population]. [Control] failed; [broader interpretation] is not supported." |
| The primary metric or test changed after results were seen | `exploratory` | "Exploratory result under [revised metric]. [Original result and failed rule] remain in the record." |
| No listed fault limits the checked claim | `candidate for human review` | State the exact result, scope, controls, and remaining checks. |

Record every condition that applies. If the source is absent, do not report its result. If a check is missing, name it and limit any claim that needs it. Keep failed controls and original tests visible. Never call a changed test confirmatory.

An agent's critique is not independent human review. Record an independent human reviewer's name and date, or `none`. A human must decide what an important result means. Without that review, avoid definitive wording. Use `presenting-results` only after this ledger and challenge.
