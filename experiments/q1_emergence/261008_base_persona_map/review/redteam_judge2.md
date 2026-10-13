# Red-team review of `judge2.py`

**Verdict:** Do not run this judge yet. I found **12 must-fix problems**.

The proposed score can measure target-role evidence in a reply. It does not yet validate the claim that the model *plays the role*.

## Review scope

I reviewed these files:

- `judge2.py`, including preparation, batching, keys, validation, and the run loop.
- `NOTE.md` and Part 1 of `THRESHOLDS.md`.
- The three rendered samples under `review/judge2_sample/`.
- All 20 source `eval_prompt` values.
- Moskvoretskii et al. 2026, Appendices D.1 and F.2-F.3.
- Lu et al. 2026, including its role-evaluation rubric.

I ran only the permitted `prep` command. I did not run either judge.

| Preparation mode | Replies | Items | Role batches | Coherence batches |
|---|---:|---:|---:|---:|
| All three donors | 14,400 | 48,600 | 342 | 144 |
| One sampled donor | 10,800 | 34,200 | 234 | 108 |

## Bottom line on the approach

The approach is directionally reasonable for **role-consistent behavior**. The separate coherence score and hidden item kinds are good choices.

The approach is not yet sound for **role enactment**. The new prompt joins two incompatible constructs without validating the result.

[Moskvoretskii et al.](https://arxiv.org/pdf/2605.13329) score four behavioral traits. [Lu et al.](https://arxiv.org/pdf/2601.10387) use identity-sensitive role labels from 0 to 3.

The new judge keeps Lu's definition but removes Lu's identity rubric. It then applies Moskvoretskii's uncalibrated cutoff.

---

# Design problems

## 1. [MUST FIX] The score does not operationalize “plays the role”

**Evidence.** The prompt asks whether a reply “displays” a role. Its endpoints are “role absent” and “role strongly present.”

That can reward one matching metaphor or topic. It does not require the speaker to adopt the role's identity, perspective, or stable behavior.

Lu's discarded rubric makes identity central. Its top label means the model is fully playing the role and is not presenting as an AI.

**Impact.** A generic answer can pass as `assistant`, `consultant`, `counselor`, or `optimist`. A subtle in-role answer can fail.

**Concrete fix.** Choose and name one construct before scoring:

1. Use **target-role evidence** for the current 0-100 prompt. Do not call it role-playing.
2. For **plays the role**, add an enactment rubric with anchored identity and behavior levels.

The cleanest design uses two axes. Score role evidence from 0-100 and enactment from 0-3.

Require a pre-registered enactment result for the “plays the role” claim. Keep role evidence as a secondary measure.

## 2. [MUST FIX] The cutoff at 50 has no relevant validation

**Evidence.** Moskvoretskii et al. use 50 as an extraction filter for four traits. Their judge was GPT-4.1-mini.

Their 91% human agreement result uses pairs below 20 and above 70. It does not test absolute labels near 50.

It also does not test these 20 roles, these questions, or either proposed judge. It does not validate coherence at 50.

**Impact.** Score compression can cause false fails. Lenient scoring can cause false passes.

Using the same cutoff for both judge families assumes equal calibration. There is no evidence for that assumption.

**Concrete fix.** Freeze a human rubric and sampling plan before judge results are visible.

Keep human labels blind to condition and judge outputs. Sample across roles, conditions, seeds, and expected cue strength.

Use two human raters per item. Adjudicate disagreements.

Use one human subset for calibration and a separate subset for validation. Estimate sensitivity and specificity near the selected cutoff.

Calibrate each judge separately. Justify the human sample size with a precision or power target.

## 3. [MUST FIX] The replacement measure has no frozen analysis plan

**Evidence.** `THRESHOLDS.md` still defines blind `hit@5` as the primary Part 1 measure. It contains no rule for `judge2` scores.

`judge2.py` defines only a reply-level pass. It does not define a condition-level decision.

There is no frozen rule for two judges, disagreements, `REFUSAL`, missing batches, or human corrections.

**Impact.** Many reasonable analysis choices remain after results appear. This creates researcher degrees of freedom.

**Concrete fix.** Update Part 1 before any judge run. Define all of these items:

- The exact construct and primary measure.
- The reply, role, question, and condition aggregation rules.
- The target-versus-control statistic.
- The two-judge combination rule.
- The treatment of `REFUSAL` and missing scores.
- The bootstrap, null, multiplicity, and replication rules.
- The copied-text and self-naming sensitivity rules.

Keep the old blind `hit@5` as a secondary convergent measure. It answers a different question.

## 4. [MUST FIX] The controls do not establish discriminant role evidence

**Evidence.** Each reply gets one random wrong role. Most wrong roles are easy and semantically distant.

The wrong-role assignment is not blocked or balanced. In the full preparation, wrong-role counts range from 657 to 780.

The one-donor sample is also imbalanced. Selected donor frequencies range from 77 to 104.

**Impact.** A low random-wrong pass rate can coexist with poor discrimination among close roles.

For example, generic advice can fit assistant, counselor, consultant, grandparent, optimist, or stoic. A distant `vampire` control will not detect this.

**Concrete fix.** Use a fixed, balanced control design.

Score every reply against the target, a close confusable role, and a distant role. Add the donor for `ex2_other`.

Use a balanced incomplete block or Latin-square assignment. Match assignments across conditions and seeds.

Pre-register a target-minus-close-control margin. Report random-wrong scores only as a secondary diagnostic.

## 5. [MUST FIX] Role difficulty and question observability are not modeled

**Evidence.** The definitions differ greatly in breadth and visible cues. `assistant` is broad, while `void` is narrow and theatrical.

The shared questions are useful for comparison. However, many valid role answers can be neutral on one question.

The current design turns each reply into a pass or fail. It then treats score 50 as comparable across all roles.

**Impact.** Broad roles get false passes. Subtle roles get false fails.

The result can measure stereotype visibility instead of role enactment. Question choice can dominate role score.

**Concrete fix.** Make the role-by-question response pattern the unit of evidence.

Use role-specific positive and negative human anchors. Estimate per-role calibration or model judge score with role effects.

Require coverage across questions. Do not claim role enactment from one reply-level pass.

Report role-level results before any macro average. Show broad-role and narrow-role results separately.

## 6. [MUST FIX] A 100-item context does not make judgments independent

**Evidence.** The header tells the judge to score items independently. The same context still contains about 100 role criteria and replies.

Random order reduces one bias. It does not remove range anchoring, carryover, or relative score normalization.

The role batches are also not type-balanced. With one donor, their sizes range from 97 to 103 items.

Research finds that LLM judge scores change with context and anchors. Instructions to ignore anchors do not reliably remove the effect.

See [Kapetanovic et al. 2026](https://arxiv.org/abs/2608.25869) and [Bhat and Varma 2026](https://aclanthology.org/2026.findings-acl.1929/).

**Impact.** A strong early item can shift later scores. Different condition mixtures can create batch effects.

**Concrete fix.** Use smaller, balanced batches. Randomize item order separately for each judge.

Repeat a pre-registered audit subset in different batches and positions. Compare scores and pass labels.

If batching stays large, include batch and position in the validation analysis. Set an acceptable repeatability limit before running.

---

# Code and template bugs

## 7. [MUST FIX] The “isolated” judge is not isolated

**Evidence.** Codex runs with `workspace-write`. Claude receives `Read` and `Write` tools.

Changing the working directory does not restrict readable files. The instruction “Do not open any other file” is only text.

`NOTE.md` already records that the earlier CLI judge ignored a similar rule. It opened skill files outside the allowed set.

**Impact.** A judge can read other batches, earlier answers, keys, project instructions, or unrelated files.

Hidden system prompts and installed skills also make this unlike the cited direct GPT-4.1-mini judge.

**Concrete fix.** Give each call an operating-system sandbox or container. Mount only one batch and one empty output directory.

Disable all tools if possible. Prefer sending the full batch as input and parsing standard output.

Add a canary file outside the mount. Run a non-scoring isolation test that proves neither CLI can read it.

Record the CLI version, resolved model version, system configuration, and prompt hash.

## 8. [MUST FIX] Valid-looking stale answers can be silently reused

**Evidence.** `prep()` does not require an empty judge directory. It does not remove old batches or answers.

`job()` skips any answer with the expected ID set. IDs hash only `(run, i, kind)`.

The ID does not hash the scored role, question, answer, definition, prompt, judge, or batch content.

**Impact.** A changed prompt can receive an old score. A changed wrong-role assignment can reuse the same wrong-item ID.

Old batch files can also remain and enter a later run. The run loop scans every `*.md` file.

**Concrete fix.** Refuse non-empty output directories by default. Add an explicit `--resume` mode.

In resume mode, require an exact manifest match. Hash every batch, prompt template, source file, model setting, and item payload.

Use a content-derived item ID. Drive the run from the manifest, not from a directory glob.

## 9. [MUST FIX] The preparation path guard is wrong and non-atomic

**Evidence.** Line 89 checks only `judge_dir.parents`. It allows `judge_dir == REPO` and directories above the repository.

It uses `assert` for a runtime safety check. Python can disable assertions.

Preparation writes directly into its final locations. A failure can leave mixed old and new artifacts.

**Impact.** A mistaken path can place or overwrite judge artifacts in unsafe locations. Partial preparation can corrupt provenance.

**Concrete fix.** Use explicit path checks, not `assert`.

Reject any path that equals, contains, or sits inside the repository. Require a new empty directory under an approved scratch root.

Write into a temporary sibling directory. Validate counts and hashes, then rename it atomically.

Never delete an existing directory automatically.

## 10. [MUST FIX] Command validation can launch the wrong judge or report false success

**Evidence.** The `judge` argument is optional for `run`. A missing value falls into the Claude branch.

It then targets `answers_None`, which preparation never creates. This can launch an unintended, costly call.

An unknown `--batches` selection becomes an empty list. The program then prints `0/0 batches valid` and exits successfully.

The docstring advertises `--batches 1-2`. The implementation accepts exact comma-separated names only.

**Impact.** An operator can spend judge calls on the wrong path. A no-op can look like a successful run.

**Concrete fix.** Use separate `prep` and `run` subparsers. Require the judge for `run`.

Reject unknown or empty batch selections. Validate `workers > 0` and `donor_sample` against explicit choices.

Make the documented range syntax work, or remove it. Print the resolved batch list before starting any call.

## 11. [MUST FIX] Output validation does not enforce the stated format or source scale

**Evidence.** `json.loads()` silently accepts duplicate object keys. `check()` does not require a JSON object before using it.

The validator rejects every non-integer score. The cited prompt asks for a “single number,” not specifically an integer.

Moskvoretskii et al. report individual example scores with decimals. The current integer restriction is an undocumented change.

**Impact.** Duplicate keys can pass unnoticed. Valid source-style scores can trigger three wasted retries.

A non-object response can raise an uncaught exception. This stops the run instead of recording a failed batch.

**Concrete fix.** Parse with an object-pairs hook that rejects duplicate keys. Require exactly one JSON object.

Accept finite, non-boolean real numbers from 0 through 100. Alternatively, document and validate a deliberate integer rubric.

Reject extra text and non-finite values. Add unit tests for duplicates, lists, booleans, decimals, `NaN`, and missing IDs.

## 12. [MUST FIX] Raw replies can break item boundaries and ID parsing

**Evidence.** Questions and replies are inserted without escaping. The parser finds expected IDs with `^### (\S+)$` over the batch text.

A reply can contain `### fake_id`, `[CONTINUATION END]`, or judge instructions. This can add an ID or escape the data block.

The current replies contain no exact tested delimiter. That is a property of this data copy, not a code guarantee.

Prompt-injection attacks against LLM judges are documented. See [Maloyan and Namiot 2025](https://arxiv.org/abs/2504.18333).

**Impact.** A generated reply can change the prompt, the expected ID set, or another item's score.

**Concrete fix.** Store items in a structured format with escaped values. Treat all question and answer text as untrusted data.

Get expected IDs from the hashed manifest, never from rendered text. Reject forbidden delimiters during preparation.

Put judge instructions before the data. Add adversarial boundary and instruction-injection tests.

---

# Other problems to fix soon

These problems do not enter the must-fix count above.

## A. Worker exceptions bypass the retry and summary logic

`subprocess.TimeoutExpired`, missing executables, and file errors can escape `job()`. One future can abort the full `map()` call.

**Fix.** Catch per-job exceptions. Record them as batch failures and keep the other jobs running.

## B. The rendered header sample is internally inconsistent

`header_role_1.txt` says 100 items. `role_1_head.md` says the batch has 99 items.

The runtime code would calculate 99. The review sample was not rendered through that exact runtime path.

**Fix.** Generate all review artifacts with one dry-run renderer. Add the batch hash and exact item count.

## C. The manifest is too weak for reproduction

It lacks input hashes, prompt hashes, batch hashes, role-definition hashes, source revisions, and software versions.

**Fix.** Record all of these fields. Include the selected reply keys and wrong-role assignments.

## D. The refusal wording no longer matches the task

Moskvoretskii's rubric asks whether the model refused to continue a hypothetical situation. This task contains interview answers.

“I don't know” can be a coherent guest answer. The current wording can turn it into `REFUSAL`.

**Fix.** Define refusal for an interview. State whether in-role refusal counts as enactment, failure, or missing evidence.

## E. The role batches are only approximately balanced

The `BATCH + 5` capacity produces 97-103 role items per batch in the tested sampled-donor preparation.

Some sampled batches contain 39 wrong items. Others contain 55.

**Fix.** Stratify by item kind, scored role, condition, run, and approximate length. Keep a fixed size where possible.

---

# Template audit

| Check | Result | Comment |
|---|---|---|
| Condition label hidden | Pass | No `desc`, `ex2`, `ex2_other`, or `ex2_filler` label appears. |
| Item kind hidden | Pass | Target, wrong, and donor items use the same wording. |
| Donor identity hidden | Pass | The prompt gives only the role being scored. |
| Opaque IDs | Pass | IDs do not reveal condition or role. |
| Same reply repeated in one batch | Pass | The tested preparation had zero repeated reply keys per batch. |
| Target criterion hidden | Not applicable | A pointwise rubric must state the scored role. This creates unavoidable criterion priming. |
| Role-template fidelity | Fail | This is a new hybrid, not either cited role rubric. |
| Coherence-template fidelity | Partial | The item wording matches. The outer integer rule changes the accepted scale. |
| Refusal fidelity | Fail | “Continue the situation” became an ambiguous interview refusal rule. |
| Batch independence | Fail | One context contains about 100 judgments. |
| Untrusted-text boundary | Fail | Reply text is not escaped or structurally isolated. |
| Rendered header count | Fail | The sample says 100 while its batch says 99. |

There is no direct condition or item-kind leakage. That part is well designed.

The target role and definition create semantic anchoring. This is necessary for pointwise scoring, but it needs calibrated controls.

---

# Likely false passes and false fails

| Failure | Example mechanism | Best defense |
|---|---|---|
| False pass | A generic helpful answer fits `assistant` or `counselor`. | Close-role controls and human calibration. |
| False pass | One keyword or metaphor matches `chef`, `bartender`, or `pirate`. | Multi-facet anchors and target-control margins. |
| False pass | The reply names the role without enacting it. | Separate self-identification from behavioral evidence. |
| False pass | Example copying supplies obvious role cues. | Keep the copy-free sensitivity analysis primary. |
| False pass | A judge uses scores relative to nearby batch items. | Smaller batches and repeated position audits. |
| False pass | Incoherent keyword text gets a high role score. | Keep coherence separate and validate its cutoff. |
| False fail | A role gives a normal answer on a neutral question. | Aggregate across questions and model observability. |
| False fail | A subtle role lacks stereotyped words. | Use human anchors with subtle positive examples. |
| False fail | One judge compresses most scores below 50. | Calibrate each judge separately. |
| False fail | An in-role refusal becomes `REFUSAL`. | Define role-consistent refusal explicitly. |
| False fail | Base-model truncation lowers coherence despite clear role evidence. | Report role and coherence separately before conjunction. |
| False fail | Requiring both judges rejects valid but ambiguous roles. | Pre-register and validate the combination rule. |

---

# Minimum acceptable design before running

1. Define “role evidence” and “role enactment” separately.
2. Freeze the new Part 1 analysis in `THRESHOLDS.md`.
3. Freeze a blinded human calibration and validation plan.
4. Replace random wrong roles with balanced close and distant controls.
5. Reduce batches or validate batch and position stability.
6. Make each CLI call genuinely isolated.
7. Make preparation fresh, atomic, hashed, and resumable only by exact manifest match.
8. Fix command and output validation.
9. Escape item data and test prompt-boundary attacks.
10. Regenerate the review samples from the exact runtime path.

After these changes, the judge can support a careful claim about role-consistent behavior. The stronger “plays the role” claim still needs human-validated enactment labels.
