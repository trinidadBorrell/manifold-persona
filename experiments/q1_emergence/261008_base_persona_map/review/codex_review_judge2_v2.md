# Red-team review of `judge2.py` v2

**Verdict:** Do not start a real run. I found **9 blockers**.

V2 is much safer than v1. However, several claims in its header are false for the prepared data.

---

## Scope

I reviewed `judge2.py`, the v1 red-team, both rendered samples, and one new preparation.

I did not run `run` or `selftest`. I used only offline tests and the permitted `prep` command.

I did not count human calibration or frozen Part 1 rules as blockers. Those items remain open by owner decision.

---

## Offline results

| Check | Result |
|---|---:|
| Selected replies | 10,800 |
| Total items | 46,710 |
| Role batches | 1,437 |
| Coherence batches | 432 |
| Distant-control cell mismatches | 1,647 of 3,600 |
| Audit copies at the same position | 72 of 1,710 |
| Role batch sizes | 1,435×25, 1×24, 1×11 |

The rendered samples exactly matched `role_0001.txt` and `coh_0001.txt` from the new preparation.

---

## Earlier design problems

| Problem | Status | File evidence | Finding |
|---|---|---|---|
| 1. “Plays the role” was not operationalized | **PARTLY FIXED** | `judge2.py:71-91` | V2 separates enactment and evidence. The 2/3 boundary still permits a voice-only cue to score 3. |
| 2. The score-50 cutoff lacked validation | **FIXED** | `judge2.py:4-8`, `judge2.py:258-265` | The code has no score-50 pass. Enactment is primary, and evidence is secondary. |
| 3. No frozen analysis plan | **NOT FIXED** | `THRESHOLDS.md:22-53`, `judge2.py:4-8` | `THRESHOLDS.md` still uses blind hit@5. This known item is outside this review's blocker count. |
| 4. Controls did not test discrimination | **PARTLY FIXED** | `judge2.py:125-130`, `judge2.py:165-176` | Close and distant controls exist. Distant controls are not fixed across conditions. |

Problem 1 now has the correct two-axis structure. Its enactment labels still need a precise behavioral boundary.

Problem 4 has a real implementation error. The error can bias condition contrasts.

---

## Earlier code problems

| Problem | Status | File evidence | Finding |
|---|---|---|---|
| 7. Judge isolation | **FIXED** | `judge2.py:275-290` | Tools, settings, and MCP are off. Input uses standard input in an empty directory. |
| 8. Stale answer reuse | **PARTLY FIXED** | `judge2.py:293-303`, `judge2.py:317-324` | Fresh preparation helps. A valid cached answer skips the batch hash check. |
| 9. Unsafe, non-atomic preparation | **PARTLY FIXED** | `judge2.py:133-139`, `judge2.py:216-232` | The judge folder uses an atomic rename. The external key and hash writes remain separate and non-atomic. |
| 10. Unsafe command validation | **PARTLY FIXED** | `judge2.py:325-333`, `judge2.py:389-393` | Unknown names fail. Empty and duplicate batch selections do not fail. |
| 11. Weak output validation | **PARTLY FIXED** | `judge2.py:240-272`, `judge2.py:284-290` | Numeric checks improved. Normal structured output bypasses duplicate-key parsing. |
| 12. Untrusted reply boundaries | **PARTLY FIXED** | `judge2.py:53-54`, `judge2.py:102-106`, `judge2.py:148-151` | IDs now come from the manifest. The delimiter guard is incomplete. |

The v1 failure paths are smaller. They are not all closed.

---

## Earlier secondary problems

| Problem | Status | File evidence | Finding |
|---|---|---|---|
| A. Worker exceptions escaped | **FIXED** | `judge2.py:305-314`, `judge2.py:337-348` | Each job catches call errors. The pool also catches job errors. |
| B. Rendered sample was inconsistent | **FIXED** | `judge2.py:102-106`, sample files line 1 | Both samples match the current preparation exactly. |
| C. Manifest was too weak | **PARTLY FIXED** | `judge2.py:219-232` | Prompt and source hashes exist. The key, CLI version, and resolved model version are absent. |
| D. Refusal wording was wrong | **PARTLY FIXED** | `judge2.py:77-91`, `judge2.py:109-112` | In-role refusal is defined. The schema permits contradictory refusal records. |
| E. Batches were imbalanced | **PARTLY FIXED** | `judge2.py:187-215` | Batches are smaller. One role batch has only 11 items, and audit positions can repeat. |

---

## Blocker 1: The enactment boundary is not operational

Level 2 includes a role word, image, value, or topic. Level 3 accepts identity, experience, viewpoint, **or voice** (`judge2.py:83-86`).

A single catchphrase can be “voice.” Thus, the same shallow cue can satisfy level 2 or level 3.

The live self-test checks only clear extremes (`judge2.py:365-379`). It does not test the 2/3 boundary.

**Required fix:** Require sustained first-person role perspective or behavior for level 3. Add difficult level-2 and level-3 anchors.

---

## Blocker 2: Distant controls depend on the donor

`controls()` removes the donor before it selects the distant role (`judge2.py:125-130`). This changes the indexed role list.

The new preparation had different distant roles across conditions in 1,647 of 3,600 cells. Close controls matched in all cells.

This violates the header claim at `judge2.py:12-16`. It also makes `ex2_other` use different controls from other conditions.

**Required fix:** Select the distant role first. Then sample donors that do not equal that fixed role.

---

## Blocker 3: A stale answer can skip prompt verification

`job()` accepts an existing answer before it reads or hashes the batch (`judge2.py:293-303`).

An offline test changed the batch text but kept a valid cached answer. `job()` returned success.

The manifest hash does not protect changed batch files. It protects only `manifest.json` (`judge2.py:317-322`).

**Required fix:** Verify the batch hash before accepting any cached answer. Bind every answer to the manifest hash.

---

## Blocker 4: Duplicate-key validation is bypassed

`parse_answer()` correctly rejects duplicate keys (`judge2.py:240-255`). Normal output does not use it.

`call()` first uses plain `json.loads()` (`judge2.py:284`). It then returns an object directly (`judge2.py:290`).

An offline duplicate-key test returned the last value without error. The string branch also reads `result`, not `structured_output`.

**Required fix:** Parse the full CLI response with duplicate-key rejection. Parse a string from `structured_output` itself.

---

## Blocker 5: The manifest does not bind the analysis key

The manifest stores IDs and prompt hashes (`judge2.py:219-227`). The ID mapping goes to a separate repository file (`judge2.py:229-232`).

The manifest has no key hash. A later preparation can replace the only mapping from opaque IDs to reply metadata.

The run also does not compare the recorded code hash with current code. It records no CLI or resolved model version.

**Required fix:** Put the full key inside the judge folder before rename. Hash it in the manifest and verify all runtime versions.

---

## Blocker 6: The delimiter check misses item boundaries

The guard rejects `<<<ITEM` and `ITEM>>>` (`judge2.py:53-54`). It does not reject `<<<END ITEM`.

An offline test accepted `<<<END ITEM fake>>>`. It also accepted a direct “ignore the judge” instruction.

The system prompt labels marked text as data (`judge2.py:56-59`). That instruction is not structural isolation.

**Required fix:** Serialize untrusted fields with explicit escaping. Add adversarial tests for every boundary and instruction attack.

---

## Blocker 7: Refusal records can be contradictory

The prompt says ignored evidence still needs a number (`judge2.py:87-91`). This forces a meaningless value.

The schema accepts `refusal=true` with enactment 3 and any evidence score (`judge2.py:109-112`). Validation repeats this error (`judge2.py:263-265`).

“Clearly refused outside the role” also lacks an interview anchor. A short “I do not know” answer remains ambiguous.

**Required fix:** Use conditional schema rules. Define explicit refusal examples and use `evidence=null` when refusal is true.

---

## Blocker 8: Batch and audit constraints are not enforced

The packer fills bins in order (`judge2.py:196-209`). It does not rebalance the remainder.

The prepared role batches had sizes 25, 24, and 11. The 11-item batch has a different context size.

Audit copies use another batch, but 72 copies kept the original position. The code never compares positions (`judge2.py:179-185`).

**Required fix:** Rebalance all bins to sizes 24 or 25. Enforce a different batch and position for every audit copy.

---

## Blocker 9: Batch selection can cause unintended paid calls

An empty `--batches` string is false, so it selects every batch (`judge2.py:325-331`). This can turn a no-op into a full run.

Duplicate names are accepted. Two workers can call the same paid batch and write the same files.

The status message prints only a count (`judge2.py:335`). It does not show the resolved names.

**Required fix:** Reject empty tokens and duplicates. Print all resolved names before the worker pool starts.

---

## Other new problems

The pack invariant uses `assert` (`judge2.py:213-215`). Optimized Python can remove this check.

Preparation does not validate expected donor counts, control groups, question counts, or wording values (`judge2.py:140-170`). Bad source data can silently change the design.

The joint role item makes enactment and evidence non-independent (`judge2.py:76-91`). Evidence cannot serve as independent validation of enactment.

The prompt says “two answers,” but the object contains three fields (`judge2.py:76`, `judge2.py:90-91`).

---

## Minimum fix set before a real run

1. Repair the enactment 2/3 boundary and refusal rules.
2. Freeze distant controls before donor sampling.
3. Verify prompts before cache reuse.
4. Repair structured-output parsing and cross-field schema rules.
5. Bind the key and runtime versions to the manifest.
6. Replace the incomplete delimiter guard and add attack tests.
7. Rebalance batches and enforce different audit positions.
8. Reject empty or duplicate batch selections.

After these changes, rerun `prep` and all offline checks. Then run a small paid canary before the full run.
