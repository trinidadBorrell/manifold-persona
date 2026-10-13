# Red-team review of `judge2.py` v3

**One-line answer:** v3 has **0 blockers** before a small paid canary run.

Two old fixes remain incomplete. One old measurement limit remains by design.

---

**Scope**

I reviewed v3 against every v2 blocker and every “other new problem.”

I excluded the two human raters and frozen Part 1 rules. The owner controls those items.

I did not run `judge2.py run` or `judge2.py selftest`.

---

**Offline checks**

I ran the permitted `prep` command in a new `/tmp` folder.

| Check | Result |
|---|---:|
| Selected replies | 10,800 |
| Total items | 46,710 |
| Role batches | 1,437 |
| Coherence batches | 432 |
| Batches with 24 items | 15 |
| Batches with 25 items | 1,854 |
| Variable close-control cells | 0 |
| Variable distant-control cells | 0 |
| Donor and distant collisions | 0 |
| Audit batch or position collisions | 0 |
| Selected weak injection-word matches | 18 |
| Direct judge-instruction attacks found | 0 |

I also ran offline tests for parsing, cache checks, and design validation.

The parser rejected duplicate keys in the CLI envelope and nested structured output.

A changed batch stopped before cache acceptance. The design checker accepted one invalid synthetic design.

---

**V2 blockers 1–9**

| # | Status | File evidence | Finding |
|---:|---|---|---|
| 1 | **FIXED** | `judge2.py:84-102`, `judge2.py:468-485` | Level 3 now needs the whole answer to use the role viewpoint. The catchphrase anchor tests level 2. |
| 2 | **FIXED** | `judge2.py:136-142`, `judge2.py:234-247` | Controls exclude donor data. Sampling removes the fixed distant role from donor choices. |
| 3 | **FIXED** | `judge2.py:368-380`, `judge2.py:389-392` | `job()` checks the batch hash before it reads a cached answer. Cached records bind both hashes. |
| 4 | **FIXED** | `judge2.py:307-323`, `judge2.py:346-365` | The CLI envelope uses duplicate-key rejection. String structured output uses the same strict parser. |
| 5 | **FIXED** | `judge2.py:280-298`, `judge2.py:399-406` | The folder contains the key. The manifest binds key, code, prompts, and CLI version. Run checks them. |
| 6 | **PARTLY FIXED** | `judge2.py:56-62`, `judge2.py:228-230`, `judge2.py:460-461` | All structural delimiters are blocked. Direct instruction attacks still have no test or separate handling. |
| 7 | **FIXED** | `judge2.py:98-102`, `judge2.py:120-124`, `judge2.py:326-336` | Stored refusal records cannot be contradictory. Runtime validation enforces the conditional rule. |
| 8 | **FIXED** | `judge2.py:167-209`, `judge2.py:268-279` | Packing gives 24–25 items. Audit copies use another batch and another position. |
| 9 | **PARTLY FIXED** | `judge2.py:410-422` | Empty and repeated names fail. The message shows only the first ten selected names. |

---

**Blocker 1 details**

The new level boundary is operational. Level 2 covers isolated cues at `judge2.py:90-92`.

Level 3 requires the answer as a whole at `judge2.py:93-95`.

The live result supplied by the owner supports this boundary. I did not repeat that paid test.

---

**Blocker 2 details**

`controls()` uses only role, wording, and question rank at `judge2.py:136-142`.

Donor sampling reads the fixed distant role at `judge2.py:243-245`.

The prepared output had no variable control cells. It also had no donor and distant collision.

---

**Blockers 3–5 details**

The prompt hash check is the first operation after path setup at `judge2.py:368-372`.

The cached answer then needs matching manifest and prompt hashes at `judge2.py:373-378`.

The strict outer parser also checks nested objects. My two duplicate-key tests both failed as required.

The run checks key, code, and CLI version at `judge2.py:399-406`.

Each model call checks the used model at `judge2.py:355-360`.

---

**Blocker 6 details**

The delimiter regex now blocks `<<<`, `>>>`, and every inner marker at `judge2.py:56`.

This closes the concrete item-boundary bug from v2.

The self-test checks delimiter attacks only at `judge2.py:460-461`. It does not test instruction-level attacks.

The manifest records a weak match count at `judge2.py:291`. It does not change or block matched replies.

I inspected the 18 selected matches. They use “instructions” in ordinary emergency-response answers.

I found no direct instruction to the judge. This remaining gap does not block the canary.

---

**Blocker 7 details**

The rubric defines out-of-role refusal and null evidence at `judge2.py:98-102`.

`valid_value()` enforces the cross-field rule at `judge2.py:331-336`.

The JSON Schema permits null evidence, but it does not enforce the conditional rule by itself.

This is safe because every stored answer must pass `check()` at `judge2.py:339-343`.

---

**Blocker 8 details**

The packer computes balanced capacities at `judge2.py:176-189`.

Preparation checks every final size at `judge2.py:272-279`.

The current output has 15 batches of 24 items. All other batches contain 25 items.

All 1,710 audit copies use another batch and another position.

---

**Blocker 9 details**

The parser rejects empty and repeated selections at `judge2.py:410-416`.

However, `judge2.py:421-422` prints only ten names. A full selection still ends with an ellipsis.

This misses the exact v2 requirement to print all resolved names.

A small canary with ten or fewer explicit names shows every name. The defect does not block that canary.

---

**V2 “other new problems”**

| Problem | Status | File evidence | Finding |
|---|---|---|---|
| Removable production assertions | **FIXED** | `judge2.py:266-279` | Preparation uses explicit errors. Assertions remain only inside `selftest()`. |
| Missing design checks | **PARTLY FIXED** | `judge2.py:149-164`, `judge2.py:231-247` | Counts and cells are checked. Allowed wordings and donor groups are not checked. |
| Joint-score dependence | **NOT FIXED** | `judge2.py:8-9`, `judge2.py:78-102` | Both scores still share one item. The header now states this limit correctly. |
| “Two answers” wording | **FIXED** | `judge2.py:4`, `judge2.py:83`, `judge2.py:101-102` | The text now says three fields. |

---

**Design-check residual**

`check_design()` verifies counts, distinct donors, and cell counts at `judge2.py:149-164`.

It does not require wordings `{0,1,2,3,4}`. It also does not require donors from another group.

My synthetic complete grid used wording `99` and same-group donors. `check_design()` accepted it.

The current real data use the correct five wordings. All selected donors are from another group.

This gap does not block this canary. Fix it before preparation must protect new input data.

---

**New bugs**

I found no new bug distinct from the residual v2 items above.

The two code gaps are incomplete old fixes. They are not new regressions.

---

**Canary decision**

The small paid canary can start with explicit, unique batch names.

| Gate | Status |
|---|---|
| Rubric boundary | Ready |
| Refusal storage | Ready |
| Cache binding | Ready |
| Batch balance | Ready |
| Current data design | Ready |
| Canary payment safety | Ready |

Use ten or fewer explicit batch names. Then the current status line shows every selected name.

Do not treat the evidence score as an independent validation of enactment.

**Blockers before a small paid canary run: 0.**
