# Challenge of the pod A family result (06/11/2026)

Reviewer: Claude session ad361f36 (agent). **This is an agent review, not an independent human review.** Human review: none.

Method: `/challenge-result`. I read `THRESHOLDS.md` v1.5, the D1 scorer `analysis_naming.py`, the saved full results and the D1 outputs. I read 9 raw replies with their judge names (translator, assistant, pirate). I did not change the frozen code, rules or results.

## Start here

| | |
|---|---|
| Short answer | Holm at alpha 0.05: D2 passes and D1 does not. D1 clears chance on all 20 roles for both judges, but its interval is wider than the frozen precision target. D3, a gate, also misses its precision target. |
| What holds | The Instruct role map repeats across halves (r 0.901). Blind judges name the role far above chance (hit@5 0.687, null p99 0.07). |
| What does not hold | "D1 passed." "OLMo-2-7B Instruct has an Assistant Axis." The rule says to stop interpreting D1 and D3 after a missed precision target. |
| Status | D2 candidate for human review. D1 and D3 narrowed. |

---

## Evidence source

| Item | Value |
|---|---|
| Statistics | HF `brend0nc/manifold-persona-experiment-artifacts` @ `b182584cf168aad9fad7e2805747e87e88199276`, `analysis_full_261030_01a188ac/results.json` (sha256 `d682d0e7…`). Per-phase files and decoder checkpoints have no copy. |
| D1 outputs | Same dataset @ `beacc63c09abcd6175675b833b58700538b356e8`, `d1_naming_261106/`, 13 files, hashes checked. |
| Replies | Same dataset @ `60a81b9b2e337a46a0a2c41862357164273da5ff`, `run_full/`. |
| Family | Holm at alpha 0.05 (owner, 5 Nov) over D1 and D2. Adjusted p 0.0002 for both. Decisions: D1 false, D2 true. |

---

## Claim ledger

| Claim text | Source file / run | Population and method | Control and baseline | Contrary evidence | Open alternatives | Missing checks | Status | Allowed wording | Human-review status |
|---|---|---|---|---|---|---|---|---|---|
| D1: blind judges name the role from the reply | `d1_miss.json`, `d1_hit.json` | 12,000 role replies, replicates 0–2, 20 roles; top-5 of 276 names; Terra and Sonnet | Terra 0.687 [0.580, 0.795]; Sonnet 0.687 [0.576, 0.794]; null p99 0.07; 20 of 20 roles above their own null; p 0.0001 | Precision missed: ±10.7 points against ±7.5. Roles span 0.20 to 1.00: translator 0.25 / 0.20, assistant 0.42 / 0.34, void 0.39 / 0.40. 10.8% of replies use a role name; without them, 0.655 / 0.654 | Most of the interval comes from which roles are drawn, so more replies may not narrow it. This is my reading and I did not test it | A human check of the names. The frozen rules do not say how to treat 13 invalid Sonnet answers | narrowed | "Both judges named the role above chance on all 20 roles (hit@5 0.69, null p99 0.07), but the interval (±10.7 points) misses the frozen ±7.5 target, so D1 did not pass." | none |
| D2: the role map repeats | `results.json`, `primary.D2` | All 5 replicates, state 17, 50 fixed halves | r 0.901 [0.797, 0.939], p 0.0001, null p99 0.22; precision met; Holm adjusted p 0.0002 | Pod C: text-only maps reach most of this agreement. Word n-grams decode the role at 0.797 against 0.918 for state 17 | The map may mostly reflect the words in the replies | None for repeatability | candidate for human review | "In OLMo-2-7B Instruct, the 20-role map repeats across question and wording halves (r = 0.90 [0.80, 0.94]). This shows repeatability, and it does not show a persona or a function." | none |
| D3: a default-vs-role contrast | `results.json`, `primary.D3` | Cross-half axis cosine | 0.881 [0.763, 0.884]; floor met, precision missed (width 0.121 against 0.10). A fixed offset passes this rule with no role differences | The axis diagnostic shows both roles and questions widen the interval. The point sits at the top of its own interval | Default wording drives the point (0.92 to 0.94 with the same wording on both sides) | None; the gate is closed | narrowed | "D3 did not pass its precision gate, so pod C uses map comparisons only." | none |

---

## Checks

| Step | What I did | Result |
|---|---|---|
| 1. Source | Read D1 outputs and `results.json` | Numbers above match. `results.json` hash equals the saved record. |
| 2. Raw cases | Read 9 replies with judge names | Pirate replies show the role plainly. Translator and assistant replies read as generic answers, and both judges put `default` first. The low roles look like weak role signal in the reply, not judge error. Nine replies are a small sample. |
| 3. Leakage | Judges saw the reply only, with no role name in the prompt. Without role-name replies, hit@5 falls by 0.03 | No leakage found. |
| 4. Plan | Compared with `THRESHOLDS.md` v1.5 | Panel, judges and prompt were fixed before any answer. Alpha was set before D1 was scored. The bound treatment of the 13 invalid answers is new and report only. |
| 5. Interpretation | Separated naming from enactment | Naming says a reader can tell the role. It does not measure how well the reply enacts it. Terra's 61.9% pass rate on these replies is a raw judge rate. |

---

## Deviation: 13 invalid Sonnet answers

The frozen payload step needs five distinct catalog names for every item. After the one allowed retry, 13 of 13,230 Sonnet answers still failed: 12 repeat a name and 1 is empty. All 13 are panel items (teenager 6, cynic 6, translator 1), and Terra had none. The payload step refused to write a file.

A second session (76e7aa2b) ran the same retry at the same time, in the same folder. Some answers got two retries, against the one allowed, and up to 2 valid retry answers were overwritten and counted among the 13. I can no longer tell which of the 11 valid retry answers came from which attempt. At most 11 of 12,600 items are affected, so I do not expect a change, but I did not bound them separately.

I did not pick a rule. I scored both extremes: every invalid answer as a miss, and every one as a hit. Sonnet macro hit@5 is 0.6867 or 0.6878. Every decision is the same in both. The frozen `payload.json` does not exist, and `d1_miss.json` and `d1_hit.json` take its place. The values above use the miss bound.

---

## What the D1 reading depends on

`precision_met` comes from the rule "If a target is not met, report that and stop interpreting that measure". The D1 text says D1 passes only if (a) and (b) hold, and both hold. The scorer also requires precision, so `passed` is false. A new owner decision is needed for any change, including more roles or questions.

The rule "If D1 is not met: stop before pod C" names the D1 criteria. I read it as not triggered, because (a) and (b) hold. This is an agent reading, for the owner to confirm.
