# Plan: concurrent-work post with Han Xiao (draft, 12/11/2026)

Status: proposal for review. Nothing here is approved or run. Owner asked: "Review this plan with fable 5.2 and with codex, and lets make sure that they are no dangling experiment that we would like to finalize first".

## Advice behind it

Mentor message relayed by the owner (12 Nov, 9:46 PM): "it's fine to release ours as concurrent work and just mention that there is this concurrent work" (crediting Ivan Arcuschin), "and we confirm their findings but in addition add X and Y finding".

## Lead question

When does the Assistant become privileged across OLMo-2-7B training (base, SFT, DPO, Instruct)? Concurrent work: Han Xiao, "How the assistant axis shifts across post-training stages" (LessWrong, 5 Nov 2026; Olmo 3 7B SFT, DPO, RLVR; data `han-xiao/assistant-axis-shift`, CC-BY-4.0).

## Han Xiao's three findings against OLMo-2 data

| His finding | OLMo-2 data now | Needed |
|---|---|---|
| RLVR barely changes anything | Consistent on fixed text: DPO to Instruct map change 0.000004; RL steps 60-360 match Instruct (pod C, `261029_fixed_text_read_through/records_full/analysis_full_v2/results_full_v2.json`). Nik Khah and Bahreini (arXiv 2609.32196): OLMo-2 final close to DPO for stories | Own-reply run gives his exact design |
| After DPO, desirable roles move toward the Assistant axis more than undesirable roles (+0.19 [0.14, 0.23]; gap p = 0.066) | Not tested (no SFT or DPO own replies) | Own-reply run with his desirable and undesirable roles |
| DPO raises willingness to role play (pass rate 0.26 to 0.52 to 0.53) | Not tested | Same run, judge pass rate per stage |

## Our additions ("X and Y")

| Addition | Status | Source |
|---|---|---|
| Base model (he names it open): weak role play from plain text (blind naming 14.9% / 17.2% vs 2.4% / 2.9% shuffled; owner yes on 3 of 20 fresh replies); base states carry the role mostly from the prompt (pod B: prompt alone 0.91-0.98, reply alone 0.43-0.47; E3: 91.8% vs 15.2%) | Done | `review/proof_ledger_261112.md`, `owner_review/NOTE.md` |
| Writer vs reader: on the same Instruct text every stage reads the same map, and no stage sets the default apart (position 1.08 role SDs at every stage; default cloud about 12% wider) | Done, exploratory | `../261112_free_checks/NOTE.md` |
| Text baseline: reply words name the role about as well as the states (0.80 vs 0.76) | Done | pod C results file |
| Judge check: judge yes right 13 of 13 on fresh chat replies, 1 of 4 on fresh base replies | Done | `owner_review/NOTE.md` |

## Runs proposed

1. Free check on Han Xiao's released files ($0): download his per-role vectors and axes (`*/vectors-glm-5.3-flash/`, `*/axis-glm-5.3-flash.pt`, small files) and his role labels (from his figure data or repo). Compute his SFT-to-DPO desirable and undesirable shifts in role-SD units, and use our check 1 noise (detectable shift about 0.85 role SDs at 20 roles, about 0.6 at 40) to set how many roles our run needs.
2. Own-reply stage run on OLMo-2 (GPU): SFT, DPO and Instruct each write their own chat replies, pod A design (5 wordings, 40 questions, no example turns), with roles chosen from his desirable and undesirable lists plus the default. Our role and question banks are the same Lu et al. banks (`refs/assistant-axis/`: 275 roles + default, 240 questions), so his labels apply directly. Measures: projection of each role on each stage's own Assistant axis (his measure), with and without his judge filter; judge pass rate per stage (behaviour); text baseline; our default position and spread. Pod A Instruct replies exist, so Instruct may reuse them if the role set is covered.
3. Base model: keep the done results; the redesigned base default (read without the description) is optional for this post.
4. Extras, not needed for the headline: coin-flip preference test per stage (Nguyen), writer-by-reader familiarity read (Asvin G. and Lindsey).

Code changes before run 2 (from the Fable check, `review/fable_rerun_check_261112.md`): model switch and weights-only load check in the pod A generator; new run folder and per-model run names (upload path); role list from his labels; gates per stage.

## Open items found so far (possible dangling experiments)

| Item | State |
|---|---|
| Codex review of the v3 outline | Running |
| Pod B C2 at 9,999 draws | Not run; recommended skip (more draws only lower p) |
| Pod A prompt-only control on Instruct (about 4,200 passes) | Never run; bears on whether the Instruct map is a map of prompts |
| T-090 Persona Geology port on one OLMo-2 stage | Designed, 51 CPU tests pass, no GPU run |
| Manifold-test design session (curvature port) | Waiting on the owner |
| T-071: 17 design items from the 8 Oct code review | Decisions pending |
| T-084: E3 leftovers (3 Part 2 runs, `judge2.py`, 200-item sheet) | No files |
| T-078 far-target steering, T-055/057/059, T-019/039 | Design or partly; after 16 Nov |
| HF copies of 12 Nov outputs (free checks, owner review labels) | Not uploaded |
| Fork's duplicate files in `261112_free_checks/` | Owner to decide |

## Review outcome (12/11/2026)

Reviews: Fable 5.2 (`review/fable_plan_concurrent_261112.md`) and Codex (`review/codex_plan_concurrent_261112.md`). Agent reviews, not human reviews. Both say: do not run this plan as written, and do not write "we confirm".

Agreed by both and checked by me:
- Han Xiao's DPO gap depends on the unit. From his figure data (layer 16, 48 desirable, 25 undesirable): raw gap +0.172 (shuffle p 0.071); in role-SD units -0.022 (p 0.886). The role spread grows from 0.613 to 0.709 at DPO. His own notebook lists z-units as "Not yet checked".
- A like-for-like test needs all his labelled roles (48 or 49 desirable, 25 undesirable) plus his neutral anchors (the axis is default minus the mean of all roles), his judge rule (score 3, at least 50 per role), his sampling (temperature 0.7, top-p 0.9, 512 tokens; ours 0.5, penalty 1.1, 128) and his statistics (100 question half-splits, label shuffle). Our 20 roles hold only 5 desirable and 5 undesirable.
- Check 1's noise is the wrong noise for his group test, and its planted check used the signed score while the statistic is absolute. Neither 20 nor 40 roles is justified; Fable's normal-approximation power is about 53% even at 48+25.
- Pod A's generator, gates and analysis all assume pod A's model, role set and run name; an SFT run in pod A's folder would land on pod A's HF path (`--roles` default "full").
- Fixed-text "no RLVR change" is below the equal-map control and confirms nothing on its own.
- The OLMo-2-1B example-turn figure comes from a pre-5-Oct audit and should leave the evidence.

Corrections to this plan: "more draws only lower p" is wrong (a Monte Carlo p can move either way); T-084 has files (scripts and the 200-item sheet); T-071 has 19 finding rows and 7 method rows, not 17.
