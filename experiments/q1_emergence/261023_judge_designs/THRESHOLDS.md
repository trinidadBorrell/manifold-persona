# Rules for the judge-design test (fixed 23/10/2026, before any judge output)

Owner decisions of 23/10/2026: Trini's J8 and Gio's ranking judge (J10), with only the opening lines changed for
interview replies; Claude Sonnet 5.5; cost cap $5; role scores reported only. Nothing below is tuned on the labels.

## Replies and labels

- 130 `run_main` replies of the E3 pilot. None is in both label sets.
- Broken labels: set 1 = the owner's 50 desc labels (`review/attrition/owner_labels_desc_2026-10-16_final.json`,
  17 broken); set 2 = the owner's 30 fresh labels (`review/attrition/owner_labels_fresh30_2026-10-16.json`, 9 broken).
- Role scores: the owner's first 50 scores (`human/owner_role50_scores_2026-10-16.json`), 0-3 for the scored role
  (35 target, 8 close, 7 distant). The revised scores are not used: the owner saw Jev's scores before revising.

## Broken (the gate)

- J8 says broken when `persona_kind` is `collapsed`. J10 says broken when `nonsense` is true.
- A judge is usable for broken replies if Cohen's kappa against the owner label is >= 0.6 on set 1 and on set 2.
  This is the rule of 16/10 (`review/role_judge_rule_2026-10-16.md`).

## Naming (report only)

- J8: share of replies where `persona_kind` is the reply's role. J10: recall@1 and recall@5 of the reply's role.
  By condition; for ex2_other also against the donor role. Chance: J8 1/277; J10 1/276 and 5/276.

## Role scores (report only, no pass rule)

- On the 35 target items: owner score >= 2 and owner score = 3, each against "J8 names the role" and against
  "the role is in J10's top 5". Cohen's kappa with a 95% bootstrap interval, and the full cross-table.
- On the 15 close and distant items: how often each judge names the scored (wrong) role.

## Missing answers

- An item with no valid answer after 3 tries is missing and is left out of every number. If more than 5% of the
  items of a judge are missing, that judge's result is marked incomplete.

## Owner decisions after the run (23/10/2026)

- The κ ≥ 0.6 bar above has no recorded reason. Owner: "it makes no sense to make the bar k >= 0.6 there is no logic to it". The results above are kept as reported against it; the bar is not used for later decisions.
- Owner: "We can do both": (1) re-score E3 naming with each broken list (done, see NOTE.md), and (2) a neutral model relabels the owner's first 50 broken labels under the 16/10 definition (design open). A model-made label set is a judge output, not a human answer key; anything checked against it is exploratory.
