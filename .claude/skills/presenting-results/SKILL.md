---
name: presenting-results
description: Checklist for writing up empirical results (reports, Slack takeaways, figures) so they survive supervisor review. Use when drafting any result, takeaway, report section, figure, or Slack message about findings for Thomas or Trinidad, or when answering their questions about a result.
---

# Presenting Results

Distilled from the Aug 15–26 Thomas/Trinidad Q&A and the MATS
"Tips on Empirical Research" slides (LW post i3b9uQfjJjJkwZF4f).
Update LESSONS.md every time a reviewer question exposes a gap.

## Before posting any result

1. **Numbers come from artifacts, never memory.** Copy each number from
   its file in `output/` and name the file. Stale Slack numbers caused
   three corrections in one week.
2. **No qualitative claim without a number.** If unmeasured, say
   "impression, not measured" — or measure first. Thomas always asks
   "did you measure this?"
3. **One name per object.** Pick a name at first use, keep it. The
   default/assistant conflation cost three thread messages to repair.
4. **Define every metric at first use.** Plain name = technical name,
   what high/low means, expected direction. Keep a glossary line handy:
   bunched-ness = H0 total persistence, evenness = H0 persistence
   entropy, simplicity = MLE intrinsic dimension, rank = tightness
   percentile of the default among 276 clouds.
5. **Every number says what it is.** Data, stage, axis, controls.
   "−0.393" alone forced a follow-up question; "closeness vs evenness
   at DPO, own axis, controls on" would not have.
6. **Every takeaway names its result.** Link claim → figure/file →
   control. If two plots disagree on the story, say so yourself first.
7. **Surprise → confound check before posting.** Ask: did two things
   change at once? The RLVR-roleplay artifact was a prompt change riding
   on a stage change.
8. **Report confounds as findings.** "Close roles have tighter clouds"
   was worth stating on its own, not only controlling away.
9. **Trends are staircases, not switches.** Report the full ladder
   (base → SFT → DPO → RLVR) and let the reader see where it moves;
   "DPO makes it" undersold SFT's 0.44 cosine.
10. **State how the method maps to prior work.** One sentence: same as
    the paper / differs how. Difference-of-means vs PCA took a
    three-person thread to settle.
11. **Check the figure/table version before posting.** A wrong table
    version created the false RLVR-roleplay claim.
12. **Run the cheapest killing null first.** Before a claim carries a
    story, ask what one afternoon of compute could refute it, and run
    that. Q9 sat unrun for 19 days, then fired.
13. **Read the transcripts before the numbers.** For every model and
    condition: render the exact input with the run's own code, read 10
    random raw outputs, and measure that the intended behaviour happened
    (blind judge or parse rate) against a positive control. Base models
    read a chat template as plain text and never played the characters
    (Ivan, 01/10). Scripts: `output/results_2026-09/audit_2026-10-01/`.

## Figures and stats (from the MATS slides)

- Show the exact prompt next to the result.
- Error bars: SE = sqrt(p(1−p)/N), 95% CI = ±1.96·SE.
- Label axes with metric and direction ("lower = tighter").
- Print values on bars; ≤5 colors; bar charts over heatmaps.
- Strongest result first; minimal text per slide/section.
- End with next steps and concrete asks.

## Pre-empt the predictable questions

Thomas's pattern, in order of frequency — have these answers ready
before posting:

| He asks | Have ready |
|---|---|
| "what does X mean here?" | metric/term definition |
| "did you measure/check this?" | the number + its file |
| "can two causes be disentangled?" | one-variable-changed control |
| "what result backs this takeaway?" | claim → figure map |
| "same methodology as the paper?" | mapping sentence |

## Before sending any multi-result report

Run the loop in [report_gate.md](report_gate.md): ledger of earlier reports,
results inventory with importance scores, questions first, then
`gate_check.py` and a fresh-context reviewer until every gate passes (max 3
rounds). Never send a report that has only been checked by its author.

## Updating this skill

New reviewer question exposes a gap → add one dated line to
[LESSONS.md](LESSONS.md), and if it generalizes, fold it into the
checklist above. Keep SKILL.md under 100 lines.
