# Lessons log

One dated line per reviewer question that exposed a gap.
Fold recurring ones into SKILL.md.

- 2026-08-19 Thomas: "weird that RLVR teaches roleplay" → wrong table
  version + prompt/stage confound. Rules 7, 11.
- 2026-08-20 Thomas: "did you measure vivid or just qualitatively?" →
  claim was qualitative; measured next day, r≈0, claim retired. Rule 2.
- 2026-08-21 Thomas: "shouldn't assistant be rank 1?" → default vs
  assistant naming collision. Rule 3.
- 2026-08-22 Thomas: "what does 'controlled' mean here?" → control not
  defined where used. Rules 4, 5.
- 2026-08-23 Thomas: "not sure what −0.393 / −0.380 are" → bare numbers
  without object/axis/controls. Rule 5.
- 2026-08-24 Thomas: "SFT contributes too, no?" → staircase collapsed
  into a binary claim. Rule 9.
- 2026-08-24 Thomas: "same methodology as the paper? they did PCA" →
  mapping to prior work unstated; Trinidad had to settle it. Rule 10.
- 2026-08-25 Thomas: "'near roles have smaller clouds' is itself a
  finding no?" → confound controlled away instead of reported. Rule 8.
- 2026-08-25 Trinidad: "what are bunched-ness/evenness/simplicity/rank?"
  → invented names shipped without a glossary. Rule 4.
- 2026-08-26 self-audit: Slack quoted stale duplicate ladder and pirate
  shares vs verified artifacts. Rule 1.
- 2026-08-26 Q9: the reviewer's "cheap and decisive" null sat unrun for 19
  days, then fired — a sentence embedder on the answer text rebuilds the axis
  (0.92) and most of the tightness link. Nineteen days of writing rested on a
  claim one afternoon of compute could have qualified. New rule: run the
  cheapest null that could kill a claim BEFORE building the story on it, not
  after. See [[run-cheap-nulls-first]].
- 2026-08-31 meeting-notes correction: Granola assigned persona interpolation
  and centroid matching to Brendon, but both belonged to Giovanni. Treat
  generated meeting notes as a draft; confirm owners from the conversation.
- 2026-09-24 Brendon, eight rounds on one report: repeats of the 17/09 and 23/09
  docs (never read them), missing results (worked from memory, no inventory),
  overstatements (headline without scope), number-only takeaways, results that
  answered no stated question, tropes, length. This skill was not invoked at
  all. New: report_gate.md loop + gate_check.py; run it before any report.
- 2026-09-25 reviewer, after adding "why + source paper" sections: three paper
  descriptions drifted from the notes (Lu "only finished models", Moskvoretskii
  "harder to steer through post-training", a control attributed to Drake &
  Eberstadt that was our own idea). New gate J8: check every paper sentence
  against the notes' line.
- 2026-10-01 Ivan: "have you checked that the base model actually behaves like the persona?" -> no. Base models got the
  instruct chat template (plain text to them) and never played the characters (blind judge top-5 at or below chance).
  The audit then found the Exp 3 fine-tunes don't play characters either, and a regex that hid the SFT AI-disclaimer
  link. New rule 13 and gate Stage 0 item 6: transcripts and a behaviour check before any number.
- 2026-10-02 full-history audit: the two plain example turns added on 14/08 for base models went to every stage of every later
  run, and they stop trained models playing the character (OLMo-1B final: judge top-5 2.9% with them, 55.4% without). The
  August report had already said 'few-shot pulls roles together' and nobody acted on it. Also: corrected angles used
  split-half reliability without the Spearman-Brown step-up, so corrections overshot by up to 32 degrees. Rule 13 covers
  the first; for the second, step split-half reliability up (2r/(1+r)) before any disattenuation.
