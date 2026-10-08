# Report gate: the loop every mentor report goes through

Type: goal-based loop. The stop condition is "every gate passes", checked by a script plus a fresh-context reviewer. Cap: 3 review rounds, then ship with the open FAILs listed for Brendon.

Why it exists: on 24 Sep 2026 one report went through eight rounds of Brendon finding holes (repeats of earlier reports, missing results, overstatements, weak takeaways, results that didn't answer the question, tropes, length). Each hole was a check nobody ran before sending.

## Stage 0: before drafting (unknown-finding, not writing)

1. **Prior-claims ledger.** Put the text of every earlier report and shared doc in `<writeup>/gate/prior/*.txt`. Search Drive for docs modified after the last report. The 11/09 report text is lost; its claims are listed in the `previous-weekly-reports` memory.
2. **Results inventory.** List every result produced since the last report in `<writeup>/gate/INVENTORY.md`, one row per finding, with its source file(s). Walk the result folder by modified date; do not work from memory.
3. **Importance triage (researcher lens).** Score each row 0-2 on N (new against the literature notes and earlier reports), C (causal or controlled), G (both families and layers), S (changes the story or next experiment), Sf (safety or monitoring). Lead with the highest totals.
4. **Questions first.** Group rows by the mentor question they answer (Q1 emergence, Q2 robustness, Q3 reward cases, method). One experiment per question. Write each concrete question before choosing its results; a row that answers no question is `out`.
5. **Status.** Every row is `in`, `context` (one clause, already known) or `out` (with a reason).
6. **Transcript audit.** Every row needs a transcript page: the exact rendered prompt per model, 10 random raw outputs per model and condition, and the intended behaviour measured (blind judge or parse rate) with a positive control. A row whose premise fails (the model did not do the thing the result assumes) is `out` or reworded. Reuse the scripts in `output/results_2026-09/audit_2026-10-01/`.

## Stage 1: draft

- Template: `Experiment: [short name]` / TLDR / Background / Concrete question / Methodology / Results (figure, one `Plot:` line, takeaways) / Global takeaways / Next steps / Questions.
- Tag each `### Result` with `<!-- inv: id, id -->` (pandoc drops it from the docx).
- Takeaways must do at least one of: say what the result means for the question; compare with a named paper from the literature notes (verify the line); give the strongest alternative explanation and the test that separates it; say what it changes next. A bullet that only restates a figure number fails.
- Scope every headline: self-written replies only, which family, which layers were measured, whether a random-direction control exists, shared post-training recipe.

## Stage 2: deterministic gates

```
.venv/bin/python .claude/skills/presenting-results/gate_check.py <report.md> \
  --inventory <writeup>/gate/INVENTORY.md --sources output/results_2026-09 \
  --ledger <writeup>/gate/prior --max-words 2200
```
G1 tropes, G2 length, G3 structure, G4 coverage must PASS. The default limit is 2,200 words for three experiments; raise it only when Brendon asks for more content (e.g. 2,800 on 25/09 for motivation and paper sources), and tell him the new count. G5 (numbers traced to the tagged sources) and G6 (numbers also in earlier reports) print WARN lines for the reviewer.

## Stage 3: fresh-context reviewer (Codex gpt-6-astra, effort xhigh)

Give it the report, INVENTORY.md, the prior ledger, the literature notes, CLAIMS_AND_TASKS "do not say" column, and the tropes list. It returns PASS or FAIL with line numbers for:
- J1 question fit: every result answers part of its experiment's concrete question, and every part of the question is answered or marked open.
- J2 takeaway quality: every bullet meets the Stage 1 rule; no point repeats across results.
- J3 scope: no overstatement against the "do not say" column or the scope list.
- J4 repeats: rule on each G6 WARN (new, context, or a real repeat).
- J5 importance: the strongest results lead; no inventory row scored 6+ is missing; nothing low-value is taking space.
- J6 tropes the regex can't see (rule of three, one-liners, self-echo, premise stacking, tie-backs).
- J7 reader test: from the TLDRs and figure titles alone, can a mentor say what was found, how sure, and what is next?
- J8 citations: every sentence that describes a paper matches the literature notes line by line (what they studied, which direction an effect goes, who recommended what).

Fix every FAIL, rerun Stage 2 and 3. Stop when all pass or after round 3.

## Stage 4: ship

Rebuild the docx (`pandoc -f markdown-smart X.md -o X.docx --resource-path=.`), import it to Google Docs through Codex (`google_drive_import_document`, native Google Docs), and send Brendon the link with the gate table.

## After shipping

Every hole Brendon still finds is a missing check. Add one dated line to LESSONS.md and a regex to `gate_check.py` or a line to the reviewer brief, so the next report can't repeat it.
