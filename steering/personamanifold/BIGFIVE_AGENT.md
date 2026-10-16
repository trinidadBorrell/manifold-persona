# Big Five judge subagent instructions (PersonaManifold TS measure, 2026-10-16)

You get a directory DIR and ONE batch name NAME (e.g. bf_007).

1. Read `DIR/sys.txt` in full: your rating instructions.
2. Read `DIR/batches/NAME.json` (a list of `{key, question, response}`) IN FULL with the Read tool; if it
   is long, read it in parts until you have seen every item from start to end.
3. Rate EVERY item yourself by reading its whole response. Do NOT use keyword searches, regexes, length
   heuristics or any script to rate. Read to the end of each answer before deciding.
4. Write `DIR/labels/NAME.json` as ONE JSON object `{"labels": [...]}` with exactly one entry per item:
   `{"key": "<copied verbatim>", "evidence": "...", "openness": 1-5, "conscientiousness": 1-5,
   "extraversion": 1-5, "agreeableness": 1-5, "neuroticism": 1-5, "degenerate": true|false}`
   (integers). A script may be used ONLY to write this file; give it a new name containing NAME and a
   random number, in the scratch directory. Do not modify any other file.

Reply with one line: `NAME: <n rated> of <n items>`.
