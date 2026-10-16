# Judge subagent instructions — judge prompt sweep (2026-10-14)

You label LLM answers. You get a judge directory JUDGE, ONE job name NAME and ONE batch name BATCH.

1. Read `JUDGE/sys/NAME.txt` in full. These are your judging instructions: the options, the definitions, the
   examples and the output fields. Follow them exactly. The examples only illustrate the format and the
   labels; they are not items to label.
2. Read `JUDGE/batches/BATCH.json` (a list of `{key, question, response}`) IN FULL. Use the Read tool; if the
   file is long, read it in parts until you have seen every item from start to end.
3. Judge EVERY item yourself by reading its whole response. Do NOT label with keyword searches, regexes,
   length heuristics, "by pattern" or any script. Many answers start "I am Qwen ..." and drift into a persona
   later, or break down halfway; read to the end of each one before deciding.
4. Write `JUDGE/labels/NAME.json` as ONE JSON object `{"labels": [ ... ]}`, with exactly one entry per item.
   Each entry is the JSON object the instructions ask for, with one extra first key `"key"` copied verbatim
   from the item, e.g. `{"key": "...", "evidence": "...", "analysis": "...", ...}`. Keep the field order the
   instructions give. A script may be used ONLY to write this file.

Scripts: if you write a helper script, give it a NEW filename that contains NAME and a random number
(e.g. write_<NAME>_<5 digits>.py) in the scratch directory, and first check that no file of that name exists (other agents share the directory and have left scripts there; running one of theirs overwrites another job's result). Use a random number of at least 8 digits and never run a file you did not create in this
task: other agents share that directory. Do not modify any other file.
Reply with one line: `NAME: <n labelled> of <n items>`.
