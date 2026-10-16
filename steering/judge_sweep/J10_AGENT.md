# J10 judge subagent instructions (Giovanni's ranking judge, exact port)

You get a directory DIR and ONE chunk number N (three digits, e.g. 007).

1. Read `DIR/chunk_N.txt` IN FULL with the Read tool. It is the complete judging prompt followed by
   15 numbered answers. Follow the prompt exactly.
2. Judge EVERY numbered item yourself by reading its text. Do NOT use keyword searches, regexes or any
   script to label.
3. Write `DIR/verdict_N.json`: exactly the STRICT JSON the prompt asks for, one key per item number
   ("1" ... last), each `{"why": "...", "rank": [10 distinct names from the list], "nonsense": true|false}`.
   A script may be used ONLY to write this file; give it a new name containing N and a random number,
   in the scratch directory. Do not modify any other file.

Reply with one line: `N: <n judged> of <n items>`.
