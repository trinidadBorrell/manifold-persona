# My transcript review of pod A and pod B (12/11/2026)

This is the note for `261029_stage_persona_plan/owner_review`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | Does the Terra judge agree with my own yes or no labels on 20 pod A replies and 20 pod B replies? |
| Result | I said yes to 16 of 20 pod A replies and to 3 of 20 pod B replies. Terra and I agree on 32 of 40 replies: 17 of 20 on pod A and 15 of 20 on pod B. |
| How much I trust it | The judge's labels were hidden until I finished. Every reply was checked against its source row. I am one reader with 20 replies per pod, so each count is rough. |
| What I do not know | How these counts hold with more replies or a second reader. |
| Status (12/11/2026) | Done. All 40 labels are in the Google Sheet tab "Review 12 Nov" and match the local file. |

---

## What I did

1. I chose the setup on 12 Nov: 20 replies per pod, my usual labels, a local labelling page, and an upload to the judge-work Google Sheet.
2. A script drew 20 pod A replies and 20 pod B replies with a fixed seed.
3. I labelled all 40 replies on the local page. The judge's labels stayed hidden until I finished.
4. My labels were compared with Terra's labels for the same replies.
5. The labels went to the Google Sheet. A read-back showed that all 40 rows match the local file.
6. I asked for a column that names the model per row. The prompt format of both pods was checked the same day.

---

## Results

### Terra and I agree on 32 of 40 replies

Terra's pass rule is: class E or P, and coherence at least 50.

| Pod | My yes | Terra's yes | Agree |
|---|---|---|---|
| A (chat) | 16 / 20 | 13 | 17 / 20 |
| B (base) | 3 / 20 | 4 | 15 / 20 |
| All | 19 / 40 | 17 | 32 / 40 |

- On pod A, Terra never said yes where I said no. It said no to 3 replies that I passed.
- On pod B, Terra said yes to 3 replies that I failed, and no to 2 replies that I passed.
- Two of those 3 extra pod B yes replies are ones that I marked U (cannot score): rows 2 and 37.
- Where I picked a class, Terra picked the same class on 16 of 35 replies.

### Each pod's prompts have the format that its model expects

- All 21,000 pod A prompts equal the Instruct model's official chat template.
- None of the 24,000 pod B prompts contains a chat tag, and each starts with one `<|endoftext|>`.

---

## How much I trust this

- The draw was random with a fixed seed, and the pod A items were never shown to me before.
- Every reply was checked against its row in `run_full/rows.jsonl`.
- I labelled before I saw any judge label.
- One reader and 20 replies per pod give rough counts only.

---

## Words used here

| Word | Meaning |
|---|---|
| Pod A | My run `../../261029_instruct_persona_map/`: OLMo-2-7B Instruct, chat format, with a role system prompt. |
| Pod B | My run `../../261029_base_role_vs_format/`: OLMo-2-7B base, plain text, with a role description in four formats. |
| Terra | The LLM judge, GPT-6.1 Terra. It gives each reply a class, a coherence score and a reason. |
| Classes E, P, G, W, U | The rubric classes of a reply: E in role, P partly in role, G not enough evidence, W a different role, U cannot score. |
| Pass | Terra's yes: class E or P, and coherence at least 50. |
| Practice labels | My 40 earlier yes or no labels that check the judge. |
| `pop_poda`, `pop_podb` | The sets of Terra labels on all pod A replies and all pod B replies. |
| Chat tag | One of the system, user and assistant markers of the OLMo-2 chat template. Section 5 gives their exact text. |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

## 1. My choices (12 Nov)

20 replies per pod; labels in the established scheme (yes/no as in the 40 practice labels, optional rubric class E/P/G/W/U, note); local labelling page; labels uploaded to the judge-work Google Sheet.

## 2. Sample

- Pod A (OLMo-2-7B Instruct, chat, role system prompt): one reply per role from the 80 sealed pod A items (random, never shown to me).
- Pod B (OLMo-2-7B base, plain text, role description): two replies per role, five per format, each a random valid reply.
- Seed 20261112. Draw: `draw_review_261112.py`. Every reply was checked against its row in `run_full/rows.jsonl`.
- Files (outside the repo): `/Users/brendon/persona-manifolds/judge_validation_261029/review261112_items.jsonl` (sha256 `3cffd1347421…`), `review261112_key.json` (source rows, exact model input, raw continuation), `review261112_owner_labels.json` (saved 12 Nov 14:13 UTC, 40 of 40).
- Page: `label_sheet.py serve` (launch config `judge-labels-review261112`, port 8767). The judge's labels were hidden until I finished.

## 3. Result: my labels against Terra (pass = E or P and coherence at least 50)

| Pod | My yes | Judge yes | Both yes | Agree | Judge yes but my no | My yes but judge no |
|---|---|---|---|---|---|---|
| A (chat) | 16 / 20 | 13 | 13 | 17 / 20 | 0 | 3 |
| B (base) | 3 / 20 | 4 | 1 | 15 / 20 | 3 | 2 |
| All | 19 / 40 | 17 | 14 | 32 / 40 | 3 | 5 |

- Class agreement: 16 of the 35 replies where I picked a class.
- Pod B: two of the judge's three extra yes replies are ones that I marked U (cannot score): rows 2 and 37.
- One reader, 20 replies per pod: each count is rough. Terra labels come from `pop_poda` (matched by source row) and `pop_podb` (same ids).

## 4. Google Sheet

Tab "Review 12 Nov" in "Judge work - how we decide if a model plays a role (manifold-persona)", https://docs.google.com/spreadsheets/d/1Rje1OqpfCvDjii6KqC_9ujpvMHRi4KnJVXfNDVEFozI.

- One row per reply: exact model input, reply, my label, class, note, judge label, class, coherence, reason, agreement; summary below.
- Read back on 12 Nov: all 40 rows match the local file (short fields exact; lengths of the input and reply texts equal).

## 5. Model column and prompt format check (12 Nov)

My request: column G, "Model that got this input", names the model per row: base (`allenai/OLMo-2-1124-7B`) for pod B, Instruct (`allenai/OLMo-2-1124-7B-Instruct`) for pod A.

Checked the same day:

- All 21,000 pod A prompts equal the Instruct model's official chat template (template sha256 `fe689ffb…`, as in the run manifest).
- None of the 24,000 pod B prompts contains a chat tag, and each starts with one `<|endoftext|>`.
- In OLMo-2, `<|system|>`, `<|user|>` and `<|assistant|>` are plain text pieces, not special tokens.
