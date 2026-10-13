# E3 `gen_extract.py` re-review

## One-line result

Four requested findings remain partly fixed. I found seven distinct open issues.

---

## Status

| Finding | Status | Main result |
|---|---|---|
| 1 | PARTLY FIXED | Activations use generated IDs, but saved `reply` text does not match those IDs. |
| 2 | PARTLY FIXED | The fingerprint works for normal runs, but resume can still accept stale or incomplete files. |
| 3 | FIXED | The activation pass uses right padding and the reply slice is checked. |
| 4 | PARTLY FIXED | The reported pronoun errors are gone, but one reviewed description is still ungrammatical. |
| 9 | PARTLY FIXED | `desc` is matched, but `ex2` still uses a different question set for each role. |
| 13 | FIXED | Invalid replies have a saved flag, NaN rows, counts, IDs, and final checks. |
| 14 | FIXED | Batch construction is locked and the requested provenance fields are present. |

---

## Finding 1 — PARTLY FIXED

The important activation error is fixed. `cut_ids()` keeps generated IDs and stops at EOS, pad, or a marker boundary (`gen_extract.py:85-101`).

The activation input uses those IDs directly (`gen_extract.py:188-207`). It does not tokenize decoded reply text again.

The smoke data has 34 marker stops and 6 limit stops. No decoded `reply_ids` sequence contains either stop marker.

However, exact decoding fails for all 40 smoke rows. `tok.decode(reply_ids).strip() == reply` holds for 40 rows, but exact equality holds for 0 rows.

The cause is the unconditional `.strip()` at `gen_extract.py:193`. For example, the IDs at `smoke/replies.jsonl:1` decode with a leading space, while `reply` has no leading space.

Pad handling also loses provenance. A pad stop is reported as `ended_by="eos"` at `gen_extract.py:89-92`, and `raw` removes every pad ID at `gen_extract.py:192`.

The smoke run cannot test EOS or pad endings because it has none. A direct `cut_ids()` check confirms that a pad-first sequence returns `ended_by="eos"`.

---

## Finding 2 — PARTLY FIXED

The fingerprint covers the model, revision, generation settings, batch size, code, item IDs, and prompts (`gen_extract.py:150-157`). A mismatch stops the run (`gen_extract.py:161-165`).

The smoke fingerprint recomputes exactly. Its code hash also matches the current file and `smoke/manifest.json:27-28`.

Resume still trusts old artifacts when `fingerprint.txt` is absent. The code writes a new fingerprint before it reads `replies.partial.jsonl` (`gen_extract.py:162-176`).

This case matters because outputs from the old code have no fingerprint. New code can silently mark stale old rows as complete.

Reply and activation parts are still not one atomic unit. The code saves the activation part, then appends reply rows separately (`gen_extract.py:208-211`).

Resume checks only reply IDs at `gen_extract.py:176-180`. It does not check that each completed reply batch has a matching, readable activation part before it skips the batch.

There is also a new destructive ordering error. `main()` replaces `items.jsonl` and `prompts_check.txt` before `run()` checks the fingerprint (`gen_extract.py:227-241`).

Thus, a rejected run can damage the metadata for the valid run already in that output directory.

---

## Finding 3 — FIXED

The second pass uses right padding at `gen_extract.py:198`. Real tokens therefore start at position zero.

The model receives the correct right-padding mask at `gen_extract.py:199-200`. The reply slice uses `[prompt_tokens:prompt_tokens + reply_tokens]` at `gen_extract.py:203-207`.

The ID assertion at `gen_extract.py:206` directly checks slice alignment. The smoke array has shape `(40, 33, 4096)`, type `float16`, and only finite values.

---

## Finding 4 — PARTLY FIXED

The original pirate and teenager errors are fixed. The teenager result uses “who they are” at `descriptions_third_person.json:310-312`.

All 100 runtime conversions match the 100 stored results. No stored result contains `you`, `your`, `yours`, or `yourself` as a word.

One result is still bad. Robot wording 0 becomes “a robot with advanced computational capabilities and process information” at `descriptions_third_person.json:262-265`.

It needs “processes information.” The prefix removal at `gen_extract.py:50-58` changes the subject but does not correct verb agreement.

The reviewed JSON file is not an input. `build_items()` still calls the heuristic function at `gen_extract.py:117-125`, so a correction in the JSON file would have no effect.

---

## Finding 9 — PARTLY FIXED

Every normal role and every default now uses all 40 questions in `desc` (`gen_extract.py:119-129`). The generated item set confirms 40 questions for each role, wording, and default.

Each `ex2` role still drops its own two example questions (`gen_extract.py:115-123`). The 20 roles therefore have 20 different 38-question sets.

This still confounds role means in the `ex2` persona map. The code comment only makes within-role condition comparisons matched (`gen_extract.py:9-11`).

The code also still uses the 40-question example bank at `gen_extract.py:106-107`. This conflicts with the held-out 200-question requirement at `NOTE.md:50-54`.

---

## Finding 13 — FIXED

Each row gets a `valid` flag at `gen_extract.py:196`. Invalid rows stay NaN because only valid rows receive means (`gen_extract.py:202-207`).

The final checks require finite valid rows and all-NaN invalid rows (`gen_extract.py:250-251`). The manifest records the invalid count, IDs, and row rule (`gen_extract.py:260-263`).

The smoke run has 40 valid rows and no invalid rows. It verifies the finite path, but it cannot exercise the invalid path.

---

## Finding 14 — FIXED

The batch size is part of the fingerprint at `gen_extract.py:153`. Changing it now prevents resume into the old run.

The seed rule remains batch-based at `gen_extract.py:177-184`, but the locked item order and batch size make it reproducible for that fingerprint. The manifest states this rule at `gen_extract.py:257-260`.

The manifest records Python, PyTorch, CUDA, Transformers, Tokenizers, and NumPy versions (`gen_extract.py:264-267`). It also records model and tokenizer revisions, settings, fingerprint, and code SHA-256 (`gen_extract.py:257-267`).

The smoke manifest contains all these fields at `smoke/manifest.json:2-35`. Its code SHA-256 matches the current script.

---

## Smoke checks

| Check | Result |
|---|---|
| Reply and item order | PASS, 40 of 40 |
| `len(reply_ids) == reply_tokens` | PASS, 40 of 40 |
| Exact `decode(reply_ids) == reply` | FAIL, 0 of 40 |
| Stripped decode equals `reply` | PASS, 40 of 40 |
| Kept decode contains a stop marker | PASS, 0 of 40 |
| Kept IDs contain EOS or pad | PASS, 0 of 40 |
| Stop results | 34 marker, 6 limit, 0 EOS, 0 pad |
| Activation shape and type | PASS, `(40, 33, 4096)`, `float16` |
| Valid activation values | PASS, all finite |
| Fingerprint and code hash | PASS, both match current code and inputs |

---

## New bugs introduced by the changes

1. **Medium — saved text and IDs disagree.** `reply` strips whitespace from `reply_ids` at `gen_extract.py:193`; all 40 smoke rows fail exact decoding.
2. **Low — pad provenance is false.** A pad stop becomes `ended_by="eos"`, and `raw` removes pad IDs at `gen_extract.py:89-92,192`.
3. **Medium — a fingerprint rejection changes old metadata.** `items.jsonl` and `prompts_check.txt` are replaced before validation at `gen_extract.py:227-241`.

---

## Open issues

| ID | Open issue |
|---|---|
| O1 | Make `reply` an exact decode of `reply_ids`, or document and save the display transform separately. |
| O2 | Report pad endings as pad and preserve the true raw generated IDs or text. |
| O3 | Reject existing run artifacts when the fingerprint file is absent. |
| O4 | Save and validate reply and activation parts as one atomic batch unit. |
| O5 | Check the fingerprint before replacing any existing run metadata. |
| O6 | Fix robot wording 0 and use the reviewed third-person data as the source. |
| O7 | Use one matched held-out question set for every role in each persona map. |
