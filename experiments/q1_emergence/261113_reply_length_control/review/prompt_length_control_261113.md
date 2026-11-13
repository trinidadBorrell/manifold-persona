My friend wrote this GPU job and already ran it. He wants a blunt adversarial code review: are the outputs correct and usable? Find what is wrong. Do not praise. Read the files; do not run GPU code. You may run the CPU tests.

Goal: a reply length control. OLMo-2-1124-7B Instruct, SFT and DPO each wrote 21,000 role replies (same prompts and seeds per key). Their stored states are means over all kept reply tokens, and SFT replies are much shorter (median 45 kept tokens vs 128). The job re-reads each stage's own stored replies (prompt ids + kept ids) with that stage's model and saves the mean over only the first 16 and the first 32 kept tokens, at hidden states 0, 9, 17, 23. The full-reply mean from the same pass must reproduce the stored mean (pod A's C0 gate). Rows shorter than the cut get NaN. A cut is "available" for a stage only if every role keeps at least half its rows.

Files (repo root = working directory):
- experiments/q1_emergence/261113_reply_length_control/THRESHOLDS.md (rules, written before the run), length_control.py (the job), modal_length.py (Modal launcher: fetch, smoke, full), test_length_control.py (6 CPU tests; run with `.venv/bin/python -m pytest experiments/q1_emergence/261113_reply_length_control/test_length_control.py`), NOTE.md (run record and results).
- Pod A code it must match: src/manifold_persona/rows.py (pool_spans, c0_stats, C0_GATE, row_key), experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py (how the stored states were made), gates.py (G4 uses the same C0 gate).
- The SFT and DPO runs: experiments/q1_emergence/261112_stage_own_replies/ (stage_run.py, NOTE.md). Pinned revisions: experiments/q1_emergence/261112_coinflip_stages/stages.json.
- Pod C's earlier length control (F4, first 32 ids) for comparison: experiments/q1_emergence/261029_fixed_text_read_through/THRESHOLDS.md and extract.py.
- Local copies of the outputs' small files: experiments/q1_emergence/261113_reply_length_control/full_copies/{instruct,sft,dpo}/ (check.json, support.json, manifest.json, keys.json, SHA256SUMS.txt). The rows are in experiments/q1_emergence/261029_instruct_persona_map/run_full/rows.jsonl and experiments/q1_emergence/261112_stage_own_replies/full_copies/{sft,dpo}/rows.jsonl.

Check, with file:line evidence for each finding:
1. Pooling: does pool_multi give exactly what rows.pool_spans would give for each span (padding side, attention mask, position ids, span bounds, the hidden-state index convention where 0 is the embedding output)? Are the first-N spans the first N kept reply tokens, not prompt tokens or an off-by-one?
2. Row order and keys: can any output row be misaligned with rows.jsonl, keys.json or the stored states (the length sort inside extract_core, fancy-index assignment, the --limit sample)?
3. The C0 check: is it a real test of the re-read, or could it pass while the cut states are wrong? Is it run on the right rows and states? Is the stored-state read from the memory-mapped file correct?
4. Support rule: does it match pod C's F4 rule? Are invalid rows handled correctly? Are the "available" flags in full_copies/*/support.json right, given the rows files?
5. Precision and storage: float16 casting, NaN handling, the float16 range check.
6. Launcher and provenance: input checksums, pinned revisions, the fetched pod A copy, the smoke-before-full rule, the attempt-1 cancellation (NOTE.md) and the rerun of two stages into the same folders. Could any output be from a mix of runs or partial?
7. Is the design a sound length control for comparing stages? What can it not show (for example: the first 16 tokens of a reply differ in content from the whole reply; stage differences in what the first tokens say)? What should the later comparison do to use it correctly?
8. Anything else that makes the outputs wrong or unusable.

Output: findings ranked by severity. For each: the claim, file:line evidence, why it matters, and the smallest fix. Say "none found" for a check with no finding. End with a one-line verdict: outputs usable as is / fix first.
