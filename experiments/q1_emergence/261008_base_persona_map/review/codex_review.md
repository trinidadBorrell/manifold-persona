# E3 code review

## Result

I found 14 findings. Eight are code defects. Six are design concerns.

## Confirmed bugs

1. **High — `gen_extract.py:153-170` — The activation tokens are not the generated tokens.**

   - **What is wrong:** The code decodes generated IDs, strips text, and tokenizes the text again. This can change token IDs and whitespace.
   - **Failing case:** Token `382` decodes as `".\n"`. Cutting before `Interviewer:` gives `"."`, which tokenizes as token `13`.
   - **Smoke evidence:** Item `331` has this boundary pattern. All 40 raw texts re-tokenize to their saved generation counts.
   - **Other failure:** A reply that starts with two spaces becomes a reply that starts with one space.
   - **Suggested fix:** Keep the generated ID prefix from `out`. Define stop cuts on token boundaries, and save the kept token IDs.
   - **Suggested fix:** Decode those saved IDs only for display. Do not remove every sampled pad token from the generated prefix.

2. **High — `gen_extract.py:136-143,171-174,206-217` — Resume can silently use stale data.**

   - **What is wrong:** Resume checks only item IDs. It does not check prompts, generation settings, code, or part contents.
   - **Failing case:** Run once, change temperature or prompt text, and use the same output directory. Every completed batch is skipped.
   - **Result:** The manifest records the new settings, but replies and activations still use the old settings.
   - **Suggested fix:** Store and check a run fingerprint before resume. Include prompt hashes, item IDs, model revision, settings, and code revision.
   - **Suggested fix:** Store item IDs beside each activation part. Write each reply part and activation part as one atomic unit.

3. **Medium — `gen_extract.py:162-170` — The activation pass uses wrong position IDs after left padding.**

   - **What is wrong:** The direct model call gets an attention mask, but it gets no mask-based `position_ids`.
   - **Failing case:** A 50-token sequence in a 263-token batch starts at position 213, not position zero.
   - **Effect:** Generation and activation extraction do not use the same positions. Results can also depend on batch composition.
   - **Suggested fix:** Compute positions with `mask2.cumsum(-1) - 1`. Mask pad positions, and pass `position_ids` to the model.

4. **Medium — `gen_extract.py:46-59` — The third-person conversion damages valid descriptions.**

   - **What is wrong:** The code changes every `you` to `them`. It does not handle subject and object grammar.
   - **Failing case:** Pirate wording 4 becomes `taking what them want`. Teenager wording 2 becomes `who them are`.
   - **Smoke evidence:** These bad prompts occur at `smoke/items.jsonl:22`, `:26`, and `:27`.
   - **Suggested fix:** Store reviewed third-person descriptions. Do not make them with a global pronoun replacement.

5. **Medium — `auto_examples.py:75-107` — The filters do not implement their written rules.**

   - **What is wrong:** `AI_RE` rejects any AI topic. The rule rejects only AI self-reference.
   - **Failing case:** `AI assistants can help people, but they also create risks.` is not self-reference, but it fails.
   - **What is wrong:** The list rules miss common bullets and lettered lists.
   - **Failing case:** A 30-word reply beginning with `• First point` passes `LIST_RE` and `FORMAT_RE`.
   - **What is wrong:** The name rule adds only `s` or `es`. It misses other plural forms.
   - **Failing case:** A void reply that says `between realities` passes the `between reality` name rule.
   - **Suggested fix:** Add tests for positive and negative cases. Use self-reference patterns and explicit name variants.
   - **Suggested fix:** Reject Unicode bullets and standard ordered-list forms.

6. **Medium — `auto_examples.py:147-160`; `run_judge.sh:12` — Judge output has no schema check.**

   - **What is wrong:** A nonempty answer file is treated as complete. `pick()` checks only missing IDs.
   - **Failing case:** `{"id": ["pirate"]}` can give rank one. Duplicate, unknown, or fewer than five names can also pass.
   - **Suggested fix:** Require exact batch keys. Require five distinct strings from `candidates.txt` for each key.
   - **Suggested fix:** Run this check before resume skips a batch and before `pick()` scores it.

7. **Medium — `run_judge.sh:13-20` — Judge command failures are reported as success.**

   - **What is wrong:** `job()` runs `echo` after Codex. The echo usually makes the function return zero.
   - **Failing case:** Codex exits with status one and writes no answer. The script still prints `__JUDGE_DONE__`.
   - **Suggested fix:** Check the Codex status directly. Validate the answer file, and return a nonzero status on failure.

8. **Low — `auto_examples.py:25,115-129` — Fixed-seed preparation is not fully repeatable.**

   - **What is wrong:** Reply sampling uses a fixed NumPy seed, but item IDs use `secrets.token_hex()`.
   - **Failing case:** Two `prep` runs select the same replies but create different pools, batches, and answer keys.
   - **Suggested fix:** Make IDs from stable fields and a content hash. Check for a hash collision.

## Design concerns

9. **High — `NOTE.md:50-54`; `gen_extract.py:86-109` — The question sets confound the persona map.**

   - **What is wrong:** The note requires the 200 questions outside the example bank. The code instead uses the 40-question bank.
   - **Failing case:** Each role omits its own two example questions. Thus, different roles use different 38-question sets.
   - **Failing case:** Default roles use all 40 questions. Role and default means do not use matched question sets.
   - **Suggested fix:** Use the same 200 held-out questions for every role, condition, wording, and default.

10. **Medium — `gen_extract.py:68-72` — Condition effects include large length and topic effects.**

   - **What is wrong:** `ex2` adds two role-specific questions and answers. It also adds about 180 prompt tokens.
   - **Failing case:** Role prompts average 52.7 tokens in `desc` and 232.2 tokens in `ex2`.
   - **Effect:** An activation difference can reflect length, example topics, or role evidence.
   - **Suggested fix:** Add a length-matched control with non-role examples. Use common example questions across roles where possible.
   - **Suggested fix:** Limit claims if the study does not add these controls.

11. **High — `run_judge.sh:3-15`; `auto_examples.py:133-141` — The blind judge can read the answer keys.**

   - **What is wrong:** The judge runs in the full repository. The same run directory contains `pool.json` with ID-to-role mappings.
   - **Failing case:** The judge can open `v2/pool.json` and map each opaque ID to its true role.
   - **Evidence:** `NOTE.md:48` says the judge already ignored the file-access rule for other files.
   - **Suggested fix:** Run the judge in an isolated directory. Give it only the prompt, candidates, batch, and output path.

12. **Medium — `auto_examples.py:160-185` — Selection and later evaluation can be circular.**

   - **What is wrong:** Examples are selected by judge rank. Replacement roles also require two rank-one results.
   - **Failing case:** Easy roles replace weak roles. A later result from the same judge is biased upward by construction.
   - **Suggested fix:** Use one judge for selection and an independent judge or blinded humans for evaluation.
   - **Suggested fix:** Do not make population claims from this selected pilot.

13. **Medium — `gen_extract.py:167-170,214-217` — Empty replies can silently poison later analysis.**

   - **What is wrong:** Empty replies become all-NaN rows. The manifest does not record their IDs or count.
   - **Failing case:** One empty reply makes a normal mean, covariance, or decomposition return NaN values.
   - **Smoke evidence:** The 40 smoke rows are finite. This does not protect the full run.
   - **Suggested fix:** Save a validity mask and an empty-row count. Require finite rows before persona-map analysis.

14. **Medium — `gen_extract.py:147-149,214-217` — Sampling and provenance do not support exact reproduction.**

   - **What is wrong:** Each batch uses its start index as one shared seed. An item's sample depends on batch size and batch position.
   - **Failing case:** Change batch size from 20 to 32. Boundary items receive a different seed and random-number stream.
   - **What is wrong:** The manifest omits Python, PyTorch, Transformers, CUDA, tokenizer revision, code revision, and the seed rule.
   - **Suggested fix:** Use stable per-item seeds or lock batch construction. Record all software and code versions.

## Checks that passed

| Check | Result |
|---|---|
| Plain text and no chat template | Pass |
| No automatic BOS token | Pass |
| Exactly one starting `<|endoftext|>` in smoke prompts | Pass |
| Left-pad values and attention masks | Pass |
| Reply suffix slice after left padding | Pass, subject to findings 1 and 3 |
| Exact example question excluded for each role and condition | Pass |
| Smoke reply order equals smoke item order | Pass |
| Smoke activation shape | Pass: `40 x 33 x 4096` |
| Smoke activation values | Pass: no NaN or infinity |
| Stored v2 and v3 judge answer shape | Pass for current files |
