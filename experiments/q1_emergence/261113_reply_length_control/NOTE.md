# Reply length control for the stage replies (13/11/2026)

This is the note for `261113_reply_length_control`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | Are the states of SFT, DPO and Instruct comparable when their replies have different lengths? This run makes states over the same number of tokens for every reply: the first 16 and the first 32. |
| Result | The first 16 tokens are the one cut that all three stages support. The first 32 tokens can compare only Instruct and DPO, because too many SFT replies are shorter. The re-read reproduces each stage's stored states. This run computes no metric on the replies. |
| How much I trust it | The rules and the code were done before any run. Every stage passes the C0 check. A Codex review found no error and judged the "outputs usable as is". Its three findings are fixed. A new check that reloads the saved arrays passes for all three stages. |
| What I do not know | Whether the stages differ. The comparison itself is in the own-reply analysis. Equal token counts do not remove content differences, so a claim that does not depend on content needs the fixed-text comparison (pod C). |
| Status (13/11/2026) | Done: the full run, the Codex review, the fixes, the saved-array check and the HF copies. The cost was $2.03 on Modal. |

---

## What I did

1. The rules and the code were written before any run. The mode is de-risk: one forward-pass job that answers one question.
2. A Modal job fetched pod A's rows and states from HF. A smoke run on 64 rows per stage passed.
3. I gave the go for the full run: "lets run these length controls". Each stage read its own 21,000 replies again with its own model.
4. Attempt 1 finished SFT only, because my Mac's Modal client disconnected. Attempt 2 ran Instruct and DPO detached, with the same code.
5. Codex reviewed the run ("Yes, review it"). I had its three findings fixed ("Yes, fix them").
6. The outputs went to HF and were checked.

---

## Results

### The first 16 tokens work for all three stages, and the first 32 do not work for SFT

A cut is available for a stage only if every role keeps at least half of its rows. The table gives the role that keeps the smallest share.

| Stage | first16: lowest role share | Rows kept | first32: lowest role share | Rows kept |
|---|---|---|---|---|
| Instruct | toddler 0.965, available | 20,925 | toddler 0.796, available | 20,549 |
| SFT | toddler 0.576, available | 19,153 | toddler 0.186, not available | 14,512 |
| DPO | translator 0.978, available | 20,951 | toddler 0.843, available | 20,650 |

- At SFT, cynic also fails first32 (0.340).
- This is what the rules expected from a count of the rows before the run.
- "Available" is a check of role coverage only. An analysis that uses a cut must also check its halves and its draws.

### The re-read reproduces each stage's stored states

The C0 check compares the full reply mean from the re-read with the stored reply mean. The numbers are for state 17.

| Stage | Min cosine | Median cosine | Median relative norm | Check |
|---|---|---|---|---|
| Instruct | 0.99984 | 0.99999 | 0.0030 | Pass |
| SFT | 0.99497 | 0.99998 | 0.0037 | Pass |
| DPO | 0.99955 | 0.99999 | 0.0028 | Pass |

- States 9 and 23 also pass in every stage.
- C0 passes on medians and a minimum cosine. The largest relative error of a single row is 0.083 (SFT, state 17).

### The cut values themselves pass an exact-length check

- When a reply has exactly 16 or 32 kept tokens, the cut is the whole reply. So its cut state must equal the stored reply mean.
- All four states pass pod A's C0 gate in every stage and cut.
- This is the first check of the cut values themselves, not only of the full mean.

---

## How much I trust this

- The rules were fixed before any run, and the row counts matched what the rules expected.
- Codex found no error in pooling, row order, the C0 check, invalid rows, float16 handling or provenance.
- `length_control.py` did not change after the run. Its hash is in every manifest.
- The saved arrays were reloaded and checked against the rows: the finite rows are exactly the eligible rows in every stage and cut.
- The smoke and the full runs used two forms of the A100 card. The fingerprint rule changed after the result to leave the card form out. The C0 check and the exact-length check show that the card did not change the states.
- No human has reviewed this run.

---

## Open points

- The comparison itself (each planned measure on full replies against first16, on the same rows) is set in the own-reply analysis rules.
- The own-reply analysis v1.2 checks half support. It records empty draws but does not stop on them.

---

## Words used here

| Word | Meaning |
|---|---|
| Stage | One step of OLMo-2-7B post-training: SFT, then DPO, then Instruct. |
| Own replies | The replies that each stage wrote itself (`../261112_stage_own_replies/`). Instruct's replies come from pod A. |
| Pod A | My run `261029_instruct_persona_map`: 21,000 Instruct replies with their stored states. |
| Cut | A state averaged over the first N kept tokens of a reply, in place of all kept tokens. |
| first16, first32 | The two cuts: the first 16 and the first 32 kept tokens. A shorter reply gets no state for that cut (NaN). |
| Re-read | A forward pass over the stored prompt and reply tokens, with no new generation. |
| C0 check | Pod A's gate for a re-read: it must reproduce the stored reply mean (min cosine at least 0.95, median cosine at least 0.99, median relative norm difference at most 0.02). |
| Support, available | A cut is available for a stage if every role keeps at least half of its rows. This is the role-share part of pod C's F4 rule. |
| Eligible row | A valid row whose reply has at least N kept tokens. |
| Exact-length check | A check on replies with exactly N kept tokens, where the cut and the whole reply are the same. |
| Smoke | A small test run (64 rows per stage) before the full run. |
| Fingerprint | The code hashes, input hashes, model, revision and versions of a run. The full run must match the smoke. |
| Pod C | My run `261029_fixed_text_read_through`, where every stage reads the same text. |
| Modal | The cloud service where the jobs ran. |
| HF | The private Hugging Face dataset `brend0nc/manifold-persona-experiment-artifacts` that keeps my run files. |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

Rules: `THRESHOLDS.md` v1.0. De-risk mode: one forward-pass job that answers one question (are the stage states comparable when the reply lengths differ?), est. $1 to $2.

## 1. Code and steps

Code: `length_control.py` (re-read and pooling; `pool_multi` is pod A's `rows.pool_spans` with several spans from one forward pass), `modal_length.py` (fetch, smoke, full), `test_length_control.py` (6 pass on CPU: `pool_multi` equals `rows.pool_spans`; cut spans; NaN rows; the C0 check passes on equal states and fails on other states; the support rule; the real-row support counts match the rules). Added after the Codex review: `verify_outputs.py` (reads the saved outputs again; launcher `--step verify`) and `test_verify_outputs.py` (10 pass on CPU); see "Fixes after the Codex review" (section 4).

| Step | Where | Date | State |
|---|---|---|---|
| Rules and code | here | 13/11 | Done, before any run |
| Fetch pod A rows and states from HF @ `60a81b9b…` | Modal CPU, app `ap-VWp9k58aThGvV68PcEziAA` | 13/11 | Done: SHA256SUMS.txt matches the HF receipt; rows, states and manifest match it. Volume path `poda_dl/experiments/q1_emergence/261029_instruct_persona_map/run_full/`. Log `fetch.log` |
| Smoke: 64 rows per stage (fixed sample, seed 0), three stages | Modal, three A100-80GB, app `ap-qLIrtiDBDo4Rk2vt5Ha1uj` | 13/11 | All three pass the C0 check: min cosine 0.99999 to 1.0, median relative norm 0.0002 at states 9, 17 and 23. 27 to 37 s per stage. The support numbers on 64 rows are too small to mean anything (one or two rows per role); the full run decides. Log `smoke.log` |
| Full, attempt 1: 21,000 rows per stage | Modal, three A100-80GB, app `ap-NqbIb6PnLkN3KXq6U4AC38` (not detached) | 13/11 | Started on my go ("lets run these length controls"). SFT done: 221 s, C0 check passes (state 17: min cosine 0.99497, median 0.99998, median relative norm 0.0037; states 9 and 23 similar), first16 available (lowest toddler 0.576, 19,153 rows), first32 not available (toddler 0.186, cynic 0.340; 14,512 rows), as expected in the rules. Instruct and DPO were cancelled at batch 251 of 657: the Mac's Modal client disconnected at 06:11:40 UTC and Modal stopped the app (container log: "Stopping app - local client disconnected"). They wrote no outputs. Each stage printed one CUDA allocator warning (a failed 1 GB allocation, then a retry); SFT finished after its warning, so the warning did not stop the run. Logs: `full.log`, `full_attempt1_logs/`; SFT small files copied to `full_copies/sft/` |
| Full, attempt 2: Instruct and DPO only, same code | Modal, two A100-80GB, detached, app `ap-wrkPzZPi2d4ewmqycLtywu` | 13/11 | Done. Both pass the C0 check. Log `full_v2.log`; small files copied to `full_copies/{instruct,dpo}/`, checksums match |
| Saved-array check, after the Codex review | Modal CPU, app `ap-44escgpkdtdIDP1k1K1jrg` (attempt 2) | 13/11 | Done. All three stages pass. See "Fixes after the Codex review" (section 4) |

## 2. Full run: check and support (no metric computed)

| Stage | Rows | Time | C0 check, state 17: min cos / median cos / median rel. norm | first16: lowest role share, rows | first32: lowest role share, rows |
|---|---|---|---|---|---|
| Instruct | 21,000 | 346 s | 0.99984 / 0.99999 / 0.0030, pass | toddler 0.965, 20,925, available | toddler 0.796, 20,549, available |
| SFT | 21,000 | 221 s | 0.99497 / 0.99998 / 0.0037, pass | toddler 0.576, 19,153, available | toddler 0.186, 14,512, not available |
| DPO | 21,000 | 357 s | 0.99955 / 0.99999 / 0.0028, pass | translator 0.978, 20,951, available | toddler 0.843, 20,650, available |

- States 9 and 23 also pass in every stage (`full_copies/*/check.json`).
- So: the re-read reproduces each stage's stored states, and first16 is the one cut that all three stages support. first32 can compare only Instruct and DPO.
- Outputs per stage: `first16_acts.npy` and `first32_acts.npy` [21000, 4, 4096] float16 (states 0, 9, 17, 23; NaN where the reply is shorter than the cut), rows in `rows.jsonl` order (`keys.json`).
- Next: the comparison itself (each planned measure on full replies against first16, on the same rows) is set in the own-reply analysis rules.

## 3. Codex review (13/11/2026, after the run)

I asked for it ("Yes, review it"). Prompt `review/prompt_length_control_261113.md`, review `review/codex_length_control_261113.md` (read-only sandbox, reasoning xhigh; log beside it). Verdict: "outputs usable as is". No error found in pooling, row order, the C0 check itself, invalid rows, float16 handling or provenance. Three findings, each checked by Claude against the code here:

| # | Severity | Finding | Claude's check | Effect on these outputs |
|---|---|---|---|---|
| 1 | Medium | The save step does not assert which rows of a cut must be finite; an all-NaN cut would pass C0 and the save guard (`length_control.py:173–177`). | Confirmed. | None found: an ineligible row is NaN by construction, and each saved `n_finite_rows` equals the eligible count, so the finite rows are exactly the eligible rows. The saved arrays themselves were not reloaded at that time. |
| 2 | Low | The full run checks only the smoke's `length_control.py` hash, not `rows.py`, inputs, revision or runtime (`modal_length.py:97–101`). | Confirmed. | None: Codex found the recorded code and input hashes match. |
| 3 | Low | "Pod C's F4 rule" names only its role-share part; pod C also needs common-question support in fixed halves and bootstrap draws. | Confirmed (pod C `THRESHOLDS.md` lines 23 to 25). | The flags are role-coverage checks. Half and draw support must be enforced in the analysis. Codex's CPU check with pod C's fixed-half code found no extra failure. |

- Design advice for the later comparison: one common key mask across the compared stages; report full-reply results before and after that mask (selection effect); same units and paired resamples; first32 only for Instruct vs DPO. Equal token counts do not remove content differences, so a claim independent of content needs the fixed-text comparison (pod C).
- Note: C0 passes on medians and a minimum cosine; the largest single-row relative error is 0.083 (SFT, state 17).

## 4. Fixes after the Codex review (13/11/2026)

My words: "Yes, fix them". `length_control.py` is unchanged; its hash `11076616…` is in every manifest.

| Finding | Fix | Result |
|---|---|---|
| 1 | New `verify_outputs.py` (reads only) and launcher step `--step verify` (Modal CPU). It reloads the saved arrays and checks them against the rows. `test_verify_outputs.py`: 10 pass on CPU. Good outputs pass. A NaN eligible row, a partly NaN row, wrong values on exact-length rows, equal cut vectors, a changed file, a wrong dtype and a changed library version each fail. A different GPU card form passes and is recorded | All three stages pass (table below). Log `verify.log`, records `verify/{stage}.json` |
| 2 | Launcher full step: before the run, `smoke_gate` compares the smoke's code hashes (`length_control.py`, `rows.py`), input hashes, model and revision with the full run's. After the run, the full manifest must match the smoke's fingerprint: every field except row count, limit, results, time and GPU card form. The launcher as it ran on 13/11 is kept in `review/code_before_fixes_261113/modal_length.py` (sha `7e43e111…`) | `verify_outputs.py` also checks the fingerprint after the fact: smoke and full match in all three stages |
| 3 | `THRESHOLDS.md` v1.1, wording only: "available" is a role-coverage check, and an analysis that uses a cut must enforce half and draw support | The own-reply analysis v1.2 checks half support (`support_check` in `../261112_stage_own_replies/analysis_v12.py`: every cloud keeps rows in every wording and in each half of the 50 fixed splits). It records empty draws (`dropped_draws`) but does not stop on them |

| Stage | Cut | Finite rows = eligible rows | Partly finite rows | Rows with exactly N kept tokens | Exact-length check, state 17: min cos / max rel. norm |
|---|---|---|---|---|---|
| Instruct | first16 | 20,925 = 20,925 | 0 | 17 | 0.99991 / 0.0073 |
| Instruct | first32 | 20,549 = 20,549 | 0 | 26 | 0.99996 / 0.0053 |
| SFT | first16 | 19,153 = 19,153 | 0 | 237 | 0.99983 / 0.0134 |
| SFT | first32 | 14,512 = 14,512 | 0 | 339 | 0.99996 / 0.0065 |
| DPO | first16 | 20,951 = 20,951 | 0 | 16 | 0.99985 / 0.0077 |
| DPO | first32 | 20,650 = 20,650 | 0 | 19 | 0.99995 / 0.0062 |

- Exact-length check: when a reply has exactly N kept tokens, the cut is the whole reply. So its first-N state must equal the stored reply mean. All four states pass pod A's C0 gate in every stage and cut. State 0 is exact (relative norm 0.0000). This is the first check of the cut values themselves, not only of the full mean.
- Also checked: every output file matches `SHA256SUMS.txt`; inputs, model, revision and code hashes match the manifest; `keys.json` covers all 21,000 rows in order; arrays are float16 [21000, 4, 4096]; `support.json` equals a recount; first16 and first32 differ on every row longer than 32 kept tokens.
- Attempt 1 (app `ap-qLraG1BF7nkuagyFTKxfi9`, `verify_attempt1.log`, `verify_attempt1/`): every array check passed. SFT and DPO failed the fingerprint check on `versions`. The only difference is the GPU card: the smoke ran on an A100 80GB PCIe and the SFT and DPO full runs on an A100-SXM4-80GB. Library versions are equal. Modal picks the card form for an "A100-80GB" request, so the card form left the fingerprint and stays in the record (`devices`). This change came after the result. The C0 check and the exact-length check against the stored states show that the card did not change the states. The array results of both attempts are identical.
- HF: revision `e6dccf4fbb217b0f14d8bb1bb9d9d5a0cfb70600`, 3 files, SHA-256 and size match. Receipt `../261112_modal_gpu/hf_receipt_261113_length_verify.json`; uploader `hf_upload.py --batch 261113_length_verify`; dataset folder `experiments/q1_emergence/261113_reply_length_control/verify/`. Attempt 2 overwrote attempt 1 on the volume; attempt 1 is kept here only.

## 5. HF copy (13/11/2026)

Revision `08ae476f7bdb08c4eedb183e36a0cfbc56db0d34`, 48 files, 4.14 GB, SHA-256 and size of all 48 match the volume. Receipt `../261112_modal_gpu/hf_receipt_261113_length.json`; uploader `hf_upload.py --batch 261113_length`. Dataset folders: `experiments/q1_emergence/261113_reply_length_control/run_full/{instruct,sft,dpo}/`, `run_smoke/{instruct,sft,dpo}/`, `logs/`. Pod A's inputs fetched to the Modal volume (`poda_dl/`) are a copy of HF @ `60a81b9b…` and were not uploaded again.

## 6. Cost (13/11/2026)

Modal $2.03, from `modal billing report --start 2026-11-13 --end 2026-11-14 --resolution h --show-resources` (read 12:13 UTC), summed by app id:

| App | Step | Cost |
|---|---|---|
| `ap-VWp9k58aThGvV68PcEziAA` | fetch | $0.004 |
| `ap-qLIrtiDBDo4Rk2vt5Ha1uj` | smoke | $0.136 |
| `ap-NqbIb6PnLkN3KXq6U4AC38` | full attempt 1 | $1.012 |
| `ap-wrkPzZPi2d4ewmqycLtywu` | full attempt 2 | $0.709 |
| `ap-qLraG1BF7nkuagyFTKxfi9` | checks | $0.008 |
| `ap-44escgpkdtdIDP1k1K1jrg` | checks | $0.006 |
| `ap-puRBzzZvqa6NuLhp1eJHfA` | HF uploads | $0.151 |
| `ap-8H1EPc97YDhEFn2PYDadSo` | HF uploads | $0.0002 |

Vault: `Reference/Cost log.md`, `Weeks/2026-W46.md` (session ad361f36), `Reference/Experiment index.md`.
