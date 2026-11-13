# Pod B `base_role_vs_format`: role against prompt format in the base model (29/10/2026)

This is the note for `261029_base_role_vs_format`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | In the OLMo-2-7B base model, do the reply states carry the role more than the prompt format? Does the role carry over to a format that the decoder did not see? Do the role maps agree across formats? |
| Result | With Holm at alpha 0.05, D2 passes, and D1 and D3 do not. A decoder that never saw a format still finds the role in that format (D2). Role does not explain more state variance than format (D1). The role maps of the formats agree with a mean r of 0.71, but the lower bound of 0.34 is under my floor of 0.5 (D3). |
| How much I trust it | I fixed the rules before the run. The full calculation is verified, and its 60 files are on HF. An agent challenge narrowed D2: a control with the prompt and no reply decodes the role at 91 to 98%. The D3 test has low power with 10 roles. Terra passed 15.4% of the replies, so most replies are not in role. No human has reviewed this. I make no external claim. |
| What I do not know | How much role information the replies hold beyond the prompt. Whether the maps agree across formats, because the D3 failure says little. The billed cost of the Modal run. |
| Status (6 Nov 2026) | The generation (29/10), the full analysis on Modal (5 Nov), the HF backup, the Holm verdict, the agent challenge, the Terra labels and the D3 power check are done. |

---

## What I did

1. The base model got each role in four prompt formats: interview, script, story and description. I froze the templates after three canary runs and two changes to the stop rule.
2. Codex reviewed the generation code in 3 rounds. I had all findings fixed.
3. The full run on 29/10 made 24,000 replies: 10 roles, 4 formats, 5 wordings, 40 questions and 3 replicates. One reply is invalid. The run went from the pod to HF.
4. The analysis code was written and reviewed on 30/10. Its null needs 1,999,800 classifier fits, so the speed was fixed three times on the pod. On 3 Nov the pod refused SSH, and I could not confirm a finished run.
5. The unchanged analysis ran again on a Modal H100 on 5 Nov. It took about 3 hours 20 minutes, and every check passed.
6. I set alpha 0.05 for Holm before anyone read the family verdict.
7. On 6 Nov Terra labelled all valid replies, and a power check for D3 ran.

---

## Results

### Only D2 passes

| Test | Number | Verdict |
|---|---|---|
| D1: role against format, share of variance | Role share minus format share 0.007, 95% interval -0.043 to 0.027, raw p 0.5865 | Not passed |
| D2: role carries over to an unseen format | Raw p 0.0001, Holm p 0.0003 | Passed |
| D3: role maps agree across formats | Mean cross-format r 0.71, lower bound 0.34, floor 0.5 | Not passed |

- For D3 the p-value is small (raw p 0.0001), but the rule also needs the lower bound to reach 0.5.
- I set alpha 0.05 before anyone read the family verdict. The raw p-values were already in `decisions.json` at that time.

### D2 is narrower than it looks, because the prompt alone carries the role

Reply-only decoding uses the C2 states: the description is removed from the prompt, and the model reads the same reply.

| Format | Reply-only decoding | Gain over the best text reader |
|---|---|---|
| Interview | 0.434 | 0.071 [0.006, 0.131] |
| Script | 0.470 | 0.076 [0.012, 0.131] |
| Story | 0.426 | -0.003 [-0.059, 0.046] |
| Description | 0.434 | 0.047 [-0.024, 0.113] |

- A control with the full prompt and no reply (C3) decodes the role at 91 to 98%. So much of D2 can come from the description in the prompt.
- Without the description, the states still decode the role (p = 0.01 in each format).
- The states beat the best text reader in interview and script. In story and description the interval includes 0.
- These numbers are report only, with 99 draws and state 16.

### Most base replies are not in role

| Format | Passed by Terra | Coherent |
|---|---|---|
| Interview | 14.1% | 97.5% |
| Script | 14.0% | 96.8% |
| Story | 18.6% | 99.7% |
| Description | 14.9% | 98.3% |
| All | 15.4% (3,689) | |

- 83% of the replies are class G: clear text, but too generic to show the role.
- The rate differs by role, from 3.8% for teenager to 34.1% for optimist.
- The rate is likely low. On 20 E3 base practice replies Terra said yes to 3, and I said yes to 8.
- Reply-only decoding is better where Terra passes a reply: 66.7% with at least one passing reply, against 32.3% with none. This split was chosen after the labels were seen.

### The D3 failure says little about the map

- I ran simulated grids of the real shape with the frozen D3 code. All formats shared one map exactly.
- At 10 roles the lower bound reaches 0.5 in only 8% of the datasets. At 30 roles it is 78%.
- D3 stays not passed.

---

## How much I trust this

- The rules are THRESHOLDS v1.7. The full counts are 2,000 bootstrap draws, 9,999 permutations, 50 splits and 50 halves.
- The gates G1 to G5 pass on the full run. I read 40 raw replies before any metric.
- The Modal result passed all six verification phases. The local copy passed 60 file hashes, 23 artifact seals and 54 dependency checks.
- The fast GPU fits were checked against CPU fits: all 200 observed prediction comparisons pass.
- The reviews of the speed changes were implementer reviews, not independent reviews.
- The judge and the decoder read the same reply, so the Terra split is not independent evidence.
- No label of mine is on a pod B reply.
- No human has reviewed the analysis.

---

## Open points

- The billed amount for the Modal run is missing.
- C2 at 9,999 draws was not run. State 17 does not exist for C2, and state 16 needs a paid GPU run.
- No text reader has had the same split by Terra pass.
- Two items from the list of 30/10 are not recorded as done. One is my OK to remove the Mac copies of the runs. The other is the stop of the second pod (`185.216.23.189`).

---

## Words used here

| Word | Meaning |
|---|---|
| Pod B | This run: the OLMo-2-7B base model writes replies for 10 roles in 4 prompt formats. |
| Base | `allenai/OLMo-2-1124-7B`, the model before chat training. It gets plain text. |
| Format | The frame around the role description: interview, script, story or description. In the last three the character is called Alex. |
| Wording, replicate | Each role has 5 wordings of its description. Each prompt is answered 3 times (replicates). |
| D1 | Rule 1: role explains a larger share of the state variance than format. |
| D2 | Rule 2: a role decoder trained on three formats works on the fourth. |
| D3 | Rule 3: the role maps of the four formats agree (lower bound of r at least 0.5). |
| Holm | A correction for several tests at once. Alpha is the error rate that I allow for the family. |
| Null | The result that chance alone gives. Here it comes from 9,999 permutations, each with a full refit of the classifiers. |
| C2 | Context control: the prompt without the description line, with the same reply. |
| C3 | Context control: the full prompt with a fixed neutral text in place of the reply. |
| Text reader | A decoder that sees the reply text only, with no model states. |
| State 16, state 17 | Hidden states of the 7B model. Whole replies use state 17. C2 and C3 exist at states 8, 16 and 24 only. |
| G1 to G5 | Gates on a generation run: tokens and grid, valid rate, seed rerun, state reconstruction, and model and template hashes. |
| G6 | Synthetic checks of the analysis code. |
| Canary | A small run (4 roles, 10 questions, 1,600 replies) that tests the code and gives the raw read. It cannot test D1 to D3. |
| Templates v1.4, v1.5, v1.6 | The prompt and stop rules. v1.5 adds a stop at the first blank line. v1.6 keeps a blank line that comes right after a colon (the colon rule). |
| R1 | The rule that rows made with a template that later changes are discarded. |
| Quote states | States pooled over the quoted speech only, in story and description replies. |
| Fingerprint | A hash of the code, templates, prompts and versions. A run resumes only if it matches. |
| Terra | GPT-6.1 Terra, my LLM judge. Pass means class E or P and coherence at least 50. Class G means generic. |
| E3 | My earlier base-model run `261008_base_persona_map`. The interview format is E3's `desc` prompt. |
| A, C | Pod A (`261029_instruct_persona_map`) and pod C (`261029_fixed_text_read_through`). They shared the rented pod. |
| Modal | The cloud service where the final analysis ran, on an H100 GPU. |
| Agent challenge | A critical review of the result by an agent (`review/challenge_261105.md`). It is not a human review. |
| HF | The private Hugging Face dataset `brend0nc/manifold-persona-experiment-artifacts` that keeps my run files. |

---

# Run record

Everything below is the detailed record, oldest step first. I only need it to check a step, a file or a number.

Status line of the old note (6 Nov 2026): the full Modal H100 calculation completed on 5 Nov at 12:00:32 SAST with exit code 0 and verified full outputs. All four 9,999-draw null files, frozen settings and 200 observed CPU/batched prediction comparisons pass checks. The 60-file numeric backup is on HF at `12d932bd6d5a38bba2f48efc676afba902a0f9ab`, checked. Holm at alpha 0.05: D2 passes; D1 and D3 do not. Agent challenge (`review/challenge_261105.md`): D2 narrowed, because a no-reply prompt control (C3) decodes the role at 91 to 98%; Terra passed 15.4% of the replies (raw rate, likely low; 6 Nov); no human review, no external claim. Design: `../261029_stage_persona_plan/DESIGN.md` (pod B). Rules: `THRESHOLDS.md` (v1.7).

## 1. Code (29/10/2026, offline, $0)

Mode: extended (shared stage-pod code). The code follows pod A (`../261029_instruct_persona_map/`) and the shared `src/manifold_persona/rows.py`. At that time the analysis (D1 to D3, G6) was not written.

| File | What it does |
|---|---|
| `gen_extract.py` | Builds the grid, renders the 4 format prompts, generates with one seed per reply, cuts each reply at its format's stop markers, and stores the whole-reply mean at all 33 hidden states. Also stores the quoted-speech states, the role-name flag and the D1 support. Stops the full run (exit code 4, `stopped_d1.json`) once more than 10 questions have a prompt with no valid replicate. Resumes after a crash. |
| `gates.py` | G1 tokens and grid, G2 ≥ 95% valid per role x format, G3 seed rerun, G4 reconstruction at states 9, 17, 23 (E3 C0 rule) plus the invalid-row check, G5 model, tokenizer and template hashes. D1 support is a report. `raw_read.md` holds 10 random replies per format. Exit code 3 if a gate fails. |
| `context_extract.py` | C2 (prompt without the description line, same reply ids) per row and C3 (full prompt plus a fixed neutral text) per prompt, in each format's own frame. Forward passes only. |
| `test_podb_*.py` | Offline tests: the real OLMo-2 base tokenizer and a tiny random Llama on the CPU. |
| `offline_checks/` | $0 re-cuts of saved canary raw ids under other stop rules (blank line; blank line after a colon), and the changed rows. Not part of a run. |

My decisions during the code (29/10/2026):

- Templates (approved as drafted). Interview = E3 `desc` exactly (a test checks all 2,000 interview prompts against the E3 code). Script, story and description use one description rule, "Alex is {desc}.", and only the frame changes. Stop markers: interview as E3; script `\nFriend:`, `\nThe following is`; story `\nThe following is`; description `\nIf Alex`, `\nThe following is`. Since v1.5 (after the canary), script, story and description also stop at `\n\n`; since v1.6, not at a `\n\n` right after a colon. Precedents: Argyle et al. 2023 and 2608.03044 (interview); Moskvoretskii et al. 2026 Dialogue and Shanahan et al. 2023 (script); Moskvoretskii Description ("If Alex …"). The story frame and the question inside the "If Alex" line have no exact precedent.
- Canary: toddler, consultant, chef, pirate; the first 10 ids of `questions_40.json`; 2 replicates; 1,600 replies. The canary keys are full-run keys, so their seeds are the same.
- Quoted speech: the whole kept reply in the interview and script; the first double-quoted passage in the story and description (an unclosed quote runs to the reply end); no quote gives NaN. A token counts when one of its bytes is inside the quote marks.
- Extra states (quoted speech, C2, C3) at 0.25, 0.5, 0.75 of model depth: hidden states 8, 16, 24 on the 7B (THRESHOLDS v1.4). Whole-reply states keep all 33.
- C3 neutral texts: E3's C3 speech in every format; the story and description put it in quotes after `said,` and `say,`.

Run outputs (`run_<grid>/`): `rows.jsonl` (shared row schema), `reply_mean_acts.npy` [rows, 33, 4096], `quote_mean_acts.npy` [rows, 3, 4096], `reply_extras.jsonl` (role-name flag, quote span and text), `grid.json`, `manifest.json`, `SHA256SUMS.txt`; then `gates.json`, `raw_read.md` and `context/` (`ctx_c2.npy`, `ctx_c3.npy`, `c3_prompts.json`, its own manifest and SHA256SUMS).

Commands on the pod (from this folder, `PYTHONPATH=<repo>/src`):

```
python gen_extract.py --grid canary --limit 40 --out run_smoke && python gates.py --run run_smoke && python context_extract.py --run run_smoke
python gen_extract.py --grid canary --out run_canary
python gates.py --run run_canary
python context_extract.py --run run_canary
```

- `../261029_stage_persona_plan/pod.sh` pushes, tests and pulls pod B (patch applied by its owner session on 29/10/2026; `pull HOST RUN b` also checks `context/SHA256SUMS.txt`).
- Storage estimate (float16): whole-reply states about 6.0 GiB, quote, C2 and C3 states about 1.3 GiB (C3 has one row per prompt, 8,000 in the full run).
- `fingerprint_parts.json` holds the code, template and tokenizer hashes. It is written when the run folder is claimed, before the first reply (THRESHOLDS, "Templates").

Offline tests (29/10/2026):

- `.venv/bin/python -m pytest experiments/q1_emergence/261029_base_role_vs_format/`: 39 pass before v1.5, 43 after, 48 after v1.6 (`test_podb_gen_extract.py` 25, then 29, `test_podb_gates.py` 9, `test_podb_context_extract.py` 5).
- They cover: interview = E3 `desc` on all 2,000 prompts; C2 drops only the description line; no frame names a role; one `<|endoftext|>` first in all 8,000 full prompts (all under 200 tokens); full grid 24,000 and canary 1,600 keys; each format's stop markers; the D1 support rule, with the Codex O-B case (99.5% valid, 0 complete questions); the stop rule (exit code 4); the same replies at batch sizes 4 and 5; resume after a cut line; quote spans (first passage, curly quotes, unclosed, none, empty, multi-byte characters before the quote); quote, C2 and C3 states equal a direct mean over the right positions; each gate fails on its kind of damage.
- With the patched `pod.sh`, `pod.sh check` passes 119 and skips 1 from the clean bundle (rows, pods A, B and C).
- The test files have unique base names (`test_podb_*`), and the scripts load their siblings by path, because pods A and B both have `gen_extract.py` and `gates.py` and `pod.sh` runs every test in one pytest session.

Tiny-model dry run (29/10/2026, CPU, $0). The real `main()` of each script, with a tiny random 32-layer Llama (width 16) and the real tokenizer in place of the 7B. The driver is outside the repo (`<scratchpad>/podb_dry/tiny_driver.py`); outputs were not kept.

- `--dry`: canary 1,600 items, grid `2ed93b34…`; full 24,000 items, grid `e47488cf…`. `prompts_check.txt` shows the 4 approved prompts.
- Canary (1,600 replies): 0 invalid; 299 s generation. Gates: G1, G2, G3 (64 of 64 identical in whole batches, 64 of 64 one at a time), G4 (min cosine ≥ 0.999999, median relative norm 0.0002), G5 all pass; quote rebuild (reported) passes at states 8, 16, 24 on 626 rows. D1 support: 10 of 10 questions complete. Context: C2 1,600 rows, C3 800 prompts, states 8, 16, 24. All SHA256SUMS lines check.
- The canary ran before the `fingerprint_parts.json` change. A 40-reply smoke run after it passed every gate and both SHA256SUMS checks.
- A resume with other settings is refused.

Not verified (this list dates from before the canary):

- The canary sections that follow cover the 7B, the GPU and real text.
- The local transformers is 4.46.3; the pod pins 4.47.1.
- At that time the analysis (D1 to D3, G6 synthetic checks, text baselines) was not written.

## 2. Engineering canary `run_canary` (29/10/2026, v1.4 templates, discarded)

Mode: extended (code shared with pods A and C; paid pod). Purpose: test tokens, seeds, reconstruction, storage and copy-back on the 7B, and give the raw read that freezes the templates (THRESHOLDS, "Templates"). It cannot test D1 to D3 (4 roles, 10 questions). No metric was computed.

- Model `allenai/OLMo-2-1124-7B` @ `7df9a82518afdecae4e8c026b27adccc8c1f0032`, bfloat16, sdpa. Roles toddler, consultant, chef, pirate; 4 formats; 5 wordings; questions 10, 18, 24, 25, 35, 37, 55, 64, 66, 72 (the first 10 of `questions_40.json`); 2 replicates; 1,600 replies. Sampling: `rows.SeededSampler`, temperature 0.5, repetition penalty 1.1, top_k 0, top_p 1, 128 new tokens, seed = `item_seed(item_key)`. Batch size 32.
- Code: not committed; sha256 `gen_extract.py` `0fe91d77…`, `gates.py` `c8a8bb8b…`, `context_extract.py` `26c667ff…`, `src/manifold_persona/rows.py` `b1e4c29b…` (the same on the pod and the Mac). Grid `2ed93b34…`, templates `57318564…`, fingerprint `c76b1740…`.
- Pod: RunPod A100-SXM4-80GB (the pod of pod A's full run, `213.173.102.5`), Python 3.12.3, torch 2.13.0+cu129, transformers 4.47.1, tokenizers 0.21.0, numpy 2.5.2 (all packages in the manifest). `pod.sh push`: 119 offline tests passed on the pod, 1 skipped.
- Commands (pod, tmux, `/root/podb_canary.sh`): a 40-reply smoke run (`--limit 40 --out run_smoke`, gates and context all passed; it stays on the pod), then `gen_extract.py --grid canary --out run_canary`, `gates.py --run run_canary`, `context_extract.py --run run_canary`.
- Time: 335 s for 1,600 replies. At this rate the full run (24,000) needs about 84 min of generation, plus the gates and context.
- Outcome: 1,600 rows, 0 invalid; states [1600, 33, 4096] float16. Stop reasons: interview marker 354, limit 45, eos 1; script marker 278, limit 117, eos 5; story limit 383, marker 10, eos 7; description eos 215, limit 130, marker 55.
- Gates (`run_canary/gates.json`): G1 pass; G2 pass (100% valid in every role x format cell); G3 pass (64 rows in 2 whole original batches, 64/64 identical; one at a time, reported only: 15/64, as in pod A); G4 pass at states 9, 17, 23 (min cosine 1.0000, median relative norm 0.0002); quote rebuild (reported) min cosine ≥ 0.9999 on 705 rows; G5 pass. D1 support: 10 of 10 questions complete, 0 lost prompts.
- Context (`run_canary/context/`): C2 1,600 rows, C3 800 prompts, hidden states 8, 16, 24; 30 s.
- Copy-back: `pod.sh pull … run_canary b`; SHA256SUMS 7/7 and `context/` 4/4 OK on the Mac. HF: not uploaded at that time (it needed my go).

Raw read (`run_canary/raw_read.md`, 40 replies, read 29/10 before any metric):

- Interview: answers to the question, mostly in a plain assistant-like voice (the pirate and chef give plain facts); the toddler reply is partly childlike. Clean ends at the next `Interviewer:` line.
- Script: short answers; some in role (toddler: "I want to go outside"). The model often adds a second `Alex:` line after a blank line, which the stop markers keep (pirate: a plain answer, then "I would like to have a pirate ship…").
- Story: 96% end at 128 tokens. In all 10, the first quote is Alex's answer; then come the friend's lines, more narration, and sometimes unrelated text ("Based on the paragraph, choose the best answer", "[Insert Diagram Here]").
- Description: Alex's answer comes first ("Alex would say, "…""). Then the model often drifts into unrelated instruction-dataset text (a Q&A with options, a footballer's bio, "what is a good title", "Answer: Option C … Click for Answer") or repeats "Alex is a pirate." many times.

Counts on all 1,600 replies (a text pattern, so approximate; not a D1 to D3 metric):

| Format | Dataset-like text | 4-word repeat ≥ 3 | Names its role | Has a quote | At 128 tokens |
|---|---|---|---|---|---|
| Interview | 2% | 14% | 3% | 100% (whole reply) | 11% |
| Script | 2% | 19% | 11% | 100% (whole reply) | 29% |
| Story | 7% | 14% | 10% | 96% | 96% |
| Description | 27% | 18% | 32% | 80% | 32% |

Offline check of one fix ($0; stops cut the generated ids after generation, so the same raw ids can be cut again): ending the 3 Alex formats at the first blank line (`"\n\n"` as an extra stop) gives description 4% dataset-like text, 5% repeats, 3% role naming, median 19 kept tokens, 8% under 5 words; story 2%, 2%, 3%, median 34 tokens; script 2%, 10%, 3%, median 36 tokens. The interview stays as E3 (median 49 tokens). The "as run" rows of this check reproduce the run's own counts.

My decision (29/10/2026, "Add blank-line stop"): the script, story and description replies also end at the first blank line (`"\n\n"` stop marker, THRESHOLDS v1.5). The interview does not change. The `run_canary` rows are discarded (THRESHOLDS R1); the folder stays on the Mac and the pod as a record, not as data.

## 3. Template change v1.5 (29/10/2026, offline, $0)

- `gen_extract.py` `1ca84972…`: `"\n\n"` added first to the stops of script, story and description. Template hash `a935962e…` (was `57318564…`), so the fingerprint changes and the old canary cannot resume into a new run.
- Tests: 4 new (a blank line cuts script, story and description; it does not cut the interview). 43 pass.
- Re-cut of the `run_canary` raw ids with the new stops (scratchpad script, not kept): 0 invalid rows of 1,600; every role x format cell 100% valid; D1 support 10 of 10. Stop reasons: script marker 367, limit 29, eos 4; story marker 390, limit 10; description marker 292, eos 100, limit 8. Median kept tokens: interview 49, script 36, story 34, description 19. Story replies with a quote 90%, description 77%. Under 5 words: script 6%, description 10%.
- A re-cut is not a new canary: the sampled ids stay the same, but the run itself (rows, gates, states) must be made with the new templates.

## 4. Engineering canary `run_canary_v2` (29/10/2026, v1.5 templates, discarded)

Mode: extended (as `run_canary`). Purpose: the canary the THRESHOLDS R1 rule asks for after the v1.5 template change. My answer to "Which templates do you freeze for the full run?" was the blank-line stop, so the templates are frozen at v1.5 (template hash `a935962e…`). No metric was computed.

- Same model, grid (`2ed93b34…`), sampling, batch size 32 and pod as `run_canary`. Code: `gen_extract.py` `1ca84972…` (the only changed file; `test_podb_gen_extract.py` `68116dfc…`), `gates.py` `c8a8bb8b…`, `context_extract.py` `26c667ff…`, `rows.py` `b1e4c29b…`. Fingerprint `002e348d…`.
- Pod: only the 2 changed files were copied with scp (the pod's old `gen_extract.py` `0fe91d77…` is in the scratchpad, `podb_v14/`); 74 pod B and `rows` tests passed on the pod. The pod C session agreed to the use of the pod. Command: `/root/podb_canary_v2.sh` in tmux (gen_extract, gates, context; no smoke run).
- Outcome: 1,600 rows, 0 invalid, 332 s. Gates (`run_canary_v2/gates.json`): G1 to G5 pass; G2 100% valid in every cell; G3 64/64 identical in whole batches (one at a time 15/64, reported only); G4 min cosine 1.0000, median relative norm 0.0002 at states 9, 17, 23; quote rebuild min cosine ≥ 0.9999 on 665 rows; D1 support 10 of 10 questions. Context: C2 1,600 rows, C3 800 prompts, states 8, 16, 24; 55 s.
- Copy-back: `pod.sh pull … run_canary_v2 b`, SHA256SUMS 7/7 and `context/` 4/4 OK; 514 MB on the Mac. HF: not uploaded at that time (it needed my go).

Counts on all 1,600 replies (the same text patterns as for `run_canary`; approximate):

| Format | Dataset-like text | 4-word repeat ≥ 3 | Names its role | Has a quote | At 128 tokens | Under 5 words | Median tokens |
|---|---|---|---|---|---|---|---|
| Interview | 2% | 14% | 3% | 100% (whole reply) | 11% | 2% | 49 |
| Script | 2% | 10% | 3% | 100% (whole reply) | 7% | 6% | 36 |
| Story | 2% | 2% | 3% | 90% | 2% | 1% | 34 |
| Description | 4% | 5% | 3% | 76% | 2% | 10% | 19 |

The interview rows are the same as in `run_canary` (same seeds and stops). The other counts equal the offline re-cut, because the seeds give the same sampled ids.

Raw read (`run_canary_v2/raw_read.md`, 40 replies, read 29/10 before any metric):

- Interview: as in `run_canary`; plain answers, mostly out of role.
- Script: one short line per reply. A few are in role (toddler: "I want to go outside"); most are plain answers. A second `Alex:` line is now rare (4%).
- Story: Alex's answer in quotes, then the cut. Some are in role (pirate: "It's like finding a lost treasure"; chef: savings "like baking a cake"). One pirate story has no quote ("Alex, being a pirate, was intrigued by this question").
- Description: mostly one quoted answer. Left over: one reply drifts into dataset text after a single newline ("This justifies what answer for what question? Q & A: …"); one repeats "Alex is a pirate." on the same line to 128 tokens; some replies carry almost no content (`" say"`, `" be most likely to say that they are"`). These count as valid, so their whole-reply states are means over 1 to 8 tokens.
- The cut at a blank line often drops the final full stop or closing quote mark (BPE merges it with `\n\n`; the shared `rows.cut_reply` rule). An unclosed quote runs to the reply end, as designed.

Offline check: a blank line after a colon (29/10/2026, $0). I asked for this check ("Check fix offline"). Rule tested: a `\n\n` right after a colon (spaces ignored) does not stop the reply; the next one does. Script `offline_checks/colon_rule_sim.py` re-cuts the `run_canary_v2` raw ids; the changed rows are in `offline_checks/colon_rule_changed.json`. The v1.5 rows of the script reproduce the run's own counts.

- 25 rows change: 24 description, 1 story; 0 in interview and script; 0 invalid.
- Description: under 5 words from 10% to 5%; median from 19 to 22 tokens; has a quote from 76% to 78%; dataset-like text from 4% to 5%; names its role from 3% to 4%; at 128 tokens from 2% to 3%.
- Read all 25: 18 are real answers (first-person prose or an advice list; some in role, e.g. pirate: "I feel like I've just found a treasure map with a big X on it"). 6 are quiz text (A to D or 1 to 5 options, two with "Answer: …"). 1 repeats one bullet.
- Not code at that time. It needed a pod B-local cut after `rows.make_row` (the shared `rows.cut_reply` matches fixed text markers only), new tests, a THRESHOLDS v1.6 line and a third canary (R1).

## 5. Template change v1.6 (29/10/2026, offline, $0)

My decision ("Adopt colon rule"): a blank line right after a colon (spaces ignored) does not end a script, story or description reply; the next blank line does (THRESHOLDS v1.6). The `run_canary_v2` rows are discarded (R1).

- `gen_extract.py` `a7239b8a…`: `"\n\n"` left the `stops` tuples; a `blank_stop` flag (true for script, story, description) and `blank_stop()`, `cut_reply()`, `make_row()` and `check_row()` do the cut in pod B only. `cut_reply` finds the earliest of the text markers and the blank-line stop in one pass and drops the token where it starts, as `rows.cut_reply` does. `check_row` also requires the kept ids and stop reason to equal `cut_reply` of the raw ids. `rows.py` is not changed. Template hash `20f5e2ac…`.
- `gates.py` `9465f20f…`: G1 uses `gen_extract.check_row`.
- Tests: `test_podb_gen_extract.py` `63c6968f…`, 5 new (a blank line after a colon, also with a space before it, in the description and the story; a colon inside a quote; a reply that runs past its cut fails `check_row`). 48 pass.
- The new code on the `run_canary_v2` raw ids gives the offline check's cut on all 1,600 rows (25 rows differ from `run_canary_v2`).

## 6. Engineering canary `run_canary_v3` (29/10/2026, second pod)

Mode: extended. I chose a second pod ("Rent second pod") and created it: RunPod A100-SXM4-80GB, `185.216.23.189`, Ubuntu 24.04, Python 3.12.3, torch 2.8.0+cu128 (the image in DESIGN.md; the first pod had torch 2.13.0), transformers 4.47.1, tokenizers 0.21.0, numpy 2.1.2. Code: round-1 fixes of the code review in section 7 (`gen_extract.py` `66b019e7…`, `gates.py` `20598698…`), templates v1.6 (`20f5e2ac…`); fingerprint `af3c1cd5…`.

- Smoke (40 replies, `run_smoke_v3`, stays on the pod): G1 to G5 pass, context done.
- Canary: 1,600 rows, 0 invalid, 238 s generation. G1 to G5 pass: G3 64/64; G4 min cosine 1.0000, median relative norm 0.0002; quote rebuild (then reported with the centred rule) on 674 rows, largest per-row relative norm 0.015; D1 support 10 of 10. Context: C2 1,600, C3 800, 55 s.
- Only 456 of 1,600 raw replies equal `run_canary_v2`: the torch version changes the sampled words. The text counts match the offline colon-rule prediction: description dataset-like text 5%, names its role 4%, has a quote 78%, under 5 words 5%, median 22 tokens.
- The final code would cut all 1,600 replies the same way.
- `sync_partial.sh` during the run: 50 of 50 state parts and 1,600 rows reached the Mac. Resume test: that copy, pushed back as `run_canary_v3_resume` and resumed, gave identical `rows.jsonl`, `reply_mean_acts.npy`, `quote_mean_acts.npy`, `reply_extras.jsonl` and `grid.json` (sha256).
- Copy-back with the patched `pod.sh pull`: 11/11 hashes and `--verify` in the staging folder, then published; 515 MB. HF: not uploaded at that time.
- Superseded by `run_canary_v4`, because the gates and `part_ok` changed after it (not a template change; the rows would be the same).

## 7. Code review (29/10/2026, Codex, 3 rounds, no GPU)

I asked: "have codex review the code ... and fix all findings". Reviewer: `codex exec` (codex-cli 0.165.0, its default model), read-only by instruction; no repo file changed during a review (hashes checked before and after each round). Prompts, logs and answers: `review/prompt_codex_code*_261029.txt`, `review/codex_code*_261029.log`, `review/codex_last_code*_261029.txt`. The code each round saw: `review/code_before_fixes_261029/` (round 1) and `review/code_before_fixes_r2_261029/` (round 2); test copies carry `.txt` so pytest skips them.

| Round | Findings | Result |
|---|---|---|
| 1 (high effort) | 14: 3 blockers (resume accepts changed prompts; two crash tails break resume; a lost pod loses all progress), 9 should-fix, 2 minor | All fixed (list under this table) |
| 2 (high effort) | 8 round-1 fixes confirmed, 6 partly; 5 new should-fix (pull retry keeps stale files; one corrupted quote passes; damaged `f32` member passes `part_ok`; identical quotes fail the centred rule; tabs after a colon) | All fixed |
| 3 (medium effort) | All round-2 fixes confirmed; 1 new: every quote state x 1.04 passes | Fixed (median rule) |

What changed (pod B files only, except two `pod.sh` patches applied by its owner session):

- Resume: the fingerprint adds a hash of every rendered prompt, the source files, the attention backend (`sdpa`, now set at load) and the torch, transformers and tokenizers versions; the run id is kept in `run_id.txt`; `read_done` drops a cut last line (also inside a UTF-8 character) and fails on a bad line before it; state parts are written to a temporary file, fsynced and renamed before the rows; a missing or damaged part (any member, shapes, dtype, `f32` flag, finite/NaN pattern) of a done batch is rebuilt from the stored ids.
- Lost pod: `sync_partial.sh` (Mac) copies `rows.partial.jsonl`, `acts_parts/` and the fingerprint files every few minutes.
- Reply cut: `blank_stop` tries every line break; a blank line may hold spaces, tabs or a CR; the colon exception ignores spaces only (my rule); a CRLF starts at its CR. On the saved `run_canary_v2` raw ids it gives the offline cut that I approved on all 1,600 rows.
- Gates: G1 checks array shapes and dtypes first; G4 gates the quote states (story and description: per row at most 0.05 relative norm difference, median at most 0.02; interview and script: equal to the reply states at the extra layers); G5 also checks the attention backend.
- Context: `check()` refuses non-finite C2 on valid rows, non-NaN C2 on invalid rows, non-finite C3 and wrong shapes before anything is saved.
- Copy-back: `check_sums` requires exactly the required files; `gen_extract.py --verify RUN` also requires `context/SHA256SUMS.txt` when `context/` exists; `pod.sh pull` stages each try in a fresh `mktemp` folder and publishes after the checks (patches `side-session-notes/.../pod_b_pod_sh_pull.patch` and `_r2.patch`).
- Memory: `sha256_file` reads 16 MiB chunks; pooling uses the decoder with `use_cache=False` (`Backbone`), which gives the same hidden states (test) without the LM head and KV cache.
- Not changed: `rows.py` (shared; its owner session keeps it, because pod C fingerprints its hash); `hf_archive.py` still hashes whole files (the Mac has 16 GiB; its owner session will raise it with me). A reply that starts with a blank line stays invalid.

Final code: `gen_extract.py` `5e2e879f…`, `gates.py` `49c3c466…`, `context_extract.py` `b336ef41…`, tests `a7c91005…`, `ef2d8270…`, `4e458628…`, `sync_partial.sh` `0ee2b0b0…`. Tests: 70 pass on the Mac (22 new for the review); 150 pass, 1 skipped on the new pod (`pod.sh push`). Tiny-model dry run of the real scripts (40 replies): gates pass, `--verify` passes, a second start resumes with nothing to redo.

## 8. Engineering canary `run_canary_v4` (29/10/2026, the full-run code)

Mode: extended. Purpose: the canary with exactly the code the full run uses. Pod, packages and grid as `run_canary_v3`. Code: `gen_extract.py` `5e2e879f…`, `gates.py` `49c3c466…`, `context_extract.py` `b336ef41…`, `rows.py` `b1e4c29b…`; templates v1.6 `20f5e2ac…`; fingerprint `6ad66746…`. `pod.sh push`: 150 tests passed, 1 skipped. Command: `/root/podb_canary_v4.sh` (gen_extract, gates, context).

- Outcome: 1,600 rows, 0 invalid, 239 s generation. Gates (`run_canary_v4/gates.json`): G1 to G5 pass. G1 shapes [1600, 33, 4096] and [1600, 3, 4096]. G3 64/64 identical in whole batches (one at a time 14/64, reported). G4 min cosine 1.0000, median relative norm 0.0002 at states 9, 17, 23; speech-format quote states equal the reply states (0 unequal); story and description quote rows (674): largest per-row relative norm 0.0071, 0.0114, 0.0149 at states 8, 16, 24 (rule 0.05), median 0.0002 (rule 0.02). G5 also matches the attention backend. D1 support 10 of 10. Context: C2 1,600, C3 800, 26 s.
- Reproducibility: every row equals `run_canary_v3` apart from its run id, and `reply_mean_acts.npy`, `quote_mean_acts.npy` and `reply_extras.jsonl` are byte-identical.
- Raw read (`run_canary_v4/raw_read.md`, 40 replies; the same replies as `run_canary_v3`): script replies are one short line, some in role (toddler: "I want to go outside"); story replies give Alex's answer in quotes (pirate: "It's like finding a lost treasure"); description replies are mostly one quoted answer. Left over, as the offline check predicted: quiz text after a single newline (chef: "This justifies what answer for what question? Q & A: …") and a numbered quiz after "say that they are:" ("Answer: Option C").
- Copy-back: patched `pod.sh pull … run_canary_v4 b` (staging, 11/11 hashes, `--verify`); 515 MB. HF: not uploaded at that time.

Full-run estimate on this pod: about 60 min generation (24,000 x 239 s / 1,600), plus gates and context, about 80 min in all. Pod disk: about 14 GB (70 GB free). Mac: the copy-back is about 8.6 GB, and `sync_partial.sh` needs about 6.6 GB more during the run; the Mac has 14 GiB free.

## 9. Full run `run_full` (29/10/2026)

Mode: extended. My go: "Start full run" (29/10/2026, without the Mac partial sync, because of Mac disk space). Pod, packages and code as `run_canary_v4` (hashes checked on the pod and the Mac before the start): `gen_extract.py` `5e2e879f…`, `gates.py` `49c3c466…`, `context_extract.py` `b336ef41…`, `rows.py` `b1e4c29b…`; templates v1.6 `20f5e2ac…`; fingerprint `0d23b615…`. Command: `/root/podb_full.sh` in tmux (`gen_extract.py --grid full --out run_full`, then `gates.py`, then `context_extract.py`), started 20:48 UTC.

- Grid: 10 roles x 4 formats x 5 wordings x 40 questions x 3 replicates = 24,000 replies; batch size 32; 3,459 s generation (57.7 min).
- Outcome: 24,000 rows, 1 invalid (`["robot","description",2,136,2]`: the reply "..." shares one token with the blank line after it, so the cut leaves nothing). States [24000, 33, 4096] float16.
- Stop reasons: interview marker 5,293, limit 695, eos 12; script marker 5,444, limit 517, eos 39; story marker 5,862, limit 130, eos 8; description marker 4,652, eos 1,048, limit 300. Quote rate: story 91.2%, description 82.4%.
- Gates (`gates.json`): G1 to G5 pass. G2 lowest cell 99.8%. G3 64/64 identical in whole batches (one at a time 13/64, reported). G4 min cosine 1.0000, median relative norm 0.0002 at states 9, 17, 23; speech-format quote states equal the reply states; story and description quote rows (10,414): largest per-row relative norm 0.0102, 0.0125, 0.0186 at states 8, 16, 24 (rule 0.05), median 0.0002 (rule 0.02). D1 support: 40 of 40 questions complete, 0 lost prompts; the stop rule did not fire.
- Context (`context/`): C2 24,000 rows, C3 8,000 prompts, states 8, 16, 24; 351 s.
- Raw read (`raw_read.md`, 40 replies, read 30/10 before any metric): some replies are in role (vampire, interview: "the only way to grow is to consume the essence of the living"; cynic, script: "I think it's a bunch of bull. I don't trust anyone"; robot, story: "I'm a robot, so I don't have opinions or biases"); many are plain answers. Interview replies sometimes run on with a new speaker line ("I: Can you tell me more…", E3's stops). One story reply is narration without a quote ("Alex, being a pirate, was intrigued by this question").
- Storage: 15 GB on the pod (with `acts_parts/` and `rows.partial.jsonl`). Copy-back to the Mac stopped on 30/10 at my word ("it should all be put on hugging face and we can analyze from there"); the run went to HF from the pod instead (see "On Hugging Face", section 10). A partial staging folder `.run_full.pulling.XmxfCn` (3.1 GB) is on the Mac until I OK its removal.

## 10. On Hugging Face (30/10/2026)

My rule, 30/10/2026: results go from the pod straight to HF and the analysis reads them there ("This is how we should work from now on, since having the results on this machine uses too much storage"). Dataset `brend0nc/manifold-persona-experiment-artifacts` (private), folder `experiments/q1_emergence/261029_base_role_vs_format/<run>/` plus `<run>.log`. Each upload was one commit by `hf_archive.py upload` (remote SHA-256 and size of every file checked), then `hf_archive.py restore` downloaded every file at that revision and checked its SHA-256. Receipts: `hf_receipts/<run>.json`. Not uploaded: `acts_parts/` and `rows.partial.jsonl` (resume files only).

| Run | Revision | Files | Restore | Uploaded from |
|---|---|---|---|---|
| `run_full` | `969f11a14a61cc1888d9532d5dd1a5c8cd8eba1f` | 21/21 (7.37 GiB) | 21/21 | second pod |
| `run_canary_v4` | `d7efd215c750bae15f4974eed3d8cff55845e5fb` | 21/21 | 21/21 | second pod |
| `run_canary_v3` (superseded) | `2d5ef4d1451cffce24a1b04f2bc4884972168347` | 21/21 | 21/21 | second pod |
| `run_canary_v2` (discarded) | `323df9f12412e8dd8f298736161a5a89ce7e854a` | 20/20 | 20/20 | Mac |
| `run_canary` (discarded) | `d2216198912edb8a0b530095045f9b8e2e4ff818` | 20/20 | 20/20 | Mac |

The pod uploads used the token that I saved on the pod (`hf auth login`; RunPod's login shell keeps it under `/workspace/.cache/huggingface/`, so the scripts set `HF_TOKEN_PATH`). The token was not read or printed. It is the broad `MARS-instance-02` token (write to all of my repos), not a dataset-only token.

## 11. Analysis code (30/10/2026, offline)

My request: "Let's get the analysis code written and reviewed, then we can check the data on HF." Codex owns the new analysis module and tests for this step. Existing generation, shared files and C remain with their owners.

| File | What it checks |
|---|---|
| `analysis.py` | Verified full inputs; G6; balanced D1 variance split; held-out-format D2 with full-refit null; joint D3 map null; matched text, quote, context and role-name controls; deterministic checkpoints |
| `test_podb_analysis.py` | Planted terms, missing prompts, unequal replicate counts, leakage, exact nulls, duplicate-role draws, input rejection, strict JSON, cache corruption and frozen settings |

- Mode: extended. An implementation worker wrote B; the parent manually reviewed its code against THRESHOLDS v1.7 and E3.
- Review fixes and verification are in `../261029_instruct_persona_map/review/analysis_review_261030.md`; exact hashes and test output are in the vault.
- No real arrays were analysed on the Mac. Historical generation hashes stay fixed when an analysis-only checksum helper changes.
- Fresh HF hash/size checks matched all 21 full-run receipt files at `969f11a14a61cc1888d9532d5dd1a5c8cd8eba1f`. No activation array was downloaded. The evidence is vault `Reference/2026-10-30 A-B HF preflight.json`.
- The frozen D2 null needs 1,999,800 classifier fits before observed fits and controls. A bounded pod timing pilot must precede the sustained run.
- The default reports adjusted p-values without a family verdict because the frozen rules omit Holm's family alpha.
- The all-valid unequal-weight fit reports full-model rank and residual sums of squares; it invents no nonorthogonal term allocation.
- No judge call, rental, real analysis job or deletion started here.

## 12. Pod analysis timing pilot (30/10/2026)

My request: "Start the analysis", with the restarted pod `213.173.102.6:40182`. Mode: extended; both frozen pipelines share verified inputs, source helpers and a paid pod.

| Item | Fixed for this pilot |
|---|---|
| Question | Does the reviewed pipeline execute on the full verified grid, and how long would the frozen analysis take? |
| Expected outcome | Input and synthetic checks pass; saved phases and timings. No scientific outcome assumed. |
| Controls | The original full grid and the frozen matched text/state0 controls; B also keeps quote/context/name controls. |
| Counts | A: 20 bootstrap, 99 permutations, 2 halves. B: 20 bootstrap, 19 permutations, 2 splits and 2 halves. Engineering only; no family verdict. |
| Budget and stop | 8 CPU threads; 30 minutes per analysis, 75 minutes for the driver. Stop on input/hash/source/gate failure, an exception or timeout. Only this launcher's child processes can be stopped by its budget guard. |
| Inputs | A revision `60a81b9b2e337a46a0a2c41862357164273da5ff`, 11 restored files; B revision `969f11a14a61cc1888d9532d5dd1a5c8cd8eba1f`, 21 restored files. All sizes and hashes checked on the pod. |
| Pre-run read | Codex read all 20 A samples, all 40 B samples and B's single invalid row. No quality filter added. |
| Code check | The deployed analysis suite passed 101 tests in 13.32 seconds before real metrics. Exact deployed bytes are in the pod's `source_manifest.json`. |
| Hardware and runtime | A100-SXM4-80GB, 64 visible CPUs; torch 2.13.0+cu129, transformers 4.47.1, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.1, sentence-transformers 5.7.0. Full versions are saved on the pod. |

- Pod workspace: `/root/pod-ab-261030-01a188ac/`. The launcher is A's `run_analysis_pilot.py`; it runs both scripts with explicit engineering flags. The outer tmux session is `ab-pilot`. The job specification, command logs and stage status remain separate from the input artifacts.
- Both analyses use the pinned E3 sentence reader. Its normalized float32 embeddings are prepared on the GPU, with TF32 disabled, then checked through each script's cache schema. This is recorded separately from the generation provenance. The statistical fits run on the CPU.
- Output folders: each experiment's `analysis_pilot_261030_01a188ac/` inside the deployed repo. No judge call or reply generation runs. The pilot must finish before a sustained-run budget is chosen; B's frozen null still requires 1,999,800 fits.

Pilot outcome:

- Both engineering timing pilots exited 0 (A 182.924 s; B 328.296 s), after 101 deployed tests, the full input checks and the raw read. These reduced counts give no final scientific verdict. A result/phase checksums and B integrity sidecars were verified after completion.
- Attempt 1 stopped during B input validation before any metric: the generation finalizer adds CUDA and NumPy to the manifest's version map. Its original fingerprint hash matched. The validator now compares each historical version field strictly while allowing those extra finalization fields. The fix passed 50 B tests locally and on the pod; original inputs are unchanged. Attempt records are kept under the pod workspace's `attempts/attempt1/`.
- B's full null needs 1,999,800 fits. Three representative timings (0.507, 0.539, 0.822 s per fit) give a provisional serial forecast near 299 hours; no parallel speedup was measured. Improve execution before the full B run. A's full resampling and paid naming remain separate.
- Numeric outputs were checked and saved locally before upload approval. Automatic approval review initially rejected the broader backup. I then replied "Approve upload" to the exact 79-file numeric-only proposal (1,822,199 bytes), excluding raw replies, source code, sentence embeddings and credentials.
- Upload to private dataset `brend0nc/manifold-persona-experiment-artifacts` is verified at `f6fc89b82409b0f8239a3366c0a43efd9f265326` (47 files for this experiment, 79 total); a fresh full restore matched all 79. Completed 30 Oct 2026 14:40:50 SAST. Receipt: vault `Reference/2026-10-30 A-B pilot results/approved_numeric_upload_receipt.json`. The original proposal is preserved; the earlier block is resolved.

## 13. Next steps as the old note listed them (30/10/2026)

1. My OK to remove the Mac copies, now that every run is on HF: `run_canary`, `run_canary_v2`, `run_canary_v3`, `run_canary_v4`, `run_canary_v3.partial`, `.run_full.pulling.XmxfCn` (about 5.6 GB).
2. Mine to do: stop or terminate the second pod (`185.216.23.189`) in the RunPod console; it bills while it runs, and its `/workspace` holds the broad HF token.
3. Improve full-null execution using the completed timing pilot, then agree the sustained analysis budget. The numeric pilot outputs are backed up and verified; full-resampling metrics remain.

## 14. Full statistical analysis and speed fix (30/10/2026)

My request: "Can you fix B's speed issue and get its analysis running?" Mode: extended. The frozen null still requires 1,999,800 classifier fits; no fold, draw, model or pass rule was removed. The implementation worker changed only `analysis.py` and its tests; the parent manually reviewed the change.

| Item | Verified execution record |
|---|---|
| Fix | Optional process workers share read-only transforms. Each scaler uses training X only. Every classifier refits; draw k retains seed+1+k and ordered checksummed checkpoints. Serial defaults remain. |
| Tests | All 76 analysis tests passed on the actual pod in 22.00 seconds, including process equivalence, child thread limits, convergence errors, interruption/resume and corruption refusal. |
| Real-data check | The deployed process backend matched the preserved serial observed score, every one of 32 null statistics and its p-value on three frozen folds of held-out format zero. Its checkpoint integrity passed. This was engineering timing only. |
| Measured speed | Warm 16-process/one-thread timing: 192 fits in 21.05 seconds. The small production-backend check projects about 62 hours for the full null alone. These are rough forecasts from one format and three folds; remaining folds and controls may differ. |
| Pod limits | Affinity shows 64 CPUs, but the measured cgroup quota is 15.25 CPU equivalents; memory limit is about 125 GB. Extra BLAS threads performed worse. |
| Full settings | 2,000 bootstrap draws, 9,999 permutations, 50 splits and 50 halves. Flags: `--null-backend processes --null-workers 16 --null-blas-threads 1 --cache-null-scalers`. |
| Start and cap | Started 30 Oct 15:45:52 SAST, PID 3023, tmux `B-full-analysis`. Initial cap: 7,200 seconds, ending about 17:45 SAST. The request to extend the total budget to 72 hours is unanswered; a request for clarification was not approval. |
| Source | `analysis.py` sha256 `1afedb0a3d314224743c90273756ef2e6d0bed93c28d0e30ce5ab92d1734434a`; tests sha256 `48151fa0abf390c37a41dce7eac2f65194b7a88eda1d3ce1f0d93952f7cf5bdb` |

- The launcher stops its own job on an input/hash/gate failure, exception or time cap. It makes no paid judge call or automatic HF upload. The first benchmark stopped before metrics because its cache path was absent; the first process prototype stopped at a package import. Both attempts are retained; the corrected checks passed.
- Workspace: `/root/pod-ab-261030-01a188ac/` on `213.173.102.6:40182`. Exact command, budget and progress: `full_B_spec.json`, `full_B_budget.json`, `full_B_status.json`, `full_B_analysis.log`. Output: `repo/experiments/q1_emergence/261029_base_role_vs_format/analysis_full_261030_01a188ac/` under that workspace. The checked sentence-embedding cache is reused; timing-pilot statistics are not reused.
- Evidence: `B_final_backend_smoke_261030/verification.json`, `B_warm_process_probe_261030/timings.json` and `B_process_deployment.json` in that workspace. The vault keeps their small receipts and exact source snapshots. A's separate statistical run exited 0 at 15:25 SAST; C remains with its owner.

## 15. Equivalent faster execution (30/10/2026)

My request: "Can we make B’s analysis faster without changing what we’re testing?" Mode: extended. The same inputs, 2,000 bootstrap draws, 9,999 permutations, 50 splits and 50 halves remain. Each of the 1,999,800 null classifiers still refits with the same estimator, labels, seeds, stopping tolerance and score.

| Change | Verification and limit |
|---|---|
| D1 sufficient statistics | The caller supplies the original role/question draws. Small Gram matrices replace repeated copies of the full feature grid. All 2,000 real-data draws matched within 5.52e-16; every sign, p-value and decision matched. The complete optimized bootstrap took 0.795 seconds. Numerical guards retain the original calculation near unstable cases. |
| D2 GPU execution | Optional float64 array-API execution of the same sklearn logistic regression, with deterministic CUDA and TF32 off. Training-only transforms stay fixed and labels stay on the CPU. All observed-fold predictions must match fresh CPU fits before a format's null starts. |
| Engineering checks | All 120 B tests passed on the pod in 115.62 seconds. The integrated eight-process GPU backend matched the serial observed score, every one of 32 null statistics and the p-value over three frozen folds of format zero. Its saved checkpoint integrity passed. This is a bounded equivalence check. |
| Warm timing | Eight GPU processes completed 96 fits in 5.127 seconds, about twice the warm CPU throughput. This projects 29.7 hours for the full null alone; startup and other phases add time. It is a short benchmark, not a completion promise. The 73.48-second integration smoke includes setup and must not be extrapolated as steady throughput. |
| Shared resources | Same A100 80 GB pod; one inner thread and an 8% GPU allocator cap per B process. C keeps its separate owner, process group and workspace. No extra pod was rented. |
| Switch and cap | The old B group exited at 16:54:30 SAST; the separate launcher then started the sole new B job at 16:55:07. It preserves the original 17:45:52 deadline. A request for 36 total hours is pending; neither the earlier 72-hour proposal nor this request is approval. |

- New source: `analysis.py` SHA256 `4490ff0eb1050553f279da227cdb01ecdcd51e379c25a6ca03bb05968e166bde`; `analysis_variance_fast.py` `601234187fcc1fe3994434755f66f1d33d5b9fa0b359b38021fc303690d20627`; `analysis_gpu.py` `d0c8de5469e87f899c1d0c681f29e58c827715365af5ad425c3e4add8136441b`. The parent reviewed these changes; this is an implementer review.
- New workspace: `/root/pod-ab-261030-01a188ac/repo_fast_v2/`. Output: `experiments/q1_emergence/261029_base_role_vs_format/analysis_full_fast_v2_261030_01a188ac/` under that workspace. Driver, logs, exact command and budget are in the parent workspace: `full_B_fast_v2_driver.py`, `full_B_fast_v2_status.json`, `full_B_fast_v2_spec.json`, `full_B_fast_v2.log`; tmux `B-full-analysis-fast-v2`, PID 5462. At that time G6, restored-input integrity and D1 were saved, and D2 was calculating. No full scientific verdict existed.
- Proofs: `B_variance_equivalence.json`, `B_fair_gpu_scale_261030/timings.json`, `B_fast_v2_smoke_261030/verification.json`, `B_fast_v2_deployment.json`, `B_verified_old_stop.json`. Small copies and exact source snapshots are in vault `Reference/2026-10-30 A-B pilot results/B_fast_v2/`. Original outputs and CPU partial checkpoints remain separate and unchanged; no results from a different execution fingerprint are reused. No new HF upload, judge call or deletion occurs automatically.
- Automatic approval review rejected the initial combined switch because it could overlap the old B job. A separate verified shutdown resolved that block; the new launcher refuses to start while the old group exists. C was not stopped or changed.

## 16. True GPU batches (30/10/2026)

My request: "That is not fast enough, can't it be done in batches and actually try and resolve this situation". Mode: de-risk engineering inside the existing extended experiment. The same role/question draws, all 9,999 permutations, 50 frozen splits per format and all 1,999,800 null classifier fits remain. No reply, model, statistical threshold or null draw was removed.

| Attempt | Measured outcome | Decision |
|---|---|---|
| Batch callbacks under CPU optimizers | All 96 paired prediction sets and iteration counts matched. Around 5 fits/s at 32 to 64 fits per group, below the previous GPU throughput. | Do not deploy. The prototype and raw timings remain. |
| Complete independent GPU L-BFGS states | Same ridge-softmax objective, C=1, balanced weights, zero initialization, history size 10, gradient tolerance 1e-4, function tolerance 64 float64 eps, max_iter=2,000, maxls=50 and independent More-Thuente searches. All 96 paired prediction sets and iteration counts matched; all 32 null statistics matched. | Selected execution method. Coefficients, gradients and optimizer histories stay on the GPU. |
| Compact inverse-history matrices | Predictions and iteration counts matched, but throughput was lower. | Retain as an engineering attempt; use the two-loop history in the full run. |
| Width comparison | 256 fits: 1.377 s, 185.85 fits/s. 512 fits: 2.717 s, 188.46 fits/s. | Choose 256: nearly the same speed with less memory and a shorter in-flight batch. |

- The warm 256-fit timing projects about 2.99 hours for the full null alone, roughly ten times faster than the prior process/GPU version. It covers one fixed fold of one format; other folds, startup and controls add time. It is not a completion promise. Numerical differences are checked through the original CPU predictions; an unstable or unconverged fit stops instead of dropping a case.
- All 201 production tests passed on the pod in 24.93 seconds. The exact integrated batch path also matched all 32 saved real null statistics, verified observed predictions and first/last null predictions against CPU, and passed checkpoint integrity. Tests cover independent optimizers, the weighted objective, curvature-history algebra, convergence errors, source/resource changes, corruption and interrupted-prefix recovery. The parent manually reviewed the new method; this is an implementer review.
- The isolated candidate is `/root/pod-ab-261030-01a188ac/repo_batch_v3/`. `analysis.py` SHA256 is `24efb62beeb8494468c80d4fcacf45fd883866fedece6e0f992567b80769240a`; the device optimizer is `analysis_batch_lbfgs.py` at `ed6d5b632a0d8326fce86c8dce44361b74888cc72e71740114b9cbcf434267f0`. Complete source hashes, code snapshots, failed/slower attempts and numeric proofs are in vault `Reference/2026-10-30 A-B pilot results/B_true_batch/`.
- Full command: the existing frozen CLI, with `--null-backend batches --null-workers 256 --null-blas-threads 1 --cache-null-scalers --null-device cuda`; workers denotes the independent fit batch width here. Each format verifies all observed-fold CPU predictions before its null. At entry or resume, its first and last fresh null fits are CPU-checked on every fold. Completed draw prefixes remain ordered and checksummed; source and batch settings enter their fingerprint. Old outputs are preserved and not mixed with this new fingerprint.
- The launcher `/root/pod-ab-261030-01a188ac/full_B_batch_v3_driver.py` is prepared, with a maximum four-hour cap. It requires a new explicit budget record from me before starting and refuses overlapping old B jobs. The prior 72-hour and 36-hour proposals are not approved. No new full job, HF upload, judge call, rental or deletion was made by this step; C stays with its owner.
- Three timing receipts initially hashed the callback-prototype filename rather than the loaded device optimizer. Their source fields were corrected from the checked deployment and preserved source snapshot; the original fields and correction reason remain in each receipt. The production smoke's source hash was correct.

## 17. Approved full batch launch (30/10/2026, 18:07 SAST)

My decision: "Run with a four-hour cap", in reply to the explicit true-batch run question. Mode: extended; this executes the verified candidate on the full saved grid. The same inputs, controls and all resampling counts remain. No outcome is assumed.

| Item | Execution record |
|---|---|
| Start and deadline | 30 Oct 18:07:08 SAST to 22:07:08 SAST; maximum 14,400 seconds from this launch |
| Host and job | Existing `213.173.102.6:40182`; tmux `B-full-analysis-batch-v3`; analysis PID and process group 7172 |
| Full counts | 2,000 bootstrap draws, 9,999 permutations, 50 splits and 50 halves; all 1,999,800 null classifier fits |
| Backend | 256 independent float64 fits per batch; original model objective and stopping settings; one inner thread; 20% GPU allocator cap |
| Startup evidence | The pod was idle, with 45.06 GiB free. Exact source and helper hashes, 201 passing tests and paired real-data proofs passed the launch guards. At 18:09 SAST, G6 passed and the input-integrity report was saved. |
| Stop conditions | Stop the launcher's own process group on an exception or the four-hour cap. Prediction mismatch, nonconvergence, corrupt inputs or changed source/resource fingerprints stop the calculation. |

- Workspace `/root/pod-ab-261030-01a188ac/`; code `repo_batch_v3/`; output `repo_batch_v3/experiments/q1_emergence/261029_base_role_vs_format/analysis_full_batch_v3_261030_01a188ac/`. Approval, exact command and progress are in `B_batch_v3_budget.json`, `full_B_batch_v3_spec.json`, `full_B_batch_v3_status.json` and `full_B_batch_v3.log`. The verified launcher is `full_B_batch_v3_driver.py`.
- This approval resolves the four-hour launch request. The earlier 72-hour and 36-hour proposals stay superseded. No new rental, paid judge call, automatic HF upload or deletion is included. Vault launch receipt: `Reference/2026-10-30 A-B pilot results/B_true_batch/approved_launch_1807.json`.
- At 18:21 SAST, the first format has 512 complete permutation draws saved, representing 25,600 full null fits. The checkpoint's SHA-256 and size match its sidecar. All 50 observed-fold CPU/GPU prediction hashes match; the runtime first/last null-fit checks also passed before the first batch was committed. GPU use was 94%; this establishes active collection of null results, not a final scientific verdict. Live proof: vault `Reference/2026-10-30 A-B pilot results/B_true_batch/live_1821.json`.

## 18. Status check (3 Nov 2026)

- A read-only check found `213.173.102.6:40182` refusing SSH.
- The private HF dataset HEAD is still `b343c96522365de638d7971d0a6969c294b4d41f`; B has generation and canary directories plus the numeric pilot, with no registered full-analysis directory.
- The agent's thread last saw B advancing on 30 Oct at 19:17 SAST, with format 0 at 9,999 null draws and format 1 at 1,280.
- No complete statistical output or clean exit is confirmed. Do not report the job as running or complete, infer pod billing state, or launch a replacement from this check. Recover the current pod address and final checkpoint first.
- Proof: vault `Reference/2026-10-30 A-B pilot results/status_261103.json`.

## 19. Modal recovery (4 Nov 2026)

Mode: extended, with a bounded engineering check before the full rerun. I offered $50 in Modal credits, accepted the setup, and completed Modal login and private HF access. The execution proposal keeps a four-hour cap. This recovers analysis of the archived generation; no new generation or paid judge is planned.

| Item | Record |
|---|---|
| Question and expected result | Can the unchanged B analysis run on Modal with the same CPU/GPU predictions and useful throughput? No scientific outcome is assumed. |
| Inputs and source | All 21 input-receipt files at HF `969f11a14a61cc1888d9532d5dd1a5c8cd8eba1f`. Restore exact checked initial, GPU and final-batch source archives. Analysis SHA-256 remains `24efb62beeb8494468c80d4fcacf45fd883866fedece6e0f992567b80769240a`. |
| Frozen measurement | THRESHOLDS v1.7; 2,000 bootstrap draws, 9,999 permutations, 50 splits and 50 halves. All 1,999,800 null classifier fits remain. Holm family alpha remains unspecified. |
| Hardware and budget | One A100 80 GB, eight physical CPU cores and 128 GiB RAM. Maximum 14,400 seconds from the first GPU check, including retries and time between phases. Listed-rate estimate is about $16 for four hours, plus bounded CPU setup; actual billing is missing. |
| Small check | Rerun the original 201 tests plus 15 wrapper tests. Compare 96 CPU/GPU prediction sets and iteration counts, then time full-width batches on three original folds. Stop before the full run if the check fails or its forecast lacks headroom. |
| Durable records | Modal Volume `persona-podb-analysis-261104`, run root `/data/podb-261104-01a188ac`. Commit progress and immutable, checksummed null prefixes during execution. The saved deadline cannot be reset by a retry. |
| Stop conditions | Bad file hash, changed source, failed prediction or iteration check, failed tests, nonconvergence, duplicate output, expired budget, or failed process stops the work. No automatic full-results HF upload, judge call or deletion. |

- New files: `modal_analysis.py`, `modal_runtime.py`, `modal_probe.py`, `build_modal_payload.py` and `test_modal_runtime.py`. The original statistical source is unchanged. The wrappers passed 15 real filesystem/process tests, including forced timeout, failed child exit, fixed deadline and torn-checkpoint handling. At that time the Modal remote tests and real-data timing were still pending.
- The first cloud attempt stopped because the HF secret had not been saved in the active environment. I completed that setup; the secret name and environment are now confirmed without inspecting its value. Runtime construction then started at Modal app run `ap-CytBz1phYysna3BG7xeLte`.
- Files: vault `Reference/2026-11-04 B Modal recovery/` has the source bundle, complete source manifest, new wrapper sources and preparation status. The older run history (sections 1 to 18) stays unchanged.

## 20. Modal input restore and account gate (5 Nov 2026)

The CPU-only restore completed successfully. All 21 archived files match their expected full SHA-256 and size. An independent Volume download confirms the saved preparation receipt. The forty archived raw cases were read again, without filtering or changing any reply.

| Item | Evidence at that time |
|---|---|
| Inputs | Frozen HF revision and all 21 receipt files verified. No activation array was copied to the Mac. |
| Runtime | Exact core library versions and checked statistical source are built in image `im-90VV203QI6iofDATX9gYsk`. Python patch version and all actual runtime versions will be recorded by the analysis. |
| Wrapper checks | Fifteen local tests pass. Remote 216-test verification and the 96 paired real-data predictions/iterations have not run. |
| GPU access | Provider error: "Please add a payment method to use A100-80GB GPU functions." I replied "Done" to that setup request, but a fresh retry still rejects the active `bchikavanga25` workspace. A clarification is pending; no further automatic GPU retry is planned. |
| Budget | No GPU budget file, pilot status or full status exists in the Volume. The four-hour clock starts only when the GPU function begins. |
| Storage | Volume `persona-podb-analysis-261104`; root `/data/podb-261104-01a188ac`. CPU staging app `ap-uJnQQ8UiHpclRnPCANVemd`; call `fc-01M6MQ45JMR9WFW19GGHJVWCM5`. |
| Billing evidence | Workspace monthly summary: metered $0.02190689, billed $0 after credits and allowances. This is a workspace total, not a B-only cost receipt. |

- Several image-save workers were interrupted. A persistent image build then succeeded; the cause of the interruptions is unconfirmed. Cloud validation also required the two-second idle shutdown setting and the provider's default disk allocation. These changes affect the wrapper only. The original analysis source is unchanged.
- CPU staging is `modal_prepare.py`. Its compatibility adapter uses `get_raw_f()`, which is supported by the pinned Modal 1.5.5 and deprecated for removal in 1.6; `requirements-modal.txt` pins the local SDK. Progress reads use bounded concurrent reads through the same file-download API as the CLI. The first streaming reader stalled and was stopped; its replacement successfully read the saved receipt and absence of GPU files. No compute job was stopped.
- Resume after the account gate is cleared: run `modal run --detach modal_analysis.py --phase preflight` from this folder. Inspect the returned verification, original-fold timings and remaining budget before `--phase full`. Do not reduce test counts or renew the saved budget on a retry.
- Files: vault `Reference/2026-11-04 B Modal recovery/` contains the source bundle, wrappers, input-restore proof, independent progress read, image ID and workspace billing summary. No new generation, paid judge call or HF results upload occurred.

## 21. Modal execution history (5 Nov 2026, 08:40 SAST)

The GPU account gate is resolved. The A100 passed all 216 tests and all 96 original-fold CPU/GPU predictions and iteration counts, but its slowest measured batch projected 4.10 hours for the null alone. Codex selected a short H100 comparison within the same four-hour deadline. The H100 passed the same checks and met the timing headroom rule.

| Item | Verified record |
|---|---|
| GPU and resources | NVIDIA H100 80GB HBM3; eight physical CPU cores and 128 GiB RAM. Torch 2.13.0+cu129, CUDA 12.9 and float64 null fits. |
| Actual H100 timings | Three full-width 256-fit batches took 1.4393, 1.3024 and 0.9054 seconds. The slowest projects 3.12 hours for all 1,999,800 null fits; other phases add time. This is an engineering projection, not a completion guarantee. |
| Full counts and controls | 2,000 bootstrap draws, 9,999 permutations, 50 splits and 50 halves; original 256-fit independent batches. Frozen inputs, unfiltered valid replies, all controls and Holm-alpha omission remain. |
| Clock | First GPU check began 08:22:12 SAST. Corrected full process began 08:40:38 SAST. Shared deadline is 12:22:12 SAST; the driver reserves two minutes before it for verification. No retry resets the clock. |
| Durable output | Modal Volume `persona-podb-analysis-261104`, `/data/podb-261104-01a188ac/analysis_full_batch_v3_cuda_261105`. Progress and verified null prefixes are committed during the run. |
| Active run | App `ap-7dggfA1PpwuAZk5txe4lEV`, call `fc-01M6N701WBFHN2B6E9D4AR6TGQ`, image `im-yvHTeVihDO6G2IqCh82vhN`. Full command and budget are in `full_spec.json`; live phase is in `full_status.json`. |
| Cost plan | H100 costs more per hour than A100. At listed default rates, H100 plus these CPU/RAM requests is about $5.35/hour. The remaining fixed window and small pilots imply about $22 maximum compute at those rates, within my stated $50 credits. This is an estimate; the billed amount remains missing. |
| Watch | `watch-b-modal-analysis` is confirmed ACTIVE every five minutes, with alerts only when my action is needed or the run needs handoff. The previous watch had been deleted or expired; it could not be updated. |

- The first full setup call, `fc-01M6N6RJWCA6D7WP0SK2JKR90W`, omitted the original pilot's `--embedding-device cuda`. Codex caught this during initialization and cancelled that call at 08:38:51 SAST, with terminal cancellation verified. Its partial output and call metadata are preserved. The corrected command restores the checked pilot's GPU embedding setting and uses a separate output folder; no completed scientific result was discarded.
- The H100 pilot's saved folder and `requested_GPU` label say A100 because the local selector was not forwarded into the container environment. The independent CUDA hardware field says `NVIDIA H100 80GB HBM3`, and the local dispatch requests H100. These historical receipts are unchanged. The full launch now forwards the selector and verifies that its actual device matches the successful timing proof. The pilot's legacy `runtime_info` method descriptor is not the solver specification; actual fits use the unchanged independent batched L-BFGS helper.
- Original statistical source SHA-256 remains `24efb62beeb8494468c80d4fcacf45fd883866fedece6e0f992567b80769240a`. The full saved command explicitly has `--embedding-device cuda`. Core package versions match the checked run; installation resolved tokenizers 0.21.4 and HF Hub 0.36.2. This rerun reads stored token IDs and arrays, and it records the pinned tokenizer files. No generation or paid judge call occurs. Full-result HF upload is not automatic.
- Sources: vault `Reference/2026-11-04 B Modal recovery/` has the A100/H100 receipts, corrected source bundle, cancellation record, full launch command and live status. Price reference: https://modal.com/pricing (checked 5 Nov).
- Live read at 08:47 SAST: current G6 passes, and input integrity, D1 and all-valid fit files are saved. The first format has 256 complete null draws (12,800 classifier fits), verified by the wrapper against the checksum sidecar before committing its backup. At that time the full analysis was in progress; no final verdict or HF result receipt existed. Proof: `latest_full_live.json` in the vault bundle.

## 22. Verified full calculation and backup handoff (5 Nov 2026)

- The remote full verification passed all six expected phases. The independent local copy passed 60 file hashes, 23 artifact seals and 54 dependency checks.
- Input coverage is 24,000 rows with one invalid reply, 8,000 prompt cells and all 40 questions. Every format retains 50 decoding splits, zero untested cells and eight expected baselines.
- The calculation took 11,994.54 seconds, about 3 hours 20 minutes. It finished before the original 12:22:12 SAST budget deadline. The Modal call returned completion, and the container list is empty. Experiment-specific billing remains missing.
- Quote-supported variance reports `not_estimable`. Each wording exclusion retains four wordings, so its original five-wording variance grid is unavailable by design. All requested sensitivity outputs, including C2/C3, are saved. At that time the family verdict was `pending_owner_alpha`.
- The ready numeric backup has 60 files and 8,154,637 bytes. Its destination is the existing private HF dataset under `experiments/q1_emergence/261029_base_role_vs_format/analysis_full_batch_v3_cuda_261105/`. At that time my approval of this concrete payload was pending. Exact file hashes and reading steps are in vault `Reference/2026-11-04 B Modal recovery/HANDOFF.md`, `full_results/HF_backup_proposal.json` and `full_results/local_verification.json`.

## 23. HF backup and Holm verdict (5 Nov 2026, 20:55 SAST)

My words, 5 Nov 2026: "ok, 1. 0.05, b upload to HF, so we don't lose stuff on modal". Alpha 0.05 was set before anyone read the family verdict; the raw D1 to D3 p-values were already in `decisions.json`.

| Item | Record |
|---|---|
| HF upload | The 60 files of `HF_backup_proposal.json` (8,154,637 bytes), uploaded from the verified local copy to `experiments/q1_emergence/261029_base_role_vs_format/analysis_full_batch_v3_cuda_261105/`. Revision `12d932bd6d5a38bba2f48efc676afba902a0f9ab`. The first commit attempt failed with HTTP 502 and created no commit; the retry succeeded. |
| Check | A fresh download at that revision has 60 files, none missing or extra, and every SHA-256 matches the manifest. |
| D1, role vs format variance share | Difference (role minus format share) 0.007, 95% interval -0.043 to 0.027. raw p 0.5865, Holm p 0.5865, rule not met: not passed. |
| D2, role carries over to an unseen format | raw p 0.0001, Holm p 0.0003, rule met: passed. |
| D3, role maps agree across formats | raw p 0.0001, Holm p 0.0003, rule not met: mean cross-format r 0.71, lower bound 0.34, below the 0.5 floor. Not passed. |

- Reply-only D2 (C2 states, state 16, frozen folds, 99 draws, report only; my go: "Yes, run it"): 0.434, 0.470, 0.426, 0.434 for interview, script, story, description; p = 0.01 in each (the lowest 99 draws allow); gain over the best text reader 0.071 [0.006, 0.131], 0.076 [0.012, 0.131], -0.003 [-0.059, 0.046], 0.047 [-0.024, 0.113]. Files in vault `Reference/2026-11-05 B C2 decoding/`.
- `decision_report()` from `analysis.py` gave these decisions, `family_status` `evaluated`. Files in vault `Reference/2026-11-04 B Modal recovery/full_results/`: `decisions_holm_alpha_0.05.json`, `HF_upload_receipt.json`. The original `decisions.json` is unchanged. At that time there was no human review and no `/challenge-result`; this is not an external claim.

## 24. Terra labels on the replies (6 Nov 2026, report only)

My go: "Go ahead and run it". Terra judged all 23,999 valid replies. Pass = E or P and coherence at least 50. These are raw judge rates. No label of mine is on a pod B reply.

| Format | Passed | Coherent |
|---|---|---|
| Interview | 14.1% | 97.5% |
| Script | 14.0% | 96.8% |
| Story | 18.6% | 99.7% |
| Description | 14.9% | 98.3% |
| All | 15.4% (3,689) | |

- Role spread: teenager 3.8% to optimist 34.1%. G (generic) is 83% of replies.
- The rate is likely low: on 20 E3 base practice replies Terra said yes to 3 and I said yes to 8.
- Reply-only decoding (C2, state 16, frozen D2 folds) by Terra pass, inside each role and format: 66.7% with at least one passing reply, 32.3% with none, difference 0.345 [0.317, 0.371] (questions resampled, 2,000 draws). By passing replies, 0 to 3 of 3: 31.3, 62.8, 81.5, 87.2%. Chosen after the labels were seen. The judge and the decoder read the same reply, so they are not independent. No text reader has had the same split.
- `rows.py` hash check closed: the only change between `b1e4c29b…` and `ba1e1e03…` is how `file_sha256` reads a file (in chunks). Row identity is not touched.
- Files: vault `Reference/2026-11-06 B Terra labels/`. Labels: HF `09f2898eddbd54a1b286c83945f361992ffebbd0`. Challenge: `review/challenge_261105.md`, second addendum.

## 25. D3 power check (6 Nov 2026, report only)

My answer: "1". On simulated grids of the real shape, with the frozen D3 code, the lower bound reaches 0.5 in 8% of datasets at 10 roles when all formats share one map exactly (0% to 8% across ranks 2, 4 and 8). At 30 roles it is 78%. So the D3 failure says little about the map. D3 stays not passed. Details and limits: `review/challenge_261105.md`; vault `Reference/2026-11-06 B D3 power/`. C2 at 9,999 draws was not run: state 17 does not exist for C2, and state 16 needs a paid GPU run.
