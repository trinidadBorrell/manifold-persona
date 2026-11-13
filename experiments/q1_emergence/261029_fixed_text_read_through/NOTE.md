# Pod C `fixed_text_read_through`: ten checkpoints read the same text (29/10/2026)

This is the note for `261029_fixed_text_read_through`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | When ten OLMo-2-7B checkpoints read the same replies, do they hold the same role map? The checkpoints are base, SFT, DPO, Instruct and six RL steps. |
| Result | Base and Instruct agree on the 20-role map when they read the same Instruct-written replies (r = 0.896 [0.767, 0.932]). So my rule D1 passes on pod A text. Maps from the reply text alone agree with the Instruct map almost as well. I found no measurable change in the map between the ten checkpoints. |
| How much I trust it | I fixed the rules before the run. All gates passed. All 10 checkpoints and the analysis are on HF and checked. An agent challenge found that every reported number matches HF, and it narrowed the reading. No human has reviewed it. I make no external claim. |
| What I do not know | Whether the states hold role information beyond the text. Whether a change in a few roles hides under "no measurable change". Whether base would write in role, because this run is about reading fixed text. What a reader from another model family does on the same text. |
| Status (30/10/2026) | The full run, the analysis, the agent challenge and the re-run after it are done. |

---

## What I did

1. I built the input once: pod A's 21,000 Instruct replies and 3,600 E3 base replies, in five frames (F0 to F4). I read 10 rows per frame before any metric.
2. An engineering canary ran base and Instruct only. It tested the extraction, the gates, the storage and the analysis code.
3. Codex and Claude reviewed the code before the full run. I had the findings fixed ("ok fix all").
4. The full run extracted all ten checkpoints on one A100 pod on 30/10. Each checkpoint went to HF and was checked before the next one loaded.
5. The analysis ran on the pod.
6. An agent challenged the result. The analysis then ran again with the additions from the challenge. Every earlier point value is unchanged.

---

## Results

These numbers are for pod A text. Unless I say otherwise, they use the main frame F1 and hidden state 17. Only D1 is a decision of my primary rule. The rest is exploratory.

### Base and Instruct hold the same role map on the same text

| Comparison of base and Instruct | Map agreement r |
|---|---|
| Main frame F1, state 17 | 0.896 [0.767, 0.932] |
| Reply only (F2) | 0.889 |
| State 9 | 0.888 |
| State 23 | 0.889 |
| First 32 tokens (F4) | 0.871 |

- The one-map null gives p = 0.0001. Both reliability lower bounds are 0.77. So D1 passes.
- Four components hold 70% of the role-mean variance at every checkpoint.

### The reply text alone gives almost the same map

| Text reader | Agreement with the Instruct map |
|---|---|
| Character n-grams | 0.846 |
| Hidden state 0 | 0.826 |
| Word n-grams | 0.821 |
| Sentence embedding | 0.683 |

- Base agrees with the Instruct map by 0.050 [0.014, 0.103] more than the best text reader does. After a correction for the noise in the text maps, the gap is 0.037 [0.013, 0.081].
- This gap is small and exploratory. It does not show role information beyond the text.
- Role decoding from the states does not beat word n-grams: 0.762 from base states against 0.797 from word n-grams.

### No stage moved the map in a way I could measure

| Checkpoint | Agreement with Instruct, corrected for reliability (r_norm) |
|---|---|
| Base | 0.996 [0.991, 0.999] |
| SFT | 0.999 |
| DPO | 1.000 |
| Every RL step | 1.000 [1.000, 1.000] |

- The largest step is base to SFT: 0.003 [0.000, 0.008]. A control with equal maps shows a wording bias of 0.027. So these differences are not evidence that a stage moves the map.
- The design has no calibrated test for no change. A planted change in 2 of 20 roles was not detected.
- The states themselves do differ between checkpoints. No two checkpoints are identical.

### The E3 base replies give a weaker picture

| Source | Base and Instruct r | Reading |
|---|---|---|
| E3 `desc` | 0.653 [0.187, 0.693] | Not interpreted, because its reliability is below the floor |
| E3 `ex2` | 0.942 [0.753, 0.940] | Base minus the best text reader 0.060 [0.008, 0.137] |

- These are report only.
- The E3 results exist only under the looser F3 limit of 1e-2 that I accepted after the canary.

---

## How much I trust this

- The rules (THRESHOLDS v1.7) were fixed before the full run. Two changes came after the canary: the F3 equality limit and the bootstrap centring.
- The gates G1 to G5 and the F3 check pass for every checkpoint and source.
- Codex and Claude reviewed the code before the full run. Codex rated a shifted interval a blocker. My choice for the full run was "Original interval", the plain 95% percentile interval.
- The agent challenge found that every reported number matches HF `c6755d6`. It is not a human review.
- The re-run after the challenge changed no earlier point value.
- About a third of the gap between base and the best text reader is noise in the text maps.
- This run reads fixed text. It says nothing about whether base would write in role.

---

## Open points

- No human has reviewed the run.
- A reader from another model family on the same text is not checked.
- The axis and the identifiable subset are not built. The axis needs pod A D3. The subset needs an approved judge panel.
- On 30/10 the folders `out/` and `analysis_full/` and the frames were still on the pod C pod. Their HF copies are checked.

---

## Words used here

| Word | Meaning |
|---|---|
| Pod C | This run. Each checkpoint reads the same fixed text, and I compare what it holds. |
| Pod A | My run `../261029_instruct_persona_map/`: 21,000 OLMo-2-7B Instruct replies for 20 roles plus `default`. |
| E3 | My run `../261008_base_persona_map/`: replies of the OLMo-2-7B base model. |
| `desc`, `ex2` | The two E3 prompt types: the role description only, and the role description plus two example answers. |
| Checkpoint | One saved model: base, SFT, DPO, Instruct, or one of the RL steps `step_60` to `step_360`. |
| Reader | A checkpoint, or a text method, that turns the fixed text into a role map. |
| Role map | How the 20 roles sit relative to each other in a reader's states. |
| Map agreement r | The correlation between two role maps. Each map comes from a different half of the questions and wordings. |
| Reliability | A checkpoint's own cross-half map r. Below a lower bound of 0.5 I do not interpret the checkpoint. This is the same-anchor floor. |
| r_norm | The agreement with Instruct divided by the two reliabilities. |
| Text anchor | A map built from the reply text alone: word n-grams, character n-grams, a sentence embedding, or hidden state 0. |
| One-map null | A test that permutes the role names of one map 9,999 times. |
| F0 | Pod A's own chat input with the role prompt. It is used for gate G3 only. |
| F1 | The main frame: the start token, the E3 interview header, the question, `Guest:` and the reply. It has no role description. |
| F2 | The start token and the reply only. |
| F3 | The role-free control: F1 with one fixed neutral reply in place of every reply. |
| F4 | The length control: only replies with at least 32 kept tokens, pooled over their first 32. |
| F3 check | In F3 every role gets the same input. So the states must be equal across roles, and role decoding must be at chance. |
| D1 | My primary rule: base holds the Instruct map when it reads Instruct text. |
| D2 | Report only: how r_norm changes from stage to stage. |
| G1 to G5 | The integrity gates: same input at every checkpoint, pinned revisions, Instruct reproduces pod A's states, safe storage, and synthetic tests of the analysis code. |
| Canary | A small engineering run (base and Instruct only) that tests the code. Its outputs are not findings. |
| Draw shift | The median of the bootstrap draws minus the point value. |
| A/B pod | The rented pod that ran pods A and B. The re-run after the challenge used it. |
| Agent challenge | A critical review of the result by a Claude session. It is not a human review. |
| HF | The private Hugging Face dataset `brend0nc/manifold-persona-experiment-artifacts` that keeps my run files. |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

Status line of 30/10/2026: full run done; all 10 checkpoints extracted, on HF and checked; analysis done on the pod and on HF. D1 passes on pod A text; an agent challenge (`review/challenge_261030.md`) narrowed the reading; no human review, no external claim. Design: `../261029_stage_persona_plan/DESIGN.md` (pod C; code choices and my decisions there). Rules: `THRESHOLDS.md` v1.7.

Mode: extended (shared code, ten checkpoints, collaborators read the results).

## 1. Code (29/10/2026)

- `checkpoints.py` + `checkpoints.json`: 10 checkpoints pinned to commits (G2 passes). All share one tokenizer (`tokenizer.json` `73fd5254…`). RL `step_360` is not the final Instruct (weight files and 3 of 4 sampled tensors differ). The RL step branches (`step_60` to `step_360` of the Instruct repo, added 13/03/2025) are not documented by AllenAI. SFT and DPO ship `.bin` weights; they load weights-only.
- `replies.py`: pod A rows (shared schema, every row checked) and E3 `run_main` base replies (`desc`, `ex2`; 20 pilot roles; 18 shared questions; 3,600 rows). Each reader refuses the other's schema.
- `frames.py`: F0 to F4 ids and pooling spans, built once; 50 fixed halves per source; the F4 support check; `frame_read.md`.
- `extract.py`, `pod_c_one.sh`, `run_pod_c.sh`: one checkpoint at a time; G1 (fed ids equal the frame file), G3 (Instruct reconstruction of pod A's states), G4 (fingerprinted folder, whole-file writes); the pod uploads each checkpoint to HF and checks it before the next one loads (my rule, 29/10). Deletion of weights or pod outputs only with my approval (`--delete-weights`, `--prune`).
- `fetch_slim.py`: the analysis copy from HF, one file at a time, hidden states 0, 9, 17, 23.
- `maps.py`, `analysis.py`: gates, the F3 check, D1, report-only D2, stage change, RL steps, text anchors, decoding, related pairs, components, G5. Not built: the axis (needs pod A D3) and the identifiable subset (needs an approved judge panel).
- Tests: `pod.sh check` runs the pod-side pod C tests (80 with pods A's; 119 with pod B's); the Mac-only tests (`test_maps.py`, `test_analysis.py`, `test_fetch_slim.py`) pass.

## 2. Frames `frames_full` (29/10/2026)

- Command (Mac): `python frames.py --pod_a ../261029_instruct_persona_map/run_full --out frames_full` (2 min 24 s). Inputs: pod A `run_full` (HF `60a81b9`; `rows.jsonl` `ddb34fab…`) and E3 `run_main` (`replies.jsonl` `443add45…`).
- Rows: pod A 21,000 (F0, F1, F2, F3 for every row; no invalid replies); E3 `desc` 1,800 and `ex2` 1,800 (F1, F2, F3). Canonical encoding (diagnostic): pod A 99.6%, E3 99.9 to 100%.
- F4 support: pod A available (lowest role share 0.796; every half has at least 18 common questions); E3 `desc` unavailable (toddler 0.378; 1 half has no common question); E3 `ex2` available (0.989).
- Frame read (`frames_full/frame_read.md`, 10 rows per frame and span, read 29/10 before any metric): every row has one pooled span, on the reply (F0, F1, F2), on its first 32 ids (F4) or on the neutral reply (F3). Instruct replies follow "Guest:" on a new line; E3 replies keep their leading space. The translator rows read in Russian and French (pod A's raw read found the same).

## 3. Engineering canary `run_canary` (29 to 30/10/2026)

Mode: extended. My go: "Yes, after pod B" (29/10), then "Ok lets run it then". Base and Instruct only; it tests extraction, gates, storage and the analysis code (`../261029_stage_persona_plan/DESIGN.md`, order of work, step 5).

- Pod: the pod A pod (RunPod A100-SXM4-80GB, 213.173.102.5), torch 2.13.0+cu129, transformers 4.47.1, tokenizers 0.21.0, numpy 2.5.2. Code by sha256: `extract.py` `f9c259ba…`, `frames.py` `212a2de7…`, `checkpoints.py` `9fe076fa…`, `rows.py` `b1e4c29b…`; batch size 64, bfloat16, sdpa. Frames `frames_full` (section 2). Fingerprints: Instruct `3de65b63…`, base `d06f0a41…`.
- Command (Mac, repo root): `experiments/q1_emergence/261029_fixed_text_read_through/run_pod_c.sh "ssh root@213.173.102.5 -p 19914" experiments/q1_emergence/261029_fixed_text_read_through/frames_full run_canary experiments/q1_emergence/261029_fixed_text_read_through/records_canary instruct base` (log `run_canary_driver.log`).
- HF token: my existing fine-grained token, saved by `hf auth login` under `/workspace/.cache/huggingface/` (my choice, 29/10: "We kill the machine once it has been used so I'm not particularly worried"); the driver now sets `HF_TOKEN_PATH` when that file exists.
- Smoke (`extract.py --limit 64`, Instruct): every frame passed G1 on 64 rows; 72 s. The 64 rows are the shortest replies, so the first-32 span was empty there (expected). The smoke output (15 GB, full-size arrays) was deleted on my OK.
- First try stalled: after F0 (268 s), the job spent 30 min in the kernel allocating the 12.8 GB F1 arrays (transparent huge pages; a fresh 11.4 GB `np.full` did not finish in 5 min in a separate test, while 1 GB without huge pages took 0.55 s). Fix: `NUMPY_MADVISE_HUGEPAGE=0` in `pod_c_one.sh`. Replacing the running script made bash exit 127 at the upload line; nothing was uploaded. The restart reused F0 (its record existed) and ran with no stall.
- Times: Instruct 779 s after the restart (plus F0's 268 s), base 807 s; each checkpoint was uploaded and checked before the next one loaded.
- Gates: G1 every fed row (fed ids equal the frame file); G2 pass; G3 (Instruct F0 against pod A's stored states, 21,000 valid rows) pass at states 9, 17, 23: min cosine 0.99996 / 0.99984 / 0.99969, median relative norm difference 0.0022 / 0.0031 / 0.0034, max 0.0088 / 0.0129 / 0.0209; G4 one fingerprinted folder per checkpoint.
- HF (`brend0nc/manifold-persona-experiment-artifacts`, private dataset), path `experiments/q1_emergence/261029_fixed_text_read_through/run_canary/`: frames `frames_full/` @ `d166df01032a326e1d9fb02b80cff433d125e067` (4/4 files checked), `instruct/` @ `e3548db375b6182d39f1ea80e2a680f38dfe0351` (27/27), `base/` @ `359493c6d88d788603f875bb34b3857f2a665809` (25/25), `analysis_canary/` @ `33b8f0d1bc032dba2010f2aff496e116f588aed1` (10/10). Receipts and small records: `records_canary/`.
- Slim copy: `fetch_slim.py --records records_canary --out slim_canary` (Mac, 57 min, 6.6 GB, states 0, 9, 17, 23; every file checked against the receipts). This was before my rule of 29/10 that the analysis reads from HF and heavy work runs on the pod.
- Analysis on the Mac: the run (`analysis.py --frames frames_full --out slim_canary --checkpoints base,instruct`) ran over an hour (x86 Python under Rosetta) and stopped at the E3 `desc` F3 check; it saved nothing, since results are written at the end.
- Analysis on the pod (analysis packages installed with the model packages pinned: scikit-learn 1.9.1, scipy 1.18.1, sentence-transformers 5.7.0, joblib 1.6.0; the Mac has 1.9.0, 1.13.1, 3.3.1), with the new `--sources` option: `analysis.py --frames frames_full --out out --results results_canary_pod_a.json --checkpoints base,instruct --sources pod_a` (85 min; `analysis.py` `2631b4d3…`, `maps.py` `4c5fed4c…`). G5 passed.

F3 check in the canary (state 17; rule: max relative difference ≤ 1e-3 within a question, and 20-class decoding at 5% ± 0.5 points):

| Source | Checkpoint | Max relative difference | Equal | Decoding accuracy | At chance |
|---|---|---|---|---|---|
| pod A | base / Instruct | 0 / 0 | yes | 0.0500 / 0.0500 | yes |
| E3 `desc` | base / Instruct | 0.0057 / 0.0068 | no | 0.0503 / 0.0510 | yes |
| E3 `ex2` | base / Instruct | 0.0057 / 0.0068 | no | 0.0503 / 0.0510 | yes |

- The E3 failure is batch numerics, not role leakage: each E3 question has 100 identical F3 rows, fed as one batch of 64 and one of 36, and the two batch shapes round differently in bfloat16; decoding stays at chance. Pod A's 500 rows per question gave identical stored (float16) states. DESIGN.md flagged this risk before the run; the rule was unchanged until I decided.
- Canary outputs, pod A source only (engineering checks of the analysis code; two checkpoints; not findings): D1 base vs Instruct r = 0.896 [0.797, 0.931], one-map null p = 0.0001 (rule passes); text anchors 0.68 to 0.85 (char n-grams 0.846 strongest); base minus the best anchor 0.050 [0.014, 0.087]. Same comparison on F2 0.889, at state 9 0.888, at state 23 0.889, on F4 (first 32 ids, common rows) 0.871. Role decoding from base states 0.76 against word n-grams 0.80 (gain -0.036 [-0.070, 0.000]); Instruct states 0.75. Related pairs: counselor vs consultant above the n-gram baseline; chef vs bartender and detective vs criminal not established. Components for 70%: 4 at both checkpoints.
- Issues for review before the full run: (1) the D2 question-only-halves interval excludes its own point (raw 0.984 against [0.946, 0.982]), a sign of bootstrap bias to check; (2) r_norm sits near its ceiling (0.996); (3) the analysis takes about 85 min for two checkpoints on the pod, mostly decoding fits, so ten checkpoints need faster code or more time; (4) `pool_batch` keeps the KV cache (`use_cache` is on), so GPU memory reached 80 GB on F0 (allocator OOM retries in the log).
- Rounding effect (30/10, $0, Mac, `rounding_effect.py`; reads `slim_canary/` and pod A's `run_full/`): pod A's generation-time states against pod C's F0 re-extraction of the same ids at state 17: per-row gap median 0.0031, max 0.0129; centred role means changed by 0.19%; role-map Spearman r 0.99999. E3 role spread is 13 to 28% of the mean state, against a worst F3 gap of 0.68%.
- My decision (30/10, after these numbers): "ok sounds negligible, lets proceed with next steps". `THRESHOLDS.md` v1.6: the F3 equality limit is 1e-2; decoding at chance stays the leak test. `analysis.py` `f46ed644…` (13 analysis tests pass on the pod).
- E3 sources (pod, 10 min): `analysis.py --frames frames_full --out out --results results_canary_e3.json --checkpoints base,instruct --sources e3_desc,e3_ex2`. F3 passes for both. Report-only D1 (engineering outputs): `desc` r = 0.653 [0.189, 0.679], text anchors 0.45 to 0.56, base minus the best anchor 0.092 [-0.007, 0.184] (not established); `ex2` r = 0.942 [0.780, 0.944], anchors 0.75 to 0.88, base minus the best anchor 0.060 [0.008, 0.114]. F4: `desc` unavailable (toddler share 0.378), `ex2` interval unavailable (one undefined draw under the strict rule). Again the point sits at or above the interval's upper end (issue 1). HF `run_canary/analysis_canary_e3/` @ `be2432ccf3b04f31200c94e7b4cfd4eae94dfd6a` (11/11 files checked).
- Pod copy: `out/`, `analysis_canary/`, `analysis_canary_e3/` and the frames were still on the pod; their HF copies are checked.
- Next at that time: the full run (10 checkpoints; about 250 GB disk without deletion) and its pod; issues 1 to 4 above before it.

## 4. Code review before the full run (30/10/2026)

My words: "lets have codex review the code before we run for stages as well as running /code-review yourself", then "ok fix all". Snapshot reviewed: `review/code_before_fixes_261030/`. Codex: `review/prompt_codex_code_261030.txt`, `review/codex_last_code_261030.txt` (4 blockers, 8 major, 1 minor; read-only, high effort). Claude `/code-review` (high): 10 findings, then 13 on the fixed code.

- Fixed (extraction; changes the code hash, so all ten checkpoints run again and the canary outputs are not reused): decoder-only forward without KV cache or logits; float32 states in disk work files, chunked save; G3 right after F0; smoke test without `--pod_a_run`; a frame resumes only if its arrays match the record; the fingerprint binds `frames.npz` and the runtime versions; peak GPU memory (allocated, reserved) per frame; streamed file hashes (`rows.py`).
- Fixed (analysis): gates bind every checkpoint to the analysed frames (digests and `frames.npz` hash), its pinned revision, one extraction code and runtime, and no smoke limit; D1 applies the same-anchor floors (both reliability lower bounds ≥ 0.5) and never passes on an undefined p; the null and p-value skip undefined permutations; `r_norm` withholds its draws when a reliability draw is ≤ 0; draws centre over the distinct sampled roles (E3 rule); the G5 equal-map wording case now records r_norm 1.03, the D2 difference 0.027 [0.013, 0.039] and the stage change, with both reliabilities above the floor; the embedding cache is bound to the model and texts; results are written per source with provenance and strict JSON; decoding runs in 16 worker processes (`POD_C_JOBS`) with one BLAS thread each.
- Fixed (storage and driver): `fetch_slim` builds in a staging folder and refuses a copy of another revision; receipts are checked by full HF path and hash for the frames and every checkpoint; a restart keeps checkpoints with a verified receipt; tmux sessions are named per job; a dead or unreachable pod stops the driver; the frames get a copy in every job folder; the upload log is appended. `pod.sh` ships the analysis code and tests; `requirements-pod.txt` pins the analysis packages.
- Skipped: pre-computing text features in the parent (it would run the costly fits one at a time).
- Open at that time, my decision: the interval method. The v1.7 change moves each percentile interval by the draws' shift; Codex rates it a blocker (a synthetic D1 lower bound moved from 0.4605 to 0.5405 and flipped the floor decision; no coverage check).
- Tests: 91 pass on the Mac (shared rows and pod C); `pod.sh check` 173 pass, 1 skipped.

## 5. Full run `run_full`: extraction (30/10/2026)

Mode: extended. My go: "Yes, start it" (30/10), after the code review; interval choice: "Original interval".

- Pod: RunPod A100-SXM4-80GB (213.173.102.5, port 10077), 300 GB container disk, torch 2.13.0+cu129, transformers 4.47.1, tokenizers 0.21.0, numpy 2.5.2 (the canary's versions). `pod.sh push`: 173 tests passed on the pod. Pod A `run_full` (for G3) came from HF @ `60a81b9`, 4/4 files matched their sums.
- Code: `extract.py` `f7a2fb67…` (all ten checkpoints; the canary outputs are not reused), `frames.npz` `7cbfb4a3…`, batch size 64, bfloat16, sdpa, decoder only without KV cache.
- Command (Mac, repo root): `experiments/q1_emergence/261029_fixed_text_read_through/run_pod_c.sh "ssh root@213.173.102.5 -p 10077" experiments/q1_emergence/261029_fixed_text_read_through/frames_full run_full experiments/q1_emergence/261029_fixed_text_read_through/records_full` (log `run_full_driver.log`). 10:09 to 12:22 UTC; driver exit 0.

Per checkpoint (seconds, incl. model load; HF revision under `experiments/q1_emergence/261029_fixed_text_read_through/run_full/`, every receipt checked on the Mac by full path and hash):

| Checkpoint | Seconds | HF revision | Files |
|---|---|---|---|
| frames | not timed | `1a3cc7249154fe45248778e5c19258eb2eef3e0f` | 4 |
| instruct | 1,074 | `00e4cf3f4d2b065078c0f1ac605722242f6fb4de` | 27 |
| base | 806 | `d48b29c526ded57bf45dfd2a26bfc7a3cd6ef160` | 25 |
| sft | 648 | `caee60c056862a554d29330bd47f1de22b4e5fb7` | 25 |
| dpo | 657 | `f3442ed9255ce864ceb7b23fceb86909878bd117` | 25 |
| rl_60 | 651 | `8f9a970ff52ec03002c1664cf2d14f6dd9e7831f` | 25 |
| rl_120 | 656 | `ec60563316676482f5b37a949245cdf9c9eb8ccd` | 25 |
| rl_180 | 659 | `a485a18589f008f8f51c78985b9e6f811cd08174` | 25 |
| rl_240 | 667 | `de4f6564be96b52fdc46af851af8962dfe4f03be` | 25 |
| rl_300 | 653 | `645a490caa505a8786aa9ca58e64b36aff942ce8` | 25 |
| rl_360 | 657 | `a6c9ed2fea43ec05fe5c50a883e113bdb0544b05` | 25 |

- Gates: G1 every fed row; G2 pass (each manifest's model commit equals its pinned revision); G3 at Instruct pass (min cosine 0.99996 / 0.99984 / 0.99969 at states 9 / 17 / 23; median relative norm difference 0.0022 / 0.0031 / 0.0034, as in the canary); G4 one folder per checkpoint, same code, frames file and runtime; peak GPU memory 18.6 GB allocated, 53 GB reserved (canary: allocator retries near 80 GB).
- State check (30/10, before reading any map): median relative gap between neighbouring checkpoints at state 17, pod A F1: base to SFT 0.308, SFT to DPO 0.147, DPO to Instruct 0.018, Instruct to RL 60 0.012, RL 60 to 120 0.008, RL 300 to 360 0.004, RL 360 to Instruct 0.0035. No two checkpoints are identical.

## 6. Full run: analysis and results (30/10/2026)

Analysis (pod, 12:40 to 13:19 UTC, 39 min; `POD_C_JOBS=32 analysis.py --frames frames_full --out out --results results_full.json`; `analysis.py` `a4275cfc…`, `maps.py` `718f4113…`):

- No source stopped.
- F3 passes for every checkpoint and source. Pod A: max relative difference 0 at all 10 checkpoints, so pod A also passes the old 1e-3 limit. E3: 0.0057 to 0.0071 at all 10, so the E3 results exist only under the v1.6 limit of 1e-2. E3 `desc` and `ex2` share one F3 frame, digest `ddf28676…`, so they are one check.
- G5 passes.
- HF `run_full/analysis_full/` @ `c6755d666e0bcc207eecd8af37da0e18c305c577` (14/14). Copy of the results and receipt: `records_full/analysis_full/`.
- The results file's `provenance.hf_revision` is `null` for every checkpoint, because the analysis read the pod copies; their HF revisions are the table in section 5 (bound by the receipts in `records_full/`).

Results, pod A text (exploratory; the primary rule decides D1 only):

- D1 base vs Instruct, F1, state 17: r = 0.896 [0.767, 0.932], one-map null p = 0.0001 (9,999 finite), both reliability lower bounds 0.77. So D1 passes. Text anchors: char n-grams 0.846, state 0 0.826, word n-grams 0.821, embedding 0.683; base minus the best anchor 0.050 [0.014, 0.103] ("base exceeds the strongest tested text-anchor agreement"). Draw shift -0.020.
- D2 (report only): r_norm to Instruct 0.996 [0.991, 0.999] at base, 0.999 at SFT, 1.000 at DPO; transitions base to SFT 0.003 [0.000, 0.008], SFT to DPO 0.001 [0.000, 0.002], DPO to Instruct 0.000; raw r 0.896 to 0.899 at every stage; the same with question-only halves. The G5 equal-map case shows a wording bias of the same size (0.027), so these differences are not evidence that a stage moves the map.
- RL steps: r_norm 1.000 [1.000, 1.000] to Instruct at every step; adjacent change 0.000.
- Same comparison elsewhere: F2 0.889, state 9 0.888, state 23 0.889, F4 (first 32 ids) 0.871, F1 on the F4 rows 0.899. Components for 70% of role-mean variance: 4 at every checkpoint.
- Role decoding: base states 0.762, Instruct states 0.755 (chance 0.05), against word n-grams 0.797, char 0.774, state 0 0.757, embedding 0.543; gain over the best text baseline -0.036 [-0.070, 0.000] at base and -0.043 [-0.078, -0.006] at Instruct. Related pairs: counselor vs consultant above word n-grams at every checkpoint (+0.05); chef vs bartender and detective vs criminal not established.

E3 text (report only):

- `desc`: D1 r = 0.653 [0.187, 0.693], reliabilities 0.65 [0.19, 0.69] and 0.66 [0.19, 0.70]. Not interpreted (below the floor).
- `ex2`: D1 r = 0.942 [0.753, 0.940], reliability lower bounds 0.77 and 0.75, base minus the best anchor 0.060 [0.008, 0.137]; r_norm to Instruct 0.992 at base, 0.999 at SFT, 1.000 at DPO.

What this does and does not show (wording from the agent challenge, `review/challenge_261030.md`):

- When OLMo-2-7B base and Instruct read the same Instruct-written replies (F1, state 17), their 20-role maps agree (r = 0.896 [0.767, 0.932]).
- Text-only maps of the same replies agree with the Instruct map almost as well, and a character n-gram map also clears the D1 floor (0.846 [0.696, 0.902]).
- Base agrees with the Instruct map slightly more than the four tested text readers do (0.050 [0.014, 0.103]); about a third of that gap is noise in the text maps (the challenge's anchor check, 200 unpaired draws), so this does not show information beyond the text.
- Role decoding from states does not beat word n-grams.
- I found no measurable change in the fixed-text role map between base, SFT, DPO, Instruct and the RL steps under this measure; the design has no calibrated test for no change.
- This is about reading fixed text; it says nothing about whether base would write in role (E3 and pod B).
- Not yet checked: a human review; a reader from another model family on the same text.
- The frame read (`frames_full/frame_read.md`) covers the full run, since the frames were built once.

Pod copy: `out/`, `analysis_full/` and the frames were still on the pod; their HF copies are checked.

## 7. Agent challenge and the re-run after it (30/10/2026)

Agent challenge (30/10, session 48adb487, `review/challenge_261030.md`; not a human review): every reported number matches HF `c6755d6`.

Code changes after it (not in any results file at that time):

- D1 now saves each anchor's own reliability, its r_norm to Instruct and the normalized base-minus-best-anchor gap.
- G5 adds the two cases THRESHOLDS already listed as pass conditions (uniform p under the one-map null: mean p 0.585, 6.7% below 0.1; planted same-row noise removed: cross-half r 0.036).
- G5 records planted map changes (true map r 0.93 for all roles shifted: change 0.071 [0.022, 0.145], detected; 2 of 20 roles moved, true r 0.83: change 0.156 [-0.006, 0.340], not detected).
- 91 tests pass on the Mac. `results_full.json` at `c6755d6` predates these changes.

Re-run after the challenge (my go, 30/10: "ok go ahead and do that", given through the challenge session; the run was done by the session that wrote this record):

- Where: the A/B pod (213.173.102.6), own folder `/root/mp`, 14:08 to 15:02 UTC.
- Steps: `fetch_slim.py --records records_full --out slim_full` (states 0, 9, 17, 23 from HF `run_full`, each file checked against its receipt; 31 GB), then `POD_C_JOBS=24 analysis.py --frames frames_full --out slim_full --results results_full_v2.json` (`analysis.py` `f6586f5d…`, `maps.py` `718f4113…`). No source stopped.
- HF `run_full/analysis_full_v2/` @ `b343c96522365de638d7971d0a6969c294b4d41f` (15/15); copy in `records_full/analysis_full_v2/`; `provenance.hf_revision` now names each checkpoint's HF revision. Every earlier point value is unchanged.
- Pod A anchors (own reliability; r_norm to Instruct): char 0.864 [0.706, 0.920], 0.960 [0.912, 0.982]; word 0.851, 0.939 [0.868, 0.974]; state 0 0.879, 0.929 [0.760, 0.980]; embedding 0.852, 0.780 [0.630, 0.905]. Base minus the best anchor after the same correction: 0.037 [0.013, 0.081] (raw 0.050 [0.014, 0.103]).
- E3 `ex2`: normalized gap 0.067 [0.020, 0.123]; E3 `desc`: unavailable (reliability floor).
- G5 passes, including null uniformity (mean p 0.585, 6.7% below 0.1) and planted shared noise (cross-half r 0.036). Planted changes, recorded: all roles shifted (true map r 0.93) 0.071 [0.022, 0.145]; 5 of 20 roles moved (true r 0.60) 0.362 [0.105, 0.697]; 2 of 20 roles moved (true r 0.83) 0.156 [-0.006, 0.340]. So "no measurable change" across stages can still hide a change in a few roles.
- Reading: after correcting the text maps for their noise, base still agrees with Instruct slightly more than any tested text reader (0.037, interval above 0). This is small and exploratory; it does not show role information beyond the text.

## 8. A/B pod cleanup (30/10/2026)

My answer: "1." = delete. `slim_full/` (31 GB), `analysis_full_v2/` and `results_full_v2.json` were removed from `/root/mp` on the A/B pod; their HF copies are checked (`run_full/` checkpoints and `analysis_full_v2` @ `b343c96`). Pod B's tmux and folder were not touched. The pod C code in `/root/mp` stays.
