# Instruct prompt-only control (12/11/2026)

This is the note for `261112_instruct_prompt_only`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | In OLMo-2-7B Instruct, does one fixed neutral reply after the role prompt give the same role map as the roles' own replies? If it does, pod A's role map may be only a map of the prompts. |
| Result | The prompt alone gives a role map (r = 0.905 [0.829, 0.936]). It even decodes the role better than the replies do (0.991 against 0.918). But the reply map is not the prompt map. The two maps agree at 0.706, and the paired difference of 0.195 [0.066, 0.315] is above 0. So the replies add role structure beyond the prompt. |
| How much I trust it | The rules were fixed before the run. The environment check reproduced pod A's states (cosines 1.000). The recomputed reply map equals pod A's reported value (0.901). States 9 and 23 agree in direction. This is agent analysis and report only. No human has reviewed it. |
| What I do not know | Whether another neutral reply gives the same result. I used one fixed reply. The result is for Instruct only. |
| Status (13/11/2026) | Done: the extraction on Modal (12/11), the analysis on my Mac (13/11) and the HF copy (13/11). |

---

## What I did

1. The rules (THRESHOLDS v1.0) came first.
2. A smoke run of 32 prompts on Modal checked the environment against pod A's stored states.
3. The full run gave Instruct pod A's 4,200 prompts, each followed by the same fixed neutral reply. It took 41.7 s.
4. The analysis ran on my Mac on 13/11, with 2,000 bootstraps, 9,999 permutations and 50 halves.
5. The files went to HF and were checked.

---

## Results

All numbers use hidden state 17.

### The prompt alone gives a role map

| Map | Split-half r |
|---|---|
| Reply map (pod A's D2, recomputed) | 0.901 [0.797, 0.939] |
| Prompt-only map (M1) | 0.905 [0.829, 0.936], p 0.0001 |

- My rule for "the prompt alone gives a role map" is met.
- Role decoding from the prompt-only states is 0.991 [0.981, 0.998]. From the reply states it is 0.918 [0.879, 0.950].
- So the prompt alone carries the role strongly.

### The reply map is not the prompt map

| Measure | Value |
|---|---|
| Reply map against prompt-only map (M2) | 0.706 [0.541, 0.825] |
| D2 - M2, paired | 0.195 [0.066, 0.315] |

- The interval of D2 - M2 is above 0. By my rule that reads: "the replies add role structure beyond the prompt".
- States 9 and 23 agree in direction: 0.375 [0.157, 0.552] at state 9 and 0.151 [0.053, 0.243] at state 23.

---

## How much I trust this

- The environment check passed in the smoke and the full run: 8 pod A states were reproduced with cosines 1.000.
- The recomputed D2 reproduces pod A's reported 0.901.
- The control uses one fixed neutral reply.
- The C3 states have no sampling noise.
- All results are report only, and this is agent analysis with no human review.

---

## Words used here

| Word | Meaning |
|---|---|
| Pod A | My run `261029_instruct_persona_map`: OLMo-2-7B Instruct replies to role prompts, with their stored states. |
| Role map | The table of distances between the role vectors. |
| Reply map | The role map from the states of the roles' own replies (pod A). |
| Prompt-only map | The role map from the states of one fixed neutral reply after each role prompt. |
| C3 | The name of this control in pod B: the full prompt plus one fixed neutral text in place of the reply. |
| Split-half r | The correlation of a map from one half of the questions and wordings with the map from the other half. |
| D2 | Pod A's rule that the role map repeats. Here it also names the split-half r of the reply map. |
| M1 | The split-half r of the prompt-only map. |
| M2 | The agreement between the reply map from one half and the prompt-only map from the other half. |
| M3 | Role decoding: a linear decoder predicts the role from a state. |
| State 17 | Hidden state 17, pod A's primary state. |
| Environment check | Before the run, the code pools 8 of pod A's real replies again and compares them with the stored states. |
| HF | The private Hugging Face dataset `brend0nc/manifold-persona-experiment-artifacts` that keeps my run files. |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

Rules: `THRESHOLDS.md` v1.0. Code: `extract_c3.py`, `analysis_c3.py`, tests `test_extract_c3.py` (5 pass).

## 1. Steps

| Step | Where | State |
|---|---|---|
| Smoke, 32 prompts | Modal A100, app `ap-5DURimmcAMRRVBAV6O6xMg` | Environment check: 8 pod A states reproduced, cosines 1.000 |
| Full, 4,200 prompts | Modal A100, app `ap-2wzS1GUdRmDRnBUYUZVGAM` | Done 12/11 in 41.7 s; cosines 1.000; states copied to `run_c3_full/` (checksums OK) |
| Analysis (M1-M3) | This Mac, `analysis_c3.py` (2,000 boots, 9,999 permutations, 50 halves) | Done 13/11: `results_full/c3_results.json`, log `analysis_full.log` |

## 2. HF copy (13/11/2026)

Uploaded from the Modal volume in one commit and checked (SHA-256 and size of all 57 files match): `brend0nc/manifold-persona-experiment-artifacts` revision `7e497190b03a56448f39306acd10b9550bdef9b7`. Receipt: `../261112_modal_gpu/hf_receipt_261113.json`. Uploader: `../261112_modal_gpu/hf_upload.py` (Modal secret `hf-write`).

Dataset folders: `experiments/q1_emergence/261112_instruct_prompt_only/run_c3_full/` and `run_c3_smoke/`.

## 3. Results (state 17; states 9 and 23 agree in direction)

Agent analysis, report-only; no human review.

| Measure | Value | Rule outcome |
|---|---|---|
| D2, reply map split-half r (recomputed; pod A reported 0.901) | 0.901 [0.797, 0.939] | Check: reproduces pod A |
| M1, prompt-only (C3) map split-half r | 0.905 [0.829, 0.936], p 0.0001 | "The prompt alone gives a role map": yes |
| M2, reply map vs C3 map | 0.706 [0.541, 0.825] | |
| D2 - M2, paired | 0.195 [0.066, 0.315] | Above 0: "the replies add role structure beyond the prompt" |
| M3, role decoding from C3 states | 0.991 [0.981, 0.998] | |
| M3, role decoding from reply states | 0.918 [0.879, 0.950] | |

- State 9: M1 0.827, D2 0.890, M2 0.515, D2 - M2 0.375 [0.157, 0.552]. State 23: M1 0.885, D2 0.901, M2 0.750, D2 - M2 0.151 [0.053, 0.243].
- Reading for the draft limit ("this may be a map of the prompts"): the prompt alone carries the role strongly (it decodes the role better than the replies do), but the reply map is not the prompt map; the replies add role structure that the prompt-only map does not have.
- Limits: one fixed neutral reply; C3 states have no sampling noise; report-only; Instruct only.
