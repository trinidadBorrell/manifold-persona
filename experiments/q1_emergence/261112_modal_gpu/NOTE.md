# Modal GPU runs (12/11/2026)

This is the note for `261112_modal_gpu`. It is an infrastructure note: one Modal app that ran two of my experiments. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| What this is | One Modal app (`modal_runs.py`) for two runs: the coin-flip test on the four OLMo-2-7B stages, and the Instruct prompt-only control (C3). |
| Result | Both full runs finished on 12/11. This folder holds no research result. The results are in `../261112_coinflip_stages/NOTE.md` and `../261112_instruct_prompt_only/`. |
| How much I trust it | The C3 environment check matched pod A's stored states (all cosines 1.000). The HF copy matches the Modal volume on SHA-256 and size for all 57 files. The smoke run found one flaw in its 40-item subset. It is fixed, and the full run was never affected. |
| What I do not know | Whether the weights volume is deleted. The plan was to delete it after the runs. This note does not record a deletion. |
| Status (13/11/2026) | The prefetch, the smoke run and the full run are done. The outputs are on HF at revision `7e497190b03a56448f39306acd10b9550bdef9b7`. |

---

## What I did

1. I approved one Modal app for both runs on 12 Nov: "Yes, build both". Then: "We can do that test quickly on modal no? And we can save storage here, cause it goes to modal and hugging face".
2. A CPU job downloaded the four OLMo-2-7B checkpoints to a Modal volume (prefetch).
3. A smoke run on one A100-80GB tested the code on a small sample. It found one flaw in the sample, and the flaw was fixed.
4. The full run did 400 coin-flip items on each of the 4 stages and all 4,200 C3 prompts.
5. On 13/11 the outputs went from the Modal volume to HF in one commit, and every file was checked.

---

## Words used here

| Word | Meaning |
|---|---|
| Modal | The cloud service that I rent GPUs from for these runs. |
| Stage | One OLMo-2-7B checkpoint: base, SFT, DPO or Instruct. |
| Coin-flip test | The run in `../261112_coinflip_stages/`. It reads the model's next-word probabilities for "heads" and "tails" on 400 items. |
| Pod A | My run `../261029_instruct_persona_map/`: OLMo-2-7B Instruct replies to role prompts. |
| Pod B | My run `../261029_base_role_vs_format/`: the OLMo-2-7B base model with role descriptions. |
| Pod C | My run `../261029_fixed_text_read_through/`. It used the same GPU card as pod A. |
| C3 | The prompt-only control from pod B: the model reads the prompt and no reply of its own. Here it runs on pod A's prompts with the Instruct model. |
| Prefetch | The CPU step that downloads the model weights to a Modal volume before any GPU run. |
| Smoke | A small test run that checks the code before the full run. Its numbers are not results. |
| b | The base rate in the coin-flip rules: the mean share of "heads" against "tails" (mean q). |
| Captured mass | P(heads) + P(tails): how much of the next-word probability the two outcome words hold. |
| Lens | The logit lens: the model's output layer applied to the state of each layer. |
| HF | The private Hugging Face dataset `brend0nc/manifold-persona-experiment-artifacts` that keeps my run files. |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

## 1. The two runs

One Modal app (`modal_runs.py`) for two runs. My words on 12 Nov: "Yes, build both", then "We can do that test quickly on modal no? And we can save storage here, cause it goes to modal and hugging face".

| Run | Folder | Rules |
|---|---|---|
| Coin-flip test, OLMo-2-7B base, SFT, DPO, Instruct | `../261112_coinflip_stages/` | `THRESHOLDS.md` v1.1 |
| Instruct prompt-only control (pod B's C3 on pod A's prompts) | `../261112_instruct_prompt_only/` | `THRESHOLDS.md` v1.0 |

## 2. Where things are stored

| What | Where | Detail |
|---|---|---|
| Weights | Modal volume `persona-olmo2-7b-weights-261112` | HF cache of the four pinned 7B revisions in `../261112_coinflip_stages/stages.json`, about 73 GB. Anonymous download, no token. Delete the volume after the runs. |
| Outputs | Modal volume `persona-gpu-runs-261112`, under `smoke/` and `full/` | Coin-flip JSON per stage and mode, `summary.json`, `summary.md`. C3 states, keys, manifest, `SHA256SUMS.txt`. Also `run.log`. |
| Mac | This folder | Code, rules and small summaries only. |
| HF write access | Modal secret `hf-write` | Created 12/11 23:58 SAST on my request ("I have a hf token that is under .hf_token that you can add to modal to write"), from the repo's `.hf_token` (git-ignored; token never printed). |

- The 1B pilot on the Mac was stopped, because the CPU was too slow (about 20 minutes per mode). Its files and the scratch environment were deleted.
- The secret was checked from inside Modal: user `brend0nc`, fine-grained, `repo.write`. It reads the private dataset `brend0nc/manifold-persona-experiment-artifacts` (head `beacc63c09ab`).
- On 12/11 no upload was done yet. Section 6 has the upload of 13/11.

## 3. Environment

- Image: Debian slim, Python 3.12, torch 2.13.0+cu129, transformers 4.47.1, tokenizers 0.21.0, numpy 2.5.2, huggingface_hub 0.36.2, safetensors 0.8.0, hf-xet 1.7.1 (pod A's versions).
- GPU: one A100-80GB (pod A and pod C card). Hard time limits: smoke 30 minutes, full 90 minutes. `retries=0`, one container at a time.

## 4. Steps

| Step | What | State |
|---|---|---|
| prefetch | CPU: download the four checkpoints to the weights volume | Done 12/11 (app ap-bWbHOofuwKL5eQZHFDj24u): base 29.2 GB, SFT, DPO, Instruct 14.61 GB each, at the pinned revisions; `prefetch.log` |
| smoke | GPU: coin-flip, 40 items on base and Instruct; C3, environment check and 32 prompts | Done 12/11 (app ap-5DURimmcAMRRVBAV6O6xMg, A100-SXM4-80GB); `smoke.log`. See section 5 |
| full | GPU: 400 items x 4 stages (v1.2: original paragraph layout primary, one paragraph as a side check); all 4,200 C3 prompts | Done 12/11 (app ap-2wzS1GUdRmDRnBUYUZVGAM; my go: "Ok we can do that recommended"). C3 environment check cosines all 1.000; C3 took 41.7 s. Results: `../261112_coinflip_stages/NOTE.md`; C3 states copied to `../261112_instruct_prompt_only/run_c3_full/` (checksums OK) |

## 5. Smoke result (12/11/2026): the code works, and the numbers are not results

- Coin-flip: it ran end to end on 7B base and Instruct, all modes, logit lens and analysis. Captured mass 0.88 (base) to 0.99. Lens versus the model's own bfloat16 output: at most 0.0026 (base) and 0.015 (Instruct) in probability. Time: base 233 s (float32 weights loaded as bfloat16), Instruct about 17 s per mode.
- Flaw found: the 40-item subset (every 10th item) kept only items where the harmless task gets "heads". So its "bias" equals 2b - 1, a liking for the word "heads" (base b 0.677 gives +0.354; Instruct open-user-turn b 0.377 gives -0.246).
- Fix: `limit_items` keeps whole pairs (all four 2x2 items); test added (11 tests pass). The full run uses all 400 items and was never affected.
- C3: the environment check re-pooled 8 pod A replies at state 17 and matched pod A's stored states (all cosines 1.000). 32 prompts in 8.3 s.

## 6. HF copy (13/11/2026)

Uploaded from the Modal volume in one commit and checked (SHA-256 and size of all 57 files match): `brend0nc/manifold-persona-experiment-artifacts` revision `7e497190b03a56448f39306acd10b9550bdef9b7`. Receipt: `../261112_modal_gpu/hf_receipt_261113.json`. Uploader: `../261112_modal_gpu/hf_upload.py` (Modal secret `hf-write`).

Dataset folders:

| Content | Folder in the dataset |
|---|---|
| Coin-flip | `261112_coinflip_stages/run_{full,smoke}/` |
| C3 | `261112_instruct_prompt_only/run_c3_{full,smoke}/` |
| Own-reply smoke | `261112_stage_own_replies/` |
| Logs | `261112_modal_gpu/logs/` |
