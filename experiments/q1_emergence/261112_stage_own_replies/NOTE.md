# SFT, DPO and Instruct on their own replies

This is the note for `261112_stage_own_replies`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | At which OLMo-2-7B stage does the default become special, when each stage writes its own replies? The default is the model with no role prompt. |
| Result | The default already sits near the assistant end of the role line at SFT. It sits closer at DPO and Instruct. Its replies vary more than a typical role's replies at every stage. None of my ten tests shows a change between stages. |
| How much I trust it | I fixed the rules before any number. The power check passed. A Codex review found no wrong main number. The move from SFT to DPO is a hint only, because my test is weak in that direction. No human has reviewed the analysis. |
| What I do not know | Whether the move from SFT to DPO is real. Whether the result holds with more than 20 roles. How much SFT's weaker role play changes the comparison. |
| Status (13/11/2026) | The generation, my transcript read, the analysis, the Codex review and the re-run are done. The judge pass rate (about $34) is not run. It needs my separate go. |

---

## What I did

1. SFT and DPO each wrote 21,000 replies on pod A's grid: 20 roles plus the default, 5 wordings, 40 questions and 5 replicates. The Instruct replies come from pod A.
2. I read 20 prompts with the three stages side by side, before any number.
3. I approved the analysis rules (THRESHOLDS v1.2), also before any number.
4. The analysis ran on Modal CPU. It first checked that the measures find shifts that I planted. Then it ran the real measurement.
5. Codex reviewed the analysis code. I had its code findings fixed and the analysis run again. The results did not change.

---

## Results

All numbers use hidden state 17. I give each number for full replies and for the first 16 tokens of each reply. I need both because SFT replies are much shorter: 4% of SFT replies reach the 128-token cap, against 63% for DPO and 59% for Instruct.

### The default is near the assistant end, and closer at DPO

I put the 20 roles on their main axis. One end holds the assistant-like roles. The table gives the default's distance from that end. On this scale 0 is the assistant end and 1 is the other end. Lu et al. report 0.03.

| Stage | Full replies | First 16 tokens |
|---|---|---|
| SFT | 0.110 | 0.086 |
| DPO | 0.041 | 0.045 |
| Instruct | 0.040 | 0.043 |

- The default is never outside the role range. In role SD units it sits about 1 above the role mean (0.90 to 1.31), on the assistant's side.
- My rule called the default "at one end" only above 2 role SD. That was a design mistake. It tests "outside the role range", and the assistant end is only about 1.3 role SD from the role mean. The table above is the better reading.
- Han Xiao's rule calls the default "extreme" when the upper 95% bound is at most 0.05. No stage passes it in these two views. DPO and Instruct pass it only on full replies of the prompts that all three stages share (0.048 and 0.049).
- The move from SFT to DPO has no test in my rules. I make no claim from it.

### The default's replies vary more than a typical role's

The spread compares the default's variation with the median role's variation. Above 0 means the default varies more.

| Stage | Full replies | First 16 tokens |
|---|---|---|
| SFT | 0.089 | 0.090 |
| DPO | 0.176 | 0.188 |
| Instruct | 0.179 | 0.194 |

- Every 99.5% interval is above 0. So no stage makes the default a tighter group than the roles.
- For SFT this depends on its short replies. On the shared prompts only, the SFT value is 0.033 and its interval includes 0.

### No change between stages passed my tests

| Change | Reading |
|---|---|
| Position, SFT to DPO | No change seen |
| Position, DPO to Instruct | No change seen |
| Spread, SFT to DPO | Up on the first 16 tokens, not clear on full replies. No claim |
| Spread, DPO to Instruct | No change seen |

- "No change seen" does not mean "no change". From SFT to DPO, the position test is weak for a move toward the assistant side. It would catch a move of 1.0 role SD only 3.4% of the time.
- DPO and Instruct have almost identical role scores on the main axis (r 0.9995 over the 20 roles). I did not compare their full role maps.

### The reply words alone give a similar picture

- I ran the same measures on the reply text only (character n-grams, no model states).
- The position from the text is similar: 0.95, 1.01 and 1.02 on full replies.
- The spread from the text is larger than the spread from the hidden states.
- So for position, the hidden states do not show more than the words show.

### Han Xiao's group test

- In role SD units, the gap between the desirable and the undesirable roles is near 0 for every stage pair (p 0.63 to 1.00).
- In raw units the gap is positive. That fits a change in overall scale, but this run does not test it.

### My read of 20 prompts per stage

| Stage | Yes | No | Partly in role |
|---|---|---|---|
| Instruct (pod A) | 17 | 3 | 6 |
| SFT | 13 | 7 | 10 |
| DPO | 16 | 4 | 5 |

- SFT plays the fewest roles. Most of its "no" replies give generic advice.
- Void failed at all three stages. The default reply passed at all three.

---

## How much I trust this

- I approved the rules before any number, after I read the transcripts.
- The power check passed. Planted shifts of 0.5, 1 and 2 role SD read back as about 0.50, 1.01 and 2.02.
- Codex reviewed the analysis code after the run. Its verdict was "fix first", and it found no wrong main number. Four of its ten findings were about wording, and six were code weaknesses with no effect on this run.
- I had the code fixed and the analysis run again. The ten test readings and every main number are the same.
- The position test has low power from SFT to DPO with 20 roles.
- SFT plays fewer roles than DPO and Instruct, so each stage's roles are not equally well played.
- This is agent analysis. No human has reviewed it, and the fixed code has had no second review.

---

## Open points

- The judge pass rate per stage (about $34) needs my separate go.
- A test for the SFT to DPO move on Lu's scale is not in my rules. It would need new rules before any new number.
- The fixed analysis code has had no second review.
- The working files `acts_parts/` and `rows.partial.jsonl` stay on the Modal volume until I approve removal.

---

## Words used here

| Word | Meaning |
|---|---|
| Default | Replies with no role prompt: four identity wordings and one with no system message. |
| Stage | One step of OLMo-2-7B post-training: SFT, then DPO, then Instruct. |
| Main axis | The first principal component (PC1) of the 20 role means. The roles spread out most along it. |
| Position | Where the default sits on the main axis, in role SD, measured from the role mean. Above 0 is the side the assistant role leans to. |
| Role SD | The standard deviation of the 20 roles' positions on the main axis. |
| Lu's scale | The default's place between the lowest and the highest role on the main axis, from 0 to 1 (Lu et al.). |
| Han's rule | Han Xiao's reading rule for Lu's scale: "extreme" if the upper 95% bound of the distance to the end is at most 0.05. |
| Spread | The log of the default's variance over the median role's variance, within each wording. Above 0 means the default varies more. |
| Full replies, first 16 tokens | The two views. The model state is averaged over all kept reply tokens, or over the first 16 only. |
| Shared prompts | The 19,100 of 21,000 prompts whose reply has at least 16 tokens at all three stages. Also called the common first16 mask. |
| Power check | A check that runs before the real numbers. It plants known shifts in the default's data and tests whether the measures find them (P1, P2), then estimates how often a planted change would be detected (P3). |
| Map gate | Pod A's rule D2: the role map must repeat across two halves of the questions (r lower bound at least 0.5) before I read a stage. |
| Gates G1 to G5 | Pod A's technical checks after generation: rows and grid, valid rate, same seed gives the same reply, stored states match a fresh pass, model and tokenizer pins. |
| HF | The private Hugging Face dataset `brend0nc/manifold-persona-experiment-artifacts` that keeps my run files. |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

## 1. Generation of the SFT and DPO replies (12 and 13 Nov)

Rules: `THRESHOLDS.md` v1.0 and v1.1. Code: `stage_run.py` (pod A's generator and gates, unchanged; model and revision swapped), `modal_own_replies.py`, `test_stage_run.py` (8 pass, including: SFT and DPO render pod A's 21,000 stored prompt ids exactly).

| Step | Where | Date | State |
|---|---|---|---|
| Smoke: canary roles, 40 replies per stage, then pod A's gates | Modal, two A100-80GB, app `ap-AzBbHy1XAV4XRalhJX5C4K` | 12/11 | SFT and DPO: 40 of 40 valid, G1-G5 all pass. SFT plays the roles (e.g. pirate: "Ahoy there! ..."). Outputs on volume `persona-gpu-runs-261112`, `own_replies/{sft,dpo}_smoke/`; SFT raw read copied to `smoke_copies/` |
| Codex review of the setup | `review/codex_rerun_setup_261112.md` | 12/11 | Done. Verdict "fix first": (1) high, the launcher can hide failures, misreads gate output, and does not require a passed smoke before full; (2) medium, crash-recovery edge cases in pod A's resume; (3) medium, the resume fingerprint omits prompt content and runtime; (4) low, relative paths; (5) low, storage about 11.35 GB per stage, not 5.7. No finding on model swap, revisions, .bin loading, offline mode, parallel writes, seeds, sampling or pooling. Call it a matched pod A stage comparison, not a replication of Han Xiao. Shown to me on 13/11 ("show me first") |
| Fixes from the Codex review (THRESHOLDS v1.1) | `stage_run.py`, `modal_own_replies.py` | 13/11 | Done; `test_stage_run.py` 8 pass (adds resume repair, receipt and order tests) |
| Smoke v2 attempt 1 | Modal | 13/11 | Failed, and the launcher now reported it: the receipt was written inside the new run folder, which pod A's code refuses. Fixed (receipt beside the folder; test added); the two folders held only that receipt and were removed |
| Smoke v2: canary roles, 40 per stage, v1.1 wrapper | Modal, two A100-80GB | 13/11 | SFT and DPO: fresh start, 40 of 40 valid, G1-G5 all pass; receipts `own_replies/{sft,dpo}_smoke_v2.stage_receipt.json`. DPO raw read copied to `smoke_copies/dpo_v2_raw_read.md`: toddler played in 2 of 3 shown replies; the third starts in role, then speaks like a general assistant |
| Full: 21,000 replies per stage, then gates | Modal, two A100-80GB, app `ap-FEa4n1cUOtt8q83wEFxSEF` (detached) | 13/11 | Done on my go ("Can we do the sft and dpo full run"). Both stages: fresh start, no repairs, grid `10c347a1e60d` (same as pod A), 21,000 rows, 0 invalid, states [21000, 33, 4096], G1-G5 all pass. Log `full.log`; outputs `own_replies/{sft,dpo}_full/`; small files (rows, manifest, gates, raw read, receipt) copied to `full_copies/{sft,dpo}/`, checksums match the run's `SHA256SUMS.txt` |

Stop reasons of the full run (no metric computed at this point):

| | SFT | DPO | Instruct (pod A, for reference) |
|---|---|---|---|
| Time | 3,234 s | 5,009 s | 4,053 s |
| Replies stopped at end of text | 20,162 (96%) | 7,715 (37%) | 8,546 (41%) |
| Replies cut at 128 tokens | 838 (4%) | 13,284 (63%) | 12,454 (59%) |
| Other stop | 0 | 1 (marker) | 0 |

- Reply length differs a lot by stage. SFT replies are short; DPO and Instruct replies mostly run to the 128-token cap. The reply mean is taken over a different number of tokens per stage, so the analysis rules had to handle length before any cross-stage comparison.
- Raw read (`full_copies/{sft,dpo}/raw_read.md`, the same 20 keys for both stages). This is Claude's first read. It is not a judge label, and it is not my read:
  - SFT plays some roles in short replies (chef, toddler, teenager, cynic, narcissist, one pirate). It drops others into generic advice (translator gives advice instead of a translation; criminal gives a careful, law-abiding answer; vampire and void give one plain sentence).
  - DPO plays more roles and at length: the translator translates into Russian, the criminal gives covert advice, the vampire and the pirates stay in voice. One optimist reply is a generic numbered list.
  - Default, both stages: "I don't have feelings / emotions" opening; DPO adds a numbered list.
- Rule for the next step: I read 20 transcripts per stage before any metric (THRESHOLDS v1.0, gates).

## 2. HF copies of the generation (13 Nov)

First commit: uploaded from the Modal volume in one commit and checked (SHA-256 and size of all 57 files match). Dataset `brend0nc/manifold-persona-experiment-artifacts`, revision `7e497190b03a56448f39306acd10b9550bdef9b7`. Receipt: `../261112_modal_gpu/hf_receipt_261113.json`. Uploader: `../261112_modal_gpu/hf_upload.py` (Modal secret `hf-write`). Dataset folders: `experiments/q1_emergence/261112_stage_own_replies/sft_smoke/`, `dpo_smoke/` and `logs/`.

Second commit (full runs): revision `f224ced8b6078ea1065b6978a0c94dffee5cbbb3`, 48 files, 11.47 GB, SHA-256 and size of all 48 match the volume. Receipt: `../261112_modal_gpu/hf_receipt_261113_full.json`; uploader `hf_upload.py --batch 261113_full`. Dataset folders: `sft_full/`, `dpo_full/` (rows, `reply_mean_acts.npy` 5.68 GB each, manifest, gates, raw read, checksums), `sft_smoke_v2/`, `dpo_smoke_v2/`, the four `*.stage_receipt.json` files and `logs/`.

Not uploaded: `acts_parts/` and `rows.partial.jsonl` (working files). They stay on the Modal volume until I approve removal.

## 3. My transcript read (13 Nov)

The THRESHOLDS v1.0 gate: I read 20 transcripts per stage before any metric. Done 13/11, 09:19 to 10:01 UTC, before any metric on these replies.

- Format: my choice, "Local label page (Recommended)". Page `owner_read/read_page.py` + `read_page.html` (port 8768). The 20 keys of `full_copies/side_by_side_20.md`, one prompt per screen, the three stages side by side, stage names shown (not blind).
- Exact input: rendered with the run's own code (`stage_run.load_pod_a(stage)`, then pod A's `build_items` + `render`, each stage's tokenizer at its pinned revision). All 60 rendered prompts equal the stored `prompt_ids`; the input text is the same in the three stages.
- Files (outside the repo): items `/Users/brendon/persona-manifolds/judge_validation_261029/ownreply261113_items.jsonl`, sha256 `4fb3e2ba7eab869aeeba3fd948146f065a180e1b6bee467a30292617c34bc5ad`; labels `ownreply261113_owner_labels.json`, sha256 `a02bdd1bfb3fcba3f261bfae270d64397b0ef28bf542c92b51f6c03e504c1397`, 60 of 60 labelled.
- Question per reply: role keys "Plays the {role}, and makes sense?"; the default key "Normal assistant voice, and makes sense?". Optional class: E in role, P partly in role, G not enough evidence, W a different role, U cannot score.

| Stage | Yes | No | Class P (partly in role) | Class W or U |
|---|---|---|---|---|
| Instruct (pod A) | 17 | 3 | 6 | 1 |
| SFT | 13 | 7 | 10 | 2 |
| DPO | 16 | 4 | 5 | 3 |

| Key | Role | Instruct | SFT | DPO |
|---|---|---|---|---|
| k01 | translator | yes, P | no, W | yes, P |
| k05 | criminal | yes | no, W | no, U |
| k06 | toddler | no, W | yes | no, U |
| k10 | refugee | no, P | no, P | yes |
| k12 | optimist | yes | no, P | no, W |
| k16 | pirate | yes | no, P | yes |
| k18 | vampire | yes | no | yes |
| k19 | void | no, P | no | no, P |
| k20 | default | yes | yes | yes |

The other 11 keys: yes in all three stages. Full labels are in the labels file.

My notes, word for word (the only notes; no per-prompt note):
- k01 translator, Instruct: "It didn't answer the question, but it translated the question"
- k01 translator, DPO: "It didn't answer the question, but it translated the question"
- k01 translator, SFT: "It is answering the questions, which I believe it is the result of " (as saved)

My decisions in chat: none beyond "Done reading". No judge label or metric was shown before or during the read.

What the read shows (Claude's summary of my labels, not a measure): SFT has the most "no" and the most "partly in role" labels; its "no" replies are mostly generic answers (translator, criminal, optimist, vampire). Void fails in all three stages. The default reply passes in all three.

Rules: I approved THRESHOLDS v1.2 on 13/11/2026 ("Yes, I approve the rules"), after a plain 10-line summary.

## 4. Analysis under THRESHOLDS v1.2: setup and first run (13 Nov)

Started on my go ("Go, run it now"). Code: `analysis_v12.py` (prep, source, power, measure), `modal_analysis_v12.py` (Modal CPU, detached; a remote orchestrator runs each step), `test_analysis_v12.py` (11 pass on CPU, synthetic data: planted shift and scale through the sums equal a recompute; planted shifts move the signed point by the planted amount; the sign follows the Instruct profile, not the feature sign; the map gate and Lu scale run; Han's exact permutation has 252 and 70 splits; the key, first16 and receipt checks catch a swapped row, a wrong NaN row and a changed file; P3 ignores the real offset; an end-to-end run on a fake volume).

- Statistics imported, not rewritten: `../261112_free_checks/check_role_count.py` (cell sums, position, spread, question weights, the 50 fixed splits), `../261112_free_checks/check3_lu_scale.py` (Lu scale), pod A's `analysis.py` + pod C's `maps.py` (map gate), Han's `persona_axes.axis_projections`.
- Sources: 3 stages x (full: states 17, 9, 23, text; first16: the same four; full replies on the common first16 mask: state 17, text) = 30. Shared draws: 4,000, seed 2026111312. Map gate: 1,000 draws, pod A's seed.
- Packages on Modal (the local test versions): numpy 1.26.4, scipy 1.13.1, scikit-learn 1.9.0, pandas 3.0.5, torch 2.2.2 (CPU), transformers 4.46.3, tokenizers 0.20.3. Tokenizers for the first16 text: each stage's repo at its pinned revision, offline from the weights volume.
- Outputs: volume `persona-gpu-runs-261112`, `analysis_v12/`; small files copied to `results/analysis_v12/`.

| Step | Where | State |
|---|---|---|
| Power check: prep, 30 sources, P1-P3 | Modal CPU, app `ap-mC1SpHYI8mSeuNaaqGZSW9`, log `analysis_v12_power.log` | Done, passed. Prep: all inputs match their receipts; row keys equal in the three stages and keys.json; first16 finite rows equal the eligible rows (valid, at least 16 kept tokens) in each stage; common first16 mask 19,100 of 21,000 rows; support passed (at least 71 rows per cloud and wording). P1 and P2 pass for state 17 in both views at every stage (planted 0.5/1/2 role SD read back as 0.502-0.505/1.004-1.010/2.007-2.020). Text first16 reads planted shifts 11-12% too large (P1 fails; report only). `results/analysis_v12/prep.json`, `power.json` |
| Measurement | Modal CPU, app `ap-AodMCJLiWuMdVG83k2tp6a`, log `analysis_v12_measure.log` | Done. `results/analysis_v12/results_v12.json`; per-source files in `results/analysis_v12/sources/` |
| HF copy | Modal, `../261112_modal_gpu/hf_upload.py --batch 261113_analysis` | Revision `8c41d5d1b88b7fa0fed38ffeeb89863a6798c973`, 69 files, 281.7 MB, SHA-256 and size of all 69 match the volume. Receipt `../261112_modal_gpu/hf_receipt_261113_analysis.json`. Dataset folder `experiments/q1_emergence/261112_stage_own_replies/analysis_v12/`. The 33 local JSON copies match the receipt |

## 5. Results with intervals (state 17; `results/analysis_v12/results_v12.json`)

Position in role SD (signed: above 0 is the side the Instruct `assistant` role leans to). Spread: log(default variance / median role variance), within wording. Brackets: 99.5% intervals, 4,000 crossed draws. Map gate: raw Spearman r (95% lower bound); all pass (floor 0.5).

| Stage | View | Position | Spread | Map gate |
|---|---|---|---|---|
| SFT | full | 0.901 [0.237, 1.338] | 0.089 [0.002, 0.167] | 0.884 (0.782) |
| SFT | first16 | 1.107 [-0.004, 1.419] | 0.090 [0.039, 0.185] | 0.879 (0.774) |
| DPO | full | 1.073 [-0.337, 1.510] | 0.176 [0.066, 0.278] | 0.896 (0.788) |
| DPO | first16 | 1.293 [-0.142, 1.660] | 0.188 [0.080, 0.291] | 0.871 (0.745) |
| Instruct | full | 1.073 [-0.424, 1.517] | 0.179 [0.074, 0.310] | 0.901 (0.801) |
| Instruct | first16 | 1.308 [-0.199, 1.684] | 0.194 [0.086, 0.304] | 0.877 (0.759) |

Profile correlation with the Instruct profile: 0.935-1.000 (all above the 0.5 floor). No draw dropped.

The ten tests:

| # | Test | Reading (both views must agree) | Numbers |
|---|---|---|---|
| 1-3 | Position, SFT / DPO / Instruct | Not beyond 2 role SD, at every stage, in both views (the frozen rule's reading "not at one end"; see section 6) | Every 99.5% upper bound is under 2 role SD (highest 1.684) |
| 4-6 | Spread, SFT / DPO / Instruct | Wider, at every stage, in both views | Every 99.5% interval is above 0 (table above) |
| 7 | Position change, SFT to DPO | Interval includes 0 | full 0.171 [-1.093, 0.570]; first16 0.186 [-0.733, 0.793]. P3: 1.5 role SD is the smallest tested change that reaches 80% simulated rejection (3.4% at 1.0, 94.7% at 1.5; sizes between were not tested) |
| 8 | Position change, DPO to Instruct | Interval includes 0 | full 0.001 [-0.218, 0.101]; first16 0.014 [-0.437, 0.091]. P3: 0.5 role SD reaches 80% |
| 9 | Spread change, SFT to DPO | Length-dependent, no claim | full 0.087 [-0.049, 0.198]; first16 0.098 [0.012, 0.179] |
| 10 | Spread change, DPO to Instruct | Interval includes 0 | full 0.003 [-0.020, 0.060]; first16 0.006 [-0.015, 0.036]. P3: 0.10 reaches 80% |

Reported without a claim:

| Item | Result |
|---|---|
| Text baseline, full | Position SFT 0.950, DPO 1.006, Instruct 1.021; spread 0.256, 0.578, 0.602 (`sources/*_full_text.json`) |
| Text baseline, first16 | Position 0.799, 1.071, 1.101; spread 0.339, 0.559, 0.592. P1 overshoot 11-12% |
| State minus text (95%) | Position: every interval includes 0. Spread: the state is less wide than the text in every stage and view, e.g. Instruct full -0.423 [-0.505, -0.168], SFT full -0.167 [-0.307, -0.037] (`state_minus_text`) |
| Full replies on the common first16 mask | Position SFT 1.075, DPO 1.110, Instruct 1.113. Spread SFT 0.033 [-0.031, 0.077] (99.5%), DPO 0.175, Instruct 0.193. So the SFT full-view spread depends on the short replies that the mask drops |
| Lu's 0-1 scale, state 17, first run | Distance to the nearer end, full: SFT 0.110 [0.082, 0.143], DPO 0.041 [0.025, 0.058], Instruct 0.040 [0.023, 0.059] (95%, question resamples); first16 0.096 [0.071, 0.120], 0.053 [0.035, 0.081], 0.051 [0.032, 0.080]. Always the assistant end (scaled 0.89-0.96), never outside the role range. Text full: 0.133, 0.081, 0.083. Not a paired test. In the first run the first16 values weighted questions equally, while the primary PC1 weights rows equally (Codex finding 2); the full view is balanced, so it is not affected. Section 8 has the corrected first16 values |
| Han's group test, role-SD units | Gap near 0 in every pair and view (p 0.63-1.00, exact over 252 or 70 splits) |
| Han's group test, raw units | Positive gaps (e.g. first16 SFT to DPO 0.858, p 0.024; full 0.157, p 0.317). They vanish in role-SD units (first16 -0.042, p 0.81). This is consistent with a change in overall scale; the run does not test that explanation |
| States 9 and 23 | Positions 0.716-1.107; spreads 0.067-0.226, all above 0 at 99.5% except SFT full state 9 |
| PC1 share, split-half PC1 cosine | State 17: share 0.29-0.36; cosine 0.96-0.98 |
| DPO vs Instruct | Almost identical full-view PC1 role profiles: r 0.9995 over the 20 role scores. PC1 holds about a third of the role variance, and the full role maps were not compared across stages. The inputs differ (separate hashes; different replies in my read) |

What this does not show:
- Power for a position change between SFT and DPO is low with 20 roles: 3.4% simulated rejection at 1.0 role SD, 94.7% at 1.5 (P3; the rate is the lower of the two directions).
- SFT plays fewer roles than DPO and Instruct (my read: 13 of 20 yes, against 16 and 17). Positions are relative to the roles each stage actually wrote.
- Agent analysis. Codex reviewed `analysis_v12.py` after the run (section 7). No human review.

## 6. Correction after the run: "not at one end" is too strong

The frozen rule calls the default "at one end" only beyond 2 role SD. That tests "outside the role range", not "near the end": the role scores on PC1 are lopsided, so the assistant end is about 1.3 role SD from the role mean. The v5 draft session (bc1e4a44) found the same flaw on pod C.

On Lu's 0-1 scale (reported without a claim, section 5) the default is 0.110 from the assistant end at SFT and 0.041 / 0.040 at DPO / Instruct (Lu: 0.03). Han's reading rule (`../261112_han_reanalysis/THRESHOLDS.md` line 26: extreme if the upper 95% bound is at most 0.05) is not met at any stage in the first run: upper bounds 0.143, 0.058, 0.059 (full); 0.120, 0.081, 0.080 (first16).

So the default is inside the role range and close to the assistant end at DPO and Instruct, and further from it at SFT. The SFT-to-DPO move on this scale has no paired test in the rules; no claim.

## 7. Codex review of the analysis (13 Nov, after the run)

I asked for it ("Yes, review it"). Prompt `review/prompt_analysis_v12_261113.md`, review `review/codex_analysis_v12_261113.md` (read-only sandbox, reasoning xhigh; log beside it). No file changed during the review (hashes equal before and after). Verdict: "fix first"; "I did not establish corruption of the primary saved numbers". No finding on the statistics, the shift and scale algebra, row keys, the common mask, the first16 token span, the map gate, Han's labels and permutation, the test family, or NOTE's primary tables. Ten findings, each checked by Claude against the code:

| # | Severity | Finding | Claude's check | Effect on these results |
|---|---|---|---|---|
| 1 | Medium | The full view uses SFT's `valid` flags for every stage (`meta` is built from the SFT rows) | Confirmed | None: all 21,000 rows are valid at each stage (`prep.json`) |
| 2 | Medium | Lu values on masked views weight questions equally; the primary PC1 weights rows equally | Confirmed | Full-view Lu values not affected (balanced). First16 Lu values are on a slightly different axis; now stated in the table |
| 3 | Medium | "Almost the same role map" rests on 20 PC1 scores only | Confirmed | Wording fixed above |
| 4 | Medium | "Under 1.5 role SD cannot be seen" turns a coarse grid into a hard limit; the vault left out two qualifications | Confirmed | Wording fixed here and in the vault |
| 5 | Low | The source files hold real estimates before the power step; `power.json` holds absolute responses that depend on the real sign | Confirmed | Claude printed only the pass flags, the signed responses and the centred P3 rates before the measurement. The files do not enforce that |
| 6 | Low | P3 reports the lower of the two directions' rates; the rule does not say so | Confirmed | Conservative. Now stated |
| 7 | Low | Resume skips a source when its JSON exists; arrays are not hashed into a run fingerprint | Confirmed | None seen: one run wrote all 30 sources (`analysis_v12_power.log`); all 30 arrays on the volume match the HF receipt |
| 8 | Low | State-minus-text is not gated; `interval()` drops non-finite draws without a count | Confirmed | None: all gates pass, and the 30 draw arrays hold 0 non-finite values (checked on a copy from the volume) |
| 9 | Low | Han's per-role change is saved for the labelled roles only; "follows a change in scale" is not shown | Confirmed | All 20 raw projections are saved (`projection_by_stage`). Wording fixed |
| 10 | Low | The end test accepts exactly 2.0 (`>=`); the rule says "beyond" | Confirmed | None: no value is at 2.0 |

The first run's code is kept in `review/code_before_fixes_261113/` (`.txt` copies; hashes equal the first run's recorded code hashes).

## 8. Re-run with the code fixes: revision b (13 Nov)

My decision: "Fix and re-run". Same rules (THRESHOLDS v1.2), same seed and draws. Code changes for the Codex findings: each stage uses its own valid flags (1); Lu's scale uses the same row-weighted cloud means as the position statistic (2); `power.json` holds no absolute response (5); P3 gives the rate for a planted increase and a planted decrease, and applies the reading rule as written (6); every source carries the run fingerprint and its array hash, arrays are written atomically, and a source is reused only if both match (7); non-finite draws are counted into the dropped share, and report-only comparisons carry their gates and paired draw counts (8); Han's change is saved for all 20 roles (9); the end test is strict (10). `test_analysis_v12.py`: 14 pass (adds: per-stage valid flags, Lu equals check 3 on balanced rows, non-finite count, tampered array and artifact are refused, strict end test).

| Step | Where | State |
|---|---|---|
| Power check: prep, 30 sources, P1-P3 | Modal CPU, app `ap-mt0zzGv6Wd2BLHJH6FYXx4`, log `analysis_v12b_power.log` | Done, passed. 21,000 valid rows at each stage; common first16 mask 19,100; support passed. Fingerprint `0662668e6c33…`. `results/analysis_v12b/prep.json`, `power.json` |
| Measurement | Modal CPU, app `ap-1KhW7cjdcxuY02CQrDGO8s`, log `analysis_v12b_measure.log` | Done. `results/analysis_v12b/results_v12.json` |
| HF copy | `../261112_modal_gpu/hf_upload.py --batch 261113_analysis_b` | Revision `a9e9a54c7125771825b226e0ff5d96e8135d88f9`, 70 files, 281.4 MB, all 70 match the volume. Receipt `../261112_modal_gpu/hf_receipt_261113_analysis_b.json`. Dataset folder `.../261112_stage_own_replies/analysis_v12b/`. The 33 local JSON copies match the receipt |

Volume folder `analysis_v12b/`. The first run stays in `analysis_v12/` (HF `8c41d5d1…`).

Revision b against the first run:

- The 10 confirmatory readings are the same.
- Of 1,455 shared result fields (outside P3 and the hashes), 74 differ. 72 are Lu values on the masked views, as intended by fix 2. The other 2 are the map-gate interval ends of Instruct first16, which differ by 0.0000015 (numerical noise; the point r is equal).
- Every position, spread, interval, stage change, text baseline and Han number in section 5 is unchanged. No draw is dropped or non-finite (4,000 of 4,000 in all 30 sources). State-minus-text: all gates pass.

Lu's 0-1 scale, state 17, revision b (distance to the assistant end, 95% interval; `results/analysis_v12b/results_v12.json`, `report_only.sources.*.lu`). These replace the first-run values for the masked views in section 5:

| Stage | Full replies | First 16 tokens | Full replies on the common mask |
|---|---|---|---|
| SFT | 0.110 [0.082, 0.143] | 0.086 [0.063, 0.110] | 0.079 [0.057, 0.103] |
| DPO | 0.041 [0.025, 0.058] | 0.045 [0.026, 0.072] | 0.031 [0.017, 0.048] |
| Instruct | 0.040 [0.023, 0.059] | 0.043 [0.024, 0.071] | 0.030 [0.015, 0.049] |

Han's reading rule (upper 95% bound at most 0.05) is not met in the two main views at any stage. It is met for DPO and Instruct on full replies restricted to the common mask (0.048, 0.049). That rule is not in THRESHOLDS v1.2, so this is no claim.

P3 by direction (simulated rejection rate; `results/analysis_v12b/power.json`). The position power is lopsided, because the difference draws are skewed:

| Change | Planted increase | Planted decrease |
|---|---|---|
| Position, SFT to DPO, 0.75 role SD | 0.2% | 91.3% |
| Position, SFT to DPO, 1.0 | 3.4% | 98.8% |
| Position, SFT to DPO, 1.5 | 94.7% | 99.9% |
| Position, DPO to Instruct, 0.25 | 0.0% | 99.8% |
| Position, DPO to Instruct, 0.5 | 97.0% | 99.9% |
| Spread, SFT to DPO, 0.15 / 0.20 | 71.4% / 95.4% | 73.2% / 95.9% |
| Spread, DPO to Instruct, 0.05 / 0.10 | 83.4% / 100% | 76.3% / 100% |

So a move of the default toward the assistant side between SFT and DPO is the direction this test is weak for.

Not reviewed again: revision b has had no Codex or human review.
