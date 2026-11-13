# Reply length control for the stage replies: rules (v1.0, 13/11/2026)

Written before any run. Owner, 13 Nov: "Ok, lets run these length controls to make sure eveyhtring is in order". De-risk mode: one forward-pass job, est. $1–2, and no new replies.

## Question

Each stage now writes its own replies (`../261112_stage_own_replies/`). The replies differ in length by stage: SFT median 45 kept tokens; DPO and Instruct median 128. A reply mean is taken over all kept tokens, so a difference between stages can come from length alone. This run makes states over the same number of tokens for every reply.

## Design

- Stages and inputs: Instruct (pod A `run_full`, HF `brend0nc/manifold-persona-experiment-artifacts` @ `60a81b9b…`), SFT and DPO (`own_replies/{sft,dpo}_full` on the Modal volume `persona-gpu-runs-261112`; HF @ `f224ced8…`). Each stage reads its own replies with its own model and pinned revision (`../261112_coinflip_stages/stages.json`).
- Input check: `rows.jsonl` and `reply_mean_acts.npy` must match the run's `SHA256SUMS.txt` (Instruct: the hashes in pod A's HF receipt).
- Frame: the stored `prompt_ids + kept_ids` of each row, unchanged (pod A's chat frame, with the role system prompt). No re-tokenization.
- Pooling: pod A's rule (`rows.pool_spans`: one forward pass, right padding, mean over a span). Spans, in the same pass:
  - full: all kept tokens (only for the check below);
  - first16: the first 16 kept tokens;
  - first32: the first 32 kept tokens (pod C's F4 length).
  A row with fewer kept tokens than the cut gets a NaN state for that cut. It is not padded or shortened.
- States saved: 0 (embeddings, text baseline), 9, 17 (primary), 23. Float16, rows in `rows.jsonl` order.
- Model: bfloat16, sdpa attention, batch size 32, rows sorted by length inside the job.

## Check before any use (pod A's C0 gate, `rows.C0_GATE`)

- For every valid row, the re-read full mean is compared with the stored reply mean at states 9, 17 and 23, both centred on the stored mean: minimum cosine at least 0.95, median cosine at least 0.99, median relative norm difference at most 0.02.
- A stage that fails is not used. The job exits with an error and its outputs are marked failed.

## Support rule (pod C's F4 rule, `../261029_fixed_text_read_through/THRESHOLDS.md`)

- A cut is available for a stage only if every role keeps at least half of its rows.
- Expected from the rows, counted before the run: first16 is available for all three stages (lowest: SFT toddler 0.58). first32 is not available for SFT (toddler 0.19, cynic 0.34). Instruct and DPO pass first32 (lowest: toddler 0.80 and 0.84).

## What this run does not do

- It computes no metric on the replies. The comparison (each planned measure on full replies against first16, on the same rows) is set in the own-reply analysis rules, after the owner's raw read.

## Compute

- Smoke: 64 rows per stage, all three stages, then the check on those rows.
- Full: 21,000 rows per stage, three A100-80GB in parallel, hard stop 1 hour each. Est. 10–15 minutes per stage.
- Outputs: about 1.4 GB per stage (two cuts, four states) on the Modal volume under `length_control/`, then HF with a verified receipt.

## v1.1 (13/11/2026, after the run): wording only

Owner, 13 Nov: "Yes, fix them" (Codex review, finding 3). No rule, number or result changes.

- The support rule above uses only the role-share part of pod C's F4 rule. Read "available" as a role-coverage check.
- Pod C's F4 rule also needs common-question support in each fixed half and in each bootstrap draw. This run does not check those parts.
- An analysis that uses a cut must enforce half and draw support itself.
