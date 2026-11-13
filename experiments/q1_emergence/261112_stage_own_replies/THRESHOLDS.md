# SFT and DPO write their own replies on pod A's grid: rules (v1.0, 12/11/2026)

Written before any run. This is the planned rerun (task T6, T-056) after the base and example-turn fixes. Owner, 12 Nov: "Why aren't we doing it. Isn't this just the re ran we wanted to do after we fixed base ?" This is the owner override of T6's gate. The gate said "the default beats the text floor on saved states", and the free checks of 12 Nov did not show that.

## Question

Do the role maps and the default change when each trained stage writes its own replies (Q3b)? Later, with Han Xiao's labels, this also gives his test on OLMo-2.

## Design (pod A, unchanged)

- Code: pod A's `gen_extract.py` and `gates.py`, unchanged, run through `stage_run.py`. The wrapper changes only the model and the pinned revision: SFT `allenai/OLMo-2-1124-7B-SFT` @ `1de02c01…`, DPO `allenai/OLMo-2-1124-7B-DPO` @ `e34ea60a…` (`../261112_coinflip_stages/stages.json`, from pod C).
- Grid: 20 E3 pilot roles plus `default`, 5 wordings, 40 frozen questions, 5 replicates: 21,000 replies per stage. Chat template, no example turns, "{model_name}" is "OLMo".
- Sampling: temperature 0.5, repetition penalty 1.1, 128 new tokens. Each reply's seed is `item_seed(key)`, the same as pod A, so replies pair by key across SFT, DPO and Instruct.
- States: mean over the kept reply at all 33 hidden states, float16, as in pod A.
- Check before the run (done, `test_stage_run.py`, 2 pass): SFT and DPO render exactly pod A's stored prompt ids for all 21,000 keys.

## Gates per stage (pod A's thresholds, unchanged)

- G1 tokens, G2 valid rate (at least 95% valid replies for every role), G3 seeds, G4 reconstruction, G5 snapshot, and 20 random transcripts in `raw_read.md`.
- A stage that fails a gate is not analysed. The owner reads 20 transcripts per stage before any metric.

## Analysis

- No metric is computed on these replies until the analysis rules are written. They come after the Han Xiao check, which decides the measures and any added roles. Planned: pod A's D2 per stage; the default's position and spread; the projection on each stage's own Assistant axis for his labelled roles; text baseline; judge pass rate per stage.
- Pod A's 20 roles hold only 5 desirable and 5 undesirable roles by his labels, so a group test like his has low power on this grid. His extra roles can be added later as new roles on the same grid with the same code.
- The judge pass rate (about $34 for both stages, est.) needs a separate owner go.

## Compute

- Smoke first: canary roles, 40 items per stage, then the gates.
- Full: about 1.1 A100 hours per stage (pod A measured 4,053 s). SFT and DPO run in parallel on two A100-80GB, with a hard stop of 3 hours each.
- Outputs: about 11.4 GB per stage on the Modal volume `persona-gpu-runs-261112` (`own_replies/`): the final state array (5.68 GB) plus the batch parts it is built from. The parts are removed only after a verified HF copy and the owner's approval. HF upload uses the Modal secret `hf-write`.

## v1.1 changes (13/11/2026, after the Codex setup review, before the full run; no metric computed)

Codex review `review/codex_rerun_setup_261112.md` ("fix first"), shown to the owner, who said "Fix then show me". No change to the design, the grid, the sampling or the gate thresholds.

- Launcher: a failed generation, failed gate or failed volume commit now fails the job and the command. Gate results are read from `gates.json` (`gates[name].pass`). The full run refuses to start unless the same stage's new code test (`{stage}_smoke_v2`) passed all gates with the same code hashes and runtime.
- Wrapper: `--out` and `--run` become absolute paths. A receipt `<run folder>.stage_receipt.json`, next to the run folder (ordered hash of every key and its prompt ids, code hashes, runtime), is written at the first start; a resume with any difference is refused. Before a resume, the wrapper adds a missing newline after a complete last row and removes the rows of complete batches whose state part is missing or unreadable, so pod A's code generates them again.
- Storage: about 11.4 GB per stage, not 5.7.
- Worst case: both stages at the 3-hour stop is 6 A100 hours for one full attempt; `retries=0`, and a new attempt needs a new owner go.
- Wording: this is a matched comparison with pod A's Instruct run (same grid, seeds, sampling and pooling). It is not a replication of Han Xiao's design (different model, roles, questions, sampling, length and filter).

## v1.2 analysis rules (approved 13/11/2026; written before any metric on these replies)

Status: approved by the owner on 13/11/2026 ("Yes, I approve the rules"), after a 10-line plain summary in chat. Frozen: a later change needs a new version. Nothing below was computed at approval. The owner's transcript read is done (13/11, `NOTE.md`). Changed after the Codex length-control review (`../261113_reply_length_control/review/codex_length_control_261113.md`, design advice): one common first16 mask across the stages, and a support check per wording and half.

### Question

At which stage does the default become privileged on its own replies? Two meanings, as in draft v5:
- Position: the default's replies sit at one end of the roles' main axis (PC1 of the role means, fitted on roles only).
- Spread: the default's replies vary less than the median role's (within-wording mean squared distance).

### Data

- Stages: SFT and DPO own replies (HF `f224ced8…`), Instruct pod A (HF `60a81b9b…`). All on the Modal volume `persona-gpu-runs-261112`.
- Two views, both primary:
  - full: `reply_mean_acts.npy`, mean over all kept reply tokens;
  - first16: `first16_acts.npy` from the length control (HF `08ae476f…`), mean over the first 16 kept tokens. One common mask: a key is used only if its reply has at least 16 kept tokens in all three stages, so the three stages compare the same keys (SFT toddler keeps 58% on its own; the common mask can keep fewer).
- first32 is not used: it fails the support rule for SFT.
- Hidden state 17 (pod A's primary). States 9 and 23: report only.
- Rows: all valid rows, 20 roles plus `default`, 5 wordings, 40 questions, 5 replicates.

### Input checks (before any statistic)

- SHA-256 of every input equals its HF receipt.
- The row keys are identical, in order, across the three stages' `rows.jsonl` and the length control's `keys.json`. The full key is checked (role, format, wording, question, replicate), not the role only.
- The NaN rows of first16 equal the counts in `support.json`.
- Support on the common first16 mask: every cloud (20 roles and `default`) keeps rows in every wording and in each half of the 50 fixed question splits. A failed cloud stops the first16 view.

### Statistics (from `../261112_free_checks/check_role_count.py`, imported, not rewritten)

- Position: PC1 of the centred role means on one question half; on the other half, (default projection minus mean role projection) / SD of the role projections. Unit: role SD (that stage's own).
- Signed, on a fixed orientation. Instruct's full-view PC1 is signed so that the `assistant` role projects above the role mean. Every other PC1 (stage, view, text) is signed so that its 20 role projections correlate positively with that Instruct profile. Each split or draw is signed to its stage's full-data PC1. The absolute value is reported beside it for continuity with checks 1 and 2.
- Spread: within-wording variance of each cloud; log(default / median role). Above 0: the default cloud is wider.
- Point estimate: mean over 50 fixed question splits (position); all questions (spread).
- Uncertainty: 4,000 crossed draws (roles resampled with replacement, random question halves, questions resampled within halves). The same draws serve every stage, view and the text baseline, so all differences are paired.

### Text baseline

- The same statistics on text features of the same rows: character 3-5-gram TF-IDF, 256-d truncated SVD, fitted once on the replies of all three stages together.
- first16 view: the text of the first 16 kept tokens (decoded from `kept_ids`).
- The state result adds evidence beyond the text only where the paired difference (state minus text, same draws) has a 95% interval that excludes 0. Report only.

### Gates before reading a stage

- D2 (pod A's map rule): role-role cosine distance maps from two question halves; raw Spearman r, mean over 50 halves; 95% interval from 1,000 question-half draws. A stage and view with a lower bound under 0.5 is not read.
- Orientation: a signed value is read only where its profile correlation with Instruct is at least 0.5. Otherwise only the absolute value is reported.
- Draws: a draw where a cloud has no rows in a half is dropped and counted. More than 1% dropped: that view is not read.

### Power check (runs first; prints no real estimate)

- P1, position response: add a shift of 0.5, 1 and 2 role SDs along each stage's signed PC1 to every default row. Pass: the signed point moves by the planted amount within 10%, in every stage and view. The change in the absolute value is recorded too.
- P2, spread response: scale each default row's distance to its cloud-and-wording mean so that its variance is x0.67 and x1.5. Pass: the log-ratio moves by log(0.67) and log(1.5) within 0.01; the P1 shift moves the spread by under 0.01.
- P3, rejection rate: from the paired difference draws, centred on their median (so the real difference is hidden), the rate at which a planted true change is called a change under the rule below. Position: 0.25, 0.5, 0.75, 1.0, 1.5 role SDs. Spread: 0.05, 0.10, 0.15, 0.20, 0.30 log units. Output: the smallest change with at least 80% rate, per stage pair. This is a bootstrap simulation, not a new sample of replies.
- A failed P1 or P2 stops the run before any real number.

### What counts (the confirmatory family: 10 tests)

Bonferroni over 10 tests: each interval is 99.5% (percentile, 4,000 draws). A reading holds only if it holds in both views (full and first16) with the same sign. If the views disagree, the result is called length-dependent, and no claim follows.

| # | Test | Reading |
|---|---|---|
| 1-3 | Position, per stage (SFT, DPO, Instruct) | "At one end": the signed point beyond 2 role SDs and the whole interval beyond 2 on the same side |
| 4-6 | Spread, per stage | "Tighter": the whole interval below 0. "Wider": the whole interval above 0 |
| 7-8 | Position change, SFT to DPO and DPO to Instruct | "Change": the paired difference's interval excludes 0. Size in role SDs |
| 9-10 | Spread change, the same two pairs | "Change": the paired difference's interval excludes 0. Size in log units |

- No "no change" claim. An interval that includes 0 is reported as "no change seen; the data rule out changes beyond the interval ends", with the P3 rate beside it.

### Report only (no claim)

- SFT to Instruct changes; absolute position; states 9 and 23; 95% intervals.
- Full replies on the common first16 mask (shows the effect of the row loss apart from the cut).
- Text baseline, both views, and state minus text.
- Han Xiao's group test: each stage's own Assistant axis (his `axis_projections`: default minus mean role); projection change per role, SFT to DPO and DPO to Instruct; desirable minus undesirable mean change (5 vs 5 by his labels: assistant, translator, counselor, consultant, stoic vs criminal, cynic, narcissist, pirate, vampire); raw and role-SD units; exact permutation p over all 252 label splits; with and without his borderline roles (stoic, cynic). Both views.
- PC1 share of role variance and split-half PC1 cosine, per stage and view.
- Lu's 0-1 scale on the roles-only PC1 (full data): (default minus lowest role) / (highest minus lowest role), and the default's distance to the nearer end; for comparison with Lu (0.03) and the pod C audit (`../261029_stage_persona_plan/slides_v5_261113/results_audit_261113.md`).

### Not in this run

- The judge pass rate per stage (about $34): a separate owner go.

### Compute

- Modal CPU on the volume `persona-gpu-runs-261112`; no GPU. Est. under $1. Two jobs: the power check (P1-P3), then the measurement.
- Outputs: `results/` here (small JSON and the draw arrays), then HF with a verified receipt.
