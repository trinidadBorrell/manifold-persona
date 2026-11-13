# Persona Geology across OLMo-2 stages (31/10/2026)

This is the note for `261031_persona_geology_stages`. It records a design and its preparation. No experiment has run. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | Does the trait-adapter map of the Persona Geology post exist in OLMo-2 base, SFT, DPO and Instruct, beyond the structure that the preference text predicts? |
| Result | No result yet. No OLMo-2 training or judging has run. The preparation is done: the original preference data is fetched and verified, and the local code passes its 51 CPU tests. |
| How much I trust it | The data matches the publisher's SHA-256 and size manifest. The tests ran on a tiny random model, with no OLMo-2 weights and no GPU. So OLMo-2 loading, GPU memory and training time are unverified. No independent reviewer read the code. |
| What I do not know | Whether the map exists at any stage. Whether thirteen full batches replay the original trainer exactly. Feedback from me and my collaborators on this design is pending. |
| Status (31/10/2026) | Stopped at my GPU gate (BRIEF.md section 5). The next step needs a fresh pod SSH address from me, and my approval of the pilot budget and schedule. |

---

## What I did

1. I asked for this test on 31/10/2026, in [BRIEF.md](BRIEF.md). The mode is de-risk: a small trait zoo at one stage before a costly sweep.
2. An agent read the Persona Geology post, its wiki and the source trainer.
3. The complete preference data was fetched and checked: 134 files with 59,630 pairs.
4. The design was written: 20 traits, four training arms, a text control and activation shifts.
5. The local code was built and tested on CPU. 51 tests pass.
6. The work stopped before any GPU use, as my brief says.

---

## The planned first test

| Item | Plan |
|---|---|
| Model | OLMo-2-1124-7B-Instruct only. |
| Traits | Twenty Goldberg markers: two markers per pole per factor. |
| Adapters | One LoRA adapter per trait, trained on that trait's 445 preference pairs for 13 optimizer steps. |
| Real zoo | Tests coherent preference contrasts. |
| Scrambled zoo | Swaps the labels of 222 of the 445 pairs per trait. It shows what structure remains without a coherent contrast. |
| Permuted zoo | Gives intact trait datasets wrong names. Recovered structure must follow the data labels. |
| Second real zoo | Uses another initialization seed. It tests whether the map is stable. |
| Text control | Predicts the adapter similarities from text embeddings of the same pairs. |
| Proposed cap | At most 80 adapters and four GPU hours. This is a proposal, not approved spend. |

- The expected result: coherent pairs may give more structure than scrambled pairs, and the text may explain most of it. Either result is useful.
- The 20-trait panel is a change from the post's 134-trait zoo. It cannot establish five stable named factors.
- Base gets plain text, and the trained stages get their chat template. This format difference limits what a later stage difference can say about the weights alone.

---

## How much I trust the preparation

- Every data file matches the publisher's SHA-256 and size manifest. Every trait has the same 445 unique prompts in the same order.
- These checks validate the data access and the bytes. They do not validate the post's numerical or behavioral claims.
- The final CPU suite passed (51 tests in 12.02 s). The training test uses a tiny random GPTNeoX model.
- Two implementation workers wrote separate files. Neither was an independent reviewer.
- The trainer is an explicit adaptation, not an exact replay: thirteen full batches of 32, with 416 rows used and 29 unused.
- Local Transformers 4.46.3 does not support OLMo-2. So nothing here has loaded the real model.

---

## Open points

- The GPU gate is mine. I must supply a fresh pod SSH address and approve the pilot budget and the schedule. The address 213.173.102.5 must not be used.
- I can review the choice of thirteen full batches at the GPU gate.
- Feedback from me and my collaborators on the design is pending.
- Before a sweep, the design needs a common plain-text sensitivity on all stages, or a narrower inference.
- A sweep needs a separate go after the one-stage result.

---

## Words used here

| Word | Meaning |
|---|---|
| Persona Geology | A blog post that trains one small adapter per personality trait on a Qwen model and maps how the adapters relate. |
| Trait zoo | The set of trait adapters. The post has 134 traits. My pilot has 20. |
| Adapter, LoRA | A small set of extra weights (low-rank matrices A and B) trained on top of a frozen model. |
| Trait-adapter map | The matrix of similarities between the trait adapters. |
| Goldberg markers | Trait words from Goldberg's Big Five list. Each belongs to one factor and one pole (positive or negative). |
| Preference pairs | For each prompt, a chosen answer and a rejected answer. The rejected answer portrays the opposite pole. |
| DPO | Direct preference optimization: training that makes the chosen answer more likely than the rejected one. |
| Stage | One OLMo-2-7B checkpoint: base, SFT, DPO or Instruct. |
| Arm | One training condition: real, scrambled, permuted, or second seed. |
| Derangement | A shuffle in which no trait keeps its own dataset. |
| Text control | A prediction of the adapter similarities from text embeddings of the pairs alone. |
| Activation shift | The change in the model's hidden state on fixed inputs when an adapter is added. |
| Gram matrix | The table of inner products between all adapter updates. Here it is normalized to cosines. |
| Tucker congruence | A measure of how well two factors match. |
| Parallel analysis | A rule that picks the number of factors by comparison with random data. |
| GPU gate, my gates | Points where the work must stop for my decision: missing data, any paid API call, any GPU use, and the one-stage result before a sweep. |
| L4 | An item in the wiki's list of superseded claims. |
| T-090 | The id of this task in my task tracker. |
| HF | Hugging Face. My private dataset is `brend0nc/manifold-persona-experiment-artifacts`. |

---

# Run record

Everything below is the detailed record of the design and the preparation. I only need it to check a step, a file or a number.

My request: 31/10/2026, in [BRIEF.md](BRIEF.md). Mode: de-risk. This first test asks whether a small trait zoo works at one stage before a costly sweep. This note records design and preparation. No OLMo-2 training or judging has run.

## 1. The start table of the old note

| Item | Decision |
|---|---|
| Question | Does the trait-adapter map exist in OLMo-2 base, SFT, DPO and Instruct, beyond structure predicted by the preference text? |
| First test | OLMo-2-1124-7B-Instruct only. Twenty Goldberg markers, with two markers per pole per factor. |
| Expected result | Coherent pairs may give more structure than scrambled pairs. The text may explain most of it. Either result is useful. |
| Current milestone | The original preference corpora are fetched and verified. Build and check the local apparatus, then stop before GPU use. |
| My gates | Missing data; any paid API call; any GPU use; the one-stage result before a sweep. |
| Approval scope | The brief authorizes preparation and local CPU checks. It supplies no pod or paid-call approval. |

## 2. Context and sources read

The agent read the brief in full, AGENTS.md, the global AGENTS.md, `.claude/CLAUDE.md`, `.claude/skills/research-cycle/SKILL.md`, `research/PLAN.md`, the current vault week, the vault experiment index, and `research/ARTIFACT_STORAGE.md`.

- It read the previous fixed-text experiment's complete NOTE.md before choosing the test.
- That experiment's stage comparison has a wording-bias limit; it does not establish no stage change.
- The new test must measure the data contribution rather than repeat that inference.

Primary sources read on 31/10/2026:

| Source | Use and limit |
|---|---|
| [Persona Geology post](https://www.mindscapes.world/writing/persona-geology/) | Full post read, including appendices. Its results concern one Qwen model and its recipe. |
| [Wiki](https://www.mindscapes.world/persona/wiki/) | Agent-written provenance, not independent verification. Prefer the current post where they disagree. |
| [Recorded stage-one configuration](https://www.mindscapes.world/persona/wiki/stage-one-training-config.html) | Supplies LR 5e-5, absent from the brief, and the recorded optimizer settings. |
| [Recipe](https://www.mindscapes.world/persona/wiki/zoo-training-recipe.html) | Stage-one DPO, chosen SFT term, KL and shared initialization. |
| [Recipe divergences](https://www.mindscapes.world/persona/wiki/recipe-vs-source-papers.html) | No general-prompt mix; teacher text may determine the map. |
| [Superseded claims](https://www.mindscapes.world/persona/wiki/superseded-claims.html) | L4 withdraws a claim that the preference data is on HF. Verify the current release directly. |
| [Pair generation](https://www.mindscapes.world/persona/wiki/dpo-pair-generation.html) | Rejected replies portray the opposite pole. They are not unconditioned model replies. |
| [Shared prompt pool](https://www.mindscapes.world/persona/wiki/shared-prompt-pool-445.html) | Training used the common 445-prompt intersection, not the earlier 500 prompts. |
| [Primary markers](https://www.mindscapes.world/persona/wiki/goldberg-100-primary-traits.html) | Emotional Stability has 6 positive and 14 negative markers; do not assume 10/10. |
| [Released adapters](https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35) | Adapters do not replace preference rows. Stage-two transcripts are a different dataset. |

The shared ChatGPT link did not load through the web tool. The brief supplies its proposed extension; no unseen chat details are treated as evidence.

## 3. Checkpoints and input format

Reuse `../261029_fixed_text_read_through/checkpoints.json`, without changing its pins. The pilot uses the 7B family. No 1B substitution is proposed.

| Stage | HF model | Exact revision |
|---|---|---|
| base | `allenai/OLMo-2-1124-7B` | `7df9a82518afdecae4e8c026b27adccc8c1f0032` |
| SFT | `allenai/OLMo-2-1124-7B-SFT` | `1de02c0175118a9de5854aec80a1f970e701e928` |
| DPO | `allenai/OLMo-2-1124-7B-DPO` | `e34ea60adff2e575f4fe7569eaffd1b28509b6fd` |
| Instruct | `allenai/OLMo-2-1124-7B-Instruct` | `470b1fba1ae01581f270116362ee4aa1b97f4c84` |

All four use tokenizer.json SHA-256 `73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca`. Tokenization settings and chat-template hashes remain stage-specific and must be recorded.

- Base training prefix: `<|endoftext|>{prompt}\n\n`, followed directly by the answer. There is exactly one leading end-of-text token, no chat template, and no role markup.
- The loss mask excludes the prefix. The answer terminates with one end-of-text token.
- Reject raw text containing special-token strings instead of silently changing it.
- SFT, DPO and Instruct use their pinned tokenizer's own user/assistant chat template, with no system constitution and no few-shot turns. The constitution is teacher metadata.
- Save both chosen and rejected rendered strings, IDs and answer masks before training.
- Read saved examples for five traits at each of the four stages before any GPU request.
- The format difference is a limit on attributing any later stage difference to weights alone.
- Before a sweep, add a common plain-text sensitivity on all stages or narrow that inference.

## 4. Data and recipe

- Use the exact original `data_common/<trait>.jsonl` rows if released.
- Require 445 unique prompts per trait, identical prompt bytes and canonical order across traits.
- Require nonempty, distinct chosen/rejected answers and source metadata or a pinned release.
- Do not substitute the three public examples, stage-two transcripts, or generated OLMo answers.
- If full pairs are unavailable, stop before code implementation and ask for the original data or approval of a separately costed generation design.

The pilot panel is the first two alphabetically sorted words within each of the ten Goldberg factor/pole cells in the released marker metadata, case-insensitive.

- Freeze the resulting 20 names and their exact source rows before code tests or training.
- If metadata or a cell is missing, stop instead of guessing a word or pole.
- This smaller balanced panel is an explicit change from the post's 134-trait zoo.
- It cannot establish five stable named factors or reproduce the 34-word generalization test.

| Setting | Planned value |
|---|---|
| Trainable parameters | Plain LoRA, rank 64, alpha 128, dropout 0; all attention and MLP linear projections; exclude embeddings, norms and lm_head. Enumerate the actual OLMo targets. |
| Initialization | Save one shared A/B initialization per seed, with B zero. Reload its exact tensor bytes for each trait and control. |
| Seeds | Initialization 0 for primary and controls; initialization 1 for the second zoo; data order 0 in all arms. |
| Objective | DPO sigmoid beta 0.1, plus 0.1 chosen-answer NLL, plus KL coefficient 0.001 to the unadapted checkpoint. |
| Optimizer | AdamW, LR 5e-5, betas (0.9, 0.98), gradient clip 1.0, warmup 1 step. |
| Training length | Target the source's 13 optimizer steps, effective batch 32, max sequence length 1024. Do not silently truncate inputs. |
| Precision | Pod bfloat16 planned; CPU apparatus checks float32. Pin actual runtime versions before the pod run. |
| Reference | The exact unadapted stage checkpoint, frozen throughout each adapter call. |

The source repository is now public, contrary to an old wiki sentence. The agent read its trainer at Git revision `ba1d3f017091e54cfa47c2a80909c81952e44231`.

- Its KL term is the mean squared observed-token log-ratio, averaged over completion tokens per sample, then over both chosen and rejected samples. It is not full-distribution KL.
- The source uses cosine scheduling, one warmup step and AdamW's default weight decay 0.
- It sets `dataloader_drop_last=True` and computes the recorded step count as floor(445/32).
- That record alone does not show which remainder microbatches the original TRL version used.

The new trainer therefore makes an explicit adaptation: thirteen full batches of 32 after a fixed seed-0 permutation, using 416 rows and recording the 29 unused indices.

- All traits, seeds and stages use the same index schedule. No row is truncated.
- This retains the stated optimizer budget but is not a claim of exact original trainer replay.
- I can review this choice at the GPU gate before any empirical training.

## 5. Controls and measurements

| Arm | Purpose |
|---|---|
| Real zoo, 20 traits, seed 0 | Test coherent preference contrasts. |
| Scrambled zoo, same 20, seed 0 | Swap exactly 222 of 445 pair labels per trait using a saved trait-specific seed. Keep prompts, optimizer and initialization fixed. |
| Permuted zoo, same 20, seed 0 | Use a saved derangement of intact trait datasets under wrong names. Recovered structure must follow data labels. |
| Second real zoo, same 20, seed 1 | Test stability of map structure across initialization. Keep data order fixed. |
| Data embedding prediction | Mean chosen-minus-rejected text embeddings, same saved rows. Predict held-out adapter similarities; compare their residual structure. |
| No adapter | Exact zero activation shift, and the reference for each adapter's fixed-input shift. |

Map of the adapter updates:

- Compute exact Frobenius products of scaled B*A updates without constructing dense updates.
- Normalize to a cosine Gram matrix. Zero norms are errors, with affected traits reported.
- Factor each within-stage matrix with principal axis factoring and oblimin rotation.
- Choose factor count by the frozen parallel analysis in THRESHOLDS.md, not by trait names.
- Match factors by maximum absolute Tucker congruence with sign and permutation alignment.
- Never use cross-checkpoint raw update cosine as the main stage measure.

Text control:

- Use `sentence-transformers/all-MiniLM-L6-v2` for the text control, with its exact commit and tokenizer version frozen before embedding. No paid embedding service is authorized.
- Use balanced two-fold trait holdout: one word per pole per factor in each half.
- Fit ridge prediction of adapter similarities to training adapters from text contrasts.
- Choose regularization using training traits only; score held-out traits against training reference adapters, excluding self edges. Swap folds and save every prediction and residual.
- Report held-out R2, map correlation and residual spectrum, with their small-sample limits.
- Do not treat this as a replication of the post's 100-train/34-test prediction result.

Activation shifts and behavior:

- For activation shifts, freeze a separate bank of ordinary prompts and neutral continuations before training. Do not reuse preference answers as evaluation completions.
- Use identical plain-text token IDs, masks and positions across stages and adapters.
- Pool answer tokens at layer label 16, resolved through `io.hidden_state_index`, with labels 8 and 24 as secondary measures. Save the adapter-minus-reference shift for each fixed row.
- Compare within-stage shift similarity maps across stages and seeds. Cross-stage raw shift cosines are secondary because a shared architecture does not prove shared coordinates.
- Behavior checks retain raw generations, lengths and failures. A paid Big Five judge requires a separate prompt, calibration against labeled examples, cost estimate and my approval.

## 6. Budget and stop conditions

| Work | Ceiling before a new decision |
|---|---|
| Preparation | Local CPU checks and public source downloads only; at most 250 MB of source/data downloads before reporting. |
| Paid APIs now | $0. No data-generation, embedding or judge call. |
| GPU now | None. No pod address is available; never connect to 213.173.102.5. |
| Proposed one-stage pilot | At most 80 adapters / 1,040 optimizer steps, including all four arms. Begin with two traits and their scrambled controls for timing. |
| Proposed GPU cap | Four GPU hours total, including apparatus checks and extraction. This is a proposal, not approved spend. Revise the estimate from the bounded timing run. |
| Sweep | Separate approval after the one-stage pass. No stage sweep, RL-step sweep or new generation is implied by a pilot pass. |

- Stop for missing original preference data, failed hashes or controls, nonfinite losses, changed inputs/config, truncation, a broken resume fingerprint, or a budget forecast above the cap.
- Save errors and incomplete runs. Do not relax a pass rule to continue.
- The detailed decision rules are fixed in [THRESHOLDS.md](THRESHOLDS.md).

## 7. Artifacts and reproducibility

- Local small source/data artifacts: `output/261031_persona_geology_stages/preflight/`.
- Future pod artifacts: private dataset `brend0nc/manifold-persona-experiment-artifacts`, prefix `experiments/q1_emergence/261031_persona_geology_stages/`.
- Pod outputs must upload directly, with SHA256SUMS, an exact remote revision and a verified receipt.
- No local file removal is authorized.
- Each run must save the invocation, all package versions, model/tokenizer commits, data revision and hashes, initial adapter hashes, seed/order, resolved config, module list, exact rendered inputs, raw outputs, errors, optimizer checkpoints and resume fingerprint.
- Record `git rev-parse HEAD`, `git status --porcelain` and the diff hash before each run.
- Current preparation uses a dirty shared checkout; preserve all unrelated changes.

## 8. Work still required after the data check

- The complete release was fetched and validated before implementation.
- The apparatus and local checks are recorded below. The next action is my GPU gate.
- Feedback from me and my collaborators on this new design and stepping adaptation is pending.

## 9. Data release check (31/10/2026)

| Item | Verified evidence |
|---|---|
| Complete preference release | `EternalRecursion/persona-curvature-results` at `80b7bd1c42c09248ce44b26d993e92f25e854941`, paths `qwen35/data_common/*.jsonl`. |
| Download and checks | 134 files, 59,630 pairs, 74,914,477 bytes. Every file matches the publisher's SHA-256 and size manifest. |
| Common inputs | Every trait has the same 445 unique prompts in the same order. No empty or identical chosen/rejected pair. |
| Marker release | `EternalRecursion/persona-lora-zoo-qwen35` at `251bbff85409c9567a5fa17d0c968864b8497e5b`, `traits_primary.json`. |
| Manifest source | Public GitHub repository `EternalRecursion121/persona-curvature`, commit `ba1d3f017091e54cfa47c2a80909c81952e44231`, `tools/data_manifest.json`. |
| Receipt | `output/261031_persona_geology_stages/preflight/data_receipt.json` and `SHA256SUMS_data.txt`. |
| Conclusion | The missing-data gate does not apply. No data generation or paid call is needed for the pilot. |

- The superseded L4 release warning concerns an older adapter/transcript-only release.
- The current results dataset contains the complete pairs; a current file inventory and checksum check take precedence over the stale prose.
- These checks validate data access and bytes, not the post's numerical or behavioral claims.
- Preparation revision: `0576b9f03fff45f7069b4fc9be3d013fe645c4da` in a dirty shared checkout.
- Package versions, command evidence, Git status and diff hash are saved in `output/261031_persona_geology_stages/preflight/preparation_provenance.json` and nearby `git_*.txt`.

## 10. Local apparatus and GPU stop (31/10/2026)

| File | Implemented behavior |
|---|---|
| `render.py`, `panel.json`, `controls.json` | Fixed twenty-trait selection; exact stage formats, IDs and answer masks; saved scramble seeds and trait derangement. |
| `train.py` | One adapter per call; explicit shared initialization files; DPO, chosen NLL and squared-logratio terms; thirteen full batches; optimizer checkpoints and strict resume fingerprints. |
| `maps.py` | Exact low-rank Frobenius Gram; cosine maps; Gaussian parallel analysis; iterative principal axis factoring; oblimin; Tucker matching; controls and nested held-out text prediction. |
| `activation.py`, `activation_bank.json` | Twelve fixed neutral continuations; identical IDs across stages; answer-token shifts against the unadapted checkpoint; no-adapter zero check. |
| `text_control.py`, `embedding_checkpoint.json` | Original pair embeddings and contrast vectors; pinned MiniLM; no silent encoder truncation; saved per-side embeddings and provenance. Real encoding has not run. |

- The numerical analysis uses the uncentered cosine-normalized update Gram for the pilot. The source also reports centered solutions; this pilot choice is not a claim of exact reproduction of its centered five-factor solution.
- Singular or improper factor fits are inconclusive; no smoothing is applied.
- Text residual spectra cover observed cross-fold blocks only, without invented within-fold residuals.
- The implementing agent read the new production code and callers. Two implementation workers handled separate files; neither was an independent reviewer.
- The parent integration check found and resolved a tensor-only adapter loading mismatch, and the map reader now accepts activation NPZ files while checking trait order, layer and input hash.

| Check | Evidence |
|---|---|
| Final CPU suite | 51 passed in 12.02 s, exit 0. Production training uses a tiny random GPTNeoX model; no downloaded model weights or GPU. |
| Numerical checks | Dense versus low-rank products, planted factor/no-factor cases, improper-fit refusal, sign/permutation matching, held-out fitting and control recovery. |
| Training checks | Hand-calculated loss terms, microbatch equivalence, frozen reference, shared initialization, seed guards, checkpoint resume and rejected input/config changes. |
| Rendered corpus | 80 trait/stage files, 35,600 pairs. Maximum full input length 213 tokens against the 1,024-token limit. No truncation. |
| Raw read | Five traits at each of four stages, twenty saved examples. Base is plain text with one initial end-of-text token; trained stages use their pinned chat template with no examples. Answer masks exclude prefixes. |
| Controls | 80 prepared input files across real, scrambled, permuted and second-seed arms. Every saved file hash was rechecked. |
| Activation inputs | Twelve manually written prompts, disjoint from the training prompts; end-of-text tokens excluded from pooling. |
| Compilation and whitespace | Local compileall and `git diff --check` passed. |

- Final test evidence: `output/261031_persona_geology_stages/preflight/cpu_tests_final.txt`.
- The command, source hashes, saved-input checks and missing checks are in `preflight/verification.json`.
- The MiniLM model is pinned to `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its weights and the actual text embeddings were not fetched or computed in preparation.

Commands used from the repository root:

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download EternalRecursion/persona-curvature-results --repo-type dataset --revision 80b7bd1c42c09248ce44b26d993e92f25e854941 --include 'qwen35/data_common/*.jsonl' --include README.md --include 'qwen35/results/runmeta_sweep.json' --local-dir output/261031_persona_geology_stages/preflight/results_release --max-workers 4 --quiet
TOKENIZERS_PARALLELISM=false .venv/bin/python -u experiments/q1_emergence/261031_persona_geology_stages/render.py
PYTHONPATH=src TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest experiments/q1_emergence/261031_persona_geology_stages -q
```

- The first rendering ran before the script's final control/bank wrapper was added. Those helper calls then ran locally; the final script reproduces the whole preparation.
- The scope of tests is this experiment, not the legacy repository suite.
- No existing model output or source numerical result was substituted for the pilot.

Stop gate reached: BRIEF.md section 5, before any GPU use.

- No pod was contacted, no GPU was used, and no paid API call was made.
- Actual OLMo-2 loading, GPU memory, training time, full-zoo results, text prediction and behavioral evidence remain unverified.
- Local Transformers 4.46.3 does not support OLMo-2; the pod needs a supported, pinned runtime and the same CPU checks before GPU work.
- The proposed cap is four GPU hours, not a measured forecast or approved spend.
- The first approved GPU action would time two real traits and their scrambled partners.
- If the full eighty-adapter forecast exceeds that cap, stop and return with the estimate.
- I must supply a fresh pod SSH address and approve the pilot budget and the declared schedule. Do not use 213.173.102.5. A later sweep still needs a separate go.

## 11. Vault records and the ledger

- Vault records: `Weeks/2026-W44.md` (session 01a18f0b), `Reference/Experiment index.md`, `Reference/Task tracker.md` (T-090) and `Reference/Cost log.md`; each linted with zero errors.
- There is no scientific result to add to `Reference/Claims and verdicts.md`.
- The ledger cannot identify this experiment. Its fuzzy older CPU/GPU matches were not treated as evidence, and none of its unrelated task offers was ticked.
