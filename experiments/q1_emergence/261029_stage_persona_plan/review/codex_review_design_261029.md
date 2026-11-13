**Verdict:** Do not buy the full pod runs yet: 13 blockers can produce misleading results or unusable outputs.

---

**Blockers**

**B1. Pod C removes direct description leakage, but does not isolate persona structure.**

Evidence: `DESIGN.md:64–68`; `261008_base_persona_map/THRESHOLDS.md:126–140`; `261008_base_persona_map/NOTE.md:98`.
Paths starting with `261008_base_persona_map/` refer to the sibling experiment folder.
The proposed main frame is E3 C2: it still contains the question and interview frame.
It supports a narrow claim about reading fixed, Instruct-written replies in that frame.

A failing case: pirates use nautical words, chefs discuss food, and default replies mention AI.
Even an embedding average can recover those labels and their map.
Holding ids fixed removes changing input content across checkpoints; it does not identify the type of structure being measured.
Nor does a decoder beating TF-IDF prove information beyond the text: activations are functions of that text.

Keep C2, but add question-only/neutral-reply controls, a literal reply-only sensitivity, and matched length/position sensitivities.
Compare activation maps with word/character n-grams, sentence embeddings, and hidden state 0 on the same rows.
Add a topic-matched role contrast and a role-neutral text control before claiming persona specificity.
Use a second text source, such as existing base replies, if claiming a model-general map rather than this Instruct corpus.

---

**B2. The calibration can certify shared text artifacts and mishandle genuine model changes.**

Evidence: `DESIGN.md:21,67–68`; `261008_base_persona_map/part2_analysis.py:320–390,477`.
“Shuffled role labels” does not specify what moves, or whether the same shuffle reaches both maps.
A common role permutation preserves split-half map correlation exactly.
An offline check gave 0.9856464897 before and after that shuffle.

Another failing case: identical random feature maps process the same replies at two checkpoints.
Same-row noise and word content can inflate cross-checkpoint agreement without a learned persona representation.
Compare checkpoint A on one question/wording half against checkpoint B on the other half, then reverse.
Keep each reply's seeds together, and use paired resampling across checkpoints.

Define separate nulls for label decoding, map correspondence, and model-specific structure.
For map correspondence, independently permute one map's role identities; preserve repeated-measure blocks.
Use existing E3 within-question block permutations for the different question of repeatable role structure.
Add synthetic identical, independent, rotated, and scaled representations with known expected results.

Spearman–Brown estimates full-sample reliability from a split-half correlation; it is not itself cross-checkpoint disattenuation.
Specify both checkpoints' reliabilities and the formula, such as observed correlation divided by their geometric mean.
Report raw values and refuse corrected claims when reliability is inadequate or undefined.
Raw axis cosine also requires a coordinate assumption: an orthogonal rotation preserves a map but can destroy axis cosine.

---

**B3. Axis repeatability and default projection can pass without an Assistant Axis.**

Evidence: `DESIGN.md:18,52,67`; `261008_base_persona_map/part2_analysis.py:395–412,488–495`; `refs/assistant-axis/roles/instructions/default.json:3–7`.
The E3 axis function splits questions but reuses every description wording in both halves.
Thus the design's “questions and wordings, as E3” claim is wrong for that function.
A repeated default-frame difference can produce a stable axis even when roles have no differences.

I passed role-identical vectors and a fixed default offset to that function.
Its split-half axis cosine was 1.0, above the proposed 0.80 threshold.
Split descriptions as well as questions, and test neutral/default frames with matched lengths and wording styles.
Validate the axis against independent role-expression or assistant behavior, or call it a default-versus-played-text contrast.

At the final checkpoint, default minus mean-role projection onto its own axis is positive by construction.
Specifically, `(default − role_mean) dot unit_axis = norm(default − role_mean)`.
The final-axis cosine with itself is also 1 by construction; neither result is independent validation.
Estimate the axis on disjoint data and specify projection origin, normalization, and held-out evaluation rows.

Pod C reads Instruct default replies; it cannot measure each checkpoint's own default behavior.
`research/PLAN.md:29–32,72` already makes this distinction.
If occupancy is intended, generate each stage's own default replies in a separate, validated protocol.
Otherwise label this measure “representation of fixed Instruct default text.”

---

**B4. Blind naming is not a validated “played role” filter.**

Evidence: `DESIGN.md:20,33,52,64,90`; `261008_base_persona_map/NOTE.md:95–97`; `261008_base_persona_map/THRESHOLDS.md:35`.
A reply saying “As a pirate, I cannot answer” can be easy to name without enacting the role.
Selecting easy-to-name replies can manufacture a cleaner lexical role map.
Reporting both subsets exposes selection sensitivity, but does not validate A3 or the selected subset.

Nameability, enactment, and answer quality need separate labels and rules.
Validate the actual filter on held-out owner labels covering Instruct replies and each proposed format.
Include self-naming, refusal, near-role, quoted-role, incoherent, and wrong-topic examples.
Freeze thresholds, judge version, candidate list, attrition limits, and treatment of roles with no passing replies.

Use one fixed row mask at every checkpoint and retain equal role/question weights where possible.
Report coverage and complete-cell sensitivity beside filtered maps.
Do not transfer a kappa result from broken-answer judging to role enactment or blind naming.
Until validation passes, call the subset “judge-nameable,” not “played.”

---

**B5. Pod B crosses role with format, but does not separate role from generated topic or length.**

Evidence: `DESIGN.md:57–60`; `output/independent_review_2026-10-02/review.txt:89–94`.
Shared questions balance input topics; roles can still generate different output topics.
A chef can discuss food in every format and pass held-out-format decoding with no stable role behavior.
A fixed filler controls prompt length, not role-dependent reply length, token positions, or stopping.

Choose the exact ten roles and all templates before coding.
Cross roles with controlled content/topics, including same-topic different-role and same-role different-topic examples.
Hold out question/topic and description wording as well as format during decoding.
Run separate role, format, topic, length, and stop-reason baselines on identical folds.

Define the multivariate variance decomposition, sums-of-squares denominator, interactions, and residual error explicitly.
Model question blocks, role-specific wording, and seed repeats; do not treat them as independent cell replicates.
Report role-by-question/topic effects as well as role-by-format effects.
An interaction alone establishes neither identity nor a nonlinear manifold.

The story and signed-letter formats add another problem: narration or a signature can name the role directly.
Define whether pooling covers quoted speech only or the complete document, and report that scope.
Inspect each format's raw outputs and freeze multiple stop markers, invalid-reply rules, and matched pooling spans.
C2/C3 must use each format's frame; the current C2 always uses an interview (`context_extract.py:44–45`).

---

**B6. Three generation seeds can change the questions, and Pod B seeds still depend on batching.**

Evidence: `DESIGN.md:48,58,73`; `src/manifold_persona/prompts_roles.py:49–62`; `261008_base_persona_map/gen_extract.py:188–198,255–260`.
The extractor's `--seed` selects questions; it does not seed generation.
Changing it from 0 to 1 changes the 40-question set, which I checked offline.
Adding generation seeding without separating these settings would confound sampling repeats with question changes.

Use a frozen question-id file and a separate generation-seed argument in both generators.
Hash an explicit stable item key, including role, format, wording, question, and replicate id.
Do not use Python's process-randomized `hash()`.
Verify that restart, batch-size, item-order, and subset changes preserve each item's seed and intended sampling behavior.

Seeding once per batch does not meet the proposed per-item rule.
Three calls with the same global seed can also repeat rather than provide independent samples.
If batch-independent sampling needs sequential generation, benchmark that cost before accepting the budget.
Keep all seeds of a prompt in the same training/test partition.

---

**B7. The uncertainty claim has no defined population or complete repeat structure.**

Evidence: `DESIGN.md:68,71–74`; `261008_base_persona_map/THRESHOLDS.md:13–18,119–121`; `261008_base_persona_map/part2_analysis.py:209–229`.
Three seeds are a reasonable pilot choice, but they do not establish adequate power for these new outcomes.
Fifty overlapping splits and 2,000 bootstrap draws do not create new observations.
The independent text units are role-by-question cells, with wording within role and generation repeats within prompts.

Use paired crossed resampling of question blocks and roles for matched model/format contrasts.
Keep wording and seed structure explicit; report between-seed estimates as well as pooled estimates.
The existing decoding bootstrap conditions on fitted models and pooled held-out predictions; it does not refit them.
Describe its conditional scope, or add a planned refit sensitivity for the primary contrast.

These roles are selected, not sampled from all personas; bootstrap intervals do not remove that selection.
There is one training lineage, so checkpoints and generation seeds are not independent training replicates.
Restrict training-stage claims to this lineage and text corpus.
Set a minimum detectable difference or interval-width target using saved data and synthetic nuisance cases before spending on extra seeds.

The cited 7.3 to 6.95 points are estimated standard errors, not observed third-seed interval widths.
`seed_check.py:46–53` predicts the third seed from a two-seed variance model.
The r≈0.99 result concerns per-role blind-naming rates in E3 ex2, not geometry or cross-format effects.
See `results/seed_check_261023.json:100–115`; it does not justify repeat counts for every new experiment.

---

**B8. The shared generator saves the wrong token population for the proposed comparison.**

Evidence: `src/manifold_persona/generate.py:71–91,207–239`; `261008_base_persona_map/gen_extract.py:109–125,265–282`.
The shared generator includes the stopping EOS in pooled states and its “last response token.”
E3 removes EOS/pad before pooling.
Thus Pod A and E3 do not currently measure the same token population.

A one-word answer followed by EOS can get half its mean from EOS.
A generated EOS alone decodes to empty text but still gets a finite vector.
Other zero-token cases receive the last prompt state, despite the response-only claim.
Freeze EOS/marker removal, empty-row exclusion, validity, and stop-reason rules across A, B, and C.

Save raw ids, kept ids, exact prompt ids, masks/spans, and stop reasons before decoding hides special tokens.
Use NaN plus an explicit invalid flag for unusable replies; never replace them with prompt activations.
Test EOS-only, first-token marker, pad, length-limit, and marker-crossing-token cases offline.
Verify activation agreement between stored generation spans and a same-frame forward pass before full extraction.

---

**B9. The listed Pod A code changes omit sampling settings and input-id storage.**

Evidence: `DESIGN.md:45–51`; `extraction/generate_and_extract_roles.py:201–205,294–311`; `src/manifold_persona/generate.py:57–60,181–184`.
There are no CLI settings for top_p, top_k, or repetition penalty in that extractor.
Only temperature reaches `generate()`; other settings come from model generation defaults.
Without `--do_sample`, the default is greedy, so three seeds would produce no sampling repeats.

Expose and record the full generation configuration, including effective EOS/pad ids and overridden model defaults.
Plumb every setting through both sequential and batched paths.
The proposed “save reply ids” change also omits prompt ids, which the design promises at line 46.
`RoleRecord.text` stays empty in this path; `record_to_messages` does not render or save token ids.

Add a common row schema and explicit run identity for roles, question ids, generation seeds, templates, and tokenizer files.
Persist these in shards and final metadata, and include them in resume checks.
The current resume guard does not fingerprint generator code or tokenizer contents (`generate_and_extract_roles.py:142–155,240–248`).
A resumed run can otherwise mix extraction rules after code changes.

---

**B10. Pod C needs a new input adapter and a different integrity gate, not just two CLI flags.**

Evidence: `DESIGN.md:69`; `context_extract.py:48–49,74–97,124–127,149–169,186–189`.
The script expects E3 `items.jsonl`, `replies.jsonl`, `shared_q`, `valid`, and a fingerprint.
Its integrity gate also requires E3 slim-state metadata.
Pod A writes parquet metadata and differently named arrays; the current script will fail before useful extraction.

The existing C0 gate compares a forward pass with stored states from the same model and original frame.
Applying it across checkpoints must fail when representations change: that is the proposed signal.
Applying it to Instruct chat states versus the new plain-text frame also changes the input.
Keep this gate only for a same-checkpoint, original-chat-frame reconstruction check.

Build each read-through frame once, store its ids and pooling indices, and feed those exact arrays to every checkpoint.
Recompute the final Instruct reference in that same plain frame; do not reuse Pod A chat-frame activations as the reference.
Assert vocabulary compatibility, sequence bounds, special tokens, row order, and identical masks/positions.
Do not re-tokenize reply strings and assume they equal generated ids.

---

**B11. Reusing the E3 analysis will reject the new grids or silently retain the wrong scope.**

Evidence: `part2_analysis.py:44–50,128–144,292–299,421–434,488–495`.
The analysis fixes 20 role classes, 18 shared questions, five wordings, and E3 default labels.
Pod A has 40 questions and a five-wording `default`; Pod B has ten roles and four formats.
An offline call confirmed that the grid guard rejects a changed design.

Parameterize roles, question sets, formats, default definitions, and layer indices from a frozen manifest.
Check every expected cell and duplicate explicitly, including seed and format dimensions.
`check_design()` currently checks level coverage, not unique complete combinations.
`cell_sums()` would silently pool seeds or formats if the caller did not separate them.

Define “held-out role decoding” precisely: the E3 decoder holds out questions and wordings, not role classes.
A closed-set classifier cannot predict an unseen role class without a separate target representation or transfer rule.
Add format-held-out splits with simultaneous question/wording exclusion.
Fit scaling, TF-IDF, PCA used by decoders, nuisance removal, and alignment on training data only.

---

**B12. Checkpoint identity and tokenizer pinning are not implemented safely.**

Evidence: `DESIGN.md:43,56,66,78`; `src/manifold_persona/extract.py:57–73`; `extraction/generate_and_extract_roles.py:274–285,339–340`.
The loader pins weights but loads the tokenizer from the default revision.
The tokenizer override also lacks a revision argument.
The recorded tokenizer commit is resolved afterward, rather than bound to the loaded tokenizer object.

A moved branch can therefore change tokenization during resume while the run configuration still matches.
The extractor records multiple model commits with a warning instead of refusing mixed weights.
Resolve immutable model and tokenizer snapshots before extraction; reject a changed snapshot on resume.
Store tokenizer/template hashes, full code hashes, and effective dtype/attention settings with each run.

Fix the checkpoint manifest before the pod: exact repositories, immutable revisions, parent chain, training stage, and step semantics.
Adjacent RL or pretraining checkpoints are calibration observations, not independent evidence of a stage effect.
Check whether final Instruct duplicates a listed RL endpoint, and whether each repository contains tokenizer files.
No offline file here proves the proposed complete 7B checkpoint ladder; do not substitute another size or lineage silently.

---

**B13. A fresh-pod and recovery plan is missing executable dependencies and safe extraction checkpoints.**

Evidence: `DESIGN.md:78–86`; `requirements.txt:1–27`; `pyproject.toml:10`; `context_extract.py:77–84,115–132`.
The image and HF token alone do not install this project, its analysis stack, or a compatible Transformers version.
`pip install -e .` installs no dependencies; requirements are unpinned and omit sentence-transformers.
My local Transformers 4.46.3 has no `olmo2` entry in its AutoConfig mapping, so local model-loading success is unverified.

Supply a tested install command and lock before allocation, using the known E3 4.47.1 setup as a starting point.
Verify the installed model class, tokenizer rendering, imports, and one row per template/checkpoint family before the full job.
The complete E3 analysis needs a sentence-embedding cache or an embedding-model run (`part2_analysis.py:523–543`).
A CPU rebuild must receive that cache or explicitly install and pin its CPU dependency.

Context extraction opens final memmaps in write mode and has no batch resume or completed-shard check.
A crash can leave a large partial array; rerunning before a manifest exists overwrites it.
Add atomic shards, completion metadata, per-checkpoint output namespaces, and refusal of mismatched resumes.
Copy and hash each finished checkpoint off an ephemeral pod before loading the next one.

---

**Other issues**

**What each experiment can establish**

| Claim | Assessment |
|---|---|
| A1: stable role direction | Split halves test repeatability, not persona function. Axis wordings currently leak across halves. |
| A2: role induction | Pod A can be a positive control after enactment validation. Naming alone tests nameability. |
| A3: common in-role judgment | Not established. Validation must cover each actual text/format distribution. |
| A4: comparable maps | Same ids are necessary. Calibration, reliability, reference frame, and coordinate assumptions remain missing. |
| A5: role beyond prompt/topic/format | Pod B helps with format. Topic, length, signatures, and narration remain alternative explanations. |
| F1: fixed input | A valid control for input changes. It does not identify persona-specific model change. |
| F3: reading differs from playing | Correct scope rule. Apply it to every Pod C axis and default-projection claim. |
| A5f: interview suffices | An untested restriction. Base-only Pod B cannot establish format invariance across trained checkpoints. |

No complete new experiment has a fully specified pass/fail rule yet.
`DESIGN.md:86` promises later threshold files; those files must define the actual decisions before a full run.
The certain-pass quantities are self-cosine and a default contrast projected onto its own axis.
The other tests can fail, but several can pass for nuisance structure, as the blockers show.

---

**The Qwen-name correction targets the wrong call path**

`prompts_roles.py:51` has a Qwen default argument, but the extractor overrides it at `generate_and_extract_roles.py:233–235`.
With the intended model argument, the placeholder becomes `OLMo-2-1124-7B-Instruct`, not Qwen.
Choose a model-neutral string for all stages, but do not describe this as a current Qwen-name bug.
F10 in the earlier review concerned changing model names in default prompts across compared runs (`review.txt:97–102`).

---

**Freeze the analysis choices before generation**

Define a primary layer, primary contrast, signed axis convention, null, failure outcome, and correction family for each experiment.
Treat other layers, kNN, PC ends, selected role groups, and alternative filters as exploratory unless included in that family.
Specify kNN settings and use only training data for tuning or dimensionality reduction.
Declare the minimum reliable map and interval precision needed to interpret stage differences.

With 20 centered role means, PCA rank is at most 19; with ten roles, it is at most nine.
Thus “PCs for 70%” needs a declared matrix: role means or individual replies.
Their interpretations differ, and neither proves a nonlinear manifold.
Inspect eigengaps and subspace stability before interpreting PC1/PC2 endpoints; see `part2_analysis.py:331–337`.

---

**Budget the complete job, not just generation**

The stated counts are correct: A has 4,200 replies per seed; B has 24,000 replies across three seeds.
A's mean and last arrays require about 2.11 GiB per seed; B's mean array requires about 6.04 GiB.
C's two frames require about 6.34 GiB per checkpoint for all 12,600 A replies, before controls or duplicate arrays.
These estimates assume 33 states, width 4,096, and float16 storage.

The 1.5 A100-hour estimate extrapolates E3 throughput to new templates and possibly sequential per-item sampling.
Benchmark the longest format, maximum-token replies, all-state pooling, and required controls before extrapolating.
E3 already warns of roughly 400,000 logistic fits (`NOTE.md:70`); do CPU analysis after releasing the GPU.
Disable unused cache/logit outputs where supported, and measure peak memory at the planned extraction batch size.

---

**Outside replication needs accessible inputs and a complete recipe**

Private HF uploads are backups, not an accessible replication package (`DESIGN.md:82`; E3 `NOTE.md:69,99`).
Publish an accessible frozen artifact set, or state the access requirement explicitly.
Include immutable code, prompts, raw judge answers, row masks, seed keys, input ids, arrays, hashes, and output-to-command mappings.
A dirty-git flag alone cannot reconstruct local changes (`src/manifold_persona/provenance.py:79–94`).

Give a clean-environment command that rebuilds every reported table from arrays without a model download or judge call.
Include all analysis settings, exact split ids, and tolerances for floating-point differences.
Provide separate commands for regeneration and saved-array reproduction; identical regeneration across GPU kernels is not guaranteed by seeding alone.
Verify restoration before deleting the pod; the earlier incomplete local C2 copy shows why hashes matter (`NOTE.md:100`).

---

**Suggested changes**

| Order | Concrete change | Release condition |
|---|---|---|
| 1 | Narrow Pod C to fixed-text role representation; separate occupancy. | Every claim names text source, frame, layer, and population. |
| 2 | Specify metrics, controls, exact role/template sets, checkpoint chain, and pass rules. | Synthetic nuisance cases fail the intended specificity tests. |
| 3 | Define one token/row schema and adapt generators, extraction, and analysis. | Offline cases cover EOS, empty replies, altered grids, and repeated seeds. |
| 4 | Fix sampling, immutable snapshots, resume checks, and package installation. | Clean-pod smoke commands are fixed and bounded. |
| 5 | Validate the actual judge/filter with held-out human labels. | Enactment and quality gates pass separately, or claims stay unfiltered/nameability-only. |
| 6 | Run a small A/C canary: final Instruct plus base, common held-out text halves. | Integrity checks pass; matched baselines and nulls behave as specified. |
| 7 | Run the full A/B jobs, then the checkpoint sweep. | Measured throughput, artifact recovery, and analysis precision justify the cost. |

Prepare the recipe, checks, and thresholds before any paid canary.
Do not add all checkpoints first and decide how to interpret them afterward.

---

**What you verified**

I read all eight requested sources, including the historical review and research plan.
I also read the shared generation/loading code, configuration, provenance code, dependency files, default prompts, and E3 seed-check code/results.
I inspected cached 7B tokenizer templates and generation settings as local JSON only.
The cached Instruct template supports system messages; I found no basis for claiming that interface itself will fail.

| Offline check | Result |
|---|---|
| Frozen question set versus extractor seed 0 | Exact match: 40 indices from 240 questions. |
| Same question set versus seed 1 | Different; generation and question seeds must be separate. |
| E3 axis function with no role differences | Cosine 1.0 from a fixed default offset alone. |
| Map agreement after a common role-label shuffle | Unchanged: 0.9856464897. |
| E3 grid guard with changed role/question counts | Rejects the changed design. |
| Reply counts and array sizes | Recomputed from the stated factorial designs. |
| E3 seed rationale | r≈0.99 and the predicted standard errors exist, with narrower scope than claimed. |

Python checks used `.venv/bin/python -B` and small local arrays.
I did not load model weights, use a GPU, call a paid model, or request network access for this review.
Checkpoint availability, full lineage, GPU memory, runtime, and real OLMo numerical agreement remain unverified.
Only this review file was intentionally written; no source or experiment artifact was edited.
