# A/B analysis review, 30 Oct 2026

The analysis implementations are written. Offline verification passes. No A/B empirical result has been calculated.

## Scope

The owner asked Codex to write and review A/B analysis before checking the HF data.
The parent implemented A's core and manually reviewed all new modules. Implementation workers wrote B and A's naming scorer.
No reviewer subagent was used. This is an implementer review, not an independent human review.
Thresholds, generation scripts, C code and shared row code were outside this edit scope.

| Implementation | Frozen rule | Coverage |
|---|---|---|
| A `analysis.py` | A THRESHOLDS v1.3 | D2 map/null, D3 contrast and opposite-half projections, matched decoding/text readers, spectrum, generation and wording sensitivities, checks and checkpoints |
| A `analysis_naming.py` | A D1 | Supplied approved judge outputs, fixed 276-name catalog, all attempts, two judges, crossed intervals, wrong-role null and blind repeat-audit checks |
| B `analysis.py` | B THRESHOLDS v1.7 | Balanced D1 decomposition, full-refit held-out-format D2 null, joint D3 map null, text/quote/context/name controls, checks and checkpoints |

## Review fixes

| Issue | Fix and regression evidence |
|---|---|
| A's within-model null differs from C's two-direction cross-model null | A now permutes the right half only. A direct seeded reference checks the complete null array. |
| Reusing C's map bootstrap centred all roles before the role draw | A now centres distinct sampled roles, as E3 does. Duplicate draw positions keep pair multiplicity. A direct-vector regression checks the Gram result. B has the same direct-reference check. |
| A lacked separate wording and held-out-generation results | Added descriptive maps, axis checks and activation decoding. Held-out-generation decoding trains on the other generations and tests disjoint questions and wordings. Missing support stays explicit. |
| D1(b) used role points despite A's global lower-bound rule | Each role now has a question-bootstrap interval on the shared draws. The role gate uses its lower bound. A point-clear/lower-fail regression checks this. |
| Saved phases could be changed without changing the run fingerprint | A checks result and dependent-array hashes. B seals phase JSON, resampling files, D2 inputs, reader arrays and null checkpoints with SHA-256 sidecars. Corruption tests fail before reuse. |
| B's sentence reader omitted E3 normalization | Both readers use the pinned E3 model with normalized float32 embeddings. Model files must already exist locally. |
| Historical generation identity was coupled to today's checksum helper | Manifests must name the approved generation hash. Current analysis helper hashes are recorded separately. C's new streaming checksum helper does not rewrite generation history. |
| Reduced checks or unconverged readers could look conclusive | Reduced counts cannot yield a family verdict. Nonfinite intervals remain unavailable. A decoder that does not converge stops. Real heavy analysis requires a Linux pod. |

## Verification

All tests use synthetic arrays, the cached tokenizer or tiny random CPU models. HF access was disabled for the test run.
The final full A/B suite passed 183 tests in 174.05 seconds. Compilation also passed. The verification record gives the command, output and source hashes.
The entry points also have synthetic-only and resume/corruption tests. A reduced engineering CLI fixture exercises the main A pipeline.

Before the pod restart, a concurrent C amendment changed its shared summary helper to shifted intervals.
A now computes its frozen v1.3 percentile intervals locally. It does not inherit C's amendment.
The regression first failed against the shared helper; after isolation, all 101 A/B analysis tests passed in 30.82 seconds.
The 183-test full-suite record above precedes this final isolation; the verification record keeps both commands and source versions.

Files: vault `Reference/2026-10-30 A-B analysis verification.txt` and `Reference/2026-10-30 A-B analysis source hashes.json`.

## HF input check

After the code review, Codex checked the immutable full-run revisions again.
Every A receipt file matched (11/11); every B receipt file matched (21/21).
The check used remote LFS metadata and bounded text-sidecar downloads. No activation array was downloaded.
It establishes archive integrity, not a scientific pass or a new full restore test.

| Pod | HF revision | Expected primary states |
|---|---|---|
| A | `60a81b9b2e337a46a0a2c41862357164273da5ff` | 21,000 rows, hidden state 17; states 9/23 exploratory |
| B | `969f11a14a61cc1888d9532d5dd1a5c8cd8eba1f` | 24,000 rows, hidden state 17; quote/context use state 16 from 8/16/24 |

Files: vault `Reference/2026-10-30 A-B HF preflight.json`.

## Limits before a real invocation

| Item | Treatment |
|---|---|
| Holm family significance level | The frozen text names Holm but omits its family alpha. Default output contains adjusted p-values and no family verdict. `--holm-alpha` is an explicit later choice. The owner's formatting reply selected no alpha. |
| A paid naming panel | Not run. D1 remains unavailable until the separately approved, frozen panel and both judges' outputs are supplied. The scorer makes no paid calls. |
| Fidelity and PC ends | Fidelity needs the validated labels. PC-end interpretation needs numerical eigengap/stability rules. Both remain explicitly unavailable. |
| Real-data execution | Not performed by this code task. Recheck exact source bytes, restore checksums and raw cases on the analysis pod first. Saved G3/G4 evidence has no built-in source fingerprint; its provenance still depends on the verified run and owner record. |
| B runtime | The frozen D2 null requires 4 × 9,999 × 50 = 1,999,800 classifier fits, before observed fits and controls. A bounded pod timing pilot must precede a sustained run. Do not shorten that null and call it confirmatory. |
| Current C work | The owner archived all three canary sources after the accepted F3 amendment. The owner is fixing review findings before the full ten-checkpoint run. These implementation checks establish no C finding. |

## Pod commands

Run from the relevant experiment folder with `PYTHONPATH` pointing to the deployed repo's `src`.
Use separate output folders for the pilot and frozen analysis. These commands have not been executed on real inputs.
The raw-read note must record actual case inspection. The sentence model must be cached at the pinned E3 revision.

```bash
# A bounded timing/engineering pilot on the restored FULL run, on Linux only.
python analysis.py --run RESTORED_A --out A_PILOT --execute-on-pod --raw-read-note RESTORED_A/raw_read.md --engineering --n-boot 20 --n-perm 99 --n-halves 2 --embedding-model-path LOCAL_PINNED_MODEL_SNAPSHOT

# B bounded timing/engineering pilot on the restored FULL run, on Linux only.
python analysis.py --run RESTORED_B --out B_PILOT --execute-on-pod --raw-read-note RESTORED_B/raw_read.md --allow-nonconfirmatory --n-boot 20 --n-perm 19 --n-splits 2 --n-halves 2
```

The frozen commands omit the reduced-count flags and use new output folders.
A supplied approved naming payload is added with A's `--naming` option.
No new rental, paid judge call, real analysis job or deletion was started.

## B execution-speed follow-up (30 Oct, before full B launch)

The parent manually reviewed training-only scaler reuse, ordered draw scheduling and the optional process backend. Each worker loads the checked analysis source by path, maps read-only features and enforces native thread limits. Every classifier refits under the frozen folds and seeds. Interrupted cleanup drains the bounded active draws before removing disposable transforms; no reviewer subagent was used.

All 76 B analysis tests passed on the Linux pod in 22.00 seconds, including process import, convergence warnings, interruption/resume, corruption refusal and resource provenance. On real prompt features, the deployed process backend matched the original serial observed score, all 32 null draws and their p-value over three fixed folds of one format. This is a bounded engineering check, not a scientific result. Source SHA256: `1afedb0a3d314224743c90273756ef2e6d0bed93c28d0e30ce5ab92d1734434a`.

Sixteen processes with one inner thread performed best in the checked configurations. The small sample projects roughly 60 hours for the frozen 1,999,800-fit null, with other phases additional. Full B started at 15:45 SAST under an initial two-hour cap; the request for a longer budget has not been approved. Evidence is in the vault's `Reference/2026-10-30 A-B pilot results/` folder.

## B equivalent GPU and variance execution (30 Oct, 16:55 SAST)

The parent manually reviewed the D1 weighted Gram contractions against the original role/question indexing, including duplicate draws, centring, cancellation and percentile signs. The helper receives existing draws and returns only their statistics. Conservative numerical guards call the original calculation. All 2,000 real-data draws matched within 5.52e-16, with identical signs, p-value and decision; the optimized calculation took 0.795 seconds. Synthetic tests cover planted terms, duplicate-heavy draws, offsets, degeneracy and boundary fallbacks.

The optional GPU path retains sklearn LogisticRegression, C=1, balanced weights, lbfgs, max_iter=2000 and the default tolerance. It uses the exact training-only float64 transforms, CPU labels, deterministic CUDA and TF32 off. Every observed fold is compared against a fresh CPU fit before null execution. Worker source hashes, resource settings, ordered draw seeds and checksummed prefixes remain checked; no model, split, draw count or statistic changed.

All 120 B tests passed on the actual pod in 115.62 seconds. The integrated eight-process GPU smoke matched the serial observed score, all 32 null statistics and their p-value on three fixed folds of one format; checkpoint integrity passed. Warm timing on 96 fits projects about 29.7 hours for the full null, with other phases additional. The cold integration smoke includes setup and is not a throughput forecast.

The original B group was stopped and its exit verified before this sole new run started. Original outputs remain intact. The new source lives separately in `repo_fast_v2`; C's job is untouched. The original two-hour deadline remains 17:45:52 SAST; a longer budget is pending. Exact code and small proofs are under vault `Reference/2026-10-30 A-B pilot results/B_fast_v2/`. This review establishes engineering equivalence within the checked cases, not a completed full analysis or an independent human review.

## B true-batch follow-up (30 Oct)

The owner rejected the roughly 30-hour forecast and asked for batches. The parent reviewed the new device-resident independent L-BFGS implementation against the original sklearn/SciPy objective and settings. Each fit retains its own zero initialization, balanced weights, C=1 penalty, curvature history, More-Thuente line search, gradient/function tolerances and iteration/evaluation caps. No coefficient history is shared between fits. The same labels, folds, seeds, full refits and macro accuracy remain.

The loss kernel matched sklearn's actual loss and gradient; device tests also check finite differences and forbid host transfers during evaluation. Two-loop and compact inverse products match explicit synthetic BFGS matrices; compact execution was slower and is not selected. On real inputs, all 96 prediction sets and iteration counts matched CPU fits, and all 32 null statistics matched. The integrated production path passed its CPU observed/null prediction guards and saved-checkpoint checks. All 201 production tests passed in 24.93 seconds, including exact interrupted-prefix recovery.

Width 256 measured 185.85 fits/s and width 512 measured 188.46 fits/s; 256 is selected for lower memory and shorter checkpoint intervals. The short benchmark projects roughly three hours for the null alone. The callback-only prototype was slower and is retained as a failed performance approach. These checks support this execution change within the tested cases; the full scientific analysis is not complete.

The prior B group exited at 17:18:31 SAST and all outputs remain. The isolated full candidate and four-hour guarded launcher are prepared; the requested run budget is pending. No full job has started with this backend. Exact source and corrected timing receipts are in vault `Reference/2026-10-30 A-B pilot results/B_true_batch/`.
