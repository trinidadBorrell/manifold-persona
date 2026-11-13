# A/B analysis implementation plan, 30 Oct 2026

Goal: implement the frozen analyses, review the code, then check the HF inputs.
Scope: new analysis modules and synthetic tests; no real-data analysis or judge calls in this step.
Mode: extended, because two experiments share scientific conventions and have active owners.

## Ownership and isolation

- Codex parent owns new A analysis files, integration, review and records.
- One implementation worker owns new B analysis files and tests only.
- Existing generation code, C code, shared row code, thresholds and label sets remain unchanged.
- Use the shared checkout because these experiment sources are untracked owner work; do not reset or commit.

## Spec

- A: `../THRESHOLDS.md` v1.3; B: `../../261029_base_role_vs_format/THRESHOLDS.md` v1.7.
- Reuse established E3 classifier conventions and read C map helpers without changing them.
- Frozen defaults: 2,000 crossed bootstrap draws, 9,999 permutations, 50 fixed halves/splits.
- Any reduced offline setting marks the result as non-confirmatory.
- Heavy analysis runs on a pod after this code step. Runtime inputs must pass saved gates, checksums and schema checks.

## Tasks and verification

1. Write failing synthetic tests for prompt weighting, one-map nulls, axis leakage, held-out folds and invalid input rejection.
2. Build A D2 map repeatability and D3 contrast/projection with crossed uncertainty and precision checks.
3. Build the A D1 scorer for supplied, approved judge panels; never invoke a judge. Missing panels remain unavailable.
4. Add matched decoding/text baselines and the frozen replicate sensitivities. Report unsupported interpretive criteria explicitly.
5. Build B D1 variance decomposition, D2 held-out-format decoding and joint D3 map null in parallel.
6. Add explicit run provenance, deterministic phase checkpoints and code/settings fingerprints.
7. Run both synthetic suites, inspect code against every threshold section, and fix demonstrated errors.
8. Save review findings and remaining limits. Update notes, plan and vault without disturbing concurrent edits.

## Review focus

- Equal prompt weight after averaging valid replicates; no imputation or silent loss of failed cells.
- Training-only preprocessing and simultaneous question/wording holdout.
- Correct role-block permutations, refits where required, and joint hypothesis correction.
- Duplicate role draws, undefined intervals, precision gates and conservative unavailable decisions.
- Exact layer labels, source receipts, invalid-row NaNs and stale checkpoint refusal.

No reviewer subagent is used, per AGENTS.md. The parent reviews the worker's B code and checks A against adversarial tests.
This is a code review, not an independent empirical review.
