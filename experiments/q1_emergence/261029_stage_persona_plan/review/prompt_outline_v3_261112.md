# Adversarial check of the draft outline for Thomas (12/11/2026)

A teammate wrote the outline below and says every number is checked. Assume there are mistakes and overclaims. Find them. Be blunt, specific and truthful. Agree only where you checked a file yourself.

Rules: read only. Do not edit, create or delete any file. No GPU work, no pods, no paid API calls. Small local reads only (`cat`, `sed`, `rg`, `python3 -I` or `.venv/bin/python -I` on JSON). Read each text source you cite in full; for large JSON, read the keys you use and say which.

## Inputs

- Outline: `experiments/q1_emergence/261029_stage_persona_plan/draft_outline_thomas_v3_261112.md`
- Proof table with an earlier Codex review: `experiments/q1_emergence/261029_stage_persona_plan/review/proof_ledger_261112.md`
- Free checks: `experiments/q1_emergence/261112_free_checks/NOTE.md`, `results/check1_poda.json`, `results/check1_readers/*.json`, `results/check2_readers.json`, code `check_role_count.py`, `check_default_readers.py`
- Owner review: `experiments/q1_emergence/261029_stage_persona_plan/owner_review/NOTE.md`
- Result files: pod A `/Users/brendon/research-log/manifold-persona-log/Reference/2026-10-30 A-B pilot results/full_A_results.json`; pod B `/Users/brendon/research-log/manifold-persona-log/Reference/2026-11-04 B Modal recovery/full_results/analysis_full_batch_v3_cuda_261105/` (`D2.json`, `sensitivities.json`); pod C `experiments/q1_emergence/261029_fixed_text_read_through/records_full/analysis_full_v2/results_full_v2.json`; E3 `experiments/q1_emergence/261008_base_persona_map/results/naming_v2.json` and `results/part2/part2_context_run_main.json`; judge `experiments/q1_emergence/261029_judge_validation/results/dev_judge_choice.json`; 1B audit `output/results_2026-09/audit_2026-10-02/judge_gpt.json`
- Literature texts: `output/lit_stages_2026-11-12/txt/` (2601.10387 Lu, 2605.25459 Asvin G. and Lindsey, 2609.23935 Nguyen, 2509.23024 Li, 2609.32196 Nik Khah and Bahreini), `output/lit_stages_2026-11-12/lw/axis_shift_stages_clean.md` (Han Xiao), Persona Geology `/private/tmp/claude-501/-Users-brendon-persona-manifolds-manifold-persona/ad361f36-2648-47d6-95a7-4e43e5c9a47c/scratchpad/geology/geology.md`

## Tasks

1. Every number in the outline: open its source and check value, interval, population and wording. Report each mismatch with the value you found.
2. Every claim: is its strength label honest (narrow, hedged, exploratory)? Name any sentence that claims more than the evidence shows, especially behaviour inferred from structure.
3. Check 1 and check 2 code: do `check_role_count.py` and `check_default_readers.py` compute what the outline says (position statistic, within-wording spread, crossed draws, planted shift, text baseline, row alignment between pod A rows and pod C states)? Cite `file:line` for any bug.
4. Related work: is each summary faithful to the source, and is each "supports / repeats / parallel / no prior test" label right? Is anything important missing?
5. Missing contrary evidence: anything in the repo, the vault or the sources that cuts against a claim and is not in the outline.
6. Style: the owner bans AI-writing tropes (negative parallelism "not X, it's Y", punchy fragments, runs of three, stakes inflation, vague attributions, synonym cycling) and allows bold only for the labels Claim, Context, Evidence and Limits. Flag violations.

## Output

Short sentences, plain words. One table per task: item, finding, evidence (`file:line` or file + key + value), severity (wrong / overstated / missing / ok). End with the five most important fixes and a read-depth table.
