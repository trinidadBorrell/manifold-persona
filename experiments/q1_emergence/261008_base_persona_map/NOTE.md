# E3: does OLMo-2-7B base play the roles, and does it hold the persona map? (08/10/2026)

Status: **design. No base-model run yet.** Only the pilot in-context examples are chosen.

Mode: **de-risk**. A 20-role pilot tests the prompts, the judge and the code before the full 276-role run.

## Owner decisions (08/10/2026)

- Study the base model only: `allenai/OLMo-2-1124-7B`. Step A of the first E3 draft (Instruct text read by Base, a replication of Lu et al. B.3) is dropped.
- Two parts, both on the base model's own replies: (1) does it play the role, (2) the persona map from its activations on those replies.
- Plain text only. No chat template. Instruct models get no few-shot turns.
- Two prompt conditions that differ in one factor: **Description** (third-person role description, question, answer cue; Moskvoretskii et al. 2026 main format) and **Description + 2 examples of the same role** (after Moskvoretskii's Dialogue format and Han et al. 2022).
- Examples come from existing replies, chosen by a fixed rule, not by hand.
- Kept as they are: translator example 2 (translates the question) and consultant example 2 ("consulting").

## Literature read (full text, 08/10)

- Han et al. 2022 (arXiv 2204.10825): base LMs 1.5–6B; 8 hand-picked lines of the target character as `User:`/`{name}:` pairs; strong word copying from examples (2-gram overlap 0.35 vs 0.04).
- Moskvoretskii et al. 2026 (arXiv 2605.13329): OLMo-3-7B; main Description format has no examples; Dialogue has 2 scenes of the same character; temperature 0.5, 10 samples, 64 tokens, repetition penalty 1.1, keep trait and coherence ≥ 50; the three formats give different directions.
- Neither paper uses examples of other characters (as E0 did).

## Pilot example selection

Code: `auto_examples.py`, `run_judge.sh`. Judge: E0 blind top-5 naming prompt (`prompt_judge.txt`), GPT-5.6-Sol, codex-cli 0.162.0, reasoning high.

Sources (reply text only):
- O: `brend0nc/manifold-persona-olmo1b-stages` @ `22250ccc62345c7ae84ba1cb86e52100ad83ed54`, `rlvr_zeroshot/metadata.parquet` (OLMo-2-0425-1B-Instruct, no few-shot).
- Q: `data/embeddings_roles_resp_40q/metadata.parquet` (Qwen2.5-3B-Instruct).

Rules: (1) judge puts the role in its top 5 from the reply alone; (2) no 5-word run shared with any of the role's 5 descriptions; (3) 30–70 words, whole sentences, no list or AI self-reference; (4) does not name its own role; (5) no anonymiser placeholder or inline "1)" list. Rule 5 was added after the v2 judge run; it only removes replies.

| Run | Roles | Pool | Result |
|---|---|---|---|
| v1 | 20 pilot roles | 8 O + 4 Q per role, rules 1–3 | 14 of 40 chosen examples named their own role → rule 4 added |
| v2 | same 20 | 20 O + 10 Q per role, rules 1–5 | 17 roles good; tutor, analyst, librarian weak |
| v3 | 10 replacement candidates (rule fixed before judging) | same | translator, counselor, consultant replace tutor, analyst, librarian |

Final: `pilot_examples_final.json`, 20 roles × 2 examples, every example's role is the judge's first guess. 35 examples from O, 5 from Q.

Pilot roles: assistant, translator, counselor, consultant · chef, detective, bartender, criminal · toddler, teenager, grandparent, refugee · cynic, optimist, stoic, narcissist · pirate, vampire, robot, void.

Superseded: `select_pilot_examples.py`, `pilot_examples.json`, `pilot_examples_review.md` (hand-picked draft before the automatic rule).

## Findings during selection

- Small Instruct models answer assistant-end roles (tutor, analyst, librarian, editor, programmer) in a plain assistant voice. This fits Lu et al.'s map, where these roles sit near the Assistant end.
- 5 of 55,200 stored replies in the O dataset contain `<PRESIDIO_ANONYMIZED_PERSON>`. Some reply text was changed at upload.
- The judge opened Codex skill files outside the allowed files in each run. It read replies with cat/sed and checked its output with jq; it did not score replies with a script.

## Generation code and smoke test (08/10)

- Pod: A100 80 GB, `/root/venv` (torch 2.8.0+cu128, transformers 4.47.1, tokenizers 0.21.0), model `allenai/OLMo-2-1124-7B` @ `7df9a82518afdecae4e8c026b27adccc8c1f0032`. Tokenizer: same repo and revision; no chat template; adds no start token; same vocabulary as Instruct. No `/workspace` volume: copy outputs back after each step.
- Code: `gen_extract.py`. Inputs: `questions_40.json`, `pilot_examples_final.json`, `descriptions_third_person.json` (100 reviewed rewrites; robot/0 grammar fixed by hand).
- Owner decision: use the 40-question set. desc: all 40 for every role and default. ex2: each role's 2 example questions are left out (38 per role).
- Codex review (GPT-5.6-Sol): `review/codex_review.md` (14 findings), `review/codex_rereview.md` (after fixes). Fixed: token re-encoding, resume safety, positions after padding, pronoun grammar, empty-row handling, provenance. Open design items: ex2 question sets differ per role (O7); ex2 prompts are ~180 tokens longer than desc (finding 10); examples chosen by the same judge family that may score base replies (finding 12).
- Smoke test, 40 random items (`smoke/`, last run `smoke3`): 34 replies stop at a marker, 6 at the token limit, 0 empty; all activations finite, shape 40 × 33 × 4096; reply text decodes exactly from the kept ids; resume and refusal paths work. A first smoke run (old code) showed one ex2 reply that copied an example almost word for word. No metric is computed.

## Full pilot run (started 09/10/2026)

- Thresholds and null tests fixed first: `THRESHOLDS.md` (v2 after the Codex red-team `review/redteam_thresholds.md`; owner decisions A–D). Code re-reviewed: `review/codex_review3.md`; its 5 blockers fixed and a batch-32 smoke with a forced crash and resume passed (`smoke/`).
- Pod run: `/root/e3/run_main` (15,120 items: desc 4,120, ex2 3,800, ex2_other 5,400, ex2_filler 1,800) then `/root/e3/run_rep` (3,654 items, seed offset 100000). Log `/root/e3/run.log`. Batch size 32. The pod has no persistent volume: copy outputs back and upload them when the run ends.

## Run results (09/10/2026, no metric computed yet)

- Both runs finished: run_main 15,120 items (54 min), run_rep 3,654 (13 min); 0 invalid replies. Local copies verified against the pod (`SHA256SUMS_run.txt`, 8/8 OK).
- 20 raw transcripts read first (`run_main/raw_read_ids.json`; excluded from the human sheet). Notes: desc replies short and mostly plain; ex2 replies take the Instruct examples' assistant tone and sometimes name the role; ex2_other replies follow the donor's content; ex2_filler replies are long and spoken in style. One reply carries a stray "Interview" heading that the stop markers miss.
- Judges (isolated folder `/Users/brendon/persona-manifolds/e3_judge_pilot`, keys in `judge/KEY.json`): GPT-5.6-Sol 126/126 batches valid; Claude Sonnet 5.5 126/126 valid after re-running batches 2, 108, 120 (1–2 off-list names each). Answers copied to `judge/`.
- Human sheets ready (`human/`). No judge output is shown to the owner before both passes are done.
- **Hugging Face copy:** `brend0nc/manifold-persona-experiment-artifacts` (private dataset), path `experiments/q1_emergence/261008_base_persona_map/`, revision `fb84c5028fee341aa8832620985b7a7d29c7c302` (369 files). The 8 run files match `SHA256SUMS_run.txt` at that revision.
- **Part 2 code:** `part2_analysis.py`, reviewed by Codex in three rounds (`review/codex_review_part2*.md`); the self-test asserts that a planted signal passes and noise fails for decoding, map, bootstrap intervals and gain. Not yet run on real data. It needs many CPU cores (≈4×10^5 logistic fits for the decoding nulls).

## Open before the full pilot run (historical)

- Test questions: the 40-question set (owner, 08/10).
- Base-model judge and calibration: blind naming plus human labels; the plan asks for about 200 labelled replies.
- Success thresholds and null tests, fixed before the run.
- Advisor red-team of the full design.

## Literature judge and owner scoring (10-13/10/2026)

- Judge: `judge2.py` (v3), Claude Sonnet 5.5 via `claude -p` with all tools off. Primary score: enactment 0-3 (Lu et al. labels, rewritten for interview answers); secondary: evidence 0-100 (Moskvoretskii et al.); coherence 0-100. Controls: close, distant and donor roles; 5% repeat audit. Codex reviews: `review/redteam_judge2.md` (v1, 12 problems), `review/codex_review_judge2_v2.md` (9 blockers), `review/codex_review_judge2_v3.md` (0 blockers before a canary). Synthetic self-test passed.
- Real judge folder prepared, not run: `/Users/brendon/persona-manifolds/e3_judge2_pilot` (10,800 replies, 46,710 items, 1,869 batches; manifest sha256 `c71090e744c125c55cfb70a2e2f141992c0489d29f0e15ca02a9f0a731d833cb`).
- Owner scoring sheet: `human/judge2_human_sheet.md` (200 items from that folder: 140 target, 30 close, 30 distant; condition and kind hidden; `human/judge2_human_sample.json` holds the strata and the calibration/validation halves). The owner scores these before any judge output is seen. The earlier blind-naming sheets are in `human/superseded_blind_naming/`.
- Blind top-5 naming (GPT-5.6-Sol, Claude Sonnet 5.5) stays a secondary measure. Its answers have not been compared with the keys.
- Next: owner scores 200 items; Part 1 rules written into THRESHOLDS.md; judge canary, then the full judge run; then Part 2 on the pod.

## Part 1 results, owner labels and Part 2 (14-16/10/2026)

- Write-up for the 16/10 meeting: `review/writeup_part1_2026-10-16.md`. Claim ledger: `review/writeup_part1_2026-10-16_ledger.md`. Slides: `review/writeup_part1_2026-10-16_slides/`. The numbers below come from the files named.
- **Part 1, blind naming** (`score_naming.py`; `results/naming_v2.json`, first sealed pass `results/naming_sealed_v1.json`):
  - desc: hit@5 14.9% [9.5, 21.4] (GPT) and 17.2% [11.6, 23.6] (Claude). Fails rule 1.
  - ex2: 37.6% [23.3, 52.3] and 42.3% [29.7, 55.9]. Passes every numerical rule on `run_main` for both judges. On `run_rep`, the GPT lower bound without copied and self-naming replies is 19.8%: fails by 0.2 points.
  - ex2_filler: 8.6% and 12.7%. Prompt length alone does not explain ex2.
  - ex2_other: the judges name the donor role (24.8% to 29.8%), not the described role (2.4% to 5.0%). The replies follow the examples.
- **Owner labels (16/10):** 80 Fine/Broken labels (`review/attrition/owner_labels_*`). 50 role scores 0-3 (`human/owner_role50_scores_2026-10-16.json`), scored again after a critique of the scale (`human/owner_role50_scores_revised_2026-10-16.json`). A judge is usable if Cohen's kappa against the owner is ≥ 0.6 (`review/role_judge_rule_2026-10-16.md`).
- **Broken-reply judges** (`judge_broken.py`, `jev_judge.py`, `validate_broken.py`, `finish_broken_261016.py`; `results/broken/`): no judge passed the kappa gate on both label sets. GPT-6.1-Sol 0.57 / 0.64, Opus 5.5 0.59 / 0.56, Jev 0.54 / 0.77. Final label = owner label, else the GPT round-3 call: 4.9% to 10.5% broken per condition, provisional. About half of the ex2 broken calls are translator replies that translate the question. The new definition of broken is the THRESHOLDS.md change of 16/10.
- **Role judge, Jev** (`jev_role.py`; `results/role_judge_vs_owner50_*.json`): quadratic weighted kappa 0.41 [0.21, 0.57] against the first owner scores. Not usable. 0.69 [0.49, 0.82] against the revised scores, but the owner saw the Jev scores first: exploratory only.
- **Part 2** (pod; `part2_analysis.py`, `context_extract.py`, `part2_context.py`; `results/part2/`): role decoding at state 17, chance 5%. desc 52.3% [41.4, 61.7], ex2 97.1% [92.3, 99.3]. Context controls (THRESHOLDS.md change of 16/10): C2, desc 15.2% with lower bound 9.98%, under the 10% floor; on `run_rep` 15.7% [10.1, 21.5]. C3, desc 91.8%. ex2_other within-donor target accuracy from C2: 35.1% [32.3, 37.8] against 1/3 (exploratory). Holm over the 4 decisions: naming p = 0.0004, both Part 2 tests p = 0.0010 (`results/part2/part2_report.json`).
- **Hugging Face:** Part 2 context states and outputs at revision `d248ba5ee547fbfcbfab75a6de7695df65bf1078` (`results/part2/hf_archive_receipt_1.json`). Part 1 judge outputs, `results/`, owner labels and reviews at revision `cc8a0b32f01c89194646df77842728f2d4589365` (23/10/2026; 214 files, each checked by size and hash in `results/hf_archive_receipt_2.json`). The same revision holds the text-embedding caches `p2_emb_run_main/` and `p2_emb_run_rep/`, and `run_main/` and `run_rep/reply_mean_acts_L9_17_23.npy`. `jev/` (55,856 files) is one file: `jev_261016.tar.gz` at revision `0e6256c7956a26796a98fa1f71e771c5da1d132c`, sha256 `ce3a8c9ede2ff2cb3d30ef06d543d6506f87217736f1cc3b5befaf1350ed41e5` (`results/hf_archive_receipt_3.json`). It replaces the first tarball at `cc8a0b3`, which also held 55,869 macOS `._` metadata entries. Each of the 55,856 files in the new tarball matches the local `jev/` by sha256. Restore test (23/10): the new tarball, `judge/KEY.json`, one embedding cache, `results/part2/part2_report.json` and 25 random files downloaded with matching sha256. The scripts are in git (`fd694c3`), not in this upload.
- **Local copy fault:** the local `run_main/context/ctx_c2_L9_17_23.npy` is incomplete: from row 702 on it is almost all zeros. The pod copy on HF (`part2_261016/run_main/context/ctx_c2_L9_17_23.npy`, sha256 `99fcf8bf…`, receipt 1) is complete. `results/part2/part2_context_run_main.json` and `part2_context_run_rep.json` are byte-identical to the pod outputs, so no reported number used the bad local file. On 23/10 the local file was replaced by the HF copy (sha256 checked).
- **Git:** from 23/10, git keeps the code and small records only. `judge/` (with `KEY.json`), `v1/`–`v3/`, `smoke/` and the slide images in `review/writeup_part1_2026-10-16_slides/` are on HF only (`.gitignore`). The figure images stay in git because this note and the write-up link to them.
- Not done: `judge2.py` is not run, and the 200-item owner sheet `human/judge2_human_sheet.md` is not scored. `part2_analysis.py` is not run for ex2_filler, ex2_other and `run_rep`.
- Next (16/10 meeting): stress test before 23/10 (steer toward a far persona from several starting points; check if replies stay coherent). Then the 3 missing Part 2 runs, and the trained stages with the plain prompt.

## Code fixes after the results (23/10/2026)

A code review of the 16/10 scripts found 10 items. The scripts that made the 16/10 numbers are in commit `fd694c3`. The fixes below are not committed yet. No fix changes a reported number.

- `part2_analysis.py`: the text-embedding cache now always lives in `p2_emb_<run>/` (`--emb_dir` to override), and its name depends only on the run, the embedding model and `embed()`. Before, a pod run with `--out p2_<cond>` did not find the cache and tried to embed without a GPU. A missing cache now stops the run with a message; `--embed_only` writes it. The existing caches were copied to their new names (same content).
- `part2_analysis.py`: the slim reply states are checked against the SHA-256 in `reply_mean_acts_slim.json` (as `context_extract.py` already did). Both runs match.
- `part2_analysis.py`: `fit_model()` is the one frozen classifier; `fit_predict()` and `part2_context.fit_proba()` use it. `decode()` takes `jobs`; `part2_context.py` passes it instead of setting `p2.JOBS`.
- `part2_context.py`: for ex2_other, donor decoding and within-target donor share one set of fits (argmax of the probabilities, checked equal to `predict()` in the self-test). The sweep reads each (feature, state) slice once, not once per split.
- `context_extract.py`: the hidden states are pooled from the model's tuple, without a stacked copy that doubled peak GPU memory. Not run on a GPU after the change.
- `validate_broken.py`: an owner label other than fine/broken stops the script (same rule as `judge_broken.py`). Before, it counted as Fine.
- `finish_broken_261016.py`: uses `score_naming.rep4` and `judge_broken.kappa` instead of its own copies (checked identical on 5,000 texts and 3,000 label pairs).
- `pod_part2.sh`: usage check, and it prints the pod commands.
- Not changed: `jev_role.py` and `jev_judge.py` share copied request code. Both runs are finished, so a refactor would only make the committed code differ from what ran.
- Check: `part2_analysis.py --selftest` and `part2_context.py selftest` give the same numbers before and after the fixes. Copies of the pre-fix files: `review/code_before_fixes_261023/` (also in `fd694c3`).
