# Proof ledger for the Thomas draft (12/11/2026)

Session ad361f36. Each claim X has its proof (file, key, value), a code check where one applies, and the evidence Y that could discount it. I opened every file below on 12 Nov and read the values myself. Agent check only, not a human review.

Paths: `R` = `/Users/brendon/persona-manifolds/manifold-persona`, `Q` = `R/experiments/q1_emergence`, `V` = `/Users/brendon/research-log/manifold-persona-log`.

## Our results

| Id | Claim X | Proof | Y that could discount X | Does Y discount X? | Status |
|---|---|---|---|---|---|
| C1 | On my 40 practice labels, Terra agreed on 31 (77.5%). It said yes 17 times, right 16 (94%). Recall: base 3 of 8, chat 13 of 16. | `Q/261029_judge_validation/results/dev_judge_choice.json`, `candidates["openai/gpt-6.1-terra"]`: agreement 0.775, precision 0.941, recall 0.667; groups base judge_yes 3, owner_yes 8, agree 15; instruct judge_yes 14, owner_yes 16, agree 16. Solving the counts gives true positives 3 (base) and 13 (chat). | (a) One rater, 40 non-random practice labels, so recall per group is wide. (b) `Q/261029_judge_validation/NOTE.md` "Main judge run" says "14 of his 16 chat yes"; that line is wrong (14 is the judge's yes count). | (a) Limits any rate comparison. (b) No: the JSON is the source; the NOTE line is a typo. | Verified |
| C2 | On 30 Oct I said Terra "agreed with me at a 90% accuracy". | `V/Reference/2026-11-12 Meetings 16-30 Oct.md` line 73 (from the meeting transcript; Granola notes repeat "90% agreement"). | Only the owner knows the exact words. | Owner to confirm. | Needs owner |
| C3 | OLMo-2-7B base, 10 roles, chance 10%. Full prompt plus one fixed neutral continuation, no reply (C3): a decoder trained on three formats finds the role in the fourth at 0.909 to 0.982. Reply with the description removed (C2): 0.426 to 0.470. | Vault `V/Reference/2026-11-04 B Modal recovery/full_results/analysis_full_batch_v3_cuda_261105/sensitivities.json`, `context.C3_state16.decoding.formats` (interview 0.972, script 0.982, story 0.909, description 0.930) and `context.C2_state16.decoding.formats` (0.434, 0.470, 0.426, 0.434). Code: `Q/261029_base_role_vs_format/analysis.py:1311-1312` and `descriptive_decoding` (folds by format, questions and wordings held out). | Report-only: no null, no test against D2; state 16, not the primary 17. C2 null run with 99 draws only (largest 0.11). D2 with prompt and reply: 0.823 to 0.915 at state 17 (`D2.json`). | Limits wording to "report-only". Does not reverse the order (C3 far above C2). | Narrowed |
| C4 | Ten checkpoints read the same 21,000 Instruct replies. Base and Instruct role maps agree at r = 0.896 [0.767, 0.932]; noise-corrected 0.996 [0.991, 0.999]. A character n-gram map reaches 0.960 on the corrected scale. States beat the best text map by 0.037 [0.013, 0.081]. | `Q/261029_fixed_text_read_through/records_full/analysis_full_v2/results_full_v2.json` (HF `b343c96`): `pod_a.D1.r`, `pod_a.D2.per_stage.base`, `pod_a.D1.anchor_r_norm.text_char` 0.960, `pod_a.D1.minus_best_anchor_normalized`. | Role decoding from text beats the states: word n-gram 0.797 vs base states 0.761 (gain −0.036 [−0.070, 0.000]); Instruct states 0.754 (gain −0.043 [−0.078, −0.006]); chance 0.05 (`pod_a.decoding`). | No. It strengthens X: the text carries the role at least as well as the states. | Verified |
| C5 | On fixed text the stage change is small but above zero at base to SFT (0.0027 [0.0004, 0.0079]) and SFT to DPO (0.0009 [0.0004, 0.0019]); DPO to Instruct is 0.000004 [−0.000001, 0.000014]. A planted move of 2 of 20 roles (true r 0.83) gave 0.156 [−0.006, 0.340]. | Same file: `pod_a.D2.transitions.*.r_norm_difference`; `g5.planted_change_recorded["2_of_20_roles_moved"].change`. | The v2 draft said "not reliably"; two intervals exclude zero. | Yes for "no change": wording must say "small, above zero at base to SFT and SFT to DPO; the measure can miss a 2-role move". | Narrowed |
| C6 | Base (E3), description only: hit@5 14.9% [9.5, 21.4] (GPT judge), 17.2% [11.6, 23.6] (Claude); shuffled-label mean 2.4% / 2.9%. Two same-role examples: 37.6% [23.3, 52.3] / 42.3% [29.7, 55.9]. Replicate run: 16.3 / 18.8 and 35.7 / 40.2. | `Q/261008_base_persona_map/results/naming_v2.json`, `run_main/*` and `run_rep/*`, `results.none.{gpt,claude}.all_replies`. | (a) With the clean mask, the run_rep lower bound passes the 20% rule in only 26% of 50 bootstrap seeds (`Q/261023_judge_designs/NOTE.md:147-150`). (b) Same-role examples: the model may copy them. (c) Older judges (GPT-5.6-Sol, Claude Sonnet 5.5). (d) No owner transcript read yet. | (a)-(b) limit the ex2 claim; the description-only result stands. | Candidate for human review |
| C7 | OLMo-2-1B final: hit@5 55.4% [49.5, 61.2] without the two example turns, 2.9% [1.5, 5.6] with them (276 replies each). | `R/output/results_2026-09/audit_2026-10-02/judge_gpt.json`, `olmo1b|rlvr_zeroshot` top5 0.554, `olmo1b|rlvr` top5 0.029. | One checkpoint, 1B only, one judge. | Limits it to a narrow claim. | Verified, narrow |
| C8 | Pod A, Instruct, 20 roles, 21,000 replies: the role map repeats across halves, r = 0.901 [0.797, 0.939]; null p99 0.22. | `V/Reference/2026-10-30 A-B pilot results/full_A_results.json`, `primary.D2.raw_r`. | The role prompt is in context; in base, the prompt alone decodes the role at 0.91 to 0.98 (C3). No prompt-only control on Instruct. | Yes for "persona map": it may be a map of the prompts. | Narrowed |
| C9 | Default-minus-roles direction repeats: cosine 0.881 [0.763, 0.884], precision missed. The prompted "assistant" role gives a direction at cosine 0.936 to it. | Same file, `primary.D3.cosine`, `primary.assistant_control.cosine_to_default_contrast`. | The axis ends at the default by construction. | Yes for "privilege". | Narrowed |
| C10 | Pod A naming: macro hit@5 0.687 [0.580, 0.795] (Terra), 0.687 [0.576, 0.794] (Sonnet); shuffled p99 0.069; per role 0.195 to 0.998. Role SD 0.243 gives ±10.7 points at 20 roles; ±7.5 needs about 40 roles. | `V/Reference/2026-11-06 A D1 finish/d1_miss.json`, `D1.judges`; SD from `per_role` (computed 12 Nov). | 13 Sonnet answers stayed invalid; miss and hit bounds give the same decisions (`d1_hit.json`). | No. | Verified |
| C11 | OLMo-2-1124-7B-Instruct's RLVR stage trained on `allenai/RLVR-GSM`. None of our 40 questions is a math problem. | Cached model card `~/.cache/huggingface/hub/models--allenai--OLMo-2-1124-7B-Instruct/snapshots/470b1fba…/README.md` lines 10, 23. Questions: `Q/261008_base_persona_map/questions_40.json` with `R/refs/assistant-axis/extraction_questions.jsonl` (all 40 read; two are personal finance, none is a math task). | None found. | No. | Verified |

## Related work

Read in full on 12 Nov: Persona Geology (LessWrong API copy, scratchpad), Han Xiao (`R/output/lit_stages_2026-11-12/lw/axis_shift_stages_clean.md`, prose in full, figure data parsed). Checked passages: Lu et al., Asvin G. and Lindsey, Nguyen, Li et al. (saved texts in `R/output/lit_stages_2026-11-12/txt/`).

| Id | Work | What it shows (checked) | Answers which of our questions | Supports or contradicts our results | Status |
|---|---|---|---|---|---|
| L1 | Lu et al. 2026, arXiv 2601.10387, §3.2.2 (txt lines 438-471) | Steering base Gemma 2 27B and Llama 3.1 70B along the instruct Assistant axis gives helpful human personas; they conclude the axis "mainly inherits" from base-model personas and gains AI associations later. | Is the Assistant direction there before post-training? Yes, in two other model families. | Neither. Our base-axis test did not run (pod A D3 gate failed). | Flag for owner |
| L2 | Han Xiao, LessWrong, 5 Nov 2026 | Olmo 3 7B, SFT, DPO, RLVR; 275 roles, 5 descriptions, 240 questions; judge keeps only full role play. Desirable roles +0.19 [0.14, 0.23] on the axis after DPO (interval from 100 question half-splits); label-shuffle p = 0.066. RLVR "barely changes anything". Role-play pass rate 0.26, 0.52, 0.53 (computed from the figure data: 201 neutral, 49 desirable, 25 undesirable). Base model not studied; named as open. No text baseline. | Which post-training stage moves roles on the axis? DPO, not RLVR, in Olmo 3 own generations. | Consistent with our pod C (RLVR change 0.000004 on fixed text), but not the same test: fixed text vs own generations, other model, and our measure misses a 2-role move. | Flag for owner |
| L3 | Asvin G. and Lindsey 2026, arXiv 2605.25459 (txt lines 278-292) | OLMo-3-32B: own-text entropy advantage absent at base, appears at SFT in the Assistant field only, DPO extends it to user and no-template text, RLVR widens it in all three. | When does the model treat its own Assistant text as special? SFT, then DPO, then RLVR. | No data of ours. Y against any "RLVR changes nothing" reading: here RLVR widens the gap. | Flag for owner |
| L4 | Nguyen 2026, arXiv 2609.23935 (txt lines 136-146, App. E 470-504) | Output (plaintext): harmless bias flat from base through SFT, jumps at DPO, unchanged by RLVR; bases "treat it as fair". Logit lens: base has a mid-layer bias (peak +0.31) that the final layers cancel to +0.10; SFT erases it; DPO rebuilds it (peak about +0.60, output +0.40). | When do Assistant preferences appear? Output: DPO. Inside: base, erased at SFT, rebuilt at DPO. | No data of ours. It shows that a behaviour measure and an internal measure can give different stage answers. Correction: v2 said the base model has the preference; that is true only inside the middle layers. | Flag for owner |
| L5 | Ratnam and Africa, "Persona Geology", LessWrong / Mindscapes, 30 Oct 2026 | 134 trait LoRAs on Qwen3.5-4B. Weight similarities factor into 5 factors. The text embedding of chosen-minus-rejected replies predicts 94% of held-out adapter positions (adapter geometry 98%); similarity matrices correlate 0.86; teacher prompt alone is below chance. Authors' caveat: adapters may record the teacher's text more than the model's own persona. One model, one judge, 24 scenarios. | Does persona structure follow the training text? Yes, in weights. | Same direction as our text result (C3, C4), in a different object: weights of trained adapters, not activations of one model reading text. Partial support. | Flag for owner |
| L6 | Li et al. 2025, arXiv 2509.23024 (txt lines 688, 717-718) | Effective rank (RankMe) rises at SFT and DPO and falls at RLVR (Tülu 3.1 8B); OLMo-2 only in pretraining. | Does representation spread change by stage? Globally, yes, in another model. | No data of ours (condensation not measured). | Flag for owner |

## Corrections to `overview_thomas_v2_261112.md`

- Nguyen: the base preference is internal only; the output is flat until DPO (L4).
- Pod C stage change: "not reliably" understates it; base to SFT and SFT to DPO are above zero but small (C5).
- Pod C: add that text decodes roles at least as well as the states (C4, Y column).
- Han Xiao: add the role-play pass rates and that he names the base model as open (L2).
- G. and Lindsey: RLVR widens the own-text gap (L3).

## Codex adversarial review (12/11/2026) and my checks of it

Codex (`gpt-6.1-sol`, xhigh, read-only): prompt `prompt_proof_ledger_261112.md`, report `codex_proof_ledger_261112.md`. Agent review, not a human review. I re-opened the files for each point below; all are confirmed.

| Id | Codex finding | My check | Change to the claim |
|---|---|---|---|
| C5 | `D2.transitions` is the change in agreement with Instruct, not the direct map change. Direct base to SFT change is 0.0032 [0.0010, 0.0077] (`pod_a.stage_change`). The equal-map control gives a difference of 0.027 [0.012, 0.039] with no map change. | `g5.equal_map_wording_case_recorded.d2_difference` = 0.02658 [0.01246, 0.03868]; pod C `NOTE.md:93` and `THRESHOLDS.md:72-76` already say these differences are not evidence that a stage moves the map. | Wrong as written. New wording: "On fixed text we cannot claim any stage change: the observed differences (at most 0.003) are smaller than the bias the control shows with equal maps (0.027)." |
| C4 | 21,000 includes 1,000 default replies; maps use 20,000. The main frame F1 keeps the question and an interview header, so "no prompt" is false. The equal-map case gives r_norm 1.027, so "agree up to noise" is too strong. | `THRESHOLDS.md:16-20`; `g5.equal_map_wording_case_recorded.r_norm` 1.02658. | "20,000 role replies, read with the question but no role prompt." Drop "up to noise"; report r and r_norm with the control. |
| C4 | Lu et al. App. B.3 already did this in Gemma 2 27B: base reads the Instruct rollouts; top PCs match at cosine 0.93, 0.87, 0.83; same-role vectors above 0.99. | `2601.10387.txt:1352-1362`. | Novelty: OLMo-2, ten readers across every stage, and text baselines. Not the base-reads-Instruct result itself. |
| C6 | Owner labels exist for E3 (80 Fine/Broken, 50 role scores, 16 Oct). `ex2_other`: judges name the donor role (24.8-29.8%), not the described role (2.4-5.0%). | `Q/261008_base_persona_map/NOTE.md:90-98`. | E3 has an owner check. "Copies the examples" is supported for E3 ex2 by `ex2_other`. |
| C7 | The 1B score drop does not show copying. | Only top-5 scores in `judge_gpt.json`. | Drop the copying mechanism for the 1B result. |
| C1 | The same 40 labels chose the judge, and the pass rule changed after the practice run. | `Q/261029_judge_validation/THRESHOLDS.md:104`. | Precision 16 of 17 is optimistic; say so. |
| C8 | Fidelity labels are unavailable in the pod A result. | `full_A_results.json` `fidelity`. | The repeating map uses all valid replies, not only replies that play the role. |
| C9 | All default wordings on both sides gives cosine 0.944. | `V/Reference/2026-10-30 A axis diagnostic/results.json`. | Report as a descriptive alternative only. |
| C10 | Without self-named replies, hit@5 is 0.655 for both judges. | `d1_checks.json`. | Add as a limit. |
| C11 | RLVR also moved non-math scores: AlpacaEval 27.9 to 29.1, IFEval 73.0 to 72.3 (DPO to Instruct). | Model card table, rows OLMo-2-7B-DPO and OLMo-2-7B-1124-Instruct. | The math-only explanation for "no RLVR change" is a guess, not a result. |
| L2 | p = 0.066 tests the desirable-minus-undesirable gap, not the desirable increase. The figure data gives +0.183 over 48 desirable roles; the prose says +0.19. | Han Xiao footnote 1; figure data. | Fix the p-value wording; note the small mismatch in the source. |
| L3 | G. and Lindsey measure output entropy (nats), not surprise. The effect is near absent at about 2B. The stage test uses 20 prompts. | `2605.25459.txt:116-118`, `278-290`. | Use "entropy" in every place. |
| L4 | Nguyen's base output bias is +0.10 (final layer), against a mid-layer peak of +0.31. The main text says bases stay near zero. | `2609.23935.txt:136-146`, `473-488`. | "Small at the output (+0.10), larger inside (+0.31)", not "inside only". |
| L5 | 94% is variance explained over 34 held-out adapters, not 94% of positions. | Geology post, "The text of the preference pairs" section. | Fix the wording. |
| L6 | Li et al. also run SFT on OLMo-2-1B intermediate checkpoints. | `2509.23024.txt:525-539`. | Remove "OLMo-2 only in pretraining". |

## Owner transcript review (12/11/2026)

Source: `../owner_review/NOTE.md`; labels `/Users/brendon/persona-manifolds/judge_validation_261029/review261112_owner_labels.json`. One reader, 20 fresh replies per pod.

| Claim | New evidence | Effect |
|---|---|---|
| C1 (judge precision 16 of 17) | Pod A: judge yes 13, all also owner yes (13 of 13). Pod B (base): judge yes 4, only 1 also owner yes; it found 1 of the owner's 3 yes. | The high precision holds on fresh chat replies. On fresh base replies it does not hold (1 of 4). Base judge rates, including pod B's 15.4%, need this limit. |
| Pod B role play | Owner yes on 3 of 20 base replies (15%). | Same order as the judge's 15.4% pass rate, but the two mostly pick different replies. |

## Codex review of the v3 outline (12/11/2026) and my checks of it

Codex (`gpt-6.1-sol`, xhigh, read-only): prompt `prompt_outline_v3_261112.md`, report `codex_outline_v3_261112.md`. Agent review, not a human review. Points I confirmed in the files; all apply to v4 as well and go into the next version.

| Point | My check | Fix |
|---|---|---|
| Pod B range is 0.82 to 0.91, not 0.92 | `D2.json` max 0.91495 | Round correctly |
| E3 controls use state 17; only pod B's use state 16 | E3 `part2_context_run_main.json` `_meta.layer` = 17 | Say which run uses which state |
| Direct base-to-SFT map change is 0.0032 | `pod_a.stage_change` | "about 0.003" |
| Planted-shift sanity checks the signed score; the reported statistic is absolute, so a move through zero is not seen (pod A: -1.07 to +0.94 gives 1.07 to 0.94) | `check_role_count.py` | Report the signed position against a fixed axis; drop detection guarantees until a simulated power check on the planned paired test |
| "Second separate build" reuses the same statistic functions | `check_hidden_prompt.py` imports `position`, `spread`, `cell_sums` from `check_role_count.py` | "a second script with the same statistic functions", not independent |
| Lu App. B.3 already places the default at one end when base Gemma reads Instruct rollouts | `2601.10387.txt:1352-1358` | Q4 is an extension of Lu, not "no prior test" |
| RankMe is spectral dimensionality, not total variance | Li et al. definition | Fix the Li sentence; my ratio cancels a common scale change only |
| Entropy (Asvin G. and Lindsey) and realised-token NLL (Nik Khah and Bahreini) are different measures | Both papers | Use "output entropy" and "surprise (NLL)" separately |
| "No run uses example turns" is false: E3 has example conditions | E3 NOTE | Restrict to pods A, B, C |
| Behaviour uses three judges (GPT-5.6-Sol and Claude Sonnet 5.5 for naming, Terra for role play) | E3 NOTE; judge file | Name them |
| E3 prompt alone 91.8% is higher than with the reply 52.3% | E3 context JSON | Report the numbers; drop "about as well" |
| Pod C readers' behaviour is not measured | pod C design | Pairing is claimed only where both measures exist |
| Contrary evidence missing: pod B reply-only states beat text readers in interview (+0.071 [0.006, 0.131]) and script (+0.076 [0.012, 0.131]); a 2-of-20 role move is not detected (0.156 [-0.006, 0.340]); base-written E3 text gives base-Instruct map r 0.653 [0.187, 0.693], not interpreted (reliability floor fails); the ex2 naming pass is fragile; two of the judge's three extra base yes labels are my "cannot score" cases; Lu's PC1 fails to mark the default in base Llama 3.1 70B | `C2_D2_report.json`; pod C `g5` and `e3_desc.D1_report`; E3 naming files; owner labels; Lu App. G | Add a contrary-evidence item to each claim |
| Bold "Contrary evidence" label is outside the allowed labels | v3 text | Use Limits |
| Nguyen: peak near 0.9 depth; Qwen 2.5 7B +0.235 is not small; the Think pipeline lacks the effect | `2609.23935.txt` | Fix wording |
