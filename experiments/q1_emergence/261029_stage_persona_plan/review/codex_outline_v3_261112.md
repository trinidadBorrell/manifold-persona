**Claim:** The draft needs correction. Most copied values match, but the absence claims, power claims and several literature comparisons do not hold.

I used local reads and JSON checks. I did not edit files, read activation arrays, rerun analyses or make paid calls. Source aliases resolve in the read-depth table.

---

Task 1 — Numbers, intervals and populations

| Item | Finding | Evidence | Severity |
|---|---|---|---|
| Pod B full-context range | Exact range: **0.82325–0.91495**. Ordinary two-decimal rounding gives 0.82–0.91, not 0.82–0.92. This is a minor rounding error. | `B/D2.json`, `formats.*.accuracy.point` | wrong |
| Pod B controls | The quoted ranges match: prompt control **0.90950–0.98165**; description-removed replies **0.42585–0.46960**. These are ranges across four formats, not intervals. There are 8,000 prompt controls and 23,999 valid reply records. | `B/sensitivities.json`, `context.{C2_state16,C3_state16}.{decoding.formats,prompt_count,valid_replicates}` | ok |
| E3 decoding | All three estimates and quoted intervals round correctly. Population: **1,800 description-only replies**, 20 roles. The reply-only lower bound is **9.9797%**, just below the 10% floor. | `E/results/part2/part2_context_run_main.json`, `desc.{n_replies,R,c2,c3}` | ok |
| “Base controls…use hidden state 16” | This applies to pod B. E3’s cited controls use **state 17**. | `O:24`; E3 context JSON, `_meta.layer=17`, `desc.layer=17` | wrong |
| Pod C population and decoding | **20,000 role replies**, excluding 1,000 default replies, is correct. Ten readers are recorded. Text **0.796709**, base states **0.760760**, Instruct states **0.753976**; chance **0.05**. | `C-results`, `checkpoints`, `pod_a.decoding.*.accuracy`; `C/THRESHOLDS.md:16` | ok |
| Pod C map numbers | All quoted values round correctly: base–Instruct **0.895896 [0.766924, 0.932136]**; character map **0.845756**; raw agreement advantage **0.050139 [0.013786, 0.102931]**. The last number measures agreement with the Instruct map. | `C-results`, `pod_a.D1.{r,text_anchors.text_char,minus_best_anchor}` | ok |
| “Changes…at most 0.003” | Direct base→SFT change is **0.00323548 [0.00101471, 0.00772629]**. The **0.00272449** value measures increased agreement with final Instruct. The control difference is **0.02658499 [0.01245972, 0.03868074]**. | `C-results`, `pod_a.stage_change["base->sft"].change`, `pod_a.D2.transitions`, `g5.equal_map_wording_case_recorded.d2_difference` | wrong |
| Pod A repeatability | **0.900864 [0.796555, 0.939328]** matches. The saved result lacks fidelity labels. | `A-results`, `primary.D2.raw_r`, `fidelity.available=false` | ok |
| E3 blind naming | **14.9% [9.5, 21.4]**, **17.2% [11.6, 23.6]**, and shuffled means **2.393% / 2.898%** match. These are macro scores over 20 selected target roles, with 276 candidate names. | `E/results/naming_v2.json`, `run_main/desc.results.none.{gpt,claude}.all_replies` | ok |
| E3 examples | **37.6% / 42.3%** match. Donor naming **24.8–29.8%** and described-role naming **2.4–5.0%** also match. | Same JSON, `run_main/ex2`, `run_main/ex2_other/shift{6,10,14}`, `donor_hit5` | ok |
| Fresh owner labels | Owner yes **3/20 base**, **16/20 chat**; judge-positive agreement **1/4 base**, **13/13 chat** all reproduce. Two base disagreements have owner class **U**, meaning unscorable. | `Owner` labels and key; matched `pop_poda` and `pop_podb` judge labels; `owner_review/NOTE.md:22` | ok |
| Practice judge | **31/40 = 77.5%** agreement and **16/17** precision reproduce. Overall recall is **16/24**. | `J/results/dev_judge_choice.json`, `candidates["openai/gpt-6.1-terra"]` | ok |
| Pod B judge rate | **3,689/23,999 = 15.3715%**, correctly rounded to 15.4%. It is an uncorrected judge rate. | `J/results/pop_podb_judge_rates.json`, `cells.base` | ok |
| Default position | Pod A **1.073092**; pod C **1.084880, 1.076538, 1.081223, 1.080889**; text **0.996372**. All quoted rounding matches. | `F/results/check1_poda.json`, `point.position_abs`; `check2_readers.json`, `results.*.point.position_abs` | ok |
| Default variance | Log-ratios **0.110980–0.122240** mean **11.74–13.00%** more variance. Every reader’s interval is above zero. Text **0.582646** is a log-ratio: approximately **79.08%** more variance. | `F/results/check2_readers.json`, `results.*.{point.spread_log_ratio.all,spread_all}` | ok |
| “Stage changes…within 0.02” | The paired medians meet this description. But some spread intervals exclude zero: DPO−SFT **−0.011354 [−0.023061, −0.004485]**; Instruct−DPO **−0.001031 [−0.002081, −0.0000619]**. | Same JSON, `paired.*.spread_all` | ok |
| Detectability: 0.85 and 0.6 | The 20-role formula outputs **0.828917–0.845274**. The saved 40-role outputs **0.594145–0.608699** are extrapolated interval widths. Neither establishes the promised detection guarantee. | `F/results/check1_readers/*.json`, `detectable_change_between_two_stages["20"]`, `extrapolation_1_over_sqrt_n.position_abs.width_at_40` | overstated |
| “Separate build…same numbers” | State spread points match. Text does not: **0.582646 versus 0.593897**. Both scripts also import the same statistic implementations. | `F/results/check{2_readers,2}.json`; `F/check_hidden_prompt.py:21` | overstated |
| Lu’s numbers | PC cosines **0.93, 0.87, 0.83** and edge distance **0.03** appear in the source. But 275 role names produce **377–463 role/category vectors** for PCA. Do not imply 275 unfiltered role means or 0.03 separately verified for every model. | `Lu:139`, `Lu:149`, `Lu:239`, `Lu:1354` | missing |
| Geology’s 94% | Correct: variance explained in coordinates of **34 held-out adapters**, after fitting on 100 markers. | `Geology:128` | ok |
| Han’s +0.19 and p | The prose reports **+0.19** and **p=0.066**. Embedded projection data give **+0.183217 over 48 desirable roles**. The p-value tests the desirable-minus-undesirable change gap. | `Han:74`, footnote 1; line 102 datasets | missing |
| Nguyen’s numbers | **+0.31**, **about +0.60**, and Llama 8B **+0.078** match. The +0.31 peak occurs around **0.9 normalized depth**. Qwen 7B gives **+0.235**, which limits “small at 7 to 8B.” | `Nguyen:219`, `Nguyen:220`, `Nguyen:473` | overstated |
| Asvin G. and Lindsey | The **32B** stage sequence and near-absence at approximately **2B** match. The measured quantity is output entropy. | `G&L:241`, `G&L:278` | ok |
| Nik Khah and Bahreini | **+0.434, −0.121, −0.567, −0.581 nats/token** match. These are final-reader NLL minus base-reader NLL, grouped by writer stage. | `Twin:217`; definition at `Twin:125` | ok |
| “Five definitions” | The draft neither lists five definitions nor maps them to sources. The count is unsupported as written. | `O:37` | missing |
| Prompt counts and math questions | I checked **24,000** pod B prompts: one start token, no chat tags. All **21,000** pod A prompts match the official single-turn template shape and question text. None of the selected **40** questions asks for a numerical math solution; the card names `RLVR-GSM`. | A/B `run_full/rows.jsonl`; tokenizer JSON; `E/questions_40.json`; selected question-bank records; `Card:23` | ok |
| Historical 1B audit | **153/276 = 55.4348%** and **8/276 = 2.8986%** match. Each panel includes **275 characters plus default**. | `Old/judge_gpt.json`, `olmo1b\|rlvr*`; `Old/KEY_char.json` | ok |
| GPU estimates | Two A100 hours, three GPU hours, minutes per stage and five total hours are estimates. The outline supplies no benchmark calculation for either role count. | `O:64`, `O:66`, `O:67`, `O:69` | missing |

---

Task 2 — Claim strength

| Item | Finding | Evidence | Severity |
|---|---|---|---|
| Claim 1: “most of the role signal…comes from…” | “Narrow” does not fix the attribution claim. Decoder accuracy and map correlation do not measure a fraction of activation information caused by prompt versus reply. | `O:8`, `O:18`, `O:49`; quoted decoder and map results above | overstated |
| “Prompt alone about as well” | The prompt control scores substantially higher, especially in E3: **91.8% versus 52.3%**. Say what the controls score. No equivalence test establishes “about the same.” | E3 context JSON, `desc.{R,c3}` | overstated |
| Claim 2: weak base role play | Low blind naming and owner **3/20** support a cautious claim about these prompts and samples. Hit@5 is not the proportion of replies that actually enact a role. | `O:28`; E3 naming JSON; owner labels | ok |
| “Structure shows the role in most base replies” | A classifier’s aggregate accuracy does not establish persona presence in individual replies. The fresh owner sample and E3 decoder population also differ. | `O:31`; E3 context JSON, `desc.n_replies=1800`; owner sample design | overstated |
| “Throughout, I pair” behaviour and structure | Pod C measures readers’ internal states on supplied replies. It does not measure those readers’ behaviour on the same items. Text classification is also not observed role enactment. | `O:10`; `C/THRESHOLDS.md:11` | overstated |
| Claim 3’s exploratory label | This label is appropriate. The broad title is not: two statistics at state 17 do not test every way the default could be special. All position intervals lie below the chosen 2-SD boundary, and variance is higher than the median role. | `O:34`, `O:36`; `F/NOTE.md:21`; check 2 results | overstated |
| “No stage change”; Q3 “Answered: yes”; Q8 “no change at RLVR” | These turn limited measurements into absence claims. The equal-map control invalidates a simple interpretation of small corrected differences. It does not prove equality. | `O:24`, `O:50`, `O:55`; `C/THRESHOLDS.md:76`; `C-results/g5` | overstated |
| Math training explains little RLVR change | Dataset and question facts check out. The implied explanation remains untested. Non-math evaluations also change after RLVR. | `O:55`; `Card:124`, `Card:127` | overstated |
| “No run uses example turns” | False across the runs discussed. E3 explicitly includes same-role, donor-role and filler-example conditions. Restrict this sentence to the new pod A/B generation protocol. | `O:76`; `E/NOTE.md:14`; E3 naming condition keys | wrong |
| “Behaviour…rests on one judge” | The outline uses GPT-5.6-Sol, Claude Sonnet 5.5 and Terra, under different judging tasks. State those differences. | `O:85`; `E/NOTE.md`, “Part 1 results”; `J/results/dev_judge_choice.json` | wrong |

---

Task 3 — Check 1 and check 2 code

| Item | Finding | Evidence | Severity |
|---|---|---|---|
| Position statistic | It fits centred, roles-only PC1 on one question half. It evaluates default displacement, standardized by role-projection SD, on the other half. This matches the note. | `F/check_role_count.py:98` | ok |
| Within-wording spread | It subtracts each cloud/wording mean and computes pooled mean squared residual distance. The log-ratio uses default over median role variance. | `F/check_role_count.py:89`, `F/check_role_count.py:147` | ok |
| Crossed draws | Roles and questions are resampled. Check 2 shares each draw across readers and text, then computes paired differences. Wordings and observed replicates remain fixed. | `F/check_role_count.py:111`; `F/check_default_readers.py:33`, `F/check_default_readers.py:81` | ok |
| Planted-shift check | It checks the signed score, while the reported statistic is absolute. In pod A, the 2-SD plant changes signed position **−1.073→+0.942**, but absolute position **1.073→0.942**. That does not validate detection of a 2-SD movement. | `F/check_role_count.py:157` versus `:186`; `check1_poda.json`, `planted_shift_sanity` | wrong |
| Power calculation | `2.8 × √2 × SE` assumes equal, independent errors and a suitable normal approximation. There is no simulated rejection rate, paired covariance calculation or power test for the planned own-reply experiment. | `F/check_role_count.py:172` | overstated |
| Forty-role extrapolation | The fit forces width proportional to `1/√n`, using 10, 15 and 20 roles from one selected panel. It neither observes 40 roles nor estimates an uncertainty floor. | `F/check_role_count.py:151`, `F/check_role_count.py:165` | overstated |
| Text baseline | Character 3–5-gram TF-IDF and 256-dimensional SVD match the note. Both are fitted once on all rows. The bootstrap therefore conditions on that fitted text representation. | `F/check_default_readers.py:25`, `F/check_default_readers.py:78` | ok |
| Row alignment, current data | No mismatch found. All **21,000** pod A row keys equal `grid.keys` in order. Pod C’s reader uses grid order; each checked reader metadata file reports 21,000 rows. | A `rows.jsonl`, `grid.json`; `C/replies.py:59`; `F/check_role_count.py:40` | ok |
| Row-alignment guard | The text features stay in file order, while metadata are reordered. The assertion checks role names only. A future within-role reorder could silently mislabel questions, wordings or replicates. Current files do not trigger this defect. | `F/check_default_readers.py:28`, `:62`, `:74`, `:79` | missing |
| “Second, separate build” | It reuses `position`, `spread`, `cell_sums`, `question_weights` and `interval` from check 1. This checks the surrounding script, not an independent implementation of the statistics. | `F/check_hidden_prompt.py:21` | overstated |
| Text-baseline decision rule | The note requires a reader interval to clear the text interval. The saved `reading` flags do not implement that condition. Current position flags are false, so this does not create the present headline result. | `F/NOTE.md:20`; `F/check_default_readers.py:93` | missing |
| Length control | It selects replies stopped at the 128-token limit. Those selected replies all have 128 kept tokens in the checked rows. Selection still changes the question/reply population between clouds. | `F/check_role_count.py:126`; A/B row fields `stop_reason`, `kept_ids` | ok |

---

Task 4 — Related work

| Item | Finding | Evidence | Severity |
|---|---|---|---|
| Lu: “supports and extends” / “repeats” | Fair as a partial replication. Lu compares PC directions and same-role vectors, using judged role-expression categories. Your distance-rank maps and unfiltered replies differ. | `Lu:139`, `Lu:149`, `Lu:1352`; `C/THRESHOLDS.md:16` | missing |
| Lu: “No direct prior test found” | Wrong for default position on fixed Instruct text. Appendix B.3 explicitly places the default at one end when base Gemma reads Instruct rollouts. A four-stage sweep is an extension. | `O:51`; `Lu:1354` | wrong |
| Han: “parallel” | Correct. He studies own generations across SFT, DPO and RLVR, in another model. He retains fully role-playing replies. His result does not isolate default movement from role movement or changing selection. | `Han:56`, `Han:60`, `Han:64`, `Han:74` | ok |
| Geology: “same direction” | Reasonable analogy, not confirmation of your activation claim. Its 94% result concerns adapter coordinates. It also reports behavioural effects from weight-distant adapters and behavioural weakness at rank 1 despite preserved position. | `Geology:128`, `Geology:134`, `Geology:281` | missing |
| Nguyen | The broad summary is recognizable, but “mid-layer” obscures a peak around 90% depth. Add that the Think pipeline lacks the effect, and open-user-turn RLVR increases it. “Small at 7–8B” needs qualification. | `Nguyen:145`, `Nguyen:147`, `Nguyen:220`, `Nguyen:473` | overstated |
| Asvin G. and Lindsey | The stage summary is faithful. Name output entropy explicitly. Their cross-model self-text advantage and matched-persona comparison are distinct tests; default-versus-role entropy is already studied. | `G&L:116`, `G&L:278`, `G&L:320` | missing |
| Nik Khah and Bahreini | The numbers match, but calling them a stage’s own-text familiarity obscures the design. Two fixed readers score stories from different writer stages. The paper warns that self-scoring widens endpoint gaps. | `Twin:125`, `Twin:217`, `Twin:236` | overstated |
| Li | “Overall representation spread” is the wrong object. RankMe measures normalized spectral dimensionality, not total variance. Your variance ratio cancels a common scalar scale change; it does not cancel spectral reshaping. | `Li:207`, `Li:223`, `Li:688`, `Li:717`; `F/check_role_count.py:89` | wrong |
| Experiment 3’s literature link | It mixes output entropy with realized-token NLL under “familiarity.” These require different scores and controls. Keeping an output layer does not specify which test will run. | `O:54`, `O:66`; `G&L:116`; `Twin:107` | missing |

---

Task 5 — Missing contrary evidence

| Item | Finding | Evidence | Severity |
|---|---|---|---|
| Residual signal after removing descriptions | Pod B reply-only states exceed tested text readers in interview and script: **+0.07125 [0.00644, 0.13116]** and **+0.07575 [0.01154, 0.13054]**. These are exploratory and do not prove information beyond text. They belong beside “mostly prompt and text.” | `B-C2`, `formats.{interview,script}.gain_over_best_text` | missing |
| Small role-map changes can be missed | Moving two of twenty roles produces true map correlation **0.833498**, yet the reported change interval includes zero: **0.155922 [−0.006023, 0.340101]**. | `C-results`, `g5.planted_change_recorded["2_of_20_roles_moved"]` | missing |
| Base-written text is a weaker case | For description-only base replies, base–Instruct map agreement is **0.653159 [0.186709, 0.693163]**. Both reliability lower bounds fail the floor; `interpreted=false`. | `C-results`, `e3_desc.D1_report` | missing |
| Example-assisted naming is fragile | The replicate’s cleaned GPT lower bound is **0.198204**, below 0.20. All four additional broken-reply masks make `plays_ex2=false`. The outline gives only the attractive main-run rates. | `E/results/naming_v2.json`, `run_rep/ex2.results.none.gpt.clean`; `Broken`, `primary_decisions.plays_ex2` | missing |
| Fresh base judge disagreements | Two judge-positive disagreements are owner-U replies. “Matched mine on 1/4” is correct. “Right on 1/4” treats unscorable cases as settled truth. | Owner items **2 and 37**, owner class `U`; corresponding judge `passed=true` | missing |
| RLVR can affect other tasks | The same model card reports AlpacaEval **27.9→29.1** and IFEval **73.0→72.3** from DPO to Instruct. Math-only training does not establish math-only effects. | `Card:124`, `Card:127` | missing |
| PC1 need not locate the default at an extreme | Lu reports this failure for Llama 3.1 70B. This directly limits any universal interpretation of your PC1 check. | `Lu:2831` | missing |
| Structure does not uniquely determine behaviour | Geology reports stage-one/stage-two weight cosine **0.000**, alongside activation-shift cosine **0.53**. Its rank-1 example preserves position while barely changing the reply. | `Geology:134`, `Geology:287` | missing |
| Historical tightness claims need an explicit boundary | The vault records that example turns reversed earlier default-tightness findings. Those withdrawn runs cannot repair the current small-panel result. | `Vault`, “Reported in July–August”; `Review`, F3 and C105 | missing |

---

Task 6 — Style

| Item | Finding | Evidence | Severity |
|---|---|---|---|
| Bold restriction | `Contrary evidence` is outside the permitted labels. Put all parenthetical details outside bold; bold only `Evidence`. | `O:20–23`, `O:30–32`, `O:38` | wrong |
| Vague attribution | “Recent work shows the two can disagree” needs the named study and its actual two measurements. | `O:10` | missing |
| Vague importance language | “Becomes special,” “sets…apart” and “privileged” exceed the two stated operational tests. Use the metric names in the headlines. | `O:7`, `O:9`, `O:34`, `O:37` | overstated |
| Term switching | “Familiarity” covers both entropy and NLL. “Names the role” covers both a classifier and blind top-5 judging. Use distinct, stable terms. | `O:22`, `O:30`, `O:54`, `O:66` | missing |
| Other banned tropes | I found no clear negative-parallelism slogan, forced rhetorical triplet, punchy-fragment sequence or stakes inflation. The donor-versus-described-role contrast reports a real comparison. | Full outline | ok |

---

Five most important fixes

1. Replace “same map” and “no stage change” with the measured correlations and their limits.
2. Remove detection guarantees until the absolute statistic and planned paired test have calibrated power.
3. Replace “most signal comes from” with the exact prompt-control and text-baseline findings.
4. Correct the Lu novelty label, Li metric and entropy-versus-NLL distinction.
5. State judge populations, unscorable cases and the fragile example-assisted results.

---

Read depth

`Q` = `/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence`  
`V` = `/Users/brendon/research-log/manifold-persona-log/Reference`

| Alias / source | Read depth |
|---|---|
| [O — draft outline](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_stage_persona_plan/draft_outline_thomas_v3_261112.md) | Full, 117 lines. |
| [Proof ledger](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_stage_persona_plan/review/proof_ledger_261112.md) | Full, including the earlier review and owner-review sections. Used as pointers, not proof. |
| `F = Q/261112_free_checks` | Full `NOTE.md`, `check_role_count.py`, `check_default_readers.py`, `check_hidden_prompt.py`. Full check 1 pod A JSON, four `check1_readers` JSONs and both check 2 JSONs. |
| `C = Q/261029_fixed_text_read_through` | Full `NOTE.md`, `THRESHOLDS.md`, `replies.py`. Reader metadata: `rows`, `fed`, `layers`, `g1_rows_checked`. |
| [C-results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/records_full/analysis_full_v2/results_full_v2.json) | Selected keys: `state`, `checkpoints`, `g5`, `pod_a.{D1,D2,stage_change,decoding,axis,identifiable_subset,components_70}`, `e3_desc.D1_report`. |
| [A-results](</Users/brendon/research-log/manifold-persona-log/Reference/2026-10-30 A-B pilot results/full_A_results.json>) | Selected keys: `primary.D2`, `fidelity`, `D3_axis_gate`, `limits`, provenance/model metadata. |
| `B = V/2026-11-04 B Modal recovery/full_results/analysis_full_batch_v3_cuda_261105` | `D2.json`: all format accuracy records. `sensitivities.json`: `context.C2_state16` and `C3_state16` populations, format accuracies and scope. |
| [B-C2](</Users/brendon/research-log/manifold-persona-log/Reference/2026-11-05 B C2 decoding/C2_D2_report.json>) | `scope`, `n_perm`, every format’s accuracy, text gain, full-context drop and p-value. |
| `E = Q/261008_base_persona_map` | Full `NOTE.md`. JSON: cited naming conditions, both judges’ scores, intervals, null means, donor scores, cleaned replicate results and decisions; context `desc` and metadata. |
| `J = Q/261029_judge_validation` | Full `NOTE.md`, `THRESHOLDS.md`, `dev_judge_choice.json`. Pod B rate JSON: aggregate and format/role records. |
| `Owner = /Users/brendon/persona-manifolds/judge_validation_261029` | Full owner-review note and all 40 owner label records. Matched all 40 against keys and saved judge decisions. Read exact inputs and kept replies for items 2, 5, 14, 17, 35 and 37. |
| [Broken](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/results/naming_broken_masks.json) | `settings`, `primary_decisions`; no full raw-case audit. |
| `Old = output/results_2026-09/audit_2026-10-02` | Full `judge_gpt.json`; selected `KEY_char.json` fields for population counts and default inclusion. |
| A/B generation JSONL and tokenizer JSON | All row keys, prompt IDs and length-control fields. Decoded all prompts. Compared all A keys with grid order. Read selected tokenizer vocabulary, decoder and template fields. |
| [Lu](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2601.10387.txt), [G&L](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2605.25459.txt), [Nguyen](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2609.23935.txt), [Li](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2509.23024.txt), [Twin](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2609.32196.txt) | Full saved texts, including appendices and references. No separate figure-image inspection. |
| [Han](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/lw/axis_shift_stages_clean.md) | Full prose and footnotes. Embedded JSON: line 102 projection datasets `22c4…`, `e47f…`; line 163 pass-rate datasets `8cd6…`, `9ec1…`. Used `role`, `label`, `SFT`, `DPO`, `RLVR`. |
| [Geology](/private/tmp/claude-501/-Users-brendon-persona-manifolds-manifold-persona/ad361f36-2648-47d6-95a7-4e43e5c9a47c/scratchpad/geology/geology.md) | Full saved text, 307 lines, including caveats and appendix. |
| [Card](/Users/brendon/.cache/huggingface/hub/models--allenai--OLMo-2-1124-7B-Instruct/snapshots/470b1fba1ae01581f270116362ee4aa1b97f4c84/README.md) | Full cached model card. All 40 selected question texts read. |
| [Vault](</Users/brendon/research-log/manifold-persona-log/Reference/Claims and verdicts.md>), [Review](/Users/brendon/persona-manifolds/manifold-persona/output/independent_review_2026-10-02/review.txt) | Both full. Historical results remain historical. |
| Unchecked | Activation arrays, remote artifact integrity, judge API outputs regenerated from scratch, power coverage and GPU-cost estimates. The repo/vault search was bounded, not exhaustive. |