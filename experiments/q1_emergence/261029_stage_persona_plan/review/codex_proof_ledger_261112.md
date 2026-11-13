**Verdict: most copied numbers match. C5 is mislabelled, a known failed control is omitted, and the novelty claim is too broad.**

I checked local files and code. I did not rerun analyses, inspect large activation arrays, or verify remote copies. No files were changed.

Short source names below resolve to exact paths in the read-depth table.

---

**1. Numbers, populations and wording**

| Claim | Finding | Evidence | Severity |
|---|---|---|---|
| C1 | Correct. Base: 3 true positives, 0 false positives, 5 misses. Instruct: 13 true positives, 1 false positive, 3 misses. Total agreement: **31/40**; precision: **16/17**; recall: **16/24**. | `J-choice`, `candidates["openai/gpt-6.1-terra"]` | ok |
| C2 | The quotation exists in the cited note. That verifies the written record, not the original utterance. The note says recordings were not checked. | `Meetings:17`, `Meetings:73` | missing |
| C3 | The four format scores match. Exact ranges: C3 **0.90950–0.98165**; C2 **0.42585–0.46960**. These are ranges across formats, not confidence intervals. C3 has 8,000 prompt controls; C2 has 23,999 valid reply records. | `B-sens`, `context.{C2_state16,C3_state16}.decoding.formats.*.accuracy`; corresponding counts | ok |
| C4 | Quoted correlations and intervals match. But **21,000 is the input-run size**: the role maps exclude 1,000 default replies. The primary F1 input retains the question and interview frame. Draft “with no prompt” is false. | `C-results`, `pod_a.D1`, `pod_a.D2.per_stage.base`; `C-rules:16–20`; `Draft:47` | missing |
| C5 | **0.00272449 measures increased agreement with final Instruct**, not direct base→SFT map change. Direct change is **0.00323548 [0.00101471, 0.00772629]**. Direct SFT→DPO change is **0.000912763 [0.000396921, 0.001838965]**. The planted value **0.15592168 [−0.00602321, 0.34010057]** matches. | `C-results`, `pod_a.D2.transitions` versus `pod_a.stage_change`; `g5.planted_change_recorded` | wrong |
| C6 | All quoted rates and intervals match. These are macro naming scores on **20 selected roles × 18 shared questions × 5 wordings = 1,800 replies per condition/run**. They are not results over 276 target roles. | `E3-results`, four `run_*/{desc,ex2}.results.none.{gpt,claude}.all_replies`; `E3-rules` | ok |
| C7 | Correct: **153/276** without examples; **8/276** with examples. Both intervals match. Each panel contains **275 characters plus default**. The role/question/wording keys match between conditions. | `Old-judge`, `olmo1b|rlvr*`; `Old-key`, matching condition records | ok |
| C8 | Correlation, interval and null percentile match: **0.900864 [0.796555, 0.939328]**, null p99 **0.220024**. Again, 21,000 includes default; the role map uses 20,000 replies. | `A-results`, `primary.D2`; `C-rules:16–17` | missing |
| C9 | Correct: **0.880673 [0.762746, 0.883963]**; assistant-control cosine **0.936209**. `precision_met=false` and `passed=false` are correctly acknowledged. | `A-results`, `primary.D3`, `primary.assistant_control` | ok |
| C10 | Quoted scores and intervals match. Judge-specific null p99 values are **0.069335** and **0.0680833**. The panel scores **12,000 role replies**, using replicates 0–2. The role-only approximation requires **41 roles for Terra**, **40 for Sonnet**; “about 40” is reasonable, not a verified precision guarantee. | `A-miss`, `D1.panel`, `D1.judges`; sample SDs **0.243379**, **0.241372** | missing |
| C11 | Correct for the cached card and selected questions. The card names `RLVR-GSM`. None of the 40 questions requests a numerical math solution. | `Card:10`, `Card:23`; `Q40`; `Q-bank`, selected `question` values | ok |
| L1 | The models and steering result match the source. “Mainly inherits” is the authors’ suggested interpretation, not a demonstrated training history. | `Lu:438–471` | ok |
| L2 | The prose reports **+0.19 [0.14, 0.23]** and **p=0.066**. But the embedded figure gives **+0.183217**, across **48 desirable roles**; translator is absent from the SFT→DPO plot. The 275-role mean pass rates do reproduce: **0.256392, 0.520092, 0.525134**. The prose/data discrepancy needs disclosure. | `Han:74`; line 102, figure datasets; line 163, `SFT`, `DPO`, `RLVR`, `label` | wrong |
| L3 | Faithful for OLMo-3-32B. The measured quantity is **output-distribution entropy**. The draft repeatedly substitutes **surprise**, which is a different quantity. | `G&L:280–290`; `Draft:30`, `Draft:94` | overstated |
| L4 | “True only inside the middle layers” is false. The source reports base output bias **+0.10**, against an internal peak **+0.31**. That peak occurs around **0.9 normalised depth**. A flat base→SFT trajectory does not mean zero output bias. | `Nguyen:473–488`; `Ledger:32` | wrong |
| L5 | It predicts **94% of coordinate variance across 34 held-out adapters**. It does not correctly place “94% of adapters.” The 98% comparator and 0.86 matrix correlation match. Five factors were selected using judgement calls. | `Geology:84–90`, `Geology:128`, `Geology:249` | wrong |
| L6 | The Tülu rank trend is correctly reported. **“OLMo-2 only in pretraining” is false**: the paper also performs OLMo-2-1B SFT experiments. | `Li:525–539`, `Li:688–706`, `Li:1180`, `Li:1206` | wrong |

---

**2. Evidence the ledger missed**

| Claim | Finding: does the additional evidence discount it? | Evidence | Severity |
|---|---|---|---|
| C1 | **Yes, for judge validation beyond these labels.** The same 40 labels selected the judge. The pass rule also changed after seeing practice results. The counts remain correct. | `J-rules:57`, `J-rules:60`, `J-rules:104` | missing |
| C2 | **It limits provenance.** This is an agent-written secondary record, explicitly unchecked against recordings. It cannot independently establish exact spoken words. | `Meetings:16–19` | missing |
| C3 | **It limits the “mostly prompt” story.** C2 still beats tested text readers in interview and script: gains **0.07125 [0.00644, 0.13116]** and **0.07575 [0.01154, 0.13054]**. This does not reverse C3>C2. | `B-C2-report`, `formats.{interview,script}.gain_over_best_text` | missing |
| C4 | **Yes, for “agree up to noise.”** Equal complete maps can produce corrected agreement **1.026585 [1.012460, 1.038681]**. The correction does not isolate noise. Nor does a correlation quantify the fraction of representation “explained by text.” | `C-results`, `g5.equal_map_wording_case_recorded.r_norm.value`; `Draft:8`, `Draft:47` | overstated |
| C5 | **Yes, decisively.** The equal-map control produces a nonzero corrected difference, **0.026585 [0.012460, 0.038681]**. The original note explicitly says these stage differences are not evidence of map movement. | `C-results`, `g5.equal_map_wording_case_recorded.d2_difference`; `C-note:93`; `C-rules:72–76` | wrong |
| C6 | **Yes, for “plays roles.”** Description-only fails the frozen decision. All four broader broken-reply masks also make ex2 fail. Owner labels already exist; “no owner transcript read yet” needs clarification. | `E3-results`, `primary_decisions`; `Judge-design-note:98–104`; `E3-note:95` | missing |
| C7 | **It discounts the mechanism claim.** A naming-score drop shows an example-turn effect. It does not establish that the model copies examples. | `Old-judge`; `Old-score`; unsupported mechanism at `Draft:56` | overstated |
| C8 | **It discounts persona identity, not repeatability.** The saved result says fidelity labels are unavailable. The repeating map includes all valid outputs, without verified role enactment. | `A-results`, `fidelity.available=false`; `provenance.analysis_scope` | missing |
| C9 | **It limits robustness across default prompts.** Keeping all default wordings on both sides changes the cosine from **0.880673** to **0.944155**. This is a different descriptive contrast, not a repaired primary pass. | `A-axis`, `all_default_wordings_on_each_side`; `limits` | missing |
| C10 | **It narrows interpretation.** Removing self-named replies lowers naming to **0.654958 / 0.654436**. Also, invalid-answer bounds are explicitly report-only because frozen rules did not specify this handling. Both bound decisions agree; that does not make the handling prespecified. | `A-checks`, `macro_no_self_named`; `A-bounds:1–3` | missing |
| C11 | **It discounts the proposed explanation, not the facts.** The same card reports non-math score changes after RLVR: AlpacaEval **27.9→29.1**, IFEval **73.0→72.3**. Math training does not imply math-only effects. | `Card:124`, `Card:127`; `Draft:86` | missing |
| L1 | **It limits a universal PC1 story.** In Llama 3.1, PC1 did not clearly identify Assistant-like personas, and default did not occupy one end. This does not negate the contrast-vector steering result. | `Lu:2830–2834` | missing |
| L2 | **It limits representation-only attribution.** Role-play selection changes strongly across stages. The activation maps therefore use different selected generations. Stage, output content and selection change together. | `Han:52–55`, `Han:131–137`; line 163 pass-rate data | missing |
| L3 | **It limits extrapolation.** The effect is essentially absent near 2B. The stage experiment uses 20 prompts under three formats; it does not establish an advantage on “all text.” | `G&L:243–249`, `G&L:1165–1172`; `Draft:33` | overstated |
| L4 | **It limits a general stage rule.** The Think pipeline does not install the bias. In the open-user-turn condition, RLVR adds an increase. | `Nguyen:141–161` | missing |
| L5 | **It discounts geometry→behaviour inference.** A rank-1 adapter retains its chart position while copying the base answer’s first **634 characters**. Similar position need not imply similar behavioural strength. | `Geology:132–139` | missing |
| L6 | **It limits causal and persona claims.** The authors call their findings primarily correlational. Global covariance rank does not measure whether default replies form a tighter cloud than role replies. | `Li:180–205`, `Li:793–801` | missing |

---

**3. Code paths for C3–C5**

| Claim | Finding | Evidence | Severity |
|---|---|---|---|
| C3 | **Held-out format is implemented correctly.** Training uses three formats, 20 questions and three wordings. Testing uses the fourth format, disjoint questions and two other wordings. Scaling fits training rows only. C2 retains a role-free question/frame; C3 pools the fixed continuation after the full prompt. | `B-analysis:282`, `B-analysis:316`, `B-analysis:328`, `B-analysis:1311`, `B-analysis:1453`; `B-context:71`, `B-context:81` | ok |
| C4 | The code computes cross-half Spearman agreement, then **r / √(reliability₁ × reliability₂)**. It applies reliability floors. This matches the saved numbers. Calling it complete “noise correction” exceeds what the code and equal-map control establish. | `C-maps:155`, `C-maps:177`, `C-maps:229`; `C-analysis:187`; `C-rules:74` | overstated |
| C5 | **The code separates the measures that the ledger conflates.** D2 subtracts agreement-to-Instruct scores. `stage_change` computes **1−corrected pairwise agreement**. The planted result uses the latter, with two role vectors replaced, one synthetic seed and 200 bootstrap draws. It is an example, not a power curve. | `C-analysis:214`, `C-analysis:230`, `C-analysis:410`, `C-analysis:456–465` | wrong |

---

**4. Related-work faithfulness**

| Claim | Finding | Evidence | Severity |
|---|---|---|---|
| L1 | **Neither** is correct for your unrun base-axis test. But Lu also directly precedes C4: base Gemma reads Instruct rollouts. PC cosines are **0.93, 0.87, 0.83**; same-role vector cosines exceed **0.99**. Omitting this hides the closest precedent. | `Lu:1352–1362`; `C-results`, `pod_a.axis` | missing |
| L2 | The prose summary is broadly faithful. **Qualitative consistency**, not direct support, is the right comparison with C5. Also, **p=0.066 tests the desirable-versus-undesirable change gap**; it is not the p-value for the desirable group’s increase alone. | `Han:74`, `Han:219–221`; `Draft:78` | overstated |
| L3 | Faithful when called an **entropy advantage**. **Neither** directly supports nor contradicts your activation-map result. RLVR widening entropy differences does contradict a global “RLVR changes nothing” claim. | `G&L:280–290`; `Draft:30`, `Draft:94` | overstated |
| L4 | The output/internal distinction is useful. The “internal only” correction goes too far. **Neither** is correct relative to your unmeasured harmlessness outcome. | `Nguyen:473–488`; `Ledger:32` | wrong |
| L5 | **Partial support** for a broad text–geometry link is reasonable. It is not a replication of C3 or C4: trained adapter weights differ from activations during reading. “94% of positions” overstates the metric. | `Geology:128–134`, `Geology:249–250` | overstated |
| L6 | **Neither** is correct for your unmeasured default-cloud condensation. The stage rank trend is faithful. The OLMo-2 scope statement is not. | `Li:525–539`, `Li:688–706` | wrong |

---

**5. Novelty of C3–C10**

“Adds something” below means within these five checked works. It is not a claim of worldwide priority.

| Claim | Finding | Evidence | Severity |
|---|---|---|---|
| C3 | **Adds a useful control.** Cross-format decoding after replacing the reply with neutral text directly tests prompt sufficiency in OLMo-2. The checked works do not report this exact control. It does not measure persona behaviour. | `B-context:63–84`; `B-sens`, `context` | ok |
| C4 | **The fixed-text base/Instruct result is already known in Gemma.** Your additions are OLMo-2, ten readers, matched text baselines and quantified residual differences. Geology already shows training-text embeddings can recover persona-related geometry. | `Lu:160–163`, `Lu:1352–1362`; `Geology:128–130`; `C-results` | overstated |
| C5 | **Stage dependence is already known.** Han measures axis projections; G&L measures entropy; Nguyen measures harmlessness bias. Your addition is the fixed-text map sweep and its sensitivity controls. A reliable stage-change conclusion remains unestablished. | `Han:74–125`; `G&L:280–290`; `Nguyen:141–146`; `C-results` | overstated |
| C6 | **Adds a bounded elicitation result.** These exact unsteered OLMo-2 naming rates, aligned/conflicting examples and replication checks are additional evidence. “Base models contain personas” is a much broader, already explored claim. | `E3-results`; `E3-note:90–98`; `Lu:438–471` | ok |
| C7 | **Adds a practical prompt-confound result.** Neutral assistant examples suppress naming in this checkpoint. It does not reveal a training-stage origin or prove copying. | `Old-judge`, `olmo1b|rlvr*`; `Old-key` | ok |
| C8 | **Adds OLMo-2 repeatability evidence.** Stable role geometry is already central to Lu and Geology. This result does not newly establish a persona manifold. | `A-results`, `primary.D2`; `Lu:146–148`; `Geology:113–118` | ok |
| C9 | **The axis construction is already Lu’s.** Your additions are OLMo-2 repeatability, uncertainty and the prompted-assistant control. They do not establish default privilege. | `Lu:289–310`; `A-results`, `primary.D3`, `primary.assistant_control` | ok |
| C10 | **Adds measurement detail.** Two naming judges, role variation and a failed precision target are useful. Judging role expression is already established in Lu and Han. This is not a new capability result. | `A-miss`, `D1`; `Lu:128–143`; `Han:53–55` | ok |

---

**6. Behaviour versus structure**

The draft explicitly rejects behaviour-from-geometry claims at line 11. I found no direct claim that C3–C5 or C8–C9 proves role enactment. Its main errors are structural attribution, metric substitution and the unsupported copying mechanism.

| Claim | Finding | Evidence | Severity |
|---|---|---|---|
| C1 | **Behavioural measurement calibration.** Judge decisions are compared with owner labels. This does not validate population role-play rates. | `J-choice`, Terra candidate | ok |
| C2 | **Neither.** A historical statement, recorded second-hand. | `Meetings:73` | ok |
| C3 | **Structure.** A classifier reads activations. The neutral control contains no generated role reply. | `B-context:81–83`; `B-analysis:1453` | ok |
| C4 | **Structure**, compared with text-feature geometry. It does not measure what base would generate. | `C-results`, `pod_a.D1`; `C-note:98` | ok |
| C5 | **Structure and synthetic structure.** No behavioural stage effect is measured. | `C-analysis:214–238`, `C-analysis:456–465` | ok |
| C6 | **Behaviour, indirectly scored.** Judges recognise roles from output. Recognition does not establish coherence or full enactment. | `E3-results`; `E3-note:95–97` | overstated |
| C7 | **Behaviour, indirectly scored.** The score difference is measured; copying is not established by that score. | `Old-judge`; `Draft:56` | overstated |
| C8 | **Structure.** Repeatable context-conditioned role means; fidelity remains unverified. | `A-results`, `primary.D2`, `fidelity` | ok |
| C9 | **Structure.** A repeatable contrast direction, without a behavioural privilege test. The draft correctly limits this. | `A-results`, `primary.D3`; `Draft:64` | ok |
| C10 | **Behaviour, indirectly scored.** The judge identifies a target name. This is not the model naming itself or demonstrating self-knowledge. | `A-miss`, `D1.judges` | ok |
| C11 | **Neither.** Training provenance and question-content inspection. | `Card:23`; selected `Q-bank` entries | ok |
| L1 | **Both.** Activation steering is followed by judged output changes. That supports a behavioural effect of intervention. | `Lu:438–471` | ok |
| L2 | **Both, separately.** Axis projections measure structure. Judge pass rates measure output role expression. Projection shifts alone do not measure increased harmful-role expression. | `Han:54–59`, `Han:123`, `Han:131–133` | missing |
| L3 | **Behavioural output-distribution measure.** Entropy measures uncertainty over possible next tokens. It is not observed-token surprise. | `G&L:116–118`, `G&L:1241–1269` | overstated |
| L4 | **Both.** Final-output probabilities measure the behavioural distribution. Intermediate logit-lens scores are structural proxies, not emitted preferences. | `Nguyen:166–171`, `Nguyen:473–488` | missing |
| L5 | **Structure in the ledger’s main result.** Weight geometry and text embeddings are compared. The post separately tests judged behaviour; it does not obtain behaviour merely from geometry. | `Geology:128`, `Geology:156–174`, `Geology:284–288` | ok |
| L6 | **Structure.** RankMe describes covariance dimensionality. It does not measure role enactment or default-specific cloud tightness. | `Li:180–205` | ok |

---

**Three most important fixes**

1. **Rewrite C5.** Separate agreement-to-Instruct from direct map change. Restore the equal-map failure. Do not turn positive corrected intervals into stage-change evidence.
2. **Rewrite novelty.** Cite Lu’s fixed-rollout base comparison and Geology’s text baseline. State your additions as controls, baselines and OLMo-2 measurements.
3. **Correct source meanings.** Entropy is not surprise. Nguyen’s base output is +0.10. Geology reports variance explained. Li includes OLMo-2 SFT.

---

**Read depth**

Full text means the supplied local text, including appendices. Embedded images were not visually inspected.

| Source | How much I read |
|---|---|
| [Ledger](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_stage_persona_plan/review/proof_ledger_261112.md), [Draft](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_stage_persona_plan/overview_thomas_v2_261112.md) | Both in full. |
| [Lu](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2601.10387.txt) | Full supplied text: 2,923 lines. |
| [G&L](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2605.25459.txt) | Full supplied text: 1,515 lines. |
| [Nguyen](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2609.23935.txt) | Full supplied text: 521 lines. |
| [Li](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2509.23024.txt) | Full supplied text: 1,728 lines. |
| [Geology](/private/tmp/claude-501/-Users-brendon-persona-manifolds-manifold-persona/ad361f36-2648-47d6-95a7-4e43e5c9a47c/scratchpad/geology/geology.md) | Full supplied text, including caveats and appendix. |
| [Han](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/lw/axis_shift_stages_clean.md) | Full prose and widget code. Parsed both embedded JSON specifications. Read dataset keys, role counts, labels, projections and pass-rate columns used above. Highlight-label duplicates excluded from means. |
| [B-analysis](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_base_role_vs_format/analysis.py), [B-context](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_base_role_vs_format/context_extract.py) | Both in full. No execution. |
| [C-analysis](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/analysis.py), [C-maps](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/maps.py) | Both in full. No execution. |
| [C-rules](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/THRESHOLDS.md), [C-note](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/NOTE.md) | Both in full. |
| [J-choice](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_judge_validation/results/dev_judge_choice.json) | Full JSON. Reconstructed Terra’s confusion counts from group totals. |
| [J-note](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_judge_validation/NOTE.md), [J-rules](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_judge_validation/THRESHOLDS.md) | Both in full. |
| [B-sens](</Users/brendon/research-log/manifold-persona-log/Reference/2026-11-04 B Modal recovery/full_results/analysis_full_batch_v3_cuda_261105/sensitivities.json>) | Top-level keys; `context.C2_state16` and `C3_state16` counts and decoding accuracy summaries. |
| [B-D2](</Users/brendon/research-log/manifold-persona-log/Reference/2026-11-04 B Modal recovery/full_results/analysis_full_batch_v3_cuda_261105/D2.json>) | All four `formats.*.accuracy` summaries; top-level scope/settings keys. |
| [B-C2-report](</Users/brendon/research-log/manifold-persona-log/Reference/2026-11-05 B C2 decoding/C2_D2_report.json>) | Scope, permutation count and all four format summaries: accuracy, text gain, prompt-removal drop, p and null statistics. |
| [C-results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/records_full/analysis_full_v2/results_full_v2.json) | `code`, `provenance`, all `g5`; `pod_a.D1`, `D2`, `stage_change`, decoding, question-only sensitivity and selected frame/layer summaries. |
| [E3-results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/results/naming_v2.json) | Settings; main/replicate desc/ex2 `none` naming summaries; selected conflicting/filler controls; contrasts, primary decisions, seed rules and copy-control summary. |
| [E3-note](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md), [E3-rules](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/THRESHOLDS.md), [Judge-design-note](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/NOTE.md) | All three in full. |
| [Old-judge](/Users/brendon/persona-manifolds/manifold-persona/output/results_2026-09/audit_2026-10-02/judge_gpt.json) | Top-level keys and both `olmo1b|rlvr*` result objects. |
| [Old-key](/Users/brendon/persona-manifolds/manifold-persona/output/results_2026-09/audit_2026-10-02/KEY_char.json) | Both OLMo RLVR conditions: family, stage, role, question and wording keys; checked panel matching and default inclusion. |
| [Old-score](/Users/brendon/persona-manifolds/manifold-persona/output/results_2026-09/audit_2026-10-02/score.py), [Old-prep](/Users/brendon/persona-manifolds/manifold-persona/output/results_2026-09/audit_2026-10-02/judge_prep.py), [Old-transcripts code](/Users/brendon/persona-manifolds/manifold-persona/output/results_2026-09/audit_2026-10-02/transcripts.py) | All three in full. Raw transcript JSON was not read. |
| [A-results](</Users/brendon/research-log/manifold-persona-log/Reference/2026-10-30 A-B pilot results/full_A_results.json>) | Primary D2/D3 and assistant control; provenance, replicate/noise sensitivities, fidelity, family, axis gate and limits. |
| [A-axis](</Users/brendon/research-log/manifold-persona-log/Reference/2026-10-30 A axis diagnostic/results.json>) | Original result, conditional uncertainty summaries, all-default-wordings comparison and limits. |
| [A-miss](</Users/brendon/research-log/manifold-persona-log/Reference/2026-11-06 A D1 finish/d1_miss.json>) | D1 panel, settings, decisions, judge summaries and per-role fields used above. Recomputed role SD and approximate sample sizes. |
| [A-hit](</Users/brendon/research-log/manifold-persona-log/Reference/2026-11-06 A D1 finish/d1_hit.json>) | Both judges’ macro intervals, criterion/precision flags and family decisions. Sonnet’s bound difference equals 13/12,000. |
| [A-checks](</Users/brendon/research-log/manifold-persona-log/Reference/2026-11-06 A D1 finish/d1_checks.json>), [A-bounds](</Users/brendon/research-log/manifold-persona-log/Reference/2026-11-06 A D1 finish/build_bounds.py>) | Both in full. Raw invalid judge answers were not inspected. |
| [Meetings](</Users/brendon/research-log/manifold-persona-log/Reference/2026-11-12 Meetings 16-30 Oct.md>), [Claims and verdicts](</Users/brendon/research-log/manifold-persona-log/Reference/Claims and verdicts.md>) | Both in full. Neither recordings nor original meeting transcripts were inspected. |
| [Independent review](/Users/brendon/persona-manifolds/manifold-persona/output/independent_review_2026-10-02/review.txt) | Full text. Its underlying analyses were not rerun. |
| [Card](/Users/brendon/.cache/huggingface/hub/models--allenai--OLMo-2-1124-7B-Instruct/snapshots/470b1fba1ae01581f270116362ee4aa1b97f4c84/README.md) | Full cached model card. |
| [Q40](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/questions_40.json), [Q-bank](/Users/brendon/persona-manifolds/manifold-persona/refs/assistant-axis/extraction_questions.jsonl) | All 40 selected IDs; all corresponding `id` and `question` fields. Other 200 question texts were not inspected. |