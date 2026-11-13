# What each result can honestly claim (13 Nov 2026)

Method: four Fable 5.2 agents read the code and result files of every experiment (not the NOTE summaries), with file:line for each method and a result file for each number. I re-checked by hand: pushback rates (`output/results_2026-09/q2_pod_out/flip/flip_summary.json`), the unfiltered role SD in Han Xiao's data (`261112_han_reanalysis/results/role_sd_unfiltered.json`), the owner's 40 fresh labels (`judge_validation_261029/review261112_owner_labels.json` with `review261112_key.json`), and the default on Lu et al.'s 0-1 scale (recomputed from pod C states, scratchpad script; no result file yet). Agent reads, not human review.

Strength: solid = controls and replication hold; exploratory = one run, no correction, or a known confound; weak = failed or underpowered test.

---

## A. Does the base model play roles?

| Result (model) | Value | Honest claim | Strength |
|---|---|---|---|
| E0 blind naming (OLMo-2-1B base) | hit@5 13.3% to 30.0% by example set; chance about 2%; Instruct 53.3% | 1B base replies carry some role signal; one example set fails the rule | Exploratory |
| E0 role rubric | wrong-role control scored "played" 9.8% (limit 5%) | The Assistant Axis role rubric cannot tell personas apart on base replies | Weak (failed control) |
| E3 naming, description only (7B base) | 14.9% [9.5, 21.4] and 17.2% [11.6, 23.6]; shuffled 2.4% / 2.9%; replicate 16.3% / 18.8% | From a description, base replies carry a small role signal above chance | Solid |
| E3 naming with two example answers | 37.6% / 42.3%; with another role's examples the donor is named 24.8-29.8%, the described role 2.4-5.0% | The base model follows the example answers, not the description. The examples were Instruct-written | Exploratory (lower bound 18.9-20.6% across seeds) |
| My 20 fresh base and 20 Instruct replies | Base: pass 3 of 20 (class E or P: 6, I marked 3 of those P as no). Instruct: pass 16 of 20 | In my reading the base model rarely plays the role and Instruct mostly does | Exploratory (n = 20) |
| Terra pass rates | E3 desc 12.1% (9.8% without "assistant"), pod B base 15.4%, pod A Instruct 61.9% | One judge passes far fewer base than Instruct replies | Weak for base (its yes matched mine 1 of 4 on base) |
| Judges against my labels | Jev QWK 0.41; J8, J10 and broken-reply kappas 0.49 to 0.77; no judge passes | No automatic judge matches my labels on base replies | Solid (negative) |

## B. Where does the role sit in the states?

| Result (model) | Value | Honest claim | Strength |
|---|---|---|---|
| E3 decoding (7B base, state 17) | prompt in view 52.3%; prompt + fixed line 91.8%; reply after header only 15.2% (text 10.1%, gain +5.1 [2.6, 7.8]) | In the base model, the role in the states comes mostly from the description in the prompt | Solid as a control |
| Pod B D2 (base, 10 roles, 4 formats) | unseen format 0.82 to 0.92; prompt only 0.91 to 0.98; reply only 0.43 to 0.47; reply beats text in 2 formats (+0.07, +0.08) | Across formats the reader mainly finds which description was in the prompt; the reply alone carries a smaller signal | Solid / exploratory |
| Pod B D1 | role 9.5% vs format 8.8% of variance; question 35.9% | In base states, role and format matter about equally, and the question matters most | Solid (no difference) |
| Pod B D3 | maps across formats r 0.71 [0.35, 0.85] | Maps agree above chance; too weak to say they differ | Weak (underpowered) |
| Pod A D1 naming (Instruct) | hit@5 0.687 (both judges); translator 0.25, assistant 0.42 | Blind judges name the role from Instruct replies far above chance; some roles are barely played | Solid |
| Pod A D2 | 20-role map repeats r 0.90 [0.80, 0.94] | The Instruct role map repeats across questions and wordings | Solid |
| Pod A decoding | states 0.918; word n-grams 0.797; gain 0.121 [0.055, 0.201] | The role is readable from Instruct states; most of it is in the words | Exploratory |
| Pod A D3 | default-minus-roles direction cosine 0.88 between halves; gate fails | The default direction repeats fairly well, not precisely | Weak |
| Prompt-only control (Instruct) | decoding 0.991; map r 0.905; reply map vs prompt map 0.706 | The role prompt alone gives the role map. The reply map is a different map; the cause (words, length or persona) is not known | Solid / exploratory |

## C. Stages reading the same Instruct text (pod C, OLMo-2-7B, no role prompt)

| Result | Value | Honest claim | Strength |
|---|---|---|---|
| Map agreement, base vs Instruct readers | r 0.896 [0.767, 0.932]; best text map 0.846 | Base and Instruct draw near-equal role maps from the same text, a little beyond the words | Solid |
| Map change by stage | at most 0.003; control bias 0.027; planted 2-of-20 change not detected, 5-of-20 detected | No map change is visible; only large changes could be seen | Weak |
| Role decoding | words 0.797; base states 0.762; Instruct 0.755 | On the same text, the states do not beat the reply words | Solid |
| **Default position, Lu's 0-1 scale (new, 13 Nov)** | 0.043 to 0.049 from the nearer end at every reader; only consultant, assistant, robot beyond it; reply words alone 0.085 | **Reverses v5.** The default sits near the Assistant-like end at every stage, close to Lu's 0.03, and the reply words alone put it nearly there | Exploratory (20 roles; my recompute) |
| Default position, role-SD units | 1.08 role SDs at every reader; roles span -1.27 to +2.4 SD | Same placement; "inside the role range" holds, "not at one end" does not | Exploratory |
| Default spread | about 12% more variance than the median role at every reader; words about 79% | The default's cloud is a little wider than a typical role's at every stage | Exploratory |
| E3 base-written text | with examples r 0.942; description only 0.653 (fails floor) | With examples, base and Instruct maps of base-written text agree | Exploratory |

## D. Behaviour by stage

| Result | Value | Honest claim | Strength |
|---|---|---|---|
| Sycophancy, letter-balanced (OLMo-2-7B; Tulu-3) | SFT 0.800, DPO 0.845, RLVR 0.845; Tulu 0.782, 0.843, 0.837 | Trained stages side with the user's view about 80-85% of the time; DPO adds about 0.05; RLVR adds nothing | Solid (not new: Blank et al., Yao et al.) |
| Pushback flip rate (OLMo-2-7B; Tulu-3) | 32% [28, 36], 39% [35, 43], 50% [46, 54]; Tulu 15%, 12%, 11% | OLMo-2 gives way more after each stage, most at RLVR; Tulu barely changes | Solid for OLMo; weak for the Tulu trend |
| Coin flip (OLMo-2-7B) | chat frame DPO -0.065 [-0.100, -0.030]; SFT to DPO -0.026 to -0.044 in all five frames; one plain frame DPO -0.048 | DPO moves the coin word slightly toward the harmful task in every frame; RLVR adds nothing | Exploratory (review unfinished; frame and "Human:" prefix confounded) |
| Own-reply lengths | stop on their own: SFT 96%, DPO 37%, Instruct 41% | SFT writes much shorter replies | Solid (counts) |
| Length control | lowest re-read cosine 0.995; 16-token cut keeps at least 57.6% per role | States can be compared on the first 16 tokens | Solid (check) |

## E. Link between geometry and behaviour

| Result | Value | Honest claim | Strength |
|---|---|---|---|
| Probe position vs sycophancy (OLMo DPO, Instruct) | r -0.27 to -0.39 per question set | In OLMo DPO and Instruct, prompts further along the probe get less sycophantic answers | Exploratory (probe from example-turn clouds) |
| Steering on the probe vs random | probe drift 0.11; random 0.14 to 0.15; all below the 0.17 noise floor | The probe moves replies no more than random directions | Weak (no power) |

## F. Han Xiao's Olmo 3 7B data (layer 16)

| Result | Value | Honest claim | Strength |
|---|---|---|---|
| Reproduction | desirable +0.186, undesirable +0.008, gap +0.178, p 0.068 | His raw number reproduces from his files | Solid |
| Role-SD (z) units | gap -0.026, p 0.86; unfiltered +0.034, p 0.72 | The group gap exists in raw units only. "Both groups move toward the Assistant" is not right: the undesirable raw shift is +0.008 | Exploratory |
| Role SD on his axis | all replies 0.29, 0.57, 0.56; judge-filtered 0.61, 0.71, 0.69 | The roles spread apart on his axis at DPO, as the judge pass rate doubles (0.26 to 0.52) | Exploratory |
| Default position, 0-1 scale | 0.12 to 0.14 (filtered); 0.075 to 0.10 (all replies) | Inside the role range, near one end | Exploratory |
| Default spread | +0.03, +0.125, +0.126 (all replies) | The default's cloud widens against the median role at DPO, confounded with the change in reply mix | Exploratory |

## G. July

| Result | Value | Honest claim | Strength |
|---|---|---|---|
| E8 (Qwen2.5-3B-Instruct, 276 roles) | held-out error at 5 dims: curved 0.191 vs PCA 0.259; PCA at 8 dims 0.185 | At 5 dimensions a curved map rebuilds held-out roles better than a flat one | Solid statistics, narrow meaning (code on disk was edited after the run) |

---

## v5 mismatches found by the audit

1. "No stage puts the default at one end" is wrong on Lu's own scale (0.043-0.049 vs Lu 0.03).
2. Role-SD units are z-scores on the Assistant axis, not division by an SD on the main role axis.
3. "Both groups move toward the Assistant" reads too much into the z result.
4. The Olmo 3 default spread uses all replies; v5 does not say so, and the widening is confounded with the reply mix.
5. The 19.8% lower bound is not "after broken replies are removed"; it removes copied and self-naming replies, and it moves between 18.9% and 20.6% across seeds.
6. "3 of 20" is the owner's pass field; the class rule (E or P) gives 6 of 20.
7. "The replies add role structure" (prompt-only control) is stronger than the code shows (only that the two maps differ).
8. Pod A D1 used GPT-6.1 Terra and Sonnet, not GPT-5.6-Sol.
9. Coin flip: one plain-text frame excludes 0 at DPO; the main chat frame and the "Human:" prefix are confounded.
10. v5's RLVR section omits the OLMo-2 pushback rise at RLVR (39% to 50%).
