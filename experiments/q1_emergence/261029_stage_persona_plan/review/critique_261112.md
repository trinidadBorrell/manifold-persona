# Critique of the stage plan (12/11/2026)

Reviewer: Claude session ad361f36 (agent). This is an agent review, not an independent human review. The owner asked for a blunt critique of the plan texts. Several of those texts are mine, so this review also covers my own mistakes.

Texts reviewed:

- the four-question map and the rerun options (20 roles or 276 roles) written in this session on 12 Nov;
- the Fable 5.2 advice of 12 Nov, from a brief that I wrote;
- the mars-draft-planning handoff (session 28b6a4dd): the five-part story, the "build or select" frame and the thesis "the map stays, behaviour moves";
- the pod A, B and C designs and rules (`DESIGN.md`, each `THRESHOLDS.md`) and their verdicts.

Evidence base: every result file named below, the independent review `output/independent_review_2026-10-02/review.txt`, the meeting record in the vault `Weeks/` pages, and an inventory of all experiment folders (12 Nov). The literature section is at the end.

## Start here

| | |
|---|---|
| Short answer | The plan has no stated question that its measures can answer. "Persona manifold" is never defined, and no test of manifold structure exists for OLMo. The main measure (a 20-role map) mostly tracks the words in the replies. Two frozen pass rules could not pass at their sample size. The thesis joins two different instruments. |
| What survives | The measurement lessons: example turns, the prompt in context and the reply text each change the numbers by large amounts. These are checked several times and are the strongest result in the project. |
| Biggest risk for 16 Nov | Writing a stage story from a measure that cannot detect a stage change, with a judge that has not passed its own agreement gate. |
| New literature | A 5 Nov post (Han Xiao) finds on Olmo 3 that DPO, not SFT, pulls desirable roles toward the Assistant axis (p = 0.066, judge-filtered, no base). Stage sycophancy is now answered by two papers. |
| What to do | Section 7: test the Han Xiao effect on OLMo-2 at 276 roles with no judge filter and a fixed-reader arm, on three GPUs in parallel. Drop the 20-role rerun. |

---

## 1. Framing faults

| # | Fault | Evidence |
|---|---|---|
| 1.1 | "Persona manifold" has no operational definition. The plans use "manifold", "persona space", "role map" and "map" as if they were one thing. The only measure run on OLMo since 5 Oct is a role-role cosine-distance matrix compared by Spearman r (RSA). That is a map of 20 points. A manifold claim needs a shape test, and none exists. | Pod A D2, pod C D1/D2 definitions in each `THRESHOLDS.md`. No curvature test has run on OLMo: `261030_stage_manifold_test/` has no code. Trinidad's curvature-null code exists for Qwen3-8B (`src/curvature_null`, per `261030_stage_manifold_test/LITERATURE.md`), so a test can be ported; my first draft wrongly said no curvature code exists. |
| 1.2 | "Does post-training build the space or select a place in it?" has no decision rule. No result in the plan says which outcome would mean "build" and which would mean "select". As written, any result fits. | Planning handoff, 12 Nov. |
| 1.3 | The thesis "the map stays, behaviour moves" compares a fixed-text representation measure (pod C) with own-text behaviour rates from September (sycophancy, pushback) on different items, prompts and code. It is two separate observations placed side by side. | Pod C reads Instruct-written text only. Sycophancy files `output/results_2026-09/pod/syco/`; prompt format there is not checked. |
| 1.4 | "The map stays" is close to true by construction. Pod C feeds the same text to every checkpoint, and a character n-gram map of that text already reaches r_norm 0.960 against Instruct, base 0.996. So the same text gives nearly the same map. | `261029_fixed_text_read_through/records_full/analysis_full_v2/results_full_v2.json`, `D1.anchor_r_norm`. |
| 1.5 | "The map stays" is also not what the data says. The noise-corrected change is small but above zero at base to SFT (0.0027 [0.0004, 0.0079]) and at SFT to DPO (0.0009 [0.0004, 0.0019]). No rule says how large a change matters, so neither "stays" nor "moves" can be claimed. | Same file, `D2.transitions`. |
| 1.6 | "Behaviour moves" is weak between trained stages. OLMo sycophancy is 0.802, 0.846, 0.846 (SFT, DPO, RLVR). Pushback rises in OLMo (all-correct items, logit: 0.324, 0.390, 0.502) and falls in Tulu. Yao et al. (27 Oct) already cover stage sycophancy on this ladder with more controls. | `output/results_2026-09/q2_pod_out/flip/flip_summary.csv`; vault literature note of 5 Nov. |
| 1.7 | The four-question map mixes levels. Question 1 (default position) and question 2 (shape) are about representations. Question 3 (trait sliders) is about directions that do not exist yet for OLMo. Question 4 (robustness) has no clean evidence at all. Four questions with four instruments, in four days, is not one story. | Inventory 12 Nov: no trait-axis or steering result since 5 Oct. |
| 1.8 | Thomas's direction is not served. His questions were interpretable factors with steerable control (2 Oct), trait sliders tracked over SFT, RLHF and RLVR (16 Oct), and base robustness under steering (4 Sep). The 16 Nov plan has no steering and no trait axis. | Vault `Weeks/2026-W36.md`, `W40.md`, `W42.md`. |
| 1.9 | The owner's own scope (23 Oct: "both" the 276-role map across stages and the trait sliders) and his 12 Nov words ("we are focusing too much on the role map") were missing from my first two recommendations. I recommended 20 roles twice, then 276 roles, within one hour. The criteria did not change, only the context I had read. | This session, 12 Nov. |

---

## 2. Measurement faults

| # | Fault | Evidence |
|---|---|---|
| 2.1 | Pod A D1 could not pass. Per-role hit@5 has SD 0.243 over 20 roles, so the 95% interval is ±10.7 points whatever the number of replies. The ±7.5 target needs about 40 roles. Nobody ran this arithmetic before freezing the rule. | `Reference/2026-11-06 A D1 finish/d1_miss.json`; pod A `THRESHOLDS.md`. |
| 2.2 | Pod B D3 could not pass. At 10 roles the rule passes in 8% of simulated datasets even when the maps are identical. Same cause: no power check before freezing. | `Reference/2026-11-06 B D3 power/d3_power_results.json`. |
| 2.3 | Pod A reads states with the role system prompt in context. In base, the prompt alone decodes the role at 91–98% (pod B C3). So pod A D2 ("the role map repeats", r 0.901) may be a map of the prompts. The pod A rules say this themselves. | Pod A `THRESHOLDS.md` line 9; pod B `challenge_261105.md`. |
| 2.4 | The default measure is fragile. Pod A D3 depends on which default wording sits in each half (0.92–0.94 with the same wording on both sides, 0.881 split). One of the five default wordings has no system message, so "default" mixes two conditions. | Pod A `NOTE.md`, axis diagnostic. |
| 2.5 | Several bootstrap intervals sit at or below their own point (pod A D3 0.881 in [0.763, 0.884]; pod C canary D2 0.984 in [0.946, 0.982]). The statistic is nonlinear and the resampling shifts it. Intervals of this kind need a bias check before anyone reads them. | Pod A `NOTE.md:50`; inventory item 6. |
| 2.6 | The judge is not validated. No judge passed the agreement gate (axiom A3 fails). Terra on 40 practice items: phi 0.60, and it kept 3 of the owner's 8 base "yes" replies against 13 of 16 chat ones (it said yes 14 times on chat, 13 of them right). Its error differs by group, so a raw rate difference between groups can be judge error. | `261029_judge_validation/NOTE.md`. |
| 2.7 | Twenty roles were selected, not sampled. Any statistic that resamples roles describes these 20, not role space. The role spread (0.20 to 1.00 in naming) dominates every macro number. | `DESIGN.md` line 23. |
| 2.8 | One model family, one checkpoint per stage, one training run. "Post-training does X" cannot be separated from "this OLMo-2 run did X". | All pods. |
| 2.9 | Pod C cannot detect a change in 2 of 20 roles (planted case 0.156 [−0.006, 0.340]). A null stage result from pod C is therefore uninformative about local changes. | Pod C `NOTE.md`, G5. |

---

## 3. Claims in the plan texts against the files

| Claim in a plan text | What the files show |
|---|---|
| Four-question map: question 2 evidence includes "July Qwen-3B curvature" | July compares GPLVM against PCA reconstruction (0.191 vs 0.259) on Qwen2.5-3B-Instruct, one stage. It is a model-fit comparison, not a curvature test, and not a stage result. |
| Four-question map: question 4 evidence "steering results" | The September steering drift "does not hold" (example turns). There is no clean steering result. |
| Four-question map: question 3 evidence "sycophancy and pushback" | Behaviour rates say nothing about slider geometry. They are not trait directions. |
| "The 276-role run fits the 23 Oct decision" | A 276-role own-generation run repeats independent-review F3: each stage writes its own text, so stage contrasts mix representation, content and compliance. It fits the decision only with a fixed-reader arm. |
| Option 1 "reuses pod A's frozen rules" | Those rules contain the D1 precision target that cannot pass at 20 roles (2.1). Reusing them guarantees a second fail. |
| Planning handoff: "no measurable map change across 10 checkpoints" | Base to SFT and SFT to DPO changes are measurable (1.5). They are small, and no threshold of importance exists. |
| Planning handoff: rerun "about $30 in judging" | That covers Terra labels only. D1 naming cost about $75 per stage. |
| Planning handoff: "K6 in Axioms and evidence" as the reference for the default question | That page is stale (29 Oct): it says pods A to C have not run. |
| Fable advice: "paired pass-rate difference" as the new primary | Sound structure, but it rests on Terra pass labels (2.6). Between SFT and DPO, both in chat format, the group error may be similar; that is untested. |
| My brief to Fable | I framed the plan around pods A to C only. Its advice inherited that frame. |

---

## 4. Process faults

| # | Fault |
|---|---|
| 4.1 | No human review of any pod. The owner's rule is to read raw transcripts before numbers. Nine replies and 34 labels were read by an agent, not by the owner. |
| 4.2 | Rules were frozen without a power or precision simulation. Two primary tests failed for that reason alone (2.1, 2.2). |
| 4.3 | Records are stale or inconsistent: the axioms page, the E3 status line, two experiment index rows, and pod B "Next" sections predate the 5–6 Nov verdicts. A planner that reads them gets wrong facts, as the handoff did. |
| 4.4 | Pod A per-phase files and decoder checkpoints were lost with the RunPod stop. The summary survived only as a local vault copy. |
| 4.5 | Two sessions ran the same paid retry at once (6 Nov). |
| 4.6 | 16 Nov is a self-set date (17 Sep). The plan treats it as fixed and cuts design work to meet it. The cost of a weak post may be higher than the cost of a later date. That choice belongs to the owner. |
| 4.7 | On 30 Oct Thomas asked for a high-level structure of the post "before running any more experiments", so he can comment before text is written. Since then the project ran D1 naming (about $75), three Terra label runs and two power checks, and it now plans a new GPU run. I found no structure in the files. If none was sent, the plan runs against the mentor's explicit order of work. |
| 4.8 | On 30 Oct the owner told Thomas the chosen judge "agreed with me at a 90% accuracy". The file shows agreement 0.775 (31 of 40), precision 0.94, recall 0.67, phi 0.60, and 3 of 8 base "yes" kept (`261029_judge_validation/results/dev_judge_choice.json`). The 90% figure needs a correction before the post. |
| 4.9 | Budget. On 30 Oct the agreed judge budget was $150 per person, and the team agreed to extend it by about $50 (Granola notes). The owner reported about $80 spent then. Since then: pod B Terra (about $17) and D1 naming (about $75 from token counts). The 276-role Terra labels (up to about $120) would go well past the agreed amount. Billed amounts are missing from the cost log. |

---

## 5. What survives

| Result | Why it holds |
|---|---|
| Example turns suppress role play | OLMo-2-1B final: 55.4% without them, 2.9% with them; identical replies 19.2% vs 0.2%. Found in the 1–2 Oct audits and reproduced in later design. |
| The prompt in context carries the role | Prompt-only decoding 91.8% in E3 and 91–98% in pod B, against 43–47% from the reply alone. |
| Reply text carries most of a fixed-text role map | Pod C: char n-gram map r_norm 0.960; states beat the best text map by only 0.037 [0.013, 0.081]. |
| Base plays roles weakly from a description | E3 desc hit@5 14.9–17.2%; Terra 15.4% on pod B (raw, likely low). With two examples, 37.6–42.3% (fragile). |
| Instruct role map repeats in its own frame | Pod A D2 r 0.901 [0.797, 0.939], with caveat 2.3. |

These five rows are a coherent measurement story: what a "persona representation" result measures depends on the prompt, the examples and the text, and each can move the number by more than any stage effect seen so far.

---

## 6. Literature (12 Nov search)

An agent read about 17 new sources (abstract plus method and results; four OpenReview items as abstracts only). Files: `output/lit_stages_2026-11-12/`. I checked the two that matter most against the saved text (Han Xiao, Yang et al.). The others are agent-read only.

| Source | What it does | Effect on this plan |
|---|---|---|
| Han Xiao, LessWrong, 5 Nov ([link](https://www.lesswrong.com/posts/xxCtj4rhrBoSeGy4g/how-the-assistant-axis-shifts-across-post-training-stages-1)) | Lu et al.'s pipeline on Olmo 3 7B: SFT, DPO, RLVR (no base). 275 roles, 240 questions. Only replies the judge scores 3/3 are kept. DPO pulls desirable roles toward the Assistant axis (0.57 to 0.76, CI [0.14, 0.23] for the change); undesirable roles do not move. Label-shuffle p = 0.066. RLVR changes almost nothing. Role-play pass rate rises at DPO. | Question 1 after SFT is partly scooped on Olmo 3. Its weak points are ours to fix: p above 0.05, a judge filter whose pass rate doubles at DPO (so each stage's role vectors come from a different sample), no base and no fixed-text control. |
| Yang et al., arXiv 2611.10045, 5 Nov | Sycophancy at every checkpoint, Olmo 3 7B, Tülu 3 8B, SmolLM3. SFT reaches at least 92% of the preference-tuned score. | With Yao et al., stage sycophancy is answered. "Behaviour moves" cannot be a new result. |
| Hsu et al., arXiv 2611.12937, 6 Nov | Trait-direction similarity against human scale correlations. Base r ≤ 0.14; instruct 0.37–0.66. Probes decode traits about equally in base. | Trait directions exist in base. The open slider question is how traits couple and steer per stage, not whether they exist. |
| Prompt dependence, OpenReview (abstract only) | OLMo-2-7B base, SFT, DPO, Instruct. Context sensitivity rises at every stage, about two-thirds at SFT. | Argues against a simple "early Assistant is less robust" story. |
| Goodfire, LessWrong, 29 Apr | OLMo-2-7B SFT refuses and DPO complies under a formatting distractor. | On that axis DPO is less robust. Robustness depends on the kind of pressure. |
| Judge twin, arXiv 2609.32196 | OLMo-2 base text is the most foreign to trained readers. | A fixed-text reader is confounded by how familiar the text is to each checkpoint. Pod C reads Instruct text, which is on-policy for late stages only. |
| G. & Lindsey, arXiv 2605.25459 | On-policy output entropy is 3–4× lower than off-policy. | Use an entropy or surprise covariate in any fixed-reader comparison. |
| Spokes, not a ring, arXiv 2610.37796 | A weekday "circle" is seven discrete states. | Test continuity before calling a trait slider a 1D manifold. |

Two sources the first search missed or under-read (found by the Fable 5.2 and Codex reviews; Lu et al. section 3.2.2 checked by me in `output/lit_q1_2026-10-05/2601.10387.txt`):

| Source | What it does | Effect on this plan |
|---|---|---|
| Lu et al., arXiv 2601.10387, section 3.2.2 and App. B.3, G.2 | Steers base Gemma 2 27B and Llama 3.1 70B with the instruct Assistant axis, using prefills ("My job is to"). Completions move toward helpful human roles (therapists, consultants) and agreeable traits. Their conclusion: the axis "mainly inherits from pre-existing helpful and harmless human personas in base models, later acquiring additional associations (such as being an AI) during post-training". In base Llama 3.1 70B, role PC1 did not clearly separate Assistant-like personas. | "Is the Assistant privileged before post-training?" is partly answered: the direction exists in base as a helpful-human direction. Open: when the AI association and the default occupancy appear (SFT or DPO), and on OLMo-2. |
| Li et al., arXiv 2509.23024 (NeurIPS 2025) | Effective rank of last-token representations on Llama-3.1-Tülu-3.1-8B: up in SFT and DPO, down in RLVR. OLMo-2 only in pretraining. | Global "condensation" across these stage types is studied under another name. A per-persona Assistant version is at most a narrow addition. |

The nine OpenReview papers, read in full on 12 Nov from PDFs the owner saved (`/Users/brendon/persona-research-related-work/`; text in `output/lit_stages_2026-11-12/or_txt/`). Three agent readers; I spot-checked the key numbers in the extracted text.

| Paper | What it measured | Effect on this plan |
|---|---|---|
| Forced openers (Pre-to-Post, NeurIPS 2026) | Olmo 3 7B base, SFT, DPO, Final. Math: 62 of 154 items stop capitulating from SFT to DPO, with the log-prob gap moving +0.49 toward the wrong answer. Medical capitulation (unforced): 10.9%, 17.1%, 7.6%, 5.9%. | Olmo 3 only. On this test SFT is the least robust stage and DPO repairs most of it. "Sycophancy is set at SFT" needs splitting by measure. |
| Post-Training Reshapes Truth–Sycophancy Geometry (ICLR 2027 sub.) | 22 base vs final-instruct pairs, one raw-text format. OLMo-2-7B and 13B are included, base vs Instruct only. Mean truth–sycophancy cosine −0.01 to −0.03. | OLMo-2-7B shows a small shift (about −0.015 from the figure). No SFT or DPO stage. |
| Family-Dependent Prompt Dependence (ICLR 2027 sub.) | OLMo-2-7B base, SFT, DPO, Instruct, one plain-text wrapper, teacher forcing. Fact-swap dependence: +2.59 at SFT, +0.94 at DPO, +0.25 at RLVR (of +3.78). | The "two-thirds at SFT" figure is fact-swap dependence, not sycophancy or general context sensitivity. Its sharpening control is described but not reported. |
| Persona Selection on the Persona Simplex (ICLR 2027 sub.) | Instruct models LoRA-tuned on three synthetic personas; reads a belief over personas from behaviour and activations. | No stages. It suggests a test for "privileged": the Assistant's prior weight in base. Its shape measures (planarity, mass outside the simplex) can be reused. |
| Jailbreaks Live in Persona Space (ICLR 2027 sub.) | Llama-3.1-8B-Instruct and Gemma-4-31B. A 229–240 trait span holds 41% and 27% of the refusal direction's norm (random about 5%). | Instruct only. The signal sits at chat-template tokens, so the method does not carry to plain-text base models as it is. |
| Multi-Agent Sycophancy (ICLR 2027 sub.) | Four families plus Mixtral, base and Instruct, all in chat format. Base yields more in 9 of 12 cells strictly; Mixtral named-peer goes the other way (3.9% vs 72.1%). Yield is read from an LDA, not from written answers. | "Base in chat format", not the base default persona. The sign of the post-training effect depends on family. |
| Nine Judges, Two Effective Votes (JUDGe 2026) | Nine judges give about 2.2 effective votes; errors correlate across families. | Our Terra and Sonnet naming agreement is about one check, not two. On our 40 practice items the four judges' errors correlate at phi 0.79 (reader's arithmetic). |
| Agreement With Humans Is Not a Property of the Judge (JUDGe 2026) | One judge correlates 0.43–0.61 with experts and 0.03–0.07 with crowdworkers. | Our 0.775 agreement is "agrees with the owner on 40 items from E3 and pod A". It does not transfer to pod B or to other raters. |
| Calibrate, Validate, Freeze, Then Optimize (JUDGe 2026) | Frozen rule, about 190 dev and 201 held-out rows, test scored once. | Our 40 items set the wording, the judge and the pass rule, so our numbers are in-sample. Base recall 3 of 8 has a 95% interval of about 0.14 to 0.69. |

Effect on the judge evidence: raw Terra pass rates cannot be compared across base and chat, because recall differs by group (3 of 8 against 13 of 16). Report them as "Terra pass rate", claim only the direction (chat above base, which the owner labels also show), and, if a stage comparison is needed, draw new held-out owner labels per group (about 100 each). That would reverse the 30 Oct decision against more sample labels, so it is the owner's call.

What this changes:

- Question 1 now has a sharp, falsifiable form: on OLMo-2, does the DPO step (not SFT) pull desirable roles toward the default, and does it survive with no judge filter and on fixed text? Treat Han Xiao's result as a prediction to test.
- Question 2 (shape across stages) is open in this set.
- Question 3 should change from "do sliders exist" to "does trait coupling or steerability change per stage".
- Question 4 should leave the 16 Nov post. The literature already shows the answer depends on the pressure type.

---

## 7. Runs worth doing, ranked

Ranked by what each run can decide. GPU times use measured A100 rates (pod A: 21,000 replies in 4,053 s; pod C: about 650 s per checkpoint read). H100 speed-ups are a guess (about 1.5–2× for generation) and must be measured on the first batch.

The literature changes my earlier ranking. Correction (12 Nov, after the Fable 5.2 review): I first argued that a 20-role run is too small because Han Xiao's interval would widen by √(275/20). That was wrong. His interval comes from random question half-splits (footnote 2 of the post), not from resampling roles, so my scaling does not apply. The role count still matters for a role-resampled interval and for the desirable/undesirable contrast, but I have no number for it.

| Rank | Run | What it decides | Cost and time | Needs first |
|---|---|---|---|---|
| 1 | 276 roles on SFT, DPO and Instruct, own replies, 40 questions × 5 wordings × 1 reply (55,200 per stage), no judge filter in the primary | Does DPO, not SFT, move desirable roles toward the default on OLMo-2? This tests Han Xiao's result with the confounds removed | Three GPUs in parallel. About 2.9 h per stage on A100 at pod A's rate (55,200 / 5.2 per s); H100 faster by an unmeasured factor. No judge cost for the primary | Model flag at `gen_extract.py:32-33`, `weights_only=True` at `:228`, role list through `--roles`, `--replicates 1`; desirable/undesirable role labels fixed before the run; the rule and a smallest change of interest frozen first |
| 2 | Fixed-reader arm for run 1: one checkpoint (and base) reads all three stages' replies, with n-gram maps and a surprise covariate | Whether any stage effect is in the representation or only in the words | About 10–20 min per pass on A100 (pod C rate, scaled); can start as each stage finishes | `261029_fixed_text_read_through/replies.py` reader for the new text; tested at 276 roles (not yet) |
| 3 | Default position on fixed text from pod C's saved states | The base and RL-step readers on identical default text, which Han Xiao cannot do | No GPU. Large fetch (31 GB on 30 Oct); run on a pod or Modal CPU | New analysis code and a text baseline |
| 4 | Terra labels on run 1, as a filtered sensitivity only, plus about 60 owner labels with the stage hidden | Whether the judge filter creates or hides the effect | Up to about $120; batch time about 11 h for pod B's size | Run 1 |
| 5 | Measurement write-up (section 5) | The project's most solid result | Owner time | Nothing |
| 6 | Trait coupling per stage (Hsu-style similarity matrix, or T-090) | Thomas's slider direction, in the form the literature leaves open | Untested | A design and a base format bridge. After 16 Nov |
| 7 | Robustness under steering | Thomas's 4 Sep question | Unknown | Out of the 16 Nov post |

The 20-role SFT and DPO run drops out. It would add a stage point to pods A and C, but it cannot decide the one sharp question the literature now poses.

Parallel lanes, if the owner approves runs 1 to 4:

| Lane | Work |
|---|---|
| A (no GPU, first) | Freeze the rule for run 1: role labels, primary contrast, precision target from a simulation, smallest change of interest. Make the code changes and a smoke test |
| B, C, D (GPUs 1–3) | SFT, DPO and Instruct generation at the same time; each then runs its fixed-reader pass |
| E (CPU) | Run 3 on pod C's saved states; the measurement write-up |
| F (after B–D) | Run 1 and 2 analysis; Terra batch and owner labels for run 4 in parallel |

GPU time is small. The slow parts are freezing the rule, the judge batch window and the owner's review. Lane A must finish before any GPU starts, or the rule can follow the data.

---

## 8. What the 16, 23 and 30 Oct meetings add

Source: transcripts the owner pasted on 12 Nov. The vault has no record of 30 Oct and only prep for 23 Oct. Speakers are not always named in the transcripts; I name Thomas only where the text shows it.

| Date | What was said | Effect on this plan |
|---|---|---|
| 16 Oct | Thomas asked the owner "what is the final question" and said "it should be more about understanding than finding a better method". In-context persona shift "might be a separate project". | The project still has no single final question (section 1). |
| 16 Oct | Thomas: Assistant Axis shows PC1 is the Assistant axis and other PCs are not interpretable; interpretable, steerable PGA axes "would already be really cool". Trait → 1D manifold pipeline for traits people disagree on (agreeableness: sycophantic vs harsh; persistence vs concise). | These belong to Trini and Gio's axis work. |
| 16 Oct | On manifold adherence: no clear interpretation; "if you can show that they fail, and when they fail, I think that's a post in itself." A random-basis control was asked for Gio's 32-persona basis. | Supports making the measurement failures (section 5) a result in their own right. |
| 23 Oct | Trini: PGA axes are no more interpretable than PCA axes; after about 20 components no theme appears. Thomas asked how linear the geodesics are (fit a line, look at residuals), and for a numeric 0–10 trait ladder. | Within this team, the case for a curved "manifold" is not established either. |
| 30 Oct | Gio: a five-level openness ladder fits a line within noise; steering along the curve reaches fewer personas than the line. Thomas saw a possible sine-like pattern and asked for numeric levels. | Same point. "Manifold" wording is not supported yet. |
| 30 Oct | Thomas: "I would prioritize trying to write up what you have so far"; give "a high-level structure now… before running any more experiments". | See 4.7. This is the first step, before any run. |
| 30 Oct | Thomas, on 1D traits across stages: figure out how to do it "in one stage first, and then we can maybe extend". | Drop trait sliders across stages from 16 Nov. |
| 30 Oct | Thomas: "once you've figured out a good structure of persona space, I think… Brendon maybe can look at how that changes from the pre-trained model to the post-trained model." And: "the assistant is definitely like a privileged persona after post-training but before post-training, it's unclear." The owner looks "at if there's some other personas which are in some way privileged in the pre-trained model." | This is the question Thomas assigned to the owner. It matches Han Xiao's DPO result and question 1. Shape change (question 2) waits on Trini and Gio's structure. |
| 30 Oct | Thomas named two hypotheses for emergent misalignment: the self-representation moves to the misaligned persona, or the Assistant and misaligned personas merge. He sent the persona hierarchy paper (a trait trained into the Assistant spreads to other personas). | Relevant to the reward part. Not for 16 Nov. |

The Granola notes the owner pasted later on 12 Nov add three points (vault `Reference/2026-11-12 Meetings 16-30 Oct.md`):

- One question runs through the owner's meetings since July: where the Assistant sits in base and after SFT (24 Jul), "tracking when 'assistant' becomes a privileged persona across post-training stages" (26 Aug), the Assistant direction and finer checkpoints (4 Sep), "revisit assistant manifold condensation result" (2 Oct), and "before post-training, it's unclear" (30 Oct). The team's own measure of "privileged" is condensation: Trini found the Assistant's cloud lower-dimensional and more curved than other personas' clouds (31 Jul, 14 Aug). The four-question map dropped this thread.
- The 30 Oct notes give the owner a specific next step: examine how persona space structure changes pre- vs post-training "using Giovanni's classifier approach" (judge-recall boundaries that look like cones). That needs steering runs and judge labels on each OLMo stage. It does not fit before 16 Nov.
- The notes record "90% agreement" for the judge, so the wrong figure is now in a team record (see 4.8).

What this changes in my ranking: run 1 (276 roles, SFT, DPO, Instruct) is the most direct test of Thomas's question "when does the Assistant become privileged". Run 3 (pod C default rows, read by base) is the only clean base evidence for "before post-training". Both need an operational definition of "privileged" frozen first. Two fit the team's history: position (the default against the role cloud, as in Han Xiao) and condensation (the default cloud's dimension and curvature against the role clouds, as in Trini's metrics). Run 1 stores per-reply states, so both can be measured per stage. Condensation on own replies is confounded by text (formulaic default replies give a tight cloud), so the fixed-reader arm is required for it. Review F9 also warns that dimension estimators depend on sampling density, so the roles and the default need equal reply counts. The owner's own "other privileged persona in base" idea has one weak lead: in pod B, Terra passes optimist (34.1%), robot (27.4%) and narcissist (24.3%) most often. These are raw judge rates with the role description in context, so they may only show which roles are easiest to describe.

---

## 9. Questions for the owner

1. What single sentence should the 16 Nov post be able to say if every run works? Thomas's 30 Oct framing suggests: "In OLMo-2-7B, the Assistant becomes a privileged persona at [stage], and the change is [in the representation / only in the text]." Without a sentence, no run can be ranked well.
2. Has a high-level structure gone to Thomas since 30 Oct? If not, that comes before any run.
3. Is the measurement story (section 5) acceptable as the main result, with the stage runs as a first test?
4. Should "manifold" stay out of the post until a curvature test exists on OLMo?
5. Is 16 Nov fixed?
