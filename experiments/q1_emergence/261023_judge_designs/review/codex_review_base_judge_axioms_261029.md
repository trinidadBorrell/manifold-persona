**Verdict:** The text finds real problems, but its proposed fix confuses role fidelity, role naming, and reply quality.

---

**1. Item-by-item review**

The opening is too strong. The tested judges lack validation for **base role fidelity**. That does not establish that instruct judges need a different definition of role-play.

The main defect is what the labels mean. A four-level scale is not inherently wrong.

| Item | Decision | Critique and specific fix | Evidence |
|---|---|---|---|
| **R1 — Document continuation** | **Change** | Right for this elicitation method. Too broad as a definition of all base role-play. Define the scored object: **the first guest turn, under a stated document format**. Judge the requested role, including professional and AI roles. | [E3 prompt design](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:7); [Shanahan, §§2–6](https://arxiv.org/html/2305.16367v1) |
| **R2 — Rules from failure modes** | **Change** | Failure modes should shape tests and flags. They should not define what a role is. The missing generic-speaker category is real. “Almost never say they are an AI” needs a measured frequency. E0’s **9.8% pools base and instruct controls**. | [E0 rubric](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261005_base_incontext_elicitation/prompt_rolejudge.txt:1); [aggregate result](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261005_base_incontext_elicitation/results_role.json:1) |
| **R3 — Hide all role information** | **Change** | Wrong for target-role fidelity. A judge needs the role being assessed. Hide the actual generation condition, examples, model, and checkpoint. Keep reply-only naming as a separate test. Self-naming and copying need flags, not automatic removal from every measure. | [E3 selection rules](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:30); [PersonaEval, §3.1](https://arxiv.org/html/2508.10014v1) |
| **R4 — Separate coherence and role-play** | **Keep; extend** | Correct and central. Add **question engagement** as a third dimension. Separate scores do not imply statistical independence. The 90–97% naming hits show that off-question text can retain role cues. They do not prove full role fidelity. | [28 Oct analysis](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/NOTE.md:134); [Han, §3.1](https://arxiv.org/html/2204.10825v1) |
| **R5 — Owner labels are the reference** | **Change** | Use them as the declared reference, not unquestionable truth. The Jev result shows test contamination, not that revised labels are necessarily worse. Permit revisions during development. Then use fresh, sealed validation items. | [Jev: 0.41 versus 0.69](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:95) |
| **R6 — Human validation and controls** | **Change** | Sound aim; incomplete statistics. Define “κ per class” as one class versus all others. Report precision, recall, class counts, and uncertainty too. Judge agreement is useful evidence about consistency, but cannot establish human validity alone. | [matched comparison results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/results/relabel_fable_score.json:1); [Thakur, §§4–5](https://arxiv.org/html/2406.12624v6) |
| **R7 — Separate base/instruct validation** | **Keep; extend** | Necessary, but insufficient for A3. Both validations must use the same role definition and decision rule. Also test early checkpoints. Different failure rates can change judge error rates even when the judge prompt stays fixed. | [Moskvoretskii, §§2–3 and Appendix C](https://arxiv.org/html/2605.13329v1) |
| **C1 — Drop Lu, J8, J10; check judge2** | **Change** | Retire the unchanged Lu rubric as a base fidelity gate. Retain J8/J10 as tested diagnostic methods. Their role results were **report-only**, without a frozen role pass rule. Describe judge2 accurately: its definitions already differ substantially from Lu’s. It still mixes breakage into score 0. | [test rules](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/THRESHOLDS.md:25); [judge2 definitions](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/judge2.py:77) |
| **C2 — Naming is the main measure and a lower bound** | **Change** | Keep it as **role identifiability**. Drop the unqualified lower-bound claim. Misses lower its rate; false hits raise it. The owner agreement concerns a small J8/J10 sample. It does not validate the original E3 naming judges across their full output distribution. | [35 target items and 15 controls](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/NOTE.md:60) |
| **C3 — New base rubric** | **Change** | Use a shared fidelity rubric with base-specific examples. “Does not copy” measures originality. “Engages the question” measures task engagement. Keep both separate from fidelity. Otherwise this consequence partly recreates the mixed “broken” category that R4 rejects. | [16 Oct definition](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/THRESHOLDS.md:123) |
| **C4 — Prefer fixed text and decoding** | **Change** | Useful for representation questions. They cannot replace a behavioral role measure. The project already shows strong role decoding from prompt context alone. Fixed text tests representation while reading supplied text, rather than successful role generation. | [context-control results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:98); [limits](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/THRESHOLDS.md:140) |
| **H1 — Instruct examples and assistant tone** | **Change** | Example provenance is established. Tone borrowing has qualitative support from 20 inspected transcripts. Judge preference for that tone remains a hypothesis. Test matched versions that preserve role content while changing assistant style. Also flag selection by the same judge family. | [provenance and selection](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:26); [inspection and selection bias](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:55) |
| **H2 — Role list and task roles** | **Change** | The source paper curated **275 roles**, then generated elicitation prompts. The project uses 276 candidates. “Come from instruct system prompts” oversimplifies this origin. Translating a question can express a translator role, but may violate the interview task. Specify which behavior the role requires. | [Lu, §2.1.1](https://arxiv.org/html/2601.10387v1); [translator example retained](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:14) |
| **H3 — Instruct judge discounts plain replies** | **Change** | Plausible and testable, but currently unproved here. It overlaps H1 and H4. State a prediction: equal role evidence should receive equal fidelity labels across writing styles. Judge training type alone does not establish this bias. | [Arena study, “shared taste” and methodology](https://arena.ai/blog/llm-judge-self-preference) supports testing bias, not this specific cause |
| **H4 — Short replies and clipping** | **Keep; qualify** | Correct distinction: little evidence is not evidence of another role. The 500-character limit is real. However, the local analysis found **13/15 longer broken replies** despite clipping. It does not establish clipping as the cause of these disagreements. | [length analysis](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/NOTE.md:41) |
| **H5 — Stop markers** | **Keep; strengthen** | The stray heading is documented. State the parser and scored span explicitly. Record turn length and boundary events. Cutting before a speaker switch measures fidelity **within the retained turn**; it cannot establish continued role holding after that boundary. | [raw-read finding](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:66) |
| **H6 — Scale changed once** | **Change** | Too vague. Separate the revised role scores from the revised broken-reply definition. Record each version and its exposure history. A delayed second pass tests owner stability. It does not test agreement between people. | [role-score revision](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:95); [broken-definition revision](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/THRESHOLDS.md:123) |

---

**Number and attribution corrections**

| Claim | What the records support |
|---|---|
| **E0: 9.8% wrong-role rate** | Correct: **50/510 controls**, pooled across base and instruct conditions. It is not a base-only rate. [Aggregate results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261005_base_incontext_elicitation/results_role.json:1) |
| **J8/J10: κ 0.16 and 0.35** | Correct for **owner score ≥2 versus naming success**, on **35 target items**. Against owner score 3, κ is about **0.35 and 0.40**. These are not direct four-class score agreements. [Results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/results/judge_designs_score.json:1) |
| **“When it names the role, you agree”** | True for owner score ≥2 in that sample. J8 finds **5/23** positives; J10 finds **10/23**. J10 also names five replies scored 2, so naming does not imply score 3. [Results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/NOTE.md:60) |
| **GPT/Fable κ 0.94 versus owner 0.53–0.61** | Compare the same items. On the first 50: GPT/Fable **0.944**, GPT/owner **0.57**, Fable/owner **0.61**. On the fresh 30: GPT/owner **0.64**, Fable/owner **0.53**. [Results](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/NOTE.md:113) |
| **14/40 candidate examples self-named** | These were the **chosen v1 examples**, before the revised selection rules. They are not the final 40 examples. [Selection history](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:32) |
| **Copies push the repeat below the bar** | The reported exclusion combines **copying and self-naming**. The text cannot assign the effect to copying alone. The 28 Oct result is in **judge-designs/NOTE.md**, not E3/NOTE.md. [Re-score](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/NOTE.md:142) |
| **κ ≥0.6 is the standing gate** | It was a historical gate. The owner rejected its justification on 23 Oct. Do not restore it as a scientific standard without a reason. [Decision record](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/THRESHOLDS.md:37) |

The zero wrong-role hits in 15 J8/J10 controls are encouraging. They do not establish a false-positive rate below 5%. With independent trials, zero in 15 gives a one-sided 95% upper bound of about **18%**.

---

**The context also contains one important error**

The 28 Oct mask is a **custom loop-and-nonsense analysis**. It is not an exact implementation of either cited coherence judge.

Betley scores coherence in relation to the question. Moskvoretskii describes understandable, on-topic continuation and coherence in context. Both therefore retain context sensitivity beyond detecting loops and nonsense. [Betley, Appendix B.4, Listing 2](https://arxiv.org/html/2502.17424v7); [Moskvoretskii, Appendix D.1, pp. 22–23](https://arxiv.org/pdf/2605.13329)

The **6 and 4 of 1,800** counts remain valid descriptions of that mask. They do not validate a new coherence judge. The mask also includes owner-broken replies without a stated reason. [Mask construction](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261023_judge_designs/NOTE.md:139)

---

**2. Literature fully read**

Search cutoff: **29 October 2026**. Dates below identify the versions read. Conceptual accounts and empirical tests have different evidential weight.

| Work, ID and date | Relevant evidence and location | Implication for the rules |
|---|---|---|
| **Han et al., “Meet Your Favorite Character.”** NAACL 2022 short paper; **2204.10825v1, 22 Apr 2022**. [Full text](https://arxiv.org/html/2204.10825v1) | §3 separates character style from response appropriateness. Table 3 tests pretrained GPT-J, GPT-Neo and GPT-2 XL. Appendix D.1, Table 11 reports substantial lexical reuse. | Supports document/dialogue elicitation and R4. Copy the separate human dimensions. Do not equate style imitation with deeper character fidelity. |
| **Shanahan, McDonell and Reynolds, “Role-Play with Large Language Models.”** **2305.16367v1, 25 May 2023**. [Full text](https://arxiv.org/html/2305.16367v1) | §§2–6 explain dialogue prompting, continuation, and the simulator account. §4 stresses that a model need not commit to one fixed character. | Supports R1 as a useful framework. It does not establish that any plausible document continuation counts as successful target-role enactment. |
| **Thakur et al., “Judging the Judges.”** GEM² 2025; **2406.12624v6, 18 Aug 2025**; first posted 18 Jun 2024. [Full text](https://arxiv.org/html/2406.12624v6) | §§3–5 compare human labels with judges on base and chat outputs. §4.1 distinguishes raw agreement from chance-corrected agreement. §§5.1–5.4 test error types, dummy answers, prompt sensitivity, and leniency. | Supports R6’s controls. Copy the confusion analysis. Its revised version uses Scott’s π; it does not establish a universal κ cutoff. |
| **Zhou et al., “PersonaEval.”** COLM 2025; **2508.10014v1, 6 Aug 2025**. [Full text](https://arxiv.org/html/2508.10014v1) | §3 supplies dialogue and candidate profiles. §3.4 filters short and easy cases. Table 1 reports best overall accuracy **68.8%**. §4.4 reports **90.8%** human accuracy on a selected hard subset. | Supports judge validation, close distractors, and insufficient-evidence handling. Contradicts R3’s blanket ban on descriptions. Its 69% result is not a general ceiling for today’s judges. |
| **Betley et al., “Emergent Misalignment.”** **2502.17424v7, 20 Jan 2026**; first posted 24 Feb 2025. [Full text](https://arxiv.org/html/2502.17424v7) | §3.2 and Appendix B.4 separate alignment and contextual coherence. §4.8 adapts evaluation for models fine-tuned from base checkpoints. Appendix C.2 checks threshold sensitivity. | Supports separate scores and format controls. It does not support “coherence means only loops and nonsense.” Copy the sensitivity analysis, not its task-specific thresholds. |
| **Lu et al., “The Assistant Axis.”** **2601.10387v1, 15 Jan 2026**. [Full text](https://arxiv.org/html/2601.10387v1) | §2.1 describes role filtering and vectors. Appendix B.3 runs base activations on **instruct-generated text**. §3.2.2 separately tests base steering with prefills. §3.2.1 attributes **91.6% agreement on 200 samples to harmfulness judging**. | Supports the project’s starting methods, with limits. Base representation evidence is not evidence of base generation fidelity. The 91.6% figure cannot validate the role rubric. |
| **Marks, Lindsey and Olah, “The Persona Selection Model.”** Anthropic research essay, **23 Feb 2026**; no arXiv ID. [Full text](https://alignment.anthropic.com/2026/psm/) | “Statement of the persona selection model” explicitly allows imperfect simulation and character breaks. “How exhaustive is PSM?” discusses competing accounts and uncertainty. | Supports R1 as a model to test. It argues against treating the simulator account as a complete, established mechanism. |
| **Moskvoretskii et al., “Tracing Persona Vectors Through LLM Pretraining.”** **2605.13329v1, 13 May 2026**. [Full text](https://arxiv.org/html/2605.13329v1) | §§2–3 separate elicitation success from steering effect. §6 and Appendix B compare document formats. Appendix F.3, Table 5 validates **widely separated score pairs**, below 20 versus above 70. | Supports format controls and separate coherence scoring. Its roughly 91% pairwise human agreement does not validate classification near 50. Four traits in OLMo-3/Apertus do not establish validity for 276 OLMo-2 roles. |
| **Arena Team, “Do LLM Judges Favor Themselves?”** Primary research report, **30 Sep 2026**, updated **6 Oct 2026**; no arXiv ID. [Full text](https://arena.ai/blog/llm-judge-self-preference) | “Methodology” uses 1,460 battles with swapped answer order. “Do AI judges have their own taste?” reports **79.4%** judge–judge versus **56.9%** judge–human agreement when comparing decisive choices. | Supports R6’s warning about consensus and testing H1/H3. These are preference results, not role-fidelity validation. The report is not peer reviewed. |

---

**Methods worth copying**

Use separate fidelity and response-quality labels. Include close alternatives, plain positive examples, and controlled bad examples.

Test several elicitation formats without assuming they recover the same direction. Keep extraction questions separate from evaluation questions.

Validate the boundary that the experiment actually uses. Easy comparisons between extreme scores cannot establish accuracy near a filtering threshold.

---

**3. Works found but not fully read**

I did not complete full-text review of these works. **I do not use their findings as evidence above.**

| Work | Version/date checked | Reading limit |
|---|---|---|
| [Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena”](https://arxiv.org/abs/2306.05685) | **2306.05685v4, 24 Dec 2023** | Full-text review not completed. |
| [Tu et al., “CharacterEval”](https://arxiv.org/abs/2401.01275) | **2401.01275v1, 2 Jan 2024** | Abstract and bibliographic record checked. |
| [Guerdan et al., “Validating LLM-as-a-Judge Systems under Rating Indeterminacy”](https://arxiv.org/html/2503.05965v4) | **2503.05965v4, 27 Oct 2025**, NeurIPS 2025 | Main methods and results read; full appendices not completed. |
| [Mukherjee et al., “The Geometry of LLM-as-Judge”](https://arxiv.org/abs/2606.03043) | **2606.03043v2, 2 Sep 2026** | PDF opened; full 30-page review not completed. |
| [Kang et al., “Pretrained Persona Mixture Models and Tandem Models for Human Simulation”](https://arxiv.org/html/2609.22607v1) | **2609.22607v1, 18 Sep 2026** | Full-text page opened; review not completed. |

---

**4. Missing rules**

**Define the claim before choosing the judge**

The text moves between three different claims:

| Claim | Required evidence |
|---|---|
| **Role identifiability** | A reader can infer a role from the reply. |
| **Target-role fidelity** | The reply expresses the specified role under its written definition. |
| **Original role enactment** | The reply expresses the role beyond copied material and identity claims alone. |

These claims can produce different labels for the same reply. Name the primary claim explicitly.

---

**Give uncertainty a real class**

“Compatible with the role” is weaker than “shows the role.” “Shows little role evidence” differs from “shows another role.”

A plain professional answer can be correct role behavior. It may still provide too little evidence to distinguish consultant, analyst, or assistant.

---

**Treat roles as overlapping definitions**

A different role name is not automatically a negative control. Merchant, pragmatist, and pirate can share behavior.

Use human-verified incompatible controls. Report close-role ambiguity separately from clear wrong-role errors.

Do not require a stereotype, accent, or catchphrase to establish a role.

---

**Keep the denominator fixed**

Removing short, copied, self-naming, or broken replies changes the population being measured. These exclusions can vary strongly across checkpoints.

Report all attempts first. Then report defined subsets and their sizes.

A tiny retained subset can look excellent while the model rarely produces usable role text.

---

**A3 requires stable meaning and measured error**

A3 does not require identical model-facing syntax. It requires the same meaning for a successful role response.

For a stage comparison, hold elicitation fixed or measure the format effect. Separate base/instruct validations alone cannot do this.

| Hold fixed or explicitly bridge | Why |
|---|---|
| Role definitions, questions, and role weights | Changing the role panel changes the target population. |
| Examples and their source | Examples can determine the expressed role. |
| Scored span and boundary parser | Different spans expose different evidence. |
| Judge version, rubric, and decision threshold | A stable name does not guarantee a stable instrument. |
| Treatment of generic, unreadable, and missing replies | Exclusions can create an apparent developmental trend. |
| Judge sensitivity and false-positive rate by stage | Identical judging code can make different errors on different stages. |

A judge that detects 80% of clear instruct roles but 40% of plain base roles creates a false stage difference.

Test errors within human label classes and length bands. Do not compare only overall agreement.

---

**Separate representation from behavior**

The **91.8% prompt-control decoding** is a strong warning. Role information in reply-token states can come from the context. [E3 context controls](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/NOTE.md:98)

Keep fixed-text representation tests as a separate result. Do not use them to declare successful role generation.

Failure to elicit readable role text also does not prove that the checkpoint lacks a relevant representation.

---

**Separate rules from hypotheses**

R5 and R6 are procedure choices. H1 and H3 contain causal hypotheses. R1 contains a framing choice.

Calling all of these “rules that could be false” hides their different status. State which observations could reject each empirical hypothesis.

---

**5a. Rewritten rule set**

1. **Define the target.** Role fidelity means observable expression of the specified role under a written role definition.
2. **Keep the meaning shared.** Use the same fidelity classes for base, SFT, DPO, and Instruct outputs.
3. **Specify the document task.** State who speaks, what response is expected, and which span receives a score.
4. **Supply necessary context.** Give the fidelity judge the question, retained reply, and role being assessed.
5. **Hide generation metadata.** Hide the checkpoint, condition, example answers, actual prompted role, and other judges’ labels.
6. **Separate dimensions.** Score fidelity, coherence, task engagement, copying, and speaker boundaries separately.
7. **Allow insufficient evidence.** Do not force generic or unreadable text into an out-of-role class.
8. **Require behavioral evidence.** A role name alone cannot establish enactment. An AI reference alone cannot disprove it.
9. **Develop, then seal.** Revise the rubric on development items. Freeze it before validation judgments.
10. **Check the reference.** Label validation items before seeing their judge predictions. Measure delayed owner agreement.
11. **Validate the decision.** Use class errors, controls, uncertainty, and stage-specific checks. Judge consensus cannot replace human comparison.
12. **Freeze comparisons.** Keep the population, extraction rules, and scoring protocol fixed across checkpoints.
13. **Limit the claim.** Naming measures identifiability. Fixed-text decoding measures representation. Neither alone establishes successful role enactment.

These rules permit different failure flags across models. They preserve one shared definition of success.

---

**5b. Rubric outline**

Use categories with names. Do not reuse old numbers with new meanings.

| Class | Definition | Typical case |
|---|---|---|
| **E — Enacted** | Clear evidence that the guest speaks or acts through the defined role. The evidence extends beyond a name alone. | A detective explains a choice through investigative practice and experience. |
| **P — Partial** | Some relevant role evidence, but weak or inconsistent enactment across the reply. | A generic answer with one role-specific viewpoint or action. |
| **G — Insufficient evidence** | Understandable and compatible with the role, but too generic to establish it. | “I would think carefully before deciding.” |
| **W — Conflicting role** | Clear behavior or identity conflicts with the defined role, or establishes another incompatible role. | A human character consistently answers as an unrelated software service. |
| **U — Unscorable** | No usable guest speech or interpretable evidence. | Empty text, wholly unintelligible text, or only another speaker’s turn. |

These classes are not one ordered scale. In particular, **G does not mean W**.

A partly broken reply can still contain enough evidence for E or P. Use U when no usable fidelity judgment is possible.

---

**Separate flags and outputs**

| Field | Rule |
|---|---|
| **Coherence** | Clear enough to interpret / severely degraded. Define destructive repetition separately from emphasis. |
| **Task engagement** | Engages / does not engage / unclear, under the written role task. |
| **Originality** | Record substantial copying using the frozen overlap rule and a manual audit. |
| **Identity claim** | Record self-naming. Do not treat it as sufficient evidence or automatic failure. |
| **Boundary** | Record the detected speaker switch, retained length, and stop reason. |
| **Evidence** | Require one or two short supporting spans from the reply. |

Score translator behavior under a fixed contract. Translation can express the role while failing the personal interview task.

Likewise, a robot or assistant can mention being an AI and remain fully in role.

---

**Primary reporting**

Report **E divided by all attempted completions** as the demonstrated-enactment rate. Keep G and U visible.

Also report coherent E replies and original coherent E replies, each divided by all attempts. Report question engagement separately.

For vector extraction, select the required subset explicitly. Check whether changing that selection changes the resulting map.

---

**5c. Validation design**

This design validates an operational measure. It cannot establish human consensus or precise validity for every class at every checkpoint.

**Sample plan**

| Set | Size | Purpose |
|---|---:|---|
| Development | **40 items** | Clarify definitions, inspect errors, and choose the primary judge. |
| Sealed natural outputs | **240 items** | Estimate errors on actual model replies. |
| Sealed controls | **120 items** | Test specific shortcuts and failure modes. |
| Delayed owner repeat | **80 of the sealed items** | Measure label stability after at least three days. |

This means **400 distinct items and 480 owner labeling actions**. My planning estimate is roughly **12–16 focused hours**, including rubric work.

Use 40 natural outputs from each of six groups: early base, middle base, final base, SFT, DPO, and Instruct.

Include the four base conditions. Include plain-document and native formats for trained stages where available.

Use stored outputs. If a required group is unavailable, narrow the validation claim; do not substitute another model family.

---

**Labelling procedure**

Use one human labeller: the owner. Call the result **validation against the owner’s labels**.

Label development items before development judging. After development, freeze the rubric and label the sealed set.

Repeat 80 items with new IDs and shuffled order. Resolve owner disagreements before exposing sealed judge predictions.

Record initial repeat agreement too. Do not report agreement only after adjudication.

---

**Control plan**

| Control | Count | Required test |
|---|---:|---|
| Wrong-role controls | **60** | Thirty close and thirty distant roles. Verify that each assessed role is actually unsupported or incompatible. |
| Positive controls | **30** | Human-verified E replies, including plain and short replies from both base and trained models. |
| Shortcut and format controls | **30** | Include name-only replies, copying, speaker changes, in-role refusals, and matched style variants. |

Do not declare an instruct output positive merely because it came from Instruct. Do not declare every different role negative.

Keep natural-output estimates separate from enriched control estimates. Controls do not reproduce production prevalence.

---

**Judges and isolation**

Use two existing CLI judges from different model families. Select the primary judge using development items only.

Give each judge only its item text and the fixed rubric. Enforce tool and file isolation through the harness.

Request structured labels and short evidence spans. Freeze retry rules and count invalid responses as validation failures.

Use the second judge to test robustness. Do not select whichever judge looks better on the sealed test.

---

**Agreement statistics**

| Target | Report |
|---|---|
| Full category labels | Confusion matrix and unweighted Cohen’s κ |
| Each class | One-class-versus-rest κ, precision, recall, F1, and sample count |
| Primary E decision | Precision, recall, false-positive rate, and uncertainty |
| Owner repeat | Initial category agreement and E-decision agreement |
| Base versus trained models | Errors within human class, length band, and format |
| Checkpoint rates | Judge-minus-owner rate difference, with uncertainty |

Use a bootstrap that respects role and question grouping. Keep matched variants and reused replies together.

Report undefined or poorly supported statistics as such. An absent class does not have perfect agreement.

---

**Proposed pass rule**

Fix these rules before sealed judge outputs. These are proposed error limits for this project, not standards established by the papers.

| Gate | Proposed requirement |
|---|---|
| Owner stability | At least **90% agreement** on the E decision in the delayed repeat |
| E precision | At least **90%**, separately for pooled base and pooled trained outputs |
| E recall | At least **80%**, separately for those two groups |
| Uncertainty | One-sided 95% lower bounds at least **80% precision** and **70% recall** |
| Wrong-role controls | **Zero E calls in 60 verified negatives** |
| Positive controls | At least **27/30** receive E |
| Other classes | Precision and recall at least **80%**, with at least **20 reference cases** per class |
| Invalid judge responses | No more than **1%** after the fixed retry rule |

Zero errors in 60 independent controls gives a one-sided 95% upper error bound of about **4.9%**. Shared replies or correlated controls weaken that interpretation.

Classes without enough cases remain unvalidated. Handle them manually or limit the published claim.

Do not change thresholds after failure. Revise on development data, then use a new validation set.

---

**The additional A3 gate**

Passing the detector gates does not automatically permit precise checkpoint comparisons.

For a claimed difference below ten percentage points, require the differential judge bias interval to fit within **±5 points**. Otherwise report manual estimates, wider bounds, or a weaker claim.

Audit additional checkpoints before interpreting their scores. A 40-item stage sample is a warning system, not proof of invariance.

If the deadline prevents stronger validation, publish the limited result plainly. Do not turn unchanged scoring code into evidence of unchanged measurement meaning.

---

**Schedule to 16 November**

| Dates | Work |
|---|---|
| **29–31 Oct** | Set the role contracts. Label development items. Select the judge and freeze the protocol. |
| **1–5 Nov** | Label the sealed items in short sessions. |
| **6–8 Nov** | Complete the delayed owner repeat. Seal the reference labels. |
| **9–11 Nov** | Run the fixed CLI validation and compute the registered statistics. |
| **12–15 Nov** | Write the supported result. Use manual scoring or narrower claims if gates fail. |

This leaves time for analysis and writing. It avoids depending on a large new judge run to rescue the paper.

---

**6. Open questions — Next step**

Only the owner can settle these choices:

1. Is the primary claim role fidelity, role identifiability, or original role enactment?
2. Should a translator transform the interview question, or answer it through a translator’s personal viewpoint?
3. What minimum checkpoint difference and retained role count would change the paper’s conclusion?
4. Which OLMo-2 7B checkpoint outputs will be available for sealed validation before 5 November?