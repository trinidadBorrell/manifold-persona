**Verdict:** The historical question is real. The novelty claim is overstated, and the proposed run cannot establish the proposed sentence.

---

**1. What prior work already answers**

| Source and date | What it measured |
|---|---|
| [Lu et al., 15 Jan 2026](https://arxiv.org/html/2601.10387v1) | Role-mean PCA, default position and steering in Gemma-2-27B, Qwen3-32B and Llama-3.3-70B instruct models; Gemma base also reads instruct rollouts, without intermediate stages. |
| [Han Xiao, 5 Nov 2026](https://www.lesswrong.com/posts/xxCtj4rhrBoSeGy4g/how-the-assistant-axis-shifts-across-post-training-stages-1) | Assistant-axis projections for 275 roles across Olmo-3-7B SFT→DPO→RLVR; judge-filtered own replies, no base. |
| [Asvin G., 30 May 2026](https://arxiv.org/html/2606.00545) | Cross-persona authorship claims, entropy and activation distance in Llama-3.1-70B-Instruct; no stage ladder. |
| [Li et al., 27 Sep 2025](https://arxiv.org/html/2509.23024) | Last-token effective rank and spectral decay across Tülu-3.1/Llama-3.1-8B SFT, DPO and RLVR checkpoints; additional OLMo-2-1B SFT experiments. |
| [Liu et al., 28 May 2026](https://arxiv.org/html/2605.30524) | Effective rank, participation ratio and anisotropy on fixed texts across instruction, domain, safety, reasoning and preference-tuning chains; principally Qwen2.5-1.5B and TinyLlama-1.1B. |
| [Xu et al., 28 Sep 2026](https://arxiv.org/html/2609.34571) | Dimension and graph curvature **between persona centroids** in Llama-3.1-8B, Qwen2.5-7B and Mistral-7B instruct models; no stage ladder. |
| [Costa and Vicente, revised 5 Nov 2026](https://arxiv.org/html/2605.12850v3) | Participation ratio across 1,000 persona-prompt vectors in original, secure-code and insecure-code Qwen3-235B variants; “original” already means post-trained. |

Thus, dimensional change across post-training is already studied; **Assistant-specific within-cloud condensation across this ladder** remains a possible narrower contribution, not established novelty. Han’s result warrants replication: its label-shuffle p is 0.066, and its stage-dependent filtering changes the selected population. [Han, 5 Nov](https://www.lesswrong.com/posts/xxCtj4rhrBoSeGy4g/how-the-assistant-axis-shifts-across-post-training-stages-1)

---

**2. What this repo actually measured**

Claim 2 is **partly true as research history, false as a description of completed condensation measurements**.

| Result files | Actual measurement |
|---|---|
| E0 `results.json`; E3 `part2_run_main_desc.json` | Naming, decoding, maps and default-contrast stability; E3’s axis criterion fails. [E0, 5 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261005_base_incontext_elicitation/results.json), [E3, 16 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261008_base_persona_map/results/part2/part2_run_main_desc.json) |
| A `results.json`, `d1_miss.json` | Map repeatability, naming and contrast stability; D3 misses precision. [Verdict, 6 Nov](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/review/challenge_261106.md) |
| B `D1.json`–`D3.json` | Role versus format variance, decoding and map agreement. [Review, 5–6 Nov](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_base_role_vs_format/review/challenge_261105.md) |
| C `results_full_v2.json` | Fixed-text role maps; its dimension summary concerns **role means**, and its axis is unavailable. [Results, 30 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/records_full/analysis_full_v2/results_full_v2.json) |

None establishes within-Assistant dimension or curvature across stages; stored states merely permit future analysis. [A analysis](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/analysis.py:575), [C analysis, 30 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/analysis.py:1)

---

**3. “Privileged” is undefined**

My methodological objection: a default-minus-roles axis points toward the default by construction; that alone establishes no privilege. [A axis code, 30 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/analysis.py:212)  
Position, intrinsic dimension, effective rank and curvature require separate definitions and decision rules. [Li, 2025](https://arxiv.org/html/2509.23024), [Xu, 2026](https://arxiv.org/html/2609.34571)  
Formulaic replies, refusals, duplicate answers, language, length, token pooling and unequal density are alternative explanations; equal counts do not remove them. [Raw replies, 29 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/run_full/raw_read.md), [independent review F9/F13, 2 Oct](/Users/brendon/persona-manifolds/manifold-persona/output/independent_review_2026-10-02/review.txt)  
Even “default” mixes no system message with four identity instructions. [Generator, 29 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:4)

---

**4. The run cannot identify the sentence**

A fixed reader holds weights constant while texts change; identical texts across readers provide the complementary comparison, and writer–reader interactions remain possible. [Independent review F3, 2 Oct](/Users/brendon/persona-manifolds/manifold-persona/output/independent_review_2026-10-02/review.txt:32)  
Base reading Instruct defaults measures its response to borrowed text, not its own default or another naturally privileged persona. [C scope, 30 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/THRESHOLDS.md:95)  
The teammate is right about an unfiltered primary, but this retains compliance differences rather than removing them. [Plan, 12 Nov](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_stage_persona_plan/review/critique_261112.md)

**My minimum design recommendation**

| Element | Requirement |
|---|---|
| Stages | SFT/DPO for a DPO contrast; all four for onset, including base-generated text. |
| Cross-reading | Each reader processes identical texts from each writer. |
| Pilot | Twelve stratified roles, separate default and prompted-assistant conditions; 40 questions × five wordings × three seeds. |
| Controls | Role-free question/reply and reply-only frames; chat-format bridge; matched lengths, text baselines, manual fidelity checks and pseudo-default controls. |
| Precision | Freeze a meaningful effect δ; scale only after pilot intervals and estimator calibration justify it. |

These are proposed starting counts, **not a power claim**. With one reply per cell, each persona has 200 points and centered rank at most 199; more roles do not increase that cloud’s sample size, and stochastic error cannot be separated within cells. [Grid code, 29 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:48), [F9, 2 Oct](/Users/brendon/persona-manifolds/manifold-persona/output/independent_review_2026-10-02/review.txt:89)  
A narrow equivalence interval could exclude changes exceeding δ under this protocol; a wide null establishes nothing about absence of privilege.

---

**5. A defensible post sentence**

“In OLMo-2-7B, base and Instruct recover similar 20-role maps from identical Instruct replies, and character n-grams recover nearly as much agreement.”

The measured correlations are 0.896 and 0.846, respectively; neither dates Assistant formation. [Actual results, 30 Oct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/records_full/analysis_full_v2/results_full_v2.json:465)

---

**6. What the plan sidelines**

The teammate correctly restores the July–October privileged-Assistant thread. Thomas nevertheless asked for an outline before further experiments; the record does not establish whether one was sent. The division was Trini’s geometry, Gio’s reach/classifier structure, then Brendon’s stage extension once that structure worked. The plan leaves the base “natural persona,” classifier boundaries, held-out geometry tests and steering asks unanswered; it should state those limits explicitly. [Meeting record, 12 Nov; recordings unchecked](/Users/brendon/research-log/manifold-persona-log/Reference/2026-11-12%20Meetings%2016-30%20Oct.md)