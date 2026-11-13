# Fable 5.2 adversarial review (12/11/2026), final report

Reviewer: Fable 5.2 (`claude-fable-5-2`), Agent subagent of session ad361f36, brief `prompt_adversarial_261112.md`. Agent review, not a human review. This second report replaces the first: after the owner's "always read fully" rule, the reviewer re-read its sources in full and changed three numbers. Read depth is in the last table. Verbatim below.

---

**Verdict:** The sentence is not ready. "Privileged" has no non-trivial test, condensation has no valid OLMo result, and the run cannot fill "[stage]".

Paths: `R` = `/Users/brendon/persona-manifolds/manifold-persona`, `V` = `/Users/brendon/research-log/manifold-persona-log`, `Q` = `R/experiments/q1_emergence`.

**1. Already answered? Partly. Claim 1 is too strong.**

| Work | What it measured | Effect |
|---|---|---|
| Lu et al., arXiv 2601.10387v1, 15 Jan 2026 (`R/output/lit_q1_2026-10-05/2601.10387.txt`) | Three instruct models. The default sits 0.03 from the edge of role PC1 (§2.3.1). Base Gemma 2 27B reads instruct replies: top PCs match at 0.93, 0.87, 0.83, default at one end (App. B.3). In base Llama 3.1 70B the default is not at one end (App. G.2). | Base position is done on fixed text only, with mixed results and no text baseline. |
| Han Xiao, LessWrong, 5 Nov 2026 (`R/output/lit_stages_2026-11-12/lw/axis_shift_stages_clean.md`) | Olmo 3 7B SFT, DPO, RLVR. 48 desirable, 25 undesirable, 201 neutral roles (figure data). Desirable +0.19 at DPO, p = 0.066. Full role-play pass rate 0.26, 0.52, 0.53. | Position across stages is done on a sibling model. It measures role movement, not when the Assistant becomes privileged. |
| Li et al., arXiv 2509.23024v1, NeurIPS 2025 | Effective rank on Llama-3.1-Tülu-3.1-8B: up in SFT and DPO, down in RLVR (§3.4, Fig. 6A, no numbers). OLMo-2 in pretraining only. | Global condensation across these stage types exists under another name. The repo has it (`Q/261030_stage_manifold_test/LITERATURE.md:16`). Critique section 6 omits it. |
| PersonaManifold, arXiv 2609.34571v1 | One mean vector per persona. Intrinsic dimension 15–23. Instruct only. | No Assistant cloud, no stages. |
| Asvin G., arXiv 2606.00545v1 (`R/output/lit_stages_2026-11-05/papers/2606.00545.txt`) | Llama-3.1-70B-Instruct. The Assistant is a top-3 reference for 7 of 8 distinctive evaluators, among 23 candidates (§5). | "Privileged" already has a behavioural test. |

I found no per-persona Assistant condensation across stages in two web searches. That is a thin search, not proof.

**2. Claim 2 is mostly false.**

| Result | Measures |
|---|---|
| Pod A D1, D2 | Naming and a 20-role map (`Q/261029_instruct_persona_map/review/challenge_261106.md`) |
| Pod A D3 | Position, one stage, gate failed: 0.881 [0.763, 0.884] (same file) |
| Pod B | Role against format, decoding (`Q/261029_base_role_vs_format/review/challenge_261105.md`) |
| Pod C | Fixed-text role map. Axis not built (`Q/261029_fixed_text_read_through/NOTE.md`) |
| Condensation | Nothing. T-065 "Revisit assistant condensation, team, No files" (`V/Weeks/2026-W40.md`) |

The same idea ran on OLMo-1B in August with Trini's metrics and failed the audit (`LITERATURE.md`, team table). The tightness result reversed without the scaffold, 5.5 to 23 (`V/Weeks/2026-W36.md`). The September study already used 276 × 5 × 40 = 55,200 replies per stage (`R/output/independent_review_2026-10-02/review.txt`, C01). Claim 2 is true for the meeting thread only.

**3. "Privileged" is not defined.**

| Measure | Trivial way to pass |
|---|---|
| Position | The axis is default minus roles, so the default is extreme by construction. "A fixed offset passes this rule with no role differences" (pod A challenge). |
| Position | The prompted "assistant" role gives the same axis: cosine 0.936 (`V/Reference/2026-10-30 A-B pilot results/full_A_results.json`, `primary.assistant_control`). |
| Position | The role prompt alone decodes the role at 91–98% (pod B challenge). Default prompts are short or empty. |
| Condensation | Formulaic default text. An n-gram cloud can be as tight. |
| Condensation | Reply length. 59% of pod A replies hit the 128-token cut (`Q/261029_instruct_persona_map/NOTE.md`). On Qwen, reply length tracks axis projection, r = +0.627 (`V/Weeks/2026-W31.md`, E8 length control). |
| Condensation | Five mixed default wordings, one with no system message (pod A `NOTE.md`). |
| Condensation | The estimator. The review rates "lower dimension near the assistant" partly supported, with MLE and TwoNN marked untrusted (C102). The global rank also shifts per stage (Li et al.). |

**4. The run cannot answer the sentence.**

- "[stage]" is censored. The run has no base replies. If SFT shows the effect, the answer is "at or before SFT". Thomas called only the base case unclear (`V/Reference/2026-11-12 Meetings 16-30 Oct.md`).
- "Confounds removed" is false. No filter swaps selection for compliance mixing (review F3, C22). On Olmo 3 only 26% of SFT replies are full role play. Whether OLMo-2 is similar is a guess.
- The power argument misreads Han Xiao. His interval comes from question half-splits (his footnote 2). This run has 40 questions against 240.
- The last clause has a foreseeable answer. Pod C gives r_norm 0.996–1.000 across readers, and n-grams reach 0.960 (`Q/261029_fixed_text_read_through/review/challenge_261030.md`).
- One reply per cell gives 200 points per role. In E3 base, question holds 35% of reply variance and the two-seed residual 15–26% (pod A `NOTE.md`). One record per cell cannot separate them (C98). This supports a trend over 276 roles. It does not support one default cloud.
- A null means nothing: other model, no positive control.

Smallest decisive design, no GPU: use pod A states (21 clouds × 1,000) and pod C states (10 readers). Test the default's place on role-only PC1 (Lu §2.3.1) and its cloud rank against the 20 roles. Add every role as anchor (Asvin §5), the first-32-ids frame, an n-gram floor and a 1-replicate subsample. Generate SFT and DPO only if the default beats the text floor.

**5. Better sentence**

"In OLMo-2-7B, most of what looks like a persona representation is the prompt and the reply text; a stage claim about the Assistant needs a fixed-text control and a text baseline, and Lu et al. and Han Xiao lack one." Evidence: critique section 5.

**6. Ignored asks**

| Ask | Status |
|---|---|
| Thomas, 30 Oct: structure "before running any more experiments" | Critique notes it (4.7), then ranks a GPU run first. |
| Thomas, 30 Oct: base model, other privileged personas | No base arm. |
| 17 Sep split: manifold existence is Trini's part | Condensation duplicates it. Her curvature null code exists (`LITERATURE.md`, team table). Critique 1.1 says no curvature code exists. |
| Judge budget $150 plus $50 | Terra labels up to $120 exceed it (critique 4.9). |

**Where the teammate is right:** Freeze the rule first. Equal reply counts. Han Xiao's weak points are real. 16 Nov is self-set. T-056, the rerun of trained stages without example turns, has no files, and this run would close it.

**Read depth per source**

| Source | Depth |
|---|---|
| Brief, critique, meetings page, `Meetings.md`, `Claims and verdicts.md`, 5 Nov literature page | Full |
| Pod A challenge and `NOTE.md`; pod B challenge; pod C challenge and `NOTE.md` | Full |
| Independent review `review.txt` | Full (file locations in the claim ledger skipped) |
| `gen_extract.py` | Full |
| `LITERATURE.md` (stage manifold test) | Full. Its team table rests on README text, so Trini's numbers are unchecked. |
| Weeks W31, W36, W40 | Full |
| Lu et al. | Main text and App. A–C, G full. App. D–F: numbers and headings; judge prompts and example replies skimmed. Figures not seen. |
| Han Xiao | Full text, footnotes and embedded figure data. Layer 8 and 24 figures are images, not seen. |
| Li et al. | Full, arXiv HTML v1. Proofs B.2–B.5: headings only. Fig. 6A values not readable. |
| PersonaManifold | Full, arXiv HTML v1. Figures not seen. |
| Asvin G. | Full text. Figure values garbled. |
| `full_A_results.json` | Keys `primary.D3`, `assistant_control`, `spectrum` only |
| Pod A `analysis.py` | Not read fully: `axis_statistics` and the control block only. No claim rests on the rest. |
| Pod C `extract.py`, `frames.py`, `analysis.py`, `THRESHOLDS.md`; `Axioms and evidence.md`; `DESIGN.md` | Not read fully. No claim rests on them. |
| Vault note for 2606.00545, 2607.09842; talkie blog; other search hits | Not read fully. Not used. |
