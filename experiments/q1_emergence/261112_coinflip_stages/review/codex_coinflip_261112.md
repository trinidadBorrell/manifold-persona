**Answer: The negative score reproduces. The claims about replication, absence of bias, and when bias starts need narrower wording.**

I inspected the code and saved JSON, recomputed the four available cells, and ran the 12 CPU tests. All 12 passed; no model inference or GPU code ran.

---

**1 — High: The primary chat condition changes Nguyen’s prompt content**

Nguyen says the two modes use “the same content,” but this implementation retains `Human: ` in plaintext and removes it in primary chat mode ([paper:74](/Users/brendon/persona-manifolds/manifold-persona/output/lit_stages_2026-11-12/txt/2609.23935.txt:74), [coinflip.py:77](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/coinflip.py:77)).

| DPO frame | Bias | Nominal 95% interval |
|---|---:|---:|
| Primary chat, without `Human:` | −0.065097 | [−0.099719, −0.030475] |
| Chat retaining `Human:` | −0.016872 | [−0.052120, +0.018376] |

These values come from [summary.json:38](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/results/full/summary.json:38); the paragraph-matching test validates the plaintext example, not content equivalence across modes.

**Smallest fix:** Call this a modified replication and put both frame estimates beside the headline; obtain a paired frame contrast before claiming a statistically established frame difference.

---

**2 — High: “Installs” does not establish onset, and “no bias” does not follow from these tests**

The [reading rule](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/THRESHOLDS.md:30) defines a significance threshold, not a test that an effect was previously absent.

| Claim | Evidence against that reading | Smallest correction |
|---|---|---|
| “Base has no bias” | Bias +0.015175; interval [−0.000295, +0.030645]; nominal *p*=0.0536. | “No detected harmless-option bias under this test.” |
| “DPO installs the negative bias” | SFT is already negative: −0.020695 [−0.042164, +0.000774]. | “The score shifts further negative between SFT and DPO.” |
| Generic onset at DPO | One-paragraph chat already detects a negative SFT score: −0.029776. | Restrict the threshold statement to the primary frame. |
| “Assistant preference” | The experiment measures user-continuation token probabilities; it never measures assistant task choices or refusals. | Describe a negative harmless-option token score. |

The first three rows are supported by [summary.md](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/results/full/summary.md:3); there is also no base chat cell for an unchanged-frame onset comparison.

The paired SFT→DPO change is detectable, but it does not identify a harmful preference, a persona, or a causal mechanism.

---

**3 — Medium: The intervals are approximate and pointwise**

Ten clusters per dimension provide weak grounds for treating nominal 95% coverage as exact, especially with deliberately selected tasks; [CGM’s own simulations](https://cameron.econ.ucdavis.edu/research/JBESpaper2009version.pdf) flag problems at ten-by-ten clusters.

The [rules](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/THRESHOLDS.md:30) do not define a multiple-testing family despite searching stages, modes, changes, and layers.

| Holm adjustment family | DPO chat endpoint | Instruct chat endpoint |
|---|---:|---:|
| Seven primary endpoints | 0.0149 | 0.0199 |
| Twelve primary endpoints and changes | 0.0234 | 0.0331 |
| All 29 reported endpoints and changes | 0.0533 | 0.0794 |

Thus, multiplicity does **not** erase the primary result under the narrower families, but “passes the rule” depends on the family chosen.

**Smallest fix:** Label intervals approximate and pointwise; define the hypothesis family before further testing.

---

**4 — Medium: The output endpoint uses the lens, and its agreement check cannot fail**

The analysis uses `pH[-1]` and `pT[-1]` from the float32 lens; it stores native model probabilities separately ([coinflip.py:168](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/coinflip.py:168), [coinflip.py:228](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/coinflip.py:228)).

For DPO chat, the largest native-versus-lens outcome-probability difference is **0.015580**, and the largest item bias difference is **0.031886**; [the check only records this gap](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/coinflip.py:207).

This does **not** explain the negative result: native logits give **−0.064939 [−0.099122, −0.030755]**.

**Smallest fix:** Report native logits as the output endpoint, retain the lens separately, and replace “equals” with quantified agreement.

---

**5 — Medium: The local archive cannot reproduce the full report, and resume checks permit mixed runs**

The local folder contains raw data for **four of 17 cells**; the other 13 cells can only be checked against their saved summary ([NOTE.md:11](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/NOTE.md:11)).

Existing files are skipped without validating their hashes, paired comparisons check only item IDs, and analysis accepts incomplete stage chains ([coinflip.py:202](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/coinflip.py:202), [coinflip.py:270](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/coinflip.py:270), [coinflip.py:279](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/coinflip.py:279)).

Consequently, a later resume could silently combine layouts or code versions; **no such mixture was found in the four available cells**, whose code hashes, task hashes, texts, and mappings match.

**Smallest fix:** Copy all raw cells and reject missing cells, mismatched inputs, or mismatched run metadata before analysis.

---

**6 — Low: NOTE.md has several reporting errors**

| NOTE location | Audit result |
|---|---|
| [Main table, lines 17–20](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/NOTE.md:17) | All **51 displayed numbers** match `summary.json` after rounding. |
| [Stage changes, line 22](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/NOTE.md:22) | Numbers and stated interval exclusions match; all DPO→Instruct changes have magnitude below 0.004. |
| [Mass range, line 24](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/NOTE.md:24) | Trained range is **0.934868–0.978131**; “0.94 to 0.98” overstates the lower bound. |
| [Checks, lines 28–31](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/NOTE.md:28) | Example probabilities, task-drop range, and nuisance terms reproduce; native/lens “equality” holds only after rounding. |
| [Lens, lines 35–36](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/NOTE.md:35) | Peak layers and values match the summary; Instruct’s output is **−0.064159**, not the shared shorthand −0.065. |
| [Conversion, line 44](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/NOTE.md:44) | DPO \(P(\text{harmless})=0.467451\); “about 0.47” matches. |

The 12-test count, 400 items per cell, and 32 layers were checked; discarded-run history and the precise LessWrong comment date are not established by `summary.json`.

The [summary labels its rules v1.1](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/results/full/summary.json:3765), although the run uses v1.2’s paragraph layout.

**Smallest fix:** Correct those descriptions and distinguish summary agreement from independent raw-data verification.

---

**Requested implementation checks**

| Check | Finding |
|---|---|
| Items, control, sign | **None found.** All four saved cells contain 100 pairs × four balanced mappings; task-to-outcome assignment matches the text. |
| Metric | **None found.** \(q=p_H/(p_H+p_T)\); harmless probability is \(q\) or \(1-q\); bias is mean \(2P(\text{harmless})-1\). |
| Variants | **None found.** Five distinct single-token variants match Nguyen’s Appendix B; `" Tails"` requires two tokens and is excluded. |
| BOS and chat structure | **None found.** Exactly one BOS, no assistant opening, and the final user newline is removed; this matches the [OLMo-2 template](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct/raw/main/tokenizer_config.json). |
| Task transcription and layout | Content matches Appendix A after normalization; no duplicated slot periods were found. “Verbatim” is too strong for normalized apostrophes and ellipsis spacing; the example’s paragraph layout matches [Marks et al.](https://alignment.anthropic.com/2026/psm/). |
| Position and label terms | **None found.** The two conditional differences in \(q\) match Nguyen’s definitions; they do not rule out task-dependent interactions. |
| CGM mean SE | **None found.** The implementation uses \(V_h+V_x-V_{hx}\), residual cluster sums, and division by \(n^2\). |
| Small-sample factors | **None found.** Factors are \(10/9,\ 10/9,\ 100/99\); the additional OLS factor \((n-1)/(n-K)\) is one for an intercept-only mean. |
| Fallback and t critical value | The larger-one-way fallback is a declared custom rule, not CGM’s eigenvalue repair; it activates in no reported endpoint/change or available layer. \(t_{.975,9}=2.262157\) matches Nguyen. |
| Paired stage SE | **None found.** It clusters item differences rather than combining independent-stage SEs. |
| Lens hooks and normalization | **None found.** Hooks take decoder outputs before final normalization; float32 final RMSNorm and unembedding match the stated lens method and [Transformers implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.47.1/src/transformers/models/olmo2/modeling_olmo2.py). |

---

**Recomputed saved results**

These calculations use the four [raw cell files](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_coinflip_stages/results/full); \(b\) means average normalized heads probability, **not** harmless probability.

| Cell | Bias | Nominal 95% interval | \(b\) | Captured mass |
|---|---:|---:|---:|---:|
| Base plaintext | +0.015175 | [−0.000295, +0.030645] | 0.623920 | 0.778076 |
| SFT chat | −0.020695 | [−0.042164, +0.000774] | 0.582229 | 0.958823 |
| DPO chat | −0.065097 | [−0.099719, −0.030475] | 0.455325 | 0.973156 |
| Instruct chat | −0.064159 | [−0.100819, −0.027499] | 0.424504 | 0.973781 |

SFT→DPO is **−0.044402 [−0.059777, −0.029028]**; DPO→Instruct is **+0.000938 [−0.002920, +0.004796]**.

DPO remains negative in **90/100 pair averages**, and dropping any single task gives **−0.072104 to −0.056807**.

---

**Artefact explanations and checks**

| Explanation | What the saved data establish | What would discount the remaining explanation |
|---|---|---|
| Constant tails preference | It cancels under the balanced design; centering the heads/tails log odds still gives DPO bias **−0.066003**. Both task-order halves are negative. | Neutral outcome labels and matched task controls would test label-dependent interactions. |
| Prompt frame | Retaining `Human:` gives a much smaller endpoint estimate whose interval includes zero. | Recover raw sensitivity cells, calculate paired frame contrasts, and repeat fixed paraphrases. |
| Tokenization after `came up` | Space-prefixed lowercase outcomes preserve the prompt tokenization; using only those outcomes gives **−0.065010**. | Further spelling checks cannot alone overturn this observed mean: allocating **all omitted mass** toward harmless yields an upper bound of **−0.036392**. |
| Task wording, length, or genre | Counterbalancing does not separate harmfulness from other task features. The original 5×5 subset gives **−0.056601**, but its df-4 interval crosses zero: **[−0.117478, +0.004276]**. | Use independently prepared paraphrases, matched benign tasks, and benign–benign/harmful–harmful controls. |
| Numerical readout | Native model logits retain the negative endpoint and negative SFT→DPO change. | This explanation is already discounted for the recorded sign; it still requires explicit numerical tolerances. |
| Harmful assistant preference | No assistant preference, refusal, or harmful execution was measured. | Measure assistant-turn choices separately on the same task pairs. |

The omitted-mass bound is a bound on the saved finite-set mean, **not** a confidence interval.

**Draft verdict: Yes—as a prompt-specific exploratory negative token score after DPO, with both frames reported; “DPO installs harmful preference” is unsupported.**