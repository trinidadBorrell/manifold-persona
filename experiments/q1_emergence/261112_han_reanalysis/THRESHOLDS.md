# Re-analysis of Han Xiao's public release: rules (v1.0, 12/11/2026)

Written before any number in parts B to D was computed. Owner, 12 Nov: "yes, we can do both" (this re-analysis and the OLMo-2 coin-flip test), for a post released as concurrent work with Han Xiao's.

## Data

- Release: `han-xiao/assistant-axis-shift` (CC-BY-4.0), revision `bb109847996bcfad452b9fb24c9c9451cbe909ab`: per-role vectors, axes, judge scores (GLM-5.3-flash, 0 to 3) and layer-16 per-response activations for Olmo 3 7B Instruct at SFT, DPO and RLVR. 275 roles plus `default`, 5 system prompts x 240 questions per role. No base model.
- Labels: his `configs/role_desirability.json` at commit `6f28e91fab76822b1559a3df850b774f648015f9` (49 desirable, 25 undesirable, 201 neutral; 52 marked borderline).
- Layer 16 only (his hook on block 16; our hidden-state index 17). `.pt` files are loaded with `weights_only=True`.
- This is Olmo 3, not OLMo-2. Every result is labelled with the model.

## Part A: reproduction (a gate, not a claim)

- From his released role vectors (judge score 3 only, as his pipeline) and his axis convention (axis = default minus the mean role vector, origin = the mean role vector), compute each role's projection at SFT and DPO, the mean change for desirable and undesirable roles, and the label-shuffle p (10,000 shuffles, two-sided, +1 correction; his code).
- Pass: desirable change within 0.01 of his figure value (+0.183) and p within 0.02 of his 0.066. If it fails, stop and report the mismatch.

## Part B: unit check (exploratory)

- The figure-data version of this check was seen before these rules (raw gap +0.172, p 0.071; role-SD gap -0.022, p 0.886).
- On the activations: desirable shift, undesirable shift and their gap, SFT to DPO and DPO to RLVR, in raw units and in role-SD units (projections z-scored over all roles present in both stages, within each stage). Uncertainty: his method, 100 random question halves (SE = SD of half differences / 2); p from 10,000 label shuffles. Also with borderline labels removed.
- No pass rule. Both units are reported side by side.

## Part C: at which stage is the default set apart? (exploratory)

- C1, position on the main role axis (Lu et al. §2.3.1): at each stage, PC1 of the centred role vectors (roles only, his judge-filtered vectors). Report the default's place as (i) its distance to the nearer end of the role range, on a 0 to 1 scale (Lu: 0.03), and (ii) a signed z against the role projections, with each stage's PC1 sign aligned to the RLVR stage's PC1 and the RLVR default side set positive.
  - Reading rule: "the default sits at an extreme" at a stage if the upper 95% bound of the distance is at most 0.05, or the default lies outside the role range in at least 95% of resamples.
- C2, spread: at each stage, the within-system-prompt variance of each cloud's per-response activations (all responses for every cloud, default included, so the judge filter does not differ between default and roles), as the log of the default's variance over the median role's variance.
  - Reading rule: "the default cloud is tighter" if the upper 95% bound is below 0.
- C3, stage change: SFT to DPO and DPO to RLVR, paired differences of the C1 signed z and the C2 log-ratio, from the same resamples. A change is reported as present only if its 95% interval excludes 0.
- Uncertainty for C1 to C3: 1,000 bootstrap draws over the 240 questions (shared across stages, so differences are paired), percentile intervals; roles fixed (all roles), because the role set is the population of interest here.

## Part D: judge-filter sensitivity (exploratory)

- Rebuild every role vector from all responses (no judge filter) and repeat Part A's shifts and Part C1. Report next to the filtered results.

## What these results can and cannot say

- They describe Olmo 3 7B's own replies at SFT, DPO and RLVR. They say nothing about the base model or about OLMo-2.
- All parts except A are exploratory: Han Xiao's results were known before these rules, and many statistics are reported without a family correction.
- Agent analysis; a human review and `/challenge-result` come before any external claim.
