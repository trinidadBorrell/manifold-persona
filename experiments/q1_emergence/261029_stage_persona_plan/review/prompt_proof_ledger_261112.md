# Adversarial check of a proof ledger (12/11/2026)

A teammate wrote the ledger below and says every number is checked. Assume they made mistakes. Find them. Be blunt and specific, and stay truthful. Agree only where you checked the file yourself.

Rules: read only. Do not edit, create or delete any file. No GPU work, no pods, no paid API calls. Small local reads only (`cat`, `sed`, `rg`, `python3 -I` on JSON). Read each source you cite in full where it is text; for large JSON, read the keys you use and say which.

## Inputs

- Ledger: `experiments/q1_emergence/261029_stage_persona_plan/review/proof_ledger_261112.md`
- Draft it corrects: `experiments/q1_emergence/261029_stage_persona_plan/overview_thomas_v2_261112.md`
- Persona Geology post (full text): `/private/tmp/claude-501/-Users-brendon-persona-manifolds-manifold-persona/ad361f36-2648-47d6-95a7-4e43e5c9a47c/scratchpad/geology/geology.md`
- Han Xiao post: `output/lit_stages_2026-11-12/lw/axis_shift_stages_clean.md` (two long lines are figure data)
- Paper texts: `output/lit_stages_2026-11-12/txt/` (2601.10387 Lu; 2605.25459 Asvin G. and Lindsey; 2609.23935 Nguyen; 2509.23024 Li)
- Vault: `/Users/brendon/research-log/manifold-persona-log/Reference/`

## Tasks

1. For each claim C1-C11 and L1-L6: open the cited file and key, and check the number, interval, population and wording. Report any mismatch with the exact value you found.
2. For each claim, look for evidence Y in the repo, the vault or the cited texts that discounts it and that the ledger missed. Say whether Y discounts X, and why.
3. Check the code path for C3, C4 and C5: does the analysis code compute what the ledger says (held-out format, noise correction, planted change)? Cite `file:line`.
4. Related work: for L1-L6, is the summary faithful to the source text? Is "supports", "contradicts" or "neither" correct? Name any claim that overstates the source.
5. Novelty: given Persona Geology, Han Xiao, Lu et al., Asvin G. and Lindsey, and Nguyen, which of our results (C3-C10) are already known, and which add something? Be specific.
6. Behavioural vs structural: for each claim, say whether it measures behaviour (model output, judged or scored) or structure (activations or weights). Name any place where the ledger or draft infers behaviour from structure.

## Output

Short sentences, plain words. One table per task: claim id, finding, evidence (`file:line` or file + key + value), severity (wrong / overstated / missing / ok). End with the three most important fixes. Then a read-depth table: each source and how much you read.
