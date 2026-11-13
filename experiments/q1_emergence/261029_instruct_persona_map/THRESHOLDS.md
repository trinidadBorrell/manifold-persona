# Pod A `instruct_persona_map`: measures, pass rules and gates

Status: **v1.5, 05/11/2026. Owner accepted the proposed values and the run-count plan; v1.4 and v1.5 record the D1 judge-run approvals and the judge change ("Changes").** No code, no run. Design: `../261029_stage_persona_plan/DESIGN.md`. Fixed before the run; later changes get a dated line in "Changes".

## Question and scope

Does the Assistant Axis setup replicate on OLMo-2-7B-Instruct without example turns? Pod A also makes the fixed text for pod C and the final-model reference.

- Claim scope: role information and a repeatable default-vs-role contrast **in Instruct reply states read in their own chat frame**. The system prompt is in context, so these states can carry the role from the prompt (E3 C3). Reply-only results at Instruct come from pod C, not from here.
- Not claimed: a persona, an identity, a causal function, or how well the reply enacts the role (fidelity is not validated yet).

## Data

- `allenai/OLMo-2-1124-7B-Instruct`, immutable revision in the manifest. Chat template, system prompt, no example turns.
- 20 E3 pilot roles plus `default`, 5 system prompts each, 40 frozen questions (`../261008_base_persona_map/questions_40.json`), 5 replicates (owner, 29/10; replicate ids 0–4). 21,000 replies. `{model_name}` = "OLMo".
- Sampling: temperature 0.5, repetition penalty 1.1, top_p 1, top_k 0, 128 new tokens, item seed = SHA-256 of (role, wording, question, replicate).
- Primary layer: hidden state 17 (index into `hidden_states`, where 0 is the embedding output; map labels with `io.hidden_state_index`). Layers 9 and 23 are exploratory.
- Primary set: the 20 roles on all 40 questions. Maps and decoding use the 20 roles; `default` is used only for the axis.

## Inference units

- Replies are crossed by role × question. Wordings are nested in role; replicates are nested in prompt (role, wording, question).
- Replicates are averaged inside each prompt before maps and decoding. Per-replicate estimates and the replicates 0–2 sensitivity are reported too.
- Bootstrap: resample roles and questions (crossed), keep all wordings and replicates of a sampled cell. 2,000 draws, percentile intervals. The decoding bootstrap does not refit models; this is its stated scope.
- Scope of the intervals [R3]: they keep the five wordings fixed and use the observed replicate means. They do not resample a new wording panel or new generations. Wording-held-out and replicate-held-out results are reported as separate sensitivities, not merged into one interval.
- Bootstrap p-values [R8]: for a one-sided bootstrap rule "lower bound > 0", the decision p is the share of bootstrap draws with the statistic ≤ 0 (percentile test inversion), computed on the same paired draws.
- Halves for reliability: half A = 20 questions × 2 or 3 wordings, half B = the other questions and wordings (as E3, change of 09/10). Fixed half assignments, 50 draws.
- Floors apply to the 95% lower bound, not the point estimate.

## Integrity gates (all must pass before any metric)

| Gate | Rule | Reason |
|---|---|---|
| G1 tokens | No EOS, pad or stop-marker id in any pooling span. An empty kept reply is invalid (NaN vector, flag). | Review B8. |
| G2 valid rate | ≥ 95% valid replies for every role. | Below that, cells drop unevenly across roles. |
| G3 seeds | 40 items rerun at the same batch size and seed give identical ids. Across batch sizes: agreement reported only. | Review B6; GPU kernels do not guarantee identity across batch sizes. |
| G4 reconstruction | A forward pass of the stored prompt and kept ids reproduces the stored states at hidden states 9, 17 and 23, on every valid row, with centring over valid rows only: centred cosine min ≥ 0.95, median ≥ 0.99, and median relative norm difference ≤ 0.02 (the full E3 C0 gate). Invalid rows are checked separately: invalid flag set, empty pooling span, NaN vector [round 2, O-A]. | Shows the stored states match the stored text, in direction and scale [O3]. |
| G5 snapshot | Model and tokenizer revisions, tokenizer and template hashes match the manifest. | Review B12. |
| G6 synthetic | Before real data: the map null gives uniform p on independent maps; the map statistic finds a planted map; the axis check on role-identical data with a fixed default offset is recorded (expected cosine 1.0). | Review B2, B3. |
| Raw read | 20 random transcripts read before any metric, and noted. | Project rule. |

If a gate fails: fix the code or data and rerun. No metric is reported from a failed run.

## Primary decisions

**D1. The Instruct replies carry the role in a nameable way (A2, identifiability).**

- Measure: blind naming hit@5 from the reply alone (E3 prompt, 276 names), macro over the 20 roles, on the judge set fixed at judge approval (see "Cost and approval"). Two judges: GPT-6.1-Terra and Claude Sonnet 5.5 (v1.5; E3 used GPT-5.6-Sol and Claude Sonnet 5.5). Isolated folder; keys out of reach.
- Null: wrong-role mapping permuted as whole role blocks within each question (9,999).
- Pass, for both judges: (a) macro hit@5 lower bound above the null p99; and (b) at least **15 of 20** roles above their own null p99.
- Reason for (b): pod C needs a map over most roles, with every E3 group present.
- Decision for Holm [R8]: (b) is a required gate inside D1, not a separate test. D1's p-value is the larger of the two judges' macro permutation p-values, p = (1 + #null ≥ observed) / (1 + 9,999). D1 passes only if (a) and (b) both hold for both judges.
- Reported beside it: hit@1, self-naming rate, coverage, per role.
- If D1 is not met [O4]: the run did not show that the Instruct replies carry nameable roles. Stop before pod C and read the raw replies.

**D2. The role map repeats (A1).**

- Measure: role-role cosine distance matrices (mean-centred role vectors) from half A and half B; raw Spearman r. Spearman–Brown is reported, not used for the rule.
- Null: permute the role identities of **one** half's matrix only, keeping wording blocks (9,999). A common shuffle of both halves is not a null (review B2).
- Pass: raw r lower bound ≥ 0.5 and p < 0.01.
- Reason: the E3 rule, kept so pod A and E3 compare. No stronger reason is recorded.
- Decision for Holm [R8]: the one-map permutation p, p = (1 + #null ≥ observed) / (1 + 9,999), with r averaged over the 50 fixed halves for the observed data and every permutation.
- If D2 is not met [O4]: the run did not show a repeatable map. Do not interpret the map. Any change to questions or replicates needs a new owner decision.

**D3 (gate, not a claim). A repeatable default-vs-role contrast.**

- Axis = mean of the 5 default wordings minus the mean of the 20 role vectors. Halves split questions **and** wordings, default wordings included.
- Pass: split-half axis cosine ≥ 0.80, as a point value averaged over the 50 halves (the E3 rule). Its bootstrap interval is reported separately.
- Projection, fixed now [R6]: on a training half, unit axis u = (default mean − role mean) / its norm, and origin o = the training-half role mean. Project only the other half's default and role vectors: s = (x − o) · u. Then swap the halves and average. When pod C compares another checkpoint, the Instruct training-half origin and axis stay fixed; this assumes the two checkpoints share coordinates, which is stated with every such result. It is a fixed-text coordinate comparison, not stage occupancy.
- Limit: a fixed default-prompt offset passes this rule with no role differences (review B3, offline check). So D3 only decides whether pod C may use the axis. It is reported as "a repeatable default-vs-role contrast", not as an Assistant Axis.
- Control, reported: the same axis with the `assistant` role (long, role-style prompts) in place of `default` (short prompts), and the cosine between the two.
- If D3 fails: pod C uses map comparisons (RSA, CKA) only.

**Multiple tests:** Holm over D1 and D2. D3 is a gate.

## Reported, no pass rule

- Context-conditioned role decoding at state 17 (E3 classifier and splits that hold out questions and wordings; chance 1/20). Expected near ceiling because the system prompt is in context.
- Text baselines on the same folds: word and character n-grams, a sentence-embedding model, hidden state 0.
- Spectrum of the 20 role means (rank ≤ 19 after centring). PC1 and PC2 ends against Lu et al.'s lists only if the eigengap and bootstrap subspace stability pass.
- Linear vs kNN decoding; layers 9 and 23; per-replicate estimates.
- Fidelity (E/P/G/W/U over all attempts): added when `judge_validation` has a validated rubric. Not a pass rule now.

## Cost and approval

- GPU: 21,000 replies (5 replicates); benchmark per-item seeding first.
- Judges: blind naming with two judges. The judge set is chosen at judge approval and fixed before any judge call: all 21,000 replies, or replicates 0–2 only (12,600). **Owner approval before the judge run.**
- The approved judge panel is D1's primary population [round 2, B5]. D1's precision target and stopping rule apply to it. If the panel is replicates 0–2, D1 is exempt from the five-replicate requirement; D2 and D3 still use all five replicates. The panel is frozen before any label or naming summary is seen and is not switched after results.

## Run count and noise (29/10; owner: "make sure we have enough runs to create and separate noise")

Each noise source gets its own measurement:

| Source | How it is measured |
|---|---|
| Sampling (seed) | Replicates per prompt; within-prompt variance; replicate-to-replicate map r. |
| Question and wording | Split halves over questions and wordings; crossed bootstrap. |
| Chance | Permutation nulls and the G6 synthetic cases. |
| Judge | Two judges; a blind 5% repeat audit of each judge. |

Precision targets (proposed 29/10; part of the run-count plan the owner accepted), set so a lower bound can clear its floor when the true value is clearly above it:

- D1 macro hit@5: 95% interval no wider than ±7.5 points. E3's one-seed standard error was 7.3 points (about ±14), which is too wide.
- D2 raw map r: 95% interval no wider than ±0.10.
- D3 axis cosine: 95% interval no wider than ±0.05.

Owner decision (29/10, "Accept all three"): the full 20-role run uses **5 replicates** per prompt (21,000 replies), not 3. The full run measures replicate noise on the real panel:

- per-replicate maps and the replicate-to-replicate map r;
- replicate-held-out results;
- every primary metric again on replicates 0–2 only, as a sensitivity that shows what 3 replicates give. It is reported, not used for a decision, except that D1 uses replicates 0–2 when that is the approved judge panel ("Cost and approval").

Rule: the precision targets are checked on the full 5-replicate run (D1: on the approved judge panel). If a target is not met, report that and stop interpreting that measure. There is no automatic increase of replicates or questions; any increase needs a new owner decision.

History: the owner first chose 3 replicates with a 5-replicate canary (29/10). Codex round 1 (R1–R3) showed two problems. The 4-role canary allows only 24 role-label permutations (smallest p 0.042), so it cannot measure the precision of a 20-role map; it is now an engineering canary only (`DESIGN.md`, order of work, step 5). The E3 noise check is descriptive and has no variance model, so it cannot forecast interval widths; its forecast was removed from `NOTE.md`.

## Owner decisions (29/10)

Accepted ("Ye accept"): D1(b) 15 of 20 roles; G2 valid rate 95%; the E3 judge pair (GPT-5.6-Sol, Claude Sonnet 5.5); D2 and D3 keep the E3 values 0.5 and 0.80.

Accepted ("Accept all three"): the full run uses 5 replicates, and the precision targets are checked on it.

## Changes

- 29/10/2026, v1.1, before any code or data: text fixes from Codex round 1 (`../261029_stage_persona_plan/review/codex_review_v2_r1_261029.md`): interval scope and bootstrap p (R3, R8), Holm decision p for D1 and D2 (R8), the axis projection rule (R6), the full E3 C0 gate with the norm limit (O3), wording for unmet rules (O4), and the run-count plan (R1–R3). No threshold value changed.
- 29/10/2026, v1.2, before any code or data: owner decision after round 1: 5 replicates in the full run (was 3), with per-replicate, replicate-held-out and replicates 0–2 results; the D1 judge set is chosen at judge approval. Precision targets unchanged. No threshold value changed.
- 29/10/2026, v1.3, before any code or data: text fixes from Codex round 2 (`../261029_stage_persona_plan/review/codex_review_v2_r2_261029.md`): the approved judge panel is D1's primary population (B5); reconstruction on valid rows (O-A). No threshold value changed.
- 30/10/2026, v1.4, before any judge call or naming result: owner approvals for the D1 judge run ("Approve all", 30/10, on the setup listed in `NOTE.md`, section "D1 naming runner"). (1) Judges run through OpenRouter, one reply per request; the prompt is `prompt_naming_openrouter.txt` (sha256 `d7e4c5bd…`), the frozen E3 prompt with only the batch, file and tool rules removed, the 276 names inline and the reply between markers. Judge settings as in E3: GPT-5.6-Sol (`openai/gpt-5.6-sol`) at reasoning effort high, Claude Sonnet 5.5 (`anthropic/claude-sonnet-5.5`) without extended thinking. (2) D1(b) counts roles whose hit@5 point is above their own null p99, as this file words it; the count with each role's 95% lower bound is reported beside it. (3) Holm family alpha 0.05. (4) The approved judge panel is replicates 0–2 (12,600 replies, default included). (5) A 20-reply pilot measures the cost first; the full batch goes out only if the estimate stays under $250. No threshold value changed.
- 05/11/2026, v1.5, before any D1 run answer: owner decision on the D1 judges ("1. Terra and sonnet", 05/11, via the coordinator session; checked in its transcript). GPT-6.1-Terra (`openai/gpt-6.1-terra`, reasoning effort high) replaces GPT-5.6-Sol to cut the cost from about $125 to about $75; Claude Sonnet 5.5 stays. Evidence: `naming_judge_check.py`, 200 pod A replies, with the rule fixed before any answer: Terra's hit agreement with Sol and Sonnet was 0.915 and 0.910 against a bar of 0.885 (Sol–Sonnet 0.935 minus 0.05). Disclosure: that check showed macro hit@5 on those 200 replies (Sol 0.675, Sonnet 0.710, Terra 0.690), and they stay in the panel. The panel (replicates 0–2) was fixed on 30/10, before the check, and the judge rule used agreement, not hit@5. Limit: pod A D1 now has one judge that E3 did not use, so a base-versus-Instruct naming comparison crosses judges for that judge. Claude Sonnet's `max_tokens` is 100 (check maximum 42 tokens), which only lowers OpenRouter's credit hold. No threshold value changed.
