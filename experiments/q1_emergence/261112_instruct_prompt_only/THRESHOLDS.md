# Instruct prompt-only control: rules (v1.0, 12/11/2026)

Written before any code ran. Owner, 12 Nov: "Yes, build both" (this control and the coin-flip test). It closes the limit in the draft: "this may be a map of the prompts; I have no prompt-only control on Instruct yet."

## Question

In OLMo-2-7B-Instruct with the role prompt in view (pod A), does one fixed neutral reply after the role prompt give the same role map as the roles' own replies?

## Design (pod B's C3, owner-approved 29/10/2026)

- Prompts: pod A's 4,200 stored prompts (20 roles plus `default`, 5 wordings, 40 questions) from `../261029_instruct_persona_map/run_full/rows.jsonl`, checked against that run's `SHA256SUMS.txt`. Every replicate of a prompt must have the same prompt ids, and every prompt must end with `<|assistant|>\n`.
- Continuation: pod B's approved neutral speech (E3 C3), "That is a good question. I think the answer depends on the situation, and I would want to think about it carefully before I say more." It has no leading space, because pod A's replies start right after `<|assistant|>\n`. No end token.
- State: the mean over the continuation tokens. The code uses pod A's pooling function (`manifold_persona.rows.pool_spans`) with the same model, revision (`470b1fba…`), bfloat16, sdpa and transformers 4.47.1. It saves hidden states 8, 9, 16, 17, 23 and 24 as float16.
- Environment check before the run: the code re-pools 8 of pod A's real replies, then compares the result with pod A's stored states at state 17. Pass: every cosine is at least 0.99. If the check fails, the run stops.

## Measures (state 17, pod A's primary; states 9 and 23 also reported)

- M1, the prompt-only map: pod A's D2 code (`map_statistics`) on the C3 states. It uses 50 fixed question-and-wording halves, 2,000 role bootstraps and 9,999 one-half permutations. Pod A's reply map gives r = 0.901 [0.797, 0.939].
- M2, cross-map agreement: pod A's `map_panel`, with the same halves and draws. It correlates the reply map from one half with the C3 map from the other half, both ways, then averages. The code also takes the paired difference D2 - M2 for each draw.
- M3, role decoding: pod A's linear decoder (`decode_features`, the same splits) on the C3 states, beside the same decoder on the reply states.

## Reading rules (fixed now)

- "The prompt alone gives a role map": M1's lower bound is at least 0.5 and p < 0.01 (pod A's D2 criterion).
- "The replies add role structure beyond the prompt": the 95% interval of the paired D2 - M2 is above 0.
- If the D2 - M2 interval includes 0: "no evidence that the replies add structure beyond the prompt". This is not proof that the maps are the same.
- All results are report-only, like pod B's C2 and C3. They set the wording of one limit in the post.

## Compute and approval

- 4,200 forward passes of about 75 tokens, plus 8 check passes: a few minutes on one GPU. Output: about 0.2 GB of states. It runs in the same Modal session as the coin-flip test (owner go needed for the paid run).
- The analysis runs on CPU and needs no GPU.
