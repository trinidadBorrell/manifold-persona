# Free checks on saved states (12/11/2026)

This is the note for `261112_free_checks`. The first part is the short version. The second part ("Run record") holds every step, file and number, for checking.

## Start here

| | |
|---|---|
| Question | How many roles do the default tests need? And where does the default sit when every stage reads the same Instruct replies with the role prompt hidden? The default is the model with no role prompt. |
| Result | With 20 roles I can see a change in the default's position of about 0.9 role SD between two stages. With 40 roles it is about 0.6. On the same Instruct text, the default sits near the assistant-like end of the role range at every stage (0.043 to 0.049 on Lu's 0-1 scale). Its replies vary more than a typical role's. The stages barely differ. |
| How much I trust it | Check 1 is a design check, not a result. Checks 2 and 3 are exploratory and report only, because I had seen the pod C data in other analyses. The planted-shift check passes. A second build of check 2 gives the same numbers. Check 3's reading is an agent's and is not reviewed. No human has reviewed any of this. |
| What I do not know | What each stage does on its own replies. Here every stage reads text that Instruct wrote. I also do not know whether the picture holds with more than 20 roles. |
| Status (13/11/2026) | The three checks are done. They cost $0: no GPU, no judge. |

---

## What I did

1. I chose free checks first on 12 Nov: "Yes, free checks first". They use saved states only.
2. Check 1 measured how precise the default's position and spread are with 10, 15 and 20 roles. It ran on pod A's states, then on pod C's four readers.
3. I fixed the rules of check 2 before any check 2 number.
4. Check 2 measured the default's position and spread at base, SFT, DPO and Instruct. All four read the same Instruct replies with no role prompt.
5. Check 3 (13 Nov) put the default on Lu et al.'s 0-1 scale, for the deck figures.

---

## Results

All numbers use hidden state 17 and 20 roles.

### With 20 roles I can see a position change of about 0.9 role SD

This is check 1 on pod A's states. The detectable change is the smallest change between two stages that I would see 80% of the time.

| Roles | Detectable position change (role SD) | Detectable spread change (log) |
|---|---|---|
| 10 | 1.12 | 0.23 |
| 15 | 0.96 | 0.20 |
| 20 | 0.86 | 0.17 |
| 40 (fit) | about 0.61 | 0.12 |

- This is a design check. It decides the role count for the stage reruns, and I pick the count from the curves.
- The pod C readers give the same picture: about 0.85 role SD at 20 roles and about 0.6 at 40.
- A spread change of about 17% (log 0.17) is visible at 20 roles.
- Planted shifts of 0.5, 1 and 2 role SD move the position by about those amounts, at pod A and at every pod C reader.

### With the role prompt hidden, the default is about 1 role SD from the role mean at every stage

This is check 2. Every reader reads the same Instruct replies. The spread compares the default's variation with the median role's variation. Above 0 means the default varies more.

| Source | Position | Spread, all rows |
|---|---|---|
| Base | 1.08 | 0.120 |
| SFT | 1.08 | 0.122 |
| DPO | 1.08 | 0.111 |
| Instruct | 1.08 | 0.111 |
| Reply words alone | 1.00 | 0.583 |

- No reader puts the default outside the role spread. My rule needed a position above 2 with its lower bound above 2.
- No reader shows a tighter default. At every reader the default's cloud is wider than the median role's cloud, by about 12% more variance.
- The reply words alone show a much larger spread (0.58). The model states narrow that gap and do not reverse it.
- The stage differences are tiny on this fixed text. I report them only.

### On Lu's 0-1 scale the default is near the assistant-like end

This is check 3. The number is the default's distance to the nearer end of the role range. Lu et al. report 0.03.

| Source | Distance to the nearer end [95%] |
|---|---|
| Base | 0.049 [0.030, 0.073] |
| SFT | 0.047 [0.030, 0.071] |
| DPO | 0.043 [0.025, 0.066] |
| Instruct | 0.044 [0.026, 0.067] |
| Reply words alone | 0.085 [0.055, 0.105] |

- The default is never outside the role range.
- The reply words alone put the default nearly at the same place.
- The roles beyond the default are consultant, assistant and (at base, DPO and Instruct) robot.
- This reading is an agent's and is not reviewed. It replaces the v5 wording "no stage puts the default at one end".

---

## How much I trust this

- I fixed the check 2 rules on 12 Nov, before any check 2 number.
- Checks 2 and 3 are exploratory. I had seen the pod C data in other analyses (maps, decoding). Pod C's own rule allowed an axis only if pod A D3 passed, and it did not.
- The planted-shift check passes at pod A and at every pod C reader.
- A second session (my fork) built the check 2 rules separately and got the same point spreads. Two builds of the same rules agree. This is not an independent human review.
- Every reader reads Instruct-written text, not its own replies.
- The default rows were written under identity prompts (one with no system message), so their text can differ for that reason.
- The intervals of check 3 cover question sampling only, not the choice of 20 roles.
- The console log of check 2 is not usable, because the fork wrote to the same log.

---

## Open points

- The fork's files stay as they are until I decide: `check_hidden_prompt.py`, `results/check2.json` and `results/check1_podc_*.json`.
- This note does not record which role count I picked for the stage reruns.

---

## Words used here

| Word | Meaning |
|---|---|
| Default | Replies with no role prompt. The default rows were written under identity prompts, one of them with no system message. |
| Stage | One OLMo-2-7B checkpoint: base, SFT, DPO or Instruct. |
| Reader | A stage that reads a fixed text. In pod C every reader reads the replies that Instruct wrote. |
| Pod A | My run `261029_instruct_persona_map`: OLMo-2-7B Instruct replies for 20 roles plus the default. Its states have the role prompt in context. |
| Pod C | My run `261029_fixed_text_read_through`: each checkpoint reads the same fixed text. |
| Frame F1 | Pod C's main input: the start token, the E3 interview header, the question, `Guest:` and the Instruct reply. It has no role prompt. |
| State 17 | Hidden state 17, pod A's primary state. |
| Role SD | The standard deviation of the roles' projections on the main axis. |
| Position, \|s\| | The default's projection on the main axis (PC1 of the role means) minus the mean role projection, in role SD. I report the absolute value, because the sign of a PC is arbitrary. |
| Spread, log-ratio | The log of the default's within-wording variance over the median role's. Above 0 means the default varies more. |
| Limit-length rows | Only the rows whose reply was cut at the 128-token limit. This is the length control. |
| Planted shift | A sanity check. I add a known shift to every default reply and check that the position moves by about that amount. |
| Detectable change | 2.8 x sqrt(2) x the bootstrap SE: the change between two stages that I would see with 80% power. |
| Lu's 0-1 scale | The default's place between the lowest and the highest role on the main axis (Lu et al.). I give the distance to the nearer end. |
| Text baseline | The same statistics on the reply words alone (character 3-5-gram TF-IDF, 256 dimensions), with no model states. |
| D3 | Pod A's gate for a repeatable contrast between the default and the roles. It did not pass. |
| Fork | A second session of mine that ran a duplicate script in this folder. |
| v5 | My draft post for Thomas, version 5. |
| HF | The private Hugging Face dataset that keeps my run files. |

---

# Run record

Everything below is the detailed record. I only need it to check a step, a file or a number.

## 1. Mode and my decisions (12 Nov)

Mode: de-risk. Reason: saved states can answer the design question (how many roles the default tests need) at $0 before any GPU rental. No GPU, no judge.

| Topic | My words |
|---|---|
| Order of work | "Yes, free checks first" |
| Check 1 | "Yes, build it" |
| HF | "Use this Mac's HF login" |
| Spread | "Saved averages" |

## 2. Check 1 rules: role count (design check, not a result)

- Data: pod A run `261029_instruct_persona_map/run_full/` (OLMo-2-1124-7B-Instruct, 20 roles + `default`, 5 wordings, 40 questions, 5 replicates = 21,000 rows), `reply_mean_acts.npy` (mean over reply tokens, role prompt in context), hidden state 17 (pod A's primary state). After the pod C download, the same check runs on pod C frame F1 (question and reply, no role prompt) at each reader.
- Position statistic (Lu et al. §2.3.1, roles-only axis): PC1 of the centred role means on one question half; on the other half, the default's projection minus the mean role projection, divided by the SD of the role projections. Reported as the absolute value (the PC sign is arbitrary).
- Spread statistic: within-wording variance of each cloud (mean squared distance of a reply to the mean of its cloud and wording), log of default over the median role. Two versions: all rows; rows cut at the 128-token limit only (length control).
- Precision: 1,000 draws. Each draw splits the 40 questions into random halves, resamples questions with replacement within each half, and resamples roles with replacement. Run at 10, 15 and 20 roles (a random role subset per draw). Output: the 95% interval width of each statistic against the role count, a 1/sqrt(n) fit, and the extrapolated width at 40 roles.
- Planted shift (sanity): add a known shift of 0.5, 1 and 2 role SDs along the full-data PC1 to every default reply and confirm the position statistic moves by about that amount. Detectable change between two stages: 2.8 x sqrt(2) x the bootstrap SE (80% power, two-sided 0.05, equal SE).
- What it decides: the role count for the stage reruns. I pick it from the curves.

## 3. Check 2 rules: default position and spread with the role prompt hidden (exploratory, report only)

Rules fixed on 12 Nov before any check 2 number was computed. The pod C data were seen in other analyses (maps, decoding), and pod C's own rule built an axis only if pod A D3 passed (it did not). So check 2 is exploratory.

- Data: pod C frame F1 states (`<|endoftext|>`, the E3 interview header, the question, `Guest:`, the Instruct reply; no role prompt) at base, SFT, DPO and Instruct, hidden state 17, all 21,000 pod A rows. HF revisions: base `d48b29c5`, SFT `caee60c0`, DPO `f3442ed9`, Instruct `00e4cf3f` (`data/fetch_receipt.json`).
- Statistics: the check 1 position and spread statistics, at each reader, with 1,000 crossed draws; 95% percentile intervals.
- Text baseline: the same two statistics on text features of the same replies (character 3-5-gram TF-IDF, reduced to 256 dimensions with truncated SVD fitted on all rows). A reader adds evidence only where its interval clears the text interval.
- Reading rules: position |s| above 2 with its lower bound above 2 = "the default sits outside the role spread at this reader". Spread log-ratio interval below 0 = "the default cloud is tighter than the median role cloud". Stage change: paired difference between readers (same draws), reported only; no stage pass.
- Limits stated with every number: Instruct-written text for every reader (not each stage's own replies); the default rows had identity prompts at writing time, so their text can differ for that reason; plain reply means; 20 roles.

## 4. Check 1 result, pod A states (12/11/2026)

Run: `check_role_count.py --source poda` (82 s on the Mac); output `results/check1_poda.json`, log `results_check1_poda.log`. 21,000 valid rows; 813 of 1,000 default rows hit the 128-token limit.

| Roles | Position \|s\|, median [95%] | Width | Spread log-ratio (all rows) | Width | Detectable change between two stages: position / spread |
|---|---|---|---|---|---|
| 10 | 0.81 [0.13, 1.28] | 1.15 | 0.177 [0.079, 0.300] | 0.22 | 1.12 / 0.23 |
| 15 | 0.89 [0.27, 1.34] | 1.07 | 0.178 [0.093, 0.286] | 0.19 | 0.96 / 0.20 |
| 20 | 0.97 [0.52, 1.38] | 0.85 | 0.177 [0.102, 0.265] | 0.16 | 0.86 / 0.17 |
| 40 (1/sqrt(n) fit) | | 0.61 | | 0.11 | about 0.61 / 0.12 |

- Planted shift sanity passes: shifts of 0.5, 1 and 2 role SDs move the signed position from -1.07 to -0.57, -0.07 and +0.94.
- Design reading: at 20 roles a stage change in the default's position must be about 0.9 role SDs to be seen; at 40 roles about 0.6. A spread change of about 17% (log 0.17) is visible at 20 roles.
- Descriptive only (role prompt in context, Instruct, data seen before): the default sits about 1 role SD from the role mean, inside the role spread, and its cloud is about 20% wider than the median role cloud (log-ratio 0.18; the same on limit-length rows, 0.19). Not a claim; check 2 reads the same text without the role prompt.

## 5. Check 2 result (12/11/2026, exploratory, report only)

Run: `check_default_readers.py` (301 s on the Mac); output `results/check2_readers.json` (seed 20261113, 1,000 shared draws, 20 roles, state 17). Inputs: `data/fetch_receipt.json`, all 8 files match their HF hashes.

The console log `results_check2.log` is not usable. A second session (my fork) ran a duplicate script, `check_hidden_prompt.py`, in this folder and wrote to the same log. Its outputs (`results/check2.json`, `results/check1_podc_*.json`) are that session's and are not checked here.

| Source (reads the same Instruct replies, no role prompt) | Position \|s\|, point; draws median [95%] | Spread log-ratio, all rows [95%] | Limit-length rows [95%] |
|---|---|---|---|
| Base | 1.08; 1.04 [0.69, 1.47] | 0.120 [0.033, 0.186] | 0.122 [0.057, 0.222] |
| SFT | 1.08; 1.03 [0.66, 1.42] | 0.122 [0.051, 0.223] | 0.167 [0.072, 0.253] |
| DPO | 1.08; 1.03 [0.66, 1.43] | 0.111 [0.043, 0.207] | 0.156 [0.064, 0.237] |
| Instruct | 1.08; 1.03 [0.66, 1.43] | 0.111 [0.042, 0.205] | 0.155 [0.063, 0.235] |
| Text baseline (TF-IDF char 3-5, 256-d SVD) | 1.00; 0.85 [0.27, 1.22] | 0.583 [0.284, 0.669] | 0.474 [0.253, 0.621] |

Paired differences (same draws):

- Reader minus text, position: +0.15 to +0.17 (intervals include 0, e.g. base [-0.03, 0.86]).
- Reader minus text, spread: -0.42 to -0.44 (e.g. base [-0.57, -0.19]).
- Stage to stage, position: within 0.02 (all intervals include 0).
- Stage to stage, spread: SFT-base +0.02 [-0.001, 0.046], DPO-SFT -0.011 [-0.023, -0.004], Instruct-DPO -0.001 [-0.002, -0.000].

Reading under the fixed rules:

- No reader puts the default outside the role spread (rule: |s| above 2 with lower bound above 2). The default sits about 1 role SD from the role mean at every reader, as it does in the text alone.
- No reader shows a tighter default cloud (rule: interval below 0). At every reader the default cloud is wider than the median role cloud (about 12% more variance). In the text alone the default's replies are much more varied than role replies (log-ratio 0.58); the model states narrow that gap but do not reverse it.
- Stage differences are tiny on this fixed text. They are reported only; pod C's equal-map control showed that small reader differences can come from wording sensitivity.
- Limits: every reader reads Instruct-written text, not its own replies; default rows were written under identity prompts (one with no system message), so their text can differ for that reason; plain reply means; 20 roles (check 1: a position change below about 0.9 role SDs is not detectable).
- Cross-check (12 Nov): the fork session's separate build of the same rules (`check_hidden_prompt.py`, draw seed 20261112, output `results/check2.json`) gives the same point spread log-ratios (0.120, 0.122, 0.111, 0.111 for base, SFT, DPO, Instruct), text 0.594 against 0.583, and interval ends within about 0.05. The agent that wrote this record (Claude) read the fork's `results/check2.json` itself. Two builds of the same rules agree; this is not an independent human review. The fork stopped its runs here; its files stay as they are until I decide.

## 6. Check 1 result, pod C readers (12/11/2026)

Run: `check_role_count.py --source podc:<reader> --out-dir results/check1_readers` for base, SFT, DPO and Instruct (frame F1, no role prompt; outputs in `results/check1_readers/`, separate from the fork's files).

| Reader | Position width at 20 roles | At 40 (fit) | Detectable position change at 10 / 15 / 20 roles | Detectable spread change at 20 roles |
|---|---|---|---|---|
| Base | 0.79 | 0.61 | 1.22 / 0.99 / 0.83 | 0.148 |
| SFT | 0.80 | 0.59 | 1.18 / 0.96 / 0.85 | 0.163 |
| DPO | 0.80 | 0.60 | 1.18 / 0.96 / 0.83 | 0.153 |
| Instruct | 0.80 | 0.60 | 1.18 / 0.96 / 0.83 | 0.152 |

Planted-shift sanity passes at every reader (signed position -1.08 moves to -0.58, -0.07, +0.94 for shifts of 0.5, 1, 2 role SDs). Same picture as pod A: at 20 roles a stage change in the default's position must be about 0.85 role SDs to be seen; at 40 roles about 0.6.

## 7. Check 3: the default on Lu et al.'s 0-1 scale (13/11/2026, exploratory, report only)

- Why: a code-and-results audit (13 Nov) found that check 2's role-SD units hide where the default sits in the role range; v5 compared OLMo-2 in role SDs with Lu et al. and Han Xiao on the 0-1 scale. My go on 13 Nov: "Go ahead and do these" (figures for the deck).
- Code: `check3_lu_scale.py` (PC1 of the 20 centred role means, all questions; default projected; distance to the nearer end of the role range; 1,000 question resamples, seed 20261113; text = check 2's text features). Output: `results/check3_lu_scale.json` (also PC1/PC2 coordinates per reader for figures). 193 s on the Mac.
- Result, OLMo-2-7B pod C readers, state 17, Instruct-written replies, 20 roles: base 0.049 [0.030, 0.073], SFT 0.047 [0.030, 0.071], DPO 0.043 [0.025, 0.066], Instruct 0.044 [0.026, 0.067]; reply words alone 0.085 [0.055, 0.105]. Never outside the role range. Roles beyond the default: consultant, assistant and (base, DPO, Instruct) robot. Lu et al. report 0.03.
- Reading (agent, not reviewed): the default sits near the Assistant-like end at every reader, and the words alone put it nearly there. Intervals cover question sampling only, not the choice of 20 roles. This replaces the v5 wording "no stage puts the default at one end".
