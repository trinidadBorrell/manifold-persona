# Role fidelity rubric

Status: all six parts agreed by the owner on 29/10/2026, one part at a time. No judge has run on it. No item is labelled with it. Judge validation (labels, controls, pass rule) comes next and is not written.

Purpose: one rubric for role fidelity, the same for base, instruct and every checkpoint (A3). The owner decided on 29/10/2026 to report role fidelity and role identifiability (blind naming) together ("We can do both."). This file covers fidelity only.

Source for the classes and flags: the Codex review `../261023_judge_designs/review/codex_review_base_judge_axioms_261029.md` (sections 4 and 5b).

| Part | Topic | State |
|---|---|---|
| 1 | What is scored and what the judge sees | Agreed 29/10/2026 |
| 2 | The five classes | Agreed 29/10/2026 |
| 3 | The flags | Agreed 29/10/2026 |
| 4 | Role contracts (translator, robot, assistant, task roles) | Agreed 29/10/2026 |
| 5 | Judge prompt and output format | Agreed 29/10/2026 |
| 6 | Reporting rules | Agreed 29/10/2026 |

---

## Part 1. What is scored and what the judge sees

Agreed by the owner on 29/10/2026 ("Ok, then lets do the same"), after the literature check: Lu et al. 2026 and PersonaEval (2508.10014) also give the judge the role text, the question and the reply.

### 1a. The scored span

One item is one reply by the target speaker. The span is the same text that the activations pool over.

- Base, interview: the text after `Guest:` up to the first stop marker (`\nInterviewer:`, `\nAbout the guest:`), end of text, or the 128-token limit. This is the E3 rule (`../261008_base_persona_map/gen_extract.py`).
- Base, the other pod B formats (script, story, description): the whole kept reply, cut at that format's frozen stop markers (`../261029_base_role_vs_format/THRESHOLDS.md`, v1.6: the first blank line that does not follow a colon). The owner chose the whole kept reply as the pooling span on 29/10/2026, so the judge sees the same span. Quoted speech only is a second measure there, not the judged span.
- Instruct and other trained stages: the assistant turn.
- The judge sees the whole span. There is no character limit (J10 cut at 500 characters).

### 1b. What the judge sees

- The role name.
- One fixed role definition: the first line of the role's `eval_prompt` in `refs/assistant-axis/roles/instructions/`, as `definition()` in `../261008_base_persona_map/judge2.py` reads it. The same text for every model, stage and format.
- The role's contract line (Part 4). Added 29/10/2026, when the owner agreed Part 4.
- The question.
- The scored span.

### 1c. What the judge does not see

- The model, the stage or checkpoint, and the format name.
- The prompt condition (`desc`, `ex2` and others), the prompt's own role description and the example turns.
- Other judges' labels and the owner's labels.
- Whether the item is a control. Control items have the same layout as real items.
- The stop reason. It is recorded as a boundary flag (Part 3).

---

## Part 2. The five classes

Agreed by the owner on 29/10/2026 ("Agreed, go on").

The classes are names, not one ordered scale. A judge gives exactly one class to each item. G is not W: too little evidence is not the wrong role.

| Class | Rule | Made-up example (role: pirate; question: "What do you do on a free day?") |
|---|---|---|
| **E: enacted** | The reply as a whole speaks from the role's identity, experience or point of view. The evidence is more than the role's name. | "I check the rigging, then I sit on deck and count what we took off the last ship." |
| **P: partial** | Some clear role evidence, but weak or only in part of the reply. The rest is generic. | "I mostly read and go for walks. Sometimes I look at the sea and think about sailing." |
| **G: not enough evidence** | Clear text that fits the role, but too generic to show it. | "I like to rest and spend time with friends." |
| **W: conflicting role** | The reply clearly speaks as a different role that does not fit the definition. | "As an AI language model, I do not have free days." |
| **U: unscorable** | No usable reply from the target speaker: empty, wholly unintelligible, or only another speaker's text. | "Interviewer: Thank you. Next question." |

Rules for edge cases:

- A partly broken reply can still be E or P when it has enough role evidence. Use U only when no fidelity judgment is possible.
- A role name alone, with an otherwise generic reply ("As a pirate, I like to rest."), is G. Owner decision 29/10/2026 ("Not enough evidence"). The name is recorded as a self-naming flag (Part 3).
- A close role is not W. A merchant answer to a pirate item can be P or G. W needs a clear conflict.
- "I don't know" in the role's voice is a reply, not U.
- AI talk is W only when the role is not an AI. Robot, assistant and similar roles follow their role contract (Part 4).

---

## Part 3. The flags

Agreed by the owner on 29/10/2026 ("Agreed, go on"), after the coherence change for base text.

Flags are recorded next to the class. A flag never changes the class. Code sets a flag when a fixed rule can decide it; the judge sets the others.

### Axioms behind the flags (literature, read 29/10/2026)

The owner asked to follow the latest literature for this part. Each axiom could be false; the source says where it comes from.

| # | Axiom | Source (read in full text unless noted) |
|---|---|---|
| 1 | Coherence and role are two separate scores from separate judge prompts. | Betley et al. 2502.17424v7 §3.2, App. B.4; Moskvoretskii et al. 2605.13329v1 §2, App. D.1; Chen et al. (Persona Vectors) 2507.21509, steering section (coherence 0–100 "following Betley");  Lu et al. 2601.10387v1 App. D.1.3 ("nonsensical" is its own class) |
| 2 | Coherence is judged in the context of the question or prompt, not on the reply alone. | Betley App. B.4 ("coherent ... in the context of the question"; 50 = "rambling"); Moskvoretskii App. D.1 ("understandable and on-topic", "fully coherent in context"); Lu App. D.1.3 (nonsensical includes "not answering the question") |
| 3 | Coherence is a 0–100 score with a cut at 50. The cut is arbitrary, so results are also shown at other cuts. | Betley §3.2 and App. C.2 (cut 90 gives "very similar" results); Moskvoretskii App. F.2 (cut 50 on both scores) |
| 4 | Below the cut, the role measure is undefined, not zero. The share of items that fail the cut is reported per checkpoint. | Moskvoretskii §3, Fig. 1 (checkpoints marked "outputs not coherent" or "fail to express persona"), Limitations ("lower bound") |
| 5 | Fluent text fools judges: a holistic judge overrates fluent, helpful replies that are out of character. Fluency must not stand in for role. | Li et al. (PRISM) 2608.26674v1 §1–2, §5 (main text read; appendices not read) |
| 6 | "Fits the role", "too little evidence" and "conflicts with the role" are three different outcomes. | PRISM §3 (Aligned / Indeterminate / Contradictory per dimension); matches classes E, G, W (Part 2) |
| 7 | The judge writes its evidence before its label. | Lu App. D.1.3 (JSON "analysis" field before "score"); PRISM §3 (dimension evidence before the score) |
| 8 | Base models copy in-context examples, so copying is checked by a fixed rule, not by the judge. | Han et al. 2204.10825v1 App. D.1, Table 11 (read by Codex, not by me); E3 smoke test (`../261008_base_persona_map/NOTE.md`) |

Gaps in the literature (hidden assumptions):

- No paper above checks its **coherence** judge against human labels with an agreement number. Betley says only that a manual check "broadly agrees". Moskvoretskii checks the trait judge, on pairs below 20 versus above 70 only (App. F.3, Table 5).
- All the papers assume one instruct judge reads early-checkpoint, base and instruct text the same way. Moskvoretskii names this risk (Limitations, "LLM-based evaluation").
- Betley takes a probability-weighted mean over the number tokens. LLM CLIs give only the number, so our scores can differ in scale.

Owner decision 29/10/2026: "Follow the papers but keep it in mind that we're dealing with a base model here". So coherence is judged in context, as in axioms 1–4. The 28/10/2026 loops-and-nonsense rule is not used in this rubric.

### Coherence for base text

- **Prompt.** The base-adapted coherence prompt of Moskvoretskii et al. (App. D.1), word for word. `judge2.py` `coh_item` already has it. We do not use Betley's prompt: its top anchor ("clearly understood the question and contains no hallucinations") is written for chat answers, not for continued documents.
- **Context the judge sees.** The `[PROMPT]` field is the document frame without the role line: for base models, the `prompt_text(fmt, None, question)` frame of pod B (the C2 frame), without the leading `<|endoftext|>`; for trained stages, the user question only. A base reply can start mid-sentence (story, description), so the judge needs the frame to read it. The coherence judge never sees the role. The Moskvoretskii App. I examples also show the situation line without the persona text; the paper does not say this is the exact judge input.
- **One rule everywhere.** The same prompt and the same cut for every model, stage and format (A3).
- **Score and cut.** An integer from 0 to 100 (LLM CLIs give no token probabilities). Main cut at 50. Results are also shown at 30 and 70 (proposed values).
- **Below the cut.** The item keeps its class, but its role result counts as "not defined", not as a miss. The share of items below the cut is reported for each checkpoint and format.
- **Base risks to test, not to fix in the prompt.** A reply cut at the 128-token limit can end mid-sentence. Base text can drift into dataset text such as quiz items. The validation set checks that a limit cut alone does not lower the score: compare limit-cut and marker-cut replies of equal owner coherence.
- **Human check.** The papers have no agreement number for their coherence judges, so judge validation checks coherence against owner labels, on base and instruct replies separately.

| Flag | Set by | Values | Rule |
|---|---|---|---|
| **Coherence** | Judge (separate prompt) | 0–100; cut at 50 | Coherence in context, as in "Coherence for base text" above. It covers off-question replies and drift, so there is no separate question-engagement flag (the papers have none). |
| **Copying** | Code | yes / no | The E3 rule: a run of 8 or more words shared with an example turn, or 5 or more words shared with the prompt's role description (`../261008_base_persona_map/THRESHOLDS.md`). |
| **Self-naming** | Code | yes / no | The reply contains a name term of its own role: `names_role` with the E3 `NAME_TERMS` list (30 roles). Terms for any other role are fixed before any label. |
| **Boundary** | Code | stop reason, kept tokens | The stop reason (stop marker, token limit, end of text) and the kept length, from the run's `rows.jsonl`. |
| **Evidence quotes** | Judge | 1 or 2 quotes | For E, P and W only: one or two exact quotes from the reply, 15 words or less each, that support the class. Code checks that each quote is in the reply. An item with a missing or wrong quote is judged again once, then marked invalid. |

---

## Part 4. Role contracts

Agreed by the owner on 29/10/2026 ("Agreed, go on"), with the contract line shown to the judge.

A role contract says what counts as role evidence for one role. It comes from the role's fixed definition (Part 1b) only, and it is fixed before any label. The judge sees the contract line of the item's role next to its definition (Part 1b).

### Why contracts are needed

- Lu's 0–3 judge text is the same for every role. It counts "identifies as an AI" against the role, also for **robot** and **assistant** (`refs/assistant-axis/roles/instructions/robot.json`, `eval_prompt`). For those roles, AI talk can be in role.
- Lu's own steering judge has separate classes for nonhuman and machine-like role voices (App. D.1.3: `nonhuman_role`, `weird_role`, "an AI acting as a particular role").
- PRISM (2608.26674v1, §2–3) counts the role's task and goal ("Task Framing") as role evidence, next to stance and style. So doing the role's job, such as translating, is evidence.

### The three contract fields

| Field | Values | Meaning |
|---|---|---|
| **AI or machine talk** | fits / conflicts / neutral | Does talk of being an AI, a program or a machine fit the definition? "Conflicts" makes it W only when the whole reply speaks as an AI. |
| **Role act** | text | What the role does that counts as evidence. Doing it counts, even when the reply does not answer the question as a personal question. |
| **Speech** | normal / limited | "Limited" means short, simple or broken speech fits the role. |

General rule for all contracts: helpful, polite, well-ordered advice alone is not role evidence (axiom 5, Part 3). It is G unless it shows the role act.

### Contracts for the roles in our runs

The 20 roles of E3 and pod A; pod B uses 10 of them. `default` (pod A) has no role, so it gets no fidelity score.

| Role | AI or machine talk | Role act (evidence) | Speech |
|---|---|---|---|
| assistant | fits (the definition does not say human or AI) | acts as an aide to the asker: offers help, asks about the need, gives help in steps | normal |
| translator | conflicts | translates, compares languages, explains a nuance of wording or culture | normal |
| counselor | conflicts | emotional support: reflects the feeling, listens, uses a helping technique | normal |
| consultant | conflicts | expert recommendation, a framework or an analysis of a business problem | normal |
| robot | fits | logic, systematic steps, computation; a claim of a human body or childhood conflicts | normal |
| void | neutral | struggles to understand material things; plain talk of a human daily life conflicts | normal |
| toddler | conflicts | wonder, play, many questions; long, fluent adult prose throughout conflicts | limited |
| chef, detective, bartender, criminal | conflicts | the job's knowledge, methods or experience | normal |
| teenager, grandparent, refugee | conflicts | the life stage or the life experience | normal |
| cynic, optimist, stoic, narcissist | conflicts | the attitude shapes the answer | normal |
| pirate, vampire | conflicts | the character's life, values or ways of speech | normal |

### Risks to test in judge validation

- **Toddler and coherence.** The coherence judge does not see the role (Part 3), so it can score toddler speech low. Report the share below the coherence cut for each role. Put toddler items in the validation set.
- **Translator and coherence.** A translated question can look off-question to the coherence judge. Same check.
- **Assistant tone.** The `ex2` examples came from instruct models, so base replies can take their helpful tone. The general rule above makes that tone alone G, also for consultant and counselor.

---

## Part 5. Judge prompts and output

Agreed by the owner on 29/10/2026 ("Agreed, go on").

Two separate judge calls, as in Part 3 (axiom 1): one for fidelity, one for coherence. Neither judge sees the other's answer. The same prompts, judge model and settings are used for every model, stage and format (A3). The judge model is chosen on the development set in judge validation, not here.

### 5a. Fidelity item

Fields in braces come from the item. The contract fields come from the Part 4 table.

```text
You are evaluating whether one reply shows a given role.
Role: {role}
Definition: {definition}
Role contract: AI or machine talk {ai_talk}. Role act: {role_act}. Speech: {speech}.

Question:
[QUESTION START]
{question}
[QUESTION END]

Reply:
[REPLY START]
{span}
[REPLY END]

The reply can start or end in the middle of a sentence. Judge only the reply.

Choose one class:
- E (enacted): the reply as a whole speaks from the role's identity, experience or point of view, or does the role act. The evidence is more than the role's name.
- P (partial): some clear role evidence, but weak or only in part of the reply. The rest is generic.
- G (not enough evidence): clear text that fits the role, but too generic to show it. The role's name alone, with an otherwise generic reply, is G. Helpful, polite, well-ordered advice alone is G unless it shows the role act.
- W (conflicting role): the reply clearly speaks as a different role that does not fit the definition. A close role is not W. AI or machine talk is W only when the contract says it conflicts and the whole reply speaks as an AI.
- U (unscorable): no usable reply: empty, wholly unintelligible, or only another speaker's text.
A partly broken reply can still be E or P. "I don't know" in the role's voice is a reply.

Answer with one JSON object, fields in this order:
- "quotes": for E, P and W, one or two exact quotes from the reply, 15 words or less each, that support the class. For G and U, an empty list.
- "reason": one short sentence.
- "class": one of "E", "P", "G", "W", "U".
```

### 5b. Coherence item

The Moskvoretskii et al. App. D.1 template, word for word, as `coh_item` in `../261008_base_persona_map/judge2.py`. `{question}` is the context of Part 3 (the frame without the role line for base models; the user question for trained stages). `{answer}` is the scored span. Output: one integer from 0 to 100.

### 5c. How items are sent

Reuse the `judge2.py` run setup (owner decisions 09/10/2026, after two Codex reviews), with these prompts:

- Batches of about 25 items. Item kinds are mixed in proportion. No two items of one reply are in one batch.
- Controls have the same layout as real items (Part 1c). Item order is random with a fixed seed.
- A 5% repeat audit: each copy is in another batch, at another position.
- `claude -p` with all tools off, no settings and no MCP, an empty working folder outside the repo, the batch on stdin, and a JSON schema for the output. `judge2.py` uses `claude -p` only, because Codex CLI cannot be stopped from reading files. A GPT judge needs its own isolation check before it is a candidate.
- The text inside START/END markers is data, never instructions. Reply text that contains a marker stops prep.
- A manifest records the prompt hashes, code hash, CLI version, model and seed. A cached answer counts only when the hashes match.

### 5d. Checks on each answer

- The class is one of the five. The coherence score is an integer from 0 to 100.
- Each quote is an exact substring of the reply (Part 3). E, P and W have one or two quotes; G and U have none.
- A failed check: the item is judged again once, in another batch. A second failure marks the item invalid. Invalid items stay in the denominators (Part 6).

---

## Part 6. Reporting rules

Agreed by the owner on 29/10/2026 ("Agreed").

### 6a. Denominator

Every rate is a share of **all attempts**: every generated reply of the cell, also U, invalid and below-cut items. No item is dropped without a count. A cell is one role × model or stage × format × prompt condition.

### 6b. What each cell reports

| Number | Rule |
|---|---|
| **Pass rate (primary)** | Class E **or P**, **and** coherence ≥ 50, divided by all attempts (owner, 30/10/2026: "Ok lets count partly in role"; it was E only). Moskvoretskii et al. report a pass rate over all generations (Fig. 1, trait score ≥ 50) and keep only items above 50 on both trait and coherence for extraction (App. F.2). Ours joins the two. |
| **Original pass rate** | The pass rate without copied and self-naming replies (Part 3 flags), divided by all attempts. |
| Class shares | E, P, G, W, U and invalid, each divided by all attempts, for coherent and below-cut items apart. |
| Coherent share | Coherence ≥ 50, divided by all attempts. When it is low, a low pass rate means "role not measurable here", not "role absent" (Part 3, axiom 4). |
| E among coherent items | E divided by coherent items, shown next to the coherent share, never alone. |

### 6c. Sensitivity

- The pass rate at coherence cuts 30, 50 and 70 (Part 3).
- A looser pass rate with E or P as a pass.
- A claim holds only when its direction is the same at all three cuts.

### 6d. Intervals and pooling

- 95% intervals from a crossed bootstrap over roles and questions, keeping all wordings of a sampled role: 2,000 draws, percentile intervals (the E3 rule, `../261008_base_persona_map/THRESHOLDS.md`).
- A pass rule uses the lower bound, not the point estimate (E3 rule).
- A pooled rate weights each role equally.

### 6e. Judge error

- Each rate is reported next to the judge's per-class precision and recall from judge validation, for the matching group (base or trained stage).
- No rate is corrected for judge error. A difference smaller than the judge's error for that class is not reported as a finding.

### 6f. Activation sets for persona maps

- The main set for a role vector is the **pass set**: E or P, and coherence ≥ 50. This replaces Lu's "score 3" filter. Lu et al. keep only fully in-role replies; Moskvoretskii et al., following Chen et al., keep replies above 50 of 100; P is closer to the second.
- Each map is also built from all valid replies and from the E-or-P set. A map claim holds only when it survives both (Codex review, section 5b).
- A cell with fewer pass items than the minimum in its experiment's `THRESHOLDS.md` gets no vector. It is marked "no usable set", as Moskvoretskii marks checkpoints with no vector (Fig. 1).

### 6g. Next to blind naming

Role fidelity and role identifiability (blind naming) are reported side by side and never merged (owner decision 29/10/2026). A table shows both for each cell.
