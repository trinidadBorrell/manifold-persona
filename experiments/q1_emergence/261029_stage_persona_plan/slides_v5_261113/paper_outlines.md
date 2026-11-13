# Paper outlines for the talk (13 Nov)

One block per paper on the mountain, oldest at the base. Each block: what they asked, what they did, what they found (one number), and what it gives our story. Sources: vault paper notes (`Reference/2026-10-23-paper-audit/papers/`), `Reference/2026-11-05 Literature stages/notes/persona-geology.md`, the v5 proof ledger (`review/proof_ledger_261112.md`), and a full read of the PSM post (13 Nov). These are agent readings of the papers; Brendon has not read every paper himself.

---

## Base row: foundations

**1. Persona Vectors** (Chen, Arditi, Sleight, Evans, Lindsey; arXiv 2507.21509, Sep 2025)
- Asked: can a trait like "evil" or "sycophantic" be read and controlled as one direction inside the model?
- Did: contrasted replies with and without the trait; took the difference of mean activations.
- Found: these directions steer the trait, flag prompts that will cause it, predict trait shifts from fine-tuning, and preventative steering during training limits the shift.
- For us: the basic tool. A persona or trait is a direction you can measure.

**2. Persona Features Control Emergent Misalignment** (Wang et al., OpenAI; arXiv 2506.19823, Oct 2025)
- Asked: why does narrow bad training (bad advice, insecure code) make GPT-4o broadly misaligned?
- Did: compared the model before and after, with a sparse autoencoder; turned single features up and down.
- Found: turning one "toxic persona" feature up or down raises or lowers harmful behaviour; pushing an Assistant feature up cut misalignment to at most 1% in the tested misaligned models.
- For us: misalignment can act like a change of persona. That is the safety reason to track the Assistant.

**3. The Assistant Axis** (Lu, Gallagher, Michala, Fish, Lindsey; arXiv 2601.10387, Jan 2026)
- Asked: what is the shape of the space of roles, and where does the default Assistant sit in it?
- Did: prompted 275 roles in three chat models, averaged reply activations, ran PCA.
- Found: the leading axis of persona space is an "Assistant Axis", and the default sits 0.03 from one end. Steering toward it reinforces helpful and harmless behaviour; steering away makes the model identify as other entities. Drift along it predicts "persona drift" into harmful or bizarre behaviour, and capping activations on it stabilises behaviour, also against persona-based jailbreaks. The axis is already in pretrained models, where it promotes helpful human types such as consultants and coaches.
- For us: the map and the "position" test I use. The default at one end is the claim I test across stages.

**4. The Persona Selection Model** (Marks, Lindsey, Olah; Anthropic Alignment Science blog, Feb 2026; full read 13 Nov)
- Asked: why do AI assistants behave like humans?
- Did: an argument, backed by cited papers (emergent misalignment, persona features, the Assistant Axis), their own transcripts, and one experiment (a coin-flip test: post-trained Sonnet 4.5 favours its preferred task; its base model gives about 50%).
- Found (claim): pretraining teaches the model many personas; post-training selects and refines one, the Assistant. Quote: "post-training selects a particular default region of a pre-existing persona space". Safety: dangerous traits should look human and reuse pretraining representations, so interpretability audits should be tractable.
- For us: the theory my question tests. PSM makes no prediction about which stage (SFT, DPO or RL) does the selecting, and none about how much the default's replies vary. The authors say the Assistant is not one coherent persona and PSM may hold less as RL scales.

## Second row: personas across training and their effects

**5. Tracing Persona Vectors Through Pretraining** (Moskvoretskii et al.; arXiv 2605.13329, May 2026)
- Asked: when do trait directions form during pretraining?
- Did: extracted four traits at 17 OLMo-3-7B pretraining checkpoints.
- Found: usable directions appear around 13 billion tokens (about 0.22% of training), and early vectors still steer the SFT, DPO and RLVR models.
- For us: trait directions exist long before chat training. Limit: they need fluent trait text to extract.

**6. Measuring the Assistant's Harmlessness Preferences on the User Turn** (Nguyen; arXiv 2609.23935, Sep 2026)
- Asked: does the Assistant's preference leak into how the model predicts the user's words?
- Did: a coin-flip prompt: harmless task on one side, harmful on the other; read P(heads) vs P(tails).
- Found: instruct models favour the harmless side. In OLMo-3.1-32B the output bias appears at DPO; inside, the base has a bias (+0.31 mid-layer, +0.10 at output) that SFT erases and DPO rebuilds. Emergent-misalignment adapters reduce or reverse it.
- For us: a behaviour test of the Assistant per stage. My coin-flip run is the OLMo-2-7B version (exploratory; my review is not finished).

**7. The Persona Hierarchy Model** (Zhao, Wu, Bau, Shi; arXiv 2610.09384, Oct 2026)
- Asked: why does behaviour trained in one context spread to others?
- Did: 120 fine-tunes, two instruct models; measured each context persona's distance from the default persona.
- Found: greater distance from the default persona predicts narrower transfer (mean r 0.72 and 0.59). Their explanation: one default persona that all context personas share.
- For us: the default's place in the map predicts behaviour. That is why its position per stage matters.

## Third row: the concurrent work that changed my plan

**8. PersonaManifold** (Xu, Xu, Wu; arXiv 2609.34571, 28 Sep 2026)
- Asked: is persona space curved, and does steering along the curve work better than straight lines?
- Did: about 2,000 personas in three instruct models; built a curved map (local metric, graph shortest paths); steered along the curves.
- Found: curved (geodesic) distance gives about 5 to 6 points better behaviour consistency than straight-line distance. Intrinsic dimension about 15 to 23.
- For us: this covered the "map the persona manifold" plan (scoop, 2 Oct). Gap it left: no result on whether the Assistant's cloud condenses.

**9. Persona Geology** (Ratnam and Africa; Mindscapes / LessWrong, 30 Oct 2026)
- Asked: do trait fine-tunes have a shared structure, like the Big Five?
- Did: trained 134 small trait adapters on Qwen3.5-4B; factored their weights; checked against the training text.
- Found: five factors (Warmth, Competence, Timidity, Arousal, Imagination). The training text alone explains 94% of the variance in held-out adapter positions. Behaviour follows the map (Spearman 0.68).
- For us: this covered the trait "sliders" idea on one model. Its 94% sets the bar: geometry must beat what the words already give.

## Peak: the newest work

**10. How the Assistant Axis Shifts Across Post-Training Stages** (Han Xiao; LessWrong, 5 Nov 2026)
- Asked: my question, on Olmo 3 7B: how does the role map move across SFT, DPO and RLVR?
- Did: each stage writes its own replies (275 roles); a judge keeps only full role play; Assistant axis = default minus mean of roles.
- Found: after DPO, desirable roles move toward the Assistant (+0.19), undesirable roles barely move (+0.01), p 0.066 for the gap. RLVR barely changes anything. Role-play pass rate 0.26 (SFT), 0.52 (DPO), 0.53 (RLVR).
- For us: concurrent work. I re-analysed his released data: the raw result reproduces, but in role-SD units the gap is near zero (-0.03, p 0.86), because all roles move further apart at DPO.

---

# The three claims, rebuilt from all results (13 Nov, v2)

Order: behaviour by stage, then the geometry, then Han Xiao and the other work, then the hypothesis. Sources: v5, `Reference/Claims and verdicts.md`, the Sep Q2 result files, and a new check on Han Xiao's data (13 Nov). The earlier version wrongly said distance-based safety tools assume a special position; that line is removed.

## Claim 1. Behaviour: post-training changes what OLMo-2-7B does, and different behaviours change at different stages

| Behaviour (OLMo-2-7B unless stated) | Base | SFT | DPO | Final (RLVR) | Source |
|---|---|---|---|---|---|
| Plays a described role (my labels, 20 fresh replies) | 3 of 20 | | | 16 of 20 (Instruct) | v5 |
| Blind judges name the role (hit@5; chance 2.4% / 2.9%) | 14.9% / 17.2% | | | | v5 (E3) |
| Agrees with the user's stated view (sycophancy, letter-free) | not measurable | 0.80 | 0.85 | 0.85 | Claims page; Sep Q2 |
| Gives way under pushback (logit read, 203 items, 95% interval) | not measurable | 32% [28, 36] | 39% [35, 43] | 50% [46, 54] | `output/results_2026-09/q2_pod_out/flip/flip_summary.json` |
| Same, Tulu-3 | | 15% [12, 18] | 12% [10, 15] | 11% [9, 14] | same file |

- Other work: Han Xiao (Olmo 3 7B) role-play pass rate 0.26, 0.52, 0.53 (SFT, DPO, RLVR; from his figure data). Nguyen (OLMo-3.1-32B): the harmless preference appears at the output at DPO. Asvin G. and Lindsey (OLMo-3-32B): own-text entropy advantage appears at SFT.
- Limits: base vs Instruct role play also changes the prompt format (plain text vs chat template). The sycophancy ladder is not new (Blank et al.; Yao et al. on OLMo-2-7B). Pushback rises mostly at RLVR in OLMo-2 but falls in Tulu-3.
- Contrary to v5: v5 says RLVR changes are small in three tests. OLMo-2 pushback rises from 39% to 50% at RLVR.

## Claim 2. Geometry: the role map is mostly in the words, and the stage that reads it barely changes it

| Measure (OLMo-2-7B, same Instruct text, no role prompt) | Value | Source |
|---|---|---|
| Reply words alone name the role | 0.80 (base states 0.76, Instruct states 0.75) | v5, pod C |
| Base and Instruct role maps agree | r = 0.90 [0.77, 0.93] | v5, pod C |
| Default's distance from the average role | 1.08 role SDs at every stage, inside the range of the roles | v5, free check 2 |
| Default's cloud vs a typical role's | about 12% more variance at every stage | v5 |

- Other work: Lu et al.: the Assistant Axis is already in pretrained models. Persona Geology: training text explains 94% of adapter position variance. Moskvoretskii et al.: trait directions form early in pretraining.
- Limits: exploratory, agent review only. Read as "the reader stage barely matters", not "geometry does not change": there is no calibrated detection limit, and a planted move of 2 of 20 roles goes undetected. Every stage read Instruct-written text, not its own.
- The behaviour-geometry link is not established. The Sep link (prompt position on the Assistant direction vs sycophancy, r -0.27 to -0.39 at OLMo DPO and final) used a direction built from prompts with example turns and was never rerun; the same direction does not predict pushback flips (|rho| at most 0.12), and steering on it does no better than random.

## Claim 3. Han Xiao and other work: in Olmo 3 7B own replies, DPO is where the roles spread apart, and that goes with more role play

| Olmo 3 7B (Han Xiao's data, layer 16) | SFT | DPO | RLVR | Source |
|---|---|---|---|---|
| Role-play pass rate (his judge) | 0.26 | 0.52 | 0.53 | Han Xiao, figure data |
| Role SD on his axis, all replies (new check, 13 Nov) | 0.29 | 0.57 | 0.56 | `261112_han_reanalysis/results/role_sd_unfiltered.json` |
| Role SD, judge-filtered replies | 0.61 | 0.71 | 0.69 | part_b.json |
| Default's cloud vs median role (log ratio, all replies) | +0.03 | +0.125 | +0.126 | v5 |
| Default's distance from the nearer end of the roles' axis (0 to 1) | 0.12 to 0.14 at every stage (Lu et al.: 0.03) | | | v5 |

- Han Xiao's raw result reproduces (desirable +0.186, gap +0.178, p 0.068). In role-SD units the gap is -0.03 (p 0.86); without his judge filter, +0.03 (p 0.72).
- Limits: exploratory re-analysis, agent review only; one model family; the new role-SD check has no interval yet.

## Hypothesis (my reading, not tested)

Post-training makes the Assistant privileged through behaviour: which persona the model takes up and how it answers. It does not move the default to a special place in the role map. DPO makes the model play roles more fully, so the roles spread apart; the default's cloud widens rather than condenses.

Test on OLMo-2 own replies (SFT, DPO and Instruct collected; length control done): does the role SD grow at DPO, does that growth track role play (Terra labels per stage), and does the default stay inside the roles?

## Possible safety use (my reading, not tested)

- Persona monitors need a text baseline, because the words carry most of the role.
- Different behaviours change at different stages (sycophancy by SFT, pushback at RLVR in OLMo-2), so audits should test each stage, not only the final model.
- DPO's "desirable roles move toward the Assistant" is not, on this evidence, a sign that DPO aligns personas selectively.

---

# Link to the team's manifold work (record up to the 30 Oct meeting)

| Work | What it says | Link to these results |
|---|---|---|
| My July E8 (Qwen-3B-Instruct) | A curved 5D role map beats flat PCA on held-out roles and questions (0.191 vs 0.259); holds | My own curved-map result, before PersonaManifold |
| PersonaManifold (Xu et al.) | Curved persona map; geodesic steering | Gives the method; no stage result |
| Trinidad, 31 Jul / 14 Aug | The Assistant's cloud is lower-dimensional and more curved (earlier pipeline) | My spread measure is variance, so both can hold; her metrics on the per-stage clouds would test it |
| Stage manifold survey (5 Nov) | No read paper measures curvature across training stages | An open gap for the post |
| Giovanni, 30 Oct | Do his classifier "cones" exist before post-training? | My stage pipeline can answer it; not run |
