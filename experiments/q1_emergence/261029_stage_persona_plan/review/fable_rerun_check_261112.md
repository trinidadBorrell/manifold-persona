# Fable 5.2 check of the merged rerun plan (12/11/2026)

Reviewer: Fable 5.2 (`claude-fable-5-2`), Agent subagent of session ad361f36, requested by the owner ("Do a quick check of this with Fable 5.2"). Agent review, not a human review. The plan it checked: G1-G2 (SFT and DPO write pod A chat replies), G3 (base default in pod B plain frames, role line removed, plus Lu prefills), G4-G6 (SFT, DPO, Instruct write pod B plain frames), R1-R4 (base, SFT, DPO, Instruct read the SFT, DPO and Instruct replies with the LM head), Terra labels on G1-G2, 5 GPUs in parallel.

**Verdict (Fable):** do not rent GPUs yet. The reuse code fails for G3-G6 and R1-R4 as it stands; the plain-frame default comparison measures the description line; the role count is fixed at generation, so "generate first, freeze rules later" is unsafe; five pods cost about 25-30% more GPU time than one.

## Points I checked myself (12 Nov)

| Fable point | My check | Result |
|---|---|---|
| Pod B has no default option; model and revision are constants | `261029_base_role_vs_format/gen_extract.py:169-180` (`descriptions()` needs 5 role wordings per role), `:556-565` (`load_model` asserts the pinned `REVISION`) | Confirmed |
| Pod C reader accepts Instruct text only | `261029_fixed_text_read_through/replies.py` `read_pod_a`: `check_model(manifest, "instruct")` | Confirmed |
| Upload path can overwrite pod A files | `261029_stage_persona_plan/hf_archive.py` `remote_paths`: prefix is `experiments/q1_emergence/<run folder's parent>`. A run named `run_full` inside pod A's folder would land on pod A's `run_full`. Pod A's default output name is `run_<roles>`, not `run_full`. | Risk confirmed; avoid with a new folder and per-model run names |
| G. and Lindsey's SFT effect is in the Assistant field only | `output/lit_privileged_2026-11-12/2605.25459.squeezed.txt:257-267` | Confirmed: a plain-text read frame would miss the SFT effect |
| The plain stage trend of own-text familiarity is already published for OLMo-2-7B | `output/lit_stages_2026-11-12/txt/2609.32196.txt:153, 210-245`: OLMo-2-1124-7B base, SFT, DPO, Instruct wrote stories; the trained model scores them +0.434, -0.121, -0.567, -0.581 nats per token against base | Confirmed |
| RunPod bills by the second | https://docs.runpod.io/pods/pricing ("Pods are billed by the second for compute and storage") | Confirmed. Modal not checked by me |
| SFT and DPO weight files load without a code fix | Pod C loaded both (Fable cites `checkpoints.json`, pod C `extract.py:237-238`) | Not re-checked; Fable's reading |

## Fable points not yet checked by me

- The plain-frame default confound (logic: removing the role line leaves one prompt per question and format, while role prompts carry the description; pod B C3 decodes the role from the prompt alone at 0.91-0.98). The logic follows from `gen_extract.py` templates; I agree with it.
- Effect sizes for Nguyen's coin-flip test at 7-8B (Llama 8B +0.078, Qwen 7B +0.235).
- Time and storage figures for R (about 250,000 sequences per reader; about 34 GB for 33 states).
- Five pods about 25-30% more GPU time (about $5-8 at $1.5-2.5 per hour); three pods by model about 5 hours wall clock at near-serial cost.

## Fable's three main changes

1. Before any GPU: a precision simulation and the default-position test on saved pod A and pod C states ($0); fix the role count from it; send Thomas the outline.
2. Redesign the plain-frame default: compare default and roles in a read frame without the description, with spread computed within wording; add the default path to the pod B code with tests and a canary per stage.
3. One read design: each writer's chat frame read by all four readers with a float32 LM head, plus a prompt-only frame and the coin-flip items; a new folder and per-model run names.

Suggested cuts: the Lu prefills (no measure planned; conflicts with the owner's "Option 1 + 4"), two of the four plain formats, 33-state storage, judge labels beyond 100 per role per stage, the plain stage-trend entropy (already published).
