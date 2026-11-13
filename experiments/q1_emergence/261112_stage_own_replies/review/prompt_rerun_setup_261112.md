My friend wrote this GPU run setup and wants a blunt adversarial code review before paying for the full run. Find what is wrong. Do not praise. Read the files; do not run GPU code.

Goal: OLMo-2-1124-7B SFT and DPO each write their own role replies on pod A's exact grid (20 roles + default, 5 wordings, 40 questions, 5 replicates = 21,000 replies per stage), using pod A's generator and gates unchanged. Only the model and pinned revision change.

Files (repo root = working directory):
- experiments/q1_emergence/261112_stage_own_replies/THRESHOLDS.md (rules), stage_run.py (wrapper), modal_own_replies.py (Modal launcher), test_stage_run.py (CPU test: SFT and DPO render pod A's 21,000 stored prompt ids exactly; it passes).
- Pod A code it reuses: experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py, gates.py, THRESHOLDS.md, NOTE.md; src/manifold_persona/rows.py.
- Pinned revisions: experiments/q1_emergence/261112_coinflip_stages/stages.json (from experiments/q1_emergence/261029_fixed_text_read_through/checkpoints.json).
- The weights volume was filled by experiments/q1_emergence/261112_modal_gpu/modal_runs.py (prefetch step).

Check, with file:line evidence for each finding:
1. Does the wrapper swap the model and revision everywhere it matters: generation, tokenizer hashes, fingerprint, manifest, gates G3/G4/G5? Can any path still load or record the Instruct model?
2. Does anything in pod A's code still assume Instruct or pod A (paths, run names, output folders, hard-coded checks, the stub package trick in stage_run.py)?
3. SFT and DPO ship .bin (pickle) weights. Are they loaded safely and at the pinned commit, offline, in this image (torch 2.13.0+cu129, transformers 4.47.1)?
4. Modal launcher: volumes, HF offline mode, timeouts, the 5-minute commit thread, resume after a crash, two stages in parallel on one volume, cost risks.
5. Is the run comparable to pod A's Instruct run (same seeds by key, same sampling, same pooling)? List differences from Han Xiao's design (Olmo 3; 275 roles; 240 questions; temperature 0.7, top-p 0.9, 512 tokens; judge score-3 filter) that matter for comparing with him.
6. Anything that would waste the GPU run or make its outputs unusable.

Output: findings ranked by severity. For each: the claim, file:line evidence, why it matters, and the smallest fix. Say "none found" for a check with no finding. End with a one-line verdict: run as is / fix first.
