**Fix first: failures can appear successful, and crash recovery can corrupt or block a resumed run.**

Read-only review with CPU mocks only. No GPU code, model loading, or remote jobs ran.

---

**1. High — The launcher hides failures and does not enforce smoke success**

- **Evidence:** [modal_own_replies.py:83](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/modal_own_replies.py:83) catches gate failures; [line 108](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/modal_own_replies.py:108) prints remote exceptions without failing the entrypoint; selecting `full` checks no smoke result.
- **Why it matters:** A failed smoke or generation job can leave the CLI successful, so shell automation can proceed to paid generation.
- **Additional defect:** [line 89](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/modal_own_replies.py:89) misreads the [gate output structure](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gates.py:158), producing `"gates": null` instead of individual results; CPU mocks reproduced both failures.
- **Smallest fix:** Propagate failures, read `gates["gates"][name]["pass"]`, and require successful smoke receipts for the same stage, source and image before `full`.

---

**2. Medium — Crash recovery misses valid failure states**

- **Evidence:** [gen_extract.py:85](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:85) strips newline boundaries, while [line 99](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:99) repairs only when the number of retained lines changes.
- **Why it matters:** A crash after a complete final JSON object but before its newline leaves a batch marked complete; the next append joins two objects, which I reproduced as `JSONDecodeError: Extra data`.
- **Second case:** [line 119](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:119) aborts when completed rows lack their state part, instead of regenerating that batch; five-minute volume commits do not repair either condition.
- **Smallest fix:** Add wrapper recovery that restores the trailing newline and removes completed-row claims for missing or corrupt state parts before delegating to unchanged pod A code.

---

**3. Medium — The resume fingerprint does not identify the exact inputs or runtime**

- **Evidence:** [gen_extract.py:205](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:205) fingerprints grid keys, tokenizer, source and sampling, but omits rendered prompt content, runtime versions and attention backend.
- **Why it matters:** An instruction or question can change without changing its key; a restart can then combine old and new prompts under one fingerprint.
- **Why gates miss it:** [G3 regeneration](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gates.py:90) and [G4 reconstruction](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gates.py:117) use stored prompts, so they do not establish agreement with the intended input bank.
- **Smallest fix:** Write and validate a wrapper receipt containing ordered `(key, prompt_ids)` hashes, runtime identity and wrapper hash before generation and every resume.

---

**4. Low — Relative generation and gate paths disagree**

- **Evidence:** [gen_extract.py:198](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:198) interprets relative `--out` from the working directory; [gates.py:143](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gates.py:143) interprets relative `--run` from pod A’s folder.
- **Why it matters:** Generating into `runs/sft` and gating `runs/sft` can address different directories.
- **Smallest fix:** Resolve both arguments to absolute paths in the wrapper.
- **Scope:** The Modal launcher already supplies absolute paths, so this affects direct CLI use.

---

**5. Low — Storage is underestimated by roughly twofold**

- **Evidence:** [gen_extract.py:136](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:136) saves batch arrays; [line 163](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:163) creates another complete array and retains the parts.
- **Why it matters:** `[21000, 33, 4096]` float16 occupies **5.68 GB**, giving approximately **11.35 GB per stage**, plus rows and metadata.
- **Mismatch:** [THRESHOLDS.md:32](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/THRESHOLDS.md:32) budgets about 5.7 GB per stage.
- **Smallest fix:** Budget for both copies; remove checkpoints only after a verified archive and owner-approved removal.

---

**Checks with no finding**

| Check | Result and evidence |
|---|---|
| Model swap and Instruct fallback | **None found** in the launcher path: [wrapper assignment](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/stage_run.py:34) reaches [tokenizer hashes](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:179), [fingerprint/manifest inputs](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:205), and [generation loading](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:228). |
| G3/G4/G5 using Instruct | **None found:** [gates.py:148](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gates.py:148) loads the patched repository at the manifest revision and passes that model to G3/G4; G5 receives its commit and patched tokenizer hashes. |
| Stub package trick | **None found:** [stage_run.py:25](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/stage_run.py:25) gives the package its source path; CPU checks confirmed that gates import the patched generator module. |
| Unsafe `.bin` defaults | **None found:** Transformers 4.47.1 defaults `from_pretrained` to `weights_only=True` and passes it to `torch.load`; these calls do not disable it ([versioned source](https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/modeling_utils.py#L2904-L2917)). |
| Wrong revision or online fallback | **None found:** [prefetch pins the revision](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_modal_gpu/modal_runs.py:44), [generation checks the loaded commit](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:230), and [the child environment enables offline mode](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/modal_own_replies.py:60), which Transformers [honours](https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/utils/hub.py#L77-L81). |
| Default parallel volume use | **None found:** [stage-specific directories](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/modal_own_replies.py:74) separate SFT/DPO writes; [Volumes v2 supports distinct-file concurrent writers](https://modal.com/docs/guide/volumes#in-v2-you-can-write-concurrently-from-hundreds-of-containers). |

These are source-level conclusions; the actual remote cache bytes and image execution remain unverified.

---

**Modal durability and cost**

The [commit thread](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/modal_own_replies.py:51) has no error propagation or join, so a commit exception stops that thread without failing the GPU job. Modal also provides [background commits every few seconds](https://modal.com/docs/guide/volumes#background-commits), so the custom thread is not the sole persistence mechanism.

The [three-hour timeout](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_stage_own_replies/modal_own_replies.py:18) covers generation **and gates** per stage. Two stages allow **six GPU-hours per full attempt**, plus smoke and storage; `retries=0` does not cap spending across manual relaunches.

---

**Comparability with pod A and Han Xiao**

Sampling and pooling differences from pod A: **none found**. Keys determine [the same seeds](/Users/brendon/persona-manifolds/manifold-persona/src/manifold_persona/rows.py:35), sampling uses the same [custom sampler](/Users/brendon/persona-manifolds/manifold-persona/src/manifold_persona/rows.py:195), and pooling uses the same [kept-reply mean](/Users/brendon/persona-manifolds/manifold-persona/src/manifold_persona/rows.py:223). Generator and gate hashes match archived pod A; the helper’s hash changed only because SHA-256 now reads files in chunks.

| Dimension | This run | Han Xiao | Comparison limit |
|---|---|---|---|
| Model | OLMo-2-1124-7B | Olmo 3 7B | Different model families and training histories |
| Roles | 20 + default | 275 + default | Changes role population, axis origin and scale |
| Questions | 40 × 5 replicates | 240 distinct questions | Replicates do not replace question coverage |
| Sampling | Temperature 0.5; top-p 1.0 | Temperature 0.7; top-p 0.9 | Different reply distributions |
| Length cap | 128 tokens | 512 tokens | Different truncation and pooled text |
| Filtering | Nonempty valid replies | Judge score 3 for roles; unfiltered default | Different populations enter role vectors |

Local evidence: [current sampling rules](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_instruct_persona_map/gen_extract.py:7), [Han’s released grid](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_han_reanalysis/data/han_release/README.md:20), and [filtering rule](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_han_reanalysis/data/han_release/README.md:40); Han’s sampling values are those supplied in your brief. Call this a matched pod A stage comparison; applying Han’s labels later does not make it a direct replication.

**Verdict: fix first.**