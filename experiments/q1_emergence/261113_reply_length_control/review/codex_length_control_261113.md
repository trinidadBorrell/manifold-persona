**Answer: I found no demonstrated extraction error, but the validation does not fully certify the saved cuts.**

---

**1. Medium — wrong cut states can pass the checks**

C0 checks only `res["full"]`; the save guard accepts any entirely NaN row, including an eligible row ([length_control.py:166](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:166), [length_control.py:175](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:175)).  
I reproduced this on CPU: replacing every `first16` state with NaN still passes C0 and the save guard, while availability remains based on token counts.  
This is a validation defect; **the current saved finite-row counts match the expected counts**, so I found no evidence that this happened here.  
**Smallest fix:** assert the exact expected finite-row mask for each cut, then reload saved arrays and verify shape, dtype, and NaN positions.

---

**2. Low — the smoke requirement permits a stale smoke**

The full launcher checks smoke success and the `length_control.py` hash, but omits `rows.py`, input hashes, model revision, and runtime ([modal_length.py:97](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/modal_length.py:97)).  
A changed dependency or input can therefore reach the full run without a matching smoke.  
I found no such mismatch in these outputs: their recorded code hashes match the current files, and input hashes match the pinned upload receipts.  
**Smallest fix:** compare a complete smoke fingerprint covering code, inputs, model revision, runtime, and extraction settings.

---

**3. Low — “Pod C’s F4 rule” describes only part of that rule**

This job checks overall retention per role ([length_control.py:101](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:101)); Pod C also requires common-question support in fixed halves and bootstrap draws ([Pod C THRESHOLDS.md:23](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_fixed_text_read_through/THRESHOLDS.md:23)).  
The flags are correct under this job’s stated rule, but they do not certify that every later map statistic or interval is available.  
My CPU check using Pod C’s fixed-half functions found no additional fixed-half failure for the cuts marked available; bootstrap support remains analysis-specific.  
**Smallest fix:** describe these flags as role-coverage checks, and enforce half/draw support during analysis.

---

**Checks with no finding**

| Check | Result and evidence |
|---|---|
| **1. Pooling and token bounds** | **None found.** Both implementations use right padding, identical attention masks, implicit positions, and float32 span means ([job:60](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:60), [reference:233](/Users/brendon/persona-manifolds/manifold-persona/src/manifold_persona/rows.py:233)). Cuts start at `len(prompt_ids)` and end at `start + N`; state 0 remains the embedding output ([job:45](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:45)). |
| **2. Row order, keys, and `--limit`** | **None found.** Sorted batches scatter back through their original indices; sampled rows and stored states use the same sorted selection ([job:88](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:88), [job:158](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:158)). All saved keys match their source rows; extra CPU checks covered unsorted lengths and real memory-map sample indexing. |
| **3. C0 implementation** | **None found in the full-mean check itself.** It compares the re-read against the corresponding stored rows at states 9, 17, and 23, using the shared gate ([job:116](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:116), [rows.py:255](/Users/brendon/persona-manifolds/manifold-persona/src/manifold_persona/rows.py:255)). Finding 1 limits what that success proves. |
| **4. Invalid rows and support counts** | **None found under the stated rule.** Invalid rows receive empty spans and count against coverage ([job:51](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:51), [job:106](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:106)). The actual input files contain no invalid rows. |
| **5. Float16 and range handling** | **None found in casting or overflow handling.** Means accumulate in float32; partial non-finite rows, infinities, and values outside float16 range are rejected before casting ([job:77](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:77), [job:175](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:175)). Finding 1 covers the missing NaN-mask assertion. |
| **6. Actual provenance and cancellation** | **None found indicating mixed or partial final outputs.** Extraction finishes before array writes begin, and the launcher refuses nonempty output folders ([job:163](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/length_control.py:163), [launcher:102](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/modal_length.py:102)). SFT completed attempt 1; both reruns completed, and their checksums agree with the upload receipt ([SFT log:16](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/full_attempt1_logs/sft.log:16), [rerun log:51](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/full_v2.log:51), [receipt:4](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261112_modal_gpu/hf_receipt_261113_length.json:4)). |
| **8. Other output-breaking defects** | **None found.** |

---

**Support flags: independently recounted**

These are eligible-row counts; each saved `n_finite_rows` agrees with the recount.

| Stage | First 16 | First 32 |
|---|---:|---:|
| [Instruct](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/full_copies/instruct/support.json:2) | 20,925 — available | 20,549 — available |
| [SFT](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/full_copies/sft/support.json:2) | 19,153 — available | 14,512 — unavailable |
| [DPO](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/full_copies/dpo/support.json:2) | 20,951 — available | 20,650 — available |

---

**7. Design: usable for prefix sensitivity, not an isolated length effect**

Equal token counts still leave different content, reply openings, and retained rows across stages; the cut can exclude short, successful role replies.  
Compare full and prefix states on **one common key mask across the compared stages**, and report full results before and after that mask to expose selection effects.  
Keep the same inference units and paired resamples, check question/wording support, and restrict first32 comparisons to Instruct versus DPO.  
A claim about model changes independent of generated content requires the complementary fixed-text comparison.

---

**Verification limits**

All six supplied CPU tests passed; additional CPU checks passed for unsorted assignment, invalid rows, and memory-map indexing.  
C0 means agreement within the frozen tolerance, not equality: SFT state 17 records a maximum relative vector error of 8.30% ([check.json:26](/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261113_reply_length_control/full_copies/sft/check.json:26)).  
The six cut arrays are absent locally, so I checked their recorded hashes and sizes, not their tensor contents.  
No GPU code ran.

**Verdict: outputs usable as is.**