# E3 `gen_extract.py` third review

**One-line answer** Do not start the full run. Five findings need a fix or a verified resource check.

---

**Result**

| Severity | Finding | Full-run blocker |
|---|---|---|
| High | A dry run can overwrite metadata in a valid run directory. | Yes |
| High | Final assembly has a minimum 9.5 GiB host-memory spike. | Yes |
| Medium | The replication includes 54 unintended default items. | Yes |
| Medium | A truncated reply line makes crash resume fail. | Yes |
| Medium | The smoke output is stale and used batch size 20. | Yes |

---

**Finding 1 — High — `gen_extract.py:289-301`**

- **Failure:** `--dry` skips `claim_out_dir()`, but it still opens `items.jsonl` with mode `w`.
- **Failing case:** Run the main job, then run `--dry --replicate --seed_offset 1` with the default output path.
- **Result:** The command replaces main-run metadata with replication metadata. It leaves the old fingerprint, replies, activations, and manifest in place.
- **Fix:** Refuse every nonempty output directory in dry mode, or use a separate dry-output path. Never write before this check.

---

**Finding 2 — High — `gen_extract.py:310-317`**

- **Failure:** Final assembly loads all activation parts and then makes a full concatenated array.
- **Failing case:** For 15,120 rows, the final array is 3.81 GiB. Concatenation can use 7.61 GiB, and the finite check raises the minimum peak to about 9.52 GiB.
- **Result:** A host with limited free RAM can fail after all model work finishes. The parts and final array also use about 7.61 GiB of disk together.
- **Fix:** Create the final `.npy` with `open_memmap()`. Copy and validate one part at a time. Check free RAM and disk before the run.

---

**Finding 3 — Medium — `gen_extract.py:284-286`**

- **Failure:** The replication filter also selects the three default frames because they use `cond="desc"` and shared questions.
- **Failing case:** The intended set has `20 × 5 × 2 × 18 = 3,600` items. The code builds 3,654 items: 3,600 role items plus 54 default items.
- **Result:** This does not match the stated 3,600-generation replication in `THRESHOLDS.md:112`.
- **Fix:** Also require `it["role"] in PILOT_ORDER`. Add exact assertions for 3,600 items, 20 roles, two conditions, five wordings, and 18 questions.

---

**Finding 4 — Medium — `gen_extract.py:221-265`**

- **Failure:** Resume parses every line in `replies.partial.jsonl` without handling a truncated last line.
- **Failing case:** Stop the process during the direct append at lines 263-265. A partial JSON record makes line 224 raise `JSONDecodeError` on every restart.
- **Result:** The activation-part check does not run. The job cannot resume without manual file repair.
- **Fix:** Write one reply file and one activation file per batch. Write temporary files, flush them, and atomically rename both before marking the batch complete.

---

**Finding 5 — Medium — `gen_extract.py:54-63,276`; `smoke/manifest.json:31,47,58`**

- **Failure:** The smoke output does not come from the reviewed code or the full-run batch size.
- **Failing case:** Its code hash is `3ac5...b285`; the reviewed file hash is `50a9...e4517`. Its filler has 178 words, while the current filler has 161 words.
- **Result:** The smoke used batch size 20. It gives no evidence that current code works at the default batch size 32.
- **Fix:** Run a new smoke in a new directory with current code and batch size 32. Test normal completion, forced resume, and fingerprint refusal.

---

**Checks that passed**

| Check | Result |
|---|---|
| Main item count | Pass: 15,120 unique items, IDs 0 through 15,119 |
| Condition counts | Pass: desc 4,120; ex2 3,800; ex2_other 5,400; ex2_filler 1,800 |
| Donors | Pass: three unique donors per target; no donor is the target; every donor is outside the target group of four |
| Shared questions | Pass: 18 unique questions unused by every example |
| Example leakage | Pass: no selected example question equals its test question |
| Question counts | Pass: desc 40; ex2 38; ex2_other 18; ex2_filler 18 per role and wording |
| Replication overwrite, non-dry | Pass: the fingerprint differs and blocks reuse of the main directory |
| `SEED_OFFSET` | Pass: it controls the seed and appears in the fingerprint and manifest |
| Earlier O1-O6 | The requested code changes are present; Findings 1 and 4 show remaining crash-safety gaps |

---

**Resource note**

Batch size 32 is not an obvious A100 80 GB GPU-memory failure from static sizing. The current smoke does not test it.

The main risk is host memory during final assembly. Do not rely on GPU memory size as proof of host capacity.

---

**Required before the full run**

1. Fix Findings 1, 3, and 4.
2. Stream final assembly, or prove enough free host RAM and disk for Finding 2.
3. Run the current batch-32 smoke required by Finding 5.
