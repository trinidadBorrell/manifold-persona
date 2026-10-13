# Part 2 final re-review

**The one-line answer:** One blocker remains. B4 is not fixed.

---

**Status**

| ID | Status | Blocker |
|---|---|---|
| B4 | NOT FIXED | Yes |
| N1 | FIXED | No |
| N2 | FIXED | No |
| N3 | FIXED | No |
| N4 | FIXED | No |
| New blocker | None | No |

---

**B4 — NOT FIXED**

The self-test now has real assertions. It also uses 50 halves for both the map estimate and each bootstrap draw.

The positive gain check is useful. The null gain check is not strict enough.

Line 557 only requires `ci(g2)[0] < 0.02`. A fully positive interval can pass this test.

For example, `[0.0105, 0.0295]` excludes zero but passes the current assertion. Production would treat that interval as positive gain.

The test also uses one baseline. It does not exercise the production best-of-three selection path.

B4 remains a blocker. Require the null interval to include zero and test baseline selection.

---

**N1 — FIXED**

`cache_path()` hashes the run fingerprint, this source file, tag, draw count, and fixed seeds.

Old cache names cannot silently control the new result. `load_cache()` stops on a wrong draw count.

---

**N2 — FIXED**

`check_design()` requires 20 roles, 18 questions, and five wordings. It checks each role separately.

The self-test confirms that a reduced question set stops. A separate check confirmed reduced wording and role coverage also stop.

---

**N3 — FIXED**

The embedding cache now uses `cache_path()`. Its identity includes the run fingerprint, code, model settings, row count, and seeds.

The row-count guard stops a wrong-size embedding cache.

---

**N4 — FIXED**

`usable()` is now the single validity rule. Subsetting, invalid counts, and axis selection all use it.

---

**Evidence**

| Check | Result |
|---|---|
| Supplied self-test | PASS; exit code 0; `SELFTEST PASSED` |
| Cache identity variants | PASS |
| Wrong draw count | PASS; stopped with `AnalysisError` |
| Reduced design margins | PASS; stopped with `AnalysisError` |
| B4 counterexample | FAIL; a positive interval passes the null assertion |

The Arrow cache-size warnings do not affect the self-test result.

---

**Final verdict**

Do not run the final analysis yet. Fix B4 first.
