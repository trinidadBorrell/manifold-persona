# Fable 5.2 review of the concurrent-work plan (12/11/2026)

Reviewer: Fable 5.2 (`claude-fable-5-2`), Agent subagent of session ad361f36, brief `prompt_plan_concurrent_261112.md`. Agent review, not a human review. Summary of its report, with the points I re-checked myself.

**Verdict (Fable):** not ready. The plan does not test Han Xiao's findings like for like, its role-count logic uses the wrong noise, and "we confirm their findings" is premature (0 of 3 tested on OLMo-2 own replies).

## Points I re-checked (12 Nov)

| Fable point | My check | Result |
|---|---|---|
| In role-SD units Han Xiao's desirable-versus-undesirable gap disappears | Recomputed from his figure data (`output/lit_stages_2026-11-12/lw/axis_shift_stages_clean.md`, layer 16, 274 roles, 48 desirable, 25 undesirable): raw desirable +0.183, undesirable +0.011, gap +0.172, label-shuffle p 0.071; in role-SD units (z within each stage) desirable +0.135, undesirable +0.157, gap -0.022, p 0.886. Role spread 0.613 at SFT, 0.709 at DPO | Confirmed. His notebook lists "z-score units" as "Not yet checked" (`output/lit_stages_2026-11-12/gh/q1_robustness.py.txt:86`) |
| Our 20 pilot roles under his labels: 5 desirable (translator has no SFT vector), 5 undesirable, 10 neutral | His labels for our roles | Confirmed |
| Pod A's generator writes to `run_full` by default (`--roles` default "full"), so an SFT run in pod A's folder would land on pod A's HF path | `gen_extract.py:184`; `hf_archive.py` prefix rule | Confirmed; corrects my note in `fable_rerun_check_261112.md` |
| Pod A's D3 rule: if D3 fails, pod C uses maps only | `261029_instruct_persona_map/THRESHOLDS.md:65-72` | Confirmed; a new experiment needs its own axis rule |
| T-073 promises "the 276-role map across stages" | Task tracker row T-073 | Confirmed |

## Other Fable points (not re-checked)

- His design: axis = default minus the mean of 275 role vectors; role vectors from replies scored 3 of 3 by his judge (at least 50 per role); temperature 0.7, top-p 0.9, 512 tokens; interval from 100 question half-splits; p from a label shuffle. Ours: Terra, 128 tokens, temperature 0.5.
- Power for his gap test (normal approximation): about 32% at 20+20 roles, about 53% at 48+25; 80% needs about 70+70.
- Use all 49+25 labelled roles plus at least 40 neutral roles (or all 275) for a like-for-like test; regenerate Instruct rather than reuse pod A.
- Storage: about 16 GB per stage at 60,000 rows with 33 states; store 4 states.
- Missing before run: a new experiment folder with THRESHOLDS (primary measure, unit, filter, "no change" margin, axis rule), a costed budget, code fixes (`gen_extract.py`, `gates.py`, `analysis.py`), HF token and pod.
- Keep priority claims and the mentor quote out of the post; write "concurrent". The OLMo-2-1B example-turn figure comes from the 2 Oct audit; owner call under the 5 Oct rule.
- Dangling items to settle before the post: outline to Thomas; judge-validation reporting rules; HF copies of 12 Nov outputs; fork files; `/challenge-result` and human review of the free checks; base default (Option 1 + 4) park or redesign; T-073 scope.
