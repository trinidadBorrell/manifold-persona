# axis_interpretability (Oct 20)

**Question.** Are the PGA (Fletcher) directions of persona space easier to interpret than the PCA directions?

**Data.** Qwen3-8B, 276 Assistant Axis persona vectors, layer 28, last answer token (`$PM/E1`), directions 1–3 (`$PM/AXI_L28`) and 1–50 (`$PM/AXI_L28_K50`).

**Judge.** Claude Sonnet 5.5, interactive API, blind to the method. Each role is shown as its name and first system prompt.

| Exp | What |
|---|---|
| 1 | Rate the 10 roles at each end. A 1–5 theme score per end and a 1–5 contrast score, with fixed anchors, reasoning first. 5 repeats per axis. |
| 2 | LLM-1 labels the axis from 10 + 10 of the 20 most extreme roles per end. LLM-2 sees only the label and sorts the other 20 into pole A or B. 5 splits. The score is the % put at their true end. The explain-then-predict check from automated interpretability. |
| 3 | What Lu et al. (Assistant Axis, §2.2, §2.3.1, Table 1) did: the top 5 roles per end by cosine, and the Assistant's relative position on each direction. No LLM. |
| 4 | LLM-2 gets the Exp 2 label and 20 unseen roles drawn from along the whole axis (one per score bin), and orders them from pole A to pole B. The score is Spearman ρ against the true score. |
| 5 | Like Exp 4 but with no label: 5 + 5 example roles at the poles (from the 20 most extreme), then order 20 roles from along the whole axis. |

## Files
| File | Role |
|---|---|
| `scores.py` | Step 0, run on Condor: every role's PCA / PGA score and cosine, plus the no-system-prompt PCA position. Checks the ends against `P2full`. |
| `prompts.py` | The five judge prompts: rate, label, sort, rank, anchor |
| `build.py` | Item files. Fixed seeds; PCA k and PGA k get the same shuffles. |
| `judge.py` | API calls, one JSON per call, resumable, `--dry-run` gives a $ estimate, `API_BUDGET` cap (default $10). Uses `route_judge_sweep.api_judge.clients`. |
| `analyse.py` | Figures, tables and `summary.json` into `output/20-oct`, directions 1–3 |
| `examples.py` | Real prompts and replies for 7 directions into `output/20-oct/examples` |
| `analyse_many.py` | Exp 1, 2 and 4 over directions 1–K, summarised over 1–20 and 1–50, into `output/20-oct/k50` |

## Commands
```bash
D=$PM/AXI_L28
condor_submit jobs_condor/axi_scores.submit                  # scores.npz, scores_check.json
python -m axis_interpretability.build --dir $D
python -m axis_interpretability.judge --dir $D --stage exp1   # 30 calls
python -m axis_interpretability.judge --dir $D --stage label  # 30 calls
python -m axis_interpretability.judge --dir $D --stage sort   # 60 calls (held-out + seen)
python -m axis_interpretability.analyse --dir $D --out output/20-oct
# directions 1-50 (Exp 1, 2, 4): about 2,500 calls, about $20
condor_submit jobs_condor/axi_scores_k50.submit              # $PM/AXI_L28_K50/scores.npz (--k 50)
D=$PM/AXI_L28_K50; python -m axis_interpretability.build --dir $D
for s in exp1 label sort rank anchor; do python -m axis_interpretability.judge --dir $D --stage $s; done
python -m axis_interpretability.analyse_many --dir $D --out output/20-oct/k50
```

## Result
Full write-up: `output/20-oct/RESULTS.md`.
- **Overlap:** PGA 1 ≈ −PCA 1 (ρ = −0.96) and PGA 3 ≈ PCA 2 (0.84).
- **Exp 1:** PCA scores the same as PGA or higher on all three directions. PGA 2, the only new direction, is lowest (theme score 3.0, contrast 3.2).
- **Exp 2:** held-out accuracy is 92–99% for all six axes. That is the ceiling, so this test does not separate the methods.
- **Exp 4:** ρ is 0.86 / 0.79 / 0.57 for PCA 1–3 and 0.82 / 0.68 / 0.64 for PGA 1–3.
- **Directions 1–50:** PGA is not better on any measure. Exp 4 ρ over 1–20 is 0.34 for PCA and 0.27 for PGA (p = 0.11). Over 1–50 it is 0.16 and 0.09 (p = 0.0005). Both methods become unreadable after about direction 20. Every score follows the direction's share of variance.
- **Exp 5:** ρ over 1–20 is 0.41 for PCA and 0.28 for PGA (p = 0.006). It is the clearest PCA advantage. 4 of the 500 calls are missing.
- **Exp 3:** our PCA 2 and PCA 3 ends match the paper's Qwen 3 32B lists (4 out of 5 roles on three of the ends). The no-system-prompt state sits 0.07 from the edge of PCA 1 (paper: 0.03).
