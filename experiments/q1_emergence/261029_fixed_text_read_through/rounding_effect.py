"""How much bf16 batch rounding moves the role map at state 17: pod A's generation-time states against pod C's
re-extraction of the same ids (F0 at Instruct), and the E3 F3 rounding gap against the E3 role spread."""
import sys
import numpy as np
from scipy.stats import spearmanr
sys.argv = sys.argv[:1]
import analysis as AN

HERE = AN.HERE
run = AN.Run(HERE / "frames_full", HERE / "slim_canary", ["base", "instruct"])


def role_map(X, c, C):
    M = np.array([X[c == k].astype(np.float64).mean(0) for k in range(C)])
    Mc = M - M.mean(0)
    n = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
    D = 1 - n @ n.T
    return M, Mc, D[np.triu_indices(C, 1)]


m = run.meta("pod_a")
c, C = m["c"], m["C"]
stored = np.load(HERE.parent / "261029_instruct_persona_map/run_full/reply_mean_acts.npy", mmap_mode="r")
gen = np.asarray(stored[:, 17])
re = np.asarray(run.states("instruct", "pod_a", "F0", "reply", 17))
ok = (c >= 0) & np.isfinite(re).all(1) & np.isfinite(gen).all(1)
Mg, Mgc, dg = role_map(gen[ok], c[ok], C)
Mr, Mrc, dr = role_map(re[ok], c[ok], C)
row_rel = np.linalg.norm(re[ok] - gen[ok], axis=1) / np.linalg.norm(gen[ok], axis=1)
print(f"pod A rows: median per-row rounding gap {np.median(row_rel):.4f}, max {row_rel.max():.4f}")
print(f"pod A role spread (centred role means / grand mean): {np.linalg.norm(Mgc) / np.sqrt(C) / np.linalg.norm(Mg.mean(0)):.4f}")
print(f"pod A change of the centred role means: {np.linalg.norm(Mrc - Mgc) / np.linalg.norm(Mgc):.5f}")
print(f"pod A role-map Spearman r (generation vs re-extraction): {spearmanr(dg, dr)[0]:.5f}; max |distance change| {np.abs(dg - dr).max():.5f} on distances {dg.min():.3f}-{dg.max():.3f}")

for src in ("e3_desc", "e3_ex2"):
    m = run.meta(src)
    c, C = m["c"], m["C"]
    for ck in ("base", "instruct"):
        X = np.asarray(run.states(ck, src, "F1", "reply", 17))
        ok = (c >= 0) & np.isfinite(X).all(1)
        M, Mc, _ = role_map(X[ok], c[ok], C)
        spread = np.linalg.norm(Mc) / np.sqrt(C) / np.linalg.norm(M.mean(0))
        print(f"{src} {ck}: role spread {spread:.4f} of the mean state; F3 rounding gap (max) {'0.0057' if ck == 'base' else '0.0068'}")
