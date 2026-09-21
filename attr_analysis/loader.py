"""One loader for both storage formats.

The first 54 personas were saved as raw per-response activations; the 247 that
followed were reduced on the cluster to means plus per-response points at the
three plotted layers. Downstream code should not care which is which.
"""
import glob

import numpy as np

LAYERS = (26, 28, 30)


CACHE = "attr_analysis/_merged_cache.npz"


def load_all(raw_globs=("attr_acts/*.npz", "attrx_acts/*.npz"),
             reduced=("attrall_reduced.npz",), use_cache=True, need_pts=True):
    """-> means[layer][persona] (2048,), pts[layer][persona] (n, 2048)

    Merging 300 personas from a dozen files takes minutes, so the result is
    cached; delete attr_analysis/_merged_cache.npz after adding new data.
    """
    import os
    if use_cache and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        names = list(z["names"].astype(str))
        means = {L: {p: z[f"m{L}"][i] for i, p in enumerate(names)} for L in LAYERS}
        pts = ({L: {p: z[f"p{L}_{i}"] for i, p in enumerate(names)} for L in LAYERS}
               if need_pts else {L: {} for L in LAYERS})
        return means, pts, names
    means = {L: {} for L in LAYERS}
    pts = {L: {} for L in LAYERS}
    for g in raw_globs:
        for f in sorted(glob.glob(g)):
            z = np.load(f, allow_pickle=True)
            X = z["prompted"].astype(np.float32)
            who = z["persona"].astype(str)
            for p in np.unique(who):
                Y = X[who == p]
                for L in LAYERS:
                    means[L][p] = Y[:, L, :].mean(0)
                    if need_pts:
                        pts[L][p] = Y[:, L, :].astype(np.float16)
            del X, z
    for f in reduced:
        for ff in sorted(glob.glob(f)):
            z = np.load(ff, allow_pickle=True)
            names = z["personas"].astype(str)
            allL = list(z["layers"])
            for i, p in enumerate(names):
                for L in LAYERS:
                    if L in allL:
                        means[L][p] = z["means"][i, L, :].astype(np.float32)
                        if need_pts:
                            pts[L][p] = z[f"pts_{L}"][i]
    names = sorted(means[LAYERS[0]])
    if use_cache and need_pts:
        d = {"names": np.array(names)}
        for L in LAYERS:
            d[f"m{L}"] = np.stack([means[L][p] for p in names]).astype(np.float32)
            for i, p in enumerate(names):
                d[f"p{L}_{i}"] = pts[L][p].astype(np.float16)
        np.savez(CACHE, **d)
    return means, pts, names


if __name__ == "__main__":
    m, p, names = load_all()
    print(f"{len(names)} personas")
    print(f"  example {names[0]}: mean {m[28][names[0]].shape}, pts {p[28][names[0]].shape}")
    print(f"  has default: {'default' in names}")


def load_five(raw_globs=("attr_acts/*.npz", "attrx_acts/*.npz")):
    """Means at all five steering layers, for personas with full 37-layer data."""
    import glob as _g
    means = {L: {} for L in (26, 27, 28, 29, 30)}
    for g in raw_globs:
        for f in sorted(_g.glob(g)):
            z = np.load(f, allow_pickle=True)
            X = z["prompted"].astype(np.float32); who = z["persona"].astype(str)
            for p in np.unique(who):
                Y = X[who == p]
                for L in means:
                    means[L][p] = Y[:, L, :].mean(0)
            del X, z
    return means, {L: {} for L in means}, sorted(means[26])
