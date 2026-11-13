"""Owner review 13/11/2026: a second, direct build of the pod C point values (no Gram matrices, no bootstrap).

Reads the F1 reply states at hidden state 17 for base, SFT, DPO and Instruct (HF run_full copies under
261112_free_checks/data), the fixed halves in frames_full/frames.json and the F2 reply ids for the text map.
Writes recompute.json next to this file.

    .venv/bin/python -I experiments/q1_emergence/261029_fixed_text_read_through/review/owner_review_261113/recompute.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PODC = HERE.parents[1]
Q = PODC.parent
DATA = Q / "261112_free_checks/data/experiments/q1_emergence/261029_fixed_text_read_through/run_full"
PODA = Q / "261029_instruct_persona_map/run_full"
STATE = 17
READERS = ("base", "sft", "dpo", "instruct")
CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000, sublinear_tf=True)


def load_meta():
    fr = json.loads((PODC / "frames_full/frames.json").read_text())["sources"]["pod_a"]
    keys = json.loads((PODA / "grid.json").read_text())["keys"]
    assert keys == fr["keys"], "pod A grid order differs from the frame order"
    roles, qs = fr["roles"], sorted(fr["questions"])
    ri, qi = {r: i for i, r in enumerate(roles)}, {q: i for i, q in enumerate(qs)}
    k = [json.loads(x) for x in keys]
    c = np.array([ri.get(x[0], -1) for x in k])
    q = np.array([qi[x[3]] for x in k])
    w = np.array([x[2] for x in k])
    halves = [(np.array(a, bool), np.array(b, bool)) for a, b in fr["halves"]]
    return roles, c, q, w, halves


def cells(X, c, q, w, n_roles):
    S = np.zeros((n_roles, q.max() + 1, 5, X.shape[1]))
    N = np.zeros(S.shape[:3])
    keep = c >= 0
    np.add.at(S, (c[keep], q[keep], w[keep]), X[keep])
    np.add.at(N, (c[keep], q[keep], w[keep]), 1)
    assert (N == 5).all(), "a prompt does not have 5 replicates"
    return S / N[..., None]


def distances(V):
    V = V - V.mean(0)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return 1 - V @ V.T


def tri(D):
    return D[np.triu_indices(len(D), 1)]


def cross_half(side_a, side_b):
    """side_x[h] = (map on half A, map on half B) as pair-distance vectors."""
    out = []
    for (a1, a2), (b1, b2) in zip(side_a, side_b):
        out.append((spearmanr(a1, b2)[0] + spearmanr(b1, a2)[0]) / 2)
    return float(np.mean(out))


def state_sides(S, halves):
    return [(tri(distances(S[:, qa][:, :, wa].mean((1, 2)))), tri(distances(S[:, ~qa][:, :, ~wa].mean((1, 2)))))
            for qa, wa in halves]


def text_sides(texts, c, q, w, halves, n_roles):
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    out = []
    for qa, wa in halves:
        pair = []
        for qm, wm in ((qa, wa), (~qa, ~wa)):
            sel = np.where((c >= 0) & qm[q] & wm[w])[0]
            X = TfidfVectorizer(**CHAR).fit_transform([texts[j] for j in sel])
            # every prompt has 5 replicates and every role the same prompts, so the role mean of rows equals the
            # mean of prompt means
            A = sparse.csr_matrix((np.ones(len(sel)), (c[sel], np.arange(len(sel)))), shape=(n_roles, len(sel)))
            pair.append(tri(distances(np.asarray((A @ X).todense()) / np.bincount(c[sel])[:, None])))
        out.append(tuple(pair))
    return out


def main():
    from tokenizers import Tokenizer
    roles, c, q, w, halves = load_meta()
    sides, full = {}, {}
    for ck in READERS:
        meta = json.loads((DATA / ck / "pod_a.F1.json").read_text())
        X = np.load(DATA / ck / "pod_a.F1.reply.npy", mmap_mode="r")[:, meta["layers"]["reply"].index(STATE), :]
        S = cells(np.asarray(X, np.float64), c, q, w, len(roles))
        sides[ck] = state_sides(S, halves)
        full[ck] = distances(S.mean((1, 2)))
        print(ck, "loaded", flush=True)
    tok = Tokenizer.from_file(sys.argv[1])
    z = np.load(PODC / "frames_full/frames.npz")
    ids, off = z["pod_a.F2.ids"], z["pod_a.F2.offsets"]
    texts = [tok.decode(ids[off[j] + 1:off[j + 1]].tolist()) for j in range(len(off) - 1)]
    sides["text_char"] = text_sides(texts, c, q, w, halves, len(roles))
    print("text done", flush=True)

    r = {f"{a}|{b}": cross_half(sides[a], sides[b])
         for a, b in [(x, x) for x in sides] + [(x, "instruct") for x in sides if x != "instruct"]
         + [("base", "sft"), ("sft", "dpo")]}
    norm = lambda a, b: r[f"{a}|{b}"] / np.sqrt(r[f"{a}|{a}"] * r[f"{b}|{b}"])
    out = dict(
        note="direct second build, point values only; compare with results_full_v2.json",
        r=r,
        r_norm_to_instruct={a: norm(a, "instruct") for a in ("base", "sft", "dpo", "text_char")},
        stage_change={"base->sft": 1 - norm("base", "sft"), "sft->dpo": 1 - norm("sft", "dpo"),
                      "dpo->instruct": 1 - norm("dpo", "instruct")},
        roles=roles,
        full_data_distance={k: v.tolist() for k, v in full.items()},
        full_data_map_r={f"{a}|instruct": float(spearmanr(tri(full[a]), tri(full["instruct"]))[0])
                         for a in ("base", "sft", "dpo")},
    )
    (HERE / "recompute.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("r", "r_norm_to_instruct", "stage_change", "full_data_map_r")}, indent=1))


if __name__ == "__main__":
    main()
