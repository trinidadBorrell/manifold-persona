"""Pod C analysis (CPU, after every checkpoint is copied off the pod): gates, D1, D2 and the reported measures.

Design: THRESHOLDS.md here. Hidden state 17 is primary; the 20 roles make every map and decoder (default only
for the axis). Halves, draws and the null come from maps.py; frames and halves from frames.py.
- Gates before any metric: G2 (checkpoints.json), the frame and checkpoint SHA256SUMS, G1 (every fed row matched
  its frame hash), G3 at Instruct, and the F3 check (F3 states equal across roles within a question; K-class
  decoding at chance). A failed gate stops the analysis.
- D1: cross-half map r between base and Instruct, F1, pod A text. Pass: lower bound >= 0.5 and one-map null
  p < 0.01. Reading rule against the text anchors (word and character n-grams, a sentence-embedding model and
  hidden state 0 of Instruct; n-gram vocabularies fitted on each half side's own rows).
- D2 (report only, never passed): r_norm to Instruct at base, SFT, DPO, Instruct, the transition differences in
  r_norm and raw r, and the question-only-halves sensitivity.
- Reported: stage-to-stage change and CKA, the RL steps, F2 and F4 (with F1 on the F4 rows), E3 text, states 9
  and 23, role decoding at base and Instruct with text baselines, related-topic pairs, components for 70% of
  the role-mean variance, the axis (only with a passed pod A D3 file), and the G5 synthetic cases.
- Not available here: the identifiable subset (no approved judge panel).

  python analysis.py --frames frames_full --out out --results results_full.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from manifold_persona import rows as R

import checkpoints as C
import frames as F
import maps as M

HERE = Path(__file__).resolve().parent
STATE, OTHER_STATES = 17, (9, 23)
D1_FLOOR, D1_ALPHA = 0.5, 0.01
F3_MAX_REL, F3_CHANCE_TOL = 1e-2, 0.005
STAGES = ["base", "sft", "dpo", "instruct"]
RL = [f"rl_{s}" for s in (60, 120, 180, 240, 300, 360)]
PAIRS = [("chef", "bartender"), ("detective", "criminal"), ("counselor", "consultant")]
TEXT = {"word": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True),
        "char": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000, sublinear_tf=True)}
EMBED_MODEL = ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130")
ANCHORS = ("text_word", "text_char", "text_embed", "text_state0")
N_SPLITS, SPLIT_SEED = 50, 20261029


class AnalysisError(Exception):
    pass


# ---- inputs

def check_sums(d):
    for line in (d / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        if R.file_sha256(d / name) != h:
            raise AnalysisError(f"{d / name} does not match SHA256SUMS.txt")


class Run:
    """Frames, per-checkpoint states and row facts for one finished pod C job."""

    def __init__(self, frames_dir, out_dir, checkpoints):
        check_sums(frames_dir)
        self.info, self.z, self.seqs = F.load(frames_dir)
        self.out, self.cks, self.layers = out_dir, list(checkpoints), {}
        for ck in self.cks:
            check_sums(out_dir / ck)
            for p in (out_dir / ck).glob("*.*.json"):
                self.layers[(ck, p.name[:-5])] = json.loads(p.read_text())

    def meta(self, src):
        s = self.info["sources"][src]
        roles, qs = s["roles"], s["questions"]
        ri, qi = {r: k for k, r in enumerate(roles)}, {q: k for k, q in enumerate(qs)}
        k = [json.loads(x) for x in s["keys"]]
        c = np.array([ri.get(x[0], -1) for x in k])
        q = np.array([qi[x[3]] for x in k])
        w = np.array([x[2] for x in k])
        role_all = {r: n for n, r in enumerate(dict.fromkeys(x[0] for x in k))}
        pid = np.array([(role_all[x[0]] * len(qs) + qi[x[3]]) * 5 + x[2] for x in k])
        return dict(roles=roles, questions=qs, c=c, q=q, w=w, pid=pid, role=[x[0] for x in k], C=len(roles), Q=len(qs))

    def ok(self, src, frame, span):
        sp = self.z[f"{src}.{frame}.span.{span}"]
        return sp[:, 1] > sp[:, 0]

    def states(self, ck, src, frame, span, state):
        rec = self.layers[(ck, f"{src}.{frame}")]
        lay = rec["layers"][span]
        if state not in lay:
            raise AnalysisError(f"{ck} {src}.{frame}.{span} has no hidden state {state}")
        arr = np.load(self.out / ck / f"{src}.{frame}.{span}.npy", mmap_mode="r")
        return np.asarray(arr[:, lay.index(state)], np.float32)

    def texts(self, src, tok):
        ids, off = self.z[f"{src}.F2.ids"], self.z[f"{src}.F2.offsets"]
        return [tok.decode(ids[off[j] + 1:off[j + 1]]) if off[j + 1] > off[j] else "" for j in range(len(off) - 1)]


def side_from_rows(feat, m, rows_ok, qmask, wmask, support=None):
    """One half side from row features. feat(rows) gives the features of those rows (dense or sparse).
    Replicates are averaged inside each prompt first; questions on the side (support "common": only questions
    where every role has a prompt)."""
    sel = np.where(rows_ok & (m["c"] >= 0) & qmask[m["q"]] & wmask[m["w"]])[0]
    X = feat(sel)
    pids, first, inv = np.unique(m["pid"][sel], return_index=True, return_inverse=True)
    A = sparse.csr_matrix((1.0 / np.bincount(inv)[inv], (inv, np.arange(len(sel)))), shape=(len(pids), len(sel)))
    P = A @ X
    pc, pq = m["c"][sel][first], m["q"][sel][first]
    nq = np.zeros((m["C"], m["Q"]))
    np.add.at(nq, (pc, pq), 1)
    qs = np.where(qmask)[0]
    if support == "common":
        qs = qs[(nq[:, qs] > 0).all(0)]
    col = {q: k for k, q in enumerate(qs)}
    keep = np.array([q in col for q in pq], bool)
    B = sparse.csr_matrix((np.ones(keep.sum()), (pc[keep] * len(qs) + np.array([col[q] for q in pq[keep]], int),
                                                  np.where(keep)[0])), shape=(m["C"] * len(qs), len(pids)))
    T = B @ P
    return M.Side.from_sums(qs, T, nq[:, qs])


def sides_for(feat_for_side, m, rows_ok, halves, support=None):
    """feat_for_side(sel) may fit on the side's own rows (text anchors)."""
    return [tuple(side_from_rows(feat_for_side, m, rows_ok, q, w, support) for q, w in h) for h in halves]


def finite_rows(X, ok, what):
    if not np.isfinite(X[ok]).all():
        raise AnalysisError(f"{what}: a usable row has non-finite states")
    return X


def tfidf_feat(texts, params):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return lambda sel: TfidfVectorizer(**params).fit_transform([texts[j] for j in sel])


def embed(texts, cache):
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL[0], revision=EMBED_MODEL[1])
    E = model.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
    np.save(cache, E)
    return E


# ---- panels and decisions

def panel_for(run, src, frame, span, state, cks, halves, draws, comparisons, mask=None, support=None, anchors=None):
    m = run.meta(src)
    ok = run.ok(src, frame, span) & (mask if mask is not None else True)
    sides = {}
    for ck in cks:
        X = finite_rows(run.states(ck, src, frame, span, state), ok & (m["c"] >= 0), f"{ck} {src}.{frame}")
        sides[ck] = sides_for(lambda sel, X=X: X[sel], m, ok, halves, support)
    for name, feat in (anchors or {}).items():
        sides[name] = sides_for(feat, m, ok, halves, support)
    stats = M.panel(sides, *draws, comparisons, strict=support == "common")
    return sides, stats


def comparisons_for(names, ref="instruct", extra=()):
    comps = [(n, n) for n in names] + [(n, ref) for n in names if n != ref] + list(extra)
    return list(dict.fromkeys(comps))


def d1(stats, sides, n_q, n_perm, anchors):
    r = stats[("base", "instruct")]["r"]
    nul = M.null(sides["base"], sides["instruct"], n_q, n_perm=n_perm)
    s, p = M.summary(r), M.p_value(r[0], nul)
    out = dict(r=s, p=p, n_perm=n_perm, rule="lower bound >= 0.5 and one-map null p < 0.01",
               passed=bool(s["lo"] is not None and s["lo"] >= D1_FLOOR and p < D1_ALPHA))
    if anchors:
        with np.errstate(all="ignore"):  # an anchor whose map is undefined in a draw is skipped in that draw
            best = np.nanmax([stats[(a, "instruct")]["r"] for a in anchors], axis=0)
        diff = M.summary(r - best)
        out["text_anchors"] = {a: M.summary(stats[(a, "instruct")]["r"]) for a in anchors}
        out["minus_best_anchor"] = diff
        out["reading"] = ("base exceeds the strongest tested text-anchor agreement" if diff["lo"] is not None and diff["lo"] > 0
                          else "no improvement over the tested text anchors was established")
    return out


def d2(stats, stages=STAGES):
    """Report only: r_norm to Instruct per stage and the paired transition differences. No pass is given."""
    per, vals = {}, {}
    for c in stages:
        per[c], vals[c] = M.r_norm(stats, c, "instruct")
    trans = {}
    for a, b in zip(stages, stages[1:]):
        t = dict(raw_r_difference=M.summary(stats[(b, "instruct")]["r"] - stats[(a, "instruct")]["r"]))
        if vals[a] is not None and vals[b] is not None:
            t["r_norm_difference"] = M.summary(vals[b] - vals[a])
        else:
            t["r_norm_difference"] = "unavailable: a reliability floor failed"
        trans[f"{a}->{b}"] = t
    return dict(note="report only; no transition is passed", per_stage=per, transitions=trans)


def stage_change(stats, chain):
    out = {}
    for a, b in zip(chain, chain[1:]):
        norm, vals = M.r_norm(stats, a, b)
        e = dict(r=M.summary(stats[(a, b)]["r"]), r_norm=norm, cka=M.summary(stats[(a, b)]["cka"]))
        if vals is not None:
            e["change"] = M.summary(1 - vals)
        out[f"{a}->{b}"] = e
    return out


# ---- decoding

def prompt_means(m, rows_ok):
    sel = np.where(rows_ok & (m["c"] >= 0))[0]
    pids, first, inv = np.unique(m["pid"][sel], return_index=True, return_inverse=True)
    A = sparse.csr_matrix((1.0 / np.bincount(inv)[inv], (inv, np.arange(len(sel)))), shape=(len(pids), len(sel)))
    return A, sel, dict(y=m["c"][sel][first], q=m["q"][sel][first], w=m["w"][sel][first])


def make_splits(m, n=N_SPLITS, seed=SPLIT_SEED):
    """Train: half the questions x 3 wordings; test: the other questions x the other 2 wordings (the E3 rule)."""
    rng = np.random.default_rng(seed)
    return [(rng.choice(m["Q"], m["Q"] // 2, replace=False), rng.choice(5, 3, replace=False)) for _ in range(n)]


def fit_predict(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def decode(feat_for_split, P, splits, n_classes, n_q, jobs=-1):
    """Macro accuracy over splits and the held-out (class, question) cell accuracy. feat_for_split(train_rows)
    returns prompt features (fit on training rows only for text)."""
    from joblib import Parallel, delayed

    def one(qtr, wtr):
        tr = np.isin(P["q"], qtr) & np.isin(P["w"], wtr)
        te = ~np.isin(P["q"], qtr) & ~np.isin(P["w"], wtr)
        for part in (tr, te):
            if len(np.unique(P["y"][part])) != n_classes:
                raise AnalysisError("a class has no prompt in a split")
        X = feat_for_split(tr)
        return te, fit_predict(X[tr], P["y"][tr], X[te])
    res = Parallel(n_jobs=jobs, prefer="threads")(delayed(one)(qtr, wtr) for qtr, wtr in splits)
    hit, cnt = np.zeros((n_classes, n_q)), np.zeros((n_classes, n_q))
    accs = []
    for te, pred in res:
        y = P["y"][te]
        accs.append(np.mean([np.mean(pred[y == c] == c) for c in range(n_classes)]))
        np.add.at(hit, (y, P["q"][te]), (pred == y).astype(float))
        np.add.at(cnt, (y, P["q"][te]), 1.0)
    with np.errstate(invalid="ignore"):
        return float(np.mean(accs)), hit / cnt


def draw_acc(cell, rr, W):
    """Per draw: each sampled class's mean over its weighted questions, then the mean over sampled classes."""
    ok = np.isfinite(cell)
    c = np.where(ok, cell, 0)
    num = np.einsum("bq,cq->bc", W, c)
    den = np.einsum("bq,cq->bc", W, ok.astype(float))
    with np.errstate(invalid="ignore"):
        per = num / den
    return np.take_along_axis(per, rr, 1).mean(1)


def f3_check(run, src, cks, splits, state=STATE):
    """F3 states equal across roles within a question, and K-class decoding at chance (1/K +- 0.5 points)."""
    m = run.meta(src)
    ok = run.ok(src, "F3", "neutral")
    A, sel, P = prompt_means(m, ok)
    counts = np.zeros((m["C"], m["Q"], 5))
    np.add.at(counts, (P["y"], P["q"], P["w"]), 1)
    if not (counts == counts.flat[0]).all():
        raise AnalysisError(f"{src}: the F3 grid is not balanced (an invalid control, not role leakage)")
    out = {}
    for ck in cks:
        X = finite_rows(run.states(ck, src, "F3", "neutral", state), ok & (m["c"] >= 0), f"{ck} {src}.F3")
        Xs = X[sel]
        rel = 0.0
        for q in range(m["Q"]):
            G = Xs[m["q"][sel] == q]
            mu = G.mean(0)
            rel = max(rel, float(np.linalg.norm(G - mu, axis=1).max() / np.linalg.norm(mu)))
        Xp = np.asarray(A @ Xs)
        acc, cell = decode(lambda tr, Xp=Xp: Xp, P, splits, m["C"], m["Q"])
        chance = 1 / m["C"]
        out[ck] = dict(max_relative_difference=rel, equal=rel <= F3_MAX_REL, accuracy=acc, chance=chance,
                       at_chance=abs(acc - chance) <= F3_CHANCE_TOL)
    out["pass"] = all(v["equal"] and v["at_chance"] for v in out.values() if isinstance(v, dict))
    return out


def decoding(run, src, ck, texts, emb, splits, draws, state=STATE):
    m = run.meta(src)
    ok = run.ok(src, "F1", "reply")
    X = run.states(ck, src, "F1", "reply", state)
    A, sel, P = prompt_means(m, ok)
    feats = {"states": lambda tr: np.asarray(A @ X[sel])}
    s0 = np.asarray(A @ run.states("instruct", src, "F1", "reply", 0)[sel])
    feats["text_state0"] = lambda tr: s0
    E = np.asarray(A @ emb[sel])
    feats["text_embed"] = lambda tr: E
    rows_text = [texts[j] for j in sel]
    for name, params in TEXT.items():
        def f(tr, params=params):
            from sklearn.feature_extraction.text import TfidfVectorizer
            train_rows = np.isin(np.arange(len(sel)), A[np.where(tr)[0]].indices)
            v = TfidfVectorizer(**params).fit([t for t, k in zip(rows_text, train_rows) if k])
            return np.asarray((A @ v.transform(rows_text)).todense())
        feats[f"text_{name}"] = f
    res = {k: decode(f, P, splits, m["C"], m["Q"]) for k, f in feats.items()}
    rr, W = draws
    acc = {k: draw_acc(cell, rr, W) for k, (_, cell) in res.items()}
    best = np.max([acc[k] for k in acc if k != "states"], axis=0)
    return dict(chance=1 / m["C"], accuracy={k: dict(mean_over_splits=v[0], cells=M.summary(acc[k])) for k, v in res.items()},
                gain_over_best_text=M.summary(acc["states"] - best))


def pair_decoding(run, src, cks, texts, splits, draws, state=STATE):
    """Two-way decoding per related-topic pair and checkpoint against the word n-gram baseline (questions resampled)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    m = run.meta(src)
    ok = run.ok(src, "F1", "reply")
    A, sel, P = prompt_means(m, ok)
    _, W = draws
    ident = np.tile(np.arange(2), (len(W), 1))
    out = {}
    for a, b in PAIRS:
        ia, ib = m["roles"].index(a), m["roles"].index(b)
        keep = np.isin(P["y"], [ia, ib])
        Pp = dict(y=(P["y"][keep] == ib).astype(int), q=P["q"][keep], w=P["w"][keep])
        Ak = A[np.where(keep)[0]]
        cols = np.unique(Ak.indices)  # the pair's rows within sel
        Ak = Ak[:, cols]
        rows_text = [texts[sel[k]] for k in cols]

        def text_feat(tr):
            used = np.unique(Ak[np.where(tr)[0]].indices)
            v = TfidfVectorizer(**TEXT["word"]).fit([rows_text[k] for k in used])
            return np.asarray((Ak @ v.transform(rows_text)).todense())
        _, cell_t = decode(text_feat, Pp, splits, 2, m["Q"])
        base = draw_acc(cell_t, ident, W)
        res = {}
        for ck in cks:
            Xk = np.asarray(Ak @ run.states(ck, src, "F1", "reply", state)[sel[cols]])
            _, cell = decode(lambda tr, Xk=Xk: Xk, Pp, splits, 2, m["Q"])
            g = M.summary(draw_acc(cell, ident, W) - base)
            res[ck] = dict(gain_over_word_ngrams=g, reading=("within-pair role decoding exceeds the n-gram baseline"
                                                             if g["lo"] is not None and g["lo"] > 0 else "not established"))
        out[f"{a} vs {b}"] = res
    return out


def components(run, src, ck, share=0.7, state=STATE):
    m = run.meta(src)
    ok = run.ok(src, "F1", "reply") & (m["c"] >= 0)
    X = run.states(ck, src, "F1", "reply", state)
    V = np.array([X[ok & (m["c"] == c)].mean(0) for c in range(m["C"])], np.float64)
    s = np.linalg.svd(V - V.mean(0), compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s) / s.sum(), share) + 1)


# ---- G5

def g5(n_boot=200, seed=5):
    """G5 synthetic cases at the pod A design (20 roles, 40 questions, the fixed halves)."""
    rng = np.random.default_rng(seed)
    Cn, Qn, D = 20, 40, 48
    halves = M.halves_from(F.make_halves(range(Qn), n=20))
    R, E = rng.standard_normal((Cn, D)), 4 * rng.standard_normal((Cn, D))
    z = np.zeros((Qn, D))
    a, b, ind = (M.synthetic_cells(x, z, 2.0, rng) for x in (R, R, rng.standard_normal((Cn, D))))
    O = np.linalg.qr(rng.standard_normal((D, D)))[0]
    wb = M.synthetic_cells(R, z, 2.0, rng, wording_dirs=E, k=(-3, -3, 2, 2, 2))
    cks = dict(a=a, b=b, rot=(b[0] @ O, b[1]), scl=(3 * b[0], b[1]), ind=ind, wb=wb)
    sides = {k: M.build_sides(*v, halves) for k, v in cks.items()}
    draws = M.make_draws(Cn, Qn, n=n_boot)
    st = M.panel(sides, *draws, [(k, k) for k in cks] + [("a", k) for k in cks if k != "a"])
    ps = M.p_value(st[("a", "ind")]["r"][0], M.null(sides["a"], sides["ind"], Qn, n_perm=999))
    norm = {k: M.r_norm(st, "a", k)[0] for k in ("b", "rot", "scl", "wb")}
    res = dict(identical_r_norm=norm["b"]["value"]["point"], rotated_r=st[("a", "rot")]["r"][0],
               scaled_r=st[("a", "scl")]["r"][0], identical_r=st[("a", "b")]["r"][0],
               independent_r=st[("a", "ind")]["r"][0], independent_null_p=ps,
               wording_case_recorded=dict(r_norm=norm["wb"].get("value", {}).get("point"),
                                          r_norm_minus_identical=(norm["wb"].get("value", {}).get("point", np.nan)
                                                                  - norm["b"]["value"]["point"]),
                                          raw_r=st[("a", "wb")]["r"][0]))
    res["pass"] = bool(abs(res["identical_r_norm"] - 1) < 0.1 and abs(res["rotated_r"] - res["identical_r"]) < 1e-6
                       and abs(res["scaled_r"] - res["identical_r"]) < 1e-6 and abs(res["independent_r"]) < 0.2)
    return res


# ---- main

def gates(run):
    problems = C.check(json.loads(C.MANIFEST.read_text()))
    for (ck, p), rec in run.layers.items():
        if rec["g1_rows_checked"] != rec["fed"]:
            problems.append(f"G1: {ck} {p} checked {rec['g1_rows_checked']} of {rec['fed']} fed rows")
    man = json.loads((run.out / "instruct" / "manifest.json").read_text()) if "instruct" in run.cks else {}
    if not isinstance(man.get("g3"), dict) or not man["g3"]["pass_"]:
        problems.append("G3: no passed reconstruction at Instruct")
    return problems


def analyse(run, tok, n_boot=M.N_BOOT, n_perm=M.N_PERM, pod_a_d3=None, cache=None, sources=F.SOURCES):
    res = dict(state=STATE, checkpoints=run.cks)
    if problems := gates(run):
        raise AnalysisError("; ".join(problems))
    res["g5"] = g5()
    if not res["g5"]["pass"]:
        raise AnalysisError("G5 synthetic cases failed")
    cache = cache or run.out
    for src in sources:
        m = run.meta(src)
        halves = M.halves_from(run.info["sources"][src]["halves"])
        draws = M.make_draws(m["C"], m["Q"], n=n_boot)
        splits = make_splits(m)
        r = res[src] = {}
        r["f3_check"] = f3_check(run, src, run.cks, splits)
        if not r["f3_check"]["pass"]:
            raise AnalysisError(f"{src}: F3 check failed; the frame or the row order may leak the role")
        texts = run.texts(src, tok)
        emb = embed(texts, cache / f"embed_{src}.npy")
        s0 = run.states("instruct", src, "F1", "reply", 0)
        anchors = {f"text_{k}": tfidf_feat(texts, p) for k, p in TEXT.items()}
        anchors["text_embed"] = lambda sel: emb[sel]
        anchors["text_state0"] = lambda sel: s0[sel]
        stage_cks = [c for c in STAGES if c in run.cks]
        main_cks = run.cks if src == "pod_a" else stage_cks
        extra = list(zip(STAGES, STAGES[1:])) + list(zip(RL, RL[1:])) + [(a, a) for a in ANCHORS] + [(a, "instruct") for a in ANCHORS]
        extra = [x for x in extra if all(y in main_cks or y in ANCHORS for y in x)]
        sides, st = panel_for(run, src, "F1", "reply", STATE, main_cks, halves, draws,
                              comparisons_for(main_cks, extra=extra), anchors=anchors)
        r["D1" if src == "pod_a" else "D1_report"] = d1(st, sides, m["Q"], n_perm, ANCHORS)
        r["D2"] = d2(st, stage_cks)
        r["stage_change"] = stage_change(st, stage_cks)
        if src == "pod_a":
            r["rl_steps"] = dict(to_instruct={c: M.r_norm(st, c, "instruct")[0] for c in RL if c in run.cks},
                                 adjacent=stage_change(st, [c for c in RL if c in run.cks]))
            qo = M.halves_from(run.info["sources"][src]["halves"], question_only=True)
            _, st_q = panel_for(run, src, "F1", "reply", STATE, stage_cks, qo, draws, comparisons_for(stage_cks))
            r["D2_question_only_halves"] = d2(st_q, stage_cks)
            for frame, span in (("F2", "reply"),):
                _, s2 = panel_for(run, src, frame, span, STATE, stage_cks, halves, draws, comparisons_for(stage_cks))
                r[f"{frame}"] = dict(base_instruct_r=M.summary(s2[("base", "instruct")]["r"]), D2=d2(s2, stage_cks))
            for state in OTHER_STATES:
                _, s3 = panel_for(run, src, "F1", "reply", state, stage_cks, halves, draws, comparisons_for(stage_cks))
                r[f"state_{state}"] = dict(base_instruct_r=M.summary(s3[("base", "instruct")]["r"]), D2=d2(s3, stage_cks))
            r["decoding"] = {ck: decoding(run, src, ck, texts, emb, splits, draws) for ck in ("base", "instruct")}
            r["related_pairs"] = pair_decoding(run, src, run.cks, texts, splits, draws)
            r["components_70"] = {ck: components(run, src, ck) for ck in run.cks}
            r["axis"] = "unavailable: pod A D3 has not passed" if not (pod_a_d3 and json.loads(Path(pod_a_d3).read_text()).get("pass")) \
                else "not computed in this version"
            r["identifiable_subset"] = "unavailable: no approved judge panel"
        f4 = run.info["f4_support"][src]
        if f4["available"]:
            mask = run.ok(src, "F1", "first32")
            _, s4 = panel_for(run, src, "F1", "first32", STATE, stage_cks, halves, draws, comparisons_for(stage_cks),
                              mask=mask, support="common")
            _, s1 = panel_for(run, src, "F1", "reply", STATE, stage_cks, halves, draws, comparisons_for(stage_cks),
                              mask=mask, support="common")
            r["F4"] = dict(base_instruct_r=M.summary(s4[("base", "instruct")]["r"]), D2=d2(s4, stage_cks),
                           F1_same_rows=dict(base_instruct_r=M.summary(s1[("base", "instruct")]["r"]), D2=d2(s1, stage_cks)))
        else:
            r["F4"] = dict(available=False, support=f4)
    return res


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="folder with one subfolder per checkpoint")
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--pod_a_d3", type=Path, default=None)
    ap.add_argument("--checkpoints", default=None, help="comma list (engineering canary: base,instruct); default all 10")
    ap.add_argument("--sources", default=",".join(F.SOURCES), help="comma list of reply sources; default all three")
    a = ap.parse_args()
    fix = lambda p: p if p.is_absolute() else HERE / p
    run = Run(fix(a.frames), fix(a.out), a.checkpoints.split(",") if a.checkpoints else [ck["name"] for ck in C.CHECKPOINTS])
    ck = C.CHECKPOINTS[3]
    tok = AutoTokenizer.from_pretrained(ck["repo"], revision=ck["revision"])
    res = analyse(run, tok, pod_a_d3=a.pod_a_d3, sources=a.sources.split(","))
    res["code"] = {n: R.file_sha256(HERE / n) for n in ("analysis.py", "maps.py", "frames.py")}
    fix(a.results).write_text(json.dumps(res, indent=1, default=float) + "\n")
    if "pod_a" in res:
        print(json.dumps(res["pod_a"]["D1"], indent=1, default=float))


if __name__ == "__main__":
    main()
