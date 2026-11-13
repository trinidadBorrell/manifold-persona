"""End-to-end test of analysis.py on synthetic states laid out as extract.py writes them (10 checkpoints, pod A
text only). A role map shared by every checkpoint must pass D1; D2 never gives a pass; the F3 check passes on a
balanced four-role grid with uneven reply loss and fails when F3 states carry the role."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as A  # noqa: E402
import frames as F  # noqa: E402

R = A.R
SIX = ["chef", "bartender", "detective", "criminal", "counselor", "consultant"]
TEN = SIX + ["pirate", "toddler", "robot", "stoic"]  # with few roles, some draws have under 3 distinct roles
QS = list(range(10, 18))
LAYERS = [0, 9, 17, 23]
D = 16


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    ck = A.C.CHECKPOINTS[3]
    try:
        return AutoTokenizer.from_pretrained(ck["repo"], revision=ck["revision"], local_files_only=True)
    except OSError:
        pytest.skip("OLMo-2-Instruct tokenizer is not in the local HF cache")


def write_sums(d, names):
    (d / "SHA256SUMS.txt").write_text("".join(f"{R.file_sha256(d / n)}  {n}\n" for n in names))


def make_run(root, tok, roles, f3_leak=0.0, invalid=(), seed=0):
    """Frames and per-checkpoint states for roles + default, 8 questions, 5 wordings, 2 replicates."""
    rng = np.random.default_rng(seed)
    keys = R.grid_keys(roles + ["default"], ["chat"], range(5), QS, range(2))
    n = len(keys)
    info = [json.loads(k) for k in keys]
    words = ["sea", "bread", "law", "night", "plan", "salt", "door", "rain"]
    texts = [f" The {k[0]} talks about topic {k[3]}, " + " ".join(rng.choice(words, 4)) for k in info]
    valid = np.array([j not in invalid for j in range(n)])
    fdir = root / "frames"
    fdir.mkdir(parents=True)
    eos = tok.eos_token_id
    f2 = [[eos] + tok.encode(t, add_special_tokens=False) if v else [] for t, v in zip(texts, valid)]
    span = np.array([[1, len(s)] if s else [0, 0] for s in f2], np.int32)
    first = np.array([[1, 9] if s and j % 3 else [0, 0] for j, s in enumerate(f2)], np.int32)
    arrays = {"pod_a.F2.ids": np.concatenate([np.array(s, np.int32) for s in f2 if s]),
              "pod_a.F2.offsets": np.cumsum([0] + [len(s) for s in f2]).astype(np.int64),
              "pod_a.F2.span.reply": span, "pod_a.F1.span.reply": span, "pod_a.F1.span.first32": first,
              "pod_a.F3.span.neutral": np.tile([[5, 9]], (n, 1)).astype(np.int32)}
    np.savez(fdir / "frames.npz", **arrays)
    halves = [(qa.tolist(), wa.tolist()) for qa, wa in F.make_halves(QS, n=10)]
    (fdir / "frames.json").write_text(json.dumps(dict(
        sources={"pod_a": dict(keys=keys, roles=roles, questions=QS, halves=halves)},
        f4_support={"pod_a": dict(available=True)}, frames={})))
    write_sums(fdir, ("frames.npz", "frames.json"))
    out = root / "out"
    Rm = 3 * rng.standard_normal((len(roles) + 1, D))
    qe = rng.standard_normal((len(QS), D))
    ci = {r: k for k, r in enumerate(roles + ["default"])}
    for ck in A.C.CHECKPOINTS:
        d = out / ck["name"]
        d.mkdir(parents=True)
        Rc = Rm + 0.3 * rng.standard_normal(Rm.shape)
        base = np.array([Rc[ci[k[0]]] + qe[QS.index(k[3])] for k in info])
        X = base[:, None, :] + rng.standard_normal((n, len(LAYERS), D))
        X[~valid] = np.nan
        f3 = np.array([qe[QS.index(k[3])] + f3_leak * Rm[ci[k[0]]] for k in info])[:, None, :] \
            + 1e-6 * rng.standard_normal((n, len(LAYERS), D))
        files = {"pod_a.F1.reply": X, "pod_a.F1.first32": np.where(first[:, 1:2, None] > 0, X, np.nan),
                 "pod_a.F2.reply": X + 0.1, "pod_a.F3.neutral": f3}
        for name, arr in files.items():
            np.save(d / f"{name}.npy", arr.astype(np.float16))
        for p, spans in (("pod_a.F1", ["reply", "first32"]), ("pod_a.F2", ["reply"]), ("pod_a.F3", ["neutral"])):
            (d / f"{p}.json").write_text(json.dumps(dict(layers={s: LAYERS for s in spans}, fed=n, g1_rows_checked=n)))
        (d / "manifest.json").write_text(json.dumps(dict(g3=dict(pass_=True)) if ck["name"] == "instruct" else {}))
        write_sums(d, sorted(p.name for p in d.iterdir()))
    np.save(root / "embed_pod_a.npy", rng.standard_normal((n, 8)).astype(np.float32))
    return A.Run(fdir, out, [ck["name"] for ck in A.C.CHECKPOINTS])


def test_end_to_end(tok, tmp_path):
    run = make_run(tmp_path, tok, TEN, invalid=(3, 50, 51))
    res = A.analyse(run, tok, n_boot=60, n_perm=199, cache=tmp_path, sources=["pod_a"])
    r = res["pod_a"]
    assert res["g5"]["pass"] and r["f3_check"]["pass"]
    assert r["D1"]["passed"] and r["D1"]["p"] == 1 / 200 and r["D1"]["r"]["lo"] > 0.5
    assert "reading" in r["D1"] and set(r["D1"]["text_anchors"]) == set(A.ANCHORS)
    d2_text = json.dumps(r["D2"], default=float)
    assert '"pass' not in d2_text and r["D2"]["note"].startswith("report only")
    assert set(r["D2"]["transitions"]) == {"base->sft", "sft->dpo", "dpo->instruct"}
    assert set(r["rl_steps"]["adjacent"]) == {f"rl_{a}->rl_{b}" for a, b in zip((60, 120, 180, 240, 300), (120, 180, 240, 300, 360))}
    assert r["decoding"]["base"]["accuracy"]["states"]["mean_over_splits"] > 0.5
    assert set(r["related_pairs"]) == {"chef vs bartender", "detective vs criminal", "counselor vs consultant"}
    assert all(1 <= v <= 9 for v in r["components_70"].values())
    assert "F1_same_rows" in r["F4"] and r["identifiable_subset"].startswith("unavailable")


def test_f3_four_roles_at_chance_and_leak_fails(tok, tmp_path):
    four = SIX[:4]
    invalid = tuple(range(0, 40)) + (90,)  # uneven loss of F1 replies; F3 still covers the full grid
    run = make_run(tmp_path / "a", tok, four, invalid=invalid)
    m = run.meta("pod_a")
    res = A.f3_check(run, "pod_a", ["base"], A.make_splits(m))
    assert res["pass"] and res["base"]["chance"] == 0.25
    run = make_run(tmp_path / "b", tok, four, f3_leak=1.0)
    assert not A.f3_check(run, "pod_a", ["base"], A.make_splits(m))["pass"]


def test_gates_refuse_missing_g3(tok, tmp_path):
    run = make_run(tmp_path, tok, SIX)
    (run.out / "instruct" / "manifest.json").write_text("{}")
    with pytest.raises(A.AnalysisError, match="G3"):
        A.analyse(run, tok, n_boot=10, n_perm=19, cache=tmp_path, sources=["pod_a"])


def test_canary_checkpoints_only(tok, tmp_path):
    """The engineering canary reads base and Instruct only; D2 and stage change use the checkpoints present."""
    full = make_run(tmp_path, tok, TEN)
    run = A.Run(full.out.parent / "frames", full.out, ["base", "instruct"])
    r = A.analyse(run, tok, n_boot=30, n_perm=99, cache=tmp_path, sources=["pod_a"])["pod_a"]
    assert set(r["D2"]["transitions"]) == {"base->instruct"} and set(r["stage_change"]) == {"base->instruct"}
    assert r["rl_steps"]["to_instruct"] == {} and set(r["components_70"]) == {"base", "instruct"}
