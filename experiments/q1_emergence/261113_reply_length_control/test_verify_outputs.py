"""CPU tests for verify_outputs.py on a small fake run: good outputs pass; each planted fault fails with its message.

  .venv/bin/python -m pytest experiments/q1_emergence/261113_reply_length_control/test_verify_outputs.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("verify_outputs", HERE / "verify_outputs.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
L = V.L
R = L.rows_module()
W = 8
LENS = [16, 16, 16, 32, 32, 32, 40, 40, 50, 5, 20, 20, 33, 60, 40]


def _sums(d, names):
    (d / "SHA256SUMS.txt").write_text("".join(f"{L.sha256(d / n)}  {n}\n" for n in names))


def _resum(d):
    _sums(d, sorted(V.read_sums(d / "SHA256SUMS.txt")))


@pytest.fixture
def case(tmp_path):
    rng = np.random.default_rng(0)
    rows = [dict(role="ab"[i % 2], format="chat", wording=0, question=i, replicate=0, prompt_ids=[1, 2, 3],
                 kept_ids=list(range(4, 4 + K)), span=[3, 3 + K], valid=i != len(LENS) - 1) for i, K in enumerate(LENS)]
    run, full, smoke = tmp_path / "run", tmp_path / "full", tmp_path / "smoke"
    for d in (run, full, smoke):
        d.mkdir()
    stored = rng.normal(size=(len(rows), 33, W)).astype(np.float16)
    (run / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.save(run / "reply_mean_acts.npy", stored)
    (run / "manifest.json").write_text("{}")
    _sums(run, ["rows.jsonl", "reply_mean_acts.npy", "manifest.json"])
    sup = L.support(rows)
    for c, n in L.CUTS.items():
        x = np.full((len(rows), len(L.STATES), W), np.nan, np.float16)
        for i, r in enumerate(rows):
            K = len(r["kept_ids"])
            if r["valid"] and K >= n:
                x[i] = stored[i, L.STATES] if K == n else rng.normal(size=(len(L.STATES), W))
        np.save(full / f"{c}_acts.npy", x)
        sup[c]["n_finite_rows"] = int(np.isfinite(x).all(axis=(1, 2)).sum())
    entry = L.STAGES["sft"]
    (full / "keys.json").write_text(json.dumps(dict(row_index=list(range(len(rows))), keys=[R.row_key(r) for r in rows])))
    (full / "support.json").write_text(json.dumps(sup))
    (full / "check.json").write_text("{}")
    man = dict(stage="sft", model=entry["repo"], revision=entry["revision"], dtype="bfloat16", attn_implementation="sdpa",
               cuts=L.CUTS, states=L.STATES, check_states=[9, 17, 23], n_rows=len(rows), limit=0, batch_size=32,
               check_passed=True, available={c: sup[c]["available"] for c in L.CUTS}, act_rows="x",
               inputs=dict(run=str(run), sha256=V.read_sums(run / "SHA256SUMS.txt")),
               code=dict(length_control_py=L.sha256(HERE / "length_control.py"), rows_py=L.sha256(L.ROWS_PY)),
               versions=dict(torch="2.13.0+cu129", device="A100"), seconds=1.0)
    (full / "manifest.json").write_text(json.dumps(man))
    _sums(full, sorted(V.OUT_FILES))
    (smoke / "manifest.json").write_text(json.dumps({**man, "n_rows": 64, "limit": 64, "seconds": 2.0,
                                                     "available": {c: False for c in L.CUTS}}))
    _sums(smoke, ["manifest.json"])
    return dict(run=run, full=full, smoke=smoke, rows=rows)


def _run(case):
    return V.verify_stage("sft", case["run"], case["full"], case["smoke"])


def _edit_cut(case, c, fn):
    p = case["full"] / f"{c}_acts.npy"
    x = np.load(p)
    fn(x)
    np.save(p, x)
    _resum(case["full"])


def test_good_outputs_pass(case):
    rec = _run(case)
    assert rec["passed"], rec["failures"]
    assert rec["cuts"]["first16"]["n_exact_length_rows"] == 3 and rec["cuts"]["first32"]["n_exact_length_rows"] == 3
    assert rec["n_rows_first16_equals_first32"] == 0 and rec["smoke_fingerprint_diff"] == []


def test_eligible_row_set_to_nan_fails(case):
    _edit_cut(case, "first16", lambda x: x.__setitem__(10, np.nan))
    rec = _run(case)
    assert any("first16: finite rows differ" in f for f in rec["failures"])
    assert rec["cuts"]["first16"]["n_eligible_not_finite"] == 1


def test_partly_nan_row_fails(case):
    _edit_cut(case, "first32", lambda x: x.__setitem__((6, 2, 0), np.nan))
    assert any("partly finite" in f for f in _run(case)["failures"])


def test_wrong_values_on_exact_rows_fail(case):
    _edit_cut(case, "first16", lambda x: x.__setitem__([0, 1, 2], x[[6, 7, 8]]))
    rec = _run(case)
    assert rec["failures"] == ["first16: exact-length rows do not reproduce the stored states"]


def test_same_vectors_in_both_cuts_fail(case):
    x16 = np.load(case["full"] / "first16_acts.npy")
    _edit_cut(case, "first32", lambda x: x.__setitem__([7, 8], x16[[7, 8]]))
    assert _run(case)["failures"] == ["first16 equals first32 on rows longer than 32"]


def test_changed_file_fails_checksum(case):
    p = case["full"] / "first32_acts.npy"
    x = np.load(p)
    x[6, 0, 0] += 1
    np.save(p, x)
    assert any("differ from SHA256SUMS" in f for f in _run(case)["failures"])


def test_wrong_dtype_fails(case):
    p = case["full"] / "first16_acts.npy"
    np.save(p, np.load(p).astype(np.float32))
    _resum(case["full"])
    assert any("first16: dtype float32" in f for f in _run(case)["failures"])


def test_smoke_with_other_runtime_fails(case):
    p = case["smoke"] / "manifest.json"
    m = json.loads(p.read_text())
    m["versions"]["torch"] = "2.12.0"
    p.write_text(json.dumps(m))
    _sums(case["smoke"], ["manifest.json"])
    rec = _run(case)
    assert rec["smoke_fingerprint_diff"] == ["versions"] and not rec["passed"]


def test_other_gpu_card_form_passes_and_is_recorded(case):
    p = case["smoke"] / "manifest.json"
    m = json.loads(p.read_text())
    m["versions"]["device"] = "A100 PCIe"
    p.write_text(json.dumps(m))
    _sums(case["smoke"], ["manifest.json"])
    rec = _run(case)
    assert rec["passed"] and rec["devices"] == dict(smoke="A100 PCIe", full="A100")


def test_smoke_gate():
    man = dict(check_passed=True, code=dict(a="1"), inputs=dict(sha256={"rows.jsonl": "x"}), model="m", revision="r")
    entry = dict(repo="m", revision="r")
    assert V.smoke_gate(man, dict(a="1"), {"rows.jsonl": "x", "other": "y"}, entry) == []
    assert V.smoke_gate(man, dict(a="2"), {"rows.jsonl": "x"}, entry) == ["code"]
    assert V.smoke_gate(man, dict(a="1"), {"rows.jsonl": "z"}, entry) == ["inputs"]
    assert V.smoke_gate(man, dict(a="1"), {"rows.jsonl": "x"}, dict(repo="m", revision="s")) == ["model or revision"]
    assert V.smoke_gate({**man, "check_passed": False}, dict(a="1"), {"rows.jsonl": "x"}, entry) == ["smoke check not passed"]
