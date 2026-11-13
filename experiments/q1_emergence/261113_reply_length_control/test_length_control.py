"""CPU tests: the multi-span pooling equals pod A's rows.pool_spans; cut spans, NaN rows, the C0 check and the
support rule; the support counts on the real rows match THRESHOLDS.md. A tiny random Llama stands in for OLMo-2.

  .venv/bin/python -m pytest experiments/q1_emergence/261113_reply_length_control/test_length_control.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("length_control", HERE / "length_control.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
R = L.rows_module()
ROWS = {"instruct": HERE.parent / "261029_instruct_persona_map/run_full/rows.jsonl",
        "sft": HERE.parent / "261112_stage_own_replies/full_copies/sft/rows.jsonl",
        "dpo": HERE.parent / "261112_stage_own_replies/full_copies/dpo/rows.jsonl"}


@pytest.fixture(scope="module")
def model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=24,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval()


def _row(role, P, K, valid=True, seed=0):
    rng = np.random.default_rng(seed)
    p, k = rng.integers(1, 60, P).tolist(), rng.integers(1, 60, K).tolist()
    return dict(role=role, format="chat", wording=0, question=seed, replicate=0, prompt_ids=p, kept_ids=k,
                span=[P, P + K], valid=valid)


def test_pool_multi_equals_pod_a_pooling(model):
    rows = [_row("a", 5, 40, seed=1), _row("b", 9, 20, seed=2), _row("c", 3, 7, seed=3)]
    seqs = [r["prompt_ids"] + r["kept_ids"] for r in rows]
    sp = [L.spans_for(r) for r in rows]
    names = ["full", "first16", "first32"]
    got = L.pool_multi(model, seqs, {n: [s[n] for s in sp] for n in names}, 0, {n: L.STATES for n in names})
    for n in names:
        ref = R.pool_spans(model, seqs, [s[n] for s in sp], 0, layers=L.STATES)
        np.testing.assert_array_equal(np.isnan(got[n]), np.isnan(ref))
        np.testing.assert_allclose(np.nan_to_num(got[n]), np.nan_to_num(ref), rtol=0, atol=1e-6)


def test_cut_spans_and_short_or_invalid_rows():
    assert L.spans_for(_row("a", 5, 40)) == dict(full=(5, 45), first16=(5, 21), first32=(5, 37))
    assert L.spans_for(_row("a", 5, 20)) == dict(full=(5, 25), first16=(5, 21), first32=(5, 5))
    assert L.spans_for(_row("a", 5, 16))["first16"] == (5, 21)
    assert L.spans_for(_row("a", 5, 15))["first16"] == (5, 5)
    assert set(L.spans_for(_row("a", 5, 40, valid=False)).values()) == {(5, 5)}
    bad = _row("a", 5, 40)
    bad["span"] = [4, 45]
    with pytest.raises(ValueError):
        L.spans_for(bad)


def test_extract_core_and_check(model):
    rows = [_row("a", 4 + i % 5, 8 + 7 * i, seed=i) for i in range(10)]
    res = L.extract_core(model, 0, rows, 3, [9, 17, 23], log=lambda s: None)
    stored = R.pool_spans(model, [r["prompt_ids"] + r["kept_ids"] for r in rows], [r["span"] for r in rows], 0)
    chk = L.check(R, stored[:, [9, 17, 23]], res["full"], np.ones(10, bool), [9, 17, 23])
    assert chk["passed"]
    for i, r in enumerate(rows):
        K, P = len(r["kept_ids"]), len(r["prompt_ids"])
        for c, n in L.CUTS.items():
            if K < n:
                assert np.isnan(res[c][i]).all()
            else:
                ref = R.pool_spans(model, [r["prompt_ids"] + r["kept_ids"]], [(P, P + n)], 0, layers=L.STATES)[0]
                np.testing.assert_allclose(res[c][i], ref, atol=1e-5)


def test_check_fails_on_other_states(model):
    rows = [_row("a", 5, 30, seed=i) for i in range(8)]
    res = L.extract_core(model, 0, rows, 4, [9, 17, 23], log=lambda s: None)
    wrong = np.random.default_rng(0).normal(size=res["full"].shape)
    assert not L.check(R, wrong, res["full"], np.ones(8, bool), [9, 17, 23])["passed"]


def test_support_rule():
    rows = [_row("a", 3, 40), _row("a", 3, 10), _row("b", 3, 20), _row("b", 3, 20), _row("b", 3, 5, valid=False)]
    s = L.support(rows)
    assert s["first16"]["share"] == {"a": 0.5, "b": 2 / 3} and s["first16"]["available"]
    assert s["first32"]["share"] == {"a": 0.5, "b": 0.0} and not s["first32"]["available"]


@pytest.mark.skipif(not all(p.exists() for p in ROWS.values()), reason="local rows not present")
def test_real_support_matches_the_rules():
    got = {}
    for stage, p in ROWS.items():
        got[stage] = L.support([json.loads(line) for line in p.open()])
    assert all(got[s]["first16"]["available"] for s in got)
    assert got["instruct"]["first32"]["available"] and got["dpo"]["first32"]["available"]
    assert not got["sft"]["first32"]["available"]
    assert got["sft"]["first16"]["lowest"][0] == "toddler" and round(got["sft"]["first16"]["lowest"][1], 2) == 0.58
