"""Offline tests for context_extract.py on a tiny finished run (real OLMo-2 base tokenizer, tiny random Llama)."""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


TG = load("base_role_vs_format_test_gen_extract", HERE / "test_podb_gen_extract.py")
C = load("base_role_vs_format_context_extract", HERE / "context_extract.py")
G, R = C.G, C.R
tok, tiny = TG.tok, TG.tiny


@pytest.fixture(scope="module")
def run_dir(tok, tiny, tmp_path_factory):
    return TG.finished_run(tok, tiny, tmp_path_factory.mktemp("run"))


def test_interview_neutral_is_e3s():
    src = (HERE.parent / "261008_base_persona_map/context_extract.py").read_text()
    e3 = next(ast.literal_eval(n.value) for n in ast.parse(src).body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "NEUTRAL")
    assert C.NEUTRAL["interview"] == C.NEUTRAL["script"] == e3
    assert all(C.NEUTRAL_SPEECH in t for t in C.NEUTRAL.values()) and set(C.NEUTRAL) == set(G.FORMATS)


def direct_mean(model, ids, a, b, layers):
    with torch.no_grad():
        hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
    return np.stack([hs[L][0, a:b].mean(0).numpy() for L in layers])


def test_c2_and_c3_are_means_over_the_right_positions(run_dir, tok, tiny):
    run = G.load_run(run_dir)
    rows = run["rows"]
    res = C.extract(rows, tiny, tok, 3)
    layers = res["layers"]
    assert layers == G.depth_layers(4) and res["c2"].shape == (len(rows), 3, 16)
    bank = TG.bank()
    for i, r in enumerate(rows):
        if not r["valid"]:
            assert np.isnan(res["c2"][i]).all()
            continue
        p = G.prompt_ids(tok, G.prompt_text(r["format"], None, bank[r["question"]]))
        assert len(p) < len(r["prompt_ids"]) and tok.decode(p).count("Alex is") == 0
        want = direct_mean(tiny, p + r["kept_ids"], len(p), len(p) + len(r["kept_ids"]), layers)
        assert np.allclose(res["c2"][i], want, atol=2e-3)
    prompts = {tuple(p) for p in res["prompts"]}
    assert len(prompts) == len(res["prompts"]) == len({G.prompt_of(R.row_key(r)) for r in rows})
    for j, p in enumerate(res["prompts"]):
        r = next(r for r in rows if list(G.prompt_of(R.row_key(r))) == p)
        n = res["neutral_ids"][p[1]]
        want = direct_mean(tiny, r["prompt_ids"] + n, len(r["prompt_ids"]), len(r["prompt_ids"]) + len(n), layers)
        assert np.allclose(res["c3"][j], want, atol=2e-3)


def test_c3_does_not_depend_on_the_reply(run_dir, tok, tiny):
    rows = G.load_run(run_dir)["rows"]
    changed = [dict(r, kept_ids=r["kept_ids"][:1]) for r in rows]
    a, b = C.extract(rows, tiny, tok, 4), C.extract(changed, tiny, tok, 2)
    assert np.allclose(a["c3"], b["c3"], atol=1e-5) and a["prompts"] == b["prompts"]


def test_a_changed_prompt_is_refused(run_dir, tok, tiny):
    rows = G.load_run(run_dir)["rows"]
    bad = [dict(r) for r in rows]
    bad[0]["prompt_ids"] = bad[0]["prompt_ids"][:-1]
    with pytest.raises(SystemExit):
        C.extract(bad, tiny, tok, 4)


def test_save_keeps_float32_when_values_exceed_float16(tmp_path):
    big = np.full((2, 3, 4), 1e6, dtype=np.float32)
    big[0] = np.nan
    C.save(tmp_path, dict(c2=big, c3=np.ones((1, 3, 4), np.float32), prompts=[["pirate", "story", 0, 10]]))
    assert np.load(tmp_path / "ctx_c2.npy").dtype == np.float32 and np.load(tmp_path / "ctx_c3.npy").dtype == np.float16
    assert json.loads((tmp_path / "c3_prompts.json").read_text()) == [["pirate", "story", 0, 10]]


# ---- code review 29/10 (review/codex_last_code_261029.txt)

def test_unusable_context_states_are_refused():
    """Codex 6: infinite C2 or NaN C3 must not be saved."""
    rows = [dict(valid=True), dict(valid=False)]
    good = dict(c2=np.array([np.ones((3, 4)), np.full((3, 4), np.nan)]), c3=np.ones((1, 3, 4)),
                layers=(1, 2, 3), prompts=[["pirate", "story", 0, 10]])
    C.check(good, rows, 4)
    for bad in (dict(good, c2=np.full((2, 3, 4), np.inf)), dict(good, c3=np.full((1, 3, 4), np.nan)),
                dict(good, c2=np.ones((2, 3, 4))), dict(good, c3=np.ones((1, 0, 4)))):
        with pytest.raises(SystemExit):
            C.check(bad, rows, 4)


def test_invalid_rows_get_nan_c2(tok, tiny, tmp_path, monkeypatch):
    """Codex 14: the fixture run had no invalid row, so the NaN branch of C2 never ran."""
    real = G.R.generate_batch

    def some_empty(model, prompts, seeds, eos, *a, **k):
        out = real(model, prompts, seeds, eos, *a, **k)
        return [[eos] if s % 3 == 0 else g for s, g in zip(seeds, out)]

    monkeypatch.setattr(G.R, "generate_batch", some_empty)
    out = TG.finished_run(tok, tiny, tmp_path / "run")
    rows = G.load_run(out)["rows"]
    v = np.array([r["valid"] for r in rows])
    assert v.any() and not v.all()
    res = C.extract(rows, tiny, tok, 3)
    C.check(res, rows, 16)
    assert np.isnan(res["c2"][~v]).all() and np.isfinite(res["c2"][v]).all() and np.isfinite(res["c3"]).all()
