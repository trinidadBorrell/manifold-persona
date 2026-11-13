"""CPU tests for extract_c3.py on the prepared inputs and the cached OLMo-2-7B-Instruct tokenizer (no model).

  HF_HUB_OFFLINE=1 .venv/bin/python -m pytest experiments/q1_emergence/261112_instruct_prompt_only/test_extract_c3.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("extract_c3_mod", HERE / "extract_c3.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)
INP = HERE / "inputs"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(X.MODEL, revision=X.REVISION)


@pytest.fixture(scope="module")
def prompts():
    return [json.loads(line) for line in (INP / "c3_prompts.jsonl").open()]


def test_inputs_match_their_checksums():
    for line in (INP / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        assert X.sha256(INP / name) == h


def test_full_grid(prompts):
    keys = {(p["role"], p["wording"], p["question"]) for p in prompts}
    assert len(prompts) == len(keys) == 4200
    assert len({p["role"] for p in prompts}) == 21 and "default" in {p["role"] for p in prompts}
    assert {p["format"] for p in prompts} == {"chat"}


def test_unique_prompts_rejects_differing_replicates():
    r = dict(role="a", format="chat", wording=0, question=1)
    assert len(X.unique_prompts([dict(r, prompt_ids=[1, 2]), dict(r, prompt_ids=[1, 2])])) == 1
    with pytest.raises(SystemExit):
        X.unique_prompts([dict(r, prompt_ids=[1, 2]), dict(r, prompt_ids=[1, 3])])


def test_frames_end_in_the_assistant_opening_then_the_neutral_reply(tok, prompts):
    neutral = tok.encode(X.NEUTRAL_SPEECH, add_special_tokens=False)
    assert tok.decode(neutral) == X.NEUTRAL_SPEECH
    for p in prompts[::97]:
        text = tok.decode(p["prompt_ids"])
        assert text.startswith("<|endoftext|><|system|>\n") and text.endswith("<|assistant|>\n")
        assert "<|user|>\n" in text
        full = tok.decode(p["prompt_ids"] + neutral)
        assert full.endswith("<|assistant|>\n" + X.NEUTRAL_SPEECH)


def test_check_rows_are_distinct_roles_with_stored_states():
    rows = json.loads((INP / "check_rows.json").read_text())
    states = np.load(INP / "check_states.npy")
    assert len(rows) == 8 == len({r["role"] for r in rows}) and states.shape == (8, 4096)
    assert np.isfinite(states).all()
    for r in rows:
        assert r["span"] == [len(r["prompt_ids"]), len(r["prompt_ids"]) + len(r["kept_ids"])]
