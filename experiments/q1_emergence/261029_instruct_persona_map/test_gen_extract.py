"""Offline tests for gen_extract.py: the real OLMo-2-Instruct tokenizer (from the local HF cache) and a
tiny random Llama with the same vocabulary, so the whole path runs on a CPU."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gen_extract", HERE / "gen_extract.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
R = G.R


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(G.MODEL, revision=G.REVISION, local_files_only=True)
    except OSError:
        pytest.skip("OLMo-2-Instruct tokenizer is not in the local HF cache")


@pytest.fixture(scope="module")
def tiny(tok):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return LlamaForCausalLM(cfg).eval()


def test_full_grid():
    items = G.build_items(G.ROLE_SETS["full"], G.N_REPLICATES)
    keys, _ = R.freeze_grid(it["key"] for it in items)
    assert len(keys) == 21 * 5 * 40 * 5 == 21_000
    sys_by = {(json.loads(it["key"])[0], json.loads(it["key"])[2]): it["system"] for it in items}
    assert sys_by[("default", 0)] == "" and sys_by[("default", 3)] == "You are OLMo."
    assert len(G.build_items(G.ROLE_SETS["canary"], G.N_REPLICATES)) == 5 * 5 * 40 * 5


def test_rendered_prompts(tok):
    items = {it["key"]: it for it in G.render(tok, G.build_items(["default", "pirate"], 1))}
    no_sys = tok.decode(items[R.item_key("default", "chat", 0, 10, 0)]["prompt_ids"])
    pirate = tok.decode(items[R.item_key("pirate", "chat", 0, 10, 0)]["prompt_ids"])
    q = items[R.item_key("pirate", "chat", 0, 10, 0)]["question"]
    assert no_sys == f"<|endoftext|><|user|>\n{q}\n<|assistant|>\n"
    assert pirate == ("<|endoftext|><|system|>\nYou are a pirate captain who has sailed the seven seas in search "
                      f"of treasure and adventure.\n<|user|>\n{q}\n<|assistant|>\n")


def small_items(tok, n=10):
    items = G.build_items(["pirate", "default"], 2)
    return G.render(tok, [items[k] for k in range(0, len(items), len(items) // n)][:n])


def rows_of(out):
    return {R.row_key(r): r for r in map(json.loads, open(out / "rows.jsonl"))}


def test_end_to_end_batch_independent_and_resumable(tok, tiny, tmp_path):
    items = small_items(tok)
    grid, _ = R.freeze_grid(sorted(it["key"] for it in items))
    a, b = tmp_path / "a", tmp_path / "b"
    G.run(items, a, 4, tiny, tok, "a")
    sa = G.finalize(a, grid)
    G.run(items, b, 3, tiny, tok, "b")
    G.finalize(b, grid)
    ra, rb = rows_of(a), rows_of(b)
    assert sa["n_rows"] == 10 and sa["acts_shape"] == [10, 3, 16]
    assert all(ra[k]["raw_ids"] == rb[k]["raw_ids"] for k in grid), "replies changed with the batch size"
    acts = np.load(a / "reply_mean_acts.npy")
    valid = np.array([ra[k]["valid"] for k in grid])
    assert np.isfinite(acts[valid]).all() and np.isnan(acts[~valid]).all()
    for r in ra.values():
        R.check_row(r, {tok.eos_token_id, tok.pad_token_id}, tok, G.MARKERS)

    # resume: drop the last batch's rows (as after a crash), rerun, get the same rows
    c = tmp_path / "c"
    shutil.copytree(a, c)
    lines = (c / "rows.partial.jsonl").read_text().splitlines()
    (c / "rows.partial.jsonl").write_text("\n".join(lines[:-2]) + "\n" + lines[-2][:15])  # cut mid-line
    G.run(items, c, 4, tiny, tok, "a")
    G.finalize(c, grid)
    assert rows_of(c) == ra
    assert np.array_equal(np.load(c / "reply_mean_acts.npy"), acts, equal_nan=True)


def test_resume_after_crash_inside_a_batch(tok, tiny, tmp_path):
    items = small_items(tok)
    grid, _ = R.freeze_grid(sorted(it["key"] for it in items))
    a, c = tmp_path / "a", tmp_path / "c"
    G.run(items, a, 4, tiny, tok, "a")
    G.finalize(a, grid)
    shutil.copytree(a, c)
    lines = (c / "rows.partial.jsonl").read_text().splitlines()
    # batch 0 complete, 2 complete rows of batch 1, then a cut line
    (c / "rows.partial.jsonl").write_text("\n".join(lines[:6]) + "\n" + lines[6][:15])
    G.run(items, c, 4, tiny, tok, "a")
    G.finalize(c, grid)
    assert len((c / "rows.partial.jsonl").read_text().splitlines()) == 10
    assert rows_of(c) == rows_of(a)


def test_states_beyond_float16_stay_float32(tok, tiny, tmp_path, monkeypatch):
    items = small_items(tok, 4)
    grid, _ = R.freeze_grid(sorted(it["key"] for it in items))
    real, calls = R.pool_spans, []

    def big_first_batch(*a, **k):
        calls.append(1)
        return real(*a, **k) * (1e6 if len(calls) == 1 else 1)

    monkeypatch.setattr(R, "pool_spans", big_first_batch)
    G.run(items, tmp_path, 2, tiny, tok, "x")
    s = G.finalize(tmp_path, grid)
    acts = np.load(tmp_path / "reply_mean_acts.npy")
    assert s["acts_dtype"] == "float32" and np.isfinite(acts).all() and np.abs(acts).max() > 65504


def test_finalize_refuses_missing_rows(tok, tiny, tmp_path):
    items = small_items(tok, 4)
    grid, _ = R.freeze_grid(sorted(it["key"] for it in items) + [R.item_key("pirate", "chat", 4, 227, 1)])
    G.run(items, tmp_path, 4, tiny, tok, "x")
    with pytest.raises(AssertionError):
        G.finalize(tmp_path, grid)
