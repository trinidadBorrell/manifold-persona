"""Offline tests for gates.py on a tiny finished run (real OLMo-2-Instruct tokenizer, tiny random Llama).
A clean run must pass every gate; each kind of damage must fail its gate."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gates as T  # noqa: E402
import gen_extract as G  # noqa: E402

R = G.R
LAYERS = (1, 2)  # the tiny model has hidden states 0..2


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


@pytest.fixture(scope="module")
def run_dir(tok, tiny, tmp_path_factory):
    """A finished run laid out as gen_extract.main() writes it."""
    out = tmp_path_factory.mktemp("run")
    items = G.build_items(["pirate", "chef"], 1)
    items = G.render(tok, [items[k] for k in range(0, len(items), 40)])  # 10 items, 5 per role
    grid, grid_hash = R.freeze_grid(sorted(it["key"] for it in items))
    G.run(items, out, 4, tiny, tok, "tiny")
    G.finalize(out, grid)
    (out / "grid.json").write_text(json.dumps(dict(grid_hash=grid_hash, keys=grid)))
    (out / "manifest.json").write_text(json.dumps(dict(
        revision="tiny", batch_size=4, tokenizer_files={"tokenizer.json": "x"},
        chat_template=hashlib.sha256(tok.chat_template.encode()).hexdigest())))
    names = ("rows.jsonl", "reply_mean_acts.npy", "grid.json", "manifest.json")
    (out / "SHA256SUMS.txt").write_text("".join(f"{R.file_sha256(out / n)}  {n}\n" for n in names))
    return out


def test_clean_run_passes_every_gate(run_dir, tok, tiny):
    rows, grid, acts, manifest = T.load_run(run_dir)
    assert T.g1(rows, grid, acts, tok)["pass"]
    assert T.g2(rows)["pass"]
    g3 = T.g3(rows, tiny, tok, manifest["batch_size"], n=6)
    assert g3["pass"] and g3["n_rows"] >= 6 and g3["identical_one_at_a_time_reported"] == g3["n_rows"]
    g4 = T.g4(rows, acts, tiny, tok, 3, layers=LAYERS)
    assert g4["pass"] and all(c["min_cos"] > 0.9999 for c in g4["cells"].values())
    assert T.g5(manifest, tok, "tiny", {"tokenizer.json": "x"})["pass"]
    T.raw_read(rows, tok, run_dir / "raw_read.md", n=5)
    assert (run_dir / "raw_read.md").read_text().count("## [") == 5


def test_scaled_states_fail_g4(run_dir, tok, tiny):
    rows, _, acts, _ = T.load_run(run_dir)
    assert not T.g4(rows, np.asarray(acts) * 1.1, tiny, tok, 3, layers=LAYERS)["pass"]


def test_changed_reply_fails_g3(run_dir, tok, tiny):
    rows, _, _, manifest = T.load_run(run_dir)
    batch = T.run_batches(rows, manifest["batch_size"])
    for i in batch[0] + batch[1] + batch[2]:  # g3 picks from all batches; damage every one
        rows[i]["raw_ids"] = rows[i]["raw_ids"][:-1] + [rows[i]["raw_ids"][-1] + 1]
    assert not T.g3(rows, tiny, tok, manifest["batch_size"], n=6)["pass"]


def test_bad_rows_fail_g1(run_dir, tok):
    rows, grid, acts, _ = T.load_run(run_dir)
    rows[0]["kept_ids"] = rows[0]["kept_ids"] + [tok.eos_token_id]
    assert T.g1(rows, grid, acts, tok)["row_errors"] == 1
    bad = np.array(acts)
    bad[1] = np.nan
    assert not T.g1(T.load_run(run_dir)[0], grid, bad, tok)["pass"]
    g = T.g1(rows[1:], grid + [R.item_key("chef", "chat", 4, 227, 0)], acts, tok)  # two grid keys lack rows
    assert g["absent"] == 2 and not g["pass"]


def test_low_valid_rate_fails_g2():
    rows = [dict(role="pirate", valid=True)] * 95 + [dict(role="pirate", valid=False)] * 5 + \
           [dict(role="chef", valid=True)] * 94 + [dict(role="chef", valid=False)] * 6
    g = T.g2(rows)
    assert g["rates"] == {"chef": 0.94, "pirate": 0.95} and not g["pass"]


def test_snapshot_mismatch_fails_g5(run_dir, tok):
    manifest = T.load_run(run_dir)[3]
    assert not T.g5(manifest, tok, "other", {"tokenizer.json": "x"})["pass"]
    assert not T.g5(manifest, tok, "tiny", {"tokenizer.json": "y"})["pass"]


def test_damaged_file_is_refused(run_dir, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(run_dir, d)
    with open(d / "rows.jsonl", "a") as f:
        f.write(" ")
    with pytest.raises(SystemExit):
        T.load_run(d)
