"""Offline tests for extract.py: a tiny random Llama with the OLMo-2 vocabulary reads small frames on the CPU.
Pooled states match single-row pooling, do not depend on the batch size, resume cleanly, and the G1 and G3
checks refuse damaged input."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract as E  # noqa: E402
import frames as F  # noqa: E402

R = E.R


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    ck = F.P.CK["instruct"]
    try:
        return AutoTokenizer.from_pretrained(ck["repo"], revision=ck["revision"], local_files_only=True)
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


@pytest.fixture(autouse=True)
def tiny_layers(monkeypatch):
    monkeypatch.setattr(E, "FEW", (0, 2))
    monkeypatch.setattr(E, "ALL", (0, 1, 2))
    monkeypatch.setattr(E, "G3_LAYERS", (1, 2))


@pytest.fixture(scope="module")
def frames_dir(tok, tmp_path_factory):
    """Frames for 7 pod A records (one invalid, two with at least 32 ids) and their files."""
    out = tmp_path_factory.mktemp("frames")
    texts = ["Arr, matey.", "Bake it slowly.", "Arr " * 40, "", "I like the sea.", "Hm " * 35, "Yes."]
    recs = []
    for j, t in enumerate(texts):
        ids = tok.encode(t, add_special_tokens=False)
        recs.append(dict(source="pod_a", key=R.item_key(["pirate", "chef"][j % 2], "chat", j % 5, 10 + j, 0),
                         role=["pirate", "chef"][j % 2], format="chat", wording=j % 5, question=10 + j, replicate=0,
                         reply_ids=ids, valid=bool(t),
                         prompt_ids=tok.encode("<|endoftext|><|user|>\nHi\n<|assistant|>\n", add_special_tokens=False)))
    built = {"pod_a": (recs, F.build(recs, tok, F.question_bank()), None)}
    F.save(out, built, dict(inputs=dict(pod_a=dict(sha256={"rows.jsonl": "x"}))), tok)
    (out / "SHA256SUMS.txt").write_text("".join(f"{R.file_sha256(out / n)}  {n}\n" for n in ("frames.npz", "frames.json")))
    return out, recs


def run(tiny, frames_dir, out, checkpoint="base", batch_size=3):
    out.mkdir(parents=True, exist_ok=True)
    info, z = E.check_frames(frames_dir)
    return {f: E.run_frame(tiny, info, z, s, f, checkpoint, out, 1, batch_size) for s, f in E.jobs(info, checkpoint)}


def test_states_match_single_row_pooling(tiny, frames_dir, tmp_path):
    fdir, recs = frames_dir
    run(tiny, fdir, tmp_path)
    info, z, seqs = F.load(fdir)
    for f, name, layers in (("F1", "reply", E.ALL), ("F1", "first32", E.FEW), ("F2", "reply", E.ALL), ("F3", "neutral", E.FEW)):
        got = np.load(tmp_path / f"pod_a.{f}.{name}.npy").astype(np.float32)
        sp = z[f"pod_a.{f}.span.{name}"]
        for j, s in enumerate(seqs("pod_a", f)):
            if sp[j, 1] > sp[j, 0]:
                ref = R.pool_spans(tiny, [s], [tuple(sp[j])], 1, layers)[0]
                np.testing.assert_allclose(got[j], ref, rtol=2e-3, atol=2e-3)
            else:
                assert np.isnan(got[j]).all()
    assert np.isfinite(np.load(tmp_path / "pod_a.F1.first32.npy")[[2, 5]]).all()
    assert not (tmp_path / "pod_a.F0.reply.npy").exists()  # F0 runs at Instruct only


def test_batch_size_does_not_change_states(tiny, frames_dir, tmp_path):
    fdir, _ = frames_dir
    run(tiny, fdir, tmp_path / "a", batch_size=2)
    run(tiny, fdir, tmp_path / "b", batch_size=7)
    for p in (tmp_path / "a").glob("*.npy"):
        np.testing.assert_allclose(np.load(p), np.load(tmp_path / "b" / p.name), rtol=2e-3, atol=2e-3, equal_nan=True)


def test_resume_skips_finished_frames(tiny, frames_dir, tmp_path):
    fdir, _ = frames_dir
    first = run(tiny, fdir, tmp_path)
    again = run(tiny, fdir, tmp_path)
    assert all(again[f]["resumed"] and again[f]["fed"] == first[f]["fed"] for f in first)


def test_g1_refuses_changed_ids(tiny, frames_dir, tmp_path):
    fdir, _ = frames_dir
    info, z = E.check_frames(fdir)
    z = dict(z)
    z["pod_a.F2.ids"] = z["pod_a.F2.ids"].copy()
    z["pod_a.F2.ids"][3] += 1
    with pytest.raises(SystemExit, match="G1"):
        E.run_frame(tiny, info, z, "pod_a", "F2", "base", tmp_path, 1, 3)


def test_damaged_frame_file_refused(frames_dir, tmp_path):
    fdir, _ = frames_dir
    bad = tmp_path / "bad"
    shutil.copytree(fdir, bad)
    with open(bad / "frames.json", "a") as f:
        f.write(" ")
    with pytest.raises(SystemExit, match="does not match"):
        E.check_frames(bad)


def test_layers_and_float32_fallback(tmp_path):
    assert E.layers_for("base", "F1", "reply") == E.ALL and E.layers_for("rl_60", "F1", "reply") == E.FEW
    assert E.layers_for("instruct", "F1", "first32") == E.FEW
    assert E.save_atomic(tmp_path / "a.npy", np.array([1.0, np.nan])) == "float16"
    assert E.save_atomic(tmp_path / "b.npy", np.array([1e6, np.nan])) == "float32"


def test_g3_reconstruction(tiny, frames_dir, tmp_path):
    fdir, recs = frames_dir
    out = tmp_path / "instruct"
    out.mkdir()
    run(tiny, fdir, out, checkpoint="instruct")
    info, _ = E.check_frames(fdir)
    mine = np.load(out / "pod_a.F0.reply.npy").astype(np.float32)
    lay = E.layers_for("instruct", "F0", "reply")
    for scale, ok in ((1.0, True), (1.5, False)):
        pod_a = tmp_path / f"pod_a_{scale}"
        pod_a.mkdir()
        stored = np.full((len(recs), 3, 16), np.nan, np.float16)
        stored[:, list(lay)] = (mine * scale).astype(np.float16)
        np.save(pod_a / "reply_mean_acts.npy", stored)
        (pod_a / "SHA256SUMS.txt").write_text(f"x  rows.jsonl\n{R.file_sha256(pod_a / 'reply_mean_acts.npy')}  reply_mean_acts.npy\n")
        res = E.g3(out, pod_a, info)
        assert res["pass_"] is ok and res["n_invalid"] == 1
