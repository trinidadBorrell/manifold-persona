"""Offline tests for manifold_persona.rows (DESIGN.md, "Shared row schema and token rules").

Token rules use the cached OLMo-2 tokenizer (skipped when it is not cached). Sampling and pooling use a
tiny random Llama, so no model download is needed.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from manifold_persona import rows as R

REPO = Path(__file__).resolve().parents[1]
BASE = ("allenai/OLMo-2-1124-7B", "7df9a82518afdecae4e8c026b27adccc8c1f0032")
BASE_MARKERS = ("\nInterviewer:", "\nAbout the guest:")
CHAT_MARKERS = ("<|user|>", "<|system|>", "<|assistant|>")
QUESTIONS = REPO / "experiments/q1_emergence/261008_base_persona_map/questions_40.json"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1], local_files_only=True)
    except OSError:
        pytest.skip("OLMo-2 tokenizer is not in the local HF cache")


def enc(tok, s):
    return tok(s, add_special_tokens=False).input_ids


def cut(tok, raw, markers=BASE_MARKERS, stop_ids=()):
    return R.cut_reply(tok, raw, tok.eos_token_id, tok.pad_token_id, markers, stop_ids)


# ---- token rules

def test_eos_only(tok):
    kept, why = cut(tok, [tok.eos_token_id])
    assert (kept, why) == ([], "eos") and not R.is_valid(tok, kept)


def test_eos_after_text_is_not_kept(tok):
    kept, why = cut(tok, enc(tok, "Arr, matey.") + [tok.eos_token_id] + enc(tok, " more"))
    assert (kept, why) == (enc(tok, "Arr, matey."), "eos")


def test_first_token_is_a_marker(tok):
    kept, why = cut(tok, enc(tok, "<|user|>\nHello") + [tok.eos_token_id], markers=CHAT_MARKERS)
    assert (kept, why) == ([], "marker") and not R.is_valid(tok, kept)


def test_trailing_pads(tok):
    text = enc(tok, "Hello there.")
    kept, why = cut(tok, text + [tok.pad_token_id] * 3)
    assert (kept, why) == (text, "pad")


def test_length_limit(tok):
    raw = enc(tok, "word " * 40)
    assert cut(tok, raw) == (raw, "limit")


def test_marker_split_across_tokens(tok):
    raw = enc(tok, "ok.\nInterviewer: next one")
    p = tok.decode(raw).find("\nInterviewer:")
    ends = {len(tok.decode(raw[:k + 1])) for k in range(len(raw))}
    assert p not in ends, "no token straddles the marker start, so this case tests nothing"
    kept, why = cut(tok, raw)
    assert why == "marker" and tok.decode(kept) == "ok"


def test_marker_after_eos_is_ignored(tok):
    kept, why = cut(tok, enc(tok, "Hi") + [tok.eos_token_id] + enc(tok, "\nInterviewer: x"))
    assert (tok.decode(kept), why) == ("Hi", "eos")


def test_extra_stop_id(tok):
    stop = enc(tok, " STOP")[0]
    kept, why = cut(tok, enc(tok, "Yes") + [stop] + enc(tok, " no"), stop_ids={stop})
    assert (tok.decode(kept), why) == ("Yes", "stop_id")


def test_whitespace_only_is_invalid(tok):
    kept, why = cut(tok, enc(tok, "  \n ") + [tok.eos_token_id])
    assert kept and why == "eos" and not R.is_valid(tok, kept)


# ---- rows

def good_row(tok):
    key = R.item_key("pirate", "interview", 2, 10, 1)
    prompt = [tok.eos_token_id] + enc(tok, "Interviewer: Hi?\nGuest:")
    raw = enc(tok, " Arr.") + [tok.eos_token_id, tok.pad_token_id]
    return R.make_row(tok, "run_x", key, prompt, raw, tok.eos_token_id, tok.pad_token_id, BASE_MARKERS)


def test_good_row_passes(tok):
    row = good_row(tok)
    R.check_row(row, {tok.eos_token_id, tok.pad_token_id}, tok, BASE_MARKERS)
    assert row["valid"] and row["span"] == [len(row["prompt_ids"]), len(row["prompt_ids"]) + 2]


@pytest.mark.parametrize("tamper", [
    lambda r, t: r["kept_ids"].append(t.eos_token_id),
    lambda r, t: r.__setitem__("span", [0, 2]),
    lambda r, t: r.__setitem__("item_seed", r["item_seed"] + 1),
    lambda r, t: r.__setitem__("valid", False),
    lambda r, t: r.__setitem__("stop_reason", "done"),
    lambda r, t: r.__setitem__("extra", 1),
    lambda r, t: r.__setitem__("reply", "Yo."),
])
def test_broken_row_raises(tok, tamper):
    row = good_row(tok)
    tamper(row, tok)
    with pytest.raises(ValueError):
        R.check_row(row, {tok.eos_token_id, tok.pad_token_id}, tok, BASE_MARKERS)


def test_empty_reply_marked_valid_raises(tok):
    key = R.item_key("pirate", "interview", 0, 10, 0)
    row = R.make_row(tok, "run_x", key, [1, 2], [tok.eos_token_id], tok.eos_token_id, tok.pad_token_id)
    assert not row["valid"]
    row["valid"] = True
    with pytest.raises(ValueError):
        R.check_row(row, {tok.eos_token_id, tok.pad_token_id})


# ---- keys, seeds, grids

def test_seed_is_pinned_and_strict():
    assert R.item_seed(R.item_key("pirate", "chat", 0, 10, 0)) == 7007743054200028646
    with pytest.raises(TypeError):
        R.item_key("pirate", "chat", "0", 10, 0)
    with pytest.raises(TypeError):
        R.item_key("pirate", "chat", True, 10, 0)
    assert R.item_seed(R.item_key("pirate", "chat", 0, 10, 1)) != R.item_seed(R.item_key("pirate", "chat", 0, 10, 0))


def test_questions_are_frozen():
    qs = R.load_questions(QUESTIONS)
    assert len(qs) == 40 and qs[0] == 10 and qs[-1] == 227


@pytest.mark.parametrize("roles, formats, reps, n_q, size", [
    (4 + 1, 1, 5, 40, 5_000),    # pod A engineering canary: 4 roles + default
    (20 + 1, 1, 5, 40, 21_000),  # pod A full run: 20 roles + default
    (10, 4, 3, 40, 24_000),      # pod B full run
    (4, 4, 2, 10, 1_600),        # pod B canary
])
def test_grid_sizes_and_unique_seeds(roles, formats, reps, n_q, size):
    keys, h = R.freeze_grid(R.grid_keys([f"r{i}" for i in range(roles)], [f"f{i}" for i in range(formats)],
                                        range(5), range(n_q), range(reps)))
    assert len(keys) == size and len({R.item_seed(k) for k in keys}) == size and len(h) == 64


def test_grid_rejects_duplicates_and_changes_hash():
    a = R.grid_keys(["pirate", "default"], ["chat"], range(5), range(40), range(5))
    with pytest.raises(ValueError):
        R.freeze_grid(a + a[:1])
    b = R.grid_keys(["pirate", "default"], ["chat"], range(5), range(39), range(5))
    assert R.freeze_grid(a)[1] != R.freeze_grid(b)[1]


def test_check_rows_splits_valid_invalid_absent():
    keys, _ = R.freeze_grid(R.grid_keys(["a", "b"], ["chat"], [0], [1, 2], [0]))
    rows = [dict(zip(R.KEY_FIELDS, ("a", "chat", 0, 1, 0)), valid=True),
            dict(zip(R.KEY_FIELDS, ("a", "chat", 0, 2, 0)), valid=False),
            dict(zip(R.KEY_FIELDS, ("b", "chat", 0, 1, 0)), valid=True)]
    got = R.check_rows(rows, keys)
    assert got["valid"] == [keys[0], keys[2]] and got["invalid"] == [keys[1]] and got["absent"] == [keys[3]]
    with pytest.raises(ValueError):
        R.check_rows(rows + rows[:1], keys)
    with pytest.raises(ValueError):
        R.check_rows(rows + [dict(zip(R.KEY_FIELDS, ("c", "chat", 0, 1, 0)), valid=True)], keys)


# ---- sampling and pooling (tiny random Llama)

EOS, PAD = 1, 0


@pytest.fixture(scope="module")
def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                      bos_token_id=2, eos_token_id=EOS, pad_token_id=PAD)
    return LlamaForCausalLM(cfg).eval()


PROMPTS = [[2, 5, 9], [2, 7, 7, 8, 30, 31, 12], [2, 40, 41, 42, 43], [2, 3, 4, 5, 6, 7, 8, 9, 10]]
SEEDS = [R.item_seed(R.item_key("r", "chat", w, 10, 0)) for w in range(4)]


def gen(model, prompts, seeds):
    return R.generate_batch(model, prompts, seeds, EOS, PAD, temperature=0.5, repetition_penalty=1.1,
                            max_new_tokens=24)


def upto_stop(g):
    return g[:g.index(EOS) + 1] if EOS in g else g


def test_reply_does_not_depend_on_batch(tiny):
    together = [upto_stop(g) for g in gen(tiny, PROMPTS, SEEDS)]
    alone = [upto_stop(gen(tiny, [p], [s])[0]) for p, s in zip(PROMPTS, SEEDS)]
    reordered = []  # batches of 2, each in reverse order
    for k in (0, 2):
        out = gen(tiny, PROMPTS[k:k + 2][::-1], SEEDS[k:k + 2][::-1])
        reordered += [upto_stop(g) for g in out[::-1]]
    assert together == alone == reordered


def test_same_seed_same_reply_new_seed_new_reply(tiny):
    a, b = gen(tiny, PROMPTS, SEEDS), gen(tiny, PROMPTS, SEEDS)
    c = gen(tiny, PROMPTS, [s + 1 for s in SEEDS])
    assert a == b and a != c


def test_draws_follow_softmax_with_temperature():
    torch.manual_seed(1)
    scores, ids = torch.randn(1, 8), torch.tensor([[3]])
    s = R.SeededSampler([123], temperature=0.5, repetition_penalty=1.0, pad_id=PAD)
    counts = np.bincount([int(s(ids, scores).argmax()) for _ in range(20_000)], minlength=8) / 20_000
    assert np.abs(counts - torch.softmax(scores[0] / 0.5, -1).numpy()).max() < 0.015


def test_penalty_matches_transformers_except_pad():
    from transformers import RepetitionPenaltyLogitsProcessor
    torch.manual_seed(2)
    scores = torch.randn(2, 12)
    ids = torch.tensor([[PAD, PAD, 4, 7, 7], [3, 4, 5, 9, 11]])
    ours = R.SeededSampler([1, 2], temperature=0.5, repetition_penalty=1.1, pad_id=PAD).scores(ids, scores)
    ref = RepetitionPenaltyLogitsProcessor(1.1)(ids, scores.clone()) / 0.5
    ref[:, PAD] = scores[:, PAD] / 0.5
    assert torch.allclose(ours, ref)


def test_pool_spans_batched_equals_single_and_empty_is_nan(tiny):
    seqs = [[2, 5, 9, 11, 12], [2, 7, 7, 8, 30, 31, 12, 13], [2, 40, 41]]
    spans = [(3, 5), (4, 8), (3, 3)]
    batch = R.pool_spans(tiny, seqs, spans, PAD)
    single = np.concatenate([R.pool_spans(tiny, [x], [sp], PAD) for x, sp in zip(seqs, spans)])
    assert batch.shape == (3, tiny.config.num_hidden_layers + 1, 32)
    assert np.allclose(batch[:2], single[:2], atol=1e-5) and np.isnan(batch[2]).all()
    with pytest.raises(ValueError):
        R.pool_spans(tiny, seqs, [(3, 5), (4, 9), (3, 3)], PAD)
