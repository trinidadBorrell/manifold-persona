"""Offline tests for frames.py with the real OLMo-2 tokenizer (local HF cache): frame layout, spans, F4 rows,
fixed halves and the F4 support rule."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402

P = F.P


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    ck = P.CK["instruct"]
    try:
        return AutoTokenizer.from_pretrained(ck["repo"], revision=ck["revision"], local_files_only=True)
    except OSError:
        pytest.skip("OLMo-2-Instruct tokenizer is not in the local HF cache")


def rec(tok, role, text, valid=True, q=10, w=0, prompt=True):
    ids = tok.encode(text, add_special_tokens=False) if valid else []
    r = dict(source="pod_a", key=P.R.item_key(role, "chat", w, q, 0), role=role, format="chat", wording=w,
             question=q, replicate=0, reply_ids=ids, valid=valid)
    return {**r, "prompt_ids": tok.encode("<|endoftext|><|user|>\nHi\n<|assistant|>\n", add_special_tokens=False)} if prompt else r


def test_frame_layout_and_spans(tok):
    bank = F.question_bank()
    long = "Arr " * 40
    recs = [rec(tok, "pirate", "Arr, matey."), rec(tok, "chef", " Bake it slowly."), rec(tok, "chef", long),
            rec(tok, "toddler", "", valid=False)]
    fr = F.build(recs, tok, bank)
    q = bank[10]
    f1, sp = fr["F1"]
    assert tok.decode(f1[0]) == f"<|endoftext|>{F.HEADER}\n\nInterviewer: {q}\nGuest:\nArr, matey."
    assert tok.decode(f1[1]) == f"<|endoftext|>{F.HEADER}\n\nInterviewer: {q}\nGuest: Bake it slowly."
    for f, name in (("F0", "reply"), ("F1", "reply"), ("F2", "reply")):
        seqs, spans = fr[f]
        for j in range(3):
            a, b = spans[name][j]
            assert seqs[j][a:b] == recs[j]["reply_ids"], (f, j)
        assert seqs[3] == [] and list(spans[name][3]) == [0, 0]
    first = sp["first32"]
    assert list(first[0]) == [0, 0] and first[2, 1] - first[2, 0] == 32
    assert f1[2][first[2, 0]:first[2, 1]] == recs[2]["reply_ids"][:32]
    f3, s3 = fr["F3"]
    assert f3[0] == f3[1] == f3[2] == f3[3] and tok.decode(f3[3][s3["neutral"][3, 0]:]) == F.NEUTRAL


def test_e3_records_have_no_chat_frame(tok):
    fr = F.build([rec(tok, "chef", " Yes.", prompt=False)], tok, F.question_bank())
    assert fr["F0"][0] == [[]] and fr["F1"][0][0]


def test_halves_are_fixed_and_split_both_factors():
    a, b = F.make_halves(range(40)), F.make_halves(range(40))
    assert len(a) == 50 and all((x[0] == y[0]).all() and (x[1] == y[1]).all() for x, y in zip(a, b))
    assert all(qa.sum() == 20 and wa.sum() in (2, 3) for qa, wa in a)


def grid_recs(roles, n_q, keep):
    """Records on roles x n_q questions x 5 wordings; keep(role, q, w) says whether a row is an F4 row."""
    recs, mask = [], []
    for role in roles:
        for q in range(n_q):
            for w in range(5):
                recs.append(dict(role=role, question=q, wording=w))
                mask.append(keep(role, q, w))
    return recs, np.array(mask)


def test_f4_support_rules():
    roles = ["a", "b", "c", "d"]
    halves = F.make_halves(range(18))
    recs, mask = grid_recs(roles, 18, lambda r, q, w: True)
    assert F.f4_support(recs, mask, roles, halves)["available"]
    # role d keeps 40% of its rows: unavailable by the share rule
    recs, mask = grid_recs(roles, 18, lambda r, q, w: r != "d" or w < 2)
    s = F.f4_support(recs, mask, roles, halves)
    assert not s["available"] and s["min_role_share"] == 0.4
    # every role keeps at least half its rows, but role d only on wordings 0-2: some half B (wordings 3, 4
    # and maybe one more) has no question where d has a row
    recs, mask = grid_recs(roles, 18, lambda r, q, w: r != "d" or w < 3)
    s = F.f4_support(recs, mask, roles, halves)
    assert s["min_role_share"] == 0.6 and s["halves_without_common_question"] > 0 and not s["available"]


def test_row_hashes_change_with_ids():
    h = F.row_hashes([[1, 2, 3], [1, 2, 3], [1, 2], []])
    assert (h[0] == h[1]).all() and not (h[0] == h[2]).all() and h.shape == (4, 32)
