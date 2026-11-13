"""Offline tests for replies.py on small fake runs (real OLMo-2 tokenizer from the local HF cache), plus the
real E3 run and pod A canary when they are on disk. A clean source reads; each kind of damage is refused."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replies as P  # noqa: E402

R = P.R
MARKERS = ["<|user|>", "<|system|>", "<|assistant|>"]
CANARY = HERE.parent / "261029_instruct_persona_map/run_canary"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    ck = P.CK["instruct"]
    try:
        return AutoTokenizer.from_pretrained(ck["repo"], revision=ck["revision"], local_files_only=True)
    except OSError:
        pytest.skip("OLMo-2-Instruct tokenizer is not in the local HF cache")


def write_sums(base, names, out):
    (base / out).write_text("".join(f"{R.file_sha256(base / n)}  {n}\n" for n in names))


def pod_a_run(path, tok, rows=None, **manifest):
    """A finished pod A run: 2 roles x 2 wordings x 2 questions, one invalid row."""
    keys, grid_hash = R.freeze_grid(R.grid_keys(["pirate", "chef"], ["chat"], [0, 1], [10, 18], [0]))
    if rows is None:
        eos, pad = tok.eos_token_id, tok.pad_token_id
        rows = [R.make_row(tok, "t", k, tok.encode("<|endoftext|>Hi"), [eos] if j == 3 else
                           tok.encode(f" Arr, answer {j}.") + [eos], eos, pad, MARKERS) for j, k in enumerate(keys)]
    path.mkdir()
    (path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "grid.json").write_text(json.dumps(dict(grid_hash=grid_hash, keys=keys)))
    ck = P.CK["instruct"]
    (path / "manifest.json").write_text(json.dumps({**dict(model=ck["repo"], revision=ck["revision"], grid_hash=grid_hash,
                                                          roles=["pirate", "chef"], markers=MARKERS), **manifest}))
    write_sums(path, ("rows.jsonl", "grid.json", "manifest.json"), "SHA256SUMS.txt")
    return path, rows


def test_pod_a_reads(tok, tmp_path):
    path, rows = pod_a_run(tmp_path / "a", tok)
    recs, info = P.read_pod_a(path, tok)
    assert [r["key"] for r in recs] == json.loads((path / "grid.json").read_text())["keys"]
    assert [r["valid"] for r in recs].count(False) == 1 and info["roles"] == ["pirate", "chef"]
    assert {r["key"]: r["reply_ids"] for r in recs} == {R.row_key(r): r["kept_ids"] for r in rows}


def e3_row(tok, rows):
    r = dict(cond="desc", ended_by="eos", reply_ids=rows[0]["kept_ids"])
    return {**{k: v for k, v in rows[0].items() if k != "kept_ids"}, **r}


@pytest.mark.parametrize("damage,why", [
    (lambda rows, tok: rows + [e3_row(tok, rows)], "old E3 row schema"),
    (lambda rows, tok: rows[:-1], "absent"),
    (lambda rows, tok: rows + rows[:1], "duplicate row"),
    (lambda rows, tok: [{**rows[0], "kept_ids": rows[0]["kept_ids"][:-1]}] + rows[1:], "span does not match"),
    (lambda rows, tok: [{**rows[0], "reply": "x"}] + rows[1:], "reply text"),
])
def test_pod_a_damage_refused(tok, tmp_path, damage, why):
    _, rows = pod_a_run(tmp_path / "clean", tok)
    path, _ = pod_a_run(tmp_path / "a", tok, damage(rows, tok))
    with pytest.raises(ValueError, match=why):
        P.read_pod_a(path, tok)


def test_pod_a_wrong_model_or_changed_file(tok, tmp_path):
    path, _ = pod_a_run(tmp_path / "a", tok, revision="0" * 40)
    with pytest.raises(ValueError, match="is not allenai"):
        P.read_pod_a(path, tok)
    path, _ = pod_a_run(tmp_path / "b", tok)
    with open(path / "rows.jsonl", "a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="does not match"):
        P.read_pod_a(path, tok)


def e3_run(root, tok, edit=lambda rows: rows):
    """An E3-style run: 2 pilot roles x desc, ex2 x 5 wordings x 18 shared questions, plus rows to skip."""
    bank = [json.loads(l)["question"] for l in open(P.REPO / "refs/assistant-axis/extraction_questions.jsonl")]
    qs = list(range(100, 118))
    rows = []
    for role in ("pirate", "chef", "default_ai"):
        for cond in ("desc", "ex2", "ex2_other"):
            for w in range(5):
                for q in qs + [5]:
                    ids = tok.encode(f" {role} says {w} {q}")
                    rows.append(dict(role=role, cond=cond, wording=w, question_idx=q, question=bank[q],
                                     shared_q=q != 5, ended_by="eos", raw_ids=ids + [tok.eos_token_id], raw="",
                                     reply=tok.decode(ids), reply_ids=ids, valid=True))
    run = root / "run_main"
    run.mkdir(parents=True)
    (run / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in edit(rows)))
    ck = P.CK["base"]
    (run / "manifest.json").write_text(json.dumps(dict(model=ck["repo"], revision=ck["revision"],
                                                       pilot_order=["pirate", "chef"], stops=["\nInterviewer:"])))
    write_sums(root, ("run_main/replies.jsonl", "run_main/manifest.json"), "SHA256SUMS_run.txt")
    return run


def test_e3_reads_and_skips_other_rows(tok, tmp_path):
    recs, info = P.read_e3(e3_run(tmp_path, tok), tok)
    assert len(recs) == 2 * 2 * 5 * 18 and {r["source"] for r in recs} == {"e3_desc", "e3_ex2"}
    assert {r["role"] for r in recs} == {"pirate", "chef"} and info["questions"] == list(range(100, 118))
    assert all(r["replicate"] == 0 and r["valid"] for r in recs)


@pytest.mark.parametrize("edit,why", [
    (lambda rows: rows + rows[:1], "duplicate row"),
    (lambda rows: [r for r in rows if not (r["role"] == "chef" and r["question_idx"] == 100 and r["wording"] == 0)],
     "absent"),
    (lambda rows: [{**rows[0], "question": "Other?"}] + rows[1:], "question text"),
    (lambda rows: [{**rows[0], "reply": "x"}] + rows[1:], "do not agree"),
    (lambda rows: [{**rows[0], "reply_ids": rows[0]["reply_ids"] + [100257]}] + rows[1:], "free of eos"),
    (lambda rows: [{**{k: v for k, v in rows[0].items() if k != "cond"}, "kept_ids": []}] + rows[1:], "not an E3 row"),
])
def test_e3_damage_refused(tok, tmp_path, edit, why):
    with pytest.raises(ValueError, match=why):
        P.read_e3(e3_run(tmp_path, tok, edit), tok)


def test_real_sources(tok):
    if not (P.E3_RUN / "replies.jsonl").exists() or not (CANARY / "rows.jsonl").exists():
        pytest.skip("E3 run_main or the pod A canary is not on disk")
    e3, _ = P.read_e3(P.E3_RUN, tok)
    a, _ = P.read_pod_a(CANARY, tok)
    s = P.summary(e3 + a)
    assert (s["e3_desc"]["rows"], s["e3_ex2"]["rows"], s["pod_a"]["rows"]) == (1800, 1800, 5000)
    assert sum(v["invalid"] for v in s.values()) == 0
