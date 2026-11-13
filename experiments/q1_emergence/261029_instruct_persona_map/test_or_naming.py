"""Offline tests for or_naming.py with a fake OpenRouter: prep keeps keys out of the requests, malformed answers
get one retry, the payload passes analysis_naming's checks, and a still-malformed answer stops the payload."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import or_naming as O  # noqa: E402
import analysis_naming as AN  # noqa: E402

ROLES, QS = ["chef", "pirate"], [10, 18]
OTHERS = ("teacher", "detective", "doctor", "assistant", "chef", "pirate")  # all in the 276 list


@pytest.fixture
def work(tmp_path, monkeypatch):
    rows = [dict(role=r, format="chat", wording=w, question=q, replicate=k, valid=True,
                 reply=f"Reply of the {r} about {q}, wording {w}, copy {k}.")
            for r in ROLES + ["default"] for w in range(5) for q in QS for k in range(3)]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "q.json").write_text(json.dumps(QS))
    monkeypatch.setattr(O, "ROWS_PATH", tmp_path / "rows.jsonl")
    monkeypatch.setattr(O, "QUESTIONS_PATH", tmp_path / "q.json")
    w = tmp_path / "work"
    O.cmd_prep(type("A", (), dict(work=str(w), replicates="0,1,2"))())
    return w, rows


def answer(names):
    return {"body": {"choices": [{"message": {"content": json.dumps({"names": names})}}], "usage": {}}}


def fake_results(w, bad=()):
    """Every judge names the true role first; ids in `bad` get a malformed answer."""
    items = {it["id"]: it for it in map(json.loads, (w / "items.jsonl").read_text().splitlines())}
    for j in O.JUDGES:
        d = w / O.slug(j)
        res = []
        for line in (d / "requests.jsonl").read_text().splitlines():
            cid = json.loads(line)["custom_id"]
            role = json.loads(items[cid.removeprefix("rep_")]["key"])[0]
            names = [role] + [n for n in OTHERS if n != role][:4]
            res.append(dict(custom_id=cid, response=answer(["chef"] * 5 if cid in bad else names)))
        (d / "results.json").write_text(json.dumps(dict(parts=[], results=res)))
    return items


def test_prep_hides_keys_and_adds_blind_repeats(work):
    w, rows = work
    m = json.loads((w / "manifest.json").read_text())
    assert m["n_items"] == len(rows) and len(m["repeat_ids"]) == 5  # ceil(5% of 90)
    reqs = (w / O.slug("GPT-6.1-Terra") / "requests.jsonl").read_text()
    keys = [json.loads(l)["key"] for l in (w / "items.jsonl").read_text().splitlines()]
    assert '"key"' not in reqs and not any(k in reqs for k in keys)  # no item key reaches the judge
    first = json.loads(reqs.splitlines()[0])["body"]
    assert first["model"] == "openai/gpt-6.1-terra" and first["reasoning"] == {"effort": "high"}
    assert "[REPLY START]\nReply of the" in first["messages"][0]["content"]
    with pytest.raises(SystemExit, match="outside the repo"):
        O.work_dir(HERE / "naming_work")


def test_parse_rules():
    assert O.parse('{"names": ["Chef", "pirate", "teacher", "detective", "doctor"]}')[0] == "chef"
    assert O.parse('```json\n{"names": ["chef", "pirate", "teacher", "detective", "doctor"]}\n```') is not None
    for bad in ('{"names": ["chef", "chef", "teacher", "detective", "doctor"]}', '{"names": ["chef"]}',
                '{"names": ["chef", "pirate", "teacher", "detective", "not_a_role"]}', "chef, pirate", None):
        assert O.parse(bad) is None


def test_retry_then_payload_scores(work, monkeypatch):
    w, rows = work
    items = fake_results(w, bad={sorted(json.loads(l)["id"] for l in (w / "items.jsonl").read_text().splitlines())[0]})
    calls = []

    def fake_call(method, path, payload=None, timeout=300):
        calls.append(path)
        reply = payload["messages"][0]["content"].split("[REPLY START]\n")[1].split("\n[REPLY END]")[0]
        role = next(r for r in ROLES + ["default"] if f"of the {r} " in reply)
        return 200, answer([role] + [n for n in OTHERS if n != role][:4])
    monkeypatch.setattr(O.orj, "call", fake_call)
    O.cmd_retry(type("A", (), dict(work=str(w), allow_many=False))())
    assert calls == ["/chat/completions"] * len(O.JUDGES)  # one retry per judge, for the one malformed answer
    O.cmd_payload(type("A", (), dict(work=str(w), approval="test approval"))())
    payload = AN.load_payload(w / "payload.json")
    res = AN.score_naming(rows, ROLES, QS, payload, n_boot=50, n_perm=99)
    assert res["available"], res["unavailable_reasons"]
    assert all(j["macro_hit5"]["point"] == 1.0 for j in res["judges"].values())
    assert not res["confirmatory_eligible"]  # a reduced panel never gives a primary pass


def test_still_malformed_stops_payload(work, monkeypatch):
    w, _ = work
    fake_results(w, bad={json.loads((w / "items.jsonl").read_text().splitlines()[0])["id"]})
    monkeypatch.setattr(O.orj, "call", lambda *a, **k: (200, answer(["chef"] * 5)))
    O.cmd_retry(type("A", (), dict(work=str(w), allow_many=False))())
    with pytest.raises(SystemExit, match="still malformed"):
        O.cmd_payload(type("A", (), dict(work=str(w), approval="x"))())
    assert not (w / "payload.json").exists()


def test_changed_prompt_is_refused(work, monkeypatch, tmp_path):
    w, _ = work
    other = tmp_path / "prompt.txt"
    other.write_text(O.TEMPLATE_PATH.read_text() + "\nExtra rule.")
    monkeypatch.setattr(O, "TEMPLATE_PATH", other)
    with pytest.raises(SystemExit, match="changed since prep"):
        O.load(w)
