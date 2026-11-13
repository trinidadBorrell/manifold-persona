"""Owner transcript read of the own-reply run (THRESHOLDS v1.0 gate: the owner reads 20 transcripts per stage before
any metric). Owner choice, 13/11/2026: a local label page, one prompt per screen, the three stages side by side.

The 20 keys are the raw-read keys of `../full_copies/side_by_side_20.md`. For each key the exact model input is
rendered with the run's own code (`stage_run.load_pod_a(stage)` -> pod A's `build_items` + `render`, with each stage's
tokenizer at its pinned revision) and must equal the stored `prompt_ids` of that stage's `rows.jsonl`; the page shows
`tok.decode(prompt_ids)` with special tokens kept. Replies come from the same rows. No judge label and no metric.

Items and labels stay outside the repo, next to the 12 Nov review set:
    .venv/bin/python -I experiments/q1_emergence/261112_stage_own_replies/owner_read/read_page.py draw
    .venv/bin/python -I experiments/q1_emergence/261112_stage_own_replies/owner_read/read_page.py serve [--port 8768]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
Q = RUN.parent
REPO = Q.parents[1]
JV = Path("/Users/brendon/persona-manifolds/judge_validation_261029")
SET = "ownreply261113"
STAGES = ("instruct", "sft", "dpo")
STAGE_NAME = {"instruct": "Instruct (pod A)", "sft": "SFT", "dpo": "DPO"}
ROWS = {"instruct": Q / "261029_instruct_persona_map/run_full/rows.jsonl",
        "sft": RUN / "full_copies/sft/rows.jsonl", "dpo": RUN / "full_copies/dpo/rows.jsonl"}
CLASSES = [("E", "In role"), ("P", "Partly in role"), ("G", "Not enough evidence"), ("W", "A different role"),
           ("U", "Cannot score")]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def keys_20():
    heads = re.findall(r"^## (\d+)\. (\S+) \(wording (\d), question (\d+), replicate (\d)\)$",
                       (RUN / "full_copies/side_by_side_20.md").read_text(), re.M)
    out = [dict(n=int(n), role=r, wording=int(w), question=int(q), replicate=int(p)) for n, r, w, q, p in heads]
    assert [k["n"] for k in out] == list(range(1, 21)), "side_by_side_20.md must hold keys 1-20"
    return out


def draw():
    from transformers import AutoTokenizer
    sys.path.insert(0, str(RUN))
    import stage_run
    judge2 = load_module("judge2", Q / "261008_base_persona_map/judge2.py")
    con = load_module("jv_sample", Q / "261029_judge_validation/sample.py").contracts()
    want = keys_20()
    per = {}
    for stage in STAGES:
        G = stage_run.load_pod_a(stage)
        tok = AutoTokenizer.from_pretrained(G.MODEL, revision=G.REVISION, local_files_only=True)
        keys = {G.R.item_key(k["role"], "chat", k["wording"], k["question"], k["replicate"]): k["n"] for k in want}
        items = G.render(tok, [it for it in G.build_items(sorted({k["role"] for k in want}), 5) if it["key"] in keys])
        stored = {}
        for line in open(ROWS[stage]):
            r = json.loads(line)
            k = G.R.item_key(r["role"], r["format"], r["wording"], r["question"], r["replicate"])
            if k in keys:
                stored[k] = r
        for it in items:
            r = stored[it["key"]]
            if it["prompt_ids"] != r["prompt_ids"]:
                raise SystemExit(f"{stage} {it['key']}: rendered prompt differs from the stored prompt_ids")
            per.setdefault(keys[it["key"]], {})[stage] = dict(
                key=it["key"], model=G.MODEL, revision=G.REVISION,
                input=tok.decode(r["prompt_ids"], skip_special_tokens=False), question=it["question"],
                reply=r["reply"], stop=r["stop_reason"], kept=len(r["kept_ids"]), valid=r["valid"])
    lines = []
    for k in want:
        p = per[k["n"]]
        if len(p) != 3 or len({p[s]["input"] for s in STAGES}) != 1:
            raise SystemExit(f"key {k['n']}: missing stage or inputs differ across stages")
        role = k["role"]
        lines.append(dict(id=f"k{k['n']:02d}", n=k["n"], role=role, wording=k["wording"], question_id=k["question"],
                          replicate=k["replicate"], key=p["instruct"]["key"],
                          definition=None if role == "default" else judge2.definition(role),
                          contract=None if role == "default" else con[role], question=p["instruct"]["question"],
                          input=p["instruct"]["input"],
                          replies=[dict(stage=s, name=STAGE_NAME[s], model=p[s]["model"], revision=p[s]["revision"],
                                        reply=p[s]["reply"], stop=p[s]["stop"], kept=p[s]["kept"], valid=p[s]["valid"])
                                   for s in STAGES]))
    out = JV / f"{SET}_items.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; the set is drawn once")
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    print(f"20 keys x 3 stages: rendered prompts equal the stored prompt_ids; wrote {out} "
          f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


PAGE = (HERE / "read_page.html").read_text


def merge(old, new):
    """Per field id, keep the entry with the later updatedAt, so a stale tab cannot erase newer labels."""
    out = dict(old)
    for iid, lab in new.items():
        if iid not in out or str(lab.get("updatedAt", "")) >= str(out[iid].get("updatedAt", "")):
            out[iid] = lab
    return out


def serve(port):
    p = JV / f"{SET}_items.jsonl"
    raw = p.read_bytes()
    data = dict(set=SET, items_sha256=hashlib.sha256(raw).hexdigest(),
                items=[json.loads(l) for l in raw.decode().splitlines()], classes=CLASSES)
    page = PAGE().replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")).encode()
    store = JV / f"{SET}_owner_labels.json"

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(200, page, "text/html; charset=utf-8")
            elif self.path == "/labels":
                self.send(200, store.read_bytes() if store.exists() else b"{}", "application/json")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/labels":
                return self.send(404, b"not found", "text/plain")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("items_sha256") != data["items_sha256"]:
                return self.send(409, b"labels belong to another item set", "text/plain")
            old = json.loads(store.read_text())["labels"] if store.exists() else {}
            body["labels"] = merge(old, body.get("labels", {}))
            tmp = store.with_suffix(".tmp")
            tmp.write_text(json.dumps(body, indent=1, ensure_ascii=False))
            os.replace(tmp, store)
            self.send(200, json.dumps(body["labels"]).encode(), "application/json")

        def log_message(self, *args):
            pass

    print(f"http://127.0.0.1:{port}/  ->  labels saved to {store}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["draw", "serve"])
    ap.add_argument("--port", type=int, default=8768)
    a = ap.parse_args()
    draw() if a.mode == "draw" else serve(a.port)
