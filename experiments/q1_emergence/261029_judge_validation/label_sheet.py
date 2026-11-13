"""Owner labelling page for one judge_validation item set (THRESHOLDS.md v2.0: one yes/no label per reply).

The page opens with a short guide (the question, what yes and no mean, tricky cases, real examples) and then
shows one reply at a time with the role's signs in plain words. It never shows the key. The examples are
replies the owner has already seen; none is in the development or sealed sets. The page needs JavaScript:
a static file preview shows only a warning. Served by `serve`, it saves every change to
<set>_owner_labels.json next to the items (outside the repo).

    .venv/bin/python experiments/q1_emergence/261029_judge_validation/label_sheet.py build ITEMS_JSONL
    .venv/bin/python experiments/q1_emergence/261029_judge_validation/label_sheet.py serve ITEMS_JSONL [--port 8765]
"""
import argparse
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcripts import e3_replies, poda  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

CLASSES = [
    ("E", "In role"), ("P", "Partly in role"), ("G", "Not enough evidence"), ("W", "A different role"),
    ("U", "Cannot score"),
]

# (source, role shown, answer, why)
EXAMPLES = [
    (("e3", 9847), "robot", "Yes", "It answers as a machine: it checks, computes a probability, then acts."),
    (("e3", 12310), "toddler", "Yes", "Wonder and pretend play, in short, simple, childlike sentences."),
    (("a", "pirate"), "pirate", "Yes", "Pirate speech and a sailor's view of life all the way through."),
    (("e3", 13007), "translator", "Yes", "Tricky: it translates the question instead of answering it. Doing the role's job counts."),
    (("e3", 1856), "chef", "No", "Friendly, helpful advice, but nothing a chef would know or say."),
    (("e3", 6137), "grandparent", "No", "Anyone could say this. Nothing shows a grandparent."),
    (("e3", 10655), "stoic", "No", "It loops: the same sentences repeat until it is cut off."),
    (("e3", 4800), "assistant", "No", "This is clearly a cynic talking, not an assistant: a different role."),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Role labels</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root { --paper:#f5f5f5; --ink:#171717; --soft:#5c5c5c; --rule:#d4d4d4; --sig:#0b7366; --card:#ffffff; --no:#8a3b12; }
:root[data-theme="night"] { --paper:#121414; --ink:#ececec; --soft:#a3a3a3; --rule:#2e3333; --sig:#17cfb9; --card:#1a1d1d; --no:#e0956a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:18px/1.6 "Hanken Grotesk", system-ui, sans-serif; }
main { max-width:820px; margin:0 auto; padding:22px 16px 90px; }
header { display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; border-bottom:1px solid var(--rule); padding-bottom:10px; }
h1 { font-size:24px; margin:0 0 4px; }
h2 { font-size:19px; margin:28px 0 8px; }
.meta { color:var(--soft); font-size:16px; }
.num { font-variant-numeric:tabular-nums; }
button { font:inherit; color:var(--ink); background:var(--card); border:1px solid var(--rule); border-radius:4px; padding:8px 14px; cursor:pointer; }
button:hover { border-color:var(--sig); }
.link { border:none; background:none; padding:0; color:var(--sig); text-decoration:underline; text-underline-offset:3px; }
.short { background:var(--card); border-left:4px solid var(--sig); padding:14px 18px; margin:18px 0; }
.label { font-size:15px; font-weight:700; color:var(--soft); margin:24px 0 4px; }
.role { font-size:26px; font-weight:700; }
.signs { margin-top:6px; }
.signs div { margin:2px 0; }
.reply { background:var(--ink); color:var(--paper); padding:16px 18px; border-radius:4px; white-space:pre-wrap; font-size:19px; }
.frame { white-space:pre-wrap; color:var(--soft); border-left:2px solid var(--rule); padding-left:12px; font-size:16px; }
.ask { margin-top:26px; padding:16px 18px; border:1px solid var(--rule); border-radius:4px; background:var(--card); }
.ask .q { font-size:20px; font-weight:700; margin-bottom:12px; }
.yn button { font-size:20px; padding:10px 26px; margin-right:10px; }
.yn button.on.yes { background:var(--sig); border-color:var(--sig); color:var(--paper); }
.yn button.on.no { background:var(--no); border-color:var(--no); color:var(--paper); }
.choices { display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }
.choices button.on { background:var(--sig); border-color:var(--sig); color:var(--paper); }
textarea { width:100%; font:inherit; font-size:16px; color:var(--ink); background:var(--card); border:1px solid var(--rule); border-radius:4px; padding:8px; }
.nav { display:flex; gap:10px; align-items:center; margin-top:24px; flex-wrap:wrap; }
ul { padding-left:22px; } li { margin:4px 0; }
.ex { border:1px solid var(--rule); border-radius:4px; background:var(--card); padding:12px 16px; margin:12px 0; }
.ex .head { display:flex; justify-content:space-between; gap:10px; font-weight:700; }
.ex .ans.yes { color:var(--sig); } .ex .ans.no { color:var(--no); }
.ex .r { white-space:pre-wrap; font-size:16px; border-left:3px solid var(--rule); padding-left:10px; margin:8px 0; }
.ex .why { font-size:16px; }
details summary { cursor:pointer; color:var(--soft); }
.hide { display:none; }
</style></head><body><main>
<p id="nojs" style="border:1px solid var(--rule);padding:12px 14px;border-radius:4px">This page needs JavaScript. A file preview shows only this text. Ask Claude to open the labelling page, or open this file in Chrome or Safari.</p>

<section id="guide" class="hide">
<h1>Labelling: does the reply play the role?</h1>
<div class="meta">My guide. One minute to read.</div>
<div class="short">
I read one reply at a time and answer one question: <b>does this reply play the role, and does it make sense as an answer?</b>
I press <b>Y</b> for yes or <b>N</b> for no. There are <span class="num" id="g-n"></span> replies, about 25 minutes in all. Each label saves as I go, so I can stop and come back.
</div>

<h2>Yes means both</h2>
<ul>
<li>The reply shows the role clearly, in all of it or in part of it: its experience, its point of view, its way of talking, or doing its job.</li>
<li>It makes sense as an answer to the question. It may stop in the middle of a sentence; that is fine.</li>
</ul>

<h2>No means any of these</h2>
<ul>
<li>A plain answer that anyone could give.</li>
<li>Only the role's name, for example "As a pirate, I like to rest."</li>
<li>Polite, helpful advice with nothing of the role in it.</li>
<li>A different role.</li>
<li>Nonsense, the same sentences again and again, or someone else talking.</li>
</ul>

<h2>Tricky cases</h2>
<ul>
<li>Doing the role's job counts. A translator who translates the question is in role.</li>
<li>Short, simple, childlike speech fits a toddler.</li>
<li>Talk of being a machine or an AI fits the robot and the assistant. For human roles it is a different role.</li>
<li>Each reply shows the signs of its role in plain words. I use them when I am unsure.</li>
<li>When I am still unsure, I pick the closer answer and add a note.</li>
</ul>

<h2>Examples (replies I have seen before; none is in this set)</h2>
<div id="examples"></div>

<div class="nav"><button id="start">Start labelling</button></div>
</section>

<section id="work" class="hide">
<header><span><b id="setname"></b> · <span class="num" id="pos"></span></span>
<span class="meta"><span class="num" id="count"></span> done · <span id="saved"></span> · <button class="link" id="openguide">How to decide (?)</button> · <button class="link" id="theme">night</button></span></header>

<div class="label">The role</div>
<div class="role" id="role"></div>
<div id="definition"></div>
<div class="signs meta" id="signs"></div>

<div class="label">The question</div>
<div id="question"></div>

<div class="label">The reply</div>
<div class="reply" id="reply"></div>
<details style="margin-top:8px"><summary>Show the text the model was continuing</summary><div class="frame" id="frame"></div></details>

<div class="ask">
<div class="q" id="askq"></div>
<div class="yn"><button class="yes" id="yes">Yes (Y)</button><button class="no" id="no">No (N)</button></div>
</div>

<details style="margin-top:14px"><summary>Add detail (optional)</summary>
<div class="label">Which outcome?</div><div class="choices" id="classes"></div>
<div class="label">Words from the reply that decided it</div><textarea id="quotes" rows="2"></textarea>
<div class="label">Note</div><textarea id="note" rows="2"></textarea>
</details>

<div class="nav"><button id="prev">Previous</button><button id="next">Next</button><button id="nextopen">Next without a label</button>
<span style="flex:1"></span><button id="export">Export labels</button><label class="meta">Import <input type="file" id="import" accept=".json"></label></div>
<p class="meta">Keys: Y yes, N no, arrows to move, ? for the guide.</p>
</section>
</main>
<script>
document.getElementById("nojs").remove();
const DATA = __DATA__;
const SERVED = location.protocol.startsWith("http");
const KEY = "jv-labels-" + DATA.items_sha256;
let labels = {}, i = 0, timer = null;
const $ = id => document.getElementById(id);
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
try { if (localStorage.getItem("theme") === "night") document.documentElement.dataset.theme = "night"; } catch (e) {}
function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function lab(id) { return labels[id] || (labels[id] = {}); }
function done(id) { const l = labels[id]; return l && (l.pass === "yes" || l.pass === "no"); }
function save() {
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {}
  if (!SERVED) { $("saved").textContent = "in this browser only: export when done"; return; }
  clearTimeout(timer);
  timer = setTimeout(() => fetch("/labels", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ set: DATA.set, items_sha256: DATA.items_sha256, savedAt: new Date().toISOString(), labels }) })
    .then(r => r.ok ? r.json() : Promise.reject()).then(merged => { for (const k in merged) if (!labels[k] || (merged[k].updatedAt || "") > (labels[k].updatedAt || "")) labels[k] = merged[k]; $("saved").textContent = "saved"; $("count").textContent = DATA.items.filter(x => done(x.id)).length; })
    .catch(() => { $("saved").textContent = "NOT saved"; }), 300);
}
function stamp() { lab(DATA.items[i].id).updatedAt = new Date().toISOString(); save(); }
function signs(c) {
  const parts = c.role_act.split("; "), yes = [], no = [];
  for (const p of parts) { if (p.endsWith(" conflicts")) no.push(p.slice(0, -10)); else yes.push(p); }
  if (c.speech === "limited") yes.push("short, simple or broken speech");
  if (c.ai_talk === "fits") yes.push("talk of being a machine or an AI");
  if (c.ai_talk === "conflicts") no.push("talk of being an AI");
  let h = "<div><b>Signs of this role:</b> " + esc(yes.join("; ")) + ".</div>";
  if (no.length) h += "<div><b>Does not fit:</b> " + esc(no.join("; ")) + ".</div>";
  return h;
}
function show() {
  const it = DATA.items[i], l = labels[it.id] || {};
  $("setname").textContent = DATA.set === "dev" ? "Practice set" : "Sample set";
  $("pos").textContent = "Reply " + (i + 1) + " of " + DATA.items.length;
  $("count").textContent = DATA.items.filter(x => done(x.id)).length;
  $("role").textContent = it.role; $("definition").textContent = it.definition; $("signs").innerHTML = signs(it.contract);
  $("question").textContent = it.question; $("reply").textContent = it.reply; $("frame").textContent = it.frame;
  $("askq").textContent = "Does this reply play the " + it.role + ", and does it make sense as an answer?";
  $("yes").classList.toggle("on", l.pass === "yes"); $("no").classList.toggle("on", l.pass === "no");
  [...$("classes").children].forEach((b, j) => b.classList.toggle("on", l.class === DATA.classes[j][0]));
  $("quotes").value = l.quotes || ""; $("note").value = l.note || "";
}
function view(name) { $("guide").classList.toggle("hide", name !== "guide"); $("work").classList.toggle("hide", name !== "work"); window.scrollTo(0, 0); if (name === "work") show(); }
function go(d) { i = Math.max(0, Math.min(DATA.items.length - 1, i + d)); show(); window.scrollTo(0, 0); }
function setPass(v) { lab(DATA.items[i].id).pass = v; stamp(); show(); }
$("g-n").textContent = DATA.items.length;
for (const ex of DATA.examples) {
  $("examples").insertAdjacentHTML("beforeend", '<div class="ex"><div class="head"><span>Role: ' + esc(ex.role) +
    '</span><span class="ans ' + ex.answer.toLowerCase() + '">' + ex.answer + '</span></div><div class="meta">Question: ' + esc(ex.question) +
    '</div><div class="r">' + esc(ex.reply) + '</div><div class="why">' + esc(ex.why) + '</div></div>');
}
DATA.classes.forEach(([k, name]) => { const b = document.createElement("button"); b.textContent = name; b.onclick = () => { lab(DATA.items[i].id).class = k; stamp(); show(); }; $("classes").appendChild(b); });
$("yes").onclick = () => setPass("yes"); $("no").onclick = () => setPass("no");
$("quotes").oninput = e => { lab(DATA.items[i].id).quotes = e.target.value; stamp(); };
$("note").oninput = e => { lab(DATA.items[i].id).note = e.target.value; stamp(); };
$("prev").onclick = () => go(-1); $("next").onclick = () => go(1);
$("nextopen").onclick = () => { const j = DATA.items.findIndex((x, k) => k > i && !done(x.id)); const f = j < 0 ? DATA.items.findIndex(x => !done(x.id)) : j; if (f >= 0) { i = f; show(); window.scrollTo(0, 0); } };
$("start").onclick = () => view("work"); $("openguide").onclick = () => view("guide");
$("export").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({ set: DATA.set, items_sha256: DATA.items_sha256, exportedAt: new Date().toISOString(), labels }, null, 1)], { type: "application/json" }));
  a.download = DATA.set + "_owner_labels.json"; a.click();
};
$("import").onchange = async e => {
  const d = JSON.parse(await e.target.files[0].text());
  if (d.items_sha256 !== DATA.items_sha256) { alert("These labels belong to another item set."); return; }
  for (const k in d.labels) if (!labels[k] || (d.labels[k].updatedAt || "") > (labels[k].updatedAt || "")) labels[k] = d.labels[k];
  save(); show();
};
$("theme").onclick = () => { const night = document.documentElement.dataset.theme !== "night"; document.documentElement.dataset.theme = night ? "night" : ""; $("theme").textContent = night ? "paper" : "night"; try { localStorage.setItem("theme", night ? "night" : "paper"); } catch (e) {} };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT" || $("work").classList.contains("hide")) { if (e.key === "Enter" && !$("guide").classList.contains("hide")) view("work"); return; }
  const k = e.key.toUpperCase();
  if (k === "Y") setPass("yes"); else if (k === "N") setPass("no");
  else if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "?") view("guide");
});
$("saved").textContent = SERVED ? "autosave on" : "in this browser only: export when done";
function begin() { const first = DATA.items.findIndex(x => !done(x.id)); i = first < 0 ? 0 : first; view(DATA.items.some(x => done(x.id)) ? "work" : "guide"); }
if (SERVED) fetch("/labels").then(r => r.json()).then(d => { if (d.labels) labels = Object.assign(labels, d.labels); begin(); }).catch(begin);
else begin();
</script></body></html>
"""


def examples():
    e3 = e3_replies()
    out = []
    for src, role, answer, why in EXAMPLES:
        if src[0] == "e3":
            q, reply = e3[src[1]]["question"], e3[src[1]]["reply"].strip()
        else:
            q, reply = poda(src[1])
        out.append(dict(role=role, question=q, reply=reply, answer=answer, why=why))
    return out


def merge(old, new):
    """Per item, keep the label with the later updatedAt, so a stale tab cannot erase newer labels."""
    out = dict(old)
    for iid, lab in new.items():
        if iid not in out or str(lab.get("updatedAt", "")) >= str(out[iid].get("updatedAt", "")):
            out[iid] = lab
    return out


def build(items_path):
    p = Path(items_path).resolve()
    if p.is_relative_to(REPO):
        raise SystemExit("keep item sets and labels outside the repo")
    raw = p.read_bytes()
    data = dict(set=p.name.removesuffix("_items.jsonl"), items_sha256=hashlib.sha256(raw).hexdigest(),
                items=[json.loads(l) for l in raw.decode().splitlines()], classes=CLASSES, examples=examples())
    page = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out = p.with_name(data["set"] + "_labels.html")
    out.write_text(page)
    return p, data, page, out


def serve(items_path, port):
    p, data, page, _ = build(items_path)
    store = p.with_name(data["set"] + "_owner_labels.json")

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/" + data["set"]):
                self.send(200, page.encode(), "text/html; charset=utf-8")
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
    ap.add_argument("mode", choices=["build", "serve"])
    ap.add_argument("items")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.mode == "build":
        print(build(a.items)[3])
    else:
        serve(a.items, a.port)
