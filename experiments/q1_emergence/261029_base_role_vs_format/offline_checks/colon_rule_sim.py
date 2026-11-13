"""Offline: re-cut the run_canary_v2 raw ids with a rule that skips a blank line right after a colon."""
import importlib.util, json, re, statistics, sys
from collections import Counter
from pathlib import Path
HERE = Path("/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_base_role_vs_format")
spec = importlib.util.spec_from_file_location("base_role_vs_format_gen_extract", HERE / "gen_extract.py")
G = importlib.util.module_from_spec(spec); sys.modules[spec.name] = G; spec.loader.exec_module(G)
R = G.R
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(G.MODEL, revision=G.REVISION)
EOS, PAD = tok.eos_token_id, tok.pad_token_id
DATASET = re.compile(r"(Answer:|Options:|Question:|\bQ:|\bA:|Based on (this|the)|choose the best|What type of details|Click for|Context:|Q & A|\nInput:|\nOutput:|Title:|Q\.|Choose)", re.I)
def rep4(t):
    w = re.findall(r"[a-z']+", t.lower()); c = Counter(tuple(w[i:i+4]) for i in range(len(w)-3)); return bool(c) and max(c.values()) >= 3

def cut_colon(raw, markers):
    """rows.cut_reply, except that a "\n\n" whose text before it ends with ":" (spaces ignored) is not a stop."""
    ids, reason = list(raw), "limit"
    for k, t in enumerate(ids):
        if t in (EOS, PAD):
            ids, reason = ids[:k], "eos" if t == EOS else "pad"; break
    text = tok.decode(ids)
    hits = [p for p in (text.find(m) for m in markers if m != "\n\n") if p != -1]
    if "\n\n" in markers:
        for m in re.finditer(r"\n\n", text):
            if not text[:m.start()].rstrip(" ").endswith(":"):
                hits.append(m.start()); break
    if hits:
        p, k = min(hits), 0
        while k < len(ids) and len(tok.decode(ids[:k + 1])) <= p:
            k += 1
        ids, reason = ids[:k], "marker"
    return ids, reason

rows = [json.loads(l) for l in open(HERE / "run_canary_v2/rows.jsonl")]
new, changed = [], []
for r in rows:
    kept, reason = cut_colon(r["raw_ids"], G.TEMPLATES[r["format"]]["stops"])
    n = dict(r, kept_ids=kept, stop_reason=reason, reply=tok.decode(kept), valid=R.is_valid(tok, kept),
             span=[len(r["prompt_ids"]), len(r["prompt_ids"]) + len(kept)])
    new.append(n)
    if kept != r["kept_ids"]:
        changed.append((r, n))
def stats(rs, label):
    for f in G.FORMATS:
        s = [r for r in rs if r["format"] == f]; N = len(s); pc = lambda k: f"{100*k/N:.0f}%"
        q = [G.quote_span(tok, r) for r in s]
        print(f"{label:6s} {f:12s} dataset {pc(sum(bool(DATASET.search(r['reply'])) for r in s)):>4s}"
              f" rep4 {pc(sum(rep4(r['reply']) for r in s)):>4s} names {pc(sum(G.names_role(r['role'], r['reply']) for r in s)):>4s}"
              f" quote {pc(sum(x is not None for x in q)):>4s} at128 {pc(sum(r['stop_reason']=='limit' for r in s)):>4s}"
              f" <5w {pc(sum(len(r['reply'].split())<5 for r in s)):>4s} median {statistics.median(len(r['kept_ids']) for r in s)}"
              f" invalid {sum(not r['valid'] for r in s)}")
stats(rows, "v1.5"); stats(new, "colon")
print("changed rows by format:", Counter(r["format"] for r, _ in changed))
json.dump([dict(key=R.row_key(a), before=a["reply"], after=b["reply"], stop=b["stop_reason"]) for a, b in changed],
          open(Path(__file__).parent / "colon_rule_changed.json", "w"), indent=1)
