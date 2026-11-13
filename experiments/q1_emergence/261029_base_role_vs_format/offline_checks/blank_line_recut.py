"""Re-cut the canary raw ids with the new stop markers; report valid rate, stops, quotes, length."""
import importlib.util, json, sys
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path("/Users/brendon/persona-manifolds/manifold-persona/experiments/q1_emergence/261029_base_role_vs_format")
spec = importlib.util.spec_from_file_location("base_role_vs_format_gen_extract", HERE / "gen_extract.py")
G = importlib.util.module_from_spec(spec); sys.modules[spec.name] = G; spec.loader.exec_module(G)
R = G.R
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(G.MODEL, revision=G.REVISION)
rows = [json.loads(l) for l in open(HERE / "run_canary/rows.jsonl")]
new = []
for r in rows:
    k = R.row_key(r)
    n = R.make_row(tok, "recut", k, r["prompt_ids"], r["raw_ids"], tok.eos_token_id, tok.pad_token_id,
                   G.TEMPLATES[r["format"]]["stops"])
    new.append(n)
cells = defaultdict(list)
for n in new:
    cells[(n["role"], n["format"])].append(n["valid"])
print("min valid per role x format:", min(sum(v)/len(v) for v in cells.values()))
for fmt in G.FORMATS:
    f = [n for n in new if n["format"] == fmt]
    inv = sum(not n["valid"] for n in f)
    q = [G.quote_span(tok, n) for n in f if n["valid"]]
    short = sum(len(n["reply"].split()) < 5 for n in f)
    lens = sorted(len(n["kept_ids"]) for n in f)
    worst = min(sum(v)/len(v) for (ro, fo), v in cells.items() if fo == fmt)
    print(f"{fmt:12s} invalid {inv:3d}/{len(f)}  worst cell {worst:.3f}  stops {dict(Counter(n['stop_reason'] for n in f))}"
          f"  quote {sum(x is not None for x in q)/len(q):.2f}  <5 words {short/len(f):.2f}  median tokens {lens[len(lens)//2]}")
grid = json.load(open(HERE / "run_canary/grid.json"))
keys = grid["keys"] if isinstance(grid, dict) else grid
s = G.support({R.row_key(n): n["valid"] for n in new}, keys)
print("D1 support: complete", len(s["complete_questions"]), "of", s["n_questions"], "lost prompts", s["n_lost_prompts"])
for n in [n for n in new if not n["valid"]][:8]:
    print("INVALID", R.row_key(n), repr(tok.decode(n["raw_ids"][:12])))
