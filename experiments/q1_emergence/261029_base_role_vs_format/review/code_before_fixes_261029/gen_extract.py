"""Pod B base_role_vs_format: OLMo-2-7B base writes role replies in 4 document formats; save rows and states.

Design: ../261029_stage_persona_plan/DESIGN.md (pod B) and THRESHOLDS.md here.
- Base model: plain text with one <|endoftext|> at the start; no chat template, no user/assistant markup.
- Formats: interview (E3 desc, exact text and stop markers), script, story, description. The character is
  "Alex" in the last three. A description sets the role in all four; only the frame changes. The
  description line is the only role text, so C2 (description removed) drops that one line.
- Templates approved by the owner on 29/10/2026, before the canary.
- 10 roles (owner, 29/10) x 4 formats x 5 E3 third-person wordings x 40 frozen questions x 3 replicates.
  Canary (owner, 29/10): toddler, consultant, chef, pirate; the first 10 question ids; 2 replicates.
- Sampling as E3: temperature 0.5, repetition penalty 1.1, top_k 0, top_p 1, 128 new tokens. Each reply's seed
  is item_seed(item_key) (manifold_persona.rows), so batch size and order do not change it.
- Reply = generated ids before eos or pad and before the token where a stop marker of its format starts. In
  script, story and description also before the first blank line that does not follow a colon (cut_reply).
  Mean over the kept reply positions at every hidden state (embeddings + 32 layers), float16; NaN rows when
  the reply is invalid.
- Second measure, quoted speech (owner, 29/10): in the interview and script the whole kept reply (the speaker
  label makes it the character's speech); in the story and description the first double-quoted passage
  (" or curly quotes; an unclosed quote runs to the reply end). A token counts when one of its bytes is inside
  the quote marks. No quote, or an empty one, gives a NaN row. Stored at the EXTRA_DEPTHS fractions of model
  depth only (depth_layers).
- D1 support (THRESHOLDS.md): the full run stops once more than 10 questions have a prompt (role, format,
  wording) with no valid replicate, because D1 can then not keep 30 of 40 questions.
- Resume only when the run fingerprint (model, tokenizer, templates, code, settings, grid) matches.

Run on the GPU pod:
  python gen_extract.py --grid canary --dry              # items, grid and rendered prompts; no model
  python gen_extract.py --grid canary --limit 40         # smoke test
  python gen_extract.py --grid canary                    # canary, 1,600 replies
  python gen_extract.py                                  # full run, 24,000 replies
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from manifold_persona import rows as R

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E3 = HERE.parent / "261008_base_persona_map"
MODEL = "allenai/OLMo-2-1124-7B"
REVISION = "7df9a82518afdecae4e8c026b27adccc8c1f0032"
EOS_TEXT = "<|endoftext|>"
ROLES = ["toddler", "robot", "narcissist", "cynic", "optimist", "teenager", "vampire", "consultant", "chef", "pirate"]
N_WORDINGS, N_REPLICATES = 5, 3
CANARY_ROLES = ["toddler", "consultant", "chef", "pirate"]
CANARY_N_QUESTIONS, CANARY_REPLICATES = 10, 2  # the first 10 ids of questions_40.json
TEMPERATURE, PENALTY, MAX_NEW = 0.5, 1.1, 128
MAX_LOST_QUESTIONS = 10
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
F16_MAX = 65504.0
EXTRA_DEPTHS = (0.25, 0.5, 0.75)  # owner, 29/10: hidden states 8, 16, 24 on the 32-layer 7B
SPEECH_FORMATS = ("interview", "script")  # the whole reply is the character's speech
OPEN_QUOTES, CLOSE_QUOTES = (b'"', "\u201c".encode()), (b'"', "\u201d".encode())
NAME_TERMS = {  # E3 score_naming.NAME_TERMS for the 10 pod B roles
    "toddler": ["toddler"], "robot": ["robot", "robotic"], "narcissist": ["narcissist", "narcissistic"],
    "cynic": ["cynic", "cynical"], "optimist": ["optimist", "optimistic"],
    "teenager": ["teenager", "teen", "teenage", "adolescent"], "vampire": ["vampire"], "consultant": ["consultant"],
    "chef": ["chef", "cook"], "pirate": ["pirate"]}

# Each prompt: EOS + header, the description line, the question turn, joined by blank lines. "{desc}" is an
# E3 third-person wording ("a pirate captain who ..."); "{q}" is the question text. The reply is the
# continuation after the turn; it ends before eos, pad or the first stop marker. With blank_stop, it also ends at
# the first blank line that does not follow a colon (owner, 29/10): the model often writes "say:", a blank line,
# then the answer.
TEMPLATES = {
    "interview": dict(header="The following is an interview. The guest answers in their own voice.",
                      about="About the guest: {desc}",
                      turn="Interviewer: {q}\nGuest:",
                      stops=("\nInterviewer:", "\nAbout the guest:"), blank_stop=False),
    "script": dict(header="The following is a script. Each line starts with the name of the speaker.",
                   about="Alex is {desc}.",
                   turn="Friend: {q}\nAlex:",
                   stops=("\nFriend:", "\nThe following is"), blank_stop=True),
    "story": dict(header="The following is a short story.",
                  about="Alex is {desc}.",
                  turn='One day, a friend said to Alex, "{q}" Alex',
                  stops=("\nThe following is",), blank_stop=True),
    "description": dict(header="The following is a description of a person.",
                        about="Alex is {desc}.",
                        turn='If Alex were asked, "{q}" Alex would',
                        stops=("\nIf Alex", "\nThe following is"), blank_stop=True),
}
FORMATS = tuple(TEMPLATES)


def blank_stop(text):
    """Start of the first blank line that does not follow a colon (spaces ignored), or -1."""
    for m in re.finditer("\n\n", text):
        if not text[:m.start()].rstrip(" ").endswith(":"):
            return m.start()
    return -1


def cut_reply(tok, raw_ids, fmt):
    """rows.cut_reply with the format's stop markers, plus its blank-line stop. One pass over the decoded text,
    so the earliest stop wins; the token where it starts is dropped, as in rows.cut_reply."""
    t = TEMPLATES[fmt]
    ids, reason = R.cut_reply(tok, raw_ids, tok.eos_token_id, tok.pad_token_id)
    text = tok.decode(ids, skip_special_tokens=False)
    hits = [p for p in [text.find(m) for m in t["stops"]] + [blank_stop(text) if t["blank_stop"] else -1] if p != -1]
    if hits:
        p, k = min(hits), 0
        while k < len(ids) and len(tok.decode(ids[:k + 1], skip_special_tokens=False)) <= p:
            k += 1
        ids, reason = ids[:k], "marker"
    return ids, reason


def make_row(tok, run_id, key, prompt_ids, raw_ids):
    """rows.make_row, with the reply cut by cut_reply."""
    row = R.make_row(tok, run_id, key, prompt_ids, raw_ids, tok.eos_token_id, tok.pad_token_id)
    kept, reason = cut_reply(tok, raw_ids, row["format"])
    start = len(prompt_ids)
    row.update(kept_ids=kept, span=[start, start + len(kept)], stop_reason=reason, valid=R.is_valid(tok, kept),
               reply=tok.decode(kept, skip_special_tokens=False))
    return row


def check_row(tok, row):
    """rows.check_row with the format's stop markers; the kept ids and stop reason must equal cut_reply of the
    raw ids."""
    R.check_row(row, {tok.eos_token_id, tok.pad_token_id}, tok, TEMPLATES[row["format"]]["stops"])
    if (row["kept_ids"], row["stop_reason"]) != cut_reply(tok, row["raw_ids"], row["format"]):
        raise ValueError(f"{R.row_key(row)}: kept ids are not the format's cut of the raw ids")


def templates_hash():
    return hashlib.sha256(json.dumps(TEMPLATES, sort_keys=True).encode()).hexdigest()


def prompt_text(fmt, desc, question):
    """The plain-text prompt of one item. desc=None gives the C2 frame: the same text without the
    description line. str.replace, not str.format, so braces in a question stay literal."""
    t = TEMPLATES[fmt]
    about = [t["about"].replace("{desc}", desc)] if desc is not None else []
    return EOS_TEXT + "\n\n".join([t["header"]] + about + [t["turn"].replace("{q}", question)])


def prompt_ids(tok, text):
    """Token ids of a prompt; the one <|endoftext|> is its first token (the tokenizer adds none)."""
    ids = tok(text, add_special_tokens=False).input_ids
    assert ids[0] == tok.eos_token_id and ids.count(tok.eos_token_id) == 1, text[:80]
    return ids


def descriptions(roles, repo=REPO):
    """E3 third-person wordings per role. The reviewed file must match the refs instruction line it came from."""
    desc3 = json.load(open(E3 / "descriptions_third_person.json"))
    out = {}
    for role in roles:
        lines = [d["pos"] for d in json.load(open(repo / f"refs/assistant-axis/roles/instructions/{role}.json"))["instruction"]]
        assert len(lines) == N_WORDINGS, role
        for w, line in enumerate(lines):
            rec = desc3[f"{role}/{w}"]
            assert rec["original"] == line, (role, w)
            out[role, w] = rec["third_person"]
    return out


def grid_spec(which):
    """(roles, questions, replicates) of the full run or the canary; formats and wordings are always all."""
    qs = R.load_questions(E3 / "questions_40.json")
    if which == "full":
        return ROLES, qs, N_REPLICATES
    if which == "canary":
        return CANARY_ROLES, qs[:CANARY_N_QUESTIONS], CANARY_REPLICATES
    raise ValueError(f"unknown grid {which!r}")


def build_items(roles, questions, n_rep, repo=REPO):
    """One item per (role, format, wording, question, replicate), in grid order (rows.grid_keys)."""
    bank = [json.loads(l)["question"] for l in open(repo / "refs/assistant-axis/extraction_questions.jsonl")]
    desc = descriptions(roles, repo)
    keys = R.grid_keys(roles, FORMATS, range(N_WORDINGS), questions, range(n_rep))
    items = []
    for k in keys:
        role, fmt, w, q, _ = json.loads(k)
        items.append(dict(key=k, fmt=fmt, desc=desc[role, w], question=bank[q],
                          text=prompt_text(fmt, desc[role, w], bank[q])))
    return items


def names_role(role, text):
    """E3 self-naming rule: the reply text names its own role (score_naming.flags)."""
    names = "|".join(re.escape(t) for t in NAME_TERMS[role])
    return bool(re.search(rf"\b({names})(s|es)?\b", text, re.I))


def render(tok, items):
    """Prompt ids; items are then sorted by prompt length (ties by key) so batches carry little padding.
    The replicates of one prompt have equal length and adjacent keys, so they stay together."""
    for it in items:
        it["prompt_ids"] = prompt_ids(tok, it["text"])
    return sorted(items, key=lambda it: (len(it["prompt_ids"]), it["key"]))


def depth_layers(n_layers, depths=EXTRA_DEPTHS):
    """Hidden-state indices for fractions of model depth: floor(f * n_layers + 0.5). Index 0 is the embedding
    output and n_layers the last layer, so a fraction counts layers, not hidden states."""
    idx = tuple(math.floor(f * n_layers + 0.5) for f in depths)
    if not all(1 <= i <= n_layers for i in idx) or len(set(idx)) != len(idx):
        raise ValueError(f"depths {depths} give hidden states {idx} on a {n_layers}-layer model")
    return idx


# ---- quoted speech

def token_bytes(tok, ids):
    """The exact bytes of each token (byte-level BPE), so a span never depends on how a partial character
    decodes."""
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    inv = {c: b for b, c in bytes_to_unicode().items()}
    out = []
    for t in tok.convert_ids_to_tokens(ids):
        out.append(bytes(inv[c] for c in t) if all(c in inv for c in t) else t.encode())
    return out


def first_quote(tok, kept_ids):
    """Token range [i, j) of the first double-quoted passage of a reply, or None when there is none or it is
    empty. Tokens that hold a quote mark and quoted text count."""
    pieces = token_bytes(tok, kept_ids)
    text = b"".join(pieces)
    opens = [(text.find(m), m) for m in OPEN_QUOTES if text.find(m) != -1]
    if not opens:
        return None
    p, m = min(opens)
    a = p + len(m)
    closes = [text.find(c, a) for c in CLOSE_QUOTES if text.find(c, a) != -1]
    b = min(closes) if closes else len(text)
    if not text[a:b].decode("utf-8", "replace").strip():
        return None
    inside, end = [], 0
    for k, x in enumerate(pieces):
        start, end = end, end + len(x)
        if start < b and end > a:
            inside.append(k)
    return inside[0], inside[-1] + 1


def quote_span(tok, row):
    """Absolute [start, end) of the quoted-speech positions of a row, or None (its quote state is NaN)."""
    if not row["valid"]:
        return None
    if row["format"] in SPEECH_FORMATS:
        return list(row["span"])
    q = first_quote(tok, row["kept_ids"])
    return None if q is None else [row["span"][0] + q[0], row["span"][0] + q[1]]


# ---- D1 support

def prompt_of(key):
    role, fmt, w, q, _ = json.loads(key)
    return role, fmt, w, q


def support(valid, grid):
    """D1 support from {key: valid} against the grid. A prompt is lost when every grid replicate of it has a
    row and none is valid. A question is complete when every grid prompt of it has a valid replicate."""
    want = Counter(prompt_of(k) for k in grid)
    got = defaultdict(list)
    for k, v in valid.items():
        got[prompt_of(k)].append(v)
    lost = sorted(p for p, vs in got.items() if len(vs) == want[p] and not any(vs))
    ok = {p for p, vs in got.items() if any(vs)}
    questions = sorted({p[3] for p in want})
    complete = [q for q in questions if all(p in ok for p in want if p[3] == q)]
    return dict(n_questions=len(questions), complete_questions=complete,
                questions_with_lost_prompt=sorted({p[3] for p in lost}), n_lost_prompts=len(lost),
                lost_by_role_format={f"{r}/{f}": n for (r, f), n in sorted(Counter(p[:2] for p in lost).items())},
                lost_by_question={str(q): n for q, n in sorted(Counter(p[3] for p in lost).items())})


# ---- generation and pooling

def read_done(part, batches):
    """{key: valid} of complete batches in the rows part file. A crash during an append can leave a truncated
    last line and only some rows of a batch; both are removed, so that batch is generated again."""
    if not part.exists():
        return {}
    lines = [l for l in part.read_bytes().split(b"\n") if l.strip()]
    rows = []
    for k, l in enumerate(lines):
        try:
            r = json.loads(l)
            rows.append((R.row_key(r), r["valid"], l))
        except json.JSONDecodeError:
            assert k == len(lines) - 1, "corrupt line inside the part file"
    have = {k for k, _, _ in rows}
    done = set()
    for chunk in batches:
        keys = {it["key"] for it in chunk}
        if keys <= have:
            done |= keys
    keep = [l for k, _, l in rows if k in done]
    if len(keep) != len(lines):
        tmp = part.with_suffix(".tmp")
        tmp.write_bytes(b"".join(l + b"\n" for l in keep))
        os.replace(tmp, part)
        print(f"removed {len(lines) - len(keep)} lines of incomplete batches from {part.name}", flush=True)
    return {k: v for k, v, _ in rows if k in done}


def run(items, out_dir, batch_size, model, tok, run_id, grid, max_lost=None):
    """Generate and pool batch by batch. A batch counts as done only when its rows and its states exist.
    With max_lost set, stop (exit code 4, stopped_d1.json) once more questions than that have a lost prompt."""
    import numpy as np
    eos, pad = tok.eos_token_id, tok.pad_token_id
    part, acts_dir = out_dir / "rows.partial.jsonl", out_dir / "acts_parts"
    acts_dir.mkdir(parents=True, exist_ok=True)
    batches = [items[s:s + batch_size] for s in range(0, len(items), batch_size)]
    valid = read_done(part, batches)
    for b, chunk in enumerate(batches):
        keys = [it["key"] for it in chunk]
        if all(k in valid for k in keys):
            f = acts_dir / f"{b:06d}.npz"
            assert f.exists() and np.load(f)["keys"].tolist() == keys, f"state part missing for batch {b}"
    for b, chunk in enumerate(batches):
        keys = [it["key"] for it in chunk]
        if not all(k in valid for k in keys):
            raw = R.generate_batch(model, [it["prompt_ids"] for it in chunk], [R.item_seed(k) for k in keys],
                                   eos, pad, TEMPERATURE, PENALTY, MAX_NEW)
            rows = [make_row(tok, run_id, it["key"], it["prompt_ids"], g) for it, g in zip(chunk, raw)]
            for r in rows:
                check_row(tok, r)
            spans = [r["span"] if r["valid"] else (r["span"][0], r["span"][0]) for r in rows]  # invalid -> NaN
            acts = R.pool_spans(model, [r["prompt_ids"] + r["kept_ids"] for r in rows], spans, pad)
            ok = np.array([r["valid"] for r in rows])
            assert np.isfinite(acts[ok]).all() and np.isnan(acts[~ok]).all(), f"batch {b}: non-finite states"
            quotes = pool_quotes(model, tok, rows, acts)
            fits = all(not np.isfinite(x).any() or np.abs(x[np.isfinite(x)]).max() <= F16_MAX for x in (acts, quotes))
            if not fits:
                print(f"batch {b}: states exceed float16; this part stays float32", flush=True)
            np.savez(acts_dir / f"{b:06d}.npz", acts=acts.astype(np.float16) if fits else acts,
                     quotes=quotes.astype(np.float16) if fits else quotes, keys=np.array(keys), f32=np.array(not fits))
            with open(part, "a") as f:
                f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
                f.flush()
                os.fsync(f.fileno())
            valid.update((R.row_key(r), r["valid"]) for r in rows)
            print(f"batch {b + 1}/{len(batches)}", flush=True)
        if max_lost is not None:
            s = support(valid, grid)
            if len(s["questions_with_lost_prompt"]) > max_lost:
                (out_dir / "stopped_d1.json").write_text(json.dumps(dict(after_batch=b, rows=len(valid), **s), indent=1))
                print(f"STOP: {len(s['questions_with_lost_prompt'])} questions have a prompt with no valid replicate "
                      f"(limit {max_lost}); D1 cannot keep 30 of 40. See stopped_d1.json.", flush=True)
                raise SystemExit(4)


def pool_quotes(model, tok, rows, acts):
    """Quoted-speech states at depth_layers(): interview and script rows copy their reply states; story and
    description rows get a second pass over the first quote; no quote gives NaN."""
    import numpy as np
    layers = depth_layers(model.config.num_hidden_layers)
    spans = [quote_span(tok, r) for r in rows]
    out = np.full((len(rows), len(layers), acts.shape[2]), np.nan, dtype=np.float32)
    for j, (r, sp) in enumerate(zip(rows, spans)):
        if sp is not None and r["format"] in SPEECH_FORMATS:
            out[j] = acts[j, list(layers)]
    todo = [j for j, (r, sp) in enumerate(zip(rows, spans)) if sp is not None and r["format"] not in SPEECH_FORMATS]
    if todo:
        out[todo] = R.pool_spans(model, [rows[j]["prompt_ids"] + rows[j]["kept_ids"] for j in todo],
                                 [spans[j] for j in todo], tok.pad_token_id, layers)
    return out


def finalize(out_dir, grid, tok):
    """rows.jsonl, reply_mean_acts.npy, quote_mean_acts.npy and reply_extras.jsonl in grid order; every grid
    key must have a row."""
    import numpy as np
    from numpy.lib.format import open_memmap
    rows = [json.loads(l) for l in open(out_dir / "rows.partial.jsonl") if l.strip()]
    split = R.check_rows(rows, grid)
    assert not split["absent"], f"{len(split['absent'])} grid keys have no row"
    pos = {k: i for i, k in enumerate(grid)}
    by_key = {R.row_key(r): r for r in rows}
    parts = sorted((out_dir / "acts_parts").glob("*.npz"))
    dtype = np.float32 if any(bool(np.load(f)["f32"]) for f in parts) else np.float16  # reads only the flag
    spans = {k: quote_span(tok, by_key[k]) for k in grid}
    acts, quotes, filled = None, None, set()
    for f in parts:  # one part at a time, so memory stays at one batch
        z = np.load(f)
        a, qa, keys = z["acts"], z["quotes"], z["keys"].tolist()
        v = np.array([by_key[k]["valid"] for k in keys])
        assert np.isfinite(a[v]).all() and np.isnan(a[~v]).all(), f"{f.name}: states do not match the valid flags"
        hq = np.array([spans[k] is not None for k in keys])
        assert np.isfinite(qa[hq]).all() and np.isnan(qa[~hq]).all(), f"{f.name}: quote states do not match the spans"
        if acts is None:
            acts = open_memmap(out_dir / "reply_mean_acts.npy", mode="w+", dtype=dtype, shape=(len(grid),) + a.shape[1:])
            quotes = open_memmap(out_dir / "quote_mean_acts.npy", mode="w+", dtype=dtype,
                                 shape=(len(grid),) + qa.shape[1:])
        for k, x, y in zip(keys, a, qa):
            acts[pos[k]], quotes[pos[k]] = x, y
            filled.add(k)
    assert filled == set(grid), f"{len(set(grid) - filled)} grid keys have no stored states"
    acts.flush()
    quotes.flush()
    with open(out_dir / "rows.jsonl", "w") as f:
        for k in grid:
            f.write(json.dumps(by_key[k], ensure_ascii=False) + "\n")
    with open(out_dir / "reply_extras.jsonl", "w") as f:  # derived from rows.jsonl; kept beside it for the analysis
        for k in grid:
            r, sp = by_key[k], spans[k]
            quote = None if sp is None else tok.decode(r["kept_ids"][sp[0] - r["span"][0]:sp[1] - r["span"][0]])
            f.write(json.dumps(dict(key=k, names_role=names_role(r["role"], r["reply"]), quote_span=sp, quote=quote),
                               ensure_ascii=False) + "\n")
    cells = defaultdict(list)
    for k in grid:
        cells[f"{by_key[k]['role']}/{by_key[k]['format']}"].append(by_key[k]["valid"])
    return dict(n_rows=len(grid), n_invalid=len(split["invalid"]), invalid_keys=split["invalid"],
                stop_reasons={fmt: dict(Counter(by_key[k]["stop_reason"] for k in grid if by_key[k]["format"] == fmt))
                              for fmt in FORMATS},
                valid_rate_by_role_format={c: sum(v) / len(v) for c, v in sorted(cells.items())},
                d1_support=support({k: by_key[k]["valid"] for k in grid}, grid),
                quote_rate_by_format={fmt: sum(spans[k] is not None for k in grid if by_key[k]["format"] == fmt)
                                      / max(1, sum(by_key[k]["format"] == fmt for k in grid)) for fmt in FORMATS},
                acts_shape=list(acts.shape), acts_dtype=str(acts.dtype))


def load_run(run_dir):
    """Rows, grid, states and manifest of a finished run, after the SHA256SUMS check."""
    import numpy as np
    for line in (run_dir / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        if R.file_sha256(run_dir / name) != h:
            raise SystemExit(f"{name} does not match SHA256SUMS.txt")
    grid = json.loads((run_dir / "grid.json").read_text())["keys"]
    rows = [json.loads(l) for l in open(run_dir / "rows.jsonl")]
    if [R.row_key(r) for r in rows] != grid:
        raise SystemExit("rows.jsonl is not in grid order")
    acts = np.load(run_dir / "reply_mean_acts.npy", mmap_mode="r")
    quotes = np.load(run_dir / "quote_mean_acts.npy", mmap_mode="r")
    extras = [json.loads(l) for l in open(run_dir / "reply_extras.jsonl")]
    if [x["key"] for x in extras] != grid:
        raise SystemExit("reply_extras.jsonl is not in grid order")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return dict(rows=rows, grid=grid, acts=acts, quotes=quotes, extras=extras, manifest=manifest)


def tokenizer_hashes():
    from huggingface_hub import hf_hub_download
    return {f: R.file_sha256(hf_hub_download(MODEL, f, revision=REVISION)) for f in TOKENIZER_FILES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="full", choices=("full", "canary"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: a fixed random subset of the grid")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    roles, questions, n_rep = grid_spec(args.grid)
    items = build_items(roles, questions, n_rep)
    grid, grid_hash = R.freeze_grid(it["key"] for it in items)
    if args.limit:
        items = random.Random(0).sample(items, args.limit)
        keep = {it["key"] for it in items}
        grid, grid_hash = R.freeze_grid(k for k in grid if k in keep)
    max_lost = MAX_LOST_QUESTIONS if args.grid == "full" and not args.limit else None
    out = Path(args.out or HERE / f"{'dry' if args.dry else 'run'}_{args.grid}")
    run_id = out.name

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    items = render(tok, items)
    code = {p.name: R.file_sha256(p) for p in (Path(__file__).resolve(), Path(R.__file__).resolve())}
    parts = dict(model=MODEL, revision=REVISION, tokenizer_files=tokenizer_hashes(),
                 templates=templates_hash(), code=code,
                 dtype="bfloat16", sampling=R.sampling_settings(TEMPERATURE, PENALTY, MAX_NEW),
                 extra_depths=EXTRA_DEPTHS, grid_hash=grid_hash, batch_size=args.batch_size,
                 max_lost_questions=max_lost)
    if args.dry:
        if (out / "fingerprint.txt").exists():
            raise SystemExit(f"{out} is a run directory; use another --out for --dry")
    else:
        R.claim_out_dir(out, R.fingerprint(parts))
        # THRESHOLDS "Templates": code, prompt and tokenizer hashes are on disk before any reply is generated
        (out / "fingerprint_parts.json").write_text(json.dumps(dict(parts, template_text=TEMPLATES), indent=1))
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(json.dumps(dict(grid_hash=grid_hash, keys=grid)))
    show = [R.item_key("pirate", fmt, 0, questions[0], 0) for fmt in FORMATS]
    shown = [it for it in items if it["key"] in show]
    (out / "prompts_check.txt").write_text("".join(f"===== {it['key']} =====\n{tok.decode(it['prompt_ids'])}<<END>>\n\n"
                                                   for it in shown))
    print(f"{len(items)} items, {len(roles)} roles, {len(FORMATS)} formats, grid {grid_hash[:12]}")
    if args.dry:
        return

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16).cuda().eval()
    commit = getattr(model.config, "_commit_hash", None)
    assert commit == REVISION, f"loaded weights {commit} are not the pinned revision {REVISION}"
    run(items, out, args.batch_size, model, tok, run_id, grid, max_lost)
    summary = finalize(out, grid, tok)
    import platform
    from importlib import metadata
    import tokenizers
    import transformers
    manifest = dict(parts, run_id=run_id, grid=args.grid, roles=roles, formats=list(FORMATS), questions=questions,
                    replicates=n_rep, limit=args.limit, template_text=TEMPLATES,
                    attn_implementation=getattr(model.config, "_attn_implementation", None),
                    n_hidden_states=model.config.num_hidden_layers + 1, fingerprint=R.fingerprint(parts),
                    act_rows="same order as rows.jsonl and grid.json; rows with valid=false are NaN",
                    extra_layers=depth_layers(model.config.num_hidden_layers),
                    quote_rows="quote_mean_acts.npy: hidden states extra_layers, same order; NaN when "
                               "reply_extras.jsonl has quote_span null",
                    versions=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                                  transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                                  numpy=np.__version__),
                    packages=sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
                    seconds=round(time.time() - t), **summary)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    names = ("rows.jsonl", "reply_mean_acts.npy", "quote_mean_acts.npy", "reply_extras.jsonl", "grid.json",
             "fingerprint_parts.json", "manifest.json")
    (out / "SHA256SUMS.txt").write_text("".join(f"{R.file_sha256(out / n)}  {n}\n" for n in names))
    print(f"done: {summary['n_rows']} rows, {summary['n_invalid']} invalid, states {summary['acts_shape']}")


if __name__ == "__main__":
    main()
