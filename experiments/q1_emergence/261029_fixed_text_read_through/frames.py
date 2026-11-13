"""Pod C frames: the exact ids every checkpoint reads. Built once on the Mac, fed unchanged to every checkpoint.

Design: ../261029_stage_persona_plan/DESIGN.md (pod C) and THRESHOLDS.md here ("Data", G1, "Frame read").
For each text source (pod_a, e3_desc, e3_ex2):
  F1  <|endoftext|> + the E3 interview header + the question + "Guest:" + the reply ids, pooled over the
      reply. No description. A reply whose first token does not start with whitespace (Instruct replies)
      gets "\\n" after "Guest:", so the joint ids stay canonical; E3 replies keep their own leading space.
  F2  <|endoftext|> + the reply ids, pooled over the reply.
  F3  F1 with one fixed neutral reply (E3 context control c3) on every row of the complete grid.
  F4  the F1 pass pooled over the first 32 reply ids, only for valid replies with at least 32 kept ids.
  F0  pod_a only: pod A's chat prompt ids + kept ids, pooled over the reply (G3 reconstruction at Instruct).
Invalid replies get no F0, F1 or F2 sequence (NaN states later); F3 covers them.
Also: 50 fixed halves per source, the F4 support check on them, and frame_read.md (10 rows per frame).

  python frames.py --pod_a ../261029_instruct_persona_map/run_full --out frames_full
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from manifold_persona import rows as R

import replies as P

HERE = Path(__file__).resolve().parent
HEADER = "The following is an interview. The guest answers in their own voice."
F1_PREFIX = "<|endoftext|>" + HEADER + "\n\nInterviewer: {question}\nGuest:"
F2_PREFIX = "<|endoftext|>"
NEUTRAL = (" That is a good question. I think the answer depends on the situation, and I would want to think"
           " about it carefully before I say more.")
F4_TOKENS = 32
F4_MIN_ROLE_SHARE = 0.5
N_HALVES, HALF_SEED = 50, 0
N_READ = 10
SOURCES = ("pod_a", "e3_desc", "e3_ex2")
FRAMES = {"F0": ("reply",), "F1": ("reply", "first32"), "F2": ("reply",), "F3": ("neutral",)}
F4 = ("F1", "first32")


def question_bank():
    return [json.loads(l)["question"] for l in open(P.REPO / "refs/assistant-axis/extraction_questions.jsonl")]


def separator(tok, reply_ids):
    return "" if not reply_ids or tok.decode(reply_ids[:1])[:1].isspace() else "\n"


def build(recs, tok, bank):
    """{frame: (sequences, {span name: [n, 2] int array})} for one source, rows in record order."""
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    neutral = enc(NEUTRAL)
    f1_pre, out = {}, {f: ([], {s: [] for s in names}) for f, names in FRAMES.items()}
    for r in recs:
        ids = r["reply_ids"] if r["valid"] else []
        sep = separator(tok, ids)
        key = (r["question"], sep)
        if key not in f1_pre:
            f1_pre[key] = enc(F1_PREFIX.format(question=bank[r["question"]]) + sep)
        pre3 = f1_pre.setdefault((r["question"], ""), enc(F1_PREFIX.format(question=bank[r["question"]])))
        seqs = dict(F0=(r["prompt_ids"] + ids) if ids and "prompt_ids" in r else [],
                    F1=f1_pre[key] + ids if ids else [], F2=enc(F2_PREFIX) + ids if ids else [],
                    F3=pre3 + neutral)
        starts = dict(F0=len(r.get("prompt_ids", [])), F1=len(f1_pre[key]), F2=1, F3=len(pre3))
        for f, (s_list, spans) in out.items():
            s_list.append(seqs[f])
            a, n = starts[f], (len(neutral) if f == "F3" else len(ids))
            empty = [0, 0]
            for name in spans:
                if not seqs[f]:
                    spans[name].append(empty)
                elif name == "first32":
                    spans[name].append([a, a + F4_TOKENS] if n >= F4_TOKENS else empty)
                else:
                    spans[name].append([a, a + n])
    return {f: (s, {k: np.array(v, np.int32).reshape(-1, 2) for k, v in spans.items()}) for f, (s, spans) in out.items()}


def row_hashes(seqs):
    """G1: one sha256 per row over the ids (the mask is ones over them and positions run from 0)."""
    return np.array([np.frombuffer(hashlib.sha256(np.asarray(s, np.int64).tobytes()).digest(), np.uint8)
                     for s in seqs], np.uint8).reshape(-1, 32)


def make_halves(questions, n=N_HALVES, seed=HALF_SEED, n_wordings=5):
    """Half A: half the questions x 2 or 3 of the 5 wordings; half B: the other questions x other wordings.
    Masks over the sorted question ids and the wording ids (the E3 rule)."""
    qs, ws = np.array(sorted(questions)), np.arange(n_wordings)
    rng, out = np.random.default_rng(seed), []
    for _ in range(n):
        qa = rng.choice(qs, len(qs) // 2, replace=False)
        wa = rng.choice(ws, rng.choice([n_wordings // 2, n_wordings - n_wordings // 2]), replace=False)
        out.append((np.isin(qs, qa), np.isin(ws, wa)))
    return out


def common_questions(cover, qmask, wmask):
    """Questions in a half where every role has at least one kept row. cover: bool [roles, questions, wordings]."""
    return np.where(qmask & cover[:, :, wmask].any(2).all(0))[0]


def f4_support(recs, mask, roles, halves):
    """F4 availability (THRESHOLDS.md): every role keeps at least half its rows, and every half of every fixed
    split has at least one common question."""
    qs = sorted({r["question"] for r in recs})
    ri, qi = {r: k for k, r in enumerate(roles)}, {q: k for k, q in enumerate(qs)}
    cover, kept, total = np.zeros((len(roles), len(qs), 5), bool), np.zeros(len(roles)), np.zeros(len(roles))
    for r, m in zip(recs, mask):
        if r["role"] in ri:
            total[ri[r["role"]]] += 1
            kept[ri[r["role"]]] += m
            cover[ri[r["role"]], qi[r["question"]], r["wording"]] |= bool(m)
    share = kept / total
    n_common = [(len(common_questions(cover, qa, wa)), len(common_questions(cover, ~qa, ~wa))) for qa, wa in halves]
    return dict(available=bool(share.min() >= F4_MIN_ROLE_SHARE and min(min(c) for c in n_common) > 0),
                min_role_share=round(float(share.min()), 4), role_share=dict(zip(roles, np.round(share, 4).tolist())),
                halves_without_common_question=sum(min(c) == 0 for c in n_common),
                min_common_questions=int(min(min(c) for c in n_common)))


def frame_read(tok, source, frames, n=N_READ):
    """Ten rows per frame with the pooled span between ⟦ and ⟧."""
    out = []
    for f, (seqs, spans) in frames.items():
        rows = [j for j, s in enumerate(seqs) if s]
        for name, sp in spans.items():
            eligible = [j for j in rows if sp[j, 1] > sp[j, 0]]
            pick = sorted(random.Random(0).sample(eligible, min(n, len(eligible))))
            out.append(f"## {source} {f} {name}\n")
            for j in pick:
                s, (a, b) = seqs[j], sp[j]
                out.append(f"```\n{tok.decode(s[:a])}⟦{tok.decode(s[a:b])}⟧{tok.decode(s[b:])}\n```\n")
    return "\n".join(out)


def canonical_rate(tok, seqs, spans):
    """Share of rows whose ids equal the tokenizer's own encoding of their text (a diagnostic, not a gate)."""
    rows = [j for j, s in enumerate(seqs) if s]
    ok = sum(tok.encode(tok.decode(seqs[j][1:]), add_special_tokens=False) == seqs[j][1:] for j in rows)
    return round(ok / len(rows), 4) if rows else None


def save(out, built, info, tok):
    """frames.npz (ids, offsets, spans, row hashes per source and frame) and frames.json."""
    arrays, frames_info = {}, {}
    for src, (recs, frames, sinfo) in built.items():
        for f, (seqs, spans) in frames.items():
            if not any(seqs):
                continue
            p = f"{src}.{f}"
            arrays[f"{p}.ids"] = np.concatenate([np.asarray(s, np.int32) for s in seqs if s])
            arrays[f"{p}.offsets"] = np.cumsum([0] + [len(s) for s in seqs]).astype(np.int64)
            arrays[f"{p}.hash"] = row_hashes(seqs)
            for name, sp in spans.items():
                arrays[f"{p}.span.{name}"] = sp
            frames_info[p] = dict(rows=len(seqs), with_ids=sum(1 for s in seqs if s), spans=list(spans),
                                  max_len=max(len(s) for s in seqs), canonical_rate=canonical_rate(tok, seqs, spans),
                                  digest=hashlib.sha256(arrays[f"{p}.hash"].tobytes()).hexdigest())
    np.savez(out / "frames.npz", **arrays)
    (out / "frames.json").write_text(json.dumps(dict(info, frames=frames_info), indent=1) + "\n")


def load(out):
    """frames.json and a reader: seqs(source, frame) -> list of id lists; spans(source, frame, name)."""
    z = np.load(out / "frames.npz")
    info = json.loads((out / "frames.json").read_text())

    def seqs(src, f):
        ids, off = z[f"{src}.{f}.ids"], z[f"{src}.{f}.offsets"]
        return [ids[off[j]:off[j + 1]].tolist() for j in range(len(off) - 1)]
    return info, z, seqs


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod_a", required=True, type=Path)
    ap.add_argument("--e3", default=P.E3_RUN, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else HERE / a.out
    out.mkdir(exist_ok=False)
    ck = P.CK["instruct"]
    tok = AutoTokenizer.from_pretrained(ck["repo"], revision=ck["revision"])
    pod_a, a_info = P.read_pod_a(a.pod_a, tok)
    e3, e_info = P.read_e3(a.e3, tok)
    bank, built, f4, src_info = question_bank(), {}, {}, {}
    for src in SOURCES:
        recs = [r for r in pod_a + e3 if r["source"] == src]
        roles = [r for r in (a_info if src == "pod_a" else e_info)["roles"] if r != "default"]
        frames = build(recs, tok, bank)
        halves = make_halves({r["question"] for r in recs})
        f4[src] = f4_support(recs, frames[F4[0]][1][F4[1]][:, 1] > 0, roles, halves)
        built[src] = (recs, frames, None)
        src_info[src] = dict(keys=[r["key"] for r in recs], roles=roles, questions=sorted({r["question"] for r in recs}),
                             separators={repr(s): sum(1 for r in recs if r["valid"] and separator(tok, r["reply_ids"]) == s)
                                         for s in ("", "\n")},
                             halves=[(qa.tolist(), wa.tolist()) for qa, wa in halves])
    from huggingface_hub import hf_hub_download
    tok_sha = R.file_sha256(hf_hub_download(ck["repo"], "tokenizer.json", revision=ck["revision"]))
    if tok_sha != json.loads(P.C.MANIFEST.read_text())["entries"][0]["tokenizer_files"]["tokenizer.json"]:
        raise SystemExit("tokenizer.json differs from checkpoints.json")
    info = dict(header=HEADER, f1_prefix=F1_PREFIX, f2_prefix=F2_PREFIX, neutral=NEUTRAL, f4_tokens=F4_TOKENS,
                tokenizer=dict(repo=ck["repo"], revision=ck["revision"], tokenizer_json_sha256=tok_sha),
                inputs=dict(pod_a=a_info, e3={k: v for k, v in e_info.items() if k != "questions"}),
                f4_support=f4, sources=src_info, code=dict(frames_py=R.file_sha256(Path(__file__)),
                                                            replies_py=R.file_sha256(HERE / "replies.py")))
    save(out, built, info, tok)
    (out / "frame_read.md").write_text("# Frame read: 10 rows per frame, pooled span in ⟦ ⟧ (read before any metric)\n\n"
                                       + "\n".join(frame_read(tok, s, built[s][1]) for s in SOURCES))
    names = ("frames.npz", "frames.json", "frame_read.md")
    (out / "SHA256SUMS.txt").write_text("".join(f"{R.file_sha256(out / n)}  {n}\n" for n in names))
    print(json.dumps(dict(f4_support={s: {k: v for k, v in d.items() if k != "role_share"} for s, d in f4.items()},
                          frames=json.loads((out / "frames.json").read_text())["frames"]), indent=1))


if __name__ == "__main__":
    main()
