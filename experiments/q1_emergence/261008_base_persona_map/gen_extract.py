"""E3 pilot: OLMo-2-7B base writes role replies; save replies and per-layer reply activations. See NOTE.md.

Design (owner decisions 08/10/2026):
- Base model only, plain text, no chat template. Each prompt starts with one <|endoftext|>
  (the tokenizer adds none).
- One interview format, neutral speaker label "Guest". Two conditions that differ in one factor:
    desc : header + "About the guest: <third-person description>" + test question
    ex2  : the same, plus 2 example exchanges of the same role (pilot_examples_final.json)
    ex2_other : conflicting-example control: as ex2, but the 2 examples come from a donor role.
          3 donors per role, DONOR_SHIFTS places along PILOT_ORDER (always another group).
    ex2_filler : length control (owner, 09/10): as ex2, but a fixed neutral interview-notes text
          of about the same token length stands where the examples stand. No answers, no persona.
- Questions: the 40-question set used in earlier runs. desc uses all 40 for every role and
  default. In ex2 a role's 2 example questions are left out (an example never uses the tested
  question), so each role has 38 there. ex2_other uses only the 18 questions that are no role's
  example question. Main maps for ex2 and ex2_other use those 18 shared questions (owner, 08/10);
  ex2 on 38 is a check.
- 20 pilot roles x 5 description wordings. Defaults (no description, "an AI assistant",
  "a large language model") run in desc only, on all 40 questions; they have no examples.
- Sampling as Moskvoretskii et al. 2026: temperature 0.5, repetition penalty 1.1, 1 sample,
  top_k 0, top_p 1. max_new_tokens 128 (as earlier runs). Seed = batch start index, batches in
  item order with a fixed batch size (recorded; a different batch size changes the samples).
- Reply = the generated token ids before <|endoftext|> and before the token where
  "\\nInterviewer:" or "\\nAbout the guest:" starts. The ids are kept as generated (no re-tokenizing).
- Activations: a second forward pass, right-padded so positions start at 0, over prompt ids +
  kept reply ids; mean over the reply positions at every hidden state (embeddings + 32 layers),
  float16. Empty replies get NaN rows and valid=false.
- Resume only when the run fingerprint (model, settings, prompts, code hash) matches.

Run on the GPU pod from this folder's copy:
  python gen_extract.py --dry          # write prompts_check.txt and items.jsonl, no model
  python gen_extract.py --limit 40     # smoke test
  python gen_extract.py                # full pilot
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "allenai/OLMo-2-1124-7B"
REVISION = "7df9a82518afdecae4e8c026b27adccc8c1f0032"
EOS = "<|endoftext|>"
HEADER = "The following is an interview. The guest answers in their own voice."
LABEL = "Guest"
STOPS = ["\nInterviewer:", "\nAbout the guest:"]
GEN = dict(do_sample=True, temperature=0.5, top_p=1.0, top_k=0, repetition_penalty=1.1, max_new_tokens=128)
PILOT_ORDER = ["assistant", "translator", "counselor", "consultant", "chef", "detective", "bartender", "criminal",
               "toddler", "teenager", "grandparent", "refugee", "cynic", "optimist", "stoic", "narcissist",
               "pirate", "vampire", "robot", "void"]
DONOR_SHIFTS = (6, 10, 14)
FILLER = ("Interview notes: This conversation was recorded for a series of long-form interviews, and each "
          "session follows the same format. The interviewer reads the questions from a printed list, and the "
          "order of the questions is fixed in advance. The guest has not seen the questions before the session. "
          "Sessions take place in a quiet room with two chairs, a small table and a single microphone. A short "
          "break is offered after every ten questions, and water is available throughout. The recording is "
          "transcribed by hand and then checked against the audio by a second person. Small hesitations and "
          "repeated words are removed, but the wording of each answer is otherwise kept exactly as spoken. "
          "Questions that the guest chooses not to answer are marked as skipped and are not repeated later. "
          "Background information about the guest is given once, before the first question, and is not "
          "repeated in the transcript. The transcript below begins with the next question in the session.")
DEFAULT_FRAMES = [("default_none", None), ("default_ai", "an AI assistant"),
                  ("default_llm", "a large language model")]
PREFIXES = ["Please act as ", "Please be ", "Act as ", "Be ", "Take on the persona of ", "Take on the role of ",
            "Embody the role of ", "Embody ", "Respond as ", "You are "]


def third_person(line):
    """'Act as a pirate who lives by your own code.' -> 'a pirate who lives by their own code'."""
    s = line.strip().rstrip(".")
    for p in PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    else:
        raise ValueError(f"unknown description form: {line!r}")
    s = re.sub(r"\byourself\b", "themselves", s)
    s = re.sub(r"\byours\b", "theirs", s)
    s = re.sub(r"\byour\b", "their", s)
    s = re.sub(r"\byou're\b", "they're", s)
    s = re.sub(r"\byou've\b", "they've", s)
    s = re.sub(r"\byou are\b", "they are", s)
    # subject 'you' (after a relative or conjunction word) -> 'they'; any other 'you' -> 'them'
    s = re.sub(r"\b(what|who|that|when|where|whenever|wherever|as|if|because|while|how|whether|than|and|but|"
               r"until|unless|before|after|everything|anything|something)\s+you\b", r"\1 they", s)
    s = re.sub(r"\byou\b", "them", s)
    return s


def block(guest, q, answer=None):
    about = f"About the guest: {guest}\n\n" if guest else ""
    turn = f"Interviewer: {q}\n{LABEL}:" + (f" {answer}" if answer is not None else "")
    return about, turn


def render(item, examples):
    """Plain-text prompt. In ex2 the description appears once, then the 2 example turns, then the test turn."""
    about, turn = block(item["guest"], item["question"])
    if item["cond"] == "ex2_filler":
        ex = [FILLER]
    else:
        ex = [block(None, e["question"], e["answer"])[1] for e in examples] if item["cond"] != "desc" else []
    return EOS + HEADER + "\n\n" + about + "\n\n".join(ex + [turn])


def cut_ids(tok, gen, eos, pad):
    """Kept reply ids: before eos/pad, and before the token where a stop marker starts.
    Returns (ids, how it ended). A token that straddles the marker start is dropped."""
    ended = "limit"
    for k, t in enumerate(gen):
        if t in (eos, pad):
            gen, ended = gen[:k], "eos" if t == eos else "pad"
            break
    text = tok.decode(gen, skip_special_tokens=False)
    hits = [text.find(m) for m in STOPS if text.find(m) != -1]
    if hits:
        p = min(hits)
        k = 0
        while k < len(gen) and len(tok.decode(gen[:k + 1], skip_special_tokens=False)) <= p:
            k += 1
        gen, ended = gen[:k], "marker"
    return gen, ended


def build_items(repo):
    roles_dir = repo / "refs/assistant-axis/roles/instructions"
    bank = [json.loads(l)["question"] for l in open(repo / "refs/assistant-axis/extraction_questions.jsonl")]
    q40 = json.load(open(HERE / "questions_40.json"))
    ex = json.load(open(HERE / "pilot_examples_final.json"))
    desc3 = json.load(open(HERE / "descriptions_third_person.json"))
    by_role = {}
    for e in ex:
        by_role.setdefault(e["role"], []).append(e)
    assert all(len(v) == 2 for v in by_role.values()) and sorted(by_role) == sorted(PILOT_ORDER)
    n = len(PILOT_ORDER)
    donors = {r: [PILOT_ORDER[(k + d) % n] for d in DONOR_SHIFTS] for k, r in enumerate(PILOT_ORDER)}
    assert all(PILOT_ORDER.index(x) // 4 != PILOT_ORDER.index(r) // 4 for r, v in donors.items() for x in v)
    shared = [q for q in q40 if q not in {e["question_idx"] for e in ex}]
    assert len(shared) == 18, len(shared)
    items = []
    for role in sorted(by_role):
        exq = {e["question_idx"] for e in by_role[role]}
        assert exq <= set(q40)
        lines = [d["pos"] for d in json.load(open(roles_dir / f"{role}.json"))["instruction"]]
        assert len(lines) == 5
        guests = []
        for w, line in enumerate(lines):  # reviewed file is the source; it must match the original line
            rec = desc3[f"{role}/{w}"]
            assert rec["original"] == line, (role, w)
            guests.append(rec["third_person"])
        plan = [("desc", None), ("ex2", role)] + [("ex2_other", d) for d in donors[role]] + [("ex2_filler", None)]
        for cond, ex_role in plan:
            qs = shared if cond in ("ex2_other", "ex2_filler") else [q for q in q40 if not (cond == "ex2" and q in exq)]
            for w, line in enumerate(lines):
                for qi in qs:
                    items.append(dict(role=role, cond=cond, wording=w, role_line=line, guest=guests[w],
                                      example_role=ex_role, question_idx=qi, question=bank[qi],
                                      shared_q=qi in shared))
    for role, guest in DEFAULT_FRAMES:
        for qi in q40:
            items.append(dict(role=role, cond="desc", wording=0, role_line=None, guest=guest,
                              example_role=None, question_idx=qi, question=bank[qi], shared_q=qi in shared))
    for n, it in enumerate(items):
        it["i"] = n
        exs = by_role[it["example_role"]] if it["example_role"] else []
        it["prompt"] = render(it, exs)
        assert it["question"] not in {e["question"] for e in exs}
        if it["cond"] == "ex2_other":
            assert it["example_role"] != it["role"] and it["shared_q"]
    return items


def pad_batch(seqs, pad, left):
    import torch
    P = max(map(len, seqs))
    ids = torch.full((len(seqs), P), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), P), dtype=torch.long)
    for j, x in enumerate(seqs):
        sl = slice(P - len(x), P) if left else slice(0, len(x))
        ids[j, sl] = torch.tensor(x)
        mask[j, sl] = 1
    return ids, mask


SEED_OFFSET = 0  # set by --seed_offset; seed = batch start index + offset


def fingerprint(items, batch_size):
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps([MODEL, REVISION, GEN, HEADER, LABEL, STOPS, batch_size, SEED_OFFSET], sort_keys=True).encode())
    h.update(Path(__file__).read_bytes())
    for it in items:
        h.update(f"{it['i']}\t{it['prompt']}\n".encode())
    return h.hexdigest()


def read_done(part):
    """Ids in the reply part file. A truncated last line (crash during append) is cut off."""
    if not part.exists():
        return set()
    lines = part.read_bytes().split(b"\n")
    good, done = [], set()
    for k, l in enumerate(lines):
        if not l.strip():
            continue
        try:
            done.add(json.loads(l)["i"])
            good.append(l)
        except json.JSONDecodeError:
            assert k == len(lines) - 1 or all(not x.strip() for x in lines[k + 1:]), "corrupt line inside the part file"
            part.write_bytes(b"\n".join(good) + b"\n")
            print(f"cut a truncated last line from {part.name}", flush=True)
            break
    return done


def claim_out_dir(out_dir, items, batch_size):
    """Fingerprint check before anything is written. A dir with outputs but no fingerprint is refused."""
    fp = fingerprint(items, batch_size)
    fp_file = out_dir / "fingerprint.txt"
    if fp_file.exists():
        if fp_file.read_text().strip() != fp:
            raise SystemExit(f"{out_dir} holds a run with other settings, prompts or code; use a new --out")
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} has files but no fingerprint; use a new --out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_file.write_text(fp)


def run(items, out_dir, batch_size):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16).cuda().eval()
    pad, eos = tok.pad_token_id, tok.eos_token_id
    n_hs = model.config.num_hidden_layers + 1
    part = out_dir / "replies.partial.jsonl"
    acts_dir = out_dir / "acts_parts"
    acts_dir.mkdir(exist_ok=True)
    done = read_done(part)
    for s in range(0, len(items), batch_size):  # a batch counts as done only with its activation part
        ids_b = [r["i"] for r in items[s:s + batch_size]]
        if all(i in done for i in ids_b):
            f = acts_dir / f"{s:06d}.npz"
            assert f.exists() and np.load(f)["ids"].tolist() == ids_b, f"activation part missing for batch {s}"
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        if all(r["i"] in done for r in chunk):
            continue
        p_ids = [tok(r["prompt"], add_special_tokens=False).input_ids for r in chunk]
        assert all(x[0] == eos and x.count(eos) == 1 for x in p_ids)
        ids, mask = pad_batch(p_ids, pad, left=True)
        torch.manual_seed(s + SEED_OFFSET)
        with torch.no_grad():
            out = model.generate(ids.cuda(), attention_mask=mask.cuda(), pad_token_id=pad, eos_token_id=eos, **GEN)
        P = ids.shape[1]
        full = []
        for j, r in enumerate(chunk):
            gen = out[j, P:].tolist()
            kept, r["ended_by"] = cut_ids(tok, gen, eos, pad)
            r["raw_ids"] = gen
            r["raw"] = tok.decode(gen, skip_special_tokens=False)
            r["reply"] = tok.decode(kept, skip_special_tokens=False)  # exact decode of reply_ids
            r["reply_ids"] = kept
            r["prompt_tokens"], r["reply_tokens"] = len(p_ids[j]), len(kept)
            r["valid"] = bool(kept) and bool(r["reply"].strip())
            full.append(p_ids[j] + kept)
        ids2, mask2 = pad_batch(full, pad, left=False)  # right padding: positions start at 0
        with torch.no_grad():
            hs = model(ids2.cuda(), attention_mask=mask2.cuda(), output_hidden_states=True).hidden_states
        assert len(hs) == n_hs
        acts = np.full((len(chunk), n_hs, hs[0].shape[-1]), np.nan, dtype=np.float16)
        for j, r in enumerate(chunk):
            if r["valid"]:
                a, b = r["prompt_tokens"], r["prompt_tokens"] + r["reply_tokens"]
                assert ids2[j, a:b].tolist() == r["reply_ids"]
                acts[j] = torch.stack([h[j, a:b].float().mean(0) for h in hs]).cpu().numpy().astype(np.float16)
        np.savez(acts_dir / f"{s:06d}.npz", acts=acts, ids=np.array([r["i"] for r in chunk]))
        with open(part, "a") as f:
            f.write("".join(json.dumps({k: v for k, v in r.items() if k != "prompt"}, ensure_ascii=False) + "\n"
                            for r in chunk))
            f.flush()
            os.fsync(f.fileno())
        print(f"{s + len(chunk)}/{len(items)}", flush=True)
    return n_hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parents[2]), help="path holding refs/assistant-axis")
    ap.add_argument("--out", default=str(HERE / "run"))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: a fixed random subset")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--replicate", action="store_true",
                    help="second-seed replication: desc and ex2 on the 18 shared questions only")
    args = ap.parse_args()
    global SEED_OFFSET
    SEED_OFFSET = args.seed_offset
    items = build_items(Path(args.repo))
    if args.replicate:
        assert SEED_OFFSET != 0, "a replication needs --seed_offset"
        items = [it for it in items if it["cond"] in ("desc", "ex2") and it["shared_q"]]
        # 20 roles x 5 wordings x 2 conditions x 18 questions, plus the 3 default frames x 18 in desc (axis)
        assert len(items) == 20 * 5 * 2 * 18 + 3 * 18, len(items)
    if args.limit:
        items = sorted(random.Random(0).sample(items, args.limit), key=lambda r: r["i"])
    out = Path(args.out)
    if args.dry:
        if (out / "fingerprint.txt").exists():
            raise SystemExit(f"{out} is a run directory; use another --out for --dry")
    else:
        claim_out_dir(out, items, args.batch_size)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    show = [next(i for i in items if i["role"] == r and i["cond"] == c)
            for r, c in (("pirate", "desc"), ("pirate", "ex2"), ("pirate", "ex2_other"), ("pirate", "ex2_filler"),
                         ("default_none", "desc"), ("default_ai", "desc"))
            if any(i["role"] == r and i["cond"] == c for i in items)]
    (out / "prompts_check.txt").write_text("".join(f"===== {i['role']} / {i['cond']} / w{i['wording']} =====\n"
                                                   f"{i['prompt']}<<END>>\n\n" for i in show))
    print(len(items), "items")
    if args.dry:
        return
    t = time.time()
    n_hs = run(items, out, args.batch_size)
    import numpy as np
    rows = {json.loads(l)["i"]: json.loads(l) for l in open(out / "replies.partial.jsonl")}
    assert sorted(rows) == sorted(it["i"] for it in items), "missing rows"
    from numpy.lib.format import open_memmap
    valid = np.array([rows[it["i"]]["valid"] for it in items])
    want = [it["i"] for it in items]
    acts = None
    pos = 0
    for f in sorted((out / "acts_parts").glob("*.npz")):  # one part at a time
        z = np.load(f)
        a, ids_p = z["acts"], z["ids"].tolist()
        assert ids_p == want[pos:pos + len(ids_p)], f"activation rows out of item order in {f.name}"
        v = valid[pos:pos + len(ids_p)]
        assert np.isfinite(a[v]).all() and np.isnan(a[~v]).all(), f.name
        if acts is None:
            acts = open_memmap(out / "reply_mean_acts.npy", mode="w+", dtype=np.float16,
                               shape=(len(items),) + a.shape[1:])
        acts[pos:pos + len(ids_p)] = a
        pos += len(ids_p)
    assert pos == len(items) and acts.shape[1] == n_hs
    acts.flush()
    with open(out / "replies.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(rows[it["i"]], ensure_ascii=False) + "\n")
    import hashlib, platform, sys, torch, transformers, tokenizers
    json.dump(dict(model=MODEL, revision=REVISION, donor_shifts=DONOR_SHIFTS, pilot_order=PILOT_ORDER, filler=FILLER, tokenizer_revision=REVISION, gen=GEN, header=HEADER,
                   label=LABEL, stops=STOPS, batch_size=args.batch_size,
                   seed_rule="torch.manual_seed(batch start index + seed_offset); batches in item order",
                   seed_offset=SEED_OFFSET, replicate=args.replicate,
                   n_items=len(items), limit=args.limit, n_hidden_states=n_hs,
                   n_invalid=int((~valid).sum()), invalid_ids=[it["i"] for it, v in zip(items, valid) if not v],
                   act_rows="same order as replies.jsonl; rows with valid=false are NaN",
                   fingerprint=(out / "fingerprint.txt").read_text().strip(),
                   code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   versions=dict(python=platform.python_version(), torch=torch.__version__,
                                 cuda=torch.version.cuda, transformers=transformers.__version__,
                                 tokenizers=tokenizers.__version__, numpy=np.__version__),
                   seconds=round(time.time() - t)),
              open(out / "manifest.json", "w"), indent=1)
    print(f"done: {len(items)} replies, acts {acts.shape}")


if __name__ == "__main__":
    main()
