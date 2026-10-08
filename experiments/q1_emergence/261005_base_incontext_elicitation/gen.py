"""E0: does OLMo-2-1B base play a character from a plain-text in-context interview prompt? See NOTE.md.

Generates replies only (no activations). Base and final get the same plain-text prompt under 8 conditions:
example set 1/2/3 or zero-shot, crossed with speaker label = role name or neutral "Guest". Final also gets
its native chat template with the original role line.

Run from the repo root:
  .venv/bin/python experiments/q1_emergence/261005_base_incontext_elicitation/gen.py --dry
  .venv/bin/python experiments/q1_emergence/261005_base_incontext_elicitation/gen.py
"""
import argparse, json, random, re, sys, time
from pathlib import Path

A = Path(__file__).resolve().parent
REPO = A.parents[2]
sys.path[:0] = [str(REPO / "src"), str(A)]
from manifold_persona.config import ROLE_INSTRUCTIONS_DIR
from manifold_persona.prompts_roles import list_roles, load_questions
from examples import SETS

MODELS = {"base": "allenai/OLMo-2-0425-1B", "final": "allenai/OLMo-2-0425-1B-Instruct"}
PLAIN = [f"{e}_{l}" for e in ("set1", "set2", "set3", "zero") for l in ("role", "neutral")]
CONDS = {"base": PLAIN, "final": PLAIN + ["native"]}
EOS = "<|endoftext|>"
HEADER = "The following are interviews. Each guest answers in their own voice."
STOPS = ["\nInterviewer:", "\nAbout the guest:"]
TEST_Q = [26, 84, 129, 158, 215, 225]
N_SAMPLES, TEMP, MAX_NEW = 1, 0.7, 160
LABELS = {"ancient": "Ancient intelligence", "devils_advocate": "Devil's advocate", "hive": "Hive mind",
          "void": "Void entity", "pirate": "Pirate captain"}
DEFAULTS = [("default_none", None, "Guest"), ("default_person", "a person", "Respondent"),
            ("default_ai", "an AI assistant", "AI assistant")]


def third_person(line):
    """'You are a pirate who ... your ...' -> 'a pirate who ... their ...'."""
    assert line.startswith("You are "), line
    s = line[len("You are "):].rstrip(".")
    s = re.sub(r"\byour\b", "their", s)
    s = re.sub(r"\bmakes you\b", "makes them", s)
    s = re.sub(r"\byou\b", "them", s)
    return s


def sample_items(n_chars=60, n_q=2, seed=0):
    """Same 60 characters as the 02/10 plan (seed 0); each gets 2 of the 6 test questions. Defaults get all 6."""
    rng = random.Random(seed)
    roles = sorted(rng.sample([r for r in list_roles() if r != "default"], n_chars))
    bank = load_questions()
    items = []
    for role in roles:
        line = json.load(open(ROLE_INSTRUCTIONS_DIR / f"{role}.json"))["instruction"][0]["pos"]
        label = LABELS.get(role, role.replace("_", " ").capitalize())
        for qi in sorted(rng.sample(TEST_Q, n_q)):
            items.append(dict(role=role, role_line=line, guest=third_person(line), label=label,
                              question_idx=qi, question=bank[qi]["question"]))
    for role, guest, label in DEFAULTS:
        for qi in TEST_Q:
            items.append(dict(role=role, role_line=None, guest=guest, label=label,
                              question_idx=qi, question=bank[qi]["question"]))
    ex_q = {q for s in SETS.values() for _, _, q, _, _ in s}
    assert not ex_q & set(TEST_Q), "example question overlaps the test set"
    assert not {e[1].lower() for s in SETS.values() for e in s} & set(list_roles()), "example character in the 276 list"
    return items


def block(guest, q, label, answer=None):
    about = f"About the guest: {guest}\n" if guest else ""
    return f"{about}Interviewer: {q}\n{label}:" + (f" {answer}" if answer is not None else "")


def render(cond, it, tok):
    if cond == "native":
        msgs = ([{"role": "system", "content": it["role_line"]}] if it["role_line"] else []) + \
               [{"role": "user", "content": it["question"]}]
        if it["role"] == "default_ai":
            msgs = [{"role": "system", "content": "You are an AI assistant."}] + msgs
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ex_name, lab = cond.split("_")
    name = (lambda l: l) if lab == "role" else (lambda l: "Guest")
    ex = [] if ex_name == "zero" else [block(g, q, name(l), a) for g, l, _, q, a in SETS[int(ex_name[-1])]]
    return EOS + HEADER + "\n\n" + "\n\n".join(ex + [block(it["guest"], it["question"], name(it["label"]))])


def cut(cond, raw):
    """Reply = generated text up to the first stop marker or eos. Returns (reply, how it ended)."""
    stops = (["<|user|>", "<|system|>"] if cond == "native" else STOPS) + [EOS]
    ends = [(raw.find(m), m) for m in stops if raw.find(m) != -1]
    if not ends:
        return raw.strip(), "limit"
    p, m = min(ends)
    return raw[:p].strip(), "eos" if m == EOS else "marker"


def generate(stage, rows, part, batch_size=16):
    """Generate in batches; append each finished batch to `part` so a stopped run resumes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[stage])
    model = AutoModelForCausalLM.from_pretrained(MODELS[stage], torch_dtype=torch.float16).to("mps").eval()
    assert model.dtype == torch.float16, model.dtype
    pad = tok.pad_token_id
    done = {json.loads(l)["i"] for l in open(part)} if part.exists() else set()
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        if all(r["i"] in done for r in chunk):
            continue
        ids = [tok(r["prompt"], add_special_tokens=False).input_ids for r in chunk]
        P = max(map(len, ids))
        inp = torch.full((len(ids), P), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), P), dtype=torch.long)
        for j, x in enumerate(ids):
            inp[j, P - len(x):] = torch.tensor(x)
            mask[j, P - len(x):] = 1
        torch.manual_seed(s)
        with torch.no_grad():
            out = model.generate(inp.to("mps"), attention_mask=mask.to("mps"), max_new_tokens=MAX_NEW,
                                 do_sample=True, temperature=TEMP, top_p=1.0, top_k=0,
                                 pad_token_id=pad, eos_token_id=tok.eos_token_id)
        for j, r in enumerate(chunk):
            gen = [t for t in out[j, P:].tolist() if t != pad]
            r["raw"] = tok.decode(gen, skip_special_tokens=False)
            r["n_gen_tokens"], r["prompt_tokens"] = len(gen), len(ids[j])
            r["reply"], r["ended_by"] = cut(r["cond"], r["raw"])
        with open(part, "a") as f:
            for r in chunk:
                f.write(json.dumps({k: v for k, v in r.items() if k != "prompt"}, ensure_ascii=False) + "\n")
        print(f"{stage} {s + len(chunk)}/{len(rows)}", flush=True)
    del model
    torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=list(MODELS))
    ap.add_argument("--dry", action="store_true", help="write items and prompt examples only")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    items = sample_items()
    json.dump(items, open(A / "items.json", "w"), indent=1)
    toks = {st: AutoTokenizer.from_pretrained(m) for st, m in MODELS.items()}
    shown = [next(i for i in items if i["role"] == r) for r in ("pirate", "default_none", "default_ai")]
    (A / "prompt_examples.txt").write_text("".join(
        f"===== {st} / {c} / {it['role']} =====\n{render(c, it, toks[st])}<<END>>\n\n"
        for st in MODELS for c in CONDS[st] for it in shown))
    (A / "third_person_check.txt").write_text("".join(
        f"{i['role']}\n  {i['role_line']}\n  Guest: {i['guest']}\n  {i['label']}:\n" for i in items[::2] if i["role_line"]))
    print(len(items), "items;", sum(len(CONDS[s]) for s in MODELS) * len(items) * N_SAMPLES, "replies planned")
    if args.dry:
        return
    for stage in args.stages:
        out = A / f"replies_{stage}.jsonl"
        if out.exists():
            print("skip", stage, "(exists)"); continue
        rows = [dict(i=n, stage=stage, cond=c, sample=k, **it, prompt=render(c, it, toks[stage]))
                for n, (c, it, k) in enumerate((c, it, k) for c in CONDS[stage] for it in items for k in range(N_SAMPLES))]
        part = A / f"replies_{stage}.partial.jsonl"
        t = time.time()
        generate(stage, rows, part)
        got = {json.loads(l)["i"]: l for l in open(part)}
        assert sorted(got) == list(range(len(rows))), "missing rows"
        out.write_text("".join(got[n] for n in range(len(rows))))
        part.unlink()
        print(f"{stage}: {len(rows)} replies in {time.time() - t:.0f}s -> {out.name}")


if __name__ == "__main__":
    main()
