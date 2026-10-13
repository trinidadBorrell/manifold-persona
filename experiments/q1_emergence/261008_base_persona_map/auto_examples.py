"""Choose pilot in-context examples with a fixed rule, no hand picking.

A reply can be an example for its role only if:
  1. the blind-naming judge (E0 prompt, GPT-5.6-Sol) puts the role in its top 5 of 276
     from the reply alone ("default" also counts for the assistant role);
  2. it shares no run of 5+ words with any of the role's 5 descriptions;
  3. it has 30-70 words after cutting to whole sentences, and no list formatting
     or AI self-reference (except for the assistant role);
  4. (v2) it does not name its own role (NAME_TERMS, fixed before judging);
  5. (v2, added after judging) no anonymiser placeholder such as <PRESIDIO_...> and no inline "1)" list.

Pool: per role, up to N_POOL OLMo-2-1B-Instruct (no few-shot) and Qwen2.5-3B-Instruct
replies that pass rules 2-4, sampled with a fixed seed. Choice per role: the 2
passing replies with the best judge rank (top-1 first), on different questions,
OLMo before Qwen on ties.

v1 (rules 1-3, 8+4 per role) is kept in v1/.

    .venv/bin/python .../auto_examples.py prep v2   # pool + blind judge batches
    bash .../run_judge.sh v2 N                      # judge
    .venv/bin/python .../auto_examples.py pick v2   # apply rule 1, write the choice
"""
import json
import re
import secrets
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLMO = ("brend0nc/manifold-persona-olmo1b-stages", "rlvr_zeroshot/metadata.parquet",
        "22250ccc62345c7ae84ba1cb86e52100ad83ed54")
QWEN = REPO / "data/embeddings_roles_resp_40q/metadata.parquet"
ROLES_DIR = REPO / "refs/assistant-axis/roles/instructions"
E0 = REPO / "experiments/q1_emergence/261005_base_incontext_elicitation"
ROLES = ["assistant", "tutor", "analyst", "librarian", "chef", "detective", "bartender", "criminal",
         "toddler", "teenager", "grandparent", "refugee", "cynic", "optimist", "stoic", "narcissist",
         "pirate", "vampire", "robot", "void"]
# v3: replacements for tutor, analyst, librarian (v2 lacked good examples). Candidates are the
# next 10 unused roles from the start of Lu et al.'s role_list.json; the first 3 in this order
# with 2 examples ranked 1st by the judge are used.
REPLACE_OUT = ["tutor", "analyst", "librarian"]
REPLACE_CANDIDATES = ["translator", "editor", "counselor", "researcher", "programmer",
                      "consultant", "teacher", "writer", "coach", "interviewer"]
N_POOL = {"O": 20, "Q": 10}
MIN_W, MAX_W, NGRAM = 30, 70, 5
SEED = 0
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else "v2")
if OUT.name == "v3":
    ROLES = REPLACE_CANDIDATES

# words that name the role or its description noun; matched as whole words, plural allowed
NAME_TERMS = {
    "assistant": ["assistant"], "tutor": ["tutor", "tutoring"], "analyst": ["analyst"],
    "librarian": ["librarian"], "chef": ["chef", "cook"], "detective": ["detective", "investigator"],
    "bartender": ["bartender", "barkeep", "mixologist"], "criminal": ["criminal", "outlaw"],
    "toddler": ["toddler"], "teenager": ["teenager", "teen", "teenage", "adolescent"],
    "grandparent": ["grandparent", "grandmother", "grandfather", "grandma", "grandpa"],
    "refugee": ["refugee", "displaced", "displacement", "asylum"],
    "cynic": ["cynic", "cynical"], "optimist": ["optimist", "optimistic"], "stoic": ["stoic", "stoicism"],
    "narcissist": ["narcissist", "narcissistic"], "pirate": ["pirate"], "vampire": ["vampire"],
    "robot": ["robot", "robotic"], "void": ["void", "between worlds", "between reality"],
    "translator": ["translator", "interpreter"], "editor": ["editor"],
    "counselor": ["counselor", "counsellor"], "researcher": ["researcher"],
    "programmer": ["programmer", "developer", "coder"], "consultant": ["consultant"],
    "teacher": ["teacher", "educator"], "writer": ["writer", "author"], "coach": ["coach"],
    "interviewer": ["interviewer"],
}

LIST_RE = re.compile(r"(^\s*(\d+\.|-|\*)\s)|\*\*|#", re.M)
FORMAT_RE = re.compile(r"<[A-Z_]{4,}>|(^|\s)\d\)\s")  # anonymiser placeholders, inline "1)" lists
AI_RE = re.compile(r"as an ai|language model|i'm an ai|i am an ai|ai assistant", re.I)


def cut(text):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(" ".join(out + [s]).split()) > MAX_W:
            break
        out.append(s)
    return " ".join(out)


def words(t):
    return re.findall(r"[a-z']+", t.lower())


def ngrams(t, n=NGRAM):
    w = words(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def text_ok(role, raw, descs):
    if LIST_RE.search(raw[:500]) or (role != "assistant" and AI_RE.search(raw)):
        return None
    a = cut(raw)
    if not (MIN_W <= len(a.split()) <= MAX_W) or a[-1] not in ".!?" or "\n" in a or FORMAT_RE.search(a):
        return None
    if any(ngrams(a) & ngrams(d) for d in descs):
        return None
    names = "|".join(re.escape(t) for t in NAME_TERMS[role])
    if re.search(rf"\b({names})(s|es)?\b", a, re.I):
        return None
    return a


def prep():
    olmo = pd.read_parquet(hf_hub_download(OLMO[0], OLMO[1], repo_type="dataset", revision=OLMO[2]))
    src = {"O": olmo, "Q": pd.read_parquet(QWEN)}
    rng = np.random.default_rng(SEED)
    pool, ids = [], set()
    for role in ROLES:
        descs = [d["pos"] for d in json.load(open(ROLES_DIR / f"{role}.json"))["instruction"]]
        for s, df in src.items():
            ok = []
            for _, x in df[df.role == role].iterrows():
                a = text_ok(role, str(x.response), descs)
                if a:
                    ok.append({"role": role, "source": s, "instruction_idx": int(x.instruction_idx),
                               "question_idx": int(x.question_idx), "question": x.question, "answer": a})
            for k in rng.permutation(len(ok))[:N_POOL[s]]:
                while (i := secrets.token_hex(3)) in ids:
                    pass
                ids.add(i)
                pool.append({"id": i, **ok[k]})
            print(f"{role:12s} {s}: {len(ok):3d} pass rules 2-4, {min(len(ok), N_POOL[s])} in pool")
    order = rng.permutation(len(pool))
    (OUT / "batches").mkdir(exist_ok=True)
    (OUT / "answers").mkdir(exist_ok=True)
    shutil.copy(E0 / "candidates.txt", OUT / "candidates.txt")
    shutil.copy(E0 / "prompt_judge.txt", HERE / "prompt_judge.txt")
    chunks = np.array_split(order, int(np.ceil(len(pool) / 125)))
    for b, chunk in enumerate(chunks, 1):
        parts = [f"### {pool[k]['id']}\n{pool[k]['answer']}\n" for k in chunk]
        (OUT / "batches" / f"batch_{b}.md").write_text(f"# Batch {b}: {len(chunk)} items\n\n" + "\n".join(parts))
    json.dump(pool, open(OUT / "pool.json", "w"), indent=1)
    print(len(pool), "replies in pool,", len(chunks), "batches")


def pick():
    pool = json.load(open(OUT / "pool.json"))
    ans = {}
    for f in sorted((OUT / "answers").glob("batch_*.json")):
        ans.update(json.load(open(f)))
    missing = [p["id"] for p in pool if p["id"] not in ans]
    assert not missing, f"{len(missing)} ids have no judge answer"
    pool = [p for p in pool if not FORMAT_RE.search(p["answer"])]  # v2 rule added after judging
    for p in pool:
        top = [n.lower() for n in ans[p["id"]]]
        hits = [k for k, n in enumerate(top) if n == p["role"] or (p["role"] == "assistant" and n == "default")]
        p["judge_top5"], p["rank"] = top, (hits[0] + 1 if hits else None)
    chosen, summary = [], []
    for role in ROLES:
        rp = [p for p in pool if p["role"] == role]
        ok = sorted([p for p in rp if p["rank"]], key=lambda p: (p["rank"], p["source"] != "O"))
        pick2, qs = [], set()
        for p in ok:
            if p["question_idx"] not in qs:
                pick2.append(p)
                qs.add(p["question_idx"])
            if len(pick2) == 2:
                break
        chosen += pick2
        summary.append({"role": role, "pool": len(rp), "pass_rule1": len(ok),
                        "top1": sum(p["rank"] == 1 for p in rp), "chosen": len(pick2)})
    json.dump(pool, open(OUT / "pool_scored.json", "w"), indent=1)
    json.dump(chosen, open(OUT / "pilot_examples_auto.json", "w"), indent=1)
    json.dump(summary, open(OUT / "pilot_examples_auto_summary.json", "w"), indent=1)
    for s in summary:
        print(s)


def final():
    """Pilot set: v2 roles minus REPLACE_OUT, plus the first 3 v3 candidates with 2 rank-1 examples."""
    v2 = json.load(open(HERE / "v2/pilot_examples_auto.json"))
    v3 = json.load(open(HERE / "v3/pilot_examples_auto.json"))
    good = [r for r in REPLACE_CANDIDATES if [p["rank"] for p in v3 if p["role"] == r] == [1, 1]]
    use = good[:3]
    assert len(use) == 3, f"only {len(use)} replacement roles qualify: {good}"
    final = [p for p in v2 if p["role"] not in REPLACE_OUT] + [p for p in v3 if p["role"] in use]
    json.dump(final, open(HERE / "pilot_examples_final.json", "w"), indent=1)
    print("qualifying replacements in order:", good, "-> used:", use)
    print(len({p["role"] for p in final}), "roles,", len(final), "examples")


if __name__ == "__main__":
    {"prep": prep, "pick": pick, "final": final}[sys.argv[1]]()
