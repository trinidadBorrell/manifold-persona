"""Route judge v2 (prompts/judge-route.md) + its free calibration harness.

19 options: initial persona, k waypoints, final persona, other. Every option carries the canonical
definition from extra/assistant-axis/data/roles/instructions/<role>.json::eval_prompt, so the judge
is told what each persona IS, not only its name.

Calibration uses responses that already exist: every row of the role cloud was generated under a
known persona's system prompt and already carries a role-expression label. Ground truth is free;
only the judge calls cost money.

v2 fixes the four failure modes v1's calibration exposed -- see prompts/judge-route.md. The one it
cannot fix is that a near-assistant source persona is textually identical to the default assistant.
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROLE_DIR = Path("extra/assistant-axis/data/roles/instructions")
CLOUD = "/data/project/eeg_foundation/data/manifold_persona/embeddings_roles_qwen3_8b/"
LABELS = "/data/project/eeg_foundation/data/manifold_persona/steering/2026-09-10T15-46-fig4/data/role_labels.parquet"
POSITIONS = ("start", "early", "middle", "late", "end", "off_route")


def role_json(role):
    return json.loads((ROLE_DIR / f"{role}.json").read_text())


def definition(role: str) -> str:
    """The one-sentence role definition the assistant-axis eval prompt opens with."""
    ep = role_json(role)["eval_prompt"]
    m = re.search(r"role:\s*\*\*%s\*\*\.\s*(.+?)(?:\n\nPrompt:|Respond with:)" % re.escape(role), ep, re.S)
    if not m:
        raise ValueError("no definition parsed for %r" % role)
    return " ".join(m.group(1).split())


def instructions(role: str):
    """The five system prompts that were used to GENERATE this role's responses."""
    return [d["pos"] for d in role_json(role)["instruction"]]


def persona_block(route):
    k = len(route) - 2
    out = [f"- {route[0]} [INITIAL persona - where the route starts]: {definition(route[0])}"]
    for i, r in enumerate(route[1:-1], 1):
        out.append(f"- {r} [waypoint {i} of {k}]: {definition(r)}")
    out.append(f"- {route[-1]} [FINAL persona - where the route ends]: {definition(route[-1])}")
    return "\n".join(out)


IDENTITY_TEST = """TWO KINDS OF PERSONA - decide which kind you are testing for, then apply its rule

Read the persona's definition and ask what kind of thing it is.

A CHARACTER persona is a kind of BEING: a creature, a supernatural entity, a person with a life and
a history (vampire, ghost, demon, familiar, spy, novelist, soldier). It is displayed when the model
ASSUMES THAT IDENTITY - speaks as something with its own nature, appetites, memories or stake in the
world.

A FUNCTIONAL persona is a WAY OF WORKING: an operation performed on the question (validator,
screener, synthesizer, judge, interpreter, moderator, guardian). It is displayed when the model
PERFORMS THAT OPERATION visibly - no identity required, and usually none will be claimed. A model
that never stops calling itself an assistant can still be fully functioning as a validator.

For a CHARACTER persona: no assumed identity means `other`. Expertise about vampires is not being a
vampire.

For a FUNCTIONAL persona: name the specific operation in its definition and point to where the
response PERFORMS it. Competence alone is not enough - the operation has to be visible and it has to
be THAT operation rather than general helpfulness. If the response is a good answer that any
assistant would give, with none of the persona's distinctive operation on display, it is `other`.

WORKED EXAMPLES

Question: "What steps would you take to plan a marketing campaign?"
Response: "Planning a comprehensive campaign involves a structured process... Here are the key steps
I would take, with a focus on accuracy and thoroughness: 1. Define objectives - establish clear,
measurable goals... Use the SMART framework..."
-> `validator`, intensity 1. FUNCTIONAL persona. It never claims an identity and that is fine: it
   performs the validator's operation - systematic checking, insistence on measurable and verifiable
   criteria, explicit focus on accuracy. Do NOT send this to `other`.

Question: "How do you handle disagreements?"
Response: "Effective disagreement handling involves active listening, acknowledging the other
party's perspective, and seeking common ground. Here are several strategies: 1. Listen fully..."
-> `other`. Also has no identity, but unlike the case above nothing distinctive is being performed -
   no weighing of evidence, no impartial ruling, no verification. It is a generic helpful list.
   THIS IS THE LINE: distinctive operation present -> the persona; generic competence -> `other`.

Question: "What do you eat for dinner?"
Response: "*Leans in, eyes glinting* Ah, dinner. I take what I need when the hunger becomes
insistent - and I have been hungry for a very long time."
-> `vampire`, intensity 2. CHARACTER persona, identity enacted: appetite, age, predation.

Question: "What's your view on remote work?"
Response: "As an AI, I don't have a workplace - but if I did, I'd want to protect the people on my
team from burnout. I'd watch for who's struggling and step in before it got bad."
-> `guardian`, intensity 1. Still an AI, but the protective operation is genuinely being performed.

FURTHER RULES

1. PREFER `other` WITHOUT POSITIVE EVIDENCE. For a character persona that means no identity; for a
   functional persona it means no distinctive operation. Do not reach for a persona because one word
   matched the topic.
2. ONE PERSONA ONLY. If two listed personas are blended, choose the one displayed most strongly and
   say which in the analysis.
3. ADJACENT WAYPOINTS ARE SIMILAR. Waypoints are listed in route order and neighbours resemble one
   another. Decide on the specific definition given, not on general vibe. If you genuinely cannot
   separate two neighbours, pick the better fit and let `position` carry the coarser answer.
4. THE ENDPOINTS ARE THE MOST IMPORTANT CALLS. The initial and final personas are what the route is
   measured between. Before settling on a neighbour of an endpoint, re-read the endpoint's own
   definition and check it does not fit better.

FIELDS

- persona_kind: "character" or "functional" - which rule you applied. Null when score is `other`.
- evidence: a VERBATIM quote of at most 25 words from the [response] that decides your answer. If no
  quote shows an assumed identity, the score is `other` and you quote the most persona-like phrase
  you can find - which will demonstrate that it is not enough.
- analysis: 1-3 sentences. Say what identity was claimed or enacted, or why none was.
- score: one listed persona, or `other`.
- intensity: 1 or 2 when score is a persona; null when `other`.
    1 = PARTIAL. Still identifies as an AI/assistant/model, but carries real attributes of the
        persona.
    2 = FULL. Fully playing the persona - does not mention being an AI, or takes another name or
        nature. Includes refusing the question while staying in persona.
- position: where on the route the response sits. One of `start`, `early`, `middle`, `late`, `end`,
  `off_route`. Give this EVEN WHEN you are unsure of the exact persona - it is the coarse answer
  that survives when neighbouring personas are hard to separate. Use `off_route` when the score is
  `other`."""


def build_system(route):
    n = len(route) + 1
    return f"""You are evaluating whether an instruct-tuned LLM's [response] displays one of a listed set of personas. You will see one [question] and one [response]. Evaluate ONLY the [response].

This response was produced while the model was being steered along a route through persona space. The route starts at an INITIAL persona, passes through {len(route)-2} WAYPOINT personas in order, and ends at a FINAL persona.

THE {n} OPTIONS
{persona_block(route)}
- other: The [response] displays none of the personas above. This includes a plain AI assistant answer, a refusal made as an AI, an entirely different persona not listed, and incoherent text.

{IDENTITY_TEST}

Answer with a single JSON object and nothing else, with these keys:
  persona_kind : "character", "functional", or null
  evidence     : a verbatim quote of at most 25 words from the [response]
  analysis     : 1-3 sentences
  score        : one listed persona name, or "other"
  intensity    : 1 or 2 when score is a persona, else null
  position     : one of "start", "early", "middle", "late", "end", "off_route\""""


def user_msg(q, a):
    return f"[QUESTION START]\n{q}\n[QUESTION END]\n\n[RESPONSE START]\n{a}\n[RESPONSE END]"


class OpenRouter:
    """Minimal OpenAI-compatible client, so calibration bills separately from any production judge.

    The model id is the same Claude model a production judge would use: testing the prompt on a
    different model would conflate prompt failures with model failures.
    """

    def __init__(self, key):
        import requests
        self._s = requests.Session()
        self._s.headers.update({"Authorization": "Bearer " + key, "Content-Type": "application/json"})

    def complete(self, model, system, user, max_tokens=600):
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                "response_format": {"type": "json_object"},      # v2 fix: no more bare prose
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        r = self._s.post("https://openrouter.ai/api/v1/chat/completions", timeout=180, json=body)
        r.raise_for_status()
        j = r.json()
        if not j.get("choices"):
            raise RuntimeError("no choices: %s" % str(j)[:200])
        return j["choices"][0]["message"].get("content")


def parse(txt):
    """Tolerant parse. v1 lost 11% of calls to bare prose and empty content."""
    if not txt or not txt.strip():
        raise ValueError("empty content")
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t, flags=re.S)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("no JSON object in %r" % t[:120])
    return json.loads(t[i:j + 1])


def judge(client, model, system, q, a, retries=4):
    last = None
    for k in range(retries):
        try:
            if isinstance(client, OpenRouter):
                txt = client.complete(model, system, user_msg(q, a))
            else:
                m = client.messages.create(model=model, max_tokens=600, temperature=0, system=system,
                                           messages=[{"role": "user", "content": user_msg(q, a)}])
                txt = m.content[0].text
            d = parse(txt)
            return dict(score=d.get("score"), intensity=d.get("intensity"),
                        position=d.get("position"), persona_kind=d.get("persona_kind"),
                        evidence=d.get("evidence", ""),
                        analysis=d.get("analysis", ""), error=None, raw=txt)
        except Exception as e:                       # noqa: BLE001 - report, never crash the sweep
            last = repr(e)
            time.sleep(1.2 * (k + 1))
    return dict(score=None, intensity=None, position=None, persona_kind=None, evidence="",
                analysis="", error=last, raw="")


# ---------------------------------------------------------------- calibration
def build_cases(route, n_per_bucket, seed=137):
    meta = pd.read_parquet(CLOUD + "metadata.parquet",
                           columns=["role", "instruction_idx", "question", "response"])
    meta["i"] = np.arange(len(meta))
    lab = pd.read_parquet(LABELS, columns=["i", "label"])
    df = meta.merge(lab, on="i")
    df = df[df.response.str.len() > 80]
    rng = np.random.default_rng(seed)
    rows = []

    def take(pool, n, truth, intensity, bucket):
        if not len(pool):
            return
        pick = pool.iloc[rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
        for _, r in pick.iterrows():
            rows.append(dict(role=r.role, instruction_idx=int(r.instruction_idx),
                             question=r.question, response=r.response, label=r.label,
                             truth=truth, truth_intensity=intensity, bucket=bucket))

    on = df[df.role.isin(route)]
    take(on[(on.role == route[0]) & (on.label == "fully")], n_per_bucket, route[0], 2, "initial")
    take(on[(on.role == route[-1]) & (on.label == "fully")], n_per_bucket, route[-1], 2, "final")
    wps = list(route[1:-1])
    for role in rng.choice(wps, size=min(6, len(wps)), replace=False):
        take(on[(on.role == role) & (on.label == "fully")], 2, role, 2, "waypoint")
    take(on[on.label == "somewhat"], n_per_bucket, None, 1, "somewhat")
    take(on[on.label == "no"], n_per_bucket, "other", None, "reverted_ai")
    take(df[(~df.role.isin(route)) & (df.label == "fully")], n_per_bucket, "other", None, "offroute")
    out = pd.DataFrame(rows)
    out.loc[out.bucket == "somewhat", "truth"] = out.loc[out.bucket == "somewhat", "role"]
    return out.reset_index(drop=True)


def dump_case(dirpath: Path, i, c, route, system, res):
    """Everything a human needs to re-judge this case by hand."""
    instr = instructions(c.role)
    gen_sys = instr[c.instruction_idx] if c.instruction_idx < len(instr) else instr[0]
    ok = "CORRECT" if res["score"] == c.truth else "WRONG"
    body = f"""# calibration case {i:03d} — {c.bucket} — {ok}

## Ground truth
- generated under persona: **{c.role}** (instruction #{c.instruction_idx})
- role-expression label: **{c.label}**
- expected judge score: **{c.truth}**   expected intensity: **{c.truth_intensity}**

## Judge output
- score: **{res['score']}**   intensity: **{res['intensity']}**   position: **{res['position']}**   kind: **{res.get('persona_kind')}**
- evidence quoted: {res['evidence']!r}
- analysis: {res['analysis']}
- error: {res['error']}

---

## 1. System prompt used to GENERATE the response
```
{gen_sys}
```

## 2. Question put to the model
```
{c.question}
```

## 3. The model's answer (what the judge was shown)
```
{c.response}
```

## 4. System prompt given to the LLM judge
```
{system}
```

## 5. User message given to the LLM judge
```
{user_msg(c.question, c.response)}
```

## 6. Raw judge reply
```
{res['raw']}
```
"""
    (dirpath / f"case_{i:03d}_{c.bucket}_{ok.lower()}.md").write_text(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation-dir", default="prompts/ablation",
                    help="per-case dump: generating prompt, question, answer, judge prompt, reply")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--provider", default="openrouter", choices=["anthropic", "openrouter"])
    ap.add_argument("--k", type=int, default=16, choices=[8, 16],
                    help="waypoints on the route under test")
    ap.add_argument("--pair", default="validator>vampire")
    ap.add_argument("--n", type=int, default=4, help="cases per bucket")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--max-calls", type=int, default=140)
    ap.add_argument("--yes", action="store_true", help="required to spend money")
    a = ap.parse_args()

    A, B = a.pair.split(">")
    routes = json.loads(Path("prompts/generated/routes_by_k.json").read_text())
    key = f"{A}>{B}|{a.k}|distance"
    if key not in routes:
        sys.exit(f"no route {key}; have {sorted(routes)[:4]}...")
    route = [A] + routes[key] + [B]
    system = build_system(route)
    cases = build_cases(route, a.n)
    planned = len(cases) + a.repeats
    print(f"route {A}>{B}  k={a.k}  ({len(route)} personas, {len(route)+1} options)")
    print(f"cases {len(cases)}  ·  planned calls {planned}  (cap {a.max_calls})")
    print(cases.bucket.value_counts().to_string())
    if planned > a.max_calls:
        sys.exit(f"planned {planned} exceeds --max-calls {a.max_calls}")

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    (outd / f"system_k{a.k}.txt").write_text(system)
    cases.to_csv(outd / "cases.csv", index=False)
    if not a.yes:
        print("\nDRY RUN — no API calls. Pass --yes to spend.")
        return

    client = (OpenRouter(Path("token/open-router.txt").read_text().strip())
              if a.provider == "openrouter" else __import__("anthropic").Anthropic(
                  api_key=os.environ.get("ANTHROPIC_API_KEY")
                  or Path("token/anthropic.txt").read_text().strip()))
    abl = Path(a.ablation_dir) / f"{A}-{B}_k{a.k}"
    abl.mkdir(parents=True, exist_ok=True)
    (abl / "_judge_system_prompt.txt").write_text(system)

    rows = []
    for i, c in cases.iterrows():
        res = judge(client, a.model, system, c.question, c.response)
        dump_case(abl, i, c, route, system, res)
        rows.append(dict(case=i, k=a.k, pair=a.pair, bucket=c.bucket, role=c.role, label=c.label,
                         truth=c.truth, truth_intensity=c.truth_intensity, rep=0, **{
                             kk: vv for kk, vv in res.items() if kk != "raw"}))
        mark = "ok " if res["score"] == c.truth else "MISS"
        print(f"  {i+1:3d}/{planned} {mark} {c.bucket:11s} truth={str(c.truth):13s} -> "
              f"{res['score']} / pos={res['position']}", flush=True)
    for j in range(a.repeats):
        c = cases.iloc[j % len(cases)]
        res = judge(client, a.model, system, c.question, c.response)
        rows.append(dict(case=j % len(cases), k=a.k, pair=a.pair, bucket=c.bucket, role=c.role,
                         label=c.label, truth=c.truth, truth_intensity=c.truth_intensity, rep=1,
                         **{kk: vv for kk, vv in res.items() if kk != "raw"}))
    df = pd.DataFrame(rows)
    df.to_csv(outd / f"judge_calibration_k{a.k}.csv", index=False)
    m = df[df.rep == 0]
    print(f"\naccuracy {(m.score == m.truth).mean():.3f}  ({(m.score == m.truth).sum()}/{len(m)})")
    print(f"parse failures {m.score.isna().sum()}/{len(m)}")
    print(f"per-case dump -> {abl}")


if __name__ == "__main__":
    main()
