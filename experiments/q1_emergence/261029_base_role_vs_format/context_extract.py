"""Pod B context controls C2 and C3 (THRESHOLDS.md, "Reported, no pass rule"), each built in its own format's
frame. Forward passes only, on the pod, after gen_extract.py and gates.py; same model, revision and dtype.

  C2  the prompt without its description line (gen_extract.prompt_text(fmt, None, q)) + the same kept reply
      ids; mean over the reply positions. Valid rows only; invalid rows are NaN. One state per row.
  C3  the stored full prompt + one fixed neutral continuation of its format; mean over the continuation.
      One state per prompt (role, format, wording, question): it does not depend on the reply.
Neutral texts approved by the owner on 29/10/2026: E3's C3 speech in every format; the story and description
put it in quotes after the verb their prompt needs. States at gen_extract.EXTRA_DEPTHS (hidden states 8, 16, 24
on the 7B).

  python context_extract.py --run run_canary          # writes run_canary/context/
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sibling(name):
    """A pod B script by path, under a pod-unique module name: pods A and B both have gen_extract.py and gates.py,
    and one pytest session imports both."""
    key = f"base_role_vs_format_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


G = sibling("gen_extract")
R = G.R
NEUTRAL_SPEECH = ("That is a good question. I think the answer depends on the situation, and I would want to think"
                  " about it carefully before I say more.")
NEUTRAL = {"interview": " " + NEUTRAL_SPEECH, "script": " " + NEUTRAL_SPEECH,
           "story": f' said, "{NEUTRAL_SPEECH}"', "description": f' say, "{NEUTRAL_SPEECH}"'}


def check_prompts(tok, rows):
    """Every stored prompt is the approved template text of its key, so C2 and C3 frames match the run."""
    bank = [json.loads(l)["question"] for l in open(G.REPO / "refs/assistant-axis/extraction_questions.jsonl")]
    desc = G.descriptions(sorted({r["role"] for r in rows}))
    for r in rows:
        want = G.prompt_ids(tok, G.prompt_text(r["format"], desc[r["role"], r["wording"]], bank[r["question"]]))
        if r["prompt_ids"] != want:
            raise SystemExit(f"{R.row_key(r)}: stored prompt is not the template text")
    return bank


def pool_batches(model, tok, seqs, spans, layers, batch_size):
    import numpy as np
    if not seqs:
        return np.zeros((0, len(layers), model.config.hidden_size), dtype=np.float32)
    return np.concatenate([R.pool_spans(G.Backbone(model), seqs[s:s + batch_size], spans[s:s + batch_size],
                                        tok.pad_token_id, layers) for s in range(0, len(seqs), batch_size)])


def extract(rows, model, tok, batch_size):
    """C2 per row (NaN when invalid) and C3 per prompt, at depth_layers() of the model."""
    import numpy as np
    layers = G.depth_layers(model.config.num_hidden_layers)
    bank = check_prompts(tok, rows)
    idx = [i for i, r in enumerate(rows) if r["valid"]]
    c2_seqs, c2_spans = [], []
    for i in idx:
        r = rows[i]
        p = G.prompt_ids(tok, G.prompt_text(r["format"], None, bank[r["question"]]))
        c2_seqs.append(p + r["kept_ids"])
        c2_spans.append([len(p), len(p) + len(r["kept_ids"])])
    c2 = np.full((len(rows), len(layers), model.config.hidden_size), np.nan, dtype=np.float32)
    c2[idx] = pool_batches(model, tok, c2_seqs, c2_spans, layers, batch_size)
    neutral = {f: tok(t, add_special_tokens=False).input_ids for f, t in NEUTRAL.items()}
    first = {}
    for r in rows:  # replicates of one prompt share its prompt ids
        first.setdefault(G.prompt_of(R.row_key(r)), r)
    prompts = list(first)
    c3_seqs = [first[p]["prompt_ids"] + neutral[p[1]] for p in prompts]
    c3_spans = [[len(first[p]["prompt_ids"]), len(first[p]["prompt_ids"]) + len(neutral[p[1]])] for p in prompts]
    c3 = pool_batches(model, tok, c3_seqs, c3_spans, layers, batch_size)
    return dict(c2=c2, c3=c3, prompts=[list(p) for p in prompts], layers=layers, neutral_ids=neutral)


def check(res, rows, width):
    """C2 finite on valid rows and NaN on invalid rows, C3 finite everywhere, shapes as the run: a failed
    forward pass must not end in checksummed files."""
    import numpy as np
    c2, c3, n = res["c2"], res["c3"], len(res["layers"])
    v = np.array([r["valid"] for r in rows], dtype=bool)
    problems = []
    if c2.shape != (len(rows), n, width) or c3.shape != (len(res["prompts"]), n, width):
        problems.append(f"shapes C2 {c2.shape}, C3 {c3.shape}")
    elif not (np.isfinite(c2[v]).all() and np.isnan(c2[~v]).all()):
        problems.append("C2 is not finite on every valid row and NaN on every invalid row")
    elif not np.isfinite(c3).all():
        problems.append("C3 has non-finite values")
    if problems:
        raise SystemExit("context states refused: " + "; ".join(problems))


def save(out, res):
    """float16 when every finite value fits, else float32."""
    import numpy as np
    for name in ("c2", "c3"):
        a = res[name]
        fin = a[np.isfinite(a)]
        fits = not fin.size or np.abs(fin).max() <= G.F16_MAX
        np.save(out / f"ctx_{name}.npy", a.astype(np.float16) if fits else a)
    (out / "c3_prompts.json").write_text(json.dumps(res["prompts"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="context", help="folder inside the run folder")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_absolute() else HERE / args.run
    out = run_dir / args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; use another --out")
    run = G.load_run(run_dir)
    man = run["manifest"]
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(G.MODEL, revision=man["revision"])
    model = G.load_model(man["revision"])
    t = time.time()
    res = extract(run["rows"], model, tok, args.batch_size)
    check(res, run["rows"], model.config.hidden_size)
    out.mkdir(parents=True, exist_ok=True)
    save(out, res)
    import platform
    import transformers
    code = {p.name: G.sha256_file(p) for p in (Path(__file__).resolve(), Path(G.__file__).resolve(),
                                              Path(R.__file__).resolve())}
    (out / "manifest.json").write_text(json.dumps(dict(
        model=G.MODEL, revision=man["revision"], run=run_dir.name, run_fingerprint=man["fingerprint"],
        batch_size=args.batch_size, depths=G.EXTRA_DEPTHS, layers=res["layers"], neutral=NEUTRAL,
        neutral_ids=res["neutral_ids"], n_rows=len(run["rows"]), n_c2_valid=int(sum(r["valid"] for r in run["rows"])),
        n_prompts=len(res["prompts"]), code=code,
        c2_rows="ctx_c2.npy: same order as rows.jsonl; NaN where valid is false",
        c3_rows="ctx_c3.npy: same order as c3_prompts.json (role, format, wording, question)",
        versions=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                      transformers=transformers.__version__, numpy=np.__version__),
        seconds=round(time.time() - t)), indent=1))
    (out / "SHA256SUMS.txt").write_text("".join(f"{G.sha256_file(out / n)}  {n}\n" for n in G.CONTEXT_FILES))
    print(f"done: C2 {res['c2'].shape}, C3 {res['c3'].shape}, hidden states {res['layers']}")


if __name__ == "__main__":
    main()
