"""SFT and DPO write their own role replies on pod A's exact grid (task T6 / T-056). Pod A's generator and gates run
unchanged; only the model and its pinned revision change. Rules: THRESHOLDS.md.

  python -I stage_run.py --stage sft gen --out /abs/run_sft_full                     # full grid, 21,000 replies
  python -I stage_run.py --stage sft gen --roles canary --limit 40 --out /abs/smoke  # smoke test
  python -I stage_run.py --stage sft gates --run /abs/run_sft_full                   # pod A's G1-G5 and raw read

Before pod A's code runs, the wrapper (v1.1, after the Codex setup review of 13/11):
- turns --out / --run into absolute paths, so generation and gates address the same folder;
- writes `<out>.stage_receipt.json` next to the run folder (ordered hash of every (key, prompt ids), code hashes, runtime) on the first start and
  refuses to resume when any of it differs;
- repairs two crash states pod A's resume does not handle: a complete last row without its newline, and complete
  batches whose state part is missing or unreadable (their rows are removed, so pod A generates them again).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
POD_A = HERE.parent / "261029_instruct_persona_map"
STAGES = {e["stage"]: e for e in json.loads((HERE.parent / "261112_coinflip_stages/stages.json").read_text())["7b"]}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_pod_a(stage):
    """Pod A's gen_extract, registered as 'gen_extract' (pod A's gates import it by that name), with the model and
    revision of this stage. The package is loaded without its __init__ (rows.py needs only the standard library)."""
    if "manifold_persona" not in sys.modules:
        pkg = types.ModuleType("manifold_persona")
        pkg.__path__ = [str(REPO / "src/manifold_persona")]
        sys.modules["manifold_persona"] = pkg
    spec = importlib.util.spec_from_file_location("gen_extract", POD_A / "gen_extract.py")
    G = importlib.util.module_from_spec(spec)
    sys.modules["gen_extract"] = G
    spec.loader.exec_module(G)
    entry = STAGES[stage]
    G.MODEL, G.REVISION = entry["repo"], entry["revision"]
    return G


def gen_args(rest):
    """Pod A's generation arguments, parsed with pod A's defaults (gen_extract.main)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="full")
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    return ap.parse_args(rest)


def planned_items(G, tok, args):
    """The items pod A's main will generate, in its batch order (same roles, replicates, limit sample and sort)."""
    roles = G.ROLE_SETS.get(args.roles) or args.roles.split(",")
    items = G.build_items(roles, args.replicates)
    if args.limit:
        items = random.Random(0).sample(items, args.limit)
    return G.render(tok, items)


def code_hashes():
    return dict(stage_run_py=sha256_file(__file__), gen_extract_py=sha256_file(POD_A / "gen_extract.py"),
                gates_py=sha256_file(POD_A / "gates.py"), rows_py=sha256_file(REPO / "src/manifold_persona/rows.py"))


def runtime():
    import numpy
    import tokenizers
    import torch
    import transformers
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                transformers=transformers.__version__, tokenizers=tokenizers.__version__, numpy=numpy.__version__,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


def make_receipt(stage, G, items, args):
    blob = json.dumps([[it["key"], it["prompt_ids"]] for it in sorted(items, key=lambda it: it["key"])])
    return dict(stage=stage, model=G.MODEL, revision=G.REVISION, roles=args.roles, replicates=args.replicates,
                limit=args.limit, batch_size=args.batch_size, n_items=len(items),
                prompts_sha256=hashlib.sha256(blob.encode()).hexdigest(), code=code_hashes(), runtime=runtime())


def receipt_path(out):
    """Next to the run folder, not in it: pod A's code refuses a new run folder that already holds files."""
    return out.parent / f"{out.name}.stage_receipt.json"


def check_receipt(out, receipt):
    """Write the receipt on the first start; on a resume every field must match."""
    path = receipt_path(out)
    if path.exists():
        old = json.loads(path.read_text())
        if old != receipt:
            diff = sorted(k for k in set(old) | set(receipt) if old.get(k) != receipt.get(k))
            raise SystemExit(f"resume refused: stage_receipt.json differs in {diff}")
        return "resume"
    if (out / "fingerprint.txt").exists():
        raise SystemExit(f"{out} was started without this wrapper's receipt; use a new --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1))
    return "new"


def repair_partial(out, items, batch_size, row_key):
    """Fix crash states before pod A's resume reads them. Returns a list of the repairs made."""
    import numpy as np
    part, acts_dir = out / "rows.partial.jsonl", out / "acts_parts"
    if not part.exists():
        return []
    done = []
    raw = part.read_bytes()
    if raw and not raw.endswith(b"\n"):
        last = raw.rsplit(b"\n", 1)[-1]
        try:
            json.loads(last)
            part.write_bytes(raw + b"\n")
            done.append("added the missing newline after a complete last row")
        except json.JSONDecodeError:
            pass  # a truncated last line: pod A's read_done removes it
    lines = [l for l in part.read_bytes().split(b"\n") if l.strip()]
    keyed = []
    for l in lines:
        try:
            keyed.append((row_key(json.loads(l)), l))
        except json.JSONDecodeError:
            keyed.append((None, l))
    have = {k for k, _ in keyed if k is not None}
    drop = set()
    for b in range(0, len(items), batch_size):
        chunk = items[b:b + batch_size]
        keys = [it["key"] for it in chunk]
        if not set(keys) <= have:
            continue
        f = acts_dir / f"{b // batch_size:06d}.npz"
        try:
            ok = np.load(f)["keys"].tolist() == keys
        except Exception:
            ok = False
        if not ok:
            drop |= set(keys)
            if f.exists():
                f.unlink()
            done.append(f"batch {b // batch_size}: state part missing or unreadable; its rows removed for regeneration")
    if drop:
        keep = [l for k, l in keyed if k not in drop]
        tmp = part.with_suffix(".tmp")
        tmp.write_bytes(b"".join(l + b"\n" for l in keep))
        os.replace(tmp, part)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sft", "dpo"])
    ap.add_argument("cmd", choices=["gen", "gates"])
    a, rest = ap.parse_known_args()
    for flag in ("--out", "--run"):  # absolute paths, so generation and gates address the same folder
        if flag in rest:
            i = rest.index(flag) + 1
            rest[i] = str(Path(rest[i]).resolve())
    G = load_pod_a(a.stage)
    record = dict(stage=a.stage, model=G.MODEL, revision=G.REVISION, cmd=a.cmd, args=rest, code=code_hashes())
    print(json.dumps(record), flush=True)
    if a.cmd == "gen":
        args = gen_args(rest)
        out = Path(args.out)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(G.MODEL, revision=G.REVISION)
        items = planned_items(G, tok, args)
        state = check_receipt(out, make_receipt(a.stage, G, items, args))
        repairs = repair_partial(out, items, args.batch_size, G.R.row_key) if state == "resume" else []
        print(json.dumps(dict(start=state, repairs=repairs)), flush=True)
        sys.argv = [str(POD_A / "gen_extract.py")] + rest
        G.main()
        if (out / "manifest.json").exists():
            (out / "stage_run.json").write_text(json.dumps(dict(record, start=state, repairs=repairs), indent=1))
    else:
        spec = importlib.util.spec_from_file_location("poda_gates", POD_A / "gates.py")
        gates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gates)
        sys.argv = [str(POD_A / "gates.py")] + rest
        gates.main()


if __name__ == "__main__":
    main()
