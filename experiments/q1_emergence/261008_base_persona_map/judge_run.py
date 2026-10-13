"""Run the blind top-5 naming judges on the isolated judge folder made by judge_prep.py.

Judges: gpt = GPT-5.6-Sol via Codex CLI (reasoning high); claude = Claude Sonnet 5.5 via Claude Code CLI.
Both get the E0 naming prompt (prompt_judge.txt) and run with the isolated folder as working
directory, so the answer keys are not reachable by relative path. Each answer file is checked
(exact batch ids; 5 distinct names from candidates.txt). A failed batch is retried once; failures
are listed and the script exits non-zero.

    .venv/bin/python experiments/q1_emergence/261008_base_persona_map/judge_run.py JUDGE_DIR gpt|claude [--workers 8]
"""
import argparse
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULE = ("Do not write or run scripts or programs that classify, score or keyword-match the replies. Read the "
        "files named below yourself and write your answer file. Do not open any other file.")
CODEX = shutil.which("codex") or "/Users/brendon/.local/share/nvm/v22.22.2/bin/codex"
CLAUDE = shutil.which("claude") or "/Users/brendon/.local/bin/claude"


def prompt(n, n_items, out_dir):
    t = (HERE / "prompt_judge.txt").read_text()
    t = t.replace("TOOL_RULE", RULE).replace("DIR/batches/batch_N.md", f"batches/batch_{n}.md")
    t = t.replace("DIR/answers/batch_N.json", f"{out_dir}/batch_{n}.json").replace("DIR/candidates.txt", "candidates.txt")
    t = t.replace("about 150 replies", f"{n_items} replies")
    assert "DIR" not in t and "batch_N" not in t
    return t


def check(jd, n, out_dir, cands):
    f = jd / out_dir / f"batch_{n}.json"
    want = re.findall(r"^### (\S+)$", (jd / "batches" / f"batch_{n}.md").read_text(), re.M)
    try:
        a = json.loads(f.read_text())
    except Exception as e:
        return f"unreadable: {e}"
    if set(a) != set(want):
        return f"ids differ: {len(set(want) - set(a))} missing, {len(set(a) - set(want))} extra"
    bad = [k for k, v in a.items() if not (isinstance(v, list) and len(v) == 5 and len({x.lower() for x in v}) == 5
                                           and all(isinstance(x, str) and x.lower() in cands for x in v))]
    return f"{len(bad)} bad entries" if bad else None


def call(jd, judge, n, n_items, out_dir, log):
    p = prompt(n, n_items, out_dir)
    if judge == "gpt":
        cmd = [CODEX, "exec", "-m", "gpt-5.6-sol", "-c", "model_reasoning_effort=high", "-s", "workspace-write",
               "--ephemeral", "--skip-git-repo-check", "-C", str(jd), p]
    else:
        cmd = [CLAUDE, "-p", p, "--model", "claude-sonnet-5-5", "--allowedTools", "Read", "Write",
               "--disallowedTools", "Bash", "--permission-mode", "acceptEdits"]
    with open(log, "w") as lf:
        return subprocess.run(cmd, cwd=jd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                              timeout=3600).returncode


def job(jd, judge, n, cands):
    out_dir = f"answers_{judge}"
    f = jd / out_dir / f"batch_{n}.json"
    if f.exists() and check(jd, n, out_dir, cands) is None:
        return n, None
    n_items = (jd / "batches" / f"batch_{n}.md").read_text().count("\n### ")
    err = None
    for attempt in (1, 2):
        f.unlink(missing_ok=True)
        rc = call(jd, judge, n, n_items, out_dir, jd / "logs" / f"{judge}_{n}_try{attempt}.log")
        err = check(jd, n, out_dir, cands) or (None if rc == 0 else f"exit {rc}")
        if err is None:
            return n, None
    return n, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judge_dir")
    ap.add_argument("judge", choices=["gpt", "claude"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batches", default=None, help="e.g. 1-3 for a test")
    a = ap.parse_args()
    jd = Path(a.judge_dir).resolve()
    (jd / "logs").mkdir(exist_ok=True)
    cands = set((jd / "candidates.txt").read_text().split())
    n_all = len(list((jd / "batches").glob("batch_*.md")))
    ns = range(1, n_all + 1)
    if a.batches:
        lo, hi = map(int, a.batches.split("-"))
        ns = range(lo, hi + 1)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda n: job(jd, a.judge, n, cands), ns))
    fails = [(n, e) for n, e in res if e]
    print(f"{a.judge}: {len(res) - len(fails)}/{len(res)} batches valid")
    for n, e in fails:
        print(f"  batch {n}: {e}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
