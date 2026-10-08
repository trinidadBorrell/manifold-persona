"""Deterministic gates for a mentor report. Exit 1 if any FAIL.

  G1 tropes      regex hits from the tropes list (mechanical ones only)
  G2 length      prose words <= --max-words
  G3 structure   template headings per experiment; every "### Result" has <!-- inv: id[, id] -->
  G4 coverage    every inventory row has a status; every `in` id is tagged in the report; every tag is in the inventory
  G5 sourced     every decimal or percent in a result section appears (rounded) in that section's inventory sources
  G6 repeats     numbers that also appear in the prior-report ledger must sit in a sentence that labels them as earlier
G5 and G6 print WARN lines for the reviewer; they fail only on G3/G4-style breakage.

Run from the repo root:
  .venv/bin/python .claude/skills/presenting-results/gate_check.py \
      output/results_2026-09/writeup/REPORT_persona_stages.md \
      --inventory output/results_2026-09/writeup/gate/INVENTORY.md \
      --sources output/results_2026-09 --ledger output/results_2026-09/writeup/gate/prior
"""
import argparse, re, sys
from pathlib import Path

TEMPLATE = ["TLDR", "Background / Motivation", "Concrete question we are answering", "Methodology",
            "Results", "Global takeaways", "Next steps", "Questions"]
TROPES = [
    (r"[–—]", "em/en dash"), (r"[←-⇿]", "unicode arrow"), (r"°", "degree sign"),
    (r"[“”‘’]", "curly quote"),
    (r"\b(?:we|our|us)\b", "collaborative we"),
    (r"\bnot (?:just|only|merely)\b|n't just\b", "negative parallelism"),
    (r"\b(?:two|three|four|five|six|seven)\s+(?:things|results|reasons|points|corrections|ways|lessons|takeaways)\b", "compulsive counting"),
    (r"\b(?:notably|importantly|interestingly|crucially|fundamentally|remarkably|arguably|quietly|deeply)\b", "magic adverb / filler"),
    (r"it'?s worth noting|it bears mentioning|here'?s the (?:thing|kicker|catch)|let'?s (?:dive|unpack|break)", "filler transition"),
    (r"\b(?:delve|leverage|tapestry|landscape|paradigm|synergy|load-bearing|serves as|stands as)\b", "AI vocabulary"),
    (r"\b(?:in conclusion|to sum up|in summary)\b", "signposted conclusion"),
    (r"^#+\s*(?:[A-Za-z]+ \d+:\s*)?(?:What|Where|Why|How|When|Which|Whether)\b", "Wh-header"),
    (r"\bthe [a-z]+ (?:paradox|trap|creep|divide|vacuum)\b", "invented label"),
]
ALLOW_WE = "Concrete question we are answering"
LABEL = re.compile(r"earlier|reported|correction|from my|before|as I|\b\d{2}/\d{2}\b|previous|rechecks?", re.I)
NUM = re.compile(r"(?<![\w.\-/])[-+]?\d+\.\d+%?|(?<![\w.\-/])\d+(?:\.\d+)?%")
SKIP = {("pct", 95.0), ("pct", 95)}
ARXIV = re.compile(r"\d{4}\.\d{4,5}")


def prose(md):
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    return re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", "", md)


def rounded_forms(x):
    """Signed rounded forms of a source number, plus ("abs", ...) magnitude forms.

    A report number must match with its sign; only a number written as a
    magnitude (e.g. "|r| = 0.66") may match on absolute value.
    """
    out = set()
    for d in (1, 2, 3):
        out.add(round(x, d)); out.add(("abs", round(abs(x), d)))
    for d in (0, 1):
        out.add(("pct", round(100 * x, d))); out.add(("abs", ("pct", round(100 * abs(x), d))))
    return out


def file_numbers(paths):
    s = set()
    for p in paths:
        if not p.exists():
            continue
        for tok in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", p.read_text(errors="ignore")):
            try:
                s |= rounded_forms(float(tok))
            except ValueError:
                pass
    return s


def report_number_key(tok, magnitude=False):
    t = tok.lstrip("+")
    if t.endswith("%"):
        v = float(t[:-1]); d = len(t[:-1].split(".")[1]) if "." in t else 0
        return ("abs", ("pct", round(abs(v), d))) if magnitude else ("pct", round(v, d))
    v = float(t); d = len(t.split(".")[1])
    return ("abs", round(abs(v), d)) if magnitude else round(v, d)


def parse_inventory(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 12 or cells[0] in ("id", "---") or set(cells[0]) <= {"-"}:
            continue
        rows[cells[0]] = {"sources": [s.strip() for s in cells[2].split(";") if s.strip()],
                          "status": cells[10], "reason": cells[11]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report"); ap.add_argument("--inventory", required=True)
    ap.add_argument("--sources", required=True); ap.add_argument("--ledger", required=True)
    ap.add_argument("--max-words", type=int, default=2000)
    a = ap.parse_args()
    md = Path(a.report).read_text(); lines = md.splitlines()
    fails, warns = {}, {}
    add = lambda d, g, m: d.setdefault(g, []).append(m)

    for i, ln in enumerate(lines, 1):  # G1
        if ln.startswith("![") or "<!--" in ln:
            continue
        for pat, name in TROPES:
            for m in re.finditer(pat, ln, flags=re.I | re.M):
                if name == "collaborative we" and ALLOW_WE in ln:
                    continue
                add(fails, "G1 tropes", f"L{i} {name}: '{m.group(0)}'")

    n = len(prose(md).split())  # G2
    if n > a.max_words:
        add(fails, "G2 length", f"{n} prose words > {a.max_words}")

    exps = re.split(r"\n(?=# )", md)  # G3
    tags = {}
    for e in exps:
        title = e.splitlines()[0]
        if not title.startswith("# "):
            continue
        heads = re.findall(r"^## (.+)$", e, flags=re.M)
        for h in TEMPLATE:
            if not any(x.startswith(h) for x in heads):
                add(fails, "G3 structure", f"{title[:50]}: missing '## {h}'")
        for sec in re.split(r"\n(?=### )", e)[1:]:
            head = sec.splitlines()[0]
            if not head.startswith("### Result"):
                continue
            m = re.search(r"<!--\s*inv:\s*([^>]+?)\s*-->", sec)
            if not m:
                add(fails, "G3 structure", f"{head[:60]}: no <!-- inv: id --> tag")
                continue
            body = re.split(r"\n(?=## )", sec)[0]
            tags[head] = (body, [t.strip() for t in m.group(1).split(",")])

    inv = parse_inventory(a.inventory)  # G4
    tagged = {t for _, ids in tags.values() for t in ids}
    for rid, r in inv.items():
        if r["status"] not in ("in", "out", "context"):
            add(fails, "G4 coverage", f"{rid}: status '{r['status']}'")
        if r["status"] == "out" and not r["reason"]:
            add(fails, "G4 coverage", f"{rid}: out with no reason")
        if r["status"] == "in" and rid not in tagged and "methods" not in r["reason"]:
            add(fails, "G4 coverage", f"{rid}: status in, but no result is tagged with it")
    for t in tagged - set(inv):
        add(fails, "G4 coverage", f"tag '{t}' is not in the inventory")

    src = Path(a.sources)  # G5
    for head, (sec, ids) in tags.items():
        pool = file_numbers([src / s for rid in ids for s in inv.get(rid, {}).get("sources", [])])
        body = "\n".join(l for l in prose(sec).splitlines() if not l.startswith("Plot:"))
        seen = set()
        for m in NUM.finditer(body):
            tok = m.group(0)
            if ARXIV.fullmatch(tok):  # arXiv ids are citations, not measurements
                continue
            mag = "|" in body[max(0, m.start() - 6):m.start()]   # "|r| = 0.66"
            k = report_number_key(tok, magnitude=mag)
            if (tok, mag) in seen:
                continue
            seen.add((tok, mag))
            if k not in pool and report_number_key(tok) not in SKIP:
                add(warns, "G5 sourced", f"{head[:45]}: '{tok}' not found in its sources"
                    + (" (with this sign)" if not mag else ""))

    ledger = file_numbers(sorted(Path(a.ledger).glob("*.txt")))  # G6
    for i, ln in enumerate(lines, 1):
        if ln.startswith(("#", "![", "Plot:")) or "<!--" in ln:
            continue
        for tok in NUM.findall(ln):
            k = report_number_key(tok)
            if k in ledger and k not in (0.5, 1.0, ("pct", 0.0)) and not LABEL.search(ln):
                add(warns, "G6 repeats", f"L{i} '{tok}' is in an earlier report; label it or drop it")

    print(f"{'gate':<16}{'result':<8}count")
    for g in ["G1 tropes", "G2 length", "G3 structure", "G4 coverage"]:
        print(f"{g:<16}{'FAIL' if g in fails else 'PASS':<8}{len(fails.get(g, []))}")
    for g in ["G5 sourced", "G6 repeats"]:
        print(f"{g:<16}{'WARN' if g in warns else 'PASS':<8}{len(warns.get(g, []))}")
    for d in (fails, warns):
        for g, ms in d.items():
            print(f"\n[{g}]"); [print("  " + m) for m in ms]
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
