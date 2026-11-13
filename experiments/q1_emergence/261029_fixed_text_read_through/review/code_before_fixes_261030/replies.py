"""Pod C reply reader: the fixed reply texts that every checkpoint reads.

Design: ../261029_stage_persona_plan/DESIGN.md (pod C) and THRESHOLDS.md here ("Data").
- Source 1 (main): one finished pod A run in the shared row schema (manifold_persona.rows). Its files must
  match SHA256SUMS.txt, its model must be the Instruct commit in checkpoints.py, and its rows must fill its
  frozen grid exactly. Invalid rows stay, flagged, because F3 uses the complete grid.
- Source 2: E3 base replies (run_main; conditions desc and ex2; the 20 pilot roles; the 18 shared
  questions), reply ids only. E3 rows have their own schema, and only read_e3 reads them.
- Both give one record per reply: source, key, role, format, wording, question, replicate, reply_ids, valid.
  The format is "chat" for pod A and the E3 condition for E3. Pod A records also keep prompt_ids (the chat
  frame, for the G3 reconstruction). Ids are never rebuilt from text.

  python replies.py --pod_a ../261029_instruct_persona_map/run_full
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from manifold_persona import rows as R

import checkpoints as C

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E3_RUN = HERE.parent / "261008_base_persona_map/run_main"
E3_CONDS = ("desc", "ex2")
E3_N_QUESTIONS = 18
E3_ONLY = {"cond", "ended_by", "reply_ids"}
N_WORDINGS = 5
CK = {ck["name"]: ck for ck in C.CHECKPOINTS}


def verify_sums(sums_file, base, names):
    """sha256 of each named file, checked against a SHA256SUMS file."""
    want = {name: h for h, name in (line.split("  ", 1) for line in Path(sums_file).read_text().splitlines())}
    for n in names:
        if n not in want:
            raise ValueError(f"{n} is not in {sums_file}")
        if R.file_sha256(Path(base) / n) != want[n]:
            raise ValueError(f"{n} does not match {sums_file}")
    return {n: want[n] for n in names}


def check_model(manifest, name):
    ck = CK[name]
    if (manifest.get("model"), manifest.get("revision")) != (ck["repo"], ck["revision"]):
        raise ValueError(f"model {manifest.get('model')}@{manifest.get('revision')} is not "
                         f"{ck['repo']}@{ck['revision']} ({name} in checkpoints.py)")


def record(source, key, reply_ids, valid):
    role, fmt, wording, question, replicate = json.loads(key)
    return dict(source=source, key=key, role=role, format=fmt, wording=wording, question=question,
                replicate=replicate, reply_ids=list(reply_ids), valid=bool(valid))


def in_grid_order(recs, keys):
    """Records sorted to the frozen grid; a duplicate, outside or absent key raises."""
    split = R.check_rows(recs, keys)
    if split["absent"]:
        raise ValueError(f"{len(split['absent'])} grid rows are absent, first {split['absent'][0]}")
    by = {r["key"]: r for r in recs}
    return [by[k] for k in keys]


def read_pod_a(run_dir, tok):
    run_dir = Path(run_dir)
    sums = verify_sums(run_dir / "SHA256SUMS.txt", run_dir, ("rows.jsonl", "grid.json", "manifest.json"))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    check_model(manifest, "instruct")
    grid = json.loads((run_dir / "grid.json").read_text())
    keys, grid_hash = R.freeze_grid(grid["keys"])
    if not grid_hash == grid["grid_hash"] == manifest["grid_hash"]:
        raise ValueError("grid hash differs between grid.json, its keys and the manifest")
    banned, recs = {tok.eos_token_id, tok.pad_token_id}, []
    for line in open(run_dir / "rows.jsonl"):
        row = json.loads(line)
        if E3_ONLY & set(row):
            raise ValueError(f"old E3 row schema in {run_dir}; E3 replies are read only by read_e3")
        R.check_row(row, banned, tok, manifest["markers"])
        recs.append({**record("pod_a", R.row_key(row), row["kept_ids"], row["valid"]), "prompt_ids": row["prompt_ids"]})
    recs = in_grid_order(recs, keys)
    if {r["role"] for r in recs} != set(manifest["roles"]):
        raise ValueError("row roles differ from the manifest roles")
    return recs, dict(source="pod_a", run=str(run_dir), sha256=sums, roles=manifest["roles"])


def read_e3(run_dir, tok, conds=E3_CONDS, n_questions=E3_N_QUESTIONS):
    run_dir = Path(run_dir)
    sums = verify_sums(run_dir.parent / "SHA256SUMS_run.txt", run_dir.parent,
                       (f"{run_dir.name}/replies.jsonl", f"{run_dir.name}/manifest.json"))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    check_model(manifest, "base")
    roles = manifest["pilot_order"]
    bank = [json.loads(l)["question"] for l in open(REPO / "refs/assistant-axis/extraction_questions.jsonl")]
    banned, recs = {tok.eos_token_id, tok.pad_token_id}, []
    for line in open(run_dir / "replies.jsonl"):
        r = json.loads(line)
        if not E3_ONLY <= set(r) or "kept_ids" in r:
            raise ValueError(f"not an E3 row in {run_dir}; pod A rows are read by read_pod_a")
        if r["cond"] not in conds or not r["shared_q"] or r["role"] not in roles:
            continue
        key, ids = R.item_key(r["role"], r["cond"], r["wording"], r["question_idx"], 0), r["reply_ids"]
        text = tok.decode(ids, skip_special_tokens=False)
        if r["question"] != bank[r["question_idx"]]:
            raise ValueError(f"{key}: question text differs from the question bank")
        if r["raw_ids"][:len(ids)] != ids or set(ids) & banned:
            raise ValueError(f"{key}: reply ids are not a raw-id prefix free of eos and pad")
        if text != r["reply"] or r["valid"] != R.is_valid(tok, ids) or any(s in text for s in manifest["stops"]):
            raise ValueError(f"{key}: reply ids, text, valid flag and stop markers do not agree")
        recs.append(record(f"e3_{r['cond']}", key, ids, r["valid"]))
    qs = sorted({r["question"] for r in recs})
    if len(qs) != n_questions:
        raise ValueError(f"{len(qs)} shared questions, want {n_questions}")
    keys, _ = R.freeze_grid(R.grid_keys(roles, conds, range(N_WORDINGS), qs, [0]))
    return in_grid_order(recs, keys), dict(source="e3", run=str(run_dir), sha256=sums, roles=roles, questions=qs)


def summary(recs):
    """Per source: rows, invalid rows, roles, questions, median kept length, and the lowest per-role share of
    valid replies with at least 32 kept tokens (the F4 length control)."""
    out = {}
    for src in dict.fromkeys(r["source"] for r in recs):
        rs = [r for r in recs if r["source"] == src]
        long = defaultdict(list)
        for r in rs:
            long[r["role"]].append(r["valid"] and len(r["reply_ids"]) >= 32)
        out[src] = dict(rows=len(rs), invalid=sum(not r["valid"] for r in rs), roles=len(long),
                        questions=len({r["question"] for r in rs}),
                        median_kept=statistics.median(len(r["reply_ids"]) for r in rs if r["valid"]),
                        min_role_share_32=round(min(sum(v) / len(v) for v in long.values()), 3))
    return out


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod_a", required=True, type=Path)
    ap.add_argument("--e3", default=E3_RUN, type=Path)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(CK["instruct"]["repo"], revision=CK["instruct"]["revision"])
    a_recs, a_info = read_pod_a(a.pod_a, tok)
    e_recs, e_info = read_e3(a.e3, tok)
    print(json.dumps(dict(summary=summary(a_recs + e_recs), pod_a=a_info, e3=e_info,
                          e3_questions_in_pod_a=len(set(e_info["questions"]) & {r["question"] for r in a_recs})),
                     indent=1))


if __name__ == "__main__":
    main()
