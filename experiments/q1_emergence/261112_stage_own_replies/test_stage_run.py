"""CPU check: with the SFT and DPO tokenizers, pod A's generator renders exactly the prompt ids pod A stored for
every one of its 21,000 grid keys (same chat template, same system prompts and questions). Offline cache, no model.

  HF_HUB_OFFLINE=1 .venv/bin/python -m pytest experiments/q1_emergence/261112_stage_own_replies/test_stage_run.py
"""
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage_run_mod", HERE / "stage_run.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
POD_A_ROWS = HERE.parent / "261029_instruct_persona_map/run_full/rows.jsonl"


@pytest.fixture(scope="module")
def poda_prompts():
    from manifold_persona import rows as R  # noqa: F401  (registered by load_pod_a)
    out = {}
    for line in POD_A_ROWS.open():
        r = json.loads(line)
        out[R.row_key(r)] = r["prompt_ids"]
    return out


@pytest.mark.parametrize("stage", ["sft", "dpo"])
def test_stage_renders_pod_a_prompts(stage, poda_prompts):
    G = S.load_pod_a(stage)
    assert G.MODEL == S.STAGES[stage]["repo"] and G.REVISION == S.STAGES[stage]["revision"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(G.MODEL, revision=G.REVISION)
    items = G.render(tok, G.build_items(G.ROLE_SETS["full"], G.N_REPLICATES))
    assert len(items) == len(poda_prompts) == 21000
    assert all(it["prompt_ids"] == poda_prompts[it["key"]] for it in items)


# ---------- v1.1 wrapper: receipt, resume repair, gate reading (synthetic, CPU) ----------

def _items(n=10):
    from manifold_persona import rows as R
    return [dict(key=R.item_key("pirate", "chat", 0, q, 0), prompt_ids=[1, 2, q]) for q in range(n)]


def _row(key):
    import json as _j
    role, fmt, w, q, r = _j.loads(key)
    return dict(role=role, format=fmt, wording=w, question=q, replicate=r, valid=True)


def _write_batches(out, items, bs, good_batches):
    import numpy as np
    (out / "acts_parts").mkdir(parents=True, exist_ok=True)
    lines = []
    for b in range(0, len(items), bs):
        chunk = items[b:b + bs]
        lines += [json.dumps(_row(it["key"])) for it in chunk]
        if (b // bs) in good_batches:
            np.savez(out / "acts_parts" / f"{b // bs:06d}.npz", acts=np.zeros((len(chunk), 2, 3), np.float16),
                     keys=np.array([it["key"] for it in chunk]), f32=np.array(False))
    return lines


def test_repair_adds_newline_after_a_complete_last_row(tmp_path):
    S.load_pod_a("sft")
    from manifold_persona import rows as R
    items = _items(8)
    lines = _write_batches(tmp_path, items, 4, good_batches={0, 1})
    (tmp_path / "rows.partial.jsonl").write_text("\n".join(lines))  # no final newline
    done = S.repair_partial(tmp_path, items, 4, R.row_key)
    assert done == ["added the missing newline after a complete last row"]
    assert (tmp_path / "rows.partial.jsonl").read_bytes().endswith(b"\n")


def test_repair_removes_rows_of_batches_without_state_parts(tmp_path):
    S.load_pod_a("sft")
    from manifold_persona import rows as R
    items = _items(12)
    lines = _write_batches(tmp_path, items, 4, good_batches={0, 2})  # batch 1 lost its state part
    (tmp_path / "rows.partial.jsonl").write_text("\n".join(lines) + "\n")
    done = S.repair_partial(tmp_path, items, 4, R.row_key)
    assert len(done) == 1 and "batch 1" in done[0]
    kept = [R.row_key(json.loads(l)) for l in (tmp_path / "rows.partial.jsonl").read_text().splitlines()]
    assert kept == [it["key"] for it in items[:4] + items[8:]]


def test_repair_drops_an_unreadable_state_part(tmp_path):
    S.load_pod_a("sft")
    from manifold_persona import rows as R
    items = _items(8)
    lines = _write_batches(tmp_path, items, 4, good_batches={0, 1})
    (tmp_path / "acts_parts" / "000001.npz").write_bytes(b"not a zip")
    (tmp_path / "rows.partial.jsonl").write_text("\n".join(lines) + "\n")
    done = S.repair_partial(tmp_path, items, 4, R.row_key)
    assert len(done) == 1 and not (tmp_path / "acts_parts" / "000001.npz").exists()


def test_receipt_refuses_a_changed_resume(tmp_path):
    rec = dict(stage="sft", prompts_sha256="a", code=dict(x=1), runtime=dict(torch="1"))
    assert S.check_receipt(tmp_path, rec) == "new"
    assert S.check_receipt(tmp_path, rec) == "resume"
    with pytest.raises(SystemExit):
        S.check_receipt(tmp_path, dict(rec, prompts_sha256="b"))


def test_planned_items_match_pod_a_order(poda_prompts):
    G = S.load_pod_a("dpo")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(G.MODEL, revision=G.REVISION)
    args = S.gen_args(["--out", "/tmp/x", "--roles", "canary", "--limit", "40"])
    items = S.planned_items(G, tok, args)
    assert len(items) == 40 and [len(it["prompt_ids"]) for it in items] == sorted(len(it["prompt_ids"]) for it in items)
    assert all(it["prompt_ids"] == poda_prompts[it["key"]] for it in items)


def test_receipt_leaves_the_new_run_folder_empty(tmp_path):
    """Pod A's claim_out_dir refuses a new folder that already holds files, so the receipt sits beside it."""
    out = tmp_path / "sft_smoke"
    assert S.check_receipt(out, dict(stage="sft")) == "new"
    assert not out.exists() and (tmp_path / "sft_smoke.stage_receipt.json").exists()
