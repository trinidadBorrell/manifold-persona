"""Offline tests for checkpoints.py: the saved manifest passes G2, and each kind of damage fails it."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("checkpoints", HERE / "checkpoints.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


@pytest.fixture
def manifest():
    return json.loads(C.MANIFEST.read_text())


def test_saved_manifest_passes(manifest):
    assert C.check(manifest) == []
    assert len(manifest["entries"]) == 10
    assert len({(e["repo"], e["revision"]) for e in manifest["entries"]}) == 10
    assert all(e["branch_head_at_resolve"] == e["revision"] for e in manifest["entries"])


def test_step_360_is_not_the_final_instruct(manifest):
    cmp = manifest["step_360_vs_instruct"]
    assert cmp["weight_files_identical"] is False and not all(cmp["probe_tensors"].values())


def damage(m, name, path, value):
    m = copy.deepcopy(m)
    e = next(e for e in m["entries"] if e["name"] == name)
    *keys, last = path
    for k in keys:
        e = e[k]
    if value is None:
        del e[last]
    else:
        e[last] = value
    return m


@pytest.mark.parametrize("name,path,value,why", [
    ("sft", ["revision"], "1de02c0", "differs from CHECKPOINTS"),
    ("rl_60", ["branch"], "step_120", "differs from CHECKPOINTS"),
    ("dpo", ["tokenizer_files", "tokenizer.json"], "0" * 64, "tokenizer.json differs"),
    ("rl_360", ["tokenizer_files", "merges.txt"], None, "tokenizer files missing"),
    ("base", ["sample_ids_sha256"], "x", "different ids"),
    ("instruct", ["config", "vocab_size"], 100278, "config vocab_size"),
    ("instruct", ["weights"], {}, "no weight files"),
])
def test_damage_fails(manifest, name, path, value, why):
    problems = C.check(damage(manifest, name, path, value))
    assert any(why in p for p in problems), problems


def test_entry_order_and_comparison_required(manifest):
    m = copy.deepcopy(manifest)
    m["entries"] = m["entries"][1:]
    assert "differ from CHECKPOINTS" in C.check(m)[0]
    m = copy.deepcopy(manifest)
    del m["step_360_vs_instruct"]
    assert any("not recorded" in p for p in C.check(m))
