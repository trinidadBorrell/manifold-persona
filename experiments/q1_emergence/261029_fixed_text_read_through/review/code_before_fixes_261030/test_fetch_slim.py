"""Offline tests for fetch_slim.py and the hf_archive path prefix: a fake receipt points at local files; the slim
copy keeps exactly the analysis states, and a file that differs from its receipt is refused."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_slim as S  # noqa: E402

R = S.R
spec = importlib.util.spec_from_file_location("hf_archive", HERE.parent / "261029_stage_persona_plan/hf_archive.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def fake_checkpoint(d):
    """One checkpoint folder as extract.py writes it: F1 at all 33 states, F3 at 4."""
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(d / "pod_a.F1.reply.npy", rng.standard_normal((5, 33, 3)).astype(np.float16))
    np.save(d / "pod_a.F1.first32.npy", rng.standard_normal((5, 4, 3)).astype(np.float16))
    np.save(d / "pod_a.F3.neutral.npy", rng.standard_normal((5, 4, 3)).astype(np.float16))
    (d / "pod_a.F1.json").write_text(json.dumps(dict(layers=dict(reply=list(range(33)), first32=[0, 9, 17, 23]),
                                                     fed=5, g1_rows_checked=5)))
    (d / "pod_a.F3.json").write_text(json.dumps(dict(layers=dict(neutral=[0, 9, 17, 23]), fed=5, g1_rows_checked=5)))
    (d / "manifest.json").write_text("{}")
    names = sorted(p.name for p in d.iterdir())
    (d / "SHA256SUMS.txt").write_text("".join(f"{R.file_sha256(d / n)}  {n}\n" for n in names))
    (d.parent / f"{d.name}.log").write_text("log\n")
    return d


def receipt_for(d, prefix):
    paths = H.remote_paths(d, prefix)
    return dict(repo="x", revision="r", verified=True,
                files=[dict(path=k, sha256=R.file_sha256(p), local=str(p)) for k, p in paths.items()])


def local_fetch(receipt, path, tmp):
    f = next(x for x in receipt["files"] if x["path"] == path)
    out = Path(tmp) / Path(path).name
    shutil.copy(f["local"], out)
    if R.file_sha256(out) != f["sha256"]:
        raise SystemExit(f"{path}: download does not match the receipt")
    return out


def test_prefix_paths(tmp_path):
    d = fake_checkpoint(tmp_path / "out" / "base")
    paths = H.remote_paths(d, "experiments/q1_emergence/261029_fixed_text_read_through/run_full")
    assert "experiments/q1_emergence/261029_fixed_text_read_through/run_full/base/pod_a.F1.reply.npy" in paths
    assert "experiments/q1_emergence/261029_fixed_text_read_through/run_full/base.log" in paths
    assert all(k.startswith("experiments/q1_emergence/out/") for k in H.remote_paths(d))  # the old default


def test_slim_copy_keeps_the_analysis_states(tmp_path):
    d = fake_checkpoint(tmp_path / "out" / "base")
    out = tmp_path / "slim" / "base"
    S.slim_checkpoint(receipt_for(d, "p"), out, fetch=local_fetch)
    full, slim = np.load(d / "pod_a.F1.reply.npy"), np.load(out / "pod_a.F1.reply.npy")
    assert slim.shape == (5, 4, 3) and np.array_equal(slim, full[:, [0, 9, 17, 23]])
    assert json.loads((out / "pod_a.F1.json").read_text())["layers"]["reply"] == [0, 9, 17, 23]
    assert np.array_equal(np.load(out / "pod_a.F3.neutral.npy"), np.load(d / "pod_a.F3.neutral.npy"))
    for line in (out / "SHA256SUMS.txt").read_text().splitlines():
        h, n = line.split("  ", 1)
        assert R.file_sha256(out / n) == h
    assert json.loads((out / "provenance.json").read_text())["revision"] == "r"


def test_changed_download_is_refused(tmp_path):
    d = fake_checkpoint(tmp_path / "out" / "base")
    rec = receipt_for(d, "p")
    np.save(d / "pod_a.F3.neutral.npy", np.zeros((5, 4, 3), np.float16))  # differs from the receipt
    with pytest.raises(SystemExit, match="does not match"):
        S.slim_checkpoint(rec, tmp_path / "slim" / "base", fetch=local_fetch)
