"""Real filesystem/process checks for the new wrapper; no research calculations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('podb_modal_runtime', Path(__file__).with_name('modal_runtime.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def sealed_null(folder, values):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / 'D2_null_format0.json'
    raw = json.dumps({'values': values}).encode()
    p.write_bytes(raw)
    p.with_name(p.name + '.integrity.json').write_text(json.dumps({
        'version': 1, 'artifact': p.name,
        'files': {p.name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}}))
    return p


def test_budget_cannot_restart_the_four_hour_clock(tmp_path):
    p = tmp_path / 'budget.json'
    first = M.start_budget(p, seconds=14400, now=1000)
    resumed = M.start_budget(p, seconds=14400, now=1100)
    assert first['deadline_epoch'] == resumed['deadline_epoch'] == 15400
    assert resumed['remaining_seconds'] == 14300


def test_expired_budget_refuses_more_compute(tmp_path):
    p = tmp_path / 'budget.json'
    M.start_budget(p, seconds=30, now=1000)
    with pytest.raises(RuntimeError, match='expired'):
        M.start_budget(p, seconds=30, now=1031)


@pytest.mark.parametrize('seconds', [0, -1, 14401, 1.5, True])
def test_invalid_budget_refused(tmp_path, seconds):
    with pytest.raises(ValueError):
        M.start_budget(tmp_path / 'budget.json', seconds=seconds, now=1000)


def test_budget_size_change_refused(tmp_path):
    p = tmp_path / 'budget.json'
    M.start_budget(p, seconds=30, now=1000)
    with pytest.raises(RuntimeError, match='changed'):
        M.start_budget(p, seconds=60, now=1001)


def test_only_checksummed_complete_draws_are_backed_up(tmp_path):
    p = sealed_null(tmp_path / 'out', [0.1, 0.2])
    counts = M.snapshot_nulls(p.parent, tmp_path / 'backup')
    saved = tmp_path / 'backup' / p.stem / '2' / p.name
    assert counts == {p.name: 2}
    assert saved.read_bytes() == p.read_bytes()
    assert saved.with_name(saved.name + '.integrity.json').is_file()


def test_torn_pair_does_not_replace_good_backup(tmp_path):
    p = sealed_null(tmp_path / 'out', [0.1, 0.2])
    M.snapshot_nulls(p.parent, tmp_path / 'backup')
    p.write_text('{"values":[0.1,0.2,0.3]}')
    assert M.snapshot_nulls(p.parent, tmp_path / 'backup') == {}
    assert not (tmp_path / 'backup' / p.stem / '3').exists()
    assert json.loads((tmp_path / 'backup' / p.stem / '2' / p.name).read_text())['values'] == [0.1, 0.2]


def test_missing_sidecar_is_not_a_valid_backup(tmp_path):
    p = sealed_null(tmp_path / 'out', [0.1])
    p.with_name(p.name + '.integrity.json').unlink()
    assert M.snapshot_nulls(p.parent, tmp_path / 'backup') == {}


def test_validation_is_copied_with_its_seal(tmp_path):
    p = sealed_null(tmp_path / 'out', [0.1])
    validation = p.with_name(p.stem + '.batch_validation.json')
    raw = b'{"observed_folds":[]}'
    validation.write_bytes(raw)
    sidecar = validation.with_name(validation.name + '.integrity.json')
    sidecar.write_text(json.dumps({'version': 1, 'artifact': validation.name, 'files': {
        validation.name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}}))
    M.snapshot_nulls(p.parent, tmp_path / 'backup')
    target = tmp_path / 'backup' / p.stem / '1'
    assert (target / validation.name).read_bytes() == raw
    assert (target / sidecar.name).read_bytes() == sidecar.read_bytes()


def test_time_cap_stops_child_and_commits_final_status(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    events = []
    result = M.run_capped([sys.executable, '-c', 'import time; time.sleep(60)'],
        cwd=tmp_path, out=out, status_path=tmp_path / 'status.json',
        log_path=tmp_path / 'run.log', seconds=0.15,
        commit=lambda: events.append(json.loads((tmp_path / 'status.json').read_text())['stage']), poll=0.03)
    assert result['stage'] == 'time_cap_reached'
    assert result['exit_code'] != 0
    assert result['elapsed_seconds'] < 5
    assert events[-1] == 'time_cap_reached'


def test_failed_child_is_not_reported_complete(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    result = M.run_capped([sys.executable, '-c', 'raise SystemExit(7)'],
        cwd=tmp_path, out=out, status_path=tmp_path / 'status.json',
        log_path=tmp_path / 'run.log', seconds=3, poll=0.03)
    assert result['stage'] == 'failed'
    assert result['exit_code'] == 7


def test_success_requires_successful_child_exit(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    result = M.run_capped([sys.executable, '-c', 'print("finished")'],
        cwd=tmp_path, out=out, status_path=tmp_path / 'status.json',
        log_path=tmp_path / 'run.log', seconds=3, poll=0.03)
    assert result['stage'] == 'complete'
    assert result['exit_code'] == 0
    assert 'finished' in (tmp_path / 'run.log').read_text()
