"""Small process-budget and checkpoint helpers for the Modal execution wrapper."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def start_budget(path, seconds=14400, now=None):
    """A resumed invocation inherits the original deadline, including any retries."""
    if type(seconds) is not int or not 0 < seconds <= 14400:
        raise ValueError('Budget must be an integer from 1 to 14400 seconds')
    path = Path(path)
    now = time.time() if now is None else now
    if path.exists():
        budget = json.loads(path.read_text())
        if budget['max_runtime_seconds'] != seconds:
            raise RuntimeError('Compute budget changed; a new decision is required')
    else:
        budget = dict(started_epoch=now, deadline_epoch=now + seconds,
                      max_runtime_seconds=seconds)
        atomic_json(path, budget)
    remaining = budget['deadline_epoch'] - now
    if remaining <= 0:
        raise RuntimeError('The original compute budget has expired')
    return dict(budget, remaining_seconds=remaining)


def snapshot_nulls(out, backup):
    """Retain immutable copies of verified prefixes; ignore a transient torn pair."""
    counts = {}
    for p in sorted(Path(out).glob('D2_null_format[0-3].json')):
        sidecar = p.with_name(p.name + '.integrity.json')
        try:
            raw, seal = p.read_bytes(), sidecar.read_bytes()
            metadata = json.loads(seal)
            expected = metadata['files'][p.name]
            if (metadata.get('version') != 1 or metadata.get('artifact') != p.name
                    or set(metadata['files']) != {p.name}
                    or len(raw) != expected['bytes']
                    or hashlib.sha256(raw).hexdigest() != expected['sha256']):
                continue
            count = len(json.loads(raw)['values'])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        target = Path(backup) / p.stem / str(count)
        target.mkdir(parents=True, exist_ok=True)
        destination = target / p.name
        if destination.exists() and destination.read_bytes() != raw:
            raise RuntimeError('A saved prefix changed without changing its draw count')
        destination.write_bytes(raw)
        (target / sidecar.name).write_bytes(seal)
        validation = p.with_name(p.stem + '.batch_validation.json')
        validation_seal = validation.with_name(validation.name + '.integrity.json')
        try:
            vraw, vseal = validation.read_bytes(), validation_seal.read_bytes()
            meta = json.loads(vseal)
            item = meta['files'][validation.name]
            if (meta.get('version') == 1 and meta.get('artifact') == validation.name
                    and set(meta['files']) == {validation.name} and len(vraw) == item['bytes']
                    and hashlib.sha256(vraw).hexdigest() == item['sha256']):
                (target / validation.name).write_bytes(vraw)
                (target / validation_seal.name).write_bytes(vseal)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        counts[p.name] = count
    return counts


def run_capped(command, *, cwd, out, status_path, log_path, seconds, commit=None, poll=5):
    """Run one process group and commit status/checkpoint prefixes until it exits."""
    if seconds <= 0 or poll <= 0:
        raise ValueError('Remaining time and polling interval must be positive')
    out, status_path = Path(out), Path(status_path)
    started = time.monotonic()
    status = dict(stage='launching', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  remaining_seconds=seconds, command=command, out=str(out))
    atomic_json(status_path, status)
    process = None

    def save():
        status['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status['elapsed_seconds'] = time.monotonic() - started
        status['null_draw_counts'] = snapshot_nulls(out, out.parent / 'verified_prefixes')
        status['phase_files'] = [p.name for p in sorted(out.glob('*.json'))
                                 if not p.name.endswith('.integrity.json')]
        atomic_json(status_path, status)
        if commit:
            commit()

    def stop():
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    try:
        with Path(log_path).open('a') as log:
            process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            status.update(stage='running', pid=process.pid)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= seconds:
                    status['stage'] = 'time_cap_reached'
                    stop()
                    break
                save()
                time.sleep(min(poll, max(0.001, seconds - (time.monotonic() - started))))
            rc = process.wait()
            status.update(exit_code=rc)
            if status['stage'] != 'time_cap_reached':
                status['stage'] = 'complete' if rc == 0 else 'failed'
    except BaseException as error:
        stop()
        status.update(stage='wrapper_failed', error_type=type(error).__name__)
        save()
        raise
    finally:
        stop()
    save()
    return status
