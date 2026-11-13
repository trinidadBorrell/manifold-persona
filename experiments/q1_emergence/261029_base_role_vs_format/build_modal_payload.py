"""Build a small source-only payload from checked historical sources, never arrays."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RECORDS = Path('/Users/brendon/research-log/manifold-persona-log/Reference/2026-10-30 A-B pilot results')
DEST = Path('/private/tmp/podb-modal-261104/payload.tar.gz')
B = 'experiments/q1_emergence/261029_base_role_vs_format/'


def archive_files(path, strip=''):
    result = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or '..' in PurePosixPath(member.name).parts or member.name.startswith('/'):
                raise RuntimeError('Unexpected member in frozen source archive')
            name = member.name.removeprefix(strip)
            result[name] = archive.extractfile(member).read()
    return result


def build():
    initial = archive_files(RECORDS / 'source_snapshots/initial_deployment.tar.gz', 'repo/')
    # The outer manifest was updated after the original launch. The archive's
    # own manifest seals the initial bytes; subsequent upgrades have their own seals.
    recorded = json.loads(initial['source_manifest.json'])
    assert set(initial) - set(recorded) == {'input_receipts/A.json', 'input_receipts/B.json', 'source_manifest.json'}
    assert not set(recorded) - set(initial)
    initial = {name: content for name, content in initial.items() if name in recorded}
    for name, content in initial.items():
        assert hashlib.sha256(content).hexdigest() == recorded[name], name
    # Restore each recorded upgrade in its original order.
    fast = archive_files(RECORDS / 'B_fast_v2/source_snapshots/B_fast_v2_source_261030.tar.gz')
    fast_hashes = json.loads((RECORDS / 'B_fast_v2/source_snapshots/B_fast_v2_source_hashes_261030.json').read_text())
    for name, content in fast.items():
        wanted = fast_hashes.get(name, fast_hashes.get(Path(name).name))
        assert hashlib.sha256(content).hexdigest() == wanted, name
        initial[name] = content
    batch = archive_files(RECORDS / 'B_true_batch/source_snapshots/B_true_batch_production_source_final_261030.tar.gz')
    batch_hashes = json.loads((RECORDS / 'B_true_batch/source_snapshots/B_true_batch_production_source_final_hashes_261030.json').read_text())
    for name, content in batch.items():
        assert hashlib.sha256(content).hexdigest() == batch_hashes[name], name
        initial[B + name] = content
    # These wrappers are new; all research source bytes above remain unchanged.
    for name in ['modal_runtime.py', 'test_modal_runtime.py', 'modal_probe.py', 'modal_analysis.py']:
        initial[B + name] = (HERE / name).read_bytes()
    files = {'repo/' + name: content for name, content in initial.items()}
    files['input_receipt.json'] = (HERE / 'hf_receipts/run_full.json').read_bytes()
    metadata = dict(input_revision='969f11a14a61cc1888d9532d5dd1a5c8cd8eba1f',
                    source_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                    local_git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                    local_git_status=subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True),
                    scientific_source='Checked 30 Oct initial + GPU + final batch source archives',
                    new_wrappers=['modal_runtime.py', 'test_modal_runtime.py', 'modal_probe.py', 'modal_analysis.py'])
    files['source_manifest.json'] = (json.dumps(metadata, indent=2) + '\n').encode()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(DEST, 'w:gz') as archive:
        for name, content in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
    proof = dict(path=str(DEST), bytes=DEST.stat().st_size, files=len(files),
                 sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(), contains_activation_arrays=False)
    (DEST.parent / 'payload_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(proof))
    return proof


if __name__ == '__main__':
    build()
