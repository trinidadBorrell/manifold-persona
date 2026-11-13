"""Run unchanged, checked Pod B analysis on Modal with a persistent four-hour budget.

Build the small source bundle first: python build_modal_payload.py
Then: modal run --detach modal_analysis.py --phase prepare|preflight|full
No HF upload, judge call, new generation, or automatic budget renewal is performed.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

import modal

APP_NAME = 'persona-podb-analysis-261104'
VOLUME_NAME = 'persona-podb-analysis-261104'
ROOT = Path('/data/podb-261104-01a188ac')
SOURCE = Path('/opt/podb')
REPO = SOURCE / 'repo'
B = REPO / 'experiments/q1_emergence/261029_base_role_vs_format'
INPUT_PREFIX = 'experiments/q1_emergence/261029_base_role_vs_format/run_full'
RUN = ROOT / 'inputs' / INPUT_PREFIX
OUT = ROOT / 'analysis_full_batch_v3_cuda_261105'
PAYLOAD = Path('/private/tmp/podb-modal-261104/payload.tar.gz')
MAX_SECONDS = 14400
GPU_TYPE = os.environ.get('PODB_MODAL_GPU', 'A100-80GB')
if GPU_TYPE not in ('A100-80GB', 'H100'):
    raise ValueError('Only the explicitly checked A100-80GB and H100 candidates are allowed')
ENV = dict(PYTHONPATH='/opt/podb/repo/src', HF_HOME=str(ROOT / 'hf'), HF_HUB_CACHE=str(ROOT / 'hf/hub'),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', SCIPY_ARRAY_API='1',
           CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
           HF_HUB_DISABLE_PROGRESS_BARS='1')

image = (modal.Image.debian_slim(python_version='3.12').apt_install('git')
         .pip_install('torch==2.13.0+cu129', index_url='https://download.pytorch.org/whl/cu129')
         .pip_install('numpy==2.5.2', 'scipy==1.18.1', 'scikit-learn==1.9.1',
                      'transformers==4.47.1', 'sentence-transformers==5.7.0',
                      'joblib==1.6.0', 'threadpoolctl==3.6.0', 'pytest', 'array-api-compat')
         .add_local_file(PAYLOAD, '/opt/podb/payload.tar.gz', copy=True)
         .run_commands('tar -xzf /opt/podb/payload.tar.gz -C /opt/podb')
         .env(ENV))
app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


def helpers():
    sys.path.insert(0, str(B))
    import modal_runtime as M
    return M


def verify_source():
    manifest = json.loads((SOURCE / 'source_manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        p = SOURCE / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Packaged source checksum differs: ' + name)
    return manifest


@app.function(cpu=(2, 2), memory=(16384, 16384), timeout=1800, retries=0,
              max_containers=1, scaledown_window=2, volumes={'/data': volume},
              secrets=[modal.Secret.from_name('hf-read', required_keys=['HF_TOKEN'])])
def prepare():
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    M = helpers()
    verify_source()
    ROOT.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((SOURCE / 'input_receipt.json').read_text())
    if receipt['revision'] != '969f11a14a61cc1888d9532d5dd1a5c8cd8eba1f':
        raise RuntimeError('Unexpected frozen input revision')
    M.atomic_json(ROOT / 'prepare_status.json', {'stage': 'downloading_frozen_inputs', 'revision': receipt['revision']})
    volume.commit()
    snapshot_download(receipt['repo'], repo_type='dataset', revision=receipt['revision'],
                      allow_patterns=[f['path'] for f in receipt['files']], local_dir=str(ROOT / 'inputs'),
                      max_workers=4)
    verified = []
    for item in receipt['files']:
        p = ROOT / 'inputs' / item['path']
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        if p.stat().st_size != item['size'] or h.hexdigest() != item['sha256']:
            raise RuntimeError('Restored input checksum differs: ' + item['path'])
        verified.append({'path': item['path'], 'sha256': h.hexdigest(), 'bytes': p.stat().st_size})
    cache = str(ROOT / 'hf/hub')
    for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        hf_hub_download('allenai/OLMo-2-1124-7B', name,
                        revision='7df9a82518afdecae4e8c026b27adccc8c1f0032', cache_dir=cache)
    model, revision = 'sentence-transformers/all-mpnet-base-v2', 'e8c3b32edf5434bc2275fc9bab85f82640a19130'
    names = {x.rfilename for x in HfApi().model_info(model, revision=revision).siblings}
    weights = 'model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
    patterns = [name for name in names if '/' not in name and (name.endswith(('.json', '.txt', '.model')) or name == weights)]
    patterns += [name for name in names if name.startswith('1_Pooling/') and name.endswith('.json')]
    snapshot_download(model, revision=revision, allow_patterns=patterns, cache_dir=cache)
    # Carry forward the exact archived raw-read record; return its raw cases for inspection.
    raw = (RUN / 'raw_read.md').read_text()
    note = ROOT / 'raw_read_attestation.md'
    note.write_text('Frozen input and raw cases were read in Codex session 01a188ac on 30 Oct.\n'
                    'This rerun uses the same archived generation files and the following unchanged raw-read record.\n\n' + raw)
    proof = dict(stage='prepared', verified_files=len(verified), files=verified, revision=receipt['revision'],
                 raw_read_sha256=hashlib.sha256(raw.encode()).hexdigest(), raw_read_text=raw,
                 source_manifest_sha256=hashlib.sha256((SOURCE / 'source_manifest.json').read_bytes()).hexdigest())
    M.atomic_json(ROOT / 'prepared.json', proof)
    M.atomic_json(ROOT / 'prepare_status.json', {'stage': 'prepared', 'verified_files': len(verified)})
    volume.commit()
    return proof


@app.function(gpu=GPU_TYPE, cpu=(8, 8), memory=(131072, 131072),
              timeout=14400, retries=0, max_containers=1,
              scaledown_window=2, volumes={'/data': volume},
              env={'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PODB_MODAL_GPU': GPU_TYPE})
def analyze(phase):
    import time
    import torch
    import importlib.util
    M = helpers()
    verify_source()
    prepared = json.loads((ROOT / 'prepared.json').read_text())
    if prepared['verified_files'] != 21:
        raise RuntimeError('Complete input receipt is required')
    if not torch.cuda.is_available() or not any(name in torch.cuda.get_device_name(0) for name in ('A100', 'H100')):
        raise RuntimeError('A checked A100 or H100 CUDA execution device is required')
    if torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
        raise RuntimeError('Expected the 80 GiB A100')
    budget = M.start_budget(ROOT / 'gpu_budget.json', MAX_SECONDS)
    volume.commit()

    def remaining():
        return M.start_budget(ROOT / 'gpu_budget.json', MAX_SECONDS)['remaining_seconds']

    if phase == 'preflight':
        work = ROOT / ('preflight_' + GPU_TYPE)
        work.mkdir(exist_ok=False)
        test_names = ['test_podb_analysis.py', 'test_podb_variance_fast.py', 'test_podb_gpu.py',
                      'test_podb_batch_loss.py', 'test_podb_batch_device.py', 'test_podb_batch_lbfgs.py',
                      'test_modal_runtime.py']
        tests = work / 'tests'
        tests.mkdir()
        command = [sys.executable, '-m', 'pytest', *[str(B / name) for name in test_names], '-q',
                   '-W', 'error::pytest.PytestUnhandledThreadExceptionWarning']
        result = M.run_capped(command, cwd=REPO, out=tests, status_path=ROOT / 'tests_status.json',
                              log_path=work / 'tests.log', seconds=min(900, remaining()-120), commit=volume.commit)
        if result['stage'] != 'complete' or '216 passed' not in (work / 'tests.log').read_text():
            raise RuntimeError('The complete 216-test apparatus check did not pass')
        probe = work / 'probe'
        probe.mkdir()
        result = M.run_capped([sys.executable, str(B / 'modal_probe.py'), str(RUN),
                               str(ROOT / 'raw_read_attestation.md'), str(probe / 'verification.json')],
            cwd=REPO, out=probe, status_path=ROOT / 'probe_status.json', log_path=work / 'probe.log',
            seconds=min(900, remaining()-120), commit=volume.commit)
        if result['stage'] != 'complete':
            raise RuntimeError('The real-data pilot did not complete')
        proof = json.loads((probe / 'verification.json').read_text())
        proof['tests_passed'] = 216
        proof['requested_GPU'] = GPU_TYPE
        proof['remaining_budget_seconds'] = remaining()
        proof['ready_for_full'] = proof['passed'] and proof['full_null_projected_seconds'] + 1200 < remaining()
        M.atomic_json(ROOT / 'preflight.json', proof)
        volume.commit()
        return proof
    if phase != 'full':
        raise ValueError('GPU phase must be preflight or full')
    proof = json.loads((ROOT / 'preflight.json').read_text())
    if not proof['ready_for_full'] or not proof['passed'] or proof['tests_passed'] != 216:
        raise RuntimeError('A successful compatibility/timing pilot is required')
    if proof['GPU']['device'] != torch.cuda.get_device_name(0):
        raise RuntimeError('Full execution device differs from the successful timing pilot')
    if OUT.exists():
        raise RuntimeError('Full output already exists; inspect the saved job instead of launching a duplicate')
    OUT.mkdir()
    command = [sys.executable, str(B / 'analysis.py'), '--run', str(RUN), '--out', str(OUT),
               '--execute-on-pod', '--raw-read-note', str(ROOT / 'raw_read_attestation.md'),
               '--embedding-device', 'cuda',
               '--n-boot', '2000', '--n-perm', '9999', '--n-splits', '50', '--n-halves', '50',
               '--null-backend', 'batches', '--null-workers', '256', '--null-blas-threads', '1',
               '--cache-null-scalers', '--null-device', 'cuda']
    M.atomic_json(ROOT / 'full_spec.json', dict(command=command, input_revision=prepared['revision'],
                  budget=budget, test_counts_unchanged=True, paid_judges=False, HF_upload=False))
    volume.commit()
    result = M.run_capped(command, cwd=REPO, out=OUT, status_path=ROOT / 'full_status.json',
                          log_path=ROOT / 'full.log', seconds=remaining()-120, commit=volume.commit, poll=30)
    if result['stage'] == 'complete':
        spec = importlib.util.spec_from_file_location('podb_modal_finished', B / 'analysis.py')
        A = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(A)
        phases = ['integrity', 'D1', 'all_valid_fit', 'D2', 'D3', 'sensitivities']
        for name in phases:
            A.checked_json(OUT / (name + '.json'))
        counts = {}
        for f in range(4):
            path = OUT / f'D2_null_format{f}.json'
            counts[path.name] = len(A.checked_json(path)['values'])
            if counts[path.name] != 9999:
                raise RuntimeError('A full null draw prefix is incomplete')
        meta = json.loads((OUT / 'provenance.json').read_text())
        decisions = json.loads((OUT / 'decisions.json').read_text())
        if not meta['confirmatory'] or not decisions['confirmatory']:
            raise RuntimeError('Full frozen test counts were not used')
        M.atomic_json(ROOT / 'full_verification.json', dict(passed=True, null_draw_counts=counts,
                      verified_phases=phases, family_status=decisions['family_status'],
                      decisions_sha256=A.streamed_sha256(OUT / 'decisions.json')))
        volume.commit()
    return result


@app.local_entrypoint()
def main(phase: str = 'prepare'):
    if phase not in ('prepare', 'preflight', 'full'):
        raise ValueError('Phase must be prepare, preflight or full')
    call = prepare.spawn() if phase == 'prepare' else analyze.spawn(phase)
    record = dict(app=APP_NAME, volume=VOLUME_NAME, phase=phase, function_call_id=call.object_id,
                  requested_GPU=GPU_TYPE, max_gpu_seconds=MAX_SECONDS, automatic_HF_upload=False)
    suffix = '_' + GPU_TYPE if phase == 'preflight' else ''
    local = PAYLOAD.parent / (phase + suffix + '_call.json')
    local.write_text(json.dumps(record, indent=2) + '\n')
    print('MODAL_CALL ' + json.dumps(record), flush=True)
    result = call.get()
    (PAYLOAD.parent / (phase + suffix + '_result.json')).write_text(json.dumps(result, indent=2) + '\n')
    # Raw cases stay in the local result for inspection, not in routine progress logs.
    print(json.dumps({key: value for key, value in result.items() if key not in ('raw_read_text', 'files')}), flush=True)
