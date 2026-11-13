"""Run the approved bounded A/B timing pilot on an isolated Linux pod.

The input receipt and raw-read attestation are required. This does not generate
model replies, call judges, or change frozen counts for a confirmatory run.
Sentence embeddings use the pinned reader in float32 on the pod GPU; its cache
matches each analysis module's schema and records this preparation separately.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)) + '\n')
    os.replace(tmp, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_command(command, cwd, env, log, seconds):
    start = time.monotonic()
    with Path(log).open('w') as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    return dict(command=command, exit_code=code, seconds=time.monotonic() - start,
                log=str(log), budget_seconds=seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--seconds-per-analysis', type=int, default=1800)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = root / 'repo'
    os.environ.update(HF_HOME='/workspace/.cache/huggingface', HF_HUB_CACHE=str(root / 'hf-cache'),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8',
                      MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    env = dict(os.environ, PYTHONPATH=str(repo / 'src'))
    sys.path.insert(0, env['PYTHONPATH'])
    receipt = json.loads((root / 'restore_receipt.json').read_text())
    reviewed = json.loads((root / 'raw_cases_reviewed.json').read_text())
    status = dict(stage='input_validation', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  mode='engineering timing pilot', confirmatory=False, phases={})
    def stage(name):
        status['stage'] = name
        status['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(root / 'pilot_status.json', status)
        print(json.dumps(dict(stage=name, utc=status['updated_utc'])), flush=True)
    try:
        folders = {'A': '261029_instruct_persona_map', 'B': '261029_base_role_vs_format'}
        modules = {label: load_module(repo / 'experiments/q1_emergence' / folder / 'analysis.py', 'pilot_' + label)
                   for label, folder in folders.items()}
        paths = {label: Path(receipt['restored'][label]['run']) for label in folders}
        for label, path in paths.items():
            if not receipt['restored'][label]['verified'] or not reviewed[label]['read']:
                raise RuntimeError('Verified inputs and a raw-case read are required')
            if sha(path / 'raw_read.md') != reviewed[label]['raw_read_sha256']:
                raise RuntimeError('Raw-read input changed')
        stage('G6_synthetic_checks')
        gates = {'A': modules['A'].synthetic_gate(), 'B': modules['B'].synthetic_checks()}
        save(root / 'pilot_G6.json', gates)
        if not gates['A']['passed'] or not gates['B']['pass']:
            raise RuntimeError('Synthetic gate failed')
        stage('fresh_full_input_checks')
        a_run = modules['A'].load_run(paths['A'], root / 'raw_read_attestation.md')
        b_run = modules['B'].load_verified_run(paths['B'], root / 'raw_read_attestation.md')
        save(root / 'fresh_input_checks.json', dict(A=a_run.provenance, B=b_run['verified']))
        stamp = 'analysis_pilot_261030_01a188ac'
        outputs = {label: repo / 'experiments/q1_emergence' / folders[label] / stamp for label in folders}
        for output in outputs.values():
            output.mkdir(exist_ok=True)
        stage('pinned_sentence_embeddings_on_GPU')
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        snapshot = receipt['sentence_model_snapshot']
        texts = [r['reply'] for r in a_run.rows]
        arr = outputs['A'] / 'reply_embeddings.npy'
        model = SentenceTransformer(snapshot, local_files_only=True, device='cuda')
        vectors = model.encode(texts, batch_size=128, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        if vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
            raise RuntimeError('A sentence reader failed shape/finite checks')
        with arr.with_suffix('.tmp').open('wb') as file:
            np.save(file, vectors)
        os.replace(arr.with_suffix('.tmp'), arr)
        save(outputs['A'] / 'reply_embeddings.meta.json', dict(model=modules['A'].EMBED_MODEL[0],
             revision=modules['A'].EMBED_MODEL[1], texts_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
             array_sha256=sha(arr)))
        save(root / 'embedding_precompute.json', dict(model=list(modules['A'].EMBED_MODEL), device='cuda',
             dtype='float32', normalized=True, batch_size=128, torch=torch.__version__, tf32=False,
             A_array_sha256=sha(arr), rows_sha256=sha(paths['A'] / 'rows.jsonl')))
        del model, vectors, a_run
        torch.cuda.empty_cache()
        modules['B'].embedding_cache(b_run, outputs['B'], device='cuda')
        del b_run
        torch.cuda.empty_cache()
        for label in ('A', 'B'):
            stage(label + '_analysis_running')
            command = [sys.executable, str(repo / 'experiments/q1_emergence' / folders[label] / 'analysis.py'),
                       '--run', str(paths[label]), '--out', str(outputs[label]), '--execute-on-pod',
                       '--raw-read-note', str(root / 'raw_read_attestation.md'), '--n-boot', '20', '--n-halves', '2']
            if label == 'A':
                command += ['--engineering', '--n-perm', '99', '--embedding-model-path', snapshot]
            else:
                command += ['--allow-nonconfirmatory', '--n-perm', '19', '--n-splits', '2', '--embedding-device', 'cuda']
            result = run_command(command, repo, env, root / (label + '_analysis.log'), args.seconds_per_analysis)
            result['out'] = str(outputs[label])
            status['phases'][label] = result
            save(root / 'pilot_status.json', status)
            if result['exit_code']:
                raise RuntimeError(label + ' analysis stopped with exit ' + str(result['exit_code']))
        stage('pilot_complete')
    except Exception as error:
        status['error'] = dict(type=type(error).__name__, message=str(error))
        stage('failed')
        raise


if __name__ == '__main__':
    main()
