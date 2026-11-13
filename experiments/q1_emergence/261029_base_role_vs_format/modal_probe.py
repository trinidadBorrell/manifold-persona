"""Bounded real-data compatibility/timing check; it emits no scientific verdict."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('podb_modal_probe_target', HERE / 'analysis.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def main(run_path, note, destination):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    run = A.load_verified_run(run_path, note)
    cells = A.prompt_cells(run['rows'], run['acts'][:, A.STATE], run['roles'], run['formats'], run['questions'])
    p = A.flatten_cells(cells)
    folds = A.make_splits(p, p['F'], A.N_SPLITS, A.SEED)[0][:3]
    B = A.sibling('analysis_batch_lbfgs')
    labels = np.stack([A.block_permute(p['y'], p['q'], np.random.default_rng(A.SEED + 1 + k))
                       for k in range(256)])
    matches, timings, iterations = [], [], []
    with tempfile.TemporaryDirectory(prefix='podb-modal-probe-') as temp:
        prepared = A.prepare_null_folds(p['X'], p, folds, Path(temp))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            for (tr, te), pair in zip(folds, prepared):
                pred, info = B.fit_predict_batch(pair['Xtr'], labels[:32, tr], pair['Xte'], device='cuda')
                for k in range(32):
                    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
                    clf.fit(pair['Xtr'], labels[k, tr])
                    matches.append(bool(np.array_equal(pred[k], clf.predict(pair['Xte']))))
                    iterations.append(int(clf.n_iter_[0]) == info['n_iter'][k])
                # Time complete, full-width fresh fits on three original folds.
                torch.cuda.synchronize()
                start = time.perf_counter()
                B.fit_predict_batch(pair['Xtr'], labels[:, tr], pair['Xte'], device='cuda')
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
    proof = dict(passed=all(matches) and all(iterations), prediction_sets=len(matches),
                 all_predictions_match=all(matches), all_iteration_counts_match=all(iterations),
                 batch_width=256, timed_fits=768, batch_seconds=timings,
                 full_null_projected_seconds=1999800 * max(timings) / 256,
                 projection_limit='Slowest of three folds; excludes other phases and is not a completion guarantee.',
                 actual_feature_shape=list(p['X'].shape), train_test_sizes=[[len(tr), len(te)] for tr, te in folds],
                 GPU=A.sibling('analysis_gpu').runtime_info(), source_sha256=A.streamed_sha256(HERE / 'analysis.py'),
                 elapsed_seconds=time.perf_counter() - started, scientific_verdict=None)
    A.atomic_json(destination, proof)
    print(json.dumps(proof), flush=True)
    if not proof['passed']:
        raise RuntimeError('Original CPU and batched GPU classifier predictions/iterations differ')


if __name__ == '__main__':
    main(*sys.argv[1:])
