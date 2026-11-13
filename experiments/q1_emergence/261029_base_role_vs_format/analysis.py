"""Frozen Pod B D1--D3 analysis. Run on the pod; synthetic checks need no data.

  python analysis.py --synthetic-only
  python analysis.py --run RUN --execute-on-pod --raw-read-note READ_NOTE

The frozen balanced D1 decomposition has equal prompt weights after replicate averaging.
The descriptive all-valid fit reports rank and RSS only: a nonorthogonal SS allocation
was not specified. D3 uses the same 50 question/wording halves throughout. A missing
half or bootstrap draw makes the interval unavailable; it is never silently omitted.
E3's balanced C=1 logistic regression is refit on every fold of every null draw.
Optional scaler caching fits only on a fold's training X and reuses those exact
label-independent transforms; workers share read-only features or temporary memmaps.
The default null remains serial with a fresh scaler on every fit. Reduced run counts are explicitly non-confirmatory.
No model, judge, HF array, or sentence model is downloaded by this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from scipy import sparse
from scipy.stats import rankdata, kstest

HERE = Path(__file__).resolve().parent
N_BOOT, N_PERM, N_SPLITS, N_HALVES, SEED = 2000, 9999, 50, 50, 20261029
STATE, EXTRA_STATE = 17, 16
# Historical generation helper, frozen in the approved 29/10 run manifests.
GENERATION_ROWS_SHA256 = 'b1e4c29bba60a49b785b4847c818ffff623dd5e96253b05e03df8df2a9eb0c4a'
EMBED_MODEL = ('sentence-transformers/all-mpnet-base-v2', 'e8c3b32edf5434bc2275fc9bab85f82640a19130')
TEXT = {'word': dict(analyzer='word', ngram_range=(1,2), min_df=2, max_features=20000, sublinear_tf=True),
        'char': dict(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=20000, sublinear_tf=True)}
TERMS = ('role','format','role_format','question','role_question','format_question','wording_within_role','residual')
METHOD_LIMITS = [
    'The frozen design omits the Holm family alpha. Adjusted p values are reported, but the family verdict '
    'stays pending unless --holm-alpha is explicitly supplied.',
    'All-valid unequal-weight fit reports full-model rank and RSS; no nonorthogonal term shares were frozen.',
    'Undefined fixed halves or bootstrap draws make the interval unavailable; counts are reported.',
    'Intervals hold five wordings fixed and use observed replicate means; they exclude wording/replicate uncertainty.',
    'Gate JSON has no built-in source fingerprint; its saved evidence is checked against current counts and shapes, '
    'but G3/G4 execution provenance depends on the supplied verified run and raw-read attestation.',
]


class AnalysisError(ValueError):
    pass


def sibling(name):
    key = f'podb_analysis_{name}'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f'{name}.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def map_module():
    key = 'podb_readonly_podc_maps'
    if key not in sys.modules:
        p = HERE.parent / '261029_fixed_text_read_through/maps.py'
        spec = importlib.util.spec_from_file_location(key, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False))
    os.replace(tmp, path)


def artifact_sidecar(path):
    path=Path(path)
    return path.with_name(path.name+'.integrity.json')


def streamed_sha256(path):
    """Bounded memory: never load a saved array or large checkpoint merely to hash it."""
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def seal_artifact(path,dependencies=()):
    """Seal a completed artifact and its immediate same-folder dependencies atomically."""
    path=Path(path)
    folder=path.parent.resolve()
    names=sorted(set(Path(d).name for d in dependencies))
    files={}
    for name in [path.name]+names:
        file=path.parent/name
        if file.resolve().parent!=folder or not file.is_file():
            raise AnalysisError(f'integrity: artifact is missing or outside its folder: {name}')
        files[name]=dict(sha256=streamed_sha256(file),bytes=file.stat().st_size)
    atomic_json(artifact_sidecar(path),dict(version=1,artifact=path.name,files=files))


def verify_artifact(path,dependencies=None):
    """Reject missing, truncated, changed, or unsealed saved work before any resume use."""
    path=Path(path)
    try:
        saved=json.loads(artifact_sidecar(path).read_text())
        if saved.get('version')!=1 or saved.get('artifact')!=path.name or path.name not in saved['files']:
            raise AnalysisError(f'integrity: invalid sidecar for {path.name}')
        if dependencies is not None and set(saved['files'])!={path.name,*map(str,dependencies)}:
            raise AnalysisError(f'integrity: dependency set differs for {path.name}')
        folder=path.parent.resolve()
        for name,want in saved['files'].items():
            file=path.parent/name
            if Path(name).name!=name or file.resolve().parent!=folder or not file.is_file():
                raise AnalysisError(f'integrity: missing or unsafe artifact {name}')
            if file.stat().st_size!=want['bytes'] or streamed_sha256(file)!=want['sha256']:
                raise AnalysisError(f'integrity checksum differs: {name}')
    except (OSError,ValueError,KeyError,TypeError) as e:
        if isinstance(e,AnalysisError):
            raise
        raise AnalysisError(f'integrity: invalid or missing sidecar for {path.name}: {e}') from e


def checked_json(path,dependencies=None):
    verify_artifact(path,dependencies)
    try:
        return json.loads(Path(path).read_text())
    except (OSError,ValueError) as e:
        raise AnalysisError(f'integrity: invalid JSON artifact {Path(path).name}: {e}') from e


def save_checked_json(path,value,dependencies=()):
    atomic_json(path,value)
    seal_artifact(path,dependencies)


def artifact_exists(path):
    return Path(path).exists() or artifact_sidecar(path).exists()


def phase_dependencies(out,name):
    """Artifacts needed to audit/reuse this phase; the sidecar itself is never its own dependency."""
    out=Path(out)
    if name=='D1':
        names=['D1_bootstrap.npy']
    elif name=='D3':
        names=['D3_resampling.npz']
    elif name=='D2':
        names=[p.name for p in out.glob('D2_*') if not p.name.endswith('.integrity.json')]
    else:
        names=[]
    return sorted(n for n in names if (out/n).is_file())


def interval(observed, draws):
    draws = np.asarray(draws, float)
    bad = int((~np.isfinite(draws)).sum())
    ok = len(draws) and not bad
    return dict(point=float(observed) if np.isfinite(observed) else None,
                lo=float(np.percentile(draws,2.5)) if ok else None,
                hi=float(np.percentile(draws,97.5)) if ok else None,
                n_draws=len(draws), n_undefined=bad)


def confirmatory_parameters(n_boot, n_perm, n_splits, n_halves):
    return (n_boot,n_perm,n_splits,n_halves) == (N_BOOT,N_PERM,N_SPLITS,N_HALVES)


def prompt_cells(rows, X, roles, formats, questions, n_wordings=5, mask=None):
    """[role,format,question,wording,feature] means; absent prompts are NaN, never zero/imputed."""
    if X.shape[0] != len(rows) or X.ndim != 2:
        raise AnalysisError('feature rows do not match rows')
    lookup = [{v:i for i,v in enumerate(ids)} for ids in (roles,formats,questions)]
    shape = (len(roles),len(formats),len(questions),n_wordings)
    sums, counts = np.zeros(shape + (X.shape[1],),np.float64), np.zeros(shape,int)
    groups = {}
    for i,r in enumerate(rows):
        c = (lookup[0][r['role']],lookup[1][r['format']],lookup[2][r['question']],r['wording'])
        if not 0 <= c[3] < n_wordings:
            raise AnalysisError('wording outside frozen grid')
        if r['valid'] and (mask is None or mask[i]) and np.isfinite(X[i]).all():
            sums[c] += X[i]
            counts[c] += 1
            groups.setdefault(c,[]).append(i)
    means = np.full_like(sums,np.nan)
    np.divide(sums,counts[...,None],out=means,where=counts[...,None]>0)
    return dict(means=means,counts=counts,groups=groups,roles=list(roles),formats=list(formats),questions=list(questions))


def complete_grid(means, min_questions=30):
    keep = np.isfinite(means).all(axis=(0,1,3,4))
    ids = np.where(keep)[0].tolist()
    return dict(status='estimable' if len(ids)>=min_questions else 'not_estimable',
                kept_questions=ids,dropped_questions=np.where(~keep)[0].tolist(),
                n_questions=means.shape[2],minimum=min_questions,grid=means[:,:,keep])


def variance_split(x, check_orthogonality=True):
    """The eight mutually orthogonal frozen effects on the balanced prompt grid."""
    x = np.asarray(x,np.float64)
    if x.ndim != 5 or not np.isfinite(x).all():
        raise AnalysisError('D1 needs a finite balanced [R,F,Q,W,D] prompt grid')
    grand = x.mean((0,1,2,3),keepdims=True)
    centred = x-grand
    role = x.mean((1,2,3),keepdims=True)-grand
    fmt = x.mean((0,2,3),keepdims=True)-grand
    q = x.mean((0,1,3),keepdims=True)-grand
    effects = [role,fmt,x.mean((2,3),keepdims=True)-grand-role-fmt,q,
               x.mean((1,3),keepdims=True)-grand-role-q,
               x.mean((0,3),keepdims=True)-grand-fmt-q,
               x.mean((1,2),keepdims=True)-grand-role]
    residual = centred.copy()
    ss = {}
    for name,e in zip(TERMS,effects):
        residual -= e
        ss[name] = float(np.square(e).sum() * np.prod([x.shape[a] for a in range(4) if e.shape[a]==1]))
    ss['residual'] = float(np.square(residual).sum())
    total = float(np.square(centred).sum())
    out = dict(total_ss=total,ss=ss,shares={k:(v/total if total>0 else None) for k,v in ss.items()})
    out['role_minus_format'] = (ss['role']-ss['format'])/total if total>0 else None
    if check_orthogonality:
        full = [np.broadcast_to(e,x.shape) for e in effects]+[residual]
        out['max_abs_cross_term'] = max(abs(float(np.sum(a*b))) for a,b in itertools.combinations(full,2))
        if not np.isclose(sum(ss.values()),total,rtol=1e-10,atol=1e-10):
            raise AnalysisError('D1 sums of squares fail the partition identity')
    return out


def crossed_draws(R,Q,n=N_BOOT,seed=SEED):
    rng = np.random.default_rng(seed)
    return [(rng.integers(R,size=R),rng.integers(Q,size=Q)) for _ in range(n)]


def d1(means, n_boot=N_BOOT, seed=SEED, min_questions=30, artifacts_dir=None):
    grid = complete_grid(means,min_questions)
    report = {k:v for k,v in grid.items() if k!='grid'}
    if grid['status']!='estimable':
        return dict(report,reason='Fewer than the frozen minimum complete questions; no imputation.',p=None)
    x = grid['grid']
    fit = variance_split(x)
    if fit['total_ss'] == 0:
        return dict(report,status='not_estimable',reason='Zero total variance.',p=None)
    values,bootstrap_execution=sibling('analysis_variance_fast').bootstrap_differences(
        x,crossed_draws(len(x),x.shape[2],n_boot,seed),
        reference=lambda sampled:variance_split(sampled,False),return_info=True)
    if artifacts_dir is not None:
        np.save(Path(artifacts_dir)/'D1_bootstrap.npy',np.asarray(values))
    ci = interval(fit['role_minus_format'],values)
    p = float(np.mean(np.asarray(values)<=0)) if np.isfinite(values).all() else None
    return dict(report,fit=fit,difference=ci,p=p,raw_rule_met=ci['lo'] is not None and ci['lo']>0,
                bootstrap_execution=bootstrap_execution)


def flatten_cells(cells):
    idx = np.argwhere(cells['counts']>0)
    r,f,q,w = idx.T
    return dict(y=r,f=f,q=q,w=w,X=cells['means'][r,f,q,w],indices=idx,
                groups=[cells['groups'][tuple(i)] for i in idx],
                R=cells['means'].shape[0],F=cells['means'].shape[1],Q=cells['means'].shape[2],W=cells['means'].shape[3])


def make_splits(p,n_formats,n=N_SPLITS,seed=SEED):
    """E3: half questions x three wordings train, disjoint questions x two wordings test."""
    rng = np.random.default_rng(seed)
    base = [(rng.choice(p['Q'],p['Q']//2,replace=False),rng.choice(p['W'],3,replace=False)) for _ in range(n)]
    result = {}
    for f in range(n_formats):
        folds=[]
        for qs,ws in base:
            tr=np.where((p['f']!=f)&np.isin(p['q'],qs)&np.isin(p['w'],ws))[0]
            te=np.where((p['f']==f)&~np.isin(p['q'],qs)&~np.isin(p['w'],ws))[0]
            for part in (tr,te):
                if len(np.unique(p['y'][part]))!=p['R']:
                    raise AnalysisError('A role has no train/test prompt on a frozen D2 fold')
            folds.append((tr,te))
        result[f]=folds
    return result


def _fit_predict_scaled(Xtr,ytr,Xte):
    """A fresh frozen E3 classifier; caller owns the convergence-warning context."""
    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000)
    clf.fit(Xtr,ytr)
    return clf.predict(Xte)


def fit_predict_scaled(Xtr,ytr,Xte):
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        return _fit_predict_scaled(Xtr,ytr,Xte)


def fit_predict(Xtr,ytr,Xte):
    from sklearn.preprocessing import StandardScaler
    # With an unpenalised intercept, omitting sparse centring preserves the E3 linear model.
    sc=StandardScaler(with_mean=not sparse.issparse(Xtr)).fit(Xtr)
    return fit_predict_scaled(sc.transform(Xtr),ytr,sc.transform(Xte))


def macro_accuracy(y,pred,R):
    if any(not (y==r).any() for r in range(R)):
        raise AnalysisError('Macro accuracy needs every role')
    return float(np.mean([np.mean(pred[y==r]==r) for r in range(R)]))


def decode(X,p,folds,y=None):
    """Fresh training-only feature fit and E3 classifier on each fold; prompt weight is equal."""
    y=p['y'] if y is None else y
    hit,cnt=np.zeros((p['R'],p['Q'])),np.zeros((p['R'],p['Q']))
    scores=[]
    for tr,te in folds:
        a,b=X(tr,te) if callable(X) else (X[tr],X[te])
        pred=fit_predict(a,y[tr],b)
        scores.append(macro_accuracy(y[te],pred,p['R']))
        np.add.at(hit,(y[te],p['q'][te]),pred==y[te])
        np.add.at(cnt,(y[te],p['q'][te]),1)
    cell=np.full_like(hit,np.nan)
    np.divide(hit,cnt,out=cell,where=cnt>0)
    return dict(accuracy=float(np.mean(scores)),cells=cell,counts=cnt,split_scores=scores)


def block_permute(y,q,rng):
    yp=y.copy()
    for question in np.unique(q):
        m=q==question
        yp[m]=rng.permutation(int(y.max())+1)[y[m]]
    return yp


def null_execution_settings(workers=1,blas_threads=None,cache_scalers=False,backend='threads',device='cpu'):
    if isinstance(workers,bool) or not isinstance(workers,int) or workers<1:
        raise AnalysisError('Null workers must be a positive integer')
    if blas_threads is not None and (isinstance(blas_threads,bool) or not isinstance(blas_threads,int) or blas_threads<1):
        raise AnalysisError('Null BLAS threads must be a positive integer')
    if backend not in ('threads','processes','batches'):
        raise AnalysisError('Null backend must be threads, processes or batches')
    if (workers>1 or backend=='processes') and blas_threads is None:
        raise AnalysisError('Parallel/process null workers require an explicit inner thread resource budget')
    if device not in ('cpu','cuda'):
        raise AnalysisError('Null device must be cpu or cuda')
    execution=dict(workers=workers,blas_threads=blas_threads,cache_scalers=bool(cache_scalers),backend=backend)
    if backend=='batches':
        if workers>512 or not cache_scalers or blas_threads!=1:
            raise AnalysisError('Batched fits require at most 512 fits, cached transforms and one inner thread')
        execution.update(precision='float64',history_mode='loop',gpu_memory_fraction=.2,
            batch_source_hashes={name:streamed_sha256(HERE/name) for name in
                ('analysis_batch_lbfgs.py','analysis_batch_device.py','analysis_batch_loss.py')})
    if device=='cuda':
        if not cache_scalers or blas_threads!=1:
            raise AnalysisError('CUDA null requires cached transforms and one inner thread')
        execution.update(device='cuda',precision='float64',gpu_memory_fraction=.2 if backend=='batches' else .08)
    return execution


def validate_null_inputs(X,p,folds):
    """Check the frozen holdouts before preparing transforms or starting any workers."""
    if not isinstance(X,np.ndarray) or X.ndim!=2 or X.shape[0]!=len(p['y']):
        raise AnalysisError('Null features must be a dense array with one row per prompt')
    for name in ('y','q','w','f'):
        if np.shape(p[name])!=(len(X),):
            raise AnalysisError(f'Null {name} metadata does not match feature rows')
    for start in range(0,len(X),256):
        if not np.isfinite(X[start:start+256]).all():
            raise AnalysisError('Null features contain non-finite prompt values')
    if not folds:
        raise AnalysisError('Null needs at least one fixed held-out fold')
    for tr,te in folds:
        for idx in (tr,te):
            idx=np.asarray(idx)
            if idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer) or not len(idx):
                raise AnalysisError('A held-out fold has invalid row indices')
            if idx.min()<0 or idx.max()>=len(X) or len(np.unique(idx))!=len(idx):
                raise AnalysisError('A held-out fold has repeated or out-of-range rows')
            if set(np.unique(p['y'][idx]))!=set(range(p['R'])):
                raise AnalysisError('A role is absent from a held-out fold')
        if np.intersect1d(tr,te).size:
            raise AnalysisError('Train/test row overlap in a held-out fold')
        for name in ('q','w'):
            if np.intersect1d(p[name][tr],p[name][te]).size:
                raise AnalysisError(f'Train/test {name} holdouts must be disjoint')
        formats=np.unique(p['f'][te])
        if len(formats)!=1 or formats[0] in p['f'][tr] or set(np.unique(p['f'][tr]))!=set(range(p['F']))-set(formats):
            raise AnalysisError('Format held-out leakage or missing training format')


def feature_sha256(X):
    """The same C-order feature hash as before, without a full-array byte copy."""
    h=hashlib.sha256()
    for start in range(0,len(X),256):
        h.update(np.ascontiguousarray(X[start:start+256]).tobytes())
    return h.hexdigest()


def prepare_null_folds(X,p,folds,folder):
    """Fit training-X-only scalers once; map exact transforms read-only for all draws/workers.

    These are disposable per-call files, not resumable caches. On interruption or
    a fresh call they are rebuilt from the checked source features and fixed folds.
    No labels are supplied to StandardScaler.fit and no test row enters that fit.
    """
    from sklearn.preprocessing import StandardScaler
    validate_null_inputs(X,p,folds)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    prepared=[]
    for i,(tr,te) in enumerate(folds):
        scaler=StandardScaler().fit(X[tr])
        paths=[]
        for name,idx in (('train',tr),('test',te)):
            path=folder/f'fold{i}_{name}.npy'
            np.save(path,scaler.transform(X[idx]))
            paths.append(path)
        prepared.append(dict(Xtr=np.load(paths[0],mmap_mode='r'),Xte=np.load(paths[1],mmap_mode='r')))
    return prepared


def _null_accuracy(X,p,folds,y,prepared=None,device='cpu'):
    """Exactly decode()'s mean macro accuracy, without unused per-cell bookkeeping."""
    from sklearn.preprocessing import StandardScaler
    scores=[]
    for i,(tr,te) in enumerate(folds):
        if prepared is None:
            scaler=StandardScaler().fit(X[tr])
            a,b=scaler.transform(X[tr]),scaler.transform(X[te])
        else:
            a,b=prepared[i]['Xtr'],prepared[i]['Xte']
        pred=(sibling('analysis_gpu').fit_predict_scaled_array_api(a,y[tr],b) if device=='cuda'
              else _fit_predict_scaled(a,y[tr],b))
        scores.append(macro_accuracy(y[te],pred,p['R']))
    return float(np.mean(scores))


def verified_cuda_observed(p,folds,prepared,cuda_prepared):
    """Verify every actual observed-fold prediction against a fresh CPU fit."""
    scores=[]
    for j,(tr,te) in enumerate(folds):
        cpu=_fit_predict_scaled(prepared[j]['Xtr'],p['y'][tr],prepared[j]['Xte'])
        gpu=sibling('analysis_gpu').fit_predict_scaled_array_api(
            cuda_prepared[j]['Xtr'],p['y'][tr],cuda_prepared[j]['Xte'])
        if not np.array_equal(cpu,gpu):
            raise AnalysisError(f'CUDA observed predictions differ from CPU on fixed fold {j}')
        scores.append(macro_accuracy(p['y'][te],cpu,p['R']))
    return float(np.mean(scores))


_PROCESS_NULL_STATE=None
_PROCESS_NULL_LIMITER=None


def _initialise_process_null(descriptor,source_sha256):
    """One loky child: checked source/descriptor, read-only maps, native limits, warnings as errors."""
    global _PROCESS_NULL_STATE,_PROCESS_NULL_LIMITER
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    if streamed_sha256(Path(__file__))!=source_sha256:
        raise RuntimeError('Process analysis source checksum changed before worker initialisation')
    info=checked_json(descriptor,dependencies=())
    if info.get('schema')!=1 or info.get('analysis_sha256')!=source_sha256:
        raise RuntimeError('Process null descriptor has an invalid source fingerprint')
    warnings.filterwarnings('error',category=ConvergenceWarning)
    _PROCESS_NULL_LIMITER=threadpool_limits(limits=info['blas_threads'])  # BLAS AND OpenMP, for child lifetime.
    meta={name:np.asarray(values,dtype=int) for name,values in info['metadata'].items()}
    for values in meta.values():
        values.flags.writeable=False
    p=dict(meta,**info['dimensions'])
    folds=[(np.asarray(tr,int),np.asarray(te,int)) for tr,te in info['folds']]
    X=None if info['features'] is None else np.load(info['features'],mmap_mode='r')
    if X is not None and (list(X.shape)!=info['feature_shape'] or str(X.dtype)!=info['feature_dtype'] or X.flags.writeable):
        raise RuntimeError('Process feature memmap shape/dtype/readonly check failed')
    prepared=None
    if info['prepared'] is not None:
        prepared=[]
        for paths,(tr,te) in zip(info['prepared'],folds):
            pair={name:np.load(file,mmap_mode='r') for name,file in paths.items()}
            for name,idx in (('Xtr',tr),('Xte',te)):
                a=pair[name]
                if a.shape!=(len(idx),info['feature_shape'][1]) or str(a.dtype)!=info['feature_dtype'] or a.flags.writeable:
                    raise RuntimeError('Process transformed memmap shape/dtype/readonly check failed')
            prepared.append(pair)
        if len(prepared)!=len(folds):
            raise RuntimeError('Process transformed memmaps do not cover every fixed fold')
    device=info.get('device','cpu')
    if device=='cuda':
        helper=sibling('analysis_gpu')
        if streamed_sha256(Path(helper.__file__))!=info.get('gpu_helper_sha256'):
            raise RuntimeError('GPU helper source checksum changed before worker initialisation')
        prepared=helper.prepare_cuda_folds(prepared)
    _PROCESS_NULL_STATE=dict(X=X,p=p,folds=folds,prepared=prepared,seed=info['seed'],device=device)


def _process_null_draw(k):
    state=_PROCESS_NULL_STATE
    if state is None:
        raise RuntimeError('Process null worker was not initialised')
    yp=block_permute(state['p']['y'],state['p']['q'],np.random.default_rng(state['seed']+1+k))
    return _null_accuracy(state['X'],state['p'],state['folds'],yp,state['prepared'],state['device'])


def _new_process_null_pool(X,p,folds,prepared,folder,seed,workers,blas_threads,device='cpu'):
    """Loky workers import this checked file BEFORE unpickling the module-referenced job function.

    The initializer is builtins.exec, so no function from the caller's possibly
    unimportable importlib module must be unpickled during child startup. Parent
    and children register the file under the same content-addressed module name.
    Only draw indices travel per task; features/transforms are shared read-only files.
    No direct import of either standalone or vendored cloudpickle is needed.
    """
    from joblib.externals.loky import ProcessPoolExecutor
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve();code_hash=streamed_sha256(source)
    module_name=f'_podb_null_worker_{code_hash}'
    if module_name not in sys.modules:
        spec=importlib.util.spec_from_file_location(module_name,source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[module_name]=module
        spec.loader.exec_module(module)
    module=sys.modules[module_name]
    features=None
    if prepared is None:
        features=folder/'features.npy'
        np.save(features,X)
    mapped=None if prepared is None else [
        {name:str(Path(pair[name].filename).resolve()) for name in ('Xtr','Xte')} for pair in prepared]
    info=dict(schema=1,analysis_sha256=code_hash,seed=seed,blas_threads=blas_threads,device=device,
              features=None if features is None else str(features),prepared=mapped,
              feature_shape=list(X.shape),feature_dtype=str(X.dtype),
              dimensions={name:int(p[name]) for name in ('R','F','Q','W')},
              metadata={name:p[name].tolist() for name in ('y','q','w','f')},
              folds=[(tr.tolist(),te.tolist()) for tr,te in folds])
    if device=='cuda':
        info['gpu_helper_sha256']=streamed_sha256(HERE/'analysis_gpu.py')
    descriptor=folder/'process_descriptor.json'
    save_checked_json(descriptor,info)
    bootstrap=(
        'import importlib.util as _podb_util, sys as _podb_sys\n'
        f'_podb_spec = _podb_util.spec_from_file_location({module_name!r}, {str(source)!r})\n'
        '_podb_module = _podb_util.module_from_spec(_podb_spec)\n'
        f'_podb_sys.modules[{module_name!r}] = _podb_module\n'
        '_podb_spec.loader.exec_module(_podb_module)\n'
        f'_podb_module._initialise_process_null({str(descriptor)!r}, {code_hash!r})\n')
    env={name:str(blas_threads) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                                            'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}
    pool=ProcessPoolExecutor(max_workers=workers,initializer=exec,initargs=(bootstrap,),env=env)
    return pool,module._process_null_draw


def _batched_null(p,folds,prepared,n,seed,values,commit,workers,device,expected_observed,checkpoint,token):
    """Fresh independent fits; same draw seeds and fold-order mean as the serial path."""
    B=sibling('analysis_batch_lbfgs')
    if device=='cuda':
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.2,0)
    observed_scores=[];proof=[]
    for j,(tr,te) in enumerate(folds):
        pair=prepared[j]
        cpu=_fit_predict_scaled(pair['Xtr'],p['y'][tr],pair['Xte'])
        pred,info=B.fit_predict_batch(pair['Xtr'],p['y'][tr][None,:],pair['Xte'],device=device)
        if not np.array_equal(cpu,pred[0]):raise AnalysisError(f'Batched observed predictions differ on fold {j}')
        observed_scores.append(macro_accuracy(p['y'][te],cpu,p['R']))
        proof.append(dict(fold=j,cpu_sha256=hashlib.sha256(cpu.tobytes()).hexdigest(),
                          batched_sha256=hashlib.sha256(pred[0].tobytes()).hexdigest(),matched=True))
    observed=float(np.mean(observed_scores))
    if expected_observed is not None and observed!=expected_observed:
        raise AnalysisError('Batched observed accuracy differs from the primary decoder')
    if checkpoint is not None:
        vp=Path(checkpoint).with_name(Path(checkpoint).stem+'.batch_validation.json')
        if artifact_exists(vp):
            previous=checked_json(vp,dependencies=())
            if previous.get('fingerprint')!=token or previous.get('observed_folds')!=proof:
                raise AnalysisError('Batched observed verification changed on resume')
        else:save_checked_json(vp,dict(fingerprint=token,observed_folds=proof))
    print(f'BATCH_NULL: observed predictions verified on {len(folds)} folds',flush=True)
    first=len(values)
    for start in range(first,n,workers):
        indices=list(range(start,min(start+workers,n)))
        labels=np.stack([block_permute(p['y'],p['q'],np.random.default_rng(seed+1+k)) for k in indices])
        scores=[[] for _ in indices]
        for j,(tr,te) in enumerate(folds):
            pair=prepared[j]
            pred,info=B.fit_predict_batch(pair['Xtr'],labels[:,tr],pair['Xte'],device=device)
            # Check the first and last fresh null fit of every fold at entry or
            # resume. Failure stops the whole batch; it never drops a draw.
            if start==first:
                for k in sorted({0,len(indices)-1}):
                    cpu=_fit_predict_scaled(pair['Xtr'],labels[k,tr],pair['Xte'])
                    if not np.array_equal(cpu,pred[k]):
                        raise AnalysisError(f'Batched null predictions differ at draw {indices[k]}, fold {j}')
            for k in range(len(indices)):
                scores[k].append(macro_accuracy(labels[k,te],pred[k],p['R']))
        for per_draw in scores:commit(float(np.mean(per_draw)))
        print(f'BATCH_NULL: saved {len(values)}/{n} complete draws, {len(folds)} fits each',flush=True)
    return dict(observed=observed,null=values,p=float((1+sum(v>=observed for v in values))/(1+n)))


def decode_null(X,p,folds,n=N_PERM,seed=SEED,checkpoint=None,workers=1,blas_threads=None,cache_scalers=False,
                expected_observed=None,backend='threads',device='cpu'):
    """Full-refit block null, optionally CPU parallel, with an ordered durable draw prefix.

    Draw k always uses seed+1+k, independent of worker scheduling. At most `workers`
    draws are in flight; only the next ordered draw is committed. Failed/interrupted
    jobs cancel queued work and leave the checked prefix for an identical resume.
    BLAS limits are process-wide and held constant while the worker pool is alive.
    """
    # Import native libraries before threadpoolctl discovers their loaded BLAS pools.
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    execution=null_execution_settings(workers,blas_threads,cache_scalers,backend,device)
    validate_null_inputs(X,p,folds)
    if not isinstance(n,int) or isinstance(n,bool) or n<1:
        raise AnalysisError('Null draw count must be a positive integer')
    X=X.view();X.flags.writeable=False  # Threads share the source; never copy it per worker.
    token=digest(dict(seed=seed,n=n,folds=[(a.tolist(),b.tolist()) for a,b in folds],
                      y=p['y'].tolist(),q=p['q'].tolist(),X_hash=feature_sha256(X),execution=execution))
    values=[]
    if checkpoint is not None and artifact_exists(checkpoint):
        saved=checked_json(checkpoint,dependencies=())
        if saved['fingerprint']!=token or saved.get('execution')!=execution:
            raise AnalysisError('D2 null checkpoint fingerprint or execution resources changed')
        values=saved['values']
        if not isinstance(values,list) or len(values)>n or not np.isfinite(values).all() or any(v<0 or v>1 for v in values):
            raise AnalysisError('D2 null checkpoint has an invalid ordered prefix')
    with ExitStack() as stack:
        if blas_threads is not None:
            stack.enter_context(threadpool_limits(limits=blas_threads,user_api='blas'))
        # One warning context surrounds all workers; nested per-thread warning filters would race.
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter('error',ConvergenceWarning)
        prepared=None
        if cache_scalers or backend=='processes':
            root=str(Path(checkpoint).parent) if checkpoint is not None else None
            if root is not None:
                Path(root).mkdir(parents=True,exist_ok=True)
            folder=stack.enter_context(tempfile.TemporaryDirectory(prefix='.podb-null-folds-',dir=root))
        if cache_scalers:
            prepared=prepare_null_folds(X,p,folds,folder)
        if backend=='batches':
            def commit_batch(value):
                values.append(value)
                if checkpoint is not None:
                    save_checked_json(checkpoint,dict(fingerprint=token,execution=execution,values=values))
            return _batched_null(p,folds,prepared,n,seed,values,commit_batch,workers,device,
                                 expected_observed,checkpoint,token)
        cuda_prepared=None
        if device=='cuda':
            helper=sibling('analysis_gpu')
            cuda_prepared=helper.prepare_cuda_folds(prepared)
            observed=verified_cuda_observed(p,folds,prepared,cuda_prepared)
            if backend=='processes':
                helper.release_cuda_folds(cuda_prepared)
                cuda_prepared=None
            else:
                stack.callback(helper.release_cuda_folds,cuda_prepared)
        else:
            observed=_null_accuracy(X,p,folds,p['y'],prepared) if cache_scalers else decode(X,p,folds)['accuracy']
        if expected_observed is not None and observed!=expected_observed:
            raise AnalysisError('Null observed accuracy differs from the primary decoder under these resource settings')
        def one(k):
            yp=block_permute(p['y'],p['q'],np.random.default_rng(seed+1+k))
            return _null_accuracy(X,p,folds,yp,cuda_prepared if device=='cuda' else prepared,device)
        def commit(value):
            values.append(value)
            if checkpoint is not None:
                save_checked_json(checkpoint,dict(fingerprint=token,execution=execution,values=values))
        if workers==1 and backend=='threads':
            for k in range(len(values),n):
                # Preserve the original default serial path as well as its statistic.
                value=one(k) if cache_scalers else decode(X,p,folds,
                          block_permute(p['y'],p['q'],np.random.default_rng(seed+1+k)))['accuracy']
                commit(value)
        else:
            if backend=='processes':
                pool,task=_new_process_null_pool(X,p,folds,prepared,folder,seed,workers,blas_threads,device)
            else:
                pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='podb-null')
                task=one
            pending={}
            try:
                next_submit=len(values)
                for _ in range(min(workers,n-next_submit)):
                    pending[next_submit]=pool.submit(task,next_submit);next_submit+=1
                for k in range(len(values),n):
                    value=pending.pop(k).result()
                    commit(value)
                    if next_submit<n:
                        pending[next_submit]=pool.submit(task,next_submit);next_submit+=1
            finally:
                for future in pending.values():
                    future.cancel()
                if backend=='processes':
                    # Cancel queued work, then drain at most one bounded draw per child.
                    # Avoid forced tree-kill portability failures in loky on some Python builds.
                    pool.shutdown(wait=True)
                else:
                    pool.shutdown(wait=True,cancel_futures=True)
    return dict(observed=observed,null=values,p=float((1+sum(v>=observed for v in values))/(1+n)))


def accuracy_draws(cells,draws):
    return np.array([cells[np.ix_(r,q)].mean() for r,q in draws])


def make_halves(Q,W=5,n=N_HALVES,seed=SEED):
    rng=np.random.default_rng(seed)
    halves=[]
    for _ in range(n):
        qa=np.isin(np.arange(Q),rng.choice(Q,Q//2,replace=False))
        wa=np.isin(np.arange(W),rng.choice(W,int(rng.choice([W//2,W-W//2])),replace=False))
        halves.append((qa,wa))
    return halves


def map_distances(means,halves,qweights=None,roles=None):
    """Format x half x side x role x role distances; centre over DISTINCT drawn roles."""
    R,F,Q,W,D=means.shape
    qweights=np.ones(Q) if qweights is None else qweights
    roles=np.arange(R) if roles is None else np.asarray(roles)
    distinct=np.unique(roles)
    distances=np.full((F,len(halves),2,len(distinct),len(distinct)),np.nan)
    for f in range(F):
        for h,(qa,wa) in enumerate(halves):
            for side,(qm,wm) in enumerate(((qa,wa),(~qa,~wa))):
                x=means[distinct,f][:,:,wm]
                good=np.isfinite(x).all(-1)
                weight=qweights*qm
                den=np.einsum('rqw,q->r',good,weight)
                if (den==0).any():
                    continue
                v=np.einsum('rqwd,q->rd',np.nan_to_num(x),weight)/den[:,None]
                v-=v.mean(0)
                norm=np.linalg.norm(v,axis=1)
                if (norm==0).any():
                    continue
                distances[f,h,side]=1-(v@v.T)/np.outer(norm,norm)
    return distances


def correlations(a,b):
    a,b=rankdata(a,axis=-1),rankdata(b,axis=-1)
    a,b=a-a.mean(-1,keepdims=True),b-b.mean(-1,keepdims=True)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.sum(a*b,-1)/np.sqrt(np.sum(a*a,-1)*np.sum(b*b,-1))


def map_statistic(distances,pair_indices=None):
    F,H,_,R,_=distances.shape
    i,j=np.triu_indices(R,1) if pair_indices is None else pair_indices
    vec=distances[...,i,j]
    pairs={}
    for a,b in itertools.combinations(range(F),2):
        v=(correlations(vec[a,:,0],vec[b,:,1])+correlations(vec[b,:,0],vec[a,:,1]))/2
        pairs[f'{a}/{b}']=float(v.mean())
    reliability={str(f):float(correlations(vec[f,:,0],vec[f,:,1]).mean()) for f in range(F)}
    return dict(mean=float(np.mean(list(pairs.values()))),pairs=pairs,reliability=reliability,
                undefined_halves=int((~np.isfinite(vec).all(axis=(0,2,3))).sum()))


def joint_map_null(d,n=N_PERM,seed=SEED+1):
    """Interview fixed; one independent relabelling per other format shared by ALL its pairs/halves."""
    F,H,_,R,_=d.shape
    rng=np.random.default_rng(seed)
    i,j=np.triu_indices(R,1)
    out=[]
    for start in range(0,n,100):
        B=min(100,n-start)
        perm=np.array([[np.arange(R)]+[rng.permutation(R) for _ in range(F-1)] for _ in range(B)])
        # Each format's same permutation serves both halves, directions and every pair.
        ranks=[rankdata(d[f][...,perm[:,f,i],perm[:,f,j]],axis=-1).transpose(2,0,1,3) for f in range(F)]  # B,H,2,P
        vals=[]
        for a,b in itertools.combinations(range(F),2):
            vals.append((correlations(ranks[a][:,:,0],ranks[b][:,:,1])+
                         correlations(ranks[b][:,:,0],ranks[a][:,:,1]))/2)
        out.extend(np.asarray(vals).mean((0,2)).tolist())
    return np.asarray(out)


def d3_bootstrap(means,halves,draws):
    """Use C's exact Gram accelerator; re-centre over distinct sampled roles before distances."""
    M=map_module()
    R,F,Q,W,D=means.shape
    sides=[]
    for f in range(F):
        good=np.isfinite(means[:,f]).all(-1).astype(float)
        S=np.nan_to_num(means[:,f])
        sides.append([(M.Side(S,good,qa,wa),M.Side(S,good,~qa,~wa)) for qa,wa in halves])
    vals=[]; pairvals=[]; relvals=[]; undefined=[]
    # Gram work is vectorised in chunks; feature width does not enter bootstrap cost.
    for start in range(0,len(draws),50):
        batch=draws[start:start+50]
        Wb=np.array([np.bincount(qq,minlength=Q) for _,qq in batch],float)
        grams=np.array([[[side.grams(Wb) for side in h] for h in fmt] for fmt in sides])
        for b,(rr,_) in enumerate(batch):
            distinct,pos=np.unique(rr,return_inverse=True)
            g=grams[:,:,:,b][...,distinct,:][...,distinct]
            g=g-g.mean(-1,keepdims=True)-g.mean(-2,keepdims=True)+g.mean((-1,-2),keepdims=True)
            with np.errstate(invalid='ignore',divide='ignore'):
                diag=np.sqrt(np.maximum(np.diagonal(g,axis1=-2,axis2=-1),0))
                d=1-g/(diag[...,None]*diag[...,None,:])
            i,j=np.triu_indices(len(rr),1); keep=rr[i]!=rr[j]
            st=map_statistic(d,(pos[i[keep]],pos[j[keep]]))
            vals.append(st['mean']); pairvals.append(st['pairs']);relvals.append(st['reliability']);undefined.append(st['undefined_halves'])
    return dict(values=np.asarray(vals),pairs=pairvals,reliability=relvals,undefined_halves=undefined)


def d3(means,n_boot=N_BOOT,n_perm=N_PERM,n_halves=N_HALVES,seed=SEED,artifacts_dir=None):
    halves=make_halves(means.shape[2],means.shape[3],n_halves,seed)
    dist=map_distances(means,halves)
    st=map_statistic(dist)
    draws=crossed_draws(means.shape[0],means.shape[2],n_boot,seed)
    boot=d3_bootstrap(means,halves,draws)
    ci=interval(st['mean'],boot['values'])
    if not np.isfinite(st['mean']):
        return dict(status='not_estimable',agreement=ci,p=None,undefined_halves=st['undefined_halves'])
    null=joint_map_null(dist,n_perm,seed+1)
    if artifacts_dir is not None:
        np.savez(Path(artifacts_dir)/'D3_resampling.npz',null=null,bootstrap=boot['values'],
                 question_halves=np.array([h[0] for h in halves]),wording_halves=np.array([h[1] for h in halves]),
                 role_draws=np.array([x[0] for x in draws]),question_draws=np.array([x[1] for x in draws]))
    p=float((1+np.sum(null>=st['mean']))/(n_perm+1)) if np.isfinite(null).all() else None
    rel={f:interval(v,[b[f] for b in boot['reliability']]) for f,v in st['reliability'].items()}
    pairs={}
    for key,v in st['pairs'].items():
        a,b=key.split('/')
        out=dict(agreement=interval(v,[p[key] for p in boot['pairs']]))
        if all(rel[f]['lo'] is not None and rel[f]['lo']>=.5 for f in (a,b)):
            den=np.sqrt(np.array([r[a]*r[b] for r in boot['reliability']]))
            out['reliability_normalised']=interval(v/np.sqrt(st['reliability'][a]*st['reliability'][b]),
                                                  np.array([p[key] for p in boot['pairs']])/den)
        else:
            out['reliability_normalised']=dict(available=False,reason='Both reliability lower bounds must reach 0.5.')
        pairs[key]=out
    return dict(status='estimated',agreement=ci,p=p,pairs=pairs,reliability=rel,
                undefined_bootstrap_halves=boot['undefined_halves'],
                raw_rule_met=ci['lo'] is not None and ci['lo']>=.5 and p is not None and p<.01)


def holm(pvalues):
    ordered=sorted(pvalues,key=lambda k:1 if pvalues[k] is None else pvalues[k])
    result={}; previous=0.
    for j,k in enumerate(ordered):
        if pvalues[k] is None:
            result[k]=None
        else:
            previous=max(previous,min(1.,(len(ordered)-j)*pvalues[k]))
            result[k]=previous
    return result


def synthetic_checks():
    """G6 deterministic engineering calibration, BEFORE reading any real metric inputs."""
    rng=np.random.default_rng(719)
    shape=(6,4,8,5,9)
    tests={}
    for term,base in [('role',rng.normal(size=(6,1,1,1,9))),('format',rng.normal(size=(1,4,1,1,9))),
                      ('role_format',rng.normal(size=(6,4,1,1,9)))]:
        if term=='role_format':
            base=base-base.mean(0,keepdims=True)-base.mean(1,keepdims=True)+base.mean((0,1),keepdims=True)
        x=np.broadcast_to(base,shape).copy()
        tests[term]=variance_split(x)['shares'][term]
    role=np.broadcast_to(rng.normal(size=(6,1,1,1,9)),shape).copy()
    rows=[]; X=[]
    for r,f,q,w in np.ndindex(shape[:4]):
        for rep in range(3):
            good=not (r==0 and f==0 and q==0 and w==0) and not (rep<2 and w==1)
            rows.append(dict(role=str(r),format=str(f),question=q,wording=w,replicate=rep,valid=good))
            X.append(role[r,f,q,w] if good else np.full(9,np.nan))
    c=prompt_cells(rows,np.array(X),list(map(str,range(6))),list(map(str,range(4))),list(range(8)))
    grid=complete_grid(c['means'],7)
    missing_ok=grid['kept_questions']==list(range(1,8)) and variance_split(grid['grid'])['shares']['role']>1-1e-12
    # Same missingness/unequal-count check for each planted component, not only role.
    missing_terms={}
    for term,base in [('format',rng.normal(size=(1,4,1,1,9))),('role_format',rng.normal(size=(6,4,1,1,9)))]:
        if term=='role_format':
            base=base-base.mean(0,keepdims=True)-base.mean(1,keepdims=True)+base.mean((0,1),keepdims=True)
        x=np.broadcast_to(base,shape).copy()
        X=np.array([x[int(r['role']),int(r['format']),r['question'],r['wording']] if r['valid'] else np.full(9,np.nan) for r in rows])
        cc=prompt_cells(rows,X,list(map(str,range(6))),list(map(str,range(4))),list(range(8)))
        missing_terms[term]=variance_split(complete_grid(cc['means'],7)['grid'])['shares'][term]
    # Pure format signal yields identical features for every role in each format.
    fmt=np.broadcast_to(rng.normal(size=(1,4,1,1,9)),shape).copy()
    fake=dict(means=fmt,counts=np.ones(shape[:4],int),groups={k:[i] for i,k in enumerate(np.ndindex(shape[:4]))})
    p=flatten_cells(fake)
    scores=[decode(p['X'],p,folds)['accuracy'] for folds in make_splits(p,4,n=5,seed=87).values()]
    # Repeated independent stable format maps, not a single lucky independent-map p-value.
    ps=[]
    for _ in range(80):
        vectors=rng.normal(size=(4,8,6))
        vectors-=vectors.mean(1,keepdims=True)
        vectors/=np.linalg.norm(vectors,axis=2,keepdims=True)
        d=(1-vectors@vectors.transpose(0,2,1))[:,None,None]
        d=np.repeat(d,2,axis=2)
        obs=map_statistic(d)['mean']
        null=joint_map_null(d,n=199,seed=int(rng.integers(2**31)))
        ps.append(float((1+(null>=obs).sum())/200))
    ks=kstest(ps,'uniform')
    cal=dict(n=80,n_perm=199,mean_p=float(np.mean(ps)),ks_statistic=float(ks.statistic),ks_p=float(ks.pvalue),
             pass_rule='KS p >= 0.01 and mean p between 0.35 and 0.65; deterministic engineering check')
    cal['pass']=cal['ks_p']>=.01 and .35<=cal['mean_p']<=.65
    out=dict(variance=tests,missing_unequal_replicates=dict(role=missing_ok,**missing_terms),
             format_only_decoding=dict(accuracy=scores,chance=1/6),joint_null_calibration=cal)
    out['pass']=all(v>1-1e-12 for v in tests.values()) and missing_ok and all(v>1-1e-12 for v in missing_terms.values()) and np.allclose(scores,1/6) and cal['pass']
    return out


# ---- matched readers and report-only controls

def aggregation_matrix(groups,n_rows):
    ii=[];jj=[];vv=[]
    for i,group in enumerate(groups):
        ii.extend([i]*len(group));jj.extend(group);vv.extend([1/len(group)]*len(group))
    return sparse.csr_matrix((vv,(ii,jj)),shape=(len(groups),n_rows))


def text_featurizer(texts,groups,kind):
    """Vocabulary/IDF learn only from training replies; valid replicates then average per prompt."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    agg=aggregation_matrix(groups,len(texts))
    def feature(tr,te):
        train_rows=np.concatenate([groups[i] for i in tr])
        vectorizer=TfidfVectorizer(**TEXT[kind])
        try:
            vectorizer.fit([texts[i] for i in train_rows])
        except ValueError as e:
            raise AnalysisError(f'{kind} reader has no training vocabulary: {e}') from e
        # Transform only selected rows, using the SAME training vocabulary and IDF.
        def side(indices):
            used=np.unique(np.concatenate([groups[i] for i in indices]))
            feats=vectorizer.transform([texts[i] for i in used])
            return agg[indices][:,used]@feats
        return side(tr),side(te)
    return feature


def all_valid_fit(cells,raw_features=None):
    """Weighted full-model fit to all valid replicate rows, with no invented allocation of overlapping SS.

    Integer prompt counts are frequency weights. Fitting their means has the same
    coefficients as fitting all rows. RSS below covers between-prompt lack of fit;
    within-prompt replicate SS must be added by the caller for raw-row residual SS.
    """
    p=flatten_cells(cells)
    factors=[np.eye(p[n])[p[k]] for n,k in [('R','y'),('F','f'),('Q','q'),('W','w')]]
    r,f,q,w=factors
    matrix=np.column_stack([np.ones(len(r)),r,f,q,
                            (r[:,:,None]*f[:,None,:]).reshape(len(r),-1),
                            (r[:,:,None]*q[:,None,:]).reshape(len(r),-1),
                            (f[:,:,None]*q[:,None,:]).reshape(len(r),-1),
                            (r[:,:,None]*w[:,None,:]).reshape(len(r),-1)])
    counts=cells['counts'][tuple(p['indices'].T)]
    root=np.sqrt(counts)
    design=matrix*root[:,None]
    coef,_,rank,_=np.linalg.lstsq(design,p['X']*root[:,None],rcond=None)
    residual=p['X']-matrix@coef
    between=float(np.sum(residual**2*counts[:,None]))
    within=0.0 if raw_features is not None else None
    if raw_features is not None:
        for i,group in enumerate(p['groups']):
            within+=float(np.square(np.asarray(raw_features[group],float)-p['X'][i]).sum())
    return dict(n_rows=int(counts.sum()),n_prompts=len(counts),rank=int(rank),columns=matrix.shape[1],
                between_prompt_residual_ss=between,within_prompt_ss=within,
                residual_ss=between+(within or 0.0),
                convention='Frequency-weighted full model; raw-row RSS includes within-prompt SS when supplied; no nonorthogonal term shares.')


def context_cells(X,keys,roles,formats,questions,n_wordings=5):
    if len(keys)!=len(X):
        raise AnalysisError('C3 keys and state rows differ')
    seen=set();rows=[]
    for key in keys:
        k=tuple(key)
        if k in seen:
            raise AnalysisError('duplicate C3 prompt key')
        seen.add(k)
        r,f,w,q=k
        rows.append(dict(role=r,format=f,wording=w,question=q,valid=True))
    return prompt_cells(rows,X,roles,formats,questions,n_wordings)


def finite_json(value):
    """Save unavailable values as null, retaining explicit counts/reasons beside summaries."""
    if isinstance(value,dict):
        return {str(k):finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    return value


def describe_maps(means,halves):
    """Point estimates only for a report-only sensitivity, with explicit missing support."""
    d=map_distances(means,halves)
    report=map_statistic(d)
    report['status']='estimated' if np.isfinite(report['mean']) else 'not_estimable'
    # Full-question role vector cosine per role across centred format-specific vectors.
    good=np.isfinite(means).all(-1)
    count=good.sum((2,3))
    v=np.nansum(means,axis=(2,3))/np.maximum(count[...,None],1)
    v-=v.mean(0,keepdims=True)
    cos={}
    for a,b in itertools.combinations(range(means.shape[1]),2):
        den=np.linalg.norm(v[:,a],axis=-1)*np.linalg.norm(v[:,b],axis=-1)
        with np.errstate(divide='ignore',invalid='ignore'):
            c=np.sum(v[:,a]*v[:,b],axis=-1)/den
        c[(count[:,a]==0)|(count[:,b]==0)]=np.nan
        cos[f'{a}/{b}']=c.tolist()
    report['per_role_format_cosine']=cos
    report['complete_question_count']=len(complete_grid(means)['kept_questions'])
    report['interpretation']='Report only; missing cells stay missing. No interval or pass rule is inferred.'
    return finite_json(report)


def decode_report(X,p,folds,draws):
    out=decode(X,p,folds)
    values=accuracy_draws(out['cells'],draws)
    return dict(accuracy=interval(out['accuracy'],values),split_scores=out['split_scores'],
                untested_cells=int(np.isnan(out['cells']).sum())),out['cells'],values


def d2(cells,rows,state0,emb,n_boot=N_BOOT,n_perm=N_PERM,n_splits=N_SPLITS,seed=SEED,out_dir=None,
       null_workers=1,null_blas_threads=None,cache_null_scalers=False,null_backend='threads',null_device='cpu'):
    execution=null_execution_settings(null_workers,null_blas_threads,cache_null_scalers,null_backend,null_device)
    p=flatten_cells(cells)
    splits=make_splits(p,p['F'],n_splits,seed)
    draws=crossed_draws(p['R'],p['Q'],n_boot,seed)
    agg=aggregation_matrix(p['groups'],len(rows))
    readers={'word':text_featurizer([r['reply'] for r in rows],p['groups'],'word'),
             'char':text_featurizer([r['reply'] for r in rows],p['groups'],'char'),
             'sentence':agg@emb,'hidden_state0':agg@np.nan_to_num(state0)}
    length=np.array([[len(r['kept_ids'])] for r in rows],float)
    reasons=sorted({r['stop_reason'] for r in rows})
    stops=np.array([[r['stop_reason']==s for s in reasons] for r in rows],float)
    readers.update(length=agg@length,stop_reason=agg@stops,
                   question_id=np.eye(p['Q'])[p['q']],format_id=np.eye(p['F'])[p['f']])
    if out_dir:
        inputs=dict(null_execution=execution,prompt_indices=p['indices'].tolist(),valid_replicate_rows=p['groups'],
                    folds={str(f):[(a.tolist(),b.tolist()) for a,b in folds] for f,folds in splits.items()},
                    role_draws=[r.tolist() for r,q in draws],question_draws=[q.tolist() for r,q in draws])
        ip=Path(out_dir)/'D2_inputs.json'
        if artifact_exists(ip):
            if checked_json(ip,dependencies=())!=finite_json(inputs):
                raise AnalysisError('D2 input splits/draws changed; use a new output folder')
        else:
            save_checked_json(ip,inputs)
        # Validate all existing D2 artifacts before reusing a completed format or partial null.
        for file in Path(out_dir).glob('D2_*'):
            if not file.name.endswith('.integrity.json') and file.is_file():
                verify_artifact(file)
    formats={}
    for f,folds in splits.items():
        cached=Path(out_dir)/f'D2_format{f}.json' if out_dir else None
        if cached is not None and artifact_exists(cached):
            formats[cells['formats'][f]]=checked_json(cached,
                    dependencies=['D2_inputs.json',f'D2_null_format{f}.json',f'D2_reader_cells_format{f}.npz'])
            continue
        primary,primary_cells,primary_draws=decode_report(p['X'],p,folds,draws)
        nr=decode_null(p['X'],p,folds,n_perm,seed+100*f,
                       Path(out_dir)/f'D2_null_format{f}.json' if out_dir else None,
                       workers=null_workers,blas_threads=null_blas_threads,cache_scalers=cache_null_scalers,
                       expected_observed=primary['accuracy']['point'],backend=null_backend,device=null_device)
        baselines={};text_draws=[];text_points=[];baseline_cells={}
        for name,X in readers.items():
            report,reader_cells,vals=decode_report(X,p,folds,draws)
            baseline_cells[name]=reader_cells
            baselines[name]=report
            if name in ('word','char','sentence','hidden_state0'):
                text_draws.append(vals);text_points.append(report['accuracy']['point'])
        gains=primary_draws-np.max(text_draws,axis=0)
        gain=interval(primary['accuracy']['point']-max(text_points),gains)
        ci=primary['accuracy']
        raw_met=ci['lo'] is not None and ci['lo']>=2/p['R'] and nr['p']<.01
        formats[cells['formats'][f]]=dict(**primary,p=nr['p'],n_perm=n_perm,baselines=baselines,
                gain_over_best_text=gain,raw_rule_met=raw_met,null_execution=execution,
                reading='role information that the reply text already carries' if gain['lo'] is None or gain['lo']<=0
                        else 'Activation decoding exceeds these text readers; this does not establish information beyond text.')
        if out_dir:
            reader_path=Path(out_dir)/f'D2_reader_cells_format{f}.npz'
            np.savez(reader_path,activation=primary_cells,**baseline_cells)
            seal_artifact(reader_path)
            save_checked_json(Path(out_dir)/f'D2_format{f}.json',finite_json(formats[cells['formats'][f]]),
                              dependencies=['D2_inputs.json',f'D2_null_format{f}.json',reader_path.name])
    return dict(formats=formats,p=max(v['p'] for v in formats.values()),
                raw_rule_met=all(v['raw_rule_met'] for v in formats.values()),chance=1/p['R'],
                interval_scope='Crossed role/question bootstrap on held-out cells pooled over fixed splits.',
                null_splits=n_splits,split_seed=seed,null_execution=execution)


# ---- verified input and resumable pod execution

def validate_gate_evidence(report,n_rows,n_valid,n_quote_other=None):
    try:
        gates=report['gates']
        if report['passed'] is not True or any(gates[f'G{i}']['pass'] is not True for i in range(1,6)):
            raise AnalysisError('G1--G5 evidence did not pass')
        if gates['G1']['n_rows']!=n_rows or gates['G1']['n_state_rows']!=n_rows or gates['G1']['n_quote_rows']!=n_rows:
            raise AnalysisError('G1 evidence row count differs')
        if gates['G3']['n_rows']<40 or gates['G3']['identical_same_batches']!=gates['G3']['n_rows']:
            raise AnalysisError('G3 requires at least 40 identical same-batch rows')
        if gates['G4']['n_invalid']!=n_rows-n_valid:
            raise AnalysisError('G4 invalid count differs')
        for layer in (9,17,23):
            c=gates['G4']['cells'][f'L{layer}']
            if c['n']!=n_valid or c['min_cos']<.95 or c['median_cos']<.99 or c['median_rel_norm']>.02:
                raise AnalysisError('G4 reconstruction does not cover every valid row or fails tolerances')
        if gates['G4']['speech_quote_rows_unequal']!=0:
            raise AnalysisError('G4 speech quote states differ from reply states')
        if n_quote_other:
            for layer in (8,16,24):
                c=gates['G4']['quotes'][f'L{layer}']
                if c['n']!=n_quote_other or c['max_rel_norm']>.05 or c['median_rel_norm']>.02:
                    raise AnalysisError('G4 quote reconstruction count or tolerance failed')
        if gates['G5']['mismatch']:
            raise AnalysisError('G5 snapshot mismatch')
    except (KeyError,TypeError) as e:
        raise AnalysisError(f'Incomplete saved gate evidence: {e}') from e


def validate_generation_code(generation,current):
    """Keep historical generation bytes distinct from the current analysis helpers.

    rows.file_sha256 changed only its reading strategy after generation. The run
    must retain its approved original helper hash. Current helper bytes are still
    recorded and participate in the analysis fingerprint; they never replace the
    historical manifest entry. Other recorded generation scripts must still match.
    """
    if not {'gen_extract.py','rows.py'}<=set(generation) or set(generation)-set(current):
        raise AnalysisError('Generation code hash list is incomplete or contains unknown scripts')
    if not {'gen_extract.py','gates.py','rows.py'}<=set(current):
        raise AnalysisError('Current analysis code hash list is incomplete')
    if generation['rows.py']!=GENERATION_ROWS_SHA256:
        raise AnalysisError('Unknown generation rows.py hash; historical provenance must not be rewritten')
    for name,expected in generation.items():
        if name!='rows.py' and current[name]!=expected:
            raise AnalysisError(f'Generation code changed: {name}')
    return dict(generation_code=dict(generation),current_analysis_code=dict(current))


def validate_fingerprint_parts(recorded,manifest):
    """Check the exact fingerprint and the generation finalizer's richer manifest.

    gen_extract.finalize adds CUDA and NumPy to manifest['versions']; the four
    original version fields must still equal the fingerprint's historical
    values. Extra finalization metadata must not be mistaken for a new run.
    """
    parts={k:v for k,v in recorded.items() if k!='template_text'}
    if sibling('gen_extract').R.fingerprint(parts)!=manifest.get('fingerprint'):
        raise AnalysisError('Run fingerprint hash differs from its recorded parts')
    for name,value in parts.items():
        got=manifest.get(name)
        if name=='versions':
            same=isinstance(value,dict) and isinstance(got,dict) and all(got.get(k)==v for k,v in value.items())
        else:
            same=got==value
        if not same:
            raise AnalysisError(f'Run fingerprint parts differ from manifest: {name}')
    return parts


def load_verified_run(path,raw_read_note):
    """Checksums, exact frozen full grid, current cached tokenizer, schema, arrays, gates, and raw-read note."""
    G=sibling('gen_extract'); gates=sibling('gates'); R=G.R
    path=Path(path).resolve()
    if not Path(raw_read_note).is_file() or not Path(raw_read_note).read_text().strip():
        raise AnalysisError('A nonempty raw-read attestation note is required (10 random replies per format).')
    run=G.load_run(path)
    man=run['manifest']
    roles,questions,n_rep=G.grid_spec('full')
    expected=R.grid_keys(roles,G.FORMATS,range(5),questions,range(n_rep))
    if run['grid']!=expected or man['grid']!='full' or man.get('limit') is not None:
        raise AnalysisError('Only the frozen full 10 x 4 x 40 x 5 x 3 grid is confirmatory; canaries are refused')
    if man['model']!=G.MODEL or man['revision']!=G.REVISION or man['templates']!=G.templates_hash():
        raise AnalysisError('Model/revision/template differs from frozen design')
    if (man['roles'],man['formats'],man['questions'],man['replicates'])!=(roles,list(G.FORMATS),questions,n_rep):
        raise AnalysisError('Manifest factors differ from grid')
    validate_fingerprint_parts(json.loads((path/'fingerprint_parts.json').read_text()),man)
    for name,h in man['sources'].items():
        if G.sha256_file(G.REPO/name)!=h:
            raise AnalysisError(f'Frozen source changed: {name}')
    code_record=validate_generation_code(man['code'],
                    {name:G.sha256_file(file) for name,file in
                     [('gen_extract.py',HERE/'gen_extract.py'),('gates.py',HERE/'gates.py'),('rows.py',Path(R.__file__))]})
    # Offline only: require the exact tokenizer snapshot already in the pod cache.
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    if set(man['tokenizer_files'])!=set(G.TOKENIZER_FILES):
        raise AnalysisError('Tokenizer snapshot hash list is incomplete')
    for name,h in man['tokenizer_files'].items():
        file=cached_file(G.MODEL,name,revision=G.REVISION,local_files_only=True)
        if file is None or G.sha256_file(file)!=h:
            raise AnalysisError(f'Cached tokenizer hash differs: {name}')
    tok=AutoTokenizer.from_pretrained(G.MODEL,revision=G.REVISION,local_files_only=True)
    if man.get('extra_layers')!=[8,16,24] or man.get('n_hidden_states')!=33:
        raise AnalysisError('Hidden-state labels differ from frozen design')
    fresh=gates.g1(run,tok,33,3,4096)
    rate=gates.g2(run['rows'])
    if not fresh['pass'] or not rate['pass']:
        raise AnalysisError('Current G1/G2 verification failed')
    for row,extra in zip(run['rows'],run['extras']):
        if extra['names_role']!=G.names_role(row['role'],row['reply']):
            raise AnalysisError('Role-name mask differs from E3 NAME_TERMS')
    report=json.loads((path/'gates.json').read_text())
    validate_gate_evidence(report,len(run['rows']),sum(r['valid'] for r in run['rows']),
                           sum(x['quote_span'] is not None and r['format'] not in G.SPEECH_FORMATS
                               for r,x in zip(run['rows'],run['extras'])))
    run.update(path=path,roles=roles,formats=list(G.FORMATS),questions=questions,
               verified=dict(G1=fresh,G2=rate,saved_gates=report,code_provenance=code_record,raw_read_note=str(Path(raw_read_note).resolve()),
                             raw_read_sha256=G.sha256_file(Path(raw_read_note))))
    return run


def embedding_cache(run,out,device='cpu'):
    """Pinned sentence reader, cached per exact row text; local model files only."""
    texts=[r['reply'] if r['valid'] else '' for r in run['rows']]
    metadata=dict(model=list(EMBED_MODEL),text_hash=digest(texts),n_rows=len(texts),
                  normalize_embeddings=True,dtype='float32')
    arr=Path(out)/'sentence_embeddings.npy';mp=Path(out)/'sentence_embeddings.json'
    if arr.exists() or mp.exists():
        if not arr.exists() or not mp.exists():
            raise AnalysisError('Incomplete sentence embedding cache')
        saved=json.loads(mp.read_text())
        if any(saved.get(k)!=v for k,v in metadata.items()) or saved.get('sha256')!=sibling('gen_extract').sha256_file(arr):
            raise AnalysisError('Sentence embedding cache provenance changed')
        emb=np.load(arr,mmap_mode='r')
    else:
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(EMBED_MODEL[0],revision=EMBED_MODEL[1],device=device,local_files_only=True)
        emb=model.encode(texts,batch_size=64,show_progress_bar=True,convert_to_numpy=True,
                         normalize_embeddings=True).astype(np.float32)
        np.save(arr,emb)
        atomic_json(mp,dict(metadata,sha256=sibling('gen_extract').sha256_file(arr)))
    if emb.ndim!=2 or emb.shape[0]!=len(texts) or emb.dtype!=np.float32 or not np.isfinite(emb).all():
        raise AnalysisError('Sentence embeddings shape/finite check failed')
    return emb


def report_sensitivities(run,cells,args):
    G=sibling('gen_extract')
    rows=run['rows'];roles=run['roles'];fmts=run['formats'];qs=run['questions']
    halves=make_halves(len(qs),5,args.n_halves,args.seed)
    def cells_at(X,mask=None):
        return prompt_cells(rows,X,roles,fmts,qs,mask=mask)
    def describe(c):
        complete=complete_grid(c['means'])
        ss=variance_split(complete['grid']) if complete['status']=='estimable' else None
        return dict(prompt_count=int((c['counts']>0).sum()),valid_replicates=int(c['counts'].sum()),
                    D1_status=complete['status'],D1=ss,maps=describe_maps(c['means'],halves),
                    decoding=descriptive_decoding(c,args.n_splits,args.n_boot,args.seed))
    reports={'all_replies_state17':describe(cells)}
    reports['without_role_names_state17']=describe(cells_at(run['acts'][:,STATE],
                                         ~np.array([x['names_role'] for x in run['extras']])))
    reports['quoted_speech_state16']=describe(cells_at(run['quotes'][:,1]))
    reports['whole_reply_state16_on_quote_support']=describe(cells_at(run['acts'][:,EXTRA_STATE],
                                              np.array([x['quote_span'] is not None for x in run['extras']])))
    reports['whole_reply_state16']=describe(cells_at(run['acts'][:,EXTRA_STATE]))
    for layer in (9,23):
        reports[f'exploratory_state{layer}']=describe(cells_at(run['acts'][:,layer]))
    reports['replicate_held_out']={}
    for rep in range(3):
        reports['replicate_held_out'][str(rep)]=describe(cells_at(run['acts'][:,STATE],
                                                                np.array([r['replicate']!=rep for r in rows])))
    reports['wording_held_out']={}
    for w in range(5):
        # Keep the original wording axis: the complete 5-wording D1 grid is then unavailable.
        c=cells_at(run['acts'][:,STATE],np.array([r['wording']!=w for r in rows]))
        reports['wording_held_out'][str(w)]=describe(c)
    context=run['path']/'context'
    if not context.exists():
        reports['context']=dict(status='unavailable',reason='Context extraction files are absent; no C2/C3 substitution.')
    else:
        G.check_sums(context,G.CONTEXT_FILES)
        man=json.loads((context/'manifest.json').read_text())
        if man['model']!=G.MODEL or man['revision']!=G.REVISION or man['run_fingerprint']!=run['manifest']['fingerprint'] or man['layers']!=[8,16,24]:
            raise AnalysisError('Context model/run/layers differ from verified run')
        c2=np.load(context/'ctx_c2.npy',mmap_mode='r');c3=np.load(context/'ctx_c3.npy',mmap_mode='r')
        keys=json.loads((context/'c3_prompts.json').read_text())
        expected=set(G.prompt_of(k) for k in run['grid'])
        if len(keys)!=len(expected) or set(map(tuple,keys))!=expected:
            raise AnalysisError('C3 prompt keys differ from full grid')
        if c2.shape!=(len(rows),3,4096) or c3.shape!=(len(keys),3,4096):
            raise AnalysisError('Context arrays have wrong shapes')
        valid=np.array([r['valid'] for r in rows])
        if not (np.isfinite(c2[valid]).all() and np.isnan(c2[~valid]).all() and np.isfinite(c3).all()):
            raise AnalysisError('Context arrays have wrong finite/NaN mask')
        reports['context']=dict(C2_state16=describe(cells_at(c2[:,1])),
                               C3_state16=describe(context_cells(c3[:,1],keys,roles,fmts,qs)))
    reports['scope']='All controls are descriptive. Quote/C2/C3 use state16, not primary state17. Decoding sensitivity is not inferred from map changes.'
    return finite_json(reports)


def null_resource_provenance(args):
    from threadpoolctl import threadpool_info
    execution=null_execution_settings(getattr(args,'null_workers',1),getattr(args,'null_blas_threads',None),
                                      getattr(args,'cache_null_scalers',False),getattr(args,'null_backend','threads'),getattr(args,'null_device','cpu'))
    env_names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
    pools=[{key:pool.get(key) for key in ('user_api','internal_api','prefix','version','num_threads')}
           for pool in threadpool_info()]
    pools.sort(key=lambda pool:(str(pool['user_api']),str(pool['prefix'])))
    available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    profile=dict(execution=execution,max_in_flight_draws=execution['workers'],available_cpus=available,
                blas_limit_scope='process_wide_for_duration_of_null_only',initial_native_pools=pools,
                thread_environment={name:os.environ.get(name) for name in env_names},
                scaler_cache_scope='temporary_per_call_rebuilt_on_resume' if execution['cache_scalers'] else 'none')
    if execution['backend']=='processes':
        profile.update(process_implementation='joblib.loky',inner_limit_scope='BLAS_and_OpenMP_in_each_child')
    if execution.get('device')=='cuda':
        profile['gpu']=sibling('analysis_gpu').runtime_info()
        if execution['backend']=='batches':
            profile['gpu'].update(memory_fraction=.2,execution='device-resident independent L-BFGS',
                                  classifier='frozen ridge-softmax, batched independent L-BFGS',
                                  label_device='cuda fixed label vectors',array_api_dispatch=False)
    return profile


def provenance(run,args):
    G=sibling('gen_extract')
    files=[Path(__file__),HERE/'analysis_variance_fast.py',HERE/'analysis_gpu.py',HERE/'THRESHOLDS.md',HERE/'gen_extract.py',HERE/'gates.py',HERE/'context_extract.py',
           HERE.parent/'261029_fixed_text_read_through/maps.py',Path(G.R.__file__)]
    if getattr(args,'null_backend','threads')=='batches':
        files.extend(HERE/name for name in ('analysis_batch_lbfgs.py','analysis_batch_device.py','analysis_batch_loss.py'))
    def git(*words):
        r=subprocess.run(['git',*words],cwd=G.REPO,text=True,capture_output=True)
        return r.stdout.strip() if r.returncode==0 else 'unavailable'
    from importlib.metadata import version,PackageNotFoundError
    versions={}
    for package in ('numpy','scipy','scikit-learn','sentence-transformers','transformers','torch','joblib','threadpoolctl'):
        try: versions[package]=version(package)
        except PackageNotFoundError: versions[package]=None
    sources={str(p):G.sha256_file(p) for p in files}
    if (run['path']/'context/SHA256SUMS.txt').exists():
        sources[str(run['path']/'context/SHA256SUMS.txt')]=G.sha256_file(run['path']/'context/SHA256SUMS.txt')
    sources.update({str(run['path']/n):G.sha256_file(run['path']/n) for n in ('SHA256SUMS.txt','gates.json','raw_read.md') if (run['path']/n).exists()})
    code_record=validate_generation_code(run['manifest']['code'],
                    {name:sources[str(file)] for name,file in
                     [('gen_extract.py',HERE/'gen_extract.py'),('gates.py',HERE/'gates.py'),('rows.py',Path(G.R.__file__))]})
    return dict(run=str(run['path']),run_fingerprint=run['manifest']['fingerprint'],
                code_provenance=code_record,source_hashes=sources,null_resources=null_resource_provenance(args),settings={k:v for k,v in vars(args).items() if k!='out'},
                model=G.MODEL,revision=G.REVISION,tokenizer_files=run['manifest']['tokenizer_files'],
                sentence_reader=list(EMBED_MODEL),versions=dict(python=platform.python_version(),**versions),
                command=sys.argv,git_revision=git('rev-parse','HEAD'),git_status=git('status','--porcelain=v1'),
                git_diff_sha256=hashlib.sha256(git('diff','--binary','HEAD').encode()).hexdigest(),
                method_limits=METHOD_LIMITS,raw_read=run['verified']['raw_read_note'],
                raw_read_sha256=run['verified']['raw_read_sha256'],
                confirmatory=confirmatory_parameters(args.n_boot,args.n_perm,args.n_splits,args.n_halves))


def phase(out,name,fn):
    path=Path(out)/f'{name}.json'
    if artifact_exists(path):
        return checked_json(path)
    print(f'{name}: starting',flush=True)
    value=finite_json(fn())
    save_checked_json(path,value,phase_dependencies(out,name))
    print(f'{name}: saved {path}',flush=True)
    return value


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--synthetic-only',action='store_true')
    ap.add_argument('--run')
    ap.add_argument('--out')
    ap.add_argument('--execute-on-pod',action='store_true',help='Explicitly permit heavy verified-run analysis on this host')
    ap.add_argument('--raw-read-note',help='Existing owner/analyst note confirming 10 random replies per format read')
    ap.add_argument('--allow-nonconfirmatory',action='store_true')
    ap.add_argument('--n-boot',type=int,default=N_BOOT)
    ap.add_argument('--n-perm',type=int,default=N_PERM)
    ap.add_argument('--n-splits',type=int,default=N_SPLITS)
    ap.add_argument('--n-halves',type=int,default=N_HALVES)
    ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--holm-alpha',type=float,default=None,help='Explicit owner-selected family alpha; default: pending family verdict')
    ap.add_argument('--embedding-device',default='cpu')
    ap.add_argument('--null-device',choices=('cpu','cuda'),default='cpu',help='Optional identical float64 sklearn array-API execution; CUDA observed labels are CPU-verified')
    ap.add_argument('--null-backend',choices=('threads','processes','batches'),default='threads',help='Optional loky processes or true independent GPU batches')
    ap.add_argument('--null-workers',type=int,default=1,help='Concurrent null draws; batch width for batches backend (default: serial)')
    ap.add_argument('--null-blas-threads',type=int,default=None,help='BLAS threads per worker, also OpenMP in process children; required for parallel/process modes')
    ap.add_argument('--cache-null-scalers',action='store_true',help='Reuse exact training-X-only transforms via temporary read-only memmaps')
    args=ap.parse_args(argv)
    try:
        null_execution_settings(args.null_workers,args.null_blas_threads,args.cache_null_scalers,args.null_backend,args.null_device)
    except AnalysisError as e:
        ap.error(str(e))
    if args.holm_alpha is not None and not 0<args.holm_alpha<1:
        ap.error('--holm-alpha must be between 0 and 1')
    if min(args.n_boot,args.n_perm,args.n_splits,args.n_halves)<1:
        ap.error('Run counts must be positive')
    out=Path(args.out) if args.out else (Path(args.run)/'analysis' if args.run else HERE/'analysis_synthetic')
    if not args.synthetic_only and (not args.run or not args.execute_on_pod or not args.raw_read_note):
        ap.error('--run, --execute-on-pod and --raw-read-note are required for real metrics')
    if not args.synthetic_only and not confirmatory_parameters(args.n_boot,args.n_perm,args.n_splits,args.n_halves) and not args.allow_nonconfirmatory:
        ap.error('Reduced/changed run counts require --allow-nonconfirmatory; no confirmatory decisions will be emitted')
    if not args.synthetic_only and platform.system()!='Linux':
        ap.error('Real heavy analysis requires a Linux pod; this host may run --synthetic-only')
    out.mkdir(parents=True,exist_ok=True)
    g6=synthetic_checks()  # Always rerun current code BEFORE loading real inputs or resuming a metric.
    atomic_json(out/'G6.json',finite_json(g6))
    if not g6['pass']:
        raise AnalysisError('G6 synthetic checks failed; no real metric may run')
    if args.synthetic_only:
        print(f'G6 PASS: {out / "G6.json"}')
        return
    run=load_verified_run(args.run,args.raw_read_note)
    meta=provenance(run,args)
    identity={k:v for k,v in meta.items() if k not in ('command','git_status','git_diff_sha256')}
    fingerprint=digest(identity)
    mp=out/'provenance.json'
    if mp.exists() and json.loads(mp.read_text())['analysis_fingerprint']!=fingerprint:
        raise AnalysisError('Analysis provenance changed; choose a new output folder')
    atomic_json(mp,dict(meta,analysis_fingerprint=fingerprint))
    cells=prompt_cells(run['rows'],run['acts'][:,STATE],run['roles'],run['formats'],run['questions'])
    np.save(out/'prompt_counts.npy',cells['counts'])
    phase(out,'integrity',lambda:run['verified'])
    one=phase(out,'D1',lambda:dict(d1(cells['means'],args.n_boot,args.seed,artifacts_dir=out),
                                              question_ids=run['questions']))
    phase(out,'all_valid_fit',lambda:all_valid_fit(cells,run['acts'][:,STATE]))
    emb=embedding_cache(run,out,args.embedding_device)
    two=phase(out,'D2',lambda:d2(cells,run['rows'],run['acts'][:,0],emb,args.n_boot,args.n_perm,args.n_splits,args.seed,out,
                           args.null_workers,args.null_blas_threads,args.cache_null_scalers,args.null_backend,args.null_device))
    three=phase(out,'D3',lambda:dict(d3(cells['means'],args.n_boot,args.n_perm,args.n_halves,args.seed,out),
                                                format_names=run['formats']))
    phase(out,'sensitivities',lambda:report_sensitivities(run,cells,args))
    report=decision_report({'D1':one,'D2':two,'D3':three},meta['confirmatory'],args.holm_alpha)
    atomic_json(out/'decisions.json',report)
    print(f'Analysis saved: {out}; confirmatory={meta["confirmatory"]}')


def descriptive_decoding(cells,n_splits=N_SPLITS,n_boot=N_BOOT,seed=SEED):
    """Same fixed question/wording/format folds, point and interval; no report-only permutation claim."""
    p=flatten_cells(cells)
    try:
        folds=make_splits(p,p['F'],n_splits,seed)
        draws=crossed_draws(p['R'],p['Q'],n_boot,seed)
        reports={cells['formats'][f]:decode_report(p['X'],p,fs,draws)[0] for f,fs in folds.items()}
    except AnalysisError as e:
        return dict(status='not_estimable',reason=str(e))
    return dict(status='estimated',formats=reports,chance=1/p['R'],n_splits=n_splits,
                scope='Report-only same-fold accuracy; no null or confirmatory decision.')


def decision_report(results,confirmatory,alpha=None):
    ps={name:result.get('p') for name,result in results.items()}
    adjusted=holm(ps)
    status='pending_owner_alpha' if alpha is None else ('evaluated' if confirmatory else 'non_confirmatory')
    decisions={}
    for name,value in results.items():
        rule=value.get('raw_rule_met')
        family=None if alpha is None or not confirmatory or ps[name] is None else bool(rule and adjusted[name]<alpha)
        decisions[name]=dict(raw_p=ps[name],holm_p=adjusted[name],raw_rule_met=rule,family_decision=family)
    return dict(confirmatory=confirmatory,family_status=status,holm_alpha=alpha,decisions=decisions,
                warning=None if confirmatory else 'NON-CONFIRMATORY: changed run counts')


if __name__=='__main__':
    try:
        main()
    except AnalysisError as e:
        raise SystemExit(f'Analysis refused: {e}')
