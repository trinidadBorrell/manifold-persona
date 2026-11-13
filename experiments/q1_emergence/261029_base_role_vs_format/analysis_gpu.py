"""Float64 array-API execution of the unchanged frozen sklearn classifier.

Labels stay on the CPU, so balanced class counts use the original NumPy path.
Only the already fitted training-X transforms move to CUDA. No draws, model
parameters, stopping tolerances, scores or statistical rules are chosen here.
"""
from __future__ import annotations

import os
import numpy as np


class GPUAnalysisError(ValueError):
    pass


def runtime_info():
    if os.environ.get('SCIPY_ARRAY_API')!='1':
        raise GPUAnalysisError('Set SCIPY_ARRAY_API=1 before starting Python')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise GPUAnalysisError('Set deterministic CUBLAS_WORKSPACE_CONFIG before starting Python')
    import torch
    if not torch.cuda.is_available():
        raise GPUAnalysisError('CUDA is unavailable; no automatic CPU substitution')
    return dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                precision='float64',classifier='sklearn.LogisticRegression',solver='lbfgs',
                class_weight='balanced',label_device='cpu',array_api_dispatch=True,
                enforced_at_fit=dict(deterministic_algorithms=True,TF32=False),
                initial_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                memory_fraction=.08)


def prepare_cuda_folds(prepared,memory_fraction=.08):
    """Copy checked, exact training-only transforms once; share them across draws."""
    if not 0<memory_fraction<=.1:
        raise GPUAnalysisError('CUDA memory fraction must be positive and at most 0.1')
    runtime_info()
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(memory_fraction,0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    result=[]
    for pair in prepared:
        out={}
        for name in ('Xtr','Xte'):
            value=pair[name]
            if value.ndim!=2 or value.dtype!=np.float64 or not np.isfinite(value).all():
                raise GPUAnalysisError('CUDA requires finite float64 training/test transforms')
            out[name]=torch.tensor(np.asarray(value),dtype=torch.float64,device='cuda')
        result.append(out)
    return result


def fit_predict_scaled_array_api(Xtr,ytr,Xte):
    """A fresh identical estimator; callers keep convergence warnings as errors.

    Torch CPU tensors also work for offline tests of this exact code path.
    Production callers obtain CUDA tensors only through prepare_cuda_folds.
    """
    import torch
    from sklearn import config_context
    from sklearn.linear_model import LogisticRegression
    if not all(isinstance(x,torch.Tensor) and x.dtype==torch.float64 and x.ndim==2 for x in (Xtr,Xte)):
        raise GPUAnalysisError('Array API classifier requires float64 tensor matrices')
    if Xtr.device!=Xte.device or Xtr.shape[1]!=Xte.shape[1]:
        raise GPUAnalysisError('Training and test tensors must share device and feature width')
    if isinstance(ytr,torch.Tensor) and ytr.device.type!='cpu':
        raise GPUAnalysisError('Keep labels on CPU for deterministic balanced class counts')
    labels=np.asarray(ytr)
    if labels.shape!=(len(Xtr),) or labels.dtype.kind not in 'iu':
        raise GPUAnalysisError('Expected the original integer training labels')
    with config_context(array_api_dispatch=True):
        classifier=LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000)
        classifier.fit(Xtr,labels)
        result=classifier.predict(Xte)
    return result.cpu().numpy() if isinstance(result,torch.Tensor) else np.asarray(result)


def release_cuda_folds(prepared):
    """Release only this process's disposable GPU copies after its pool finishes."""
    prepared.clear()
    import torch
    torch.cuda.empty_cache()
