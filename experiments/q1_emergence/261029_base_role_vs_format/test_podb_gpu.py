import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('podb_gpu_test',HERE/'analysis_gpu.py')
G=importlib.util.module_from_spec(spec);sys.modules[spec.name]=G;spec.loader.exec_module(G)


def test_runtime_requires_explicit_array_api_and_deterministic_configuration(monkeypatch):
    monkeypatch.delenv('SCIPY_ARRAY_API',raising=False)
    with pytest.raises(G.GPUAnalysisError,match='SCIPY_ARRAY_API'):
        G.runtime_info()
    monkeypatch.setenv('SCIPY_ARRAY_API','1');monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    with pytest.raises(G.GPUAnalysisError,match='CUBLAS'):
        G.runtime_info()


@pytest.mark.parametrize('fraction',[0,-1,.11])
def test_preparation_rejects_unbounded_memory_fraction(fraction):
    with pytest.raises(G.GPUAnalysisError,match='memory fraction'):
        G.prepare_cuda_folds([],fraction)


@pytest.mark.parametrize('dtype',['float32','numpy'])
def test_array_path_rejects_precision_or_namespace_substitution(dtype):
    import torch
    x=np.ones((12,3)) if dtype=='numpy' else torch.ones((12,3),dtype=torch.float32)
    with pytest.raises(G.GPUAnalysisError,match='float64'):
        G.fit_predict_scaled_array_api(x,np.arange(12)%3,x)


def test_array_path_refits_the_same_balanced_estimator_and_restores_config(monkeypatch):
    import scipy
    if tuple(map(int,scipy.__version__.split('.')[:2]))<(1,14):
        pytest.skip('Array API execution requires SciPy >=1.14; run this on the pinned pod runtime')
    import torch
    from sklearn import get_config
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(71);y=np.repeat(np.arange(3),30)
    x=rng.normal(size=(90,6))+np.eye(3,6)[y]*3
    expected=LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000).fit(x,y).predict(x)
    fits=[];original=LogisticRegression.fit
    def record(self,X,labels,*a,**kw):
        fits.append(self)
        assert self.C==1 and self.class_weight=='balanced' and self.max_iter==2000
        assert self.solver=='lbfgs' and self.tol==1e-4
        assert isinstance(labels,np.ndarray) and np.array_equal(labels,y)
        return original(self,X,labels,*a,**kw)
    monkeypatch.setattr(LogisticRegression,'fit',record)
    initial=get_config()['array_api_dispatch']
    xx=torch.tensor(x,dtype=torch.float64)
    for _ in range(2):
        np.testing.assert_array_equal(G.fit_predict_scaled_array_api(xx,y,xx),expected)
    assert len(fits)==2 and fits[0] is not fits[1]
    assert get_config()['array_api_dispatch']==initial


def test_labels_and_feature_width_are_preserved():
    import torch
    x=torch.ones((12,3),dtype=torch.float64)
    with pytest.raises(G.GPUAnalysisError,match='feature width'):
        G.fit_predict_scaled_array_api(x,np.arange(12)%3,torch.ones((2,4),dtype=torch.float64))
    with pytest.raises(G.GPUAnalysisError,match='integer training labels'):
        G.fit_predict_scaled_array_api(x,np.arange(12,dtype=float),x)
