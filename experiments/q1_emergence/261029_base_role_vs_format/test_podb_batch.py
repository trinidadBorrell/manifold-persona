"""The production call seam keeps sklearn/SciPy fits independent while batching callbacks."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import _logistic
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('podb_batch_under_test',HERE/'analysis_batch.py')
B=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=B
spec.loader.exec_module(B)


def fixture(seed=3):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(100,12));T=rng.normal(size=(41,12))
    Y=np.stack([rng.permutation(np.arange(100)%4) for _ in range(8)])
    return X,Y,T


def test_batched_fits_match_original_cpu_predictions_and_iterations():
    X,Y,T=fixture();original=_logistic.LinearModelLoss
    with threadpool_limits(limits=1):
        ref=[LogisticRegression(C=1,class_weight='balanced',max_iter=2000).fit(X,y) for y in Y]
    pred,info=B.fit_predict_batch(X,Y,T,device='cpu',wait_seconds=.002,verify_callbacks=2)
    np.testing.assert_array_equal(pred,np.stack([m.predict(T) for m in ref]))
    np.testing.assert_array_equal(info['n_iter'],[int(m.n_iter_[0]) for m in ref])
    assert info['fits']==8 and info['largest_callback_batch']>1
    assert info['callback_reference_checks']==2
    assert _logistic.LinearModelLoss is original


def test_unequal_balanced_class_weights_preserved():
    X,Y,T=fixture();Y=Y[:3].copy();Y[:,50:]=0
    with threadpool_limits(limits=1):
        expected=np.stack([LogisticRegression(C=1,class_weight='balanced',max_iter=2000).fit(X,y).predict(T) for y in Y])
    actual,info=B.fit_predict_batch(X,Y,T,device='cpu')
    np.testing.assert_array_equal(actual,expected)


def test_evaluator_failure_releases_all_fit_threads_and_restores_alias(monkeypatch):
    X,Y,T=fixture();original=_logistic.LinearModelLoss
    class Broken:
        def __init__(self,*args,**kwargs):pass
        def evaluate(self,*args,**kwargs):raise ValueError('injected kernel failure')
    monkeypatch.setattr(B,'loss_class',lambda:Broken)
    with pytest.raises(ValueError,match='injected kernel failure'):
        B.fit_predict_batch(X,Y,T,device='cpu')
    assert _logistic.LinearModelLoss is original


def test_each_fit_is_fresh_and_uses_frozen_estimator(monkeypatch):
    X,Y,T=fixture();seen=[];original=LogisticRegression.fit
    def record(self,*args,**kwargs):
        assert self.C==1 and self.class_weight=='balanced' and self.max_iter==2000
        assert self.solver=='lbfgs' and self.tol==1e-4 and not self.warm_start
        seen.append(self)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(LogisticRegression,'fit',record)
    B.fit_predict_batch(X,Y[:4],T,device='cpu')
    assert len(seen)==len({id(m) for m in seen})==4


@pytest.mark.parametrize('mutate',[
    lambda X,Y,T:(X.astype(np.float32),Y,T),
    lambda X,Y,T:(X,Y.astype(float),T),
    lambda X,Y,T:(X,Y[:,:-1],T),
    lambda X,Y,T:(X,Y,T[:,:-1]),
])
def test_invalid_batch_refused(mutate):
    with pytest.raises(ValueError):B.fit_predict_batch(*mutate(*fixture()),device='cpu')
