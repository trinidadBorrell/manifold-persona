import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('podb_device_lbfgs_test',HERE/'analysis_batch_lbfgs.py')
B=importlib.util.module_from_spec(spec);sys.modules[spec.name]=B;spec.loader.exec_module(B)


def fixture(seed=41):
    rng=np.random.default_rng(seed);X=rng.normal(size=(120,9));T=rng.normal(size=(51,9))
    Y=np.stack([rng.permutation(np.arange(120)%4) for _ in range(5)])
    return X,Y,T


def test_device_batch_matches_fresh_original_classifier_predictions():
    X,Y,T=fixture()
    with threadpool_limits(limits=1):
        refs=[LogisticRegression(C=1,class_weight='balanced',max_iter=2000).fit(X,y) for y in Y]
    pred,info=B.fit_predict_batch(X,Y,T,device='cpu')
    np.testing.assert_array_equal(pred,np.stack([m.predict(T) for m in refs]))
    assert info['fits']==len(Y) and info['all_converged']
    assert info['gtol']==1e-4 and info['max_iter']==2000
    assert info['ftol']==64*np.finfo(float).eps and info['history_size']==10


def test_each_optimizer_is_independent_of_other_fits():
    X,Y,T=fixture()
    actual,info=B.fit_predict_batch(X,Y,T,device='cpu',return_parameters=True)
    for i in range(len(Y)):
        pred,one=B.fit_predict_batch(X,Y[i:i+1],T,device='cpu',return_parameters=True)
        np.testing.assert_array_equal(pred[0],actual[i])
        np.testing.assert_allclose(one['parameters'][0],info['parameters'][i],rtol=1e-9,atol=1e-9)
        assert one['n_iter'][0]==info['n_iter'][i]


def test_balanced_class_weights_are_preserved():
    X,Y,T=fixture();Y=Y[:2].copy();Y[:,60:]=0
    with threadpool_limits(limits=1):
        expected=np.stack([LogisticRegression(C=1,class_weight='balanced',max_iter=2000).fit(X,y).predict(T) for y in Y])
    actual,info=B.fit_predict_batch(X,Y,T,device='cpu')
    np.testing.assert_array_equal(actual,expected)


def test_zero_gradient_stops_each_fit_without_an_update():
    X=np.zeros((30,4));Y=np.stack([np.arange(30)%3,np.arange(30)%3]);T=np.zeros((7,4))
    actual,info=B.fit_predict_batch(X,Y,T,device='cpu')
    np.testing.assert_array_equal(actual,np.zeros((2,7),dtype=int));assert info['n_iter']==[0,0]


def test_unconverged_fit_is_an_error():
    with pytest.raises(B.BatchConvergenceError):B.fit_predict_batch(*fixture(),device='cpu',max_iter=1)


def test_optimizer_on_independent_planted_quadratics(monkeypatch):
    import torch
    class Quadratics:
        def __init__(self,X,Y,weight,strength,device):
            self.device=torch.device(device);P=(X.shape[1]+1)*4
            self.goal=torch.as_tensor(np.random.default_rng(11).normal(size=(len(Y),P)),dtype=torch.float64)
            self.diag=torch.linspace(.5,4,P,dtype=torch.float64)[None,:]
        def evaluate(self,x):
            diff=x-self.goal
            return .5*(self.diag*diff*diff).sum(1),self.diag*diff
        def predict(self,x,T):return torch.zeros((len(x),len(T)),dtype=torch.long)
    monkeypatch.setattr(B,'_device_loss_class',lambda:Quadratics)
    X,Y,T=fixture();pred,info=B.fit_predict_batch(X,Y,T,device='cpu',return_parameters=True)
    expected=np.random.default_rng(11).normal(size=(len(Y),40))
    np.testing.assert_allclose(info['parameters'],expected,atol=2e-4,rtol=0)


def test_compact_history_matches_explicit_independent_bfgs_matrices():
    import torch
    rng=np.random.default_rng(91);M=4;N=3;P=9
    s=rng.normal(size=(M,N,P));a=rng.normal(size=(N,P,P))
    h=np.einsum('bji,bjk->bik',a,a)+np.eye(P)[None,:,:]
    y=np.einsum('bij,mbj->mbi',h,s)
    cursor=np.array([0,1,3]);counts=np.array([0,2,4]);gamma=np.array([1.,.4,.7]);g=rng.normal(size=(N,P))
    expected=[]
    for b in range(N):
        H=gamma[b]*np.eye(P)
        for slot in [(cursor[b]-age)%M for age in range(counts[b],0,-1)]:
            sv=s[slot,b];yv=y[slot,b];rho=1/(sv@yv);v=np.eye(P)-rho*np.outer(sv,yv)
            H=v@H@v.T+rho*np.outer(sv,sv)
        expected.append(H@g[b])
    tensor=lambda z:torch.as_tensor(z)
    got=B.compact_inverse_product(tensor(g),tensor(s),tensor(y),tensor(cursor),tensor(counts),tensor(gamma),M)
    np.testing.assert_allclose(got.numpy(),expected,rtol=2e-12,atol=2e-12)


def test_compact_fit_matches_two_loop_fit():
    X,Y,T=fixture()
    a,ia=B.fit_predict_batch(X,Y,T,device='cpu',return_parameters=True)
    b,ib=B.fit_predict_batch(X,Y,T,device='cpu',return_parameters=True,history_mode='compact')
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(ia['parameters'],ib['parameters'],atol=2e-11,rtol=2e-11)
    assert ia['n_iter']==ib['n_iter']
