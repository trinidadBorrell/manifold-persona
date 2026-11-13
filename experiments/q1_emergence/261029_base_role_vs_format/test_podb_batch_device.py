"""CPU-tensor tests for the device-resident multinomial objective."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

M=load('podb_batch_device_test_target',HERE/'analysis_batch_device.py')
R=load('podb_batch_device_verified_reference',HERE/'analysis_batch_loss.py')


def data(B=3,C=4,D=5,N=17,scale=1.):
    rng=np.random.default_rng(81)
    X=rng.normal(size=(N,D)).astype(np.float64)
    y=rng.integers(C,size=(B,N))
    w=rng.lognormal(size=(B,N)).astype(np.float64);w[:,::5]=0
    l2=np.linspace(0,.15,B,dtype=np.float64)
    coefs=(scale*rng.normal(size=(B,(D+1)*C))).astype(np.float64)
    return X,y,w,l2,coefs


@pytest.mark.parametrize('B,C',[(1,2),(3,4),(7,10)])
@pytest.mark.parametrize('scale',[0.,1.,1000.])
@pytest.mark.parametrize('weighted',[True,False])
def test_device_evaluation_matches_verified_reference(B,C,scale,weighted):
    X,y,w,l2,coefs=data(B,C,scale=scale);w=w if weighted else None
    model=M.DeviceMultinomialBatchLoss(X,y,w,l2,device='cpu')
    expected=R.MultinomialBatchLoss(X,device='cpu').evaluate(coefs,y,w,l2)
    losses,gradient=model.evaluate(torch.from_numpy(coefs))
    assert losses.device==gradient.device==model.device==torch.device('cpu')
    assert losses.dtype==gradient.dtype==torch.float64
    assert losses.shape==(B,) and gradient.shape==coefs.shape
    np.testing.assert_allclose(losses.numpy(),expected[0],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(gradient.numpy(),expected[1],rtol=3e-12,atol=3e-13)


def test_device_gradient_matches_central_finite_differences():
    X,y,w,l2,c=data(B=2,C=3,D=3,N=9)
    model=M.DeviceMultinomialBatchLoss(X,y,w,l2,device='cpu');coef=torch.from_numpy(c)
    _,gradient=model.evaluate(coef)
    for b,j in ((0,0),(0,11),(1,4),(1,10)):
        plus=coef.clone();minus=coef.clone();plus[b,j]+=1e-6;minus[b,j]-=1e-6
        numerical=(model.evaluate(plus)[0][b]-model.evaluate(minus)[0][b])/(2e-6)
        assert numerical.item()==pytest.approx(gradient[b,j].item(),rel=2e-7,abs=2e-9)


def test_evaluation_has_no_host_transfer_and_reuses_constants(monkeypatch):
    X,y,w,l2,c=data();model=M.DeviceMultinomialBatchLoss(X,y,w,l2,device='cpu');coef=torch.from_numpy(c)
    model.evaluate(coef)
    one_hot=model._one_hot.data_ptr();normalized=model._normalized_weights.data_ptr()
    def forbidden(*args,**kwargs):pytest.fail('evaluate must not transfer to host or copy constants')
    with monkeypatch.context() as patch:
        for name in ('cpu','numpy','item'):patch.setattr(torch.Tensor,name,forbidden)
        patch.setattr(model,'_tensor',forbidden)
        losses,gradient=model.evaluate(coef+.1)
    assert losses.dtype==gradient.dtype==torch.float64
    assert model._one_hot.data_ptr()==one_hot and model._normalized_weights.data_ptr()==normalized


def test_constants_are_owned_and_predict_matches_reference():
    X,y,w,l2,c=data();model=M.DeviceMultinomialBatchLoss(X,y,w,l2,device='cpu')
    reference=R.MultinomialBatchLoss(X,device='cpu');expected=reference.evaluate(c,y,w,l2)
    X[:]=1000;y[:]=0;w[:]=1;l2[:]=100
    losses,gradient=model.evaluate(torch.from_numpy(c))
    np.testing.assert_allclose(losses.numpy(),expected[0],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(gradient.numpy(),expected[1],rtol=3e-12,atol=3e-13)
    Xte=np.random.default_rng(5).normal(size=(11,5)).astype(np.float64)
    pred=model.predict(torch.from_numpy(c),Xte)
    assert pred.device==model.device and pred.dtype==torch.int64
    np.testing.assert_array_equal(pred.numpy(),reference.predict(c,Xte))


@pytest.mark.parametrize('bad',['targets_dtype','targets_shape','targets_negative','weights_shape','weights_zero','l2_shape','l2_nan'])
def test_invalid_constants_fail_clearly(bad):
    X,y,w,l2,c=data()
    if bad=='targets_dtype':y=y.astype(float)
    elif bad=='targets_shape':y=y[:,:-1]
    elif bad=='targets_negative':y[0,0]=-1
    elif bad=='weights_shape':w=w[:,:-1]
    elif bad=='weights_zero':w[0]=0
    elif bad=='l2_shape':l2=l2[:-1]
    else:l2[0]=np.nan
    with pytest.raises(ValueError):M.DeviceMultinomialBatchLoss(X,y,w,l2,device='cpu')


def test_invalid_coefficient_metadata_and_class_changes_are_rejected():
    X,y,w,l2,c=data();model=M.DeviceMultinomialBatchLoss(X,y,w,l2,device='cpu')
    for bad in (c,torch.from_numpy(c).float(),torch.from_numpy(c[:,:-1]),torch.from_numpy(c[:1])):
        with pytest.raises(ValueError):model.evaluate(bad)
    model.evaluate(torch.from_numpy(c))
    with pytest.raises(ValueError):model.evaluate(torch.zeros((3,6*5),dtype=torch.float64))


def test_missing_observed_class_can_be_inferred_from_coefficient_layout():
    X,y,w,l2,c=data();y[:]=0
    model=M.DeviceMultinomialBatchLoss(X,y,w,l2,device='cpu')
    expected=R.MultinomialBatchLoss(X,device='cpu').evaluate(c,y,w,l2)
    actual=model.evaluate(torch.from_numpy(c))
    np.testing.assert_allclose(actual[0].numpy(),expected[0],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(actual[1].numpy(),expected[1],rtol=3e-12,atol=3e-13)
