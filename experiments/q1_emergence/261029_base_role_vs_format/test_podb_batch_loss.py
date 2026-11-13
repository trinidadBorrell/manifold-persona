"""CPU-only comparisons with sklearn's actual multinomial objective."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.linear_model._linear_loss import LinearModelLoss

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('podb_batch_loss_test_target',HERE/'analysis_batch_loss.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


def fixture(B=3,C=4,N=19,D=7,scale=1.):
    rng=np.random.default_rng(513)
    X=rng.normal(size=(N,D)).astype(np.float64)
    coef=(scale*rng.normal(size=(B,(D+1)*C))).astype(np.float64)
    y=rng.integers(C,size=(B,N))
    weights=rng.lognormal(size=(B,N)).astype(np.float64)
    weights[:,::5]=0
    strengths=np.linspace(0,.27,B,dtype=np.float64)
    return X,coef,y,weights,strengths


def sklearn_values(X,coefs,targets,weights,strengths):
    C=coefs.shape[1]//(X.shape[1]+1)
    objective=LinearModelLoss(HalfMultinomialLoss(n_classes=C),fit_intercept=True)
    values=[objective.loss_gradient(np.ascontiguousarray(c),X,np.ascontiguousarray(y,dtype=np.float64),
                                   sample_weight=None if weights is None else np.ascontiguousarray(weights[i]),
                                   l2_reg_strength=float(strengths[i]),n_threads=1)
            for i,(c,y) in enumerate(zip(coefs,targets))]
    return np.array([v[0] for v in values]),np.array([v[1] for v in values])


@pytest.mark.parametrize('B,C',[(1,2),(3,4),(7,10)])
@pytest.mark.parametrize('weighted',[False,True])
@pytest.mark.parametrize('scale',[0.,1.,1000.])
def test_batched_values_match_sklearn(B,C,weighted,scale):
    X,coefs,y,weights,l2=fixture(B=B,C=C,scale=scale)
    weights=weights if weighted else None
    got=M.MultinomialBatchLoss(X,device='cpu').evaluate(coefs,y,weights,l2)
    expected=sklearn_values(X,coefs,y,weights,l2)
    assert got[0].dtype==got[1].dtype==np.float64
    assert got[0].shape==(B,) and got[1].shape==coefs.shape
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    np.testing.assert_allclose(got[0],expected[0],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(got[1],expected[1],rtol=3e-12,atol=3e-13)


def test_intercept_is_unpenalized_and_fortran_flatten_is_correct():
    X,coefs,y,weights,l2=fixture(B=3,C=4)
    obj=M.MultinomialBatchLoss(X,device='cpu')
    baseline=obj.evaluate(coefs,y,weights,np.zeros(3,dtype=np.float64))
    actual=obj.evaluate(coefs,y,weights,l2)
    params=coefs.reshape(3,X.shape[1]+1,4)
    expected_penalty=.5*l2*np.sum(params[:,:-1]**2,axis=(1,2))
    np.testing.assert_allclose(actual[0]-baseline[0],expected_penalty,rtol=2e-14,atol=2e-14)
    gradient_change=(actual[1]-baseline[1]).reshape(params.shape)
    np.testing.assert_allclose(gradient_change[:,:-1],l2[:,None,None]*params[:,:-1],rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(gradient_change[:,-1],np.zeros((3,4)))


def test_extreme_common_intercept_offset_matches_original_formula():
    X,coefs,y,weights,l2=fixture(B=3,C=4,scale=100.)
    coefs.reshape(3,8,4)[:,-1]+=1e8
    actual=M.MultinomialBatchLoss(X,device='cpu').evaluate(coefs,y,weights,l2)
    expected=sklearn_values(X,coefs,y,weights,l2)
    np.testing.assert_allclose(actual[0],expected[0],rtol=2e-12,atol=2e-8)
    np.testing.assert_allclose(actual[1],expected[1],rtol=2e-11,atol=2e-10)


def test_predict_matches_exact_logits_argmax_and_zero_ties():
    X,coefs,y,weights,l2=fixture(B=3,C=4)
    Xte=np.random.default_rng(24).normal(size=(13,7)).astype(np.float64)
    obj=M.MultinomialBatchLoss(X,device='cpu')
    expected=np.array([(Xte@p[:-1]+p[-1]).argmax(1) for p in coefs.reshape(3,8,4)])
    np.testing.assert_array_equal(obj.predict(coefs,Xte),expected)
    np.testing.assert_array_equal(obj.predict(np.zeros_like(coefs),Xte),np.zeros((3,13),dtype=int))
    assert obj.predict(coefs,np.zeros((0,7),dtype=np.float64)).shape==(3,0)


def test_training_matrix_is_owned_and_input_arrays_are_not_mutated():
    X,coefs,y,weights,l2=fixture()
    obj=M.MultinomialBatchLoss(X,device='cpu')
    arrays=[coefs.copy(),y.copy(),weights.copy(),l2.copy()]
    expected=obj.evaluate(coefs,y,weights,l2)
    X[:]=1000
    actual=obj.evaluate(coefs,y,weights,l2)
    for a,b in zip((coefs,y,weights,l2),arrays):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])


@pytest.mark.parametrize('bad',['float32','nonfinite','shape','empty'])
def test_invalid_training_matrix_is_rejected(bad):
    X=np.ones((5,3),dtype=np.float64)
    if bad=='float32':X=X.astype(np.float32)
    elif bad=='nonfinite':X[0,0]=np.nan
    elif bad=='shape':X=X.ravel()
    else:X=X[:0]
    with pytest.raises(ValueError,match='X_cpu|float64|finite|shape|samples'):
        M.MultinomialBatchLoss(X,device='cpu')


@pytest.mark.parametrize('bad',['coef_dtype','coef_shape','coef_nonfinite','target_dtype','target_shape',
                              'target_negative','target_large','weight_shape','weight_nonfinite','weight_sum',
                              'l2_shape','l2_nonfinite'])
def test_invalid_evaluation_inputs_are_rejected(bad):
    X,c,y,w,l2=fixture()
    if bad=='coef_dtype':c=c.astype(np.float32)
    elif bad=='coef_shape':c=c[:,:-1]
    elif bad=='coef_nonfinite':c[0,0]=np.inf
    elif bad=='target_dtype':y=y.astype(np.float64)
    elif bad=='target_shape':y=y[:,:-1]
    elif bad=='target_negative':y[0,0]=-1
    elif bad=='target_large':y[0,0]=4
    elif bad=='weight_shape':w=w[:,:-1]
    elif bad=='weight_nonfinite':w[0,0]=np.nan
    elif bad=='weight_sum':w[0]=0
    elif bad=='l2_shape':l2=l2[:-1]
    else:l2[0]=np.inf
    with pytest.raises(ValueError):M.MultinomialBatchLoss(X,device='cpu').evaluate(c,y,w,l2)


def test_predict_rejects_mismatched_or_nonfinite_test_matrix():
    X,c,y,w,l2=fixture();obj=M.MultinomialBatchLoss(X,device='cpu')
    for Xte in (np.zeros((5,8),dtype=np.float64),np.full((5,7),np.nan),np.zeros((5,7),dtype=np.float32)):
        with pytest.raises(ValueError):obj.predict(c,Xte)
