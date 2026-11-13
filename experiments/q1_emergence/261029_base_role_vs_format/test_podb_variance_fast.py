"""Synthetic reference comparisons for the frozen Pod B D1 bootstrap kernel."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


V=load('podb_variance_fast_tests_target',HERE/'analysis_variance_fast.py')
A=load('podb_variance_fast_direct_reference',HERE/'analysis.py')


def reference(grid):
    return A.variance_split(grid,False)


def direct_values(x,draws):
    values=[]
    for rr,qq in draws:
        value=reference(x[rr][:,:,qq])['role_minus_format']
        values.append(np.nan if value is None else value)
    return np.asarray(values)


def assert_same_decision(actual,expected):
    np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
    finite=np.isfinite(expected)
    np.testing.assert_array_equal(np.signbit(actual[finite]),np.signbit(expected[finite]))
    np.testing.assert_array_equal(actual[finite]<=0,expected[finite]<=0)
    if finite.all():
        assert np.mean(actual<=0)==np.mean(expected<=0)
        ac=np.percentile(actual,[2.5,97.5]);ec=np.percentile(expected,[2.5,97.5])
        np.testing.assert_array_equal(ac>0,ec>0)
        np.testing.assert_allclose(ac,ec,rtol=2e-12,atol=2e-14)


def test_random_full_rank_matches_original_without_per_draw_reference_calls():
    x=np.random.default_rng(41).normal(size=(6,4,9,5,13))
    draws=A.crossed_draws(6,9,n=100,seed=17)
    def no_fallback(grid):
        pytest.fail('well-conditioned random data should use sufficient statistics')
    actual,info=V.bootstrap_differences(x,draws,reference=no_fallback,return_info=True)
    expected=direct_values(x,draws)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-14)
    assert_same_decision(actual,expected)
    assert info['fallback_count']==0


def test_frozen_axis_sizes_with_small_synthetic_feature_width():
    x=np.random.default_rng(1).normal(size=(10,4,40,5,3))
    draws=A.crossed_draws(10,40,n=40,seed=10)
    actual=V.bootstrap_differences(x,draws,reference=reference)
    expected=direct_values(x,draws)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-14)
    assert_same_decision(actual,expected)


def test_large_offsets_and_small_variation_use_exact_reference_guard():
    x=1e12+np.random.default_rng(3).normal(scale=.01,size=(4,4,6,5,7))
    draws=A.crossed_draws(4,6,n=30,seed=23)
    actual,info=V.bootstrap_differences(x,draws,reference=reference,return_info=True)
    expected=direct_values(x,draws)
    np.testing.assert_array_equal(actual,expected)
    assert_same_decision(actual,expected)
    assert info['fallback_count']==len(draws)


def test_duplicate_heavy_draws_retain_crossed_multiplicities_and_order():
    x=np.random.default_rng(2).normal(size=(5,4,7,5,11))
    draws=[(np.array([0,0,0,1,4]),np.array([1,1,1,1,2,2,6])),
           (np.array([4,0,1,0,0]),np.array([6,2,1,2,1,1,1])),
           (np.array([2,2,2,2,2]),np.array([3,3,3,3,3,3,3]))]
    originals=[(r.copy(),q.copy()) for r,q in draws]
    actual=V.bootstrap_differences(x,draws,reference=reference)
    expected=direct_values(x,draws)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-14)
    assert_same_decision(actual,expected)
    for (r,q),(r0,q0) in zip(draws,originals):
        np.testing.assert_array_equal(r,r0);np.testing.assert_array_equal(q,q0)


@pytest.mark.parametrize('kind',['role','format','interaction','question','wording'])
def test_planted_effects_keep_exact_decision_p_and_signs(kind):
    rng=np.random.default_rng(31)
    shape=(6,4,8,5,9)
    if kind=='role':base=rng.normal(size=(6,1,1,1,9))
    elif kind=='format':base=rng.normal(size=(1,4,1,1,9))
    elif kind=='question':base=rng.normal(size=(1,1,8,1,9))
    elif kind=='wording':base=rng.normal(size=(6,1,1,5,9))
    else:
        base=rng.normal(size=(6,4,1,1,9))
        base=base-base.mean(0,keepdims=True)-base.mean(1,keepdims=True)+base.mean((0,1),keepdims=True)
    x=np.broadcast_to(base,shape).copy()
    draws=A.crossed_draws(6,8,n=100,seed=54)
    draws.append((np.arange(6),np.arange(8)))
    actual=V.bootstrap_differences(x,draws,reference=reference)
    expected=direct_values(x,draws)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-14)
    assert_same_decision(actual,expected)


@pytest.mark.parametrize('epsilon',[-1e-13,0,1e-13])
def test_near_zero_numerator_falls_back_and_preserves_exact_zero_rule(epsilon):
    role=np.array([-1.,1.])[:,None,None,None,None]
    fmt=np.array([-1.,1.])[None,:,None,None,None]*(1+epsilon)
    x=np.broadcast_to(role+fmt,(2,2,6,5,3)).copy()
    draws=[(np.array([0,1]),np.array([0,1,1,3,4,5])),(np.array([1,0]),np.array([5,4,3,1,1,0]))]
    actual,info=V.bootstrap_differences(x,draws,reference=reference,return_info=True)
    expected=direct_values(x,draws)
    np.testing.assert_array_equal(actual,expected)
    assert_same_decision(actual,expected)
    assert info['fallback_count']==len(draws)


@pytest.mark.parametrize('constant',[0.,1.,.1])
def test_degenerate_total_preserves_original_nans_and_roundoff(constant):
    x=np.full((4,4,6,5,7),constant)
    draws=A.crossed_draws(4,6,n=12,seed=7)
    actual,info=V.bootstrap_differences(x,draws,reference=reference,return_info=True)
    expected=direct_values(x,draws)
    np.testing.assert_array_equal(actual,expected)
    assert_same_decision(actual,expected)
    assert info['fallback_count']==len(draws)


def test_complete_question_selection_stays_with_existing_analysis_rule():
    x=np.random.default_rng(3).normal(size=(4,4,8,5,7));x[0,0,0,0]=np.nan
    assert A.complete_grid(x,min_questions=8)['status']=='not_estimable'
    complete=A.complete_grid(x,min_questions=7)
    draws=A.crossed_draws(4,7,n=20,seed=12)
    actual=V.bootstrap_differences(complete['grid'],draws,reference=reference)
    expected=direct_values(complete['grid'],draws)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-14)
    with pytest.raises(ValueError,match='finite|complete'):
        V.bootstrap_differences(x,draws,reference=reference)


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_inputs_are_rejected(value):
    x=np.ones((3,4,5,5,7));x[0,0,0,0,0]=value
    with pytest.raises(ValueError,match='finite|complete'):
        V.bootstrap_differences(x,A.crossed_draws(3,5,n=1))


def test_default_reference_and_empty_draw_list():
    x=np.ones((3,4,5,5,7));draws=A.crossed_draws(3,5,n=2)
    assert np.isnan(V.bootstrap_differences(x,draws)).all()
    assert V.bootstrap_differences(x,[]).shape==(0,)


@pytest.mark.parametrize('bad_draw',[(np.array([0,1]),np.arange(5)),
                                     (np.array([0,1,9]),np.arange(5)),
                                     (np.array([0.,1.,2.]),np.arange(5)),
                                     (np.arange(3),np.array([0,1,2,3,-1]))])
def test_malformed_draws_never_silently_change_the_bootstrap(bad_draw):
    with pytest.raises(ValueError,match='draw|indices|length'):
        V.bootstrap_differences(np.ones((3,4,5,5,7)),[bad_draw])


def test_percentile_interpolation_near_zero_uses_whole_reference_vector():
    # The 2.5th percentile interpolates -1 and 1/39: .025*(-1)+.975*(1/39)=0.
    # Neither individual difference is near zero; the CI-sign guard is necessary.
    role=np.array([-1.,1.])[:,None,None,None,None]*np.sqrt(20/19)
    fmt=np.array([-1.,1.])[None,:,None,None,None]
    x=np.broadcast_to(role+fmt,(2,2,6,5,3)).copy()
    draws=[(np.array([0,0]),np.arange(6))]+[(np.array([0,1]),np.arange(6)) for _ in range(39)]
    actual,info=V.bootstrap_differences(x,draws,reference=reference,return_info=True)
    expected=direct_values(x,draws)
    np.testing.assert_array_equal(actual,expected)
    assert info['ci_guard_fallback']
    assert info['fallback_count']==40
    assert_same_decision(actual,expected)


def test_scalar_reference_callback_keeps_zero_variance_none_behavior():
    x=np.zeros((3,4,5,5,7));draws=A.crossed_draws(3,5,n=3)
    values=V.bootstrap_differences(x,draws,reference=lambda grid:reference(grid)['role_minus_format'])
    assert np.isnan(values).all()


@pytest.mark.parametrize('offset,noise',[(1e2,.01),(1e3,.001),(1e5,.01),(1e8,1.)])
def test_offset_conditioning_also_preserves_tight_value_agreement(offset,noise):
    x=offset+np.random.default_rng(18).normal(scale=noise,size=(4,4,6,5,7))
    draws=A.crossed_draws(4,6,n=12,seed=4)
    actual=V.bootstrap_differences(x,draws,reference=reference)
    expected=direct_values(x,draws)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-14)
    assert_same_decision(actual,expected)


def test_full_frozen_draw_count_uses_no_full_grid_reference_copies():
    x=np.random.default_rng(101).normal(size=(10,4,40,5,8))
    draws=A.crossed_draws(10,40,n=2000,seed=20261029)
    def no_fallback(grid):
        pytest.fail('well-conditioned full-count synthetic run should stay on the Gram path')
    actual,info=V.bootstrap_differences(x,draws,reference=no_fallback,return_info=True)
    assert actual.shape==(2000,) and np.isfinite(actual).all()
    assert info['fallback_count']==0
    expected=direct_values(x,draws[:16])
    np.testing.assert_allclose(actual[:16],expected,rtol=2e-12,atol=2e-14)
    assert_same_decision(actual[:16],expected)
