"""Sufficient statistics for the frozen Pod B D1 crossed bootstrap.

The caller supplies the already balanced, complete [role, format, question,
wording, feature] grid and the EXISTING role/question draws. This module does not
choose questions, make draws, impute prompts, calculate intervals, or set a pass
rule. The caller retains those frozen operations.

For weights a_r and b_q, let E_rq be the sum of squared globally centred rows,
m_rq the format/wording mean, and t_frq the wording mean. Three Gram contractions
suffice: G from m, its per-role diagonal blocks, and H=sum_f Gram(t_f). With
v_rq=a_r b_q and the original R,F,Q,W sizes, the desired numerator is

  F W/Q sum_r a_r (b' G_r b) - W/(R Q) (v' H v),

and the denominator is sum_rq a_r b_q E_rq - F W/(R Q) (v' G v).
Grand-mean terms cancel from the numerator. Duplicate roles and questions remain
frequency weights; no full feature grid is copied for a well-conditioned draw.

Conservative floating-point guards fall back to the supplied original reference
for uncertain totals, differences, or percentile signs. If no callback is given,
the existing sibling analysis.variance_split is loaded lazily for those draws.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_EPS=np.finfo(np.float64).eps
_REFERENCE=None


def _gamma(n):
    value=n*_EPS
    return value/(1-value) if value<1 else np.inf


def _existing_reference(sample):
    global _REFERENCE
    if _REFERENCE is None:
        name='_podb_variance_fast_original_reference'
        if name not in sys.modules:
            spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('analysis.py'))
            module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module
            spec.loader.exec_module(module)
        _REFERENCE=sys.modules[name].variance_split
    return _REFERENCE(sample,False)


def _reference_value(x,draw,callback):
    rr,qq=draw
    result=callback(x[rr][:,:,qq])  # Preserve the original indexing and summation order.
    value=result['role_minus_format'] if isinstance(result,dict) else result
    return np.nan if value is None else float(value)


def _validate(x,draws):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=5 or any(size==0 for size in x.shape):
        raise ValueError('D1 needs a nonempty [role,format,question,wording,feature] grid')
    # Keep validation memory bounded for a pod grid with thousands of feature dimensions.
    for r in range(x.shape[0]):
        if not np.isfinite(x[r]).all():
            raise ValueError('D1 needs a complete finite grid; apply the existing complete-question rule first')
    checked=[]
    for i,draw in enumerate(draws):
        if len(draw)!=2:
            raise ValueError(f'draw {i} must contain role and question indices')
        pair=[]
        for indices,size in zip(draw,(x.shape[0],x.shape[2])):
            indices=np.asarray(indices)
            if indices.shape!=(size,) or not np.issubdtype(indices.dtype,np.integer):
                raise ValueError(f'draw {i} indices must be integer vectors of the original axis length')
            if indices.min()<0 or indices.max()>=size:
                raise ValueError(f'draw {i} contains out-of-range indices')
            pair.append(indices)
        checked.append(tuple(pair))
    return x,checked


def _statistics(x):
    R,F,Q,W,D=x.shape
    # One feature-sized centred copy; all bootstrap operations below use small matrices.
    centred=x-x.mean((0,1,2,3),keepdims=True)
    energy=np.einsum('rfqwd,rfqwd->rq',centred,centred)
    means=centred.mean((1,3))
    flat=means.reshape(R*Q,D)
    grand=flat@flat.T
    role=np.stack([grand[r*Q:(r+1)*Q,r*Q:(r+1)*Q] for r in range(R)])
    wording_means=centred.mean(3)
    formats=np.zeros_like(grand)
    for f in range(F):
        flat=np.ascontiguousarray(wording_means[:,f]).reshape(R*Q,D)
        formats+=flat@flat.T
    raw_bounds=np.maximum(np.abs(x.min((0,1,2,3))),np.abs(x.max((0,1,2,3))))
    raw_scale=float(np.linalg.norm(raw_bounds))
    return energy,grand,role,formats,raw_scale


def _quadratic(weights,gram):
    return np.einsum('bi,bi->b',weights@gram,weights)


def bootstrap_differences(x,draws,reference=None,return_info=False):
    """Return D1 differences in draw order; optionally return (values, diagnostics).

    reference(sampled_grid) may return the original variance_split dictionary, a
    scalar difference, or None for zero variance. It is called only for guarded
    cases. A None reference result becomes NaN, exactly as in the existing d1().
    Numerical guards use float64 error scales, not an experimental effect floor.
    The percentile-sign guard refers only to the existing 95% percentile interval.
    """
    x,draws=_validate(x,draws)
    n=len(draws)
    info=dict(n_draws=n,fallback_count=0,fallback_indices=[],ci_guard_fallback=False,
              max_estimated_error=None,conditioning_fallback_count=0)
    if not n:
        values=np.empty(0,dtype=np.float64)
        return (values,info) if return_info else values
    R,F,Q,W,D=x.shape
    N=R*F*Q*W
    a=np.array([np.bincount(rr,minlength=R) for rr,_ in draws],dtype=np.float64)
    b=np.array([np.bincount(qq,minlength=Q) for _,qq in draws],dtype=np.float64)
    weights=(a[:,:,None]*b[:,None,:]).reshape(n,R*Q)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        energy,grand,role,formats,raw_scale=_statistics(x)
        weighted_energy=weights@energy.ravel()
        role_quads=np.column_stack([_quadratic(b,role[r]) for r in range(R)])
        role_term=(F*W/Q)*np.einsum('br,br->b',a,role_quads)
        format_term=(W/(R*Q))*_quadratic(weights,formats)
        grand_term=(F*W/(R*Q))*_quadratic(weights,grand)
        numerator=role_term-format_term
        total=weighted_energy-grand_term
        values=numerator/total
        # Projection contractions are bounded by weighted row energy via Jensen
        # and Cauchy--Schwarz, including cancellation inside a near-zero Gram form.
        gram_error=4*_gamma(16*D+F*W*D+32*R*Q+64)*np.abs(weighted_energy)
        # The original raw marginal means and our global centring can differ by
        # forward summation/subtraction error. Bound it in vector norm, keeping
        # large-offset/small-variation grids on the exact original path.
        vector_error=8*_gamma(N+1)*raw_scale
        total_error=gram_error+2*np.sqrt(N*np.maximum(weighted_energy,0))*vector_error+N*vector_error**2
        numerator_error=gram_error+2*np.sqrt(N)*vector_error*(
            np.sqrt(np.maximum(role_term,0))+np.sqrt(np.maximum(format_term,0)))+2*N*vector_error**2
        safe_denominator=total-total_error
        errors=(numerator_error+np.abs(values)*total_error)/safe_denominator
        # Losing over one quarter of float64's significant bits to the common
        # offset can spoil tight agreement even when a decision sign is robust.
        # This is an arithmetic conditioning guard, not an effect-size floor.
        condition=raw_scale/np.sqrt(np.maximum(total,0)/N)
        poorly_conditioned=(~np.isfinite(condition))|(condition>_EPS**(-.25))
    info['conditioning_fallback_count']=int(poorly_conditioned.sum())
    uncertain=(~np.isfinite(values)|~np.isfinite(errors)|(total<=total_error)|
               (np.abs(numerator)<=numerator_error)|(weighted_energy<0)|poorly_conditioned)
    callback=_existing_reference if reference is None else reference
    for i in np.flatnonzero(uncertain):
        values[i]=_reference_value(x,draws[i],callback)
    # A percentile may interpolate positive and negative draws near zero, even
    # when their individual signs are stable. In that rare boundary case use
    # the original statistic for the entire vector instead of changing a verdict.
    remaining=~uncertain
    if np.isfinite(values).all() and remaining.any():
        max_error=float(np.max(errors[remaining]))
        ci=np.percentile(values,[2.5,97.5])
        info['max_estimated_error']=max_error
        if (np.abs(ci)<=max_error).any():
            info['ci_guard_fallback']=True
            for i in np.flatnonzero(remaining):
                values[i]=_reference_value(x,draws[i],callback)
            uncertain[:]=True
    info['fallback_indices']=np.flatnonzero(uncertain).tolist()
    info['fallback_count']=int(uncertain.sum())
    return (values,info) if return_info else values
