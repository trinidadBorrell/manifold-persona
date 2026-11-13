"""Independent float64 L-BFGS fits with batched device-resident vector operations.

The frozen ridge-softmax objective, zero initialization, history size, maxiter,
gradient/function tolerances and maximum line-search evaluations are retained.
Each fit has its own More-Thuente line search and curvature history. Scalar line
searches use SciPy DCSRCH; coefficient and gradient vectors remain on the device.
This is a candidate execution backend and must pass paired real-data checks.
"""
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np
import torch


class BatchConvergenceError(ValueError):
    pass


def compact_inverse_product(g,S,Z,cursor,counts,gamma,K):
    """Same limited-memory inverse BFGS product, using its compact matrix form."""
    if K==0:return gamma[:,None]*g
    B=len(g);ids=torch.arange(B,device=g.device)[:,None]
    age=torch.arange(K,device=g.device)[None,:]
    slots=(cursor[:,None]-K+age)%len(S)
    valid=age>=(K-counts[:,None])
    s=S[slots,ids]*valid[:,:,None]
    y=Z[slots,ids]*valid[:,:,None]
    sy=torch.bmm(s,y.transpose(1,2))
    R=torch.triu(sy)+torch.diag_embed((~valid).to(dtype=g.dtype))
    sg=torch.bmm(s,g[:,:,None])
    yg=torch.bmm(y,g[:,:,None])
    a=torch.linalg.solve_triangular(R,sg,upper=True)
    yy=torch.bmm(y,y.transpose(1,2))
    rhs=torch.diagonal(R,dim1=1,dim2=2)[:,:,None]*a-gamma[:,None,None]*(yg-torch.bmm(yy,a))
    b=torch.linalg.solve_triangular(R.transpose(1,2),rhs,upper=False)
    return gamma[:,None]*g-gamma[:,None]*torch.bmm(y.transpose(1,2),a).squeeze(2)+torch.bmm(s.transpose(1,2),b).squeeze(2)


def _device_loss_class():
    name='_podb_device_multinomial_loss'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('analysis_batch_device.py'))
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name].DeviceMultinomialBatchLoss


def fit_predict_batch(X,Y,T,*,device='cuda',max_iter=2000,return_parameters=False,
                      wait_seconds=None,verify_callbacks=0,history_mode='loop'):
    from scipy.optimize._dcsrch import DCSRCH
    from sklearn.utils.class_weight import compute_sample_weight
    started=time.perf_counter();X=np.asarray(X);Y=np.asarray(Y);T=np.asarray(T)
    if X.ndim!=2 or T.ndim!=2 or X.dtype!=np.float64 or T.dtype!=np.float64:
        raise ValueError('Original float64 training/test matrices are required')
    if not len(X) or T.shape[1]!=X.shape[1] or not np.isfinite(X).all() or not np.isfinite(T).all():
        raise ValueError('Invalid classifier feature matrices')
    if Y.ndim!=2 or not 1<=len(Y)<=512 or Y.shape[1]!=len(X) or Y.dtype.kind not in 'iu':
        raise ValueError('Expected bounded independent integer label vectors')
    C=len(np.unique(Y[0]));B=len(Y);D=X.shape[1];P=(D+1)*C
    if C<3 or any(not np.array_equal(np.unique(y),np.arange(C)) for y in Y):
        raise ValueError('Every fit needs all consecutive original class indices')
    if isinstance(max_iter,bool) or not isinstance(max_iter,int) or max_iter<1:
        raise ValueError('max_iter must be positive')
    if history_mode not in ('loop','compact'):raise ValueError('Unknown inverse-history execution')
    weight=np.stack([compute_sample_weight('balanced',y) for y in Y])
    strength=1.0/weight.sum(axis=1)  # Original C=1.0 and sklearn sample-weight sum.
    objective=_device_loss_class()(X,Y,weight,strength,device=device)
    dev=objective.device
    x=torch.zeros((B,P),dtype=torch.float64,device=dev)
    f,g=objective.evaluate(x)
    m=10;gtol=1e-4;ftol=64*np.finfo(float).eps;eps=np.finfo(float).eps
    S=torch.zeros((m,B,P),dtype=torch.float64,device=dev);Z=torch.zeros_like(S)
    rho=torch.zeros((m,B),dtype=torch.float64,device=dev)
    cursor=torch.zeros(B,dtype=torch.long,device=dev);counts=torch.zeros_like(cursor)
    ids=torch.arange(B,device=dev);gamma=torch.ones(B,dtype=torch.float64,device=dev)
    iterations=np.zeros(B,dtype=int);evaluations=np.ones(B,dtype=int)
    done=np.zeros(B,dtype=bool);reasons=['']*B;matrix_batches=1;restarts=0
    for outer in range(max_iter):
        norm=g.abs().amax(dim=1).cpu().numpy()
        for i in np.flatnonzero((~done)&(norm<=gtol)):done[i]=True;reasons[i]='gradient'
        active=~done
        if not active.any():break
        # Independent two-loop recursions, vectorized over fits. Ring slots are
        # per fit, including when a curvature update is skipped.
        if history_mode=='compact':
            r=compact_inverse_product(g,S,Z,cursor,counts,gamma,min(outer,m))
        else:
            q=g.clone();alphas=[];entries=[]
            for age in range(1,min(outer,m)+1):
                slot=(cursor-age)%m;valid=counts>=age
                sh=S[slot,ids];yh=Z[slot,ids];rh=rho[slot,ids]*valid
                alpha=rh*(sh*q).sum(dim=1)
                q=q-alpha[:,None]*yh
                alphas.append(alpha);entries.append((slot,valid))
            r=gamma[:,None]*q
            for alpha,(slot,valid) in zip(reversed(alphas),reversed(entries)):
                sh=S[slot,ids];yh=Z[slot,ids];rh=rho[slot,ids]*valid
                beta=rh*(yh*r).sum(dim=1)
                r=r+sh*(alpha-beta)[:,None]
        direction=-r
        direction[torch.as_tensor(done,device=dev)]=0
        initial_derivative=(g*direction).sum(dim=1).cpu().numpy()
        bad=active&(initial_derivative>=0)
        if bad.any():
            bi=torch.as_tensor(np.flatnonzero(bad),device=dev)
            direction[bi]=-g[bi];counts[bi]=0;cursor[bi]=0;rho[:,bi]=0;gamma[bi]=1
            initial_derivative=(g*direction).sum(dim=1).cpu().numpy();restarts+=int(bad.sum())
        if not np.isfinite(initial_derivative).all():raise BatchConvergenceError('Nonfinite descent direction')
        previous=f.cpu().numpy();steps=np.zeros(B);tasks={};search={}
        norms=direction.norm(dim=1).cpu().numpy()
        for i in np.flatnonzero(active):
            # Same initial unconstrained step and More-Thuente constants used
            # by L-BFGS-B; later iterations start at step one.
            step=min(1.0/norms[i],1e10) if iterations[i]==0 else 1.0
            ls=DCSRCH(None,None,ftol=1e-3,gtol=.9,xtol=.1,stpmin=0.,stpmax=1e10)
            step,_,_,task=ls._iterate(step,float(previous[i]),float(initial_derivative[i]),b'START')
            if task[:2]!=b'FG':raise BatchConvergenceError('Invalid initial line search: '+task.decode())
            steps[i]=step;search[i]=ls;tasks[i]=task
        pending=active.copy()
        for line_iteration in range(50):
            trial=x+torch.as_tensor(steps,dtype=torch.float64,device=dev)[:,None]*direction
            trial_f,trial_g=objective.evaluate(trial);matrix_batches+=1
            values=trial_f.cpu().numpy();derivative=(trial_g*direction).sum(dim=1).cpu().numpy()
            if not np.isfinite(values).all() or not np.isfinite(derivative).all():
                raise BatchConvergenceError('Nonfinite line-search evaluation')
            for i in np.flatnonzero(pending):
                evaluations[i]+=1
                step,_,_,task=search[i]._iterate(float(steps[i]),float(values[i]),float(derivative[i]),tasks[i])
                if task[:2]==b'FG':steps[i]=step;tasks[i]=task
                elif task[:4]==b'CONV':pending[i]=False
                else:raise BatchConvergenceError('Line search did not converge: '+task.decode())
            if not pending.any():break
        else:raise BatchConvergenceError('Original maximum 50 line-search evaluations exceeded')
        s=trial-x;y=trial_g-g
        curvature=(s*y).sum(dim=1)
        bound=eps*(-g*s).sum(dim=1)
        yy=(y*y).sum(dim=1)
        good=(curvature>bound)&(yy>0)&torch.as_tensor(active,device=dev)
        gi=ids[good];slot=cursor[good]
        S[slot,gi]=s[good];Z[slot,gi]=y[good];rho[slot,gi]=1/curvature[good]
        gamma[good]=curvature[good]/yy[good]
        cursor[good]=(cursor[good]+1)%m;counts[good]=torch.clamp(counts[good]+1,max=m)
        x,f,g=trial,trial_f,trial_g
        iterations[active]+=1
        relative=(previous-values)<=ftol*np.maximum(np.maximum(np.abs(previous),np.abs(values)),1)
        new_norm=g.abs().amax(dim=1).cpu().numpy()
        for i in np.flatnonzero(active&(new_norm<=gtol)):done[i]=True;reasons[i]='gradient'
        for i in np.flatnonzero(active&(~done)&relative):done[i]=True;reasons[i]='relative_function'
        if (evaluations>15000).any():raise BatchConvergenceError('Original maximum function evaluations exceeded')
        if done.all():break
    if not done.all():raise BatchConvergenceError('One or more independent fits reached max_iter')
    pred=objective.predict(x,T).cpu().numpy()
    info=dict(fits=B,device=str(dev),precision='float64',solver='independent batched L-BFGS, SciPy More-Thuente',
              gtol=gtol,ftol=ftol,max_iter=max_iter,max_line_search=50,history_size=m,C=1.0,
              all_converged=True,n_iter=iterations.tolist(),n_function_evaluations=evaluations.tolist(),
              convergence_reason=reasons,max_abs_gradient=g.abs().amax(dim=1).cpu().numpy().tolist(),
              callback_batches=matrix_batches,largest_callback_batch=B,history_restarts=restarts,
              history_mode=history_mode,seconds=time.perf_counter()-started)
    if return_parameters:info['parameters']=x.cpu().numpy()
    return pred,info
