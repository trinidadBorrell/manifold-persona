"""Batch only loss/gradient callbacks beneath fresh, unchanged sklearn/SciPy fits.

This helper runs in a dedicated analysis process. A temporary process-local loss
class alias routes independent sklearn fits to one float64 matrix evaluator.
The original scipy L-BFGS-B code keeps each fit's line search, state, tolerances
and stopping rule. Fits never share coefficients or optimizer history.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import queue
import sys
import threading
import time
import warnings

import numpy as np

_FIT_LOCK=threading.Lock()


def loss_class():
    name='_podb_multinomial_batch_loss'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('analysis_batch_loss.py'))
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name].MultinomialBatchLoss


class _Request:
    def __init__(self,coef,y,weight,l2,reference):
        self.coef=np.asarray(coef)
        self.y=np.asarray(y,dtype=np.int64)
        self.weight=None if weight is None else np.asarray(weight)
        self.l2=float(l2);self.reference=reference
        self.ready=threading.Event();self.value=None;self.error=None


class _Broker:
    def __init__(self,X):
        self.X=X;self.requests=queue.Queue();self.failure=None;self.lock=threading.Lock()

    def request(self,coef,X,y,weight,l2,reference):
        if X.shape!=self.X.shape or not np.shares_memory(X,self.X):
            raise ValueError('Batched fits must use the same fixed training transform')
        if np.asarray(y).dtype.kind not in 'iuf' or not np.equal(y,np.asarray(y,dtype=int)).all():
            raise ValueError('Expected original encoded integer labels')
        r=_Request(coef,y,weight,l2,reference)
        with self.lock:
            if self.failure is not None:raise self.failure
            self.requests.put(r)
        if not r.ready.wait(300):
            raise RuntimeError('Batched loss broker did not return a callback')
        if r.error is not None:raise r.error
        return r.value

    def fail(self,error):
        with self.lock:
            self.failure=error
            while True:
                try:r=self.requests.get_nowait()
                except queue.Empty:break
                r.error=error;r.ready.set()


def _validate(X,Y,T,device,wait_seconds):
    X=np.asarray(X);Y=np.asarray(Y);T=np.asarray(T)
    if X.ndim!=2 or T.ndim!=2 or X.dtype!=np.float64 or T.dtype!=np.float64:
        raise ValueError('Batched fits require original float64 train/test matrices')
    if not len(X) or not len(T) or T.shape[1]!=X.shape[1] or not np.isfinite(X).all() or not np.isfinite(T).all():
        raise ValueError('Invalid training/test dimensions or nonfinite values')
    if Y.ndim!=2 or not 1<=len(Y)<=256 or Y.shape[1]!=len(X) or Y.dtype.kind not in 'iu':
        raise ValueError('Expected a bounded batch of integer label vectors')
    classes=np.unique(Y[0])
    if len(classes)<3 or not np.array_equal(classes,np.arange(len(classes))):
        raise ValueError('Expected at least three consecutive encoded role labels')
    if any(not np.array_equal(np.unique(y),classes) for y in Y):
        raise ValueError('All original roles must occur in every training fit')
    if device not in ('cpu','cuda') or not 0<=wait_seconds<=.05:
        raise ValueError('Invalid batch device or collection wait')
    return X,Y,T


def fit_predict_batch(X,Y,T,*,device='cuda',wait_seconds=.002,verify_callbacks=0):
    """Return one prediction row per original label vector, preserving fit order.

    Each fit still calls LogisticRegression.fit with the frozen settings. Only
    the loss callback's shared matrix multiplications are grouped. This function
    must be the only classifier workload in its process while its alias is set.
    """
    from sklearn.linear_model import LogisticRegression,_logistic
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    X,Y,T=_validate(X,Y,T,device,wait_seconds)
    if isinstance(verify_callbacks,bool) or not isinstance(verify_callbacks,int) or verify_callbacks<0:
        raise ValueError('Invalid callback reference-check count')
    started=time.monotonic();evaluator=loss_class()(X,device=device)
    broker=_Broker(X)
    original=_logistic.LinearModelLoss
    info=dict(fits=len(Y),device=device,precision='float64',solver='original sklearn/scipy lbfgs',
              callback_batches=0,callback_evaluations=0,largest_callback_batch=0,
              callback_reference_checks=0,wait_seconds=wait_seconds)

    class RoutedLoss(original):
        def loss_gradient(self,coef,X,y,sample_weight=None,l2_reg_strength=0.0,n_threads=1,raw_prediction=None):
            if not self.base_loss.is_multiclass or not self.fit_intercept or raw_prediction is not None:
                raise ValueError('Unsupported objective in frozen multinomial batch')
            reference=lambda:original.loss_gradient(self,coef,X,y,sample_weight,l2_reg_strength,n_threads)
            return broker.request(coef,X,y,sample_weight,l2_reg_strength,reference)

    def fit(y):
        return LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000).fit(X,y)

    with _FIT_LOCK,threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        _logistic.LinearModelLoss=RoutedLoss
        try:
            with ThreadPoolExecutor(max_workers=len(Y),thread_name_prefix='podb-batch-fit') as pool:
                futures=[pool.submit(fit,y) for y in Y]
                active=[]
                try:
                    while not all(f.done() for f in futures):
                        for f in futures:
                            if f.done() and f.exception() is not None:raise f.exception()
                        try:first=broker.requests.get(timeout=.01)
                        except queue.Empty:continue
                        active=[first];deadline=time.monotonic()+wait_seconds
                        while len(active)<len(Y):
                            remaining=deadline-time.monotonic()
                            if remaining<=0:break
                            try:active.append(broker.requests.get(timeout=remaining))
                            except queue.Empty:break
                        coefs=np.stack([r.coef for r in active])
                        targets=np.stack([r.y for r in active])
                        weights=np.stack([np.ones(len(X)) if r.weight is None else r.weight for r in active])
                        strength=np.array([r.l2 for r in active])
                        loss,grad=evaluator.evaluate(coefs,targets,weights,strength)
                        if loss.shape!=(len(active),) or grad.shape!=coefs.shape or not np.isfinite(loss).all() or not np.isfinite(grad).all():
                            raise ValueError('Batched objective returned invalid loss or gradient')
                        for i,r in enumerate(active):
                            if info['callback_reference_checks']<verify_callbacks:
                                a,b=r.reference()
                                np.testing.assert_allclose(loss[i],a,rtol=1e-12,atol=1e-12)
                                np.testing.assert_allclose(grad[i],b,rtol=5e-10,atol=1e-11)
                                info['callback_reference_checks']+=1
                        info['callback_batches']+=1
                        info['callback_evaluations']+=len(active)
                        info['largest_callback_batch']=max(info['largest_callback_batch'],len(active))
                        for i,r in enumerate(active):
                            r.value=(float(loss[i]),grad[i].copy());r.ready.set()
                        active=[]
                    models=[f.result() for f in futures]
                except BaseException as error:
                    for r in active:r.error=error;r.ready.set()
                    broker.fail(error)
                    for f in futures:f.cancel()
                    raise
        finally:
            _logistic.LinearModelLoss=original
    coefs=np.stack([np.concatenate([m.coef_,m.intercept_[:,None]],axis=1).ravel(order='F') for m in models])
    pred=evaluator.predict(coefs,T)
    info.update(n_iter=[int(m.n_iter_[0]) for m in models],seconds=time.monotonic()-started)
    return pred,info
