"""Float64 batched multinomial loss/gradient callbacks for unchanged sklearn LBFGS.

A batch shares ONE training X. Coefficients follow sklearn's multiclass Fortran
flattening: reshape [B, D+1, C], with feature-major class groups and the intercept
in the last row. There is no optimizer, stopping rule, sampling, or scoring here.
Both dense GEMMs combine all fits' class columns; no per-fit GEMM loop is used.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def _float64(value,name,ndim):
    try:
        array=np.asarray(value)
    except (TypeError,ValueError) as error:
        raise ValueError(f'{name} must be a CPU float64 array') from error
    if array.dtype!=np.dtype(np.float64):
        raise ValueError(f'{name} must have float64 dtype')
    if array.ndim!=ndim:
        raise ValueError(f'{name} must have {ndim} dimensions; got shape {array.shape}')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite values')
    return array


class MultinomialBatchLoss:
    """Hold one finite float64 X and evaluate independent multinomial objectives.

    evaluate(coefs[B,P], targets[B,N], sample_weights[B,N] or None,
    l2_strengths[B]) returns NumPy float64 losses[B], gradients[B,P].
    The objective is weighted-average cross entropy plus lambda/2 ||W||²;
    the intercept is never penalized. CUDA uses deterministic float64 GEMMs,
    with TF32 disabled. CPU is supported for numerical reference checks.
    """

    def __init__(self,X_cpu,device='cuda'):
        X=_float64(X_cpu,'X_cpu',2)
        if X.shape[0]<1:
            raise ValueError('X_cpu needs at least one training sample')
        self.n_samples,self.n_features=X.shape
        self.device=torch.device(device)
        if self.device.type not in ('cpu','cuda'):
            raise ValueError('device must be cpu or cuda for float64 evaluation')
        if self.device.type=='cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA was requested but is unavailable; no device substitution is made')
            config=os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            if config not in (':4096:8',':16:8'):
                raise ValueError('CUDA determinism needs CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.use_deterministic_algorithms(True)
        # Own the training matrix: future caller mutations must not alter a fit.
        self._X=self._tensor(X)

    def _tensor(self,array,dtype=torch.float64):
        # One host copy also supports read-only memmaps and negative-stride inputs.
        owned=np.array(array,copy=True,order='C')
        return torch.from_numpy(owned).to(device=self.device,dtype=dtype)

    def _parameters(self,coefs):
        coefs=_float64(coefs,'coefs',2)
        B,P=coefs.shape
        if B<1 or P%(self.n_features+1):
            raise ValueError('coefs shape must be [B,(D+1)*C] with a nonempty batch')
        C=P//(self.n_features+1)
        if C<2:
            raise ValueError('coefs must encode at least two multinomial classes')
        return self._tensor(coefs).reshape(B,self.n_features+1,C),B,C

    @staticmethod
    def _logits(X,params):
        B,_,C=params.shape
        # [N,D] @ [D,B*C], rather than B separate [N,D] @ [D,C] calls.
        flat_weights=params[:,:-1].permute(1,0,2).reshape(X.shape[1],B*C)
        logits=X@flat_weights+params[:,-1].reshape(B*C)
        return logits.reshape(X.shape[0],B,C).permute(1,0,2)

    def evaluate(self,coefs,targets,sample_weights,l2_strengths):
        with torch.no_grad():
            params,B,C=self._parameters(coefs)
            targets=np.asarray(targets)
            if targets.shape!=(B,self.n_samples) or not np.issubdtype(targets.dtype,np.integer):
                raise ValueError('targets must be integer labels with shape [B,N]')
            if targets.min()<0 or targets.max()>=C:
                raise ValueError('targets must be class indices in [0,C)')
            y=self._tensor(targets.astype(np.int64,copy=False),dtype=torch.int64)
            l2=_float64(l2_strengths,'l2_strengths',1)
            if l2.shape!=(B,):
                raise ValueError('l2_strengths must have shape [B]')
            strength=self._tensor(l2)
            if sample_weights is None:
                sw=torch.ones((B,self.n_samples),dtype=torch.float64,device=self.device)
                sums=np.full(B,self.n_samples,dtype=np.float64)
            else:
                weights=_float64(sample_weights,'sample_weights',2)
                if weights.shape!=(B,self.n_samples):
                    raise ValueError('sample_weights must have shape [B,N]')
                with np.errstate(over='ignore',invalid='ignore'):
                    sums=weights.sum(axis=1)
                if not np.isfinite(sums).all() or (sums<=0).any():
                    raise ValueError('sample_weights must have positive finite weight sums')
                sw=self._tensor(weights)
            denominator=self._tensor(sums)
            logits=self._logits(self._X,params)
            maximum=logits.max(dim=2).values
            exponentials=torch.exp(logits-maximum[:,:,None])
            sum_exp=exponentials.sum(dim=2)
            # Match sklearn's log(sum(exp(raw-max))) + max - raw_target order.
            point_loss=torch.log(sum_exp)+maximum-logits.gather(2,y[:,:,None]).squeeze(2)
            losses=(point_loss*sw).sum(dim=1)/denominator
            weights=params[:,:-1]
            losses=losses+.5*strength*(weights*weights).sum(dim=(1,2))
            probabilities=exponentials/sum_exp[:,:,None]
            one_hot=torch.nn.functional.one_hot(y,num_classes=C).to(dtype=torch.float64)
            point_gradient=(probabilities-one_hot)*sw[:,:,None]
            point_gradient=point_gradient/denominator[:,None,None]
            # [D,N] @ [N,B*C] fuses all fits' weight gradients into one GEMM.
            flat_gradient=point_gradient.permute(1,0,2).reshape(self.n_samples,B*C)
            weight_gradient=(self._X.T@flat_gradient).reshape(self.n_features,B,C).permute(1,0,2)
            weight_gradient=weight_gradient+strength[:,None,None]*weights
            intercept_gradient=point_gradient.sum(dim=1)
            gradient=torch.cat((weight_gradient,intercept_gradient[:,None,:]),dim=1).reshape(B,-1)
            loss_cpu=losses.cpu().numpy()
            gradient_cpu=gradient.cpu().numpy()
            if not np.isfinite(loss_cpu).all() or not np.isfinite(gradient_cpu).all():
                raise FloatingPointError('float64 loss/gradient overflowed; no approximate result is returned')
            return loss_cpu,gradient_cpu

    def predict(self,coefs,Xte_cpu):
        with torch.no_grad():
            params,_,_=self._parameters(coefs)
            Xte=_float64(Xte_cpu,'Xte_cpu',2)
            if Xte.shape[1]!=self.n_features:
                raise ValueError('Xte_cpu must have the same number of features as X_cpu')
            logits=self._logits(self._tensor(Xte),params)
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError('float64 prediction logits overflowed')
            return logits.argmax(dim=2).cpu().numpy()
