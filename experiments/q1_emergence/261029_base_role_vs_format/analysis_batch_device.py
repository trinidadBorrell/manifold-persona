"""Device-resident float64 multinomial callbacks; no optimizer or host gradient copies.

Constants are owned and transferred once. Class count is inferred from the first
coefficient matrix, then its one-hot labels are cached. Evaluation validates tensor
metadata without host synchronization; the optimizer handles nonfinite iterates.
The objective and feature-major/Fortran coefficient layout match the frozen
analysis_batch_loss kernel. Only the placement and cached weight normalization differ.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


def _verified_kernel():
    name='_podb_verified_multinomial_batch_loss'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('analysis_batch_loss.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


_REF=_verified_kernel()


class DeviceMultinomialBatchLoss(_REF.MultinomialBatchLoss):
    """Shared X, fixed independent targets/weights/penalties, device-only evaluations.

    evaluate(coefs_tensor[B,(D+1)*C]) returns device float64 losses[B], gradients[B,P].
    Constants are copied at construction; one-hot labels are allocated on the first
    evaluation because the API has no separate class-count argument. No coefficient,
    gradient, or scalar validation value is copied to CPU in evaluate().
    """

    def __init__(self,X_cpu,targets,sample_weights,l2_strengths,device='cuda'):
        super().__init__(X_cpu,device=device)
        self.device=self._X.device  # Canonical CUDA index for metadata comparisons.
        labels=np.asarray(targets)
        if labels.ndim!=2 or labels.shape[0]<1 or labels.shape[1]!=self.n_samples or not np.issubdtype(labels.dtype,np.integer):
            raise ValueError('targets must be integer labels with shape [B,N]')
        if labels.min()<0 or int(labels.max())>np.iinfo(np.int64).max:
            raise ValueError('targets must be nonnegative int64-compatible class indices')
        self.n_fits=labels.shape[0]
        self._max_target=int(labels.max())
        self._targets=self._tensor(labels.astype(np.int64,copy=False),dtype=torch.int64)
        strengths=_REF._float64(l2_strengths,'l2_strengths',1)
        if strengths.shape!=(self.n_fits,):
            raise ValueError('l2_strengths must have shape [B]')
        self._l2=self._tensor(strengths)
        if sample_weights is None:
            weights=np.ones((self.n_fits,self.n_samples),dtype=np.float64)
        else:
            weights=_REF._float64(sample_weights,'sample_weights',2)
            if weights.shape!=(self.n_fits,self.n_samples):
                raise ValueError('sample_weights must have shape [B,N]')
        with np.errstate(over='ignore',invalid='ignore'):
            sums=weights.sum(axis=1)
        if not np.isfinite(sums).all() or (sums<=0).any():
            raise ValueError('sample_weights need positive finite weight sums')
        self._normalized_weights=self._tensor(weights/sums[:,None])
        self._one_hot=None
        self._n_classes=None

    def _device_parameters(self,coefs_tensor):
        if not isinstance(coefs_tensor,torch.Tensor) or coefs_tensor.dtype!=torch.float64:
            raise ValueError('coefs_tensor must be a float64 torch.Tensor')
        if coefs_tensor.device!=self.device:
            raise ValueError('coefs_tensor must stay on the objective device')
        if coefs_tensor.ndim!=2 or coefs_tensor.shape[0]!=self.n_fits or coefs_tensor.shape[1]%(self.n_features+1):
            raise ValueError('coefs_tensor must have shape [B,(D+1)*C] for the fixed batch')
        C=coefs_tensor.shape[1]//(self.n_features+1)
        if C<2 or self._max_target>=C:
            raise ValueError('coefficient class count must cover every target and be at least two')
        if self._n_classes is None:
            self._one_hot=torch.nn.functional.one_hot(self._targets,num_classes=C).to(dtype=torch.float64)
            self._n_classes=C
        elif self._n_classes!=C:
            raise ValueError('the objective class count cannot change between evaluations')
        return coefs_tensor.reshape(self.n_fits,self.n_features+1,C),C

    def evaluate(self,coefs_tensor):
        with torch.no_grad():
            params,C=self._device_parameters(coefs_tensor)
            logits=self._logits(self._X,params)
            maximum=logits.max(dim=2).values
            exponentials=torch.exp(logits-maximum[:,:,None])
            sums=exponentials.sum(dim=2)
            point_loss=torch.log(sums)+maximum-logits.gather(2,self._targets[:,:,None]).squeeze(2)
            weights=params[:,:-1]
            losses=(point_loss*self._normalized_weights).sum(dim=1)+.5*self._l2*(weights*weights).sum(dim=(1,2))
            point_gradient=(exponentials/sums[:,:,None]-self._one_hot)*self._normalized_weights[:,:,None]
            flat_gradient=point_gradient.permute(1,0,2).reshape(self.n_samples,self.n_fits*C)
            gradient=(self._X.T@flat_gradient).reshape(self.n_features,self.n_fits,C).permute(1,0,2)
            gradient=gradient+self._l2[:,None,None]*weights
            intercept=point_gradient.sum(dim=1)
            return losses,torch.cat((gradient,intercept[:,None,:]),dim=1).reshape(self.n_fits,-1)

    def predict(self,coefs_tensor,Xte_cpu):
        with torch.no_grad():
            params,_=self._device_parameters(coefs_tensor)
            Xte=_REF._float64(Xte_cpu,'Xte_cpu',2)
            if Xte.shape[1]!=self.n_features:
                raise ValueError('Xte_cpu must match the training feature count')
            return self._logits(self._tensor(Xte),params).argmax(dim=2)
