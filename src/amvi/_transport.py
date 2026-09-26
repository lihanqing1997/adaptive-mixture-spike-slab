"""Finite fixed-marginal transport helpers used by the reported comparator."""
import numpy as np

def logsumexp(a, axis=None):
    """Real finite-input reduction used in the hot Sinkhorn loop."""
    a=np.asarray(a)
    m=np.max(a,axis=axis,keepdims=True)
    value=m+np.log(np.sum(np.exp(a-m),axis=axis,keepdims=True))
    return np.squeeze(value,axis=axis)


def expand(v, j, p):
    shape = [1] * p
    shape[j] = len(v)
    return np.asarray(v).reshape(shape)


def log_product(vectors):
    return sum(expand(v, j, len(vectors)) for j, v in enumerate(vectors))


def feasible_round(W, targets):
    W=W.copy()
    p=W.ndim
    for j,w in enumerate(targets):
        current=W.sum(axis=tuple(k for k in range(p) if k!=j))
        scale=np.minimum(1,np.divide(w,current,out=np.ones_like(w),where=current>0))
        W*=expand(scale,j,p)
    delta=1-float(W.sum())
    deficits=[w-W.sum(axis=tuple(k for k in range(p) if k!=j)) for j,w in enumerate(targets)]
    if delta < -1e-12 or min(float(d.min()) for d in deficits)<-1e-12:
        raise FloatingPointError('Invalid marginal repair deficits')
    if delta>1e-16:
        # Roundoff only: normalized deficits retain their relative mass.
        positive=[np.maximum(d,0) for d in deficits]
        if any(d.sum()==0 for d in positive):
            raise FloatingPointError('Inconsistent marginal deficits')
        repair=np.ones_like(W)
        for j,d in enumerate(positive): repair*=expand(d/d.sum(),j,p)
        W+=delta*repair
    return W
