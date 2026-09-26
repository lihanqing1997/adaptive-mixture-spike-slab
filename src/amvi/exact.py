"""Conjugate enumeration reference for small spike-and-slab models."""
from dataclasses import dataclass
from itertools import product
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def exact_reference(model):
    """Enumerate supports at p <= 12 using the regression model's parameters."""
    if model.p > 12:
        return None
    return exact_posterior(Model(model.X, model.y, sigma=model.sigma, tau=model.tau, omega=model.omega))

def logsumexp(a, axis=None):
    """Real finite-input reduction used in the hot Sinkhorn loop."""
    a=np.asarray(a)
    m=np.max(a,axis=axis,keepdims=True)
    value=m+np.log(np.sum(np.exp(a-m),axis=axis,keepdims=True))
    return np.squeeze(value,axis=axis)


def softmax(a):
    return np.exp(a - logsumexp(a))


@dataclass
class Model:
    X: np.ndarray
    y: np.ndarray
    sigma: float = 1.
    tau: float = 1.
    omega: float = .2

    def __post_init__(self):
        self.X = np.asarray(self.X, float)
        self.y = np.asarray(self.y, float)
        self.n, self.p = self.X.shape
        assert self.y.shape == (self.n,)
        assert self.sigma > 0 and self.tau > 0 and 0 < self.omega < 1
        self.G = self.X.T @ self.X / self.sigma**2
        self.h = self.X.T @ self.y / self.sigma**2
        self.yy = float(self.y @ self.y / self.sigma**2)


def exact_posterior(model):
    start = perf_counter()
    p = model.p
    masks = np.array(list(product([0, 1], repeat=p)), int)
    means, covs, logmass = [], [], []
    for mask in masks:
        idx = np.flatnonzero(mask)
        mean, cov = np.zeros(p), np.zeros((p, p))
        logz = -.5*model.yy
        if len(idx):
            A = model.G[np.ix_(idx, idx)] + np.eye(len(idx))/model.tau**2
            cf = cho_factor(A, lower=True)
            V = cho_solve(cf, np.eye(len(idx)))
            mu = cho_solve(cf, model.h[idx])
            logdet = 2*np.log(np.diag(cf[0])).sum()
            logz += -.5*(logdet + 2*len(idx)*np.log(model.tau)) + .5*model.h[idx]@mu
            mean[idx] = mu
            cov[np.ix_(idx, idx)] = V
        means.append(mean); covs.append(cov)
        s = mask.sum()
        logmass.append(logz + s*np.log(model.omega)+(p-s)*np.log1p(-model.omega))
    logZ = float(logsumexp(logmass))
    probs = softmax(np.asarray(logmass))
    means, covs = np.asarray(means), np.asarray(covs)
    mean = probs @ means
    second = np.einsum('s,sij->ij', probs, covs + means[:, :, None]*means[:, None, :])
    return dict(logZ=logZ, support=probs, pip=probs@masks, mean=mean,
                covariance=second-np.outer(mean, mean), masks=masks,
                conditional_means=means, conditional_covariances=covs,
                seconds=perf_counter()-start)
