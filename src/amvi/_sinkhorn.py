"""Fixed-marginal Sinkhorn correction and feasibility certificate."""
from time import perf_counter
import numpy as np
from scipy.special import xlogy
from ._transport import expand, logsumexp, log_product, feasible_round
SOFT_SECONDS=60.

def certificate(C, logw, targets, u, lam, shift):
    T = 1 + lam
    mass = float(np.exp(logsumexp(logw)))
    dual = shift + T * (sum(float(v @ m) for v, m in zip(u, targets)) + 1 - mass)
    W = feasible_round(np.exp(logw), targets)
    logM = log_product([np.log(m) for m in targets])
    tc = float(np.sum(xlogy(W, W) - W * logM))
    loss = float(np.sum(W * C))
    primal = loss + T * tc
    axes = [tuple(k for k in range(C.ndim) if k != j) for j in range(C.ndim)]
    repaired = max(float(np.abs(W.sum(axis=ax) - m).sum()) for ax, m in zip(axes, targets))
    product_loss = C.copy()
    for j, m in enumerate(targets):
        product_loss *= expand(m, j, C.ndim)
    baseline = float(product_loss.sum())
    return dict(primal=primal, dual=dual, gap=primal-dual,
                relative_gap=(primal-dual)/max(1., abs(primal)),
                total_correlation=tc, expected_loss=loss,
                repaired_marginal_l1=repaired, product_objective=baseline,
                improvement_over_product=baseline-primal)


def solve(C, targets, lam, seconds=SOFT_SECONDS, tol=1e-7, gap_tol=1e-7,
          maxiter=10_000, return_logw=False):
    if lam < 0 or any(np.any(m <= 0) or abs(m.sum()-1) > 1e-10 for m in targets):
        raise ValueError('Nonnegative penalty and positive normalized marginals required')
    start = perf_counter()
    deadline = start + seconds
    T = 1 + lam
    shift = float(C.min())
    lm = [np.log(m) for m in targets]
    u = [np.zeros_like(m) for m in targets]
    logw = -(C-shift)/T
    for j, m in enumerate(lm):
        logw += expand(m, j, C.ndim)
    axes = [tuple(k for k in range(C.ndim) if k != j) for j in range(C.ndim)]
    status, residual, cert, sweeps, updates = 'iteration_limit', None, None, 0, 0
    for sweep in range(maxiter):
        for j, ax in enumerate(axes):
            if perf_counter() >= deadline:
                status = 'time_limit'
                break
            delta = lm[j] - logsumexp(logw, axis=ax)
            u[j] += delta
            logw += expand(delta, j, C.ndim)
            updates += 1
        if status == 'time_limit':
            break
        sweeps = sweep + 1
        residual = max(float(np.abs(np.exp(logsumexp(logw, axis=ax))-m).sum())
                       for ax, m in zip(axes, targets))
        if residual <= tol:
            cert = certificate(C, logw, targets, u, lam, shift)
            if cert['relative_gap'] < -1e-9:
                raise ArithmeticError('Dual exceeds feasible primal')
            if cert['relative_gap'] <= gap_tol:
                status = 'converged'
                break
        if perf_counter() >= deadline:
            status = 'time_limit'
            break
    result = dict(status=status, sinkhorn_seconds=perf_counter()-start,
                  sweeps=sweeps, coordinate_updates=updates, marginal_l1=residual,
                  certificate=cert)
    if return_logw:
        result['logw'] = logw
    return result
