"""Posterior moments, interval coverage, selection and prediction metrics."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

def summaries(weights, a, mu, var, beta, test, reference=None):
    weights, a, mu, var = [np.asarray(v, dtype=float) for v in (weights, a, mu, var)]
    cm = a*mu
    mean = weights @ cm
    within = weights @ (a*(var+mu**2)-cm**2)
    covariance = np.diag(within)+(cm.T*weights)@cm-np.outer(mean, mean)
    pip = weights @ a
    intervals = np.zeros((len(beta), 2))
    for j in range(len(beta)):
        active_weight = weights*a[:, j]
        atom = 1.-pip[j]
        below = active_weight @ norm.cdf(-mu[:, j]/np.sqrt(var[:, j]))
        low = min(-1., np.min(mu[:, j]-14*np.sqrt(var[:, j])))
        high = max(1., np.max(mu[:, j]+14*np.sqrt(var[:, j])))
        for t, prob in enumerate((.025, .975)):
            if below <= prob <= below+atom:
                intervals[j, t] = 0.
            else:
                def cdf(value):
                    return (active_weight @ norm.cdf((value-mu[:, j])/np.sqrt(var[:, j]))
                            + (atom if value >= 0 else 0.) - prob)
                intervals[j, t] = brentq(cdf, low, high, xtol=1e-11)
    active = beta != 0
    selected = pip >= .5
    covered = (intervals[:, 0] <= beta) & (beta <= intervals[:, 1])
    answer = dict(pip=pip, mean=mean, covariance=covariance, intervals=intervals,
                  prediction_mse=float(np.mean((test@(mean-beta))**2)),
                  true_positive_rate=float(selected[active].mean()),
                  false_discovery_proportion=float(np.sum(selected & ~active)/max(1, selected.sum())),
                  active_coverage=float(covered[active].mean()),
                  all_coverage=float(covered.mean()), expected_support_size=float(pip.sum()))
    if reference is not None:
        answer.update(pip_mae=float(np.abs(pip-reference['pip']).mean()),
                      covariance_error=float(np.linalg.norm(covariance-reference['covariance'])),
                      posterior_mean_error=float(np.linalg.norm(mean-reference['mean'])))
    return answer
