"""Shared spike-and-slab laws and analytic costs used by adaptive mixture VI."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.special import expit, logit, logsumexp, xlogy
from scipy.stats import norm, qmc

@dataclass
class RegressionModel:
    X: np.ndarray
    y: np.ndarray
    sigma: float = 1.
    tau: float = 1.
    omega: float = .1

    def __post_init__(self):
        self.X, self.y = np.asarray(self.X, float), np.asarray(self.y, float)
        self.n, self.p = self.X.shape
        if self.y.shape != (self.n,) or min(self.sigma, self.tau) <= 0 or not 0 < self.omega < 1:
            raise ValueError('Invalid regression inputs')
        self.diag = np.sum(self.X**2, axis=0)/self.sigma**2


@dataclass
class Mixture:
    weights: np.ndarray
    a: np.ndarray
    mu: np.ndarray
    var: np.ndarray

    def __post_init__(self):
        self.weights, self.a, self.mu, self.var = [np.asarray(x, float) for x in
                                                 (self.weights, self.a, self.mu, self.var)]
        if not self.a.shape == self.mu.shape == self.var.shape or self.a.ndim != 2:
            raise ValueError('Component arrays must be K by p')
        self.k, self.p = self.a.shape
        if (self.weights.shape != (self.k,) or np.any(self.weights <= 0)
                or not np.isclose(self.weights.sum(), 1.) or np.any(self.a <= 0)
                or np.any(self.a >= 1) or np.any(self.var <= 0)
                or not all(np.all(np.isfinite(x)) for x in (self.weights, self.a, self.mu, self.var))):
            raise ValueError('Invalid mixture parameters')

    def pack(self):
        return np.r_[np.log(self.weights), logit(self.a).ravel(), self.mu.ravel(), np.log(self.var).ravel()]

    @classmethod
    def unpack(cls, theta, k, p):
        w = np.exp(theta[:k]-logsumexp(theta[:k]))
        b = k+k*p
        return cls(w, expit(theta[k:b]).reshape(k,p), theta[b:b+k*p].reshape(k,p),
                   np.exp(theta[b+k*p:]).reshape(k,p))

    def logdensity(self, gamma, beta):
        """Density w.r.t. product of delta_0 + Lebesgue, not pure Lebesgue."""
        zero = np.log1p(-self.a)
        active = np.log(self.a)-zero-.5*np.log(2*np.pi*self.var)-.5*self.mu**2/self.var
        return zero.sum(1)[None,:] + gamma @ active.T + beta @ (self.mu/self.var).T - .5*(beta**2) @ (1/self.var).T

    def summary(self, covariance=False):
        cm = self.a*self.mu
        mean, pip = self.weights @ cm, self.weights @ self.a
        second = self.weights @ (self.a*(self.var+self.mu**2))
        result = dict(pip=pip, mean=mean, variance=second-mean**2)
        if covariance:
            within = self.a*(self.var+self.mu**2)-cm**2
            result['covariance'] = np.diag(self.weights @ within)+(cm.T*self.weights) @ cm-np.outer(mean,mean)
        intervals = np.empty((self.p,2))
        for j in range(self.p):
            masses = self.weights*self.a[:,j]
            below = masses @ norm.cdf(-self.mu[:,j]/np.sqrt(self.var[:,j]))
            for t, level in enumerate((.025,.975)):
                if below <= level <= below+1-pip[j]:
                    intervals[j,t] = 0.
                else:
                    def fun(x):
                        return masses @ norm.cdf((x-self.mu[:,j])/np.sqrt(self.var[:,j]))+(1-pip[j] if x >= 0 else 0.)-level
                    low = min(-1., np.min(self.mu[:,j]-12*np.sqrt(self.var[:,j])))
                    high = max(1., np.max(self.mu[:,j]+12*np.sqrt(self.var[:,j])))
                    intervals[j,t] = brentq(fun,low,high,xtol=1e-10)
        result['intervals'] = intervals
        return result


def component_costs(model, mix, gradient=False):
    m = mix.a*mix.mu
    variance = mix.a*(mix.var+mix.mu**2)-m**2
    residual = m @ model.X.T-model.y
    loss = .5*np.sum(residual**2,axis=1)/model.sigma**2+.5*variance @ model.diag
    normal = .5*((mix.var+mix.mu**2)/model.tau**2-1-np.log(mix.var/model.tau**2))
    bern = xlogy(mix.a,mix.a/model.omega)+xlogy(1-mix.a,(1-mix.a)/(1-model.omega))
    costs = loss+np.sum(bern+mix.a*normal,axis=1)
    if not gradient:
        return costs
    gm = residual @ model.X/model.sigma**2
    ga = gm*mix.mu+.5*model.diag*(mix.var+(1-2*mix.a)*mix.mu**2)+logit(mix.a)-logit(model.omega)+normal
    gu = gm*mix.a+model.diag*mix.a*(1-mix.a)*mix.mu+mix.a*mix.mu/model.tau**2
    gv = .5*mix.a*(model.diag+1/model.tau**2-1/mix.var)
    return costs, (ga*mix.a*(1-mix.a),gu,gv*mix.var)


class InformationCache:
    """Differentiable importance integral, including source-measure derivatives.

    Source samples/nodes remain fixed during an optimizer call. Each source has
    its own probability quadrature rule. Normalization is NOT estimated or
    self-normalized; this keeps the represented integral and its derivative
    explicit. Importance mass/ESS deviations are reported separately.
    """
    def __init__(self, reference, gamma, beta, source, measure):
        self.gamma, self.beta = gamma, beta
        self.source, self.measure = source, measure
        self.refsource = reference.logdensity(gamma,beta)[np.arange(len(source)),source]
        self.k, self.p = reference.k, reference.p

    def evaluate(self, mix, gradient=False):
        ld = mix.logdensity(self.gamma,self.beta)
        terms = ld+np.log(mix.weights)
        lm = logsumexp(terms,axis=1)
        rows = np.arange(len(self.source))
        ratio = ld[rows,self.source]-lm
        iw = np.exp(ld[rows,self.source]-self.refsource)
        mass = mix.weights[self.source]*self.measure*iw
        value = float(mass @ ratio)
        imass, ess = [], []
        for k in range(mix.k):
            sel = self.source == k
            z = self.measure[sel]*iw[sel]
            imass.append(float(z.sum()))
            # Relative ESS is meaningful for equal-weight random rules only;
            # the marginal quadrature cache does not use this for acceptance.
            ess.append(float(z.sum()**2/(len(z)*np.sum(z*z))))
        diagnostics = dict(importance_mass=imass, relative_ess=ess)
        if not gradient:
            return value, diagnostics
        responsibilities = np.exp(terms-lm[:,None])
        coef = -mass[:,None]*responsibilities
        coef[rows,self.source] += mass*(ratio+1)
        counts = self.gamma.T @ coef
        first = self.beta.T @ coef
        second = (self.beta**2).T @ coef
        sums = coef.sum(0)
        ga = counts.T-mix.a*sums[:,None]
        gu = (first.T-mix.mu*counts.T)/mix.var
        gv = .5*((second.T-2*mix.mu*first.T+mix.mu**2*counts.T)/mix.var-counts.T)
        gw = -np.sum(mass[:,None]*(responsibilities-mix.weights),axis=0)
        gw += np.bincount(self.source,weights=mass*ratio,minlength=mix.k)-mix.weights*value
        return value,(gw,ga,gu,gv),diagnostics


def make_joint_cache(mix, power=9, seed=0):
    gamma,beta,source = [],[],[]
    for k in range(mix.k):
        u = qmc.Sobol(2*mix.p,scramble=True,seed=seed+104729*k).random_base2(power)
        g = (u[:,:mix.p] < mix.a[k]).astype(float)
        z = norm.ppf(np.clip(u[:,mix.p:],1e-14,1-1e-14))
        gamma.append(g)
        beta.append(g*(mix.mu[k]+np.sqrt(mix.var[k])*z))
        source.append(np.full(len(g),k))
    return InformationCache(mix,np.vstack(gamma),np.vstack(beta),np.concatenate(source),
                            np.full(mix.k*2**power,1/2**power))


def meanfield(model,maxiter=1500,tol=1e-8,start=0,tilt=None):
    began = perf_counter()
    a = np.full(model.p,model.omega)
    var = 1/(model.diag+model.tau**-2)
    mu = np.zeros(model.p) if start == 0 else var*(model.X.T @ model.y/model.sigma**2)
    m = a*mu
    residual = model.y-model.X @ m
    tilt = np.zeros(model.p) if tilt is None else tilt
    for iteration in range(maxiter):
        old = np.r_[a,m]
        for j in range(model.p):
            x = model.X[:,j]
            mu[j] = var[j]*(x @ residual/model.sigma**2+model.diag[j]*m[j]+tilt[j])
            a[j] = np.clip(expit(logit(model.omega)+.5*(np.log(var[j]/model.tau**2)+mu[j]**2/var[j])),1e-10,1-1e-10)
            new = a[j]*mu[j]
            residual -= x*(new-m[j])
            m[j] = new
        change = float(np.max(np.abs(np.r_[a,m]-old)))
        if change < tol:
            break
    mix = Mixture(np.ones(1),a[None],mu[None],var[None])
    return dict(mixture=mix,objective=float(component_costs(model,mix)[0]),
                status='converged' if change < tol else 'iteration_limit',iterations=iteration+1,
                residual=change,seconds=perf_counter()-began)


def component_kl(mix,reference):
    bern = xlogy(mix.a,mix.a/reference.a)+xlogy(1-mix.a,(1-mix.a)/(1-reference.a))
    normal = .5*((mix.var+(mix.mu-reference.mu)**2)/reference.var-1+np.log(reference.var/mix.var))
    return np.sum(bern+mix.a*normal,axis=1)
