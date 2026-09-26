"""Reported MCMC references, diagnostics, and fixed-marginal comparator."""
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import expit
from scipy.stats import norm, qmc, rankdata
from .solver import best_meanfield

def groups_for(p,number=4,policy='interleaved',seed=0):
    if not 1<=number<=p: raise ValueError('Group count must be between one and p')
    order=np.arange(p)
    if policy=='random': np.random.default_rng(seed).shuffle(order)
    if policy=='interleaved': return [order[g::number] for g in range(number)]
    if policy in ('contiguous','random'): return list(np.array_split(order,number))
    raise ValueError('Unknown grouping policy')


def sinkhorn_correction(model,baseline=None,*,number=4,points=20,lam=0.,policy='interleaved',
                        seed=0,seconds=30.,max_entries=3_000_000,return_samples=2000):
    """Wu--Blei-style one-step correction, adapted to mixed SS marginals.

    Supplied group laws are finite direct-sampling approximations to the same
    MFVI baseline. Marginals are fixed. This is not their full outer algorithm,
    their EP initialization, or a finite-K continuous variational density.
    """
    began=perf_counter(); number=min(number,model.p)
    entries=int(points)**number
    if entries>max_entries:
        return dict(status='representation_cap',entries=entries,max_entries=max_entries,seconds=perf_counter()-began)
    base=best_meanfield(model) if baseline is None else baseline
    mix=base['mixture']; blocks=groups_for(model.p,number,policy,seed)
    nodes=[]; indicators=[]; contributions=[]
    for g,indices in enumerate(blocks):
        # Arbitrary N is obtained as a prefix of a scrambled power-of-two rule.
        u=qmc.Sobol(2*len(indices),scramble=True,seed=seed+104729*g).random_base2(int(np.ceil(np.log2(points))))[:points]
        gamma=(u[:,:len(indices)]<mix.a[0,indices]).astype(float)
        beta=gamma*(mix.mu[0,indices]+np.sqrt(mix.var[0,indices])*norm.ppf(np.clip(u[:,len(indices):],1e-14,1-1e-14)))
        nodes.append(beta); indicators.append(gamma)
        contributions.append(beta@model.X[:,indices].T/model.sigma)
    response=model.y/model.sigma
    cost=np.full((points,)*number,.5*response@response)
    def expand(v,axes):
        shape=[1]*number
        for axis in axes: shape[axis]=points
        return v.reshape(shape)
    for g,z in enumerate(contributions):
        cost+=expand(.5*np.sum(z*z,axis=1)-z@response,[g])
        for h in range(g): cost+=expand(contributions[h]@z.T,[h,g])
    from ._sinkhorn import solve
    from ._transport import feasible_round
    targets=[np.full(points,1/points) for _ in blocks]
    solved=solve(cost,targets,lam,seconds=seconds,tol=1e-8,gap_tol=1e-8,return_logw=True)
    logw=solved.pop('logw'); weight=feasible_round(np.exp(logw),targets)
    mean=np.zeros(model.p); pip=np.zeros(model.p); covariance=np.zeros((model.p,model.p))
    for g,idx in enumerate(blocks):
        mean[idx]=targets[g]@nodes[g]; pip[idx]=targets[g]@indicators[g]
        covariance[np.ix_(idx,idx)]=(nodes[g].T*targets[g])@nodes[g]-np.outer(mean[idx],mean[idx])
        for h in range(g):
            other=blocks[h]; axes=tuple(a for a in range(number) if a not in (h,g))
            joint=weight.sum(axis=axes) if axes else weight
            cross=nodes[h].T@joint@nodes[g]-np.outer(mean[other],mean[idx])
            covariance[np.ix_(other,idx)]=cross; covariance[np.ix_(idx,other)]=cross.T
    residual=max(float(np.abs(weight.sum(axis=tuple(h for h in range(number) if h!=g))-targets[g]).sum()) for g in range(number))
    result=dict(**solved,entries=entries,groups=[x.tolist() for x in blocks],points=points,lam=lam,
        grouping=policy,seed=seed,mean=mean,pip=pip,covariance=covariance,
        analytic_mf_pip=mix.a[0],analytic_mf_mean=mix.a[0]*mix.mu[0],
        group_marginal_l1=residual,seconds=perf_counter()-began,
        ordinary_reverse_kl=None,representation='discrete_group_particles',
        note='Preserves empirical group marginals; discretization error is separate from coupling error.')
    if return_samples:
        rng=np.random.default_rng(seed+9871)
        selected=rng.choice(weight.size,size=return_samples,p=weight.ravel()/weight.sum())
        positions=np.array(np.unravel_index(selected,weight.shape)).T
        beta=np.zeros((return_samples,model.p)); gamma=np.zeros_like(beta,dtype=np.uint8)
        for g,idx in enumerate(blocks):
            beta[:,idx]=nodes[g][positions[:,g]]; gamma[:,idx]=indicators[g][positions[:,g]]
        result.update(beta_samples=beta,gamma_samples=gamma)
    return result


class SupportPosterior:
    """Collapsed Gaussian support posterior, independent of the VI evaluator."""
    def __init__(self,model):
        self.model=model
        self.gram=model.X.T@model.X/model.sigma**2
        self.score=model.X.T@model.y/model.sigma**2
        self.constant=-.5*(model.y@model.y)/model.sigma**2+model.p*np.log1p(-model.omega)
        self.log_odds=np.log(model.omega)-np.log1p(-model.omega)

    @lru_cache(maxsize=50000)
    def log_mass(self,support):
        s=len(support)
        if not s: return float(self.constant)
        ids=np.asarray(support,dtype=int)
        precision=self.gram[np.ix_(ids,ids)]+np.eye(s)/self.model.tau**2
        cf=cho_factor(precision,lower=True,check_finite=False)
        score=self.score[ids]
        return float(self.constant+s*self.log_odds-s*np.log(self.model.tau)
            -np.log(np.diag(cf[0])).sum()+.5*score@cho_solve(cf,score,check_finite=False))

    def beta_draw(self,support,rng):
        beta=np.zeros(self.model.p)
        if support:
            ids=np.asarray(support,dtype=int)
            precision=self.gram[np.ix_(ids,ids)]+np.eye(len(ids))/self.model.tau**2
            chol=np.linalg.cholesky(precision)
            mean=cho_solve((chol,True),self.score[ids],check_finite=False)
            beta[ids]=mean+solve_triangular(chol.T,rng.normal(size=len(ids)),lower=False,check_finite=False)
        return beta


def _split(x):
    half=x.shape[1]//2
    return np.concatenate([x[:,:half],x[:,-half:]],axis=0)


def _rhat_and_ess(x):
    m,n=x.shape; w=float(np.var(x,axis=1,ddof=1).mean())
    b=float(n*np.var(x.mean(1),ddof=1))
    if w<1e-25:
        return (1.,float(m*n)) if b<1e-25 else (float('inf'),0.)
    varplus=(n-1)/n*w+b/n
    centered=x-x.mean(1)[:,None]
    fourier=np.fft.rfft(centered,n=2*n,axis=1)
    acov=np.fft.irfft(fourier*np.conj(fourier),n=2*n,axis=1)[:,:n]/n
    rho=1-(w-acov.mean(0))/varplus; rho[0]=1.
    pairs=[]
    for k in range(0,n-1,2):
        pair=float(rho[k]+rho[k+1])
        if pair<=0: break
        pairs.append(min(pair,pairs[-1]) if pairs else pair)
    tau=max(1.,-1+2*sum(pairs))
    return float(np.sqrt(varplus/w)),float(min(m*n,m*n/tau))


def diagnostic(x):
    # The reported runs applied this conversion before binary-event diagnostics.
    x=np.asarray(x,dtype=float)
    if x.shape[1]<20: return dict(rhat=None,bulk_ess=0.,tail_ess=0.,mean_mcse=None,constant=False)
    if np.ptp(x)==0: return dict(rhat=None,bulk_ess=None,tail_ess=None,mean_mcse=None,constant=True)
    split=_split(np.asarray(x,float))
    ranks=rankdata(split.ravel(),method='average')
    z=norm.ppf((ranks-.375)/(len(ranks)+.25)).reshape(split.shape)
    folded=np.abs(split-np.median(split))
    ranks_fold=rankdata(folded.ravel(),method='average')
    zfold=norm.ppf((ranks_fold-.375)/(len(ranks_fold)+.25)).reshape(split.shape)
    rhat,ess=_rhat_and_ess(z); folded_rhat,_=_rhat_and_ess(zfold)
    rhat=max(rhat,folded_rhat)
    tail=[]
    for probability in [.05,.95]:
        indicator=(split<=np.quantile(split,probability)).astype(float)
        if np.ptp(indicator)>0: tail.append(_rhat_and_ess(indicator)[1])
    raw_ess=_rhat_and_ess(split)[1]
    return dict(rhat=rhat if np.isfinite(rhat) else None,bulk_ess=ess,constant=False,
                tail_ess=min(tail) if tail else None,
                mean_mcse=float(np.std(x,ddof=1)/np.sqrt(raw_ess)) if raw_ess>0 else None,
                variance_failure=not np.isfinite(rhat))


def posterior_mcmc(model,baseline=None,*,seed=0,chains=4,warmup=1000,draws=2000,
                   seconds=180.,return_samples=True):
    """Collapsed random-scan Gibbs plus reversible swaps, then Gaussian draws.

    Whole coordinate sweeps update the support. Independent chains are run
    interleaved, so a deadline cannot leave only the first chain sampled.
    """
    if chains<2 or warmup<0 or draws<20: raise ValueError('At least two chains and 20 draws required')
    began=perf_counter(); target=SupportPosterior(model)
    base=best_meanfield(model) if baseline is None else baseline
    rngs=[np.random.default_rng(seed+104729*c) for c in range(chains)]
    supports=[]
    for c,rng in enumerate(rngs):
        if c==0: state=set()
        elif c==1: state=set(np.flatnonzero(base['mixture'].a[0]>.5).tolist())
        elif c==2: state=set(np.argsort(-np.abs(target.score))[:min(5,model.p)].tolist())
        else: state=set(np.flatnonzero(rng.uniform(size=model.p)<min(.1,2/model.p)).tolist())
        supports.append(state)
    beta=[]; gamma=[]; model_size=[]; log_mass=[]; swaps=0; swap_accepts=0
    status='draw_limit'
    for sweep in range(warmup+draws):
        if perf_counter()-began>=seconds: status='time_limit'; break
        beta_row=[]; gamma_row=[]; size_row=[]; log_row=[]
        for c,rng in enumerate(rngs):
            state=supports[c]
            for j in rng.permutation(model.p):
                old=tuple(sorted(state)); alt=set(state)
                if j in alt: alt.remove(int(j))
                else: alt.add(int(j))
                alternative=tuple(sorted(alt))
                log_old=target.log_mass(old); log_alt=target.log_mass(alternative)
                # Probability of switching to the alternative in this binary Gibbs conditional.
                if rng.uniform()<expit(log_alt-log_old): state=alt
            if 0<len(state)<model.p:
                swaps+=1
                included=np.array(sorted(state)); excluded=np.array(sorted(set(range(model.p))-state))
                trial=set(state); trial.remove(int(rng.choice(included))); trial.add(int(rng.choice(excluded)))
                if np.log(rng.uniform())<target.log_mass(tuple(sorted(trial)))-target.log_mass(tuple(sorted(state))):
                    state=trial; swap_accepts+=1
            supports[c]=state
            if sweep>=warmup:
                ids=tuple(sorted(state)); beta_row.append(target.beta_draw(ids,rng))
                included=np.zeros(model.p,dtype=np.uint8); included[list(ids)]=1
                gamma_row.append(included); size_row.append(len(ids)); log_row.append(target.log_mass(ids))
        if sweep>=warmup:
            beta.append(beta_row); gamma.append(gamma_row); model_size.append(size_row); log_mass.append(log_row)
    if len(beta)<20:
        target.log_mass.cache_clear()
        return dict(status=status,reference_eligible=False,retained_draws_per_chain=len(beta),
            seconds=perf_counter()-began,note='Insufficient retained draws; not a posterior reference.')
    beta=np.transpose(np.asarray(beta),(1,0,2)); gamma=np.transpose(np.asarray(gamma),(1,0,2))
    pip=gamma.mean((0,1)); means=beta.mean((0,1))
    diagnostics={'model_size':diagnostic(np.asarray(model_size).T),'log_support_mass':diagnostic(np.asarray(log_mass).T)}
    for j in range(model.p):
        diagnostics[f'beta_{j}']=diagnostic(beta[:,:,j])
        diagnostics[f'inclusion_{j}']=diagnostic(gamma[:,:,j])
    varying=[d for d in diagnostics.values() if not d['constant']]
    finite=[d for d in varying if d['rhat'] is not None]
    max_rhat=max((d['rhat'] for d in finite),default=None)
    min_ess=min((d['bulk_ess'] for d in varying),default=0.)
    min_tail=min((d['tail_ess'] for d in varying if d['tail_ess'] is not None),default=0.)
    eligible=bool(len(finite)==len(varying) and max_rhat is not None and max_rhat<=1.01
                  and min_ess>=400 and min_tail>=400 and len(beta[0])>=400)
    flat=beta.reshape(-1,model.p)
    result=dict(status=status,reference_eligible=eligible,retained_draws_per_chain=beta.shape[1],chains=chains,
        warmup=warmup,seed=seed,seconds=perf_counter()-began,pip=pip,mean=means,
        covariance=np.cov(flat,rowvar=False),intervals=np.quantile(flat,[.025,.975],axis=0).T,
        max_rank_split_rhat=max_rhat,min_bulk_ess=min_ess,min_tail_ess=min_tail,diagnostics=diagnostics,
        constant_diagnostics=sum(d['constant'] for d in diagnostics.values()),
        swap_acceptance=swap_accepts/max(1,swaps),ordinary_reverse_kl=None,
        note='Diagnostic eligibility is not proof of mixing or rare-support exploration. Constant observables carry no mixing evidence.')
    if return_samples: result.update(beta_samples=beta,gamma_samples=gamma)
    target.log_mass.cache_clear()
    return result
