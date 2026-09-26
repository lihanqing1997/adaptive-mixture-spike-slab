"""Complete frozen and stagewise procedures in the reported comparisons."""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, qmc
from .solver import Mixture, Objective, component_kl, compare, Deadline, _split, _alternative

def free_coordinates(k,p,method):
    weights=np.arange(k) if method=='frozen_refinement' else np.array([k-1])
    shapes=np.concatenate([np.arange(k+block*k*p+(k-1)*p,k+(block+1)*k*p)
                           for block in range(3)])
    return np.r_[weights,shapes]


def restricted_refine(model,initial,settings,seed,deadline,method):
    current=initial;trace=[];failures=0
    for refresh in range(settings.max_refreshes):
        if perf_counter()>=deadline:break
        power=settings.training_power+min(2,failures//2)
        cache=Objective(model,current,power,seed+1009*refresh)
        packed=current.pack();free=free_coordinates(current.k,current.p,method)
        scale=np.r_[np.ones(current.k+current.k*current.p),np.sqrt(current.var).ravel(),np.ones(current.k*current.p)]
        width=np.r_[np.ones(current.k),np.full(current.k*current.p,.5),np.full(current.k*current.p,.5),np.full(current.k*current.p,.4)]
        low,high=-width,width.copy();lo,hi=current.k,current.k+current.k*current.p
        low[lo:hi]=np.maximum(low[lo:hi],-23-packed[lo:hi]);high[lo:hi]=np.minimum(high[lo:hi],23-packed[lo:hi])
        calls=0
        def fun(x):
            nonlocal calls
            if perf_counter()>=deadline:raise Deadline()
            calls+=1;theta=packed.copy();theta[free]+=scale[free]*x
            value,grad,_=cache.evaluate(theta,True)
            return value,grad[free]*scale[free]
        record=dict(refresh=refresh,power=power,accepted=False,free_parameters=len(free))
        try:
            result=minimize(fun,np.zeros(len(free)),jac=True,method='L-BFGS-B',
                bounds=list(zip(low[free],high[free])),options=dict(maxiter=settings.inner_iterations,
                    ftol=1e-10,gtol=1e-5,maxls=15))
        except Deadline:
            record.update(status='time_limit',evaluations=calls);trace.append(record);break
        record.update(status=str(result.message),iterations=int(result.nit),evaluations=calls,attempts=[])
        delta=np.zeros_like(packed);delta[free]=scale[free]*result.x;fraction=1.
        for attempt in range(3):
            if perf_counter()>=deadline:break
            trust=False
            for _ in range(12):
                candidate=Mixture.unpack(packed+fraction*delta,current.k,current.p)
                detail=cache.evaluate(candidate.pack(),False)
                ckl=float(np.max(component_kl(candidate,current)))
                wkl=float(candidate.weights@np.log(candidate.weights/current.weights))
                trust=(ckl<=settings.max_component_kl and wkl<=settings.max_component_kl
                       and min(detail['relative_ess'])>=settings.min_relative_ess
                       and max(abs(v-1) for v in detail['importance_mass'])<=settings.mass_tolerance)
                if trust:break
                fraction*=.5
            trial=dict(fraction=fraction,trust_ok=bool(trust),component_kl=ckl,weight_kl=wkl)
            if trust and perf_counter()<deadline:
                validation=compare(model,candidate,current,power=settings.validation_power,
                    seed=seed+5000009+1009*refresh,repeats=settings.validation_repeats,
                    tolerance=settings.acceptance_tol,multiplier=settings.mcse_multiplier)
                trial['validation']=validation
                if validation['accepted']:current=candidate;record['accepted']=True
            record['attempts'].append(trial)
            if record['accepted']:break
            fraction*=.5
        trace.append(record);failures=0 if record['accepted'] else failures+1
        if failures>=settings.stalled_refreshes:break
    return current,trace


def direct_sample(mix,power,seed):
    u=qmc.Sobol(2*mix.p+1,scramble=True,seed=seed).random_base2(power)
    labels=np.searchsorted(np.cumsum(mix.weights),u[:,0],side='right')
    labels=np.minimum(labels,mix.k-1)
    gamma=(u[:,1:1+mix.p]<mix.a[labels]).astype(float)
    beta=gamma*(mix.mu[labels]+np.sqrt(mix.var[labels])*norm.ppf(np.clip(u[:,1+mix.p:],1e-14,1-1e-14)))
    return gamma,beta


def log_target(model,gamma,beta):
    loss=.5*np.sum((beta@model.X.T-model.y)**2,axis=1)/model.sigma**2
    prior=gamma*(np.log(model.omega)-.5*np.log(2*np.pi*model.tau**2)-.5*beta**2/model.tau**2)
    prior+=(1-gamma)*np.log1p(-model.omega)
    return prior.sum(1)-loss


def importance_initializer(model,current,seed,power=12):
    """Mixed-law adaptation of Miller et al. supplementary Algorithm 1.

    Two importance passes, outlier-centered proposal components and clamped-old
    weighted EM. Sampling/variance heuristics are declared in our protocol.
    """
    gamma,beta=direct_sample(current,power,seed)
    logq=logsumexp(current.logdensity(gamma,beta)+np.log(current.weights),axis=1)
    lw=log_target(model,gamma,beta)-logq;weights=np.exp(lw-logsumexp(lw))
    outliers=np.argsort(-weights,kind='stable')[:8]
    outliers=outliers[weights[outliers]>10/len(weights)]
    slabmass=current.weights@current.a
    slabvar=(current.weights@(current.a*current.var))/np.maximum(slabmass,1e-12)
    slabvar=np.maximum(slabvar,1/(model.diag+model.tau**-2))
    if len(outliers):
        ow=weights[outliers];raw_mass=float(ow.sum());mass=min(raw_mass,1-1e-4)
        ow=ow*(mass/raw_mass)
        proposal=Mixture(np.r_[(1-mass)*current.weights,ow],
            np.vstack([current.a,.05+.90*gamma[outliers]]),
            np.vstack([current.mu,beta[outliers]]),
            np.vstack([current.var,np.tile(slabvar,(len(outliers),1))]))
    else:proposal=current
    gamma,beta=direct_sample(proposal,power,seed+170003)
    logq=logsumexp(proposal.logdensity(gamma,beta)+np.log(proposal.weights),axis=1)
    lw=log_target(model,gamma,beta)-logq;weights=np.exp(lw-logsumexp(lw))
    top=int(np.argmax(weights));a=.05+.90*gamma[top];mu=beta[top].copy();var=slabvar.copy();fraction=.1
    oldlog=logsumexp(current.logdensity(gamma,beta)+np.log(current.weights),axis=1)
    for _ in range(20):
        new=Mixture(np.ones(1),a[None,:],mu[None,:],var[None,:])
        newlog=new.logdensity(gamma,beta)[:,0]
        denom=np.logaddexp(np.log1p(-fraction)+oldlog,np.log(fraction)+newlog)
        responsibility=np.exp(np.log(fraction)+newlog-denom)
        weighted=weights*responsibility;mass=float(weighted.sum())
        fraction=float(np.clip(mass,1e-4,1-1e-4))
        active=weighted@gamma
        a=np.clip(active/max(mass,1e-300),1e-6,1-1e-6)
        mu=np.divide(weighted@beta,active,out=np.zeros(model.p),where=active>1e-12)
        second=np.divide(weighted@(beta*beta),active,out=slabvar.copy(),where=active>1e-12)
        var=np.clip(second-mu**2,.1/(model.diag+model.tau**-2),4*model.tau**2)
    initial=Mixture(np.r_[(1-fraction)*current.weights,fraction],np.vstack([current.a,a]),
                    np.vstack([current.mu,mu]),np.vstack([current.var,var]))
    return initial,dict(kind='importance_weighted_em',seed=seed,power=power,
                        outliers=len(outliers),second_pass_ess=float(1/np.sum(weights**2)),
                        initial_fraction=fraction)


def fit_variant(model,settings,seed,baseline,method):
    if method not in ['stagewise_boosting','frozen_refinement']:raise ValueError(method)
    began=perf_counter();deadline=began+settings.max_seconds;current=baseline['mixture'];history=[];status='component_cap'
    for stage in range(1,settings.max_components):
        if perf_counter()>=deadline:status='time_limit';break
        accepted=False
        for number in range(settings.proposals_per_stage):
            if perf_counter()>=deadline:status='time_limit';break
            proposal_began=perf_counter()
            if method=='stagewise_boosting' and number<2:
                proposal,detail=importance_initializer(model,current,seed+stage*1000003+number*70001)
            elif number<2:proposal,detail=_split(model,current,number)
            else:proposal,detail=_alternative(model,current,number-2,seed+stage)
            trial,trace=restricted_refine(model,proposal,settings,seed+stage*1000003+number*70001,
                min(deadline,perf_counter()+settings.proposal_seconds),method)
            if not (np.allclose(trial.a[:-1],proposal.a[:-1],atol=1e-13,rtol=0)
                    and np.allclose(trial.mu[:-1],proposal.mu[:-1],atol=1e-13,rtol=0)
                    and np.allclose(trial.var[:-1],proposal.var[:-1],atol=1e-13,rtol=0)):
                raise AssertionError('Frozen component changed during refinement')
            if method=='stagewise_boosting':
                for field in ['a','mu','var']:
                    if not np.allclose(getattr(trial,field)[:-1],getattr(current,field),atol=1e-13,rtol=0):
                        raise AssertionError('Stagewise incumbent shape changed')
                if not np.allclose(trial.weights[:-1]/trial.weights[:-1].sum(),current.weights,atol=1e-13,rtol=0):
                    raise AssertionError('Stagewise incumbent relative weights changed')
            comparison=compare(model,trial,current,power=settings.validation_power+1,
                seed=seed+80000009+stage*100003+number*1009,repeats=settings.validation_repeats,
                tolerance=settings.acceptance_tol,multiplier=settings.mcse_multiplier)
            history.append(dict(stage=stage,proposal=detail,refinement=trace,validation=comparison,
                accepted=comparison['accepted'],seconds=perf_counter()-proposal_began))
            if comparison['accepted']:current=trial;accepted=True;break
        if not accepted:status='time_limit' if perf_counter()>=deadline else 'no_supported_proposal';break
    fit_seconds=perf_counter()-began
    final=compare(model,current,baseline['mixture'],power=settings.final_power,seed=seed+170000009,
        repeats=settings.final_repeats,tolerance=settings.acceptance_tol,multiplier=settings.mcse_multiplier)
    fallback=bool(current.k>1 and not final['accepted']);candidate_components=current.k
    if fallback:current=baseline['mixture']
    return dict(mixture=current,baseline=baseline['mixture'],baseline_objective=baseline['objective'],
        baseline_status=baseline['status'],candidate_components=candidate_components,components=current.k,
        final_fallback=fallback,status=status,history=history,final_validation=final,
        returned_objective=baseline['objective'] if current.k==1 else final['candidate_objective'],
        returned_delta_mf=0. if current.k==1 else final['difference'],fit_seconds=fit_seconds,
        total_seconds=perf_counter()-began,baseline_seconds=0.,method=method,certified_global_optimum=False)
