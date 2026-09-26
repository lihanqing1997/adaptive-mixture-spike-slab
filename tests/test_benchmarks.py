import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import logsumexp
from amvi.solver import RegressionModel,best_meanfield
from amvi.references import SupportPosterior,groups_for,sinkhorn_correction,diagnostic,posterior_mcmc
from amvi.exact import exact_reference


def example(p=3):
    rng=np.random.default_rng(1909)
    x=rng.normal(size=(20,p)); x[:,1]=.6*x[:,0]+.8*x[:,1]
    y=.5*x[:,0]+rng.normal(size=20)
    return RegressionModel(x,y,omega=.2)


def test_collapsed_support_masses_match_independent_exact_reference():
    model=example(); exact=exact_reference(model); target=SupportPosterior(model)
    masks=np.array(list(itertools.product([0,1],repeat=model.p)))
    values=np.array([target.log_mass(tuple(np.flatnonzero(mask))) for mask in masks])
    assert_allclose(logsumexp(values),exact['logZ'],atol=1e-11)
    assert_allclose(np.exp(values-logsumexp(values)),exact['support'],atol=1e-11)


def test_grouping_covers_coordinates_once_and_controls_group_count():
    for policy in ['interleaved','contiguous','random']:
        groups=groups_for(100,4,policy,7)
        assert len(groups)==4
        assert sorted(np.concatenate(groups).tolist())==list(range(100))
        assert all(len(g)==25 for g in groups)
    assert [x.tolist() for x in groups_for(12,4,'contiguous')]==[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]


def test_sinkhorn_preserves_empirical_marginals_and_reports_discrete_law():
    model=example(p=3)
    fit=sinkhorn_correction(model,number=3,points=8,seconds=3,return_samples=0,seed=9)
    assert fit['status']=='converged'
    assert fit['group_marginal_l1']<1e-9
    assert fit['ordinary_reverse_kl'] is None
    assert fit['certificate']['improvement_over_product']>=-1e-8
    assert np.linalg.eigvalsh(fit['covariance']).min()>-1e-8


def test_sinkhorn_memory_guard_precedes_allocation():
    result=sinkhorn_correction(example(),number=3,points=1000,max_entries=1000)
    assert result['status']=='representation_cap'


def test_chain_diagnostics_detect_shift_and_handle_constants():
    rng=np.random.default_rng(311)
    iid=rng.normal(size=(4,1000))
    assert diagnostic(iid)['rhat']<1.01
    shifted=iid+np.arange(4)[:,None]
    assert diagnostic(shifted)['rhat']>1.1
    assert diagnostic(np.zeros((4,1000)))['constant']


def test_small_collapsed_mcmc_matches_exact_inclusion_probabilities():
    model=example(); exact=exact_reference(model)
    fit=posterior_mcmc(model,seed=1729,chains=4,warmup=150,draws=700,seconds=20,return_samples=False)
    assert fit['retained_draws_per_chain']==700
    assert_allclose(fit['pip'],exact['pip'],atol=.045)
    assert_allclose(fit['mean'],exact['mean'],atol=.07)
