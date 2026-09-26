import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from time import perf_counter
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from test_solver import example
from amvi.solver import Objective,Mixture,Settings
from amvi.comparators import free_coordinates,restricted_refine,importance_initializer,direct_sample,log_target


def test_restricted_derivatives_off_reference():
    model,mix=example()
    objective=Objective(model,mix,power=7,seed=217)
    base=mix.pack()+.03*np.cos(np.arange(len(mix.pack())))
    for method in ['stagewise_boosting','frozen_refinement']:
        free=free_coordinates(mix.k,mix.p,method)
        def value(x):
            full=base.copy();full[free]=x
            return objective.evaluate(full,False)['objective']
        numeric=approx_derivative(value,base[free],method='3-point').ravel()
        analytic=objective.evaluate(base,True)[1][free]
        assert_allclose(analytic,numeric,rtol=5e-6,atol=4e-7)


def test_restricted_refinement_preserves_incumbents():
    model,mix=example()
    mix=Mixture(np.array([.2,.3,.5]),np.vstack([mix.a,mix.a[:1]]),
        np.vstack([mix.mu,mix.mu[:1]+.1]),np.vstack([mix.var,mix.var[:1]]))
    settings=Settings(max_refreshes=2,inner_iterations=3,training_power=7,
        validation_power=8,validation_repeats=2)
    for method in ['stagewise_boosting','frozen_refinement']:
        result,trace=restricted_refine(model,mix,settings,913,perf_counter()+10,method)
        assert trace
        for field in ['a','mu','var']:
            assert_allclose(getattr(result,field)[:-1],getattr(mix,field)[:-1],atol=1e-14)
        if method=='stagewise_boosting':
            assert_allclose(result.weights[:-1]/result.weights[:-1].sum(),mix.weights[:-1]/mix.weights[:-1].sum(),atol=1e-14)


def test_initializer_preserves_incumbent_distribution():
    model,mix=example()
    result,detail=importance_initializer(model,mix,876,power=8)
    assert result.k==mix.k+1
    assert np.all(result.weights>0)
    assert_allclose(result.weights.sum(),1)
    for field in ['a','mu','var']:
        assert_allclose(getattr(result,field)[:-1],getattr(mix,field))
    assert_allclose(result.weights[:-1]/result.weights[:-1].sum(),mix.weights)
    gamma,beta=direct_sample(result,8,891)
    assert np.all(beta[gamma==0]==0)
    assert np.all(np.isfinite(result.logdensity(gamma,beta)))
    assert np.all(np.isfinite(log_target(model,gamma,beta)))


def test_target_log_density_against_explicit_formula():
    model,mix=example()
    gamma,beta=direct_sample(mix,5,314)
    expected=[]
    for g,b in zip(gamma,beta):
        active=g.astype(bool)
        value=-np.sum((model.y-model.X@b)**2)/(2*model.sigma**2)
        value+=np.sum(active)*np.log(model.omega)+np.sum(~active)*np.log(1-model.omega)
        value+=np.sum(-.5*np.log(2*np.pi*model.tau**2)-b[active]**2/(2*model.tau**2))
        expected.append(value)
    assert_allclose(log_target(model,gamma,beta),expected,atol=1e-12)
