import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logsumexp
from amvi.solver import (
    RegressionModel,
    Mixture,
    Information,
    Objective,
    Settings,
    fit,
    component_costs,
    evaluate,
    compare,
    best_meanfield,
)


def example(p=3):
    rng = np.random.default_rng(721)
    model = RegressionModel(rng.normal(size=(18, p)), rng.normal(size=18), omega=0.25)
    mix = Mixture(
        np.array([0.35, 0.65]),
        rng.uniform(0.15, 0.8, (2, p)),
        rng.normal(scale=0.3, size=(2, p)),
        rng.uniform(0.3, 0.9, (2, p)),
    )
    return model, mix


def test_fixed_mixture_information_all_gradients_off_reference():
    model, mix = example()
    obj = Objective(model, mix, power=7, seed=19)
    theta = mix.pack() + 0.04 * np.sin(np.arange(len(mix.pack())) + 0.3)
    value, gradient, _ = obj.evaluate(theta)
    numeric = approx_derivative(
        lambda t: obj.evaluate(t, False)["objective"], theta, method="3-point"
    ).ravel()
    assert_allclose(gradient, numeric, rtol=4e-6, atol=3e-7)
    assert np.isfinite(value)
    assert abs(gradient[: mix.k].sum()) < 1e-10


def test_exact_information_on_shared_slab_supports():
    model, _ = example()
    ref = Mixture(
        np.array([0.4, 0.6]),
        np.array([[0.2, 0.5, 0.7], [0.7, 0.3, 0.4]]),
        np.zeros((2, 3)),
        np.ones((2, 3)),
    )
    target = Mixture(np.array([0.3, 0.7]), ref.a * 0.8 + 0.1, ref.mu, ref.var)
    patterns = np.array(list(itertools.product([0.0, 1.0], repeat=3)))
    beta = patterns * 0.37
    cache = Information(ref, power=3, seed=3)
    cache.gamma = patterns
    cache.beta = beta
    cache.beta2 = beta**2
    ld = ref.logdensity(patterns, beta)
    cache.log_r = logsumexp(ld + np.log(ref.weights), axis=1)
    discrete = patterns @ np.log(ref.a).T + (1 - patterns) @ np.log1p(-ref.a).T
    cache.measure = np.exp(logsumexp(discrete + np.log(ref.weights), axis=1))
    cache.source = np.arange(8) % 2
    cache.ref_source = ld[np.arange(8), cache.source]
    got = cache.evaluate(target)[0]
    lp = patterns @ np.log(target.a).T + (1 - patterns) @ np.log1p(-target.a).T
    lq = logsumexp(lp + np.log(target.weights), axis=1)
    expected = float(np.sum(np.exp(lp) * target.weights * (lp - lq[:, None])))
    assert_allclose(got, expected, atol=1e-12)


def test_identical_components_equal_analytic_meanfield():
    model, mix = example()
    for k in [1, 3]:
        same = Mixture(
            np.full(k, 1 / k),
            np.repeat(mix.a[:1], k, axis=0),
            np.repeat(mix.mu[:1], k, axis=0),
            np.repeat(mix.var[:1], k, axis=0),
        )
        val = evaluate(model, same, power=5)
        assert_allclose(val["objective"], component_costs(model, same)[0], atol=1e-11)
        assert abs(val["information"]) < 1e-12
        assert "total_correlation" not in val


def test_one_coordinate_still_has_nonzero_mixture_information():
    model, mix = example(p=1)
    value = evaluate(model, mix, power=10, seed=33)
    assert value["information"] > 0
    assert value["information"] < -mix.weights @ np.log(mix.weights)


def test_fresh_comparison_of_identical_law_is_unresolved():
    model, mix = example()
    comparison = compare(model, mix, mix, power=6, seed=42, repeats=3)
    assert comparison["difference"] == 0.0
    assert comparison["mcse"] == 0.0
    assert not comparison["accepted"]


def test_batched_validation_matches_full_rule():
    model, mix = example()
    full = Objective(model, mix, power=11, seed=43).evaluate(mix.pack(), False)
    batched = evaluate(model, mix, power=11, seed=43)
    assert_allclose(batched["objective"], full["objective"], atol=1e-11)


def test_adaptive_cap_one_returns_analytic_baseline():
    model, _ = example(p=2)
    base = best_meanfield(model)
    result = fit(
        model, Settings(max_components=1, final_power=5), seed=9, baseline=base
    )
    assert result["components"] == 1
    assert result["returned_delta_mf"] == 0.0
    assert not result["certified_global_optimum"]


def test_adaptive_acceptance_and_final_fallback(monkeypatch):
    from amvi import solver

    model, _ = example(p=2)
    actual = solver.compare

    def screen(m, c, b, **kw):
        result = actual(m, c, b, **kw)
        if kw["seed"] > 170000000:
            result.update(accepted=False, difference=0.1)
        else:
            result.update(accepted=True, difference=-0.1)
        return result

    monkeypatch.setattr(solver, "compare", screen)
    result = fit(
        model,
        Settings(
            max_components=2,
            proposals_per_stage=1,
            max_refreshes=0,
            training_power=4,
            validation_power=4,
            final_power=5,
        ),
        seed=3,
    )
    assert result["candidate_components"] == 2
    assert result["final_fallback"]
    assert result["components"] == 1
    assert result["returned_delta_mf"] == 0.0
