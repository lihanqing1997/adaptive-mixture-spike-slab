"""Deterministic independent numerical checks; run before source freeze."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc, norm
from amvi.objectives import *


def run_checks():
    rng = np.random.default_rng(40231)
    model = RegressionModel(
        rng.normal(size=(9, 3)) * 0.3, rng.normal(size=9) * 0.3, tau=1.3, omega=0.3
    )
    records = {}

    def make(k):
        return Mixture(
            np.arange(1, k + 1) / sum(range(1, k + 1)),
            rng.uniform(0.03, 0.8, (k, 3)),
            rng.normal(size=(k, 3)) * 0.7,
            rng.uniform(0.3, 1.8, (k, 3)),
        )

    identity = []
    for _ in range(8):
        m = make(1)
        normal = 0.5 * (
            (m.var + m.mu**2) / model.tau**2 - 1 - np.log(m.var / model.tau**2)
        )
        difference = (
            evaluate(model, m, augmented=True)["objective"]
            - evaluate(model, m)["objective"]
        )
        identity.append(abs(difference - float(np.sum((1 - m.a) * normal))))
    assert max(identity) < 1e-12
    records["single_component_identity_max_error"] = max(identity)
    gradient_records = []
    for k in (2, 3):
        reference = make(k)
        theta = reference.pack() + rng.normal(size=len(reference.pack())) * 0.025
        for augmented in (False, True):
            obj = Objective(model, reference, power=9, seed=2389, augmented=augmented)
            value, grad, _ = obj.evaluate(theta)
            numeric = []
            h = 1e-5
            for j in range(len(theta)):
                delta = np.zeros_like(theta)
                delta[j] = h
                numeric.append(
                    (
                        obj.evaluate(theta + delta, False)["objective"]
                        - obj.evaluate(theta - delta, False)["objective"]
                    )
                    / (2 * h)
                )
            numeric = np.array(numeric)
            error = np.abs(grad - numeric)
            scaled = error / (1 + np.abs(numeric))
            assert error.max() < 2e-7 and scaled.max() < 1e-7
            gradient_records.append(
                dict(
                    k=k,
                    augmented=augmented,
                    max_absolute=float(error.max()),
                    max_scaled=float(scaled.max()),
                )
            )
    records["finite_estimator_gradients"] = gradient_records
    # All points have gamma=0: augmented Gaussian scores still have nonzero gradients.
    m = Mixture(
        [0.4, 0.6],
        np.full((2, 3), 1e-9),
        np.array([[1.0, -0.5, 0.2], [-1.0, 0.8, -0.3]]),
        np.ones((2, 3)),
    )
    info = Information(m, power=8, seed=56)
    assert np.count_nonzero(info.gamma) == 0
    _, ig, _ = info.evaluate(m, True, True)
    _, dg, _ = info.evaluate(m, True, False)
    assert np.linalg.norm(ig[2]) > 0.01 and np.max(np.abs(dg[2])) == 0
    records["inactive_gaussian_information_gradient_norm"] = float(
        np.linalg.norm(ig[2])
    )
    records["full_density_identity"] = []
    for k, separation in ((2, 0.2), (3, 1.4)):
        mix = make(k)
        mix.a[:] = rng.uniform(0.02, 0.25, mix.a.shape)
        mix.mu *= separation
        diffs = []
        for rep in range(8):
            sd = seed_for(0, k, "checks", repeat=rep)
            raw = 0.0
            # Independent source-labelled full-density integral, not information architecture.
            for source in range(k):
                u = qmc.Sobol(
                    6, scramble=True, seed=source_seed(sd, source)
                ).random_base2(14)
                g = (u[:, :3] < mix.a[source]).astype(float)
                B = mix.mu[source] + np.sqrt(mix.var[source]) * norm.ppf(
                    np.clip(u[:, 3:], 1e-14, 1 - 1e-14)
                )
                beta = g * B
                loss = (
                    0.5
                    * np.sum((beta @ model.X.T - model.y) ** 2, axis=1)
                    / model.sigma**2
                )
                lprior = (
                    g * np.log(model.omega) + (1 - g) * np.log1p(-model.omega)
                ).sum(1) - 0.5 * np.sum(
                    np.log(2 * np.pi * model.tau**2) + B**2 / model.tau**2, axis=1
                )
                lq = logsumexp(logdensity_plus(mix, g, B) + np.log(mix.weights), axis=1)
                raw += mix.weights[source] * float(np.mean(loss + lq - lprior))
            analytic = evaluate(
                model, mix, power=14, seed=source_seed(sd, 99), augmented=True
            )["objective"]
            diffs.append(raw - analytic)
        se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        bias = float(np.mean(diffs))
        assert abs(bias) < max(5 * se, 2e-4)
        records["full_density_identity"].append(
            dict(k=k, separation=separation, differences=diffs, mean=bias, mcse=se)
        )
    # Permuting components and source seeds together must leave population and finite rule invariant.
    mix = make(3)
    perm = np.array([2, 0, 1])
    other = Mixture(mix.weights[perm], mix.a[perm], mix.mu[perm], mix.var[perm])
    seeds = [12, 13, 14]
    invariance = []
    for aug in (False, True):
        a = evaluate(model, mix, 12, augmented=aug, source_seeds=seeds)
        b = evaluate(
            model, other, 12, augmented=aug, source_seeds=np.array(seeds)[perm]
        )
        assert abs(a["objective"] - b["objective"]) < 1e-12
        assert -1e-12 <= a["information"] <= -mix.weights @ np.log(mix.weights) + 1e-12
        invariance.append(abs(a["objective"] - b["objective"]))
    records["permutation_max_error"] = max(invariance)
    masks = (np.arange(8)[:, None] >> np.arange(3)) & 1
    probs = np.sum(
        mix.weights[None, :]
        * np.prod(
            np.where(masks[:, None, :], mix.a[None, :, :], 1 - mix.a[None, :, :]),
            axis=2,
        ),
        axis=1,
    )
    assert abs(probs.sum() - 1) < 1e-14
    info = Information(mix, power=8, seed=231)
    assert np.all(info.beta[info.gamma == 0] == 0) and np.all(
        info.beta[info.gamma == 1] == info.B[info.gamma == 1]
    )
    overlap = info.overlap(mix)
    for space in overlap.values():
        assert np.max(np.abs(np.array(space["importance_mass"]) - 1)) < 1e-12
        assert np.max(np.abs(np.array(space["relative_ess"]) - 1)) < 1e-12
    prior = Mixture(
        mix.weights, mix.a, np.zeros_like(mix.mu), np.full_like(mix.var, model.tau**2)
    )
    assert (
        np.max(np.abs(costs(model, prior, augmented=True) - costs(model, prior)))
        < 1e-12
    )
    assert (
        abs(
            evaluate(model, prior, 12, 7, True)["objective"]
            - evaluate(model, prior, 12, 7, False)["objective"]
        )
        < 1e-12
    )
    from amvi.solver import component_kl

    target = make(3)
    assert np.all(full_component_kl(target, mix) >= component_kl(target, mix) - 1e-12)
    records["support_mask_prior_and_overlap_checks"] = "passed"
    # Own-objective decision: moving toward a near-active likelihood fit helps direct,
    # whereas the inactive latent Gaussian prior makes this move harmful augmented.
    empty = RegressionModel(np.zeros((2, 1)), np.zeros(2), omega=0.001)
    incumbent = Mixture([1.0], [[0.001]], [[3.0]], [[1.0]])
    candidate = Mixture([1.0], [[0.003]], [[2.0]], [[1.0]])
    direct = compare(empty, candidate, incumbent, repeats=4)
    augmented = compare(empty, candidate, incumbent, repeats=4, augmented=True)
    assert augmented["accepted"] and not direct["accepted"]
    records["own_objective_selection"] = dict(
        direct=direct["difference"],
        augmented=augmented["difference"],
        augmented_accepted=augmented["accepted"],
        direct_accepted=direct["accepted"],
    )
    records["passed"] = True
    return records


def test_numerical_checks():
    assert run_checks()["passed"]
