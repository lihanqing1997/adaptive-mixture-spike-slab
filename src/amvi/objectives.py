"""Frozen-reference direct/independent-augmentation objectives for this study only."""

import numpy as np
from scipy.special import logsumexp, xlogy
from scipy.stats import norm, qmc

from .solver import Mixture, RegressionModel, component_costs

MASTER_SEED = 20260925
PHASES = dict(
    candidate=1,
    training=2,
    local_validation=3,
    final_validation=4,
    assessment=5,
    checks=6,
    manifest=7,
    order=7,
)


def seed_for(rho_index, replicate, phase, refresh=0, repeat=0, source=0):
    """64-bit SeedSequence mapping; arms intentionally share common random numbers.

    Preparation enumerates every planned tuple and every derived source seed and
    verifies collision freedom, including disjoint fitting/assessment streams.
    """
    return int(
        np.random.SeedSequence(
            [
                MASTER_SEED,
                int(rho_index),
                int(replicate),
                PHASES[phase],
                int(refresh),
                int(repeat),
                int(source),
            ]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def source_seed(seed, source):
    return int(
        np.random.SeedSequence([int(seed), int(source), 104729]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def logdensity_plus(mix, gamma, B):
    return (
        np.log1p(-mix.a).sum(1)[None, :]
        + gamma @ (np.log(mix.a) - np.log1p(-mix.a)).T
        - 0.5 * np.log(2 * np.pi * mix.var).sum(1)[None, :]
        - 0.5 * (mix.mu**2 / mix.var).sum(1)[None, :]
        + B @ (mix.mu / mix.var).T
        - 0.5 * (B**2) @ (1 / mix.var).T
    )


def costs(model, mix, gradient=False, augmented=False):
    out = component_costs(model, mix, gradient)
    if not augmented:
        return out
    normal = 0.5 * (
        (mix.var + mix.mu**2) / model.tau**2 - 1 - np.log(mix.var / model.tau**2)
    )
    penalty = ((1 - mix.a) * normal).sum(1)
    if not gradient:
        return out + penalty
    values, (ga, gm, gv) = out
    return values + penalty, (
        ga - mix.a * (1 - mix.a) * normal,
        gm + (1 - mix.a) * mix.mu / model.tau**2,
        gv + 0.5 * (1 - mix.a) * (mix.var / model.tau**2 - 1),
    )


def full_component_kl(mix, reference):
    bern = xlogy(mix.a, mix.a / reference.a) + xlogy(
        1 - mix.a, (1 - mix.a) / (1 - reference.a)
    )
    normal = 0.5 * (
        (mix.var + (mix.mu - reference.mu) ** 2) / reference.var
        - 1
        + np.log(reference.var / mix.var)
    )
    return np.sum(bern + normal, axis=1)


component_kl_full = full_component_kl


class Information:
    def __init__(self, reference, power=10, seed=0, source_seeds=None):
        self.reference = reference
        gs, bs, ss = [], [], []
        for k in range(reference.k):
            sd = source_seeds[k] if source_seeds is not None else source_seed(seed, k)
            u = qmc.Sobol(2 * reference.p, scramble=True, seed=sd).random_base2(power)
            gs.append((u[:, : reference.p] < reference.a[k]).astype(float))
            bs.append(
                reference.mu[k]
                + np.sqrt(reference.var[k])
                * norm.ppf(np.clip(u[:, reference.p :], 1e-14, 1 - 1e-14))
            )
            ss.append(np.full(len(u), k))
        self.gamma = np.vstack(gs)
        self.B = np.vstack(bs)
        self.beta = self.gamma * self.B
        self.source = np.concatenate(ss)
        self.measure = reference.weights[self.source] / 2**power
        self.ref = {
            False: reference.logdensity(self.gamma, self.beta),
            True: logdensity_plus(reference, self.gamma, self.B),
        }
        self.log_r = {
            key: logsumexp(ld + np.log(reference.weights), axis=1)
            for key, ld in self.ref.items()
        }

    def _overlap(self, ld, augmented):
        rows = np.arange(len(self.source))
        lr = ld[rows, self.source] - self.ref[augmented][rows, self.source]
        mass = []
        ess = []
        for k in range(self.reference.k):
            v = lr[self.source == k]
            mass.append(float(np.exp(logsumexp(v) - np.log(len(v)))))
            ess.append(
                float(np.exp(2 * logsumexp(v) - logsumexp(2 * v) - np.log(len(v))))
            )
        return dict(importance_mass=mass, relative_ess=ess)

    def overlap(self, mix):
        return dict(
            direct=self._overlap(mix.logdensity(self.gamma, self.beta), False),
            augmented=self._overlap(logdensity_plus(mix, self.gamma, self.B), True),
        )

    def evaluate(self, mix, gradient=False, augmented=False):
        ld = (
            logdensity_plus(mix, self.gamma, self.B)
            if augmented
            else mix.logdensity(self.gamma, self.beta)
        )
        lq = logsumexp(ld + np.log(mix.weights), axis=1)
        mass = (
            self.measure[:, None]
            * np.exp(ld - self.log_r[augmented][:, None])
            * mix.weights
        )
        coefficient = mass * (ld - lq[:, None])
        value = float(coefficient.sum())
        detail = dict(information=value, **self._overlap(ld, augmented))
        if not gradient:
            return value, detail
        sums = coefficient.sum(0)
        counts = self.gamma.T @ coefficient
        values = self.B if augmented else self.beta
        first = values.T @ coefficient
        second = (values**2).T @ coefficient
        gaussian_counts = (
            np.broadcast_to(sums[:, None], mix.a.shape) if augmented else counts.T
        )
        ga = counts.T - mix.a * sums[:, None]
        gm = (first.T - mix.mu * gaussian_counts) / mix.var
        gv = 0.5 * (
            (second.T - 2 * mix.mu * first.T + mix.mu**2 * gaussian_counts) / mix.var
            - gaussian_counts
        )
        gw = (coefficient - mass).sum(0) + mix.weights * (mass.sum() - value)
        return value, (gw, ga, gm, gv), detail


class Objective:
    def __init__(
        self,
        model,
        reference,
        power=10,
        seed=0,
        augmented=False,
        source_seeds=None,
        objective=None,
    ):
        self.model = model
        self.reference = reference
        self.augmented = (
            (objective == "augmented") if objective is not None else augmented
        )
        self.information = Information(reference, power, seed, source_seeds)

    def overlap(self, mix):
        return self.information.overlap(mix)

    def evaluate(self, theta, gradient=True):
        mix = Mixture.unpack(theta, self.reference.k, self.reference.p)
        out = costs(self.model, mix, gradient, self.augmented)
        values, cg = out if gradient else (out, None)
        average = float(mix.weights @ values)
        if gradient:
            info, ig, detail = self.information.evaluate(mix, True, self.augmented)
        else:
            info, detail = self.information.evaluate(mix, False, self.augmented)
        detail.update(objective=average - info, component_average=average)
        if not gradient:
            return detail
        g = [mix.weights * (values - average)] + [mix.weights[:, None] * v for v in cg]
        return (
            average - info,
            np.concatenate([(v - i).ravel() for v, i in zip(g, ig)]),
            detail,
        )


def evaluate(
    model, mix, power=14, seed=0, augmented=False, source_seeds=None, objective=None
):
    if objective is not None:
        augmented = objective == "augmented"
    average = float(mix.weights @ costs(model, mix, augmented=augmented))
    count = 2**power
    batch = 2 ** min(power, 10)
    information = 0.0
    if mix.k > 1:
        for k in range(mix.k):
            sd = source_seeds[k] if source_seeds is not None else source_seed(seed, k)
            sampler = qmc.Sobol(2 * mix.p, scramble=True, seed=sd)
            for _ in range(count // batch):
                u = sampler.random(batch)
                g = (u[:, : mix.p] < mix.a[k]).astype(float)
                B = mix.mu[k] + np.sqrt(mix.var[k]) * norm.ppf(
                    np.clip(u[:, mix.p :], 1e-14, 1 - 1e-14)
                )
                ld = (
                    logdensity_plus(mix, g, B)
                    if augmented
                    else mix.logdensity(g, g * B)
                )
                lq = logsumexp(ld + np.log(mix.weights), axis=1)
                responsibilities = np.exp(ld + np.log(mix.weights) - lq[:, None])
                information += (
                    mix.weights[k]
                    * float(np.sum(responsibilities * (ld - lq[:, None])))
                    / count
                )
    return dict(
        objective=average - information,
        component_average=average,
        information=information,
    )


def compare(
    model,
    candidate,
    incumbent,
    *,
    power=12,
    seed=0,
    repeats=3,
    tolerance=1e-4,
    multiplier=3.0,
    augmented=False,
    seeds=None,
    objective=None
):
    if objective is not None:
        augmented = objective == "augmented"
    if seeds is None:
        seeds = [source_seed(seed, r) for r in range(repeats)]
    if len(seeds) != repeats:
        raise ValueError("One seed required per repeat")
    records = []
    for sd in seeds:
        new = evaluate(model, candidate, power, sd, augmented)
        old = evaluate(model, incumbent, power, sd, augmented)
        records.append(
            dict(
                candidate=new,
                incumbent=old,
                difference=new["objective"] - old["objective"],
                seed=int(sd),
            )
        )
    d = np.array([r["difference"] for r in records])
    se = float(d.std(ddof=1) / np.sqrt(repeats)) if repeats > 1 else float("inf")
    consistent = True
    for label, mix in [("candidate", candidate), ("incumbent", incumbent)]:
        v = np.array([r[label]["information"] for r in records])
        ise = float(v.std(ddof=1) / np.sqrt(repeats)) if repeats > 1 else float("inf")
        consistent &= bool(
            v.mean() >= -max(1e-8, multiplier * ise)
            and v.mean()
            <= -mix.weights @ np.log(mix.weights) + max(1e-8, multiplier * ise)
        )
    threshold = max(tolerance, multiplier * se)
    return dict(
        difference=float(d.mean()),
        mcse=se,
        threshold=threshold,
        accepted=bool(consistent and d.mean() < -threshold),
        information_consistent=consistent,
        power=power,
        repeats=repeats,
        candidate_objective=float(
            np.mean([r["candidate"]["objective"] for r in records])
        ),
        incumbent_objective=float(
            np.mean([r["incumbent"]["objective"] for r in records])
        ),
        records=records,
        objective_kind="augmented" if augmented else "direct",
    )
