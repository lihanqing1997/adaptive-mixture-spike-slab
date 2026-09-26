"""Unpenalized adaptive spike-and-slab mixture VI.

Uses the frozen, tested mixed-law representation and analytic regression cost.
No TC/marginal-quadrature code is called. Statistical acceptance is empirical.
"""

from dataclasses import dataclass, asdict
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm, qmc

from ._mixed_law import (
    RegressionModel,
    Mixture,
    component_costs,
    component_kl,
    make_joint_cache,
    meanfield,
)


@dataclass
class Settings:
    max_components: int = 5
    proposals_per_stage: int = 3
    max_seconds: float = 60.0
    proposal_seconds: float = 20.0
    max_refreshes: int = 16
    inner_iterations: int = 25
    training_power: int = 10
    validation_power: int = 12
    final_power: int = 14
    validation_repeats: int = 3
    final_repeats: int = 4
    acceptance_tol: float = 1e-4
    mcse_multiplier: float = 3.0
    max_component_kl: float = 0.25
    min_relative_ess: float = 0.5
    mass_tolerance: float = 0.05
    stalled_refreshes: int = 4


class Deadline(Exception):
    pass


class Information:
    """Fixed-mixture importance rule with analytically integrated labels.

    The mixture proposal r and its stratified weights are fixed. Integrate
    sum_k w_k f_k/r log(f_k/q). This gives exactly differentiated finite sums.
    """

    def __init__(self, reference, power=10, seed=0):
        self.reference = reference
        raw = make_joint_cache(reference, power=power, seed=seed)
        self.gamma, self.beta, self.source = raw.gamma, raw.beta, raw.source
        self.beta2 = self.beta**2
        self.measure = raw.measure * reference.weights[self.source]
        ld = reference.logdensity(self.gamma, self.beta)
        self.log_r = logsumexp(ld + np.log(reference.weights), axis=1)
        self.ref_source = ld[np.arange(len(ld)), self.source]

    def evaluate(self, mix, gradient=False):
        ld = mix.logdensity(self.gamma, self.beta)
        lq = logsumexp(ld + np.log(mix.weights), axis=1)
        ratios = np.exp(ld - self.log_r[:, None])
        mass = self.measure[:, None] * ratios * mix.weights
        centered = ld - lq[:, None]
        coefficient = mass * centered
        value = float(coefficient.sum())
        imass, ess = [], []
        source_ratio = np.exp(ld[np.arange(len(ld)), self.source] - self.ref_source)
        for k in range(mix.k):
            v = source_ratio[self.source == k]
            imass.append(float(v.mean()))
            ess.append(float(v.sum() ** 2 / (len(v) * np.sum(v * v))))
        detail = dict(information=value, importance_mass=imass, relative_ess=ess)
        if not gradient:
            return value, detail
        sums = coefficient.sum(0)
        counts = self.gamma.T @ coefficient
        first = self.beta.T @ coefficient
        second = self.beta2.T @ coefficient
        ga = counts.T - mix.a * sums[:, None]
        gm = (first.T - mix.mu * counts.T) / mix.var
        gv = 0.5 * (
            (second.T - 2 * mix.mu * first.T + mix.mu**2 * counts.T) / mix.var
            - counts.T
        )
        gw = (coefficient - mass).sum(0) + mix.weights * (mass.sum() - value)
        return value, (gw, ga, gm, gv), detail


class Objective:
    def __init__(self, model, reference, power=10, seed=0):
        self.model, self.reference = model, reference
        self.information = (
            None if reference.k == 1 else Information(reference, power, seed)
        )

    def evaluate(self, theta, gradient=True):
        mix = Mixture.unpack(theta, self.reference.k, self.reference.p)
        if gradient:
            costs, cg = component_costs(self.model, mix, True)
        else:
            costs = component_costs(self.model, mix)
        average = float(mix.weights @ costs)
        if self.information is None:
            info = 0.0
            detail = dict(information=0.0, importance_mass=[1.0], relative_ess=[1.0])
            if gradient:
                ig = (np.zeros(1),) + tuple(np.zeros_like(mix.a) for _ in range(3))
        elif gradient:
            info, ig, detail = self.information.evaluate(mix, True)
        else:
            info, detail = self.information.evaluate(mix)
        detail.update(objective=average - info, component_average=average)
        if not gradient:
            return detail
        g = [mix.weights * (costs - average)] + [mix.weights[:, None] * v for v in cg]
        return (
            average - info,
            np.concatenate([(v - i).ravel() for v, i in zip(g, ig)]),
            detail,
        )


def evaluate(model, mix, power=14, seed=0):
    # Fresh direct integration is batched: final validation need not store N*K*p draws.
    average = float(mix.weights @ component_costs(model, mix))
    if mix.k == 1:
        return dict(objective=average, component_average=average, information=0.0)
    count = 2**power
    batch = 2 ** min(power, 10)
    information = 0.0
    for k in range(mix.k):
        sampler = qmc.Sobol(2 * mix.p, scramble=True, seed=seed + 104729 * k)
        for _ in range(count // batch):
            u = sampler.random(batch)
            gamma = (u[:, : mix.p] < mix.a[k]).astype(float)
            beta = gamma * (
                mix.mu[k]
                + np.sqrt(mix.var[k])
                * norm.ppf(np.clip(u[:, mix.p :], 1e-14, 1 - 1e-14))
            )
            ld = mix.logdensity(gamma, beta)
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
    multiplier=3.0
):
    records = []
    for rep in range(repeats):
        sd = seed + 100003 * rep
        new = evaluate(model, candidate, power, sd)
        old = evaluate(model, incumbent, power, sd)
        records.append(
            dict(
                candidate=new,
                incumbent=old,
                difference=new["objective"] - old["objective"],
            )
        )
    d = np.array([r["difference"] for r in records])
    se = float(d.std(ddof=1) / np.sqrt(repeats)) if repeats > 1 else float("inf")
    consistent = True
    for label, mix in [("candidate", candidate), ("incumbent", incumbent)]:
        v = np.array([r[label]["information"] for r in records])
        ise = float(v.std(ddof=1) / np.sqrt(repeats)) if repeats > 1 else float("inf")
        entropy = float(-mix.weights @ np.log(mix.weights))
        consistent &= bool(
            v.mean() >= -max(1e-8, multiplier * ise)
            and v.mean() <= entropy + max(1e-8, multiplier * ise)
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
    )


def best_meanfield(model):
    starts = [meanfield(model, start=i) for i in (0, 1)]
    best = min(starts, key=lambda r: r["objective"])
    best["starts"] = [
        {k: v for k, v in r.items() if k not in ("mixture", "starts")} for r in starts
    ]
    return best


def _split(model, current, index):
    """A global component split; a small subproblem only proposes a direction."""
    k = int(np.argmax(current.weights))
    a, mu, var = current.a[k], current.mu[k], current.var[k]
    selected = np.argsort(-a, kind="stable")[: min(20, model.p)]
    xs = model.X[:, selected]
    precision = xs.T @ xs / model.sigma**2 + np.eye(len(selected)) / model.tau**2
    covariance = np.linalg.solve(precision, np.eye(len(selected)))
    delta = covariance - np.diag(var[selected])
    values, vectors = np.linalg.eigh(delta)
    amplitude = 0.5 if index % 2 == 0 else 1.0
    direction = np.zeros(model.p)
    direction[selected] = (
        amplitude
        * np.sqrt(max(values[-1], 0.05 * np.mean(var[selected])))
        * vectors[:, -1]
    )
    aa = np.vstack([current.a, a])
    mm = np.vstack([current.mu, mu])
    vv = np.vstack([current.var, var])
    mm[k] -= direction
    mm[-1] += direction
    weights = np.r_[current.weights, current.weights[k] / 2]
    weights[k] /= 2
    return Mixture(weights, aa, mm, vv), dict(
        kind="symmetric_covariance_split",
        source=k,
        direction_coordinates=selected.tolist(),
        direction_scale=amplitude,
    )


def _alternative(model, current, index, seed):
    """Release a temporary tilt before using the resulting CAVI component."""
    k = int(np.argmax(current.weights))
    a = current.a[k]
    residual = model.y - model.X @ (a * current.mu[k])
    score = np.abs(model.X.T @ residual) / np.sqrt(np.maximum(model.diag, 1e-12))
    score *= 1 - a
    order = np.argsort(-score, kind="stable")
    j = int(order[index % min(5, model.p)])
    sign = np.sign(model.X[:, j] @ residual) or 1.0
    tilt = np.zeros(model.p)
    tilt[j] = sign * 2 * np.sqrt(model.diag[j] + model.tau**-2)
    temporary = meanfield(model, start=1, tilt=tilt, maxiter=100)
    # Warm-start a standard, untilted CAVI sweep; no forced support remains.
    c = temporary["mixture"]
    alpha = c.a[0].copy()
    means = c.mu[0].copy()
    v = 1 / (model.diag + model.tau**-2)
    m = alpha * means
    r = model.y - model.X @ m
    for _ in range(300):
        old = np.r_[alpha, m]
        for t in range(model.p):
            means[t] = v[t] * (
                model.X[:, t] @ r / model.sigma**2 + model.diag[t] * m[t]
            )
            alpha[t] = np.clip(
                expit(
                    logit(model.omega)
                    + 0.5 * (np.log(v[t] / model.tau**2) + means[t] ** 2 / v[t])
                ),
                1e-10,
                1 - 1e-10,
            )
            new = alpha[t] * means[t]
            r -= model.X[:, t] * (new - m[t])
            m[t] = new
        if np.max(np.abs(np.r_[alpha, m] - old)) < 1e-8:
            break
    return Mixture(
        np.r_[0.9 * current.weights, 0.1],
        np.vstack([current.a, alpha]),
        np.vstack([current.mu, means]),
        np.vstack([current.var, v]),
    ), dict(kind="released_tilt_cavi", coordinate=j)


def refine(model, initial, settings, seed, deadline):
    current = initial
    trace = []
    failures = 0
    for refresh in range(settings.max_refreshes):
        if perf_counter() >= deadline:
            break
        power = settings.training_power + min(2, failures // 2)
        cache = Objective(model, current, power, seed + 1009 * refresh)
        packed = current.pack()
        scale = np.r_[
            np.ones(current.k + current.k * current.p),
            np.sqrt(current.var).ravel(),
            np.ones(current.k * current.p),
        ]
        width = np.r_[
            np.ones(current.k),
            np.full(current.k * current.p, 0.5),
            np.full(current.k * current.p, 0.5),
            np.full(current.k * current.p, 0.4),
        ]
        low, high = -width, width.copy()
        lo, hi = current.k, current.k + current.k * current.p
        low[lo:hi] = np.maximum(low[lo:hi], -23 - packed[lo:hi])
        high[lo:hi] = np.minimum(high[lo:hi], 23 - packed[lo:hi])
        calls = 0

        def fun(x):
            nonlocal calls
            if perf_counter() >= deadline:
                raise Deadline()
            calls += 1
            value, grad, _ = cache.evaluate(packed + scale * x, True)
            return value, grad * scale

        record = dict(refresh=refresh, power=power, accepted=False)
        try:
            result = minimize(
                fun,
                np.zeros_like(packed),
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(low, high)),
                options=dict(
                    maxiter=settings.inner_iterations, ftol=1e-10, gtol=1e-5, maxls=15
                ),
            )
        except Deadline:
            record.update(status="time_limit", evaluations=calls)
            trace.append(record)
            break
        record.update(
            status=str(result.message),
            iterations=int(result.nit),
            evaluations=calls,
            attempts=[],
        )
        delta = scale * result.x
        fraction = 1.0
        for attempt in range(3):
            if perf_counter() >= deadline:
                break
            trust = False
            for _ in range(12):
                candidate = Mixture.unpack(
                    packed + fraction * delta, current.k, current.p
                )
                detail = cache.evaluate(candidate.pack(), False)
                ckl = float(np.max(component_kl(candidate, current)))
                wkl = float(
                    candidate.weights @ np.log(candidate.weights / current.weights)
                )
                trust = (
                    ckl <= settings.max_component_kl
                    and wkl <= settings.max_component_kl
                    and min(detail["relative_ess"]) >= settings.min_relative_ess
                    and max(abs(v - 1) for v in detail["importance_mass"])
                    <= settings.mass_tolerance
                )
                if trust:
                    break
                fraction *= 0.5
            trial = dict(
                fraction=fraction, trust_ok=bool(trust), component_kl=ckl, weight_kl=wkl
            )
            if trust and perf_counter() < deadline:
                validation = compare(
                    model,
                    candidate,
                    current,
                    power=settings.validation_power,
                    seed=seed + 5000009 + 1009 * refresh,
                    repeats=settings.validation_repeats,
                    tolerance=settings.acceptance_tol,
                    multiplier=settings.mcse_multiplier,
                )
                trial["validation"] = validation
                if validation["accepted"]:
                    current = candidate
                    record["accepted"] = True
            record["attempts"].append(trial)
            if record["accepted"]:
                break
            fraction *= 0.5
        trace.append(record)
        failures = 0 if record["accepted"] else failures + 1
        if failures >= settings.stalled_refreshes:
            break
    return current, trace


def fit(model, settings=None, seed=0, baseline=None):
    settings = settings or Settings()
    if (
        settings.max_components < 1
        or settings.validation_repeats < 2
        or settings.final_repeats < 2
    ):
        raise ValueError("At least one component and two integration repeats required")
    began = perf_counter()
    baseline = best_meanfield(model) if baseline is None else baseline
    baseline_mix = baseline["mixture"]
    current = baseline_mix
    baseline_seconds = perf_counter() - began
    deadline = began + settings.max_seconds
    history = []
    status = "component_cap"
    for stage in range(1, settings.max_components):
        if perf_counter() >= deadline:
            status = "time_limit"
            break
        accepted = False
        for number in range(settings.proposals_per_stage):
            if perf_counter() >= deadline:
                status = "time_limit"
                break
            proposal_began = perf_counter()
            proposal, detail = (
                _split(model, current, number)
                if number < 2
                else _alternative(model, current, number - 2, seed + stage)
            )
            trial, trace = refine(
                model,
                proposal,
                settings,
                seed + stage * 1000003 + number * 70001,
                min(deadline, perf_counter() + settings.proposal_seconds),
            )
            # Acceptance is against the incumbent before birth, not the new initializer.
            comparison = compare(
                model,
                trial,
                current,
                power=settings.validation_power + 1,
                seed=seed + 80000009 + stage * 100003 + number * 1009,
                repeats=settings.validation_repeats,
                tolerance=settings.acceptance_tol,
                multiplier=settings.mcse_multiplier,
            )
            record = dict(
                stage=stage,
                proposal=detail,
                refinement=trace,
                validation=comparison,
                accepted=comparison["accepted"],
                seconds=perf_counter() - proposal_began,
            )
            history.append(record)
            if comparison["accepted"]:
                current = trial
                accepted = True
                break
        if not accepted:
            status = (
                "time_limit" if perf_counter() >= deadline else "no_supported_proposal"
            )
            break
    fit_seconds = perf_counter() - began
    final = compare(
        model,
        current,
        baseline_mix,
        power=settings.final_power,
        seed=seed + 170000009,
        repeats=settings.final_repeats,
        tolerance=settings.acceptance_tol,
        multiplier=settings.mcse_multiplier,
    )
    fallback = bool(current.k > 1 and not final["accepted"])
    candidate_components = current.k
    if fallback:
        current = baseline_mix
    returned_objective = (
        float(baseline["objective"]) if current.k == 1 else final["candidate_objective"]
    )
    return dict(
        mixture=current,
        baseline=baseline_mix,
        baseline_objective=float(baseline["objective"]),
        baseline_status=baseline["status"],
        candidate_components=candidate_components,
        components=current.k,
        final_fallback=fallback,
        status=status,
        history=history,
        final_validation=final,
        returned_objective=returned_objective,
        returned_delta_mf=returned_objective - float(baseline["objective"]),
        returned_mcse=0.0 if current.k == 1 else final["mcse"],
        baseline_seconds=baseline_seconds,
        fit_seconds=fit_seconds,
        total_seconds=perf_counter() - began,
        settings=asdict(settings),
        seed=seed,
        certified_global_optimum=False,
    )
