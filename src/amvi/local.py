"""Fixed-K controlled local refinement; no posterior references enter this module."""

from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .solver import Settings, Mixture, Deadline
from .objectives import Objective, compare, component_kl_full, seed_for

ARMS = ("direct_joint", "augmented_joint", "direct_frozen", "direct_stagewise")


def free_coordinates(k, p, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    if arm.endswith("_joint"):
        return np.arange(k + 3 * k * p)
    weights = np.arange(k) if arm == "direct_frozen" else np.array([k - 1])
    shapes = np.concatenate(
        [np.arange(k + b * k * p + (k - 1) * p, k + (b + 1) * k * p) for b in range(3)]
    )
    return np.r_[weights, shapes]


def check_restrictions(candidate, initial, arm):
    if candidate.k != 5 or initial.k != 5:
        raise AssertionError("Every local fit must retain K=5")
    result = dict(components=5, frozen_shapes=None, fixed_old_weight_ratios=None)
    if arm in ("direct_frozen", "direct_stagewise"):
        errors = {
            field: float(
                np.max(
                    np.abs(getattr(candidate, field)[:4] - getattr(initial, field)[:4])
                )
            )
            for field in ("a", "mu", "var")
        }
        result["old_shape_max_errors"] = errors
        if max(errors.values()) > 1e-12:
            raise AssertionError("Restricted old component shape changed")
        result["frozen_shapes"] = True
    if arm == "direct_stagewise":
        error = float(
            np.max(
                np.abs(
                    candidate.weights[:4] / candidate.weights[:4].sum()
                    - initial.weights[:4] / initial.weights[:4].sum()
                )
            )
        )
        result["old_weight_ratio_max_error"] = error
        if error > 1e-12:
            raise AssertionError("Stagewise old relative weights changed")
        result["fixed_old_weight_ratios"] = True
    return result


def fit_local(model, initial, arm, rho_index, replicate, settings=None):
    settings = settings or Settings()
    if arm not in ARMS or initial.k != 5:
        raise ValueError("Expected one authorized arm and K=5")
    began = perf_counter()
    deadline = began + settings.max_seconds
    current = initial
    history, failures = [], 0
    status = "max_refreshes"
    objective = "augmented" if arm == "augmented_joint" else "direct"
    free = free_coordinates(initial.k, initial.p, arm)
    finite_gradient = True
    for refresh in range(settings.max_refreshes):
        if perf_counter() >= deadline:
            status = "time_limit"
            break
        power = settings.training_power + min(2, failures // 2)
        training_seed = seed_for(rho_index, replicate, "training", refresh=refresh)
        cache = Objective(
            model, current, power=power, seed=training_seed, objective=objective
        )
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
            nonlocal calls, finite_gradient
            if perf_counter() >= deadline:
                raise Deadline()
            calls += 1
            theta = packed.copy()
            theta[free] += scale[free] * x
            value, gradient, _ = cache.evaluate(theta, True)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                finite_gradient = False
                raise FloatingPointError("Nonfinite training value or gradient")
            return value, gradient[free] * scale[free]

        record = dict(
            refresh=refresh,
            power=power,
            accepted=False,
            free_parameters=len(free),
            training_seed=training_seed,
        )
        try:
            result = minimize(
                fun,
                np.zeros(len(free)),
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(low[free], high[free])),
                options=dict(
                    maxiter=settings.inner_iterations, ftol=1e-10, gtol=1e-5, maxls=15
                ),
            )
        except Deadline:
            record.update(status="time_limit", evaluations=calls)
            history.append(record)
            status = "time_limit"
            break
        record.update(
            status=str(result.message),
            iterations=int(result.nit),
            evaluations=calls,
            attempts=[],
        )
        delta = np.zeros_like(packed)
        delta[free] = scale[free] * result.x
        fraction = 1.0
        for attempt in range(3):
            if perf_counter() >= deadline:
                break
            trust = False
            for reduction in range(12):
                if perf_counter() >= deadline:
                    break
                candidate = Mixture.unpack(
                    packed + fraction * delta, current.k, current.p
                )
                ckl = float(np.max(component_kl_full(candidate, current)))
                wkl = float(
                    candidate.weights @ np.log(candidate.weights / current.weights)
                )
                overlaps = cache.overlap(candidate)
                trust = (
                    ckl <= settings.max_component_kl
                    and wkl <= settings.max_component_kl
                    and all(
                        min(d["relative_ess"]) >= settings.min_relative_ess
                        and max(abs(v - 1) for v in d["importance_mass"])
                        <= settings.mass_tolerance
                        for d in overlaps.values()
                    )
                )
                if trust:
                    break
                fraction *= 0.5
            else:
                reduction = 11
            if perf_counter() >= deadline:
                break
            trial = dict(
                fraction=fraction,
                trust_ok=bool(trust),
                component_kl=ckl,
                weight_kl=wkl,
                overlap=overlaps,
                overlap_evaluations=reduction + 1,
            )
            if trust:
                seeds = [
                    seed_for(
                        rho_index,
                        replicate,
                        "local_validation",
                        refresh=refresh,
                        repeat=attempt * settings.validation_repeats + r,
                    )
                    for r in range(settings.validation_repeats)
                ]
                validation = compare(
                    model,
                    candidate,
                    current,
                    power=settings.validation_power,
                    seeds=seeds,
                    repeats=settings.validation_repeats,
                    tolerance=settings.acceptance_tol,
                    multiplier=settings.mcse_multiplier,
                    objective=objective,
                )
                trial["validation"] = validation
                trial["seeds"] = seeds
                # A validation launched before the deadline may finish afterwards, as in
                # the historical local routine. Its result remains a scientific decision.
                if validation["accepted"]:
                    trial["restriction_check"] = check_restrictions(
                        candidate, initial, arm
                    )
                    current = candidate
                    record["accepted"] = True
            record["attempts"].append(trial)
            if record["accepted"]:
                break
            fraction *= 0.5
        history.append(record)
        failures = 0 if record["accepted"] else failures + 1
        if perf_counter() >= deadline:
            status = "time_limit"
            break
        if failures >= settings.stalled_refreshes:
            status = "stalled_refreshes"
            break
    local_seconds = perf_counter() - began
    validation_began = perf_counter()
    final_seeds = [
        seed_for(rho_index, replicate, "final_validation", repeat=r)
        for r in range(settings.final_repeats)
    ]
    final = compare(
        model,
        current,
        initial,
        power=settings.final_power,
        seeds=final_seeds,
        repeats=settings.final_repeats,
        tolerance=settings.acceptance_tol,
        multiplier=settings.mcse_multiplier,
        objective=objective,
    )
    final["seeds"] = final_seeds
    fallback = not final["accepted"]
    if fallback:
        current = initial
    restriction = check_restrictions(current, initial, arm)
    return dict(
        mixture=current,
        arm=arm,
        objective=objective,
        components=current.k,
        status=status,
        history=history,
        accepted_refreshes=sum(r["accepted"] for r in history),
        final_validation=final,
        final_fallback=bool(fallback),
        returned_objective=final[
            "incumbent_objective" if fallback else "candidate_objective"
        ],
        restriction_check=restriction,
        finite_gradient=finite_gradient,
        fit_seconds=local_seconds,
        final_validation_seconds=perf_counter() - validation_began,
        local_deadline_overrun_seconds=max(0.0, local_seconds - settings.max_seconds),
        total_seconds=perf_counter() - began,
        settings=asdict(settings),
        rho_index=rho_index,
        replicate=replicate,
        certified_global_optimum=False,
    )
