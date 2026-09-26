"""Small deterministic restriction and fixed-candidate wrapper checks."""

import json
import numpy as np
from amvi.local import (
    ARMS,
    Settings,
    Mixture,
    fit_local,
    free_coordinates,
    check_restrictions,
)
from amvi.objectives import Objective, RegressionModel


def run_checks():
    rng = np.random.default_rng(83012)
    model = RegressionModel(rng.normal(size=(8, 2)), rng.normal(size=8), omega=0.5)
    initial = Mixture(
        np.array([0.1, 0.2, 0.15, 0.45, 0.1]),
        rng.uniform(0.15, 0.85, (5, 2)),
        rng.normal(size=(5, 2)) * 0.3,
        rng.uniform(0.5, 1.2, (5, 2)),
    )
    records = {}
    for arm in ARMS:
        free = free_coordinates(5, 2, arm)
        theta = initial.pack()
        changed = theta.copy()
        changed[free] += rng.normal(size=len(free)) * 0.03
        restriction = check_restrictions(Mixture.unpack(changed, 5, 2), initial, arm)
        objective = Objective(
            model,
            initial,
            power=7,
            seed=905,
            objective="augmented" if arm == "augmented_joint" else "direct",
        )
        _, grad, _ = objective.evaluate(changed)
        h = 2e-5
        fd = []
        for coordinate in free:
            plus, minus = changed.copy(), changed.copy()
            plus[coordinate] += h
            minus[coordinate] -= h
            fd.append(
                (
                    objective.evaluate(plus, False)["objective"]
                    - objective.evaluate(minus, False)["objective"]
                )
                / (2 * h)
            )
        error = float(np.max(np.abs(np.array(fd) - grad[free])))
        assert error < 2e-6, (arm, error)
        # Bounded smoke refinement: real optimization/acceptance, deliberately tiny
        # integration resolution only in this check, never production settings.
        settings = Settings(
            max_refreshes=2,
            inner_iterations=3,
            training_power=7,
            validation_power=7,
            final_power=8,
            max_seconds=10,
        )
        result = fit_local(model, initial, arm, 0, 0, settings)
        final_restriction = check_restrictions(result["mixture"], initial, arm)
        for refresh in result["history"]:
            for attempt in refresh.get("attempts", []):
                if attempt.get("validation", {}).get("accepted"):
                    assert "restriction_check" in attempt
        records[arm] = dict(
            restricted_gradient_max_absolute_error=error,
            changed_coordinate_restrictions=restriction,
            fit_restrictions=final_restriction,
            accepted_refreshes=result["accepted_refreshes"],
            final_fallback=result["final_fallback"],
            status=result["status"],
        )
    # Zero allowance tests the explicit common-candidate fallback and K retention.
    result = fit_local(
        model, initial, "augmented_joint", 0, 1, Settings(max_seconds=0, final_power=7)
    )
    assert result["final_fallback"] and result["mixture"] is initial
    assert result["status"] == "time_limit" and not result["history"]
    records["zero_allowance"] = dict(
        status=result["status"],
        initial_returned=True,
        final_objective=result["final_validation"]["objective_kind"],
    )
    return dict(passed=True, checks=records)


def test_numerical_checks():
    assert run_checks()["passed"]
