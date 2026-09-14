"""Idle tangent validity, exact repaired upper bounds, and OA solver checks."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("gurobipy")

from mlnd_path.analytic_oa import solve_analytic_oa
from mlnd_path.data import Instance


@pytest.fixture(scope="module")
def instance():
    inst = Instance("oa_analytic", 1, 2, 1, [4.0], [10.0], [1.0],
                    [0.2, 0.2], [0.6], rho_max=0.8, kappa=8.0, fleet_cost=2.0)
    inst.service = [
        dict(od=0, l=0, m=0, i=0, j=1, time=0.25, cost=2.0, swaps=(1, 0)),
        dict(od=0, l=0, m=0, i=0, j=1, time=0.35, cost=0.5, swaps=(1, 1)),
    ]
    return inst


def test_queue_tangent_is_a_global_lower_bound_by_exact_gap_identity():
    # g(r)-[g(a)+g'(a)(r-a)] = (r-a)^2 / [(1-r)(1-a)^2].
    for a in (0.0, 0.03, 0.2, 0.41, 0.7, 0.9):
        for r in np.linspace(0.0, 0.95, 40):
            tangent = a/(1.0-a)+(r-a)/(1.0-a)**2
            actual_gap = r/(1.0-r)-tangent
            identity = (r-a)**2/((1.0-r)*(1.0-a)**2)
            assert actual_gap == pytest.approx(identity, abs=2e-12)
            assert actual_gap >= -1e-12


def test_exact_idle_repair_satisfies_all_future_queue_tangents():
    for r in (0.01, 0.137, 0.37, 0.719, 0.8):
        relaxed_idle = r  # Tangent at zero, strictly below g(r) when r>0.
        repaired = max(relaxed_idle, r/(1.0-r))
        assert repaired > relaxed_idle
        for a in np.linspace(0.0, 0.9, 50):
            assert repaired >= a/(1.0-a)+(r-a)/(1.0-a)**2-1e-12
        # These are also the complete auxiliary-variable values supplied to
        # the MILP incumbent: up-I=1, low+rho=1, one=1.
        up, low, one = 1.0+repaired, 1.0-r, 1.0
        assert up-repaired == pytest.approx(1.0)
        assert low+r == pytest.approx(1.0)
        assert up*low >= one**2-1e-12


@pytest.mark.parametrize("inject", [False, True])
def test_oa_continuous_rho_matches_analytic_optimum_and_true_cost(instance, inject):
    rho = 1.0-math.sqrt(2.0/35.2)
    reference = 2.0+40.0-35.2*rho+2.0*rho/(1.0-rho)
    result = solve_analytic_oa(instance, time_limit=2.0, mip_gap=1e-6,
                               initial_tangents=(0.0,), inject_incumbent=inject)
    assert result["status"] == "OPTIMAL_TOL", result
    assert result["master_is_milp"]
    assert result["validated_incumbent"]
    assert result["objective"] == pytest.approx(reference, abs=3e-5)
    assert result["bound"] <= reference+1e-6
    assert result["rho"][0] == pytest.approx(rho, abs=3e-4)
    # The optimal availability lies between the supplied tangent locations.
    assert min(abs(result["rho"][0]-a) for a in result["initial_tangent_points"]) > 0.01
    assert result["stats"]["queue_cuts"] > 0
    assert result["stats"]["maximum_idle_repair"] > 0.1
    assert result["idle"][0] >= result["rho"][0]/(1.0-result["rho"][0])-1e-9
    actual_cost = (instance.fixed_cost(result["design"])+instance.c0+
                   instance.fleet_cost*sum(result["idle"])+sum(
                       instance.service_cost(p)*flow for p, flow in enumerate(result["theta"])))
    assert result["objective"] == pytest.approx(actual_cost, abs=1e-8)
    assert not result["stats"]["incumbent_injection_errors"]
    if inject:
        assert result["stats"]["incumbent_injection_requests"] > 0
    else:
        assert result["stats"]["incumbent_injection_requests"] == 0


def test_oa_injection_switch_preserves_two_base_empty_return_optimum():
    from scipy.optimize import minimize_scalar

    inst = Instance("oa_empty_return", 2, 2, 0, [4.0], [10.0],
                    [1.0, 1.0], [0.2, 0.2], [], rho_max=0.8, kappa=8.0, fleet_cost=2.0)
    inst.service = [dict(od=0, l=0, m=1, i=0, j=1, time=0.25, cost=2.0, swaps=(0, 1))]
    inst.empty = [dict(l=1, m=0, time=0.1, cost=0.1, swaps=(1, 0))]
    # phi=theta=4*rho0 and rho1=rho0/2, from balance and the finite clock.
    reference = minimize_scalar(
        lambda r: 42.4-28.8*r+2.0*r/(1.0-r)+r/(1.0-r/2.0),
        bounds=(0.0, 0.8), method="bounded", options={"xatol": 1e-12})
    results = []
    for inject in (False, True):
        result = solve_analytic_oa(inst, time_limit=2.0, mip_gap=1e-6,
                                   initial_tangents=(0.0,), inject_incumbent=inject)
        assert result["status"] == "OPTIMAL_TOL", result
        assert result["objective"] == pytest.approx(reference.fun, abs=3e-5)
        assert result["bound"] <= reference.fun+1e-6
        assert result["phi"][0] > 1.0
        assert result["phi"][0] == pytest.approx(result["theta"][0], abs=1e-6)
        assert result["rho"][1] == pytest.approx(result["rho"][0]/2.0, abs=2e-5)
        assert not result["stats"]["incumbent_injection_errors"]
        results.append(result)
    assert results[0]["objective"] == pytest.approx(results[1]["objective"], abs=3e-5)


def test_oa_zero_budget_returns_validated_all_ground_incumbent(instance):
    result = solve_analytic_oa(instance, time_limit=0.0, inject_incumbent=True)
    assert result["status"] == "TIME_LIMIT"
    assert result["objective"] == instance.c0
    assert result["validated_incumbent"]
    assert result["master_is_milp"]
    assert result["theta"] == [0.0]*len(instance.service)
    assert result["stats"]["incumbent_injection_requests"] == 0
