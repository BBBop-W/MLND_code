"""Independent short checks of the conic-master / LP-path decomposition."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

pytest.importorskip("gurobipy")

from mlnd_path.data import Instance
from mlnd_path.partial_benders import PathAllocationLP, solve_partial
from mlnd_path.solver import solve_full


@pytest.fixture(scope="module")
def instance():
    result = Instance("partial_analytic", 1, 2, 1, [4.0], [10.0], [1.0],
                      [0.2, 0.2], [0.6], rho_max=0.8, kappa=8.0, fleet_cost=2.0)
    result.service = [
        dict(od=0, l=0, m=0, i=0, j=1, time=0.25, cost=2.0, swaps=(1, 0)),
        dict(od=0, l=0, m=0, i=0, j=1, time=0.35, cost=0.5, swaps=(1, 1)),
    ]
    return result


def _point(rho, swap=1.0, second_hub=1.0):
    return dict(X0=1.0, Y0=1.0, Y1=second_hub, S1=swap,
                **{"B0:0:0": 1.0, "T0:0:0": 4.0*rho})


def _cut_value(certificate, point):
    return certificate["intercept"]+sum(c*point[key] for key, c in certificate["slope"].items())


def _lp_reference(point):
    """Two-column LP solved directly, including a fractional swap gate."""
    total = point["T0:0:0"]
    if total > 3.2*min(point["X0"], point["Y0"], point["Y1"])+1e-10:
        return math.inf
    cheap = min(total, 3.2*point["S1"])
    return -7.5*total-1.3*cheap


def test_lp_full_and_cg_match_direct_reference_and_global_optimality_cuts(instance):
    points = [_point(rho, swap) for rho, swap in
              itertools.product((0.0, 0.15, 0.45, 0.75), (0.0, 0.25, 1.0))]
    for use_cg in (False, True):
        allocator = PathAllocationLP(instance, use_cg=use_cg)
        try:
            for point in points:
                answer = allocator.solve(point, time_limit=1.0)
                assert answer["complete"] and answer["feasible"], answer
                assert answer["cut_type"] == "optimality"
                assert answer["objective"] == pytest.approx(_lp_reference(point), abs=1e-6)
                assert len(answer["certificate"]["rcS"]) == len(instance.service)
                for target in points:
                    assert _cut_value(answer["certificate"], target) <= _lp_reference(target)+1e-7
        finally:
            allocator.dispose()


def test_phase_one_detects_infeasible_fixed_flow_and_returns_correct_cut_sign(instance):
    allocator = PathAllocationLP(instance, use_cg=True)
    try:
        infeasible = _point(0.5, swap=1.0, second_hub=0.0)
        answer = allocator.solve(infeasible, time_limit=1.0)
        assert answer["complete"] and answer["cut_valid"]
        assert not answer["feasible"]
        assert answer["cut_type"] == "feasibility"
        assert answer["phase1_objective"] == pytest.approx(2.0, abs=1e-7)
        assert _cut_value(answer["certificate"], infeasible) > 1.0
        feasible_points = [_point(rho, swap) for rho, swap in
                           itertools.product((0.0, 0.15, 0.45, 0.75), (0.0, 1.0))]
        feasible_points.append(_point(0.0, second_hub=0.0))
        for target in feasible_points:
            # The feasibility inequality is L(point) <= 0.
            assert _cut_value(answer["certificate"], target) <= 1e-7
        # Reusing the same LP after an infeasible point must restore phase-II
        # costs and remove all artificial unmet flow from the returned primal.
        restored = allocator.solve(_point(0.5), time_limit=1.0)
        assert restored["complete"] and restored["feasible"]
        assert restored["objective"] == pytest.approx(-17.6, abs=1e-6)
    finally:
        allocator.dispose()


def test_phase_one_clips_artificial_variable_dual_row():
    empty = Instance("no_path", 1, 2, 0, [4.0], [10.0], [1.0], [0.2, 0.2], [],
                     rho_max=0.8, kappa=8.0, fleet_cost=2.0)
    allocator = PathAllocationLP(empty)
    try:
        prices = np.zeros(len(allocator.names))
        prices[0] = 1.0001  # Slight violation of the artificial u's dual row.
        certificate = allocator._certificate(prices, phase=1)
        point = empty.zero_design()
        point["T0:0:0"] = 2.0
        assert _cut_value(certificate, point) <= 2.0
        assert certificate["slope"]["T0:0:0"] <= 1.0
    finally:
        allocator.dispose()


@pytest.mark.parametrize("use_cg", [False, True])
def test_partial_matches_analytic_integer_optimum_and_full_model(instance, use_cg):
    # The best design opens all facilities. At rho*=1-sqrt(2/35.2), the
    # discounted path component is negative and must be allowed below zero.
    rho = 1.0-math.sqrt(2.0/35.2)
    expected = 2.0+40.0-35.2*rho+2.0*rho/(1.0-rho)
    result = solve_partial(instance, time_limit=2.0, mip_gap=1e-6, use_cg=use_cg)
    assert result["status"] == "OPTIMAL_TOL", result
    assert result["validated_incumbent"]
    assert result["objective"] == pytest.approx(expected, abs=2e-5)
    assert result["bound"] <= expected+1e-6
    assert result["rho"][0] == pytest.approx(rho, abs=3e-4)
    assert result["idle"][0] >= result["rho"][0]/(1.0-result["rho"][0])-2e-6
    full = solve_full(instance, time_limit=1.0, mip_gap=1e-6)
    assert full["status"] == "optimal"
    assert result["objective"] == pytest.approx(full["objective"], abs=2e-5)


def test_partial_deadline_returns_only_validated_all_ground_incumbent(instance):
    result = solve_partial(instance, time_limit=0.0)
    assert result["status"] == "TIME_LIMIT"
    assert result["objective"] == instance.c0
    assert result["validated_incumbent"]
    assert result["theta"] == [0.0]*len(instance.service)
    assert result["bound"] <= result["objective"]


def test_partial_cross_base_flow_matches_physical_full_model():
    inst = Instance("partial_empty_return", 2, 2, 0, [4.0], [10.0],
                    [1.0, 1.0], [0.2, 0.2], [], rho_max=0.8, kappa=8.0, fleet_cost=2.0)
    inst.service = [dict(od=0, l=0, m=1, i=0, j=1, time=0.25, cost=2.0, swaps=(0, 1))]
    inst.empty = [dict(l=1, m=0, time=0.1, cost=0.1, swaps=(1, 0))]
    result = solve_partial(inst, time_limit=2.0, mip_gap=1e-6, use_cg=True)
    full = solve_full(inst, time_limit=1.0, mip_gap=1e-6)
    assert result["status"] == "OPTIMAL_TOL", result
    assert full["status"] == "optimal"
    assert result["objective"] == pytest.approx(full["objective"], abs=2e-5)
    assert result["phi"][0] > 1.0
    assert result["phi"][0] == pytest.approx(result["theta"][0], abs=1e-6)
    assert result["rho"][1] == pytest.approx(result["rho"][0]/2.0, abs=2e-5)
