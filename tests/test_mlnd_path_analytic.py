"""All-pattern algebra checks and short solver checks of analytic path cuts."""
from __future__ import annotations

import itertools
import math

import pytest

pytest.importorskip("gurobipy")

from mlnd_path.analytic_benders import AnalyticPathOracle, solve_analytic
from mlnd_path.data import Instance


@pytest.fixture(scope="module")
def fixture_instance():
    inst = Instance("analytic_cut_patterns", 1, 3, 1, [4.0], [10.0], [1.0],
                    [0.2, 0.2, 0.2], [0.6], rho_max=0.8, kappa=8.0, fleet_cost=2.0)
    inst.service = [
        dict(od=0, l=0, m=0, i=0, j=1, time=0.25, cost=2.0, swaps=(1, 0)),
        dict(od=0, l=0, m=0, i=0, j=1, time=0.35, cost=0.5, swaps=(1, 1)),
        dict(od=0, l=0, m=0, i=1, j=2, time=0.25, cost=1.0, swaps=(1, 0)),
    ]
    return inst


def _required(inst, column):
    keys = {f"X{column['l']}", f"X{column['m']}", f"Y{column['i']}", f"Y{column['j']}"}
    keys.update(inst.station_key(k) for k, count in enumerate(column["swaps"]) if count)
    return keys


def _patterns(inst):
    keys = [key for key in inst.design_keys() if not key.startswith("B")]
    for values in itertools.product((0.0, 1.0), repeat=len(keys)):
        point = inst.zero_design()
        point.update(zip(keys, values))
        yield point


def _cheapest_available(inst, point):
    return min((inst.service_cost(p) for p, col in enumerate(inst.service)
                if all(point[key] for key in _required(inst, col))), default=math.inf)


def test_oracle_algebra_all_pattern_cost_cuts_and_same_pattern_tightness(fixture_instance):
    """Pure Python: no Gurobi model or optimizer is constructed."""
    inst = fixture_instance
    oracle = AnalyticPathOracle(inst)
    patterns = list(_patterns(inst))
    for origin in patterns:
        answer, _ = oracle.evaluate(origin)
        group = answer["groups"][0]
        if not group["feasible"]:
            continue
        assert group["q"] == _cheapest_available(inst, origin)
        assert answer["full_candidate_certificate"]
        for p, column in enumerate(inst.service):
            rc = inst.service_cost(p)-group["q"]+sum(
                group["penalties"].get(key, 0.0) for key in _required(inst, column))
            assert rc >= -1e-10
        for amount in (0.0, 0.1, 1.4, 3.2):
            at_origin = (group["q"]*amount-group["maximum"]*sum(
                penalty*origin[key] for key, penalty in group["penalties"].items())-group["reserve"])
            assert at_origin == pytest.approx(group["q"]*amount, abs=1e-7)
            for target in patterns:
                cost = _cheapest_available(inst, target)
                if math.isinf(cost):
                    if amount:
                        continue  # Positive flow is infeasible at this pattern.
                    actual = 0.0
                else:
                    actual = cost*amount
                lower = (group["q"]*amount-group["maximum"]*sum(
                    penalty*target[key] for key, penalty in group["penalties"].items())-group["reserve"])
                assert lower <= actual+1e-9


def test_oracle_algebra_unavailable_cut_hits_every_candidate(fixture_instance):
    """Pure Python: every missing-path cut covers the full candidate set."""
    inst = fixture_instance
    oracle = AnalyticPathOracle(inst)
    patterns = list(_patterns(inst))
    for origin in patterns:
        answer, _ = oracle.evaluate(origin)
        group = answer["groups"][0]
        if group["feasible"]:
            continue
        cover = set(group["cover"])
        assert cover and all(origin[key] == 0.0 for key in cover)
        assert all(cover & _required(inst, col) for col in inst.service)
        for target in patterns:
            if math.isfinite(_cheapest_available(inst, target)):
                assert group["maximum"] <= group["maximum"]*sum(target[key] for key in cover)+1e-9


def test_oracle_algebra_cache_depends_on_facilities_not_assignment(fixture_instance):
    inst = fixture_instance
    oracle = AnalyticPathOracle(inst)
    point = dict.fromkeys(inst.design_keys(), 1.0)
    first, hit = oracle.evaluate(point)
    assert not hit
    point["B0:0:0"] = 0.0
    second, hit = oracle.evaluate(point)
    assert hit and first is second
    assert len(oracle.cache) == 1


def test_analytic_solver_matches_enumerated_single_base_optimum(fixture_instance):
    inst = fixture_instance
    reference = inst.c0
    for point in _patterns(inst):
        cost = _cheapest_available(inst, point)
        if math.isinf(cost):
            continue
        slope = inst.demand[0]*cost
        rho = 0.0 if slope >= -inst.fleet_cost else min(
            inst.rho_max, 1.0-math.sqrt(inst.fleet_cost/(-slope)))
        reference = min(reference, inst.fixed_cost(point)+inst.c0+slope*rho+
                        inst.fleet_cost*rho/(1.0-rho))
    result = solve_analytic(inst, time_limit=2.0, mip_gap=1e-6)
    assert result["status"] == "OPTIMAL_TOL", result
    assert result["validated_incumbent"]
    assert result["objective"] == pytest.approx(reference, abs=2e-5)
    assert result["bound"] <= reference+1e-6


def test_analytic_solver_matches_full_geometric_instance():
    from mlnd_path.instances import make_instance
    from mlnd_path.solver import solve_full

    inst = make_instance(2, 3, 2, 1, 17)
    reference = solve_full(inst, time_limit=1.0, mip_gap=1e-6)
    result = solve_analytic(inst, time_limit=2.0, mip_gap=1e-6)
    assert reference["status"] == "optimal"
    assert result["status"] == "OPTIMAL_TOL", result
    assert result["objective"] == pytest.approx(reference["objective"], abs=2e-5)
    assert result["bound"] <= reference["objective"]+1e-6
    assert result["primal_residual"] <= 1e-5


def test_analytic_deadline_does_not_promote_unvalidated_master_point(fixture_instance):
    result = solve_analytic(fixture_instance, time_limit=0.0)
    assert result["status"] == "TIME_LIMIT"
    assert result["objective"] == fixture_instance.c0
    assert result["validated_incumbent"]
    assert result["theta"] == [0.0]*len(fixture_instance.service)
