"""Safe design proposals and fixed-flow improvements of actual incumbents."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.model.paths import enumerate_paths
from src.operations.validation import build_operations
from src.solvers.algorithm import solve_subproblem
from src.solvers.heuristics import cleanup_incumbent, design_neighbors, initial_design
from test_operations import all_courier, instance, single_solution
from test_solver import data


def redundant_solution(*, per_path_flow=1.0):
    d = instance(
        n_h=3,
        flight_times=[
            [0, 0.05, 0.07, 0.08],
            [0.05, 0, 0.1, 0.1],
            [0.07, 0.1, 0, 0.1],
            [0.08, 0.1, 0.1, 0],
        ],
        slot_counts=[4, 8, 100],
        slot_cost=[0.1, 0.2, 3.0],
        link_cost=0.25,
    )
    result = single_solution(d, per_path_flow=per_path_flow)
    sol = result["_solution"]
    original_cost = float(np.sum(d.slot_cost * sol["capacity_level"]))
    sol["capacity_level"][:] = 0
    sol["capacity_level"][:, -1] = 1
    result["objective"] += (
        float(np.sum(d.slot_cost * sol["capacity_level"])) - original_cost
    )
    assert build_operations(d, result)["summary"]["checks"]["passed"]
    return d, result


def test_initial_design_is_supported_binary_and_has_actual_improving_recourse():
    d = data()
    paths = enumerate_paths(d)
    proposal = initial_design(d, paths)
    assert proposal["assignment"].sum() == 2
    assert all(np.isin(value, (0, 1)).all() for value in proposal.values())
    np.testing.assert_array_equal(
        proposal["capacity_level"].sum(axis=1), proposal["facility"][d.n_l :]
    )
    answer = solve_subproblem(d, paths, proposal, time_limit=10)
    assert answer["solution"] is not None
    assert answer["solution"]["objective"] < float(d.demand @ d.ground_cost)
    assert build_operations(d, answer["solution"])["summary"]["checks"]["passed"]


def test_k2_requires_both_ordered_pair_supports_and_self_pairs():
    d = data()
    paths = enumerate_paths(d)
    no_ab = replace(
        paths,
        service_paths=tuple(
            p for p in paths.service_paths if (p.first_od, p.second_od) != (0, 1)
        ),
    )
    proposal = initial_design(d, no_ab)
    assert proposal["assignment"].sum() == 1
    assert proposal["assignment"][1, 0, 0] == 1  # higher demand is considered first
    one = data(demand=[4.0], rider_mean=0.08, kappa=[[1.0]])
    singles = replace(
        enumerate_paths(one),
        service_paths=tuple(
            p for p in enumerate_paths(one).service_paths if p.mode == "single"
        ),
    )
    assert not initial_design(one, singles)["assignment"].any()


def test_k1_ignores_continuation_and_unprofitable_offers_are_skipped():
    d = data(max_orders=1)
    assert initial_design(d, enumerate_paths(d))["assignment"].sum() == 2
    unprofitable = replace(d, ground_cost=np.zeros(d.n_od))
    proposal = initial_design(unprofitable, enumerate_paths(unprofitable))
    assert not any(value.any() for value in proposal.values())


def test_incomplete_catalogue_is_not_used_for_design_support():
    d = data()
    with pytest.raises(ValueError, match="complete"):
        initial_design(d, replace(enumerate_paths(d), complete=False))


def test_cleanup_reduces_fixed_cost_without_changing_flows_or_original():
    d, result = redundant_solution()
    original = deepcopy(result)
    cleaned = cleanup_incumbent(d, result)
    assert cleaned["objective"] < result["objective"]
    assert cleaned["cleanup"]["saving"] == pytest.approx(
        result["objective"] - cleaned["objective"]
    )
    assert build_operations(d, cleaned)["summary"]["checks"]["passed"]
    for name in (
        "assignment",
        "rho",
        "idle",
        "fleet",
        "head",
        "eta",
        "flow",
        "empty_flow",
        "courier",
    ):
        np.testing.assert_array_equal(
            cleaned["_solution"][name], original["_solution"][name]
        )
    for name in result["_solution"]:
        np.testing.assert_array_equal(
            result["_solution"][name], original["_solution"][name]
        )
    assert result["objective"] == original["objective"]
    assert cleaned["_solution"]["facility"][3] == 0  # unused third locker
    assert cleaned["_solution"]["capacity_level"][:2, 0].sum() == 2
    assert cleaned["_solution"]["airline"].sum() == 3
    assert cleaned["_missions"] == result["_missions"]


def test_cleanup_protects_every_positive_flow_even_below_validation_tolerance():
    d, result = redundant_solution(per_path_flow=1e-8)
    cleaned = cleanup_incumbent(d, result)
    assert cleaned["_solution"]["facility"][:3].all()
    for u, v in ((0, 1), (1, 2), (2, 0)):
        assert cleaned["_solution"]["airline"][u, v] == 1
    assert cleaned["_solution"]["flow"][0] == 1e-8


def test_cleanup_selects_cheapest_feasible_capacity_not_lowest_index():
    d, result = redundant_solution()
    d = replace(d, slot_cost=np.tile([0.3, 0.1, 3.0], (d.n_h, 1)))
    cleaned = cleanup_incumbent(d, result)
    assert cleaned["_solution"]["capacity_level"][:2, 1].sum() == 2
    assert build_operations(d, cleaned)["summary"]["checks"]["passed"]


def test_cleanup_preserves_unused_assignment_and_closes_only_safe_base():
    d = instance(n_l=2, flight_times=np.ones((4, 4)) - np.eye(4))
    result = all_courier(d)
    result["_solution"]["facility"][:2] = 1
    result["_solution"]["assignment"][0, 0, 0] = 1
    result["objective"] += float(d.base_cost.sum())
    cleaned = cleanup_incumbent(d, result)
    np.testing.assert_array_equal(cleaned["_solution"]["facility"][:2], [1, 0])
    np.testing.assert_array_equal(
        cleaned["_solution"]["assignment"], result["_solution"]["assignment"]
    )


def test_cleanup_rejects_invalid_incumbent_and_updates_gap_consistently():
    d, result = redundant_solution()
    result.update(bound=0.0, gap=1.0)
    cleaned = cleanup_incumbent(d, result)
    assert cleaned["gap"] == 1.0
    bad = deepcopy(result)
    bad["_solution"]["airline"][:] = 0
    with pytest.raises(ValueError, match="valid incumbent"):
        cleanup_incumbent(d, bad)


def test_bounded_neighbors_preserve_existing_assignments_and_input():
    d = data()
    paths = enumerate_paths(d)
    incumbent = dict(_solution=dict(assignment=np.zeros((2, 1, 1))))
    incumbent["_solution"]["assignment"][0, 0, 0] = 1
    neighbors = design_neighbors(d, paths, incumbent, max_neighbors=1)
    assert len(neighbors) == 1
    assert neighbors[0]["assignment"].sum() == 2
    assert incumbent["_solution"]["assignment"].sum() == 1
    no_ab = replace(
        paths,
        service_paths=tuple(
            p for p in paths.service_paths if (p.first_od, p.second_od) != (0, 1)
        ),
    )
    assert not design_neighbors(d, no_ab, incumbent)
    assert not design_neighbors(d, paths, incumbent, max_neighbors=0)
