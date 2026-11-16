"""Aggregate task allocation preserves demand while relaxing route selection."""

from dataclasses import replace

from gurobipy import GRB
import numpy as np
import pytest

from src.model.demand import demand_rates
from src.model.paths import enumerate_paths
from src.model.types import PlanningData
from src.solvers.formulation import build_model, path_column, shapes


def route_tradeoff_data(**changes):
    """OD 0 has two first routes; each supports a different second OD.

    OD 1 must use locker 1 -> 0 and OD 2 must use locker 0 -> 1. A
    zero-transfer double takes 0.30 h, but switching pickup takes 0.40 h.
    Both first routes take 0.20 h alone. No paths are manually removed.
    """
    values = dict(
        n_l=1,
        n_h=2,
        demand=[4.0, 7.0, 5.0],
        flight_times=[[0, 0.05, 0.05], [0.05, 0, 0.1], [0.05, 0.1, 0]],
        rider_mean=[[0, 0], [2, 0], [0, 2]],
        first_cost=1.0,
        terminal_cost=0.5,
        receipt_mean=0.0,
        removal_mean=0.0,
        ground_cost=5.0,
        deadline=0.3,
        base_cost=0.5,
        slot_counts=[100],
        slot_cost=[0.1],
        slot_epsilon=0.05,
        link_cost=0.0,
        kappa=[[0, 1, 1], [0, 0, 0], [0, 0, 0]],
        delta=0.04,
        operating_limit=0.31,
        fleet_cost=1.0,
        flight_cost=0.2,
        waiting_cost=0.3,
        flow_policy="aggregate",
    )
    values.update(changes)
    return PlanningData(**values)


def all_open_design(data):
    design = {name: np.ones(shape) for name, shape in shapes(data).items()}
    design["airline"] = data.allowed_links.astype(float)
    return design


def test_different_second_orders_may_select_different_first_routes():
    data = route_tradeoff_data()
    paths = enumerate_paths(data)
    first_segments = [s for s in paths.initial_segments if s.first_od == 0]
    assert len(first_segments) == 2
    branches = {
        s.id: {
            p.second_od
            for p in paths.service_paths
            if p.segment_id == s.id and p.mode == "double"
        }
        for s in first_segments
    }
    assert {frozenset(v) for v in branches.values()} == {frozenset({1}), frozenset({2})}
    rates = demand_rates(data.demand, kappa=data.kappa, delta=data.delta)
    for policy in ("prefix", "aggregate"):
        current = replace(data, flow_policy=policy)
        built = build_model(current, paths, fixed_design=all_open_design(current))
        try:
            built.model.addConstr(built.variables["rho"][0] == 0.5)
            built.model.Params.DualReductions = 0
            built.model.optimize()
            if policy == "prefix":
                assert built.model.Status == GRB.INFEASIBLE
                continue
            assert built.model.Status == GRB.OPTIMAL
            singles = np.zeros(data.n_od)
            pairs = np.zeros((data.n_od, data.n_od))
            for j, segment in enumerate(paths.initial_segments):
                actual = sum(
                    built.variables["flow"][i].X
                    for i, p in enumerate(paths.service_paths)
                    if p.segment_id == segment.id
                )
                assert built.variables["head"][j].X == pytest.approx(actual, abs=1e-7)
            for i, path in enumerate(paths.service_paths):
                value = built.variables["flow"][i].X
                if path.mode == "single":
                    singles[path.first_od] += value
                else:
                    pairs[path.first_od, path.second_od] += value
            np.testing.assert_allclose(singles, 0.5 * rates.single_rates, atol=1e-7)
            np.testing.assert_allclose(pairs, 0.5 * rates.pair_rates, atol=1e-7)
        finally:
            built.model.dispose()


def test_k1_is_unchanged_by_aggregate_relaxation():
    costs = []
    for policy in ("prefix", "aggregate"):
        data = route_tradeoff_data(max_orders=1, flow_policy=policy)
        paths = enumerate_paths(data)
        built = build_model(data, paths, fixed_design=all_open_design(data))
        try:
            built.model.addConstr(built.variables["rho"][0] == 0.5)
            built.model.optimize()
            assert built.model.Status == GRB.OPTIMAL
            costs.append(built.model.ObjVal)
            actual = np.zeros(data.n_od)
            for i, path in enumerate(paths.service_paths):
                assert path.mode == "single"
                actual[path.first_od] += built.variables["flow"][i].X
            np.testing.assert_allclose(actual, 0.5 * data.demand, atol=1e-7)
        finally:
            built.model.dispose()
    assert costs[0] == pytest.approx(costs[1], abs=1e-6)


def test_shared_task_rows_have_one_coefficient_per_path():
    data = route_tradeoff_data()
    paths = enumerate_paths(data)
    built = build_model(data, paths, fixed_design=all_open_design(data))
    try:
        first = [j for j, s in enumerate(paths.initial_segments) if s.first_od == 0]
        assert (
            built.constraints["single"][first[0]].index
            == built.constraints["single"][first[1]].index
        )
        assert (
            built.constraints["double"][first[0], 1].index
            == built.constraints["double"][first[1], 1].index
        )
        for i, path in enumerate(paths.service_paths):
            column = path_column(built, path)
            assert len({row.index for row, _ in column}) == len(column)
            expected = {row.index: value for row, value in column}
            actual = {
                row.index: coefficient
                for row in built.model.getConstrs()
                if (
                    coefficient := built.model.getCoeff(row, built.variables["flow"][i])
                )
            }
            assert actual == expected
    finally:
        built.model.dispose()
