"""Conservation, common support and independent fixed-demand service checks."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from src.model.generalized_demand import generalized_demand
from src.model.paths import enumerate_paths
from src.operations.validation import build_operations
from test_operations import instance


def test_residual_pair_demand_counts_self_pairs_twice_and_is_conserved():
    result = generalized_demand(
        [10, 8],
        0.1,
        np.ones((2, 2)),
        [3, 3],
        [[4, 3], [4, 4]],
        np.ones((2, 1), bool),
        np.ones((2, 2, 1), bool),
    )
    assert result.candidate_order[0] == (0, 1)
    assert result.pair_rates[0, 1] == pytest.approx(8 * (1 - np.exp(-0.8)))
    assert result.pair_rates[0, 0] > 0
    assert np.all(result.single_rates >= 0)
    np.testing.assert_allclose(result.reconstructed_orders, [10, 8], atol=1e-12)
    assert result.single_rates.sum() + 2 * result.pair_rates.sum() == pytest.approx(18)


def test_zero_window_recovers_singles_and_fractional_compatibility_is_rejected():
    args = (
        [10, 8],
        0.0,
        np.ones((2, 2)),
        [3, 3],
        np.full((2, 2), 4.0),
        np.ones((2, 1), bool),
        np.ones((2, 2, 1), bool),
    )
    result = generalized_demand(*args)
    np.testing.assert_array_equal(result.single_rates, [10, 8])
    assert result.pair_rates.sum() == 0
    changed = list(args)
    changed[2] = np.full((2, 2), 0.5)
    single = generalized_demand(*changed)
    np.testing.assert_array_equal(single.single_rates, [10, 8])
    assert single.pair_rates.sum() == 0
    changed[1] = 0.1
    with pytest.raises(ValueError, match="binary kappa"):
        generalized_demand(*changed)


def test_common_departure_support_prevents_individually_supported_inconsistent_branches():
    single = np.ones((3, 2), bool)
    double = np.zeros((3, 3, 2), bool)
    double[0, 1, 0] = True
    double[0, 2, 1] = True
    costs = np.full((3, 3), np.inf)
    costs[0, 1], costs[0, 2] = 2, 3
    compatibility = np.zeros((3, 3))
    compatibility[0, 1] = compatibility[0, 2] = 1
    result = generalized_demand(
        [10, 8, 7], 0.1, compatibility, [3, 3, 3], costs, single, double
    )
    assert result.pair_rates[0, 1] > 0
    assert result.pair_rates[0, 2] == 0
    assert result.processing_trace[1]["status"] == "no_common_origin"
    np.testing.assert_array_equal(result.common_origin_support[0], [True, False])
    np.testing.assert_allclose(result.reconstructed_orders, [10, 8, 7])


def test_saved_candidate_order_tolerates_only_numerical_zero_savings():
    args = ([10, 8], 0.1, np.ones((2, 2)), [3, 3],
            np.array([[4., 6.], [7., 4.]]), np.ones((2, 1), bool),
            np.ones((2, 2, 1), bool))
    # Native floating point may classify the exactly-zero proxy as positive.
    result = generalized_demand(*args, candidate_order=[(0, 0), (1, 1), (0, 1)])
    np.testing.assert_allclose(result.reconstructed_orders, [10, 8])
    for order in ([(0, 0)], [(0, 0), (1, 1), (1, 0)],
                  [(0, 0), (1, 1), (1, 1)], [(0, 0), (1, 1), (3, 0)]):
        with pytest.raises(ValueError, match="eligible pairs"):
            generalized_demand(*args, candidate_order=order)
    changed = list(args)
    changed[2] = np.array([[1, 0], [1, 1]])
    with pytest.raises(ValueError, match="eligible pairs"):
        generalized_demand(*changed, candidate_order=[(0, 0), (1, 1), (0, 1)])


def _manual_preprocessed(data=None):
    if data is None:
        data = instance(
            demand=[4.0, 3.0],
            kappa=np.ones((2, 2)),
            max_orders=2,
            flow_policy="aggregate",
        )
    paths = enumerate_paths(data)
    c1, c2 = np.full(2, np.inf), np.full((2, 2), np.inf)
    s1, s2 = np.zeros((2, data.n_l), bool), np.zeros((2, 2, data.n_l), bool)
    for p in paths.service_paths:
        if p.second_od is None:
            c1[p.first_od] = min(c1[p.first_od], p.cost)
            s1[p.first_od, p.b[0]] = True
        else:
            c2[p.first_od, p.second_od] = min(c2[p.first_od, p.second_od], p.cost)
            s2[p.first_od, p.second_od, p.b[0]] = True
    fixed = generalized_demand(data.demand, data.effective_delta, data.kappa, c1, c2, s1, s2)
    rho = np.zeros(data.n_l)
    rho[0] = 0.5
    task_count = 2 if data.max_orders == 1 else 6
    assignment = np.zeros((task_count, data.n_l))
    assignment[0, 0] = 1.0
    if data.max_orders == 2:
        assignment[2:4, 0] = 1.0
    flow = np.zeros(len(paths.service_paths))
    head = np.zeros(len(paths.initial_segments))
    served = np.zeros(2)
    for r in range(2):
        for s in (None, 0, 1):
            rate = fixed.single_rates[r] if s is None else fixed.pair_rates[r, s]
            q = r if s is None else 2 + 2*r + s
            amount = rate * assignment[q, 0] * rho[0] if rate > 0 else 0.0
            if amount == 0:
                continue
            p = min(
                (
                    p
                    for p in paths.service_paths
                    if p.first_od == r and p.second_od == s and p.b == (0, 0)
                ),
                key=lambda p: p.cost,
            )
            flow[p.id] += amount
            head[p.segment_id] += amount
            served[r] += amount
            if s is not None:
                served[s] += amount
    idle = rho / (1 - rho)
    fleet = idle.sum() + sum(p.time * flow[p.id] for p in paths.service_paths)
    costs = (
        data.base_cost.sum()
        + data.slot_cost.sum()
        + data.link_cost[data.allowed_links].sum()
    )
    costs += data.fleet_cost * fleet + data.ground_cost @ (data.demand - served)
    costs += sum(p.cost * flow[p.id] for p in paths.service_paths)
    result = dict(
        model_version="consecutive",
        max_orders=data.max_orders,
        flow_policy="aggregate",
        demand_formulation="preprocessed",
        assignment_scope="task",
        task_indexing="singles_then_row_major_pairs",
        task_table=[dict(task=q,first_od=q if q < 2 else (q-2)//2,
            second_od=-1 if q < 2 else (q-2)%2,
            rate=float(fixed.single_rates[q] if q < 2 else fixed.pair_rates[(q-2)//2,(q-2)%2]))
            for q in range(task_count)],
        objective=float(costs),
        preprocessed_demand=dict(
            single_rates=fixed.single_rates.tolist(),
            pair_rates=fixed.pair_rates.tolist(),
            head_rates=fixed.head_rates.tolist(),
            common_origin_support=fixed.common_origin_support.astype(int).tolist(),
            candidate_order=fixed.candidate_order,
        ),
        _solution=dict(
            facility=np.ones(data.n_nodes),
            capacity_level=np.ones(data.slot_counts.shape),
            assignment=assignment,
            airline=data.allowed_links.astype(float),
            rho=rho,
            idle=idle,
            fleet=float(fleet),
            head=head,
            eta=np.zeros_like(head),
            flow=flow,
            empty_flow=np.zeros(len(paths.empty_paths)),
            courier=data.demand - served,
        ),
        _segments=[asdict(s) for s in paths.initial_segments],
        _missions=[
            dict(path=asdict(p), flow=float(flow[p.id])) for p in paths.service_paths
        ],
        _relocations=[dict(path=asdict(p), flow=0.0) for p in paths.empty_paths],
    )
    return data, result


def test_single_order_uses_the_same_fixed_task_validation_and_origin_assignment():
    data, result = _manual_preprocessed(instance(
        demand=[4.0, 3.0], kappa=np.full((2, 2), 0.5),
        max_orders=1, flow_policy="aggregate",
    ))
    ops = build_operations(data, result)
    assert ops["summary"]["checks"]["passed"], ops["summary"]["checks"]
    assert ops["summary"]["assignment_scope"] == "task"
    np.testing.assert_array_equal(result["preprocessed_demand"]["single_rates"], data.demand)
    assert np.asarray(result["preprocessed_demand"]["pair_rates"]).sum() == 0
    assert all(row["path"]["second_od"] is None for row in result["_missions"])


def test_independent_audit_allows_second_order_with_unassigned_own_head():
    data, result = _manual_preprocessed()
    ops = build_operations(data, result)
    assert ops["summary"]["checks"]["passed"], ops["summary"]["checks"]
    second = ops["tables"]["orders"][1]
    assert second["assigned"] is True  # Its pair task is independently assigned.
    assert result["_solution"]["assignment"][1].sum() == 0
    assert second["drone_per_hour"] == pytest.approx(second["second_per_hour"])
    assert second["second_per_hour"] > 0
    assert ops["summary"]["unassigned_direct_orders_per_hour"] + ops["summary"][
        "unavailability_direct_orders_per_hour"
    ] == pytest.approx(ops["summary"]["direct"])


def test_independent_audit_detects_tampered_preprocessing_and_branch_flows():
    data, original = _manual_preprocessed()
    changed = deepcopy(original)
    changed["preprocessed_demand"]["pair_rates"][0][1] += 0.1
    checks = build_operations(data, changed)["summary"]["checks"]
    assert not checks["passed"]
    assert checks["residuals"]["preprocessed_pair_rates"] == pytest.approx(0.1)
    changed = deepcopy(original)
    pid = next(i for i, value in enumerate(changed["_solution"]["flow"]) if value > 0)
    changed["_solution"]["flow"][pid] += 0.1
    changed["_missions"][pid]["flow"] += 0.1
    checks = build_operations(data, changed)["summary"]["checks"]
    assert not checks["passed"]
    assert checks["residuals"]["first_leg_task_flow"] == pytest.approx(0.1)


def _split_first_legs(data, original, splits):
    """Preserve each task-class rate while moving it between two first legs."""
    result = deepcopy(original)
    sol = result["_solution"]
    totals = {}
    for row in original["_missions"]:
        key = row["path"]["first_od"], row["path"]["second_od"]
        totals[key] = totals.get(key, 0.0) + row["flow"]
    sol["flow"][:] = 0
    sol["head"][:] = 0
    for row in result["_missions"]:
        row["flow"] = 0.0
    for (r, s), amount in totals.items():
        if amount == 0:
            continue
        candidates = [
            row
            for row in result["_missions"]
            if row["path"]["first_od"] == r and row["path"]["second_od"] == s
        ]
        by_segment = {}
        for row in candidates:
            by_segment.setdefault(row["path"]["segment_id"], row)
        assert len(by_segment) == 2
        for fraction, row in zip(splits[s], by_segment.values()):
            row["flow"] = fraction * amount
            sol["flow"][row["path"]["id"]] = row["flow"]
            sol["head"][row["path"]["segment_id"]] += row["flow"]
    busy = sum(row["flow"] * row["path"]["time"] for row in result["_missions"])
    sol["fleet"] = float(sol["idle"].sum() + busy)
    result["objective"] = float(
        data.base_cost @ sol["facility"][: data.n_l]
        + np.sum(data.slot_cost * sol["capacity_level"])
        + np.sum(data.link_cost * sol["airline"])
        + data.fleet_cost * sol["fleet"]
        + data.ground_cost @ sol["courier"]
        + sum(row["flow"] * row["path"]["cost"] for row in result["_missions"])
    )
    return result


def test_first_leg_consistency_accepts_an_ex_ante_continuous_split():
    data, original = _manual_preprocessed()
    result = _split_first_legs(
        data, original, {None: (0.4, 0.6), 0: (0.4, 0.6), 1: (0.4, 0.6)}
    )
    result["first_leg_consistency"] = True
    with pytest.raises(ValueError, match="incompatible with independent tasks"):
        build_operations(data, result)
    # Independent tasks may retain the same physical first leg by choice.
    result["first_leg_consistency"] = False
    assert build_operations(data, result)["summary"]["checks"]["passed"]
    assert result["preprocessed_demand"] == original["preprocessed_demand"]


@pytest.mark.parametrize("second_split", [(0.8, 0.2), (1.0, 0.0)])
def test_first_leg_consistency_detects_future_branch_dependent_first_legs(second_split):
    data, original = _manual_preprocessed()
    result = _split_first_legs(
        data, original, {None: (0.4, 0.6), 0: (0.4, 0.6), 1: second_split}
    )
    assert build_operations(data, result)["summary"]["checks"]["passed"]
    result["first_leg_consistency"] = True
    with pytest.raises(ValueError, match="incompatible with independent tasks"):
        build_operations(data, result)


def test_first_leg_consistency_flag_requires_boolean_and_fixed_demand_mode():
    data, result = _manual_preprocessed()
    result["first_leg_consistency"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        build_operations(data, result)
    from test_operations import single_solution

    data = instance()
    result = single_solution(data)
    result["first_leg_consistency"] = True
    with pytest.raises(ValueError, match="incompatible with independent tasks"):
        build_operations(data, result)


def test_first_leg_consistency_groups_identical_routes_across_different_returns():
    data = instance(
        n_l=2,
        demand=[4.0, 3.0],
        kappa=np.ones((2, 2)),
        max_orders=2,
        flow_policy="aggregate",
        flight_times=[
            [0, 0.05, 0.05, 0.07],
            [0.05, 0, 0.06, 0.06],
            [0.05, 0.06, 0, 0.1],
            [0.07, 0.06, 0.1, 0],
        ],
    )
    data, result = _manual_preprocessed(data)
    result["first_leg_consistency"] = False
    sol = result["_solution"]
    sol["flow"][:] = 0
    sol["head"][:] = 0
    for row in result["_missions"]:
        row["flow"] = 0.0
    fixed = result["preprocessed_demand"]
    for second in (None, 0, 1):
        rate = (
            fixed["single_rates"][0]
            if second is None
            else fixed["pair_rates"][0][second]
        )
        target_return = 0 if second is None else 1
        row = next(
            row
            for row in result["_missions"]
            if (
                row["path"]["first_od"] == 0
                and row["path"]["second_od"] == second
                and row["path"]["b"] == (0, target_return)
                and row["path"]["pickups"][0] == 0
                and row["path"]["deliveries"][0] == 1
            )
        )
        amount = rate * sol["rho"][0]
        row["flow"] = amount
        sol["flow"][row["path"]["id"]] = amount
        sol["head"][row["path"]["segment_id"]] += amount
    relocation = next(
        row for row in result["_relocations"] if row["path"]["b"] == (1, 0)
    )
    relocation["flow"] = sum(
        row["flow"] for row in result["_missions"] if row["path"]["b"][1] == 1
    )
    sol["empty_flow"][relocation["path"]["id"]] = relocation["flow"]
    records = result["_missions"] + result["_relocations"]
    sol["fleet"] = float(
        sol["idle"].sum() + sum(row["flow"] * row["path"]["time"] for row in records)
    )
    result["objective"] = float(
        data.base_cost.sum()
        + data.slot_cost.sum()
        + np.sum(data.link_cost * sol["airline"])
        + data.fleet_cost * sol["fleet"]
        + data.ground_cost @ sol["courier"]
        + sum(row["flow"] * row["path"]["cost"] for row in records)
    )
    checked = build_operations(data, result)["summary"]["checks"]
    assert checked["passed"], checked
