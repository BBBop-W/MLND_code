"""Independent saved-incumbent checks, including hostile/corrupted records."""

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from src.model.resources import build_initial_segment, build_service_path
from src.model.types import PlanningData
from src.operations.validation import build_operations


def instance(**overrides):
    values = dict(
        n_l=1,
        n_h=2,
        demand=[4.0],
        flight_times=[[0, 0.05, 0.07], [0.05, 0, 0.1], [0.07, 0.1, 0]],
        rider_mean=0.08,
        first_cost=1.0,
        terminal_cost=0.5,
        receipt_mean=0.08,
        removal_mean=0.03,
        ground_cost=5.0,
        deadline=1.0,
        base_cost=0.5,
        slot_counts=[100],
        slot_cost=0.1,
        slot_epsilon=0.05,
        link_cost=0.0,
        kappa=[[1.0]],
        delta=0.04,
        operating_limit=1.0,
        fleet_cost=1.0,
        flight_cost=0.2,
        waiting_cost=0.3,
        max_orders=1,
        flow_policy="prefix",
    )
    values.update(overrides)
    return PlanningData(**values)


def all_courier(data):
    return dict(
        model_version="consecutive",
        objective=float(data.demand @ data.ground_cost),
        _solution=dict(
            facility=np.zeros(data.n_nodes),
            capacity_level=np.zeros(data.slot_counts.shape),
            assignment=np.zeros((data.n_od, data.n_l, data.n_l)),
            airline=np.zeros((data.n_nodes, data.n_nodes)),
            rho=np.zeros(data.n_l),
            idle=np.zeros(data.n_l),
            fleet=0.0,
            head=np.array([]),
            eta=np.array([]),
            flow=np.array([]),
            empty_flow=np.array([]),
            courier=data.demand.copy(),
        ),
        _segments=[],
        _missions=[],
        _relocations=[],
    )


def single_solution(data, *, number=1, per_path_flow=1.0):
    segment = build_initial_segment(data, (0, 0), 0, 0, 1, ((0, 1), (1, 2)), id=0)
    path = build_service_path(data, segment, ((0, 1), (1, 2), (2, 0)), id=0)
    result = all_courier(data)
    solution = result["_solution"]
    served = number * per_path_flow
    rho = served / data.demand[0]
    idle = rho / (1 - rho)
    fleet = idle + served * path.time
    solution.update(
        facility=np.ones(data.n_nodes),
        capacity_level=np.ones(data.slot_counts.shape),
        assignment=np.ones((1, 1, 1)),
        airline=data.allowed_links.astype(float),
        rho=np.array([rho]),
        idle=np.array([idle]),
        fleet=fleet,
        head=np.array([served]),
        eta=np.array([0.0]),
        flow=np.full(number, per_path_flow),
        courier=data.demand - served,
    )
    result["_segments"] = [asdict(segment)]
    result["_missions"] = [
        dict(path=asdict(replace(path, id=i)), flow=per_path_flow)
        for i in range(number)
    ]
    result["objective"] = float(
        data.base_cost.sum()
        + data.slot_cost.sum()
        + data.link_cost[data.allowed_links].sum()
        + data.fleet_cost * fleet
        + served * path.cost
        + data.ground_cost @ (data.demand - served)
    )
    return result


def checks(data, result):
    return build_operations(data, result)["summary"]["checks"]


def test_all_courier_and_manual_single_are_feasible():
    data = instance()
    assert checks(data, all_courier(data))["passed"]
    result = single_solution(data)
    operations = build_operations(data, result)
    assert operations["summary"]["checks"]["passed"]
    assert operations["summary"]["served"] == 1.0
    assert operations["summary"]["single_tasks"] == 1.0
    assert operations["summary"]["objective"] == pytest.approx(result["objective"])


def test_verifier_does_not_call_model_resource_builder(monkeypatch):
    data = instance()
    result = single_solution(data)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "The operational verifier must independently reconstruct resources"
        )

    monkeypatch.setattr("src.model.resources.rebuild_path", forbidden)
    monkeypatch.setattr("src.model.resources.build_service_path", forbidden)
    assert checks(data, result)["passed"]


def test_every_tiny_flow_is_counted_in_resource_totals():
    data = instance()
    result = single_solution(data, number=1000, per_path_flow=1e-6)
    operations = build_operations(data, result)
    assert operations["summary"]["checks"]["passed"]
    assert operations["summary"]["served"] == pytest.approx(0.001)
    assert len(operations["tables"]["paths"]) == 1000
    result["_solution"]["fleet"] = float(result["_solution"]["idle"].sum())
    invalid = checks(data, result)
    assert not invalid["passed"]
    assert invalid["residuals"]["fleet_budget"] > 5e-5


def test_missing_tiny_records_cannot_hide_a_large_aggregate():
    data = instance()
    result = single_solution(data, number=1000, per_path_flow=1e-6)
    result["_missions"] = []
    invalid = checks(data, result)
    assert not invalid["passed"]
    assert invalid["residuals"]["service_flow_storage_total"] == pytest.approx(0.001)


@pytest.mark.parametrize("field", ["time", "wait", "cost", "flight_time"])
def test_tampered_cached_scalar_resource_is_rejected(field):
    data = instance()
    result = single_solution(data)
    result["_missions"][0]["path"][field] += 0.1
    invalid = checks(data, result)
    assert not invalid["passed"]
    assert any(field in message for message in invalid["physical_errors"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("slot_times", ((0, 0.0), (1, 0.0))),
        ("served_counts", ((0, 2),)),
        ("arc_counts", ((0, 1, 7),)),
        ("delivery_times", (0.0,)),
        ("pickup_times", (0.0,)),
    ],
)
def test_tampered_cached_arrays_are_rejected(field, value):
    data = instance()
    result = single_solution(data)
    result["_missions"][0]["path"][field] = value
    assert not checks(data, result)["passed"]


def test_zero_flow_path_and_unused_segment_still_validate_geometry():
    data = instance()
    result = single_solution(data, per_path_flow=0.0)
    assert checks(data, result)["passed"]
    result["_missions"][0]["path"]["legs"] = ((0, 1), (), (2, 0))
    assert not checks(data, result)["passed"]
    result = single_solution(data, per_path_flow=0.0)
    result["_segments"][0]["legs"] = ((0, 1), (1, 0))
    with pytest.raises(ValueError, match="first-leg"):
        checks(data, result)


def test_fractional_identifiers_are_not_silently_truncated():
    data = instance()
    result = single_solution(data)
    result["_missions"][0]["path"]["legs"] = ((0.2, 1), (1, 2), (2, 0))
    assert not checks(data, result)["passed"]
    result = single_solution(data)
    result["_missions"][0]["path"]["id"] = 0.2
    with pytest.raises(ValueError, match="integer"):
        checks(data, result)


@pytest.mark.parametrize("field", ["objective", "bound", "served", "gap"])
def test_nonfinite_reported_values_are_rejected(field):
    data = instance()
    result = single_solution(data)
    result["bound"] = 0.0
    result[field] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        checks(data, result)


def test_bad_bounds_courier_values_and_flow_shapes_fail():
    data = instance()
    result = single_solution(data)
    result["bound"] = result["objective"] + 1
    assert not checks(data, result)["passed"]
    result = single_solution(data)
    result["_solution"]["courier"] += 1
    assert not checks(data, result)["passed"]
    result = single_solution(data)
    result["_solution"]["flow"] = np.array([[1.0]])
    with pytest.raises(ValueError, match="one-dimensional"):
        checks(data, result)


def test_head_and_branch_equations_are_checked_even_when_order_total_matches():
    data = instance()
    result = single_solution(data)
    result["_solution"]["head"] += 0.1
    invalid = checks(data, result)
    assert not invalid["passed"]
    assert invalid["residuals"]["single_branches"] > 0.09
    result = single_solution(data)
    result["_solution"]["eta"][0] = 0.1
    assert not checks(data, result)["passed"]


def test_required_double_branch_cannot_be_replaced_by_single_service():
    data = instance(max_orders=2)
    result = single_solution(replace(data, max_orders=1))
    # The single path itself remains physically valid at K=2, but the required
    # offered double branch has been deleted and demand reconstruction changes.
    invalid = checks(data, result)
    assert not invalid["passed"]
    assert invalid["residuals"]["eta_identity_and_bounds"] > 0


def test_metadata_copy_does_not_mutate_input_solution():
    data = instance()
    result = single_solution(data)
    before = deepcopy(result)
    build_operations(data, result)
    for key in result["_solution"]:
        np.testing.assert_array_equal(
            result["_solution"][key], before["_solution"][key]
        )
