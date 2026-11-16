"""Aggregate task proportions remain compulsory while prefixes can specialize."""

from copy import deepcopy
from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from src.model.resources import build_initial_segment, build_service_path
from src.model.types import PlanningData
from src.operations.validation import build_operations
from src.reporting.comparison import _label, _scenario_fingerprint
from src.data.instances import pack_planning_data


def specialized_solution():
    """Single tasks use prefix A; all double tasks use prefix B, without a solve."""
    data = PlanningData(
        n_l=1,
        n_h=2,
        demand=[4.0, 4.0],
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
        kappa=np.ones((2, 2)),
        delta=0.04,
        operating_limit=1.0,
        fleet_cost=1.0,
        flight_cost=0.2,
        waiting_cost=0.3,
        max_orders=2,
        flow_policy="aggregate",
    )
    rho = 0.5
    lam = float(data.demand.sum())
    q = -math.expm1(-lam * data.delta)
    first_rate = rho * 4 / (1 + q)
    segments, missions, heads, flows = [], [], [], []
    for r in range(2):
        for double in (False, True):
            i, j = (1, 0) if double else (0, 1)
            legs = ((0, 1 + i), (1 + i, 1 + j))
            segment = build_initial_segment(
                data, (0, 0), r, i, j, legs, id=len(segments)
            )
            segments.append(asdict(segment))
            heads.append(first_rate * (q if double else 1 - q))
            for s in range(2) if double else (None,):
                path = build_service_path(
                    data,
                    segment,
                    legs + (((1,), (1, 2), (2, 0)) if double else ((2, 0),)),
                    second_od=s,
                    second_pickup=0 if double else None,
                    second_delivery=1 if double else None,
                    id=len(missions),
                )
                assert path is not None
                flow = first_rate * (q / 2 if double else 1 - q)
                missions.append(dict(path=asdict(path), flow=flow))
                flows.append(flow)
    idle = rho / (1 - rho)
    fleet = idle + sum(row["flow"] * row["path"]["time"] for row in missions)
    result = dict(
        model_version="consecutive",
        flow_policy="aggregate",
        objective=float(
            data.base_cost.sum()
            + data.slot_cost.sum()
            + data.fleet_cost * fleet
            + sum(row["flow"] * row["path"]["cost"] for row in missions)
            + data.ground_cost @ (data.demand * (1 - rho))
        ),
        _solution=dict(
            facility=np.ones(data.n_nodes),
            capacity_level=np.ones(data.slot_counts.shape),
            assignment=np.ones((2, 1, 1)),
            airline=data.allowed_links.astype(float),
            rho=np.array([rho]),
            idle=np.array([idle]),
            fleet=fleet,
            head=np.array(heads),
            eta=np.array(heads) * q / lam,
            flow=np.array(flows),
            empty_flow=np.array([]),
            courier=data.demand * (1 - rho),
        ),
        _segments=segments,
        _missions=missions,
        _relocations=[],
    )
    return data, result


def test_specialized_prefixes_pass_aggregate_and_fail_prefix_policy():
    data, result = specialized_solution()
    payload = build_operations(data, result)
    assert payload["summary"]["flow_policy"] == "aggregate"
    assert payload["summary"]["checks"]["passed"]
    assert payload["summary"]["served"] == pytest.approx(4.0)

    prefix_result = deepcopy(result)
    prefix_result["flow_policy"] = "prefix"
    checks = build_operations(replace(data, flow_policy="prefix"), prefix_result)[
        "summary"
    ]["checks"]
    assert not checks["passed"]
    assert checks["residuals"]["double_branches"] > 0.1


def test_pair_marginals_cannot_replace_required_pair_totals():
    data, result = specialized_solution()
    # A 2x2 cycle preserves all first/second OD totals, segment starts, resource
    # use and cost here; only the required (r,s) task composition is violated.
    epsilon = 0.05
    for row in result["_missions"]:
        path = row["path"]
        if path["second_od"] is not None:
            shift = epsilon if path["first_od"] == path["second_od"] else -epsilon
            row["flow"] += shift
            result["_solution"]["flow"][path["id"]] += shift
    checks = build_operations(data, result)["summary"]["checks"]
    assert not checks["passed"]
    assert checks["residuals"]["aggregate_double_branches"] == pytest.approx(epsilon)
    assert checks["residuals"]["served_order_identity"] < 1e-10
    assert checks["residuals"]["first_leg_task_flow"] < 1e-10
    assert checks["residuals"]["objective_relative"] < 1e-10
    assert not checks["physical_errors"]


def test_aggregate_keeps_local_head_equal_to_actual_task_starts():
    data, result = specialized_solution()
    result["_solution"]["head"][:2] += [0.1, -0.1]
    q = -math.expm1(-float(data.demand.sum()) * data.delta)
    result["_solution"]["eta"] = result["_solution"]["head"] * q / data.demand.sum()
    checks = build_operations(data, result)["summary"]["checks"]
    assert not checks["passed"]
    assert checks["residuals"]["first_leg_task_flow"] == pytest.approx(0.1)
    assert checks["residuals"]["aggregate_double_branches"] < 1e-10


def test_result_policy_must_match_inputs_and_legacy_mode_is_prefix():
    data, result = specialized_solution()
    result["flow_policy"] = "prefix"
    with pytest.raises(ValueError, match="flow_policy"):
        build_operations(data, result)
    del result["flow_policy"]
    with pytest.raises(ValueError, match="flow_policy"):
        build_operations(data, result)


def test_reports_label_policy_but_fingerprint_retains_same_physical_scenario():
    data, _ = specialized_solution()
    prefix = replace(data, flow_policy="prefix")
    assert _scenario_fingerprint(pack_planning_data(data)) == _scenario_fingerprint(
        pack_planning_data(prefix)
    )
    row = dict(max_orders=2, method="full_path", flow_policy="aggregate")
    assert "aggregate" in _label(row)
