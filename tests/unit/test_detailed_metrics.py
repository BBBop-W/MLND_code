"""Meaningful accounting checks for single, continued and zero-flow plans."""

from copy import deepcopy
import json

import numpy as np
import pytest

from src.data.instances import pack_planning_data
from src.operations.export import export_operations
from src.operations.validation import build_operations
from test_aggregate_validation import specialized_solution
from test_operations import all_courier, instance, single_solution


def test_double_task_components_do_not_add_continuation_window_twice():
    data, result = specialized_solution()
    payload = build_operations(data, result)
    summary, tables = payload["summary"], payload["tables"]
    assert summary["checks"]["passed"]
    busy = {row["component"]: row["drones"] for row in tables["busy_components"]}
    expected_flight = sum(
        row["flow"]
        * sum(
            data.flight_times[u, v]
            for leg in row["path"]["legs"]
            for u, v in zip(leg, leg[1:])
        )
        for row in result["_missions"]
    )
    expected_busy = sum(
        row["flow"] * row["path"]["time"] for row in result["_missions"]
    )
    assert busy["service_flight"] == pytest.approx(expected_flight)
    assert sum(busy.values()) == pytest.approx(expected_busy)
    assert summary["busy_drones"] + summary["idle_drones"] == pytest.approx(
        summary["fleet"]
    )
    assert busy["second_pickup_wait"] > 0
    stages = {row["stage"]: row for row in tables["flight_stages"]}
    # This fixture's continuation transfer has identical endpoints.
    assert stages["continuation_transfer"]["task_legs_per_hour"] > 0
    assert stages["continuation_transfer"]["traversals_per_hour"] == 0
    assert sum(row["busy_drones"] for row in tables["arcs"]) == pytest.approx(
        expected_flight
    )
    assert sum(
        row["actual_tasks_per_hour"] for row in tables["task_pairs"]
    ) == pytest.approx(summary["double_tasks"])
    assert sum(
        row["pickup_orders_per_hour"] for row in tables["lockers"]
    ) == pytest.approx(summary["served"])
    for row in tables["lockers"]:
        assert row["pickup_reservation_mean"] + row[
            "delivery_reservation_mean"
        ] == pytest.approx(row["mean_occupancy"])


def test_closed_sites_and_all_courier_keep_zero_rows_and_finite_export(tmp_path):
    data = instance()
    payload = export_operations(data, all_courier(data), tmp_path)
    assert len(payload["tables"]["bases"]) == data.n_l
    assert len(payload["tables"]["lockers"]) == data.n_h
    assert len(payload["tables"]["arcs"]) == data.n_nodes * (data.n_nodes - 1)
    assert len(payload["tables"]["task_pairs"]) == data.n_l**2 * data.n_od**2
    assert payload["summary"]["busy_fraction"] is None
    assert payload["summary"]["unassigned_direct_orders_per_hour"] == 4
    assert payload["summary"]["terminal_choice_available"] is False
    assert (tmp_path / "metric_dictionary.csv").exists()
    json.dumps(payload, allow_nan=False)


def test_terminal_choices_direct_fallbacks_and_costs_reconcile():
    data = instance()
    result = single_solution(data)
    inputs = pack_planning_data(data)
    inputs.update(
        pickup_share=np.full((1, 2), 0.3),
        bases=np.array([[0.0, 0.0]]),
        hubs=np.array([[1.0, 0.0], [2.0, 0.0]]),
        origins=np.array([[0.5, 0.0]]),
        destinations=np.array([[2.5, 0.0]]),
        od_origin=np.array([0]),
        od_destination=np.array([0]),
    )
    payload = build_operations(inputs, result)
    summary, tables = payload["summary"], payload["tables"]
    assert summary["checks"]["passed"]
    assert summary["self_pickup_orders_per_hour"] == pytest.approx(0.3)
    assert summary["terminal_courier_orders_per_hour"] == pytest.approx(0.7)
    assert summary["unassigned_direct_orders_per_hour"] == 0
    assert summary["unavailability_direct_orders_per_hour"] == pytest.approx(3)
    assert sum(row["cny_per_hour"] for row in tables["cost_detail"]) == pytest.approx(
        summary["objective"]
    )
    assert tables["bases"][0]["availability"] == pytest.approx(0.25)
    assert tables["bases"][0]["attributed_busy_fraction"] != pytest.approx(0.25)
    assert len(tables["market_nodes"]) == 2
    assert all(
        row["drone_per_hour"] == pytest.approx(1.0) for row in tables["market_nodes"]
    )
    broken = deepcopy(inputs)
    broken["pickup_share"][0, 0] = 1.1
    with pytest.raises(ValueError, match="pickup_share"):
        build_operations(broken, result)
