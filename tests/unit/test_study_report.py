"""Multi-dataset reporting keeps controls, units, and optimization evidence."""

from copy import deepcopy
import json

import pytest

from src.experiments.runner import run_case
from src.reporting.study import (
    _cost_delta_rows,
    _normalized_fleet_rows,
    _policy_changes,
    produce_study,
)


def _row(k, objective, bound):
    row = dict(dataset="test", max_orders=k, objective=objective, bound=bound)
    for key in (
        "drone_orders_per_hour",
        "direct_orders_per_hour",
        "fleet",
        "idle_drones",
        "busy_drones",
        "double_tasks_per_hour",
        "service_tasks_per_hour",
        "mean_drone_receipt_minutes",
        "compartments",
        "cost_per_order",
    ):
        row[key] = 0.0
    return row


def test_cost_order_requires_disjoint_optimization_bounds():
    first = _row(1, 100, 90)
    overlap = _policy_changes([first, _row(2, 95, 85)])[0]
    assert overlap["delta_objective"] == -5
    assert overlap["optimal_cost_difference_lower"] == -15
    assert overlap["optimal_cost_difference_upper"] == 5
    assert overlap["certificate"] == "optimal_cost_order_unresolved"
    better = _policy_changes([first, _row(2, 89, 85)])[0]
    assert better["certificate"] == "K2_lower_optimal_cost"
    worse = _policy_changes([first, _row(2, 110, 101)])[0]
    assert worse["certificate"] == "K2_higher_optimal_cost"


def test_study_exports_all_courier_without_inventing_drone_means(tmp_path, small_input):
    for k in (1, 2):
        case = deepcopy(small_input)
        case.update(max_orders=k, flow_policy="aggregate")
        run_case(
            case, tmp_path / "runs" / f"test_K{k}", time_limit=0, make_report=False
        )
    result = produce_study(tmp_path)
    assert result["cases"] == 2 and result["independent_validation_passed"]
    for row in result["comparison"]:
        assert row["drone_orders_per_hour"] == 0
        assert row["mean_task_minutes"] is None
        assert row["mean_drone_receipt_minutes"] is None
        assert row["fleet_per_100_drone_orders"] is None
        assert row["demand_per_hour"] == 11
        assert row["cost_per_order"] == pytest.approx(5)
    out = tmp_path / "report"
    manifest = json.loads((out / "source_manifest.json").read_text())
    assert not manifest["optimization_rerun"]
    assert manifest["within_dataset_physics_identical"]
    assert manifest["tables"]["orders"] == 4
    assert (out / "tables" / "orders.csv").is_file()
    assert (out / "headline_comparison.png").stat().st_size > 1000
    assert (
        out / "cases" / "test_K2" / "path_mean_distributions.pdf"
    ).stat().st_size > 1000
    report = (out / "REPORT.md").read_text()
    assert "不是统计置信区间" in report
    assert "不是单次配送随机送达时间" in report


def test_study_rejects_uncontrolled_policy_comparison(tmp_path, small_input):
    for k in (1, 2):
        case = deepcopy(small_input)
        case.update(max_orders=k, flow_policy="aggregate")
        if k == 2:
            case["ground_cost"] = case["ground_cost"] * 2
        run_case(
            case, tmp_path / "runs" / f"test_K{k}", time_limit=0, make_report=False
        )
    with pytest.raises(ValueError, match="identical physical inputs"):
        produce_study(tmp_path)
    assert not (tmp_path / "report").exists()


def test_cost_deltas_sum_and_fleet_normalization_uses_order_throughput():
    rows = []
    for k in (1, 2):
        row = dict(
            case=f"test_K{k}",
            dataset="test",
            max_orders=k,
            objective=100 if k == 1 else 95,
            cost_direct_courier=50 if k == 1 else 40,
            cost_service=20 if k == 1 else 24,
            cost_fleet=20 if k == 1 else 21,
            cost_bases=2,
            cost_lockers=4,
            cost_links=3,
            cost_repositioning=1,
            drone_orders_per_hour=200 if k == 1 else 0,
            busy_service_flight=12,
            busy_first_pickup_wait=2,
            busy_second_pickup_wait=1,
            busy_empty_flight=1,
            idle_drones=4,
            fleet_slack=0,
            fleet=20,
        )
        rows.append(row)
    deltas = _cost_delta_rows(rows)
    assert sum(
        r["delta_cny_per_hour"] for r in deltas if r["component"] != "net_total"
    ) == pytest.approx(-5)
    assert (
        next(r for r in deltas if r["component"] == "net_total")["delta_cny_per_hour"]
        == -5
    )
    normalized = _normalized_fleet_rows(rows)
    first = [r for r in normalized if r["max_orders"] == 1]
    assert sum(
        r["drones_per_100_drone_orders_per_hour"]
        for r in first
        if r["component"] != "total"
    ) == pytest.approx(10)
    assert (
        next(r for r in first if r["component"] == "total")[
            "drones_per_100_drone_orders_per_hour"
        ]
        == 10
    )
    assert all(
        r["drones_per_100_drone_orders_per_hour"] is None
        for r in normalized
        if r["max_orders"] == 2
    )
    # Expanded service accounting must keep higher waits visible next to
    # cheaper flights; the coarse service total is not counted a second time.
    for row in rows:
        row.update(
            {
                "detail_cost_" + name: row["cost_" + name]
                for name in ("bases", "lockers", "links", "fleet", "direct_courier")
            }
        )
        row.update(
            detail_cost_empty_flight=row["cost_repositioning"],
            detail_cost_first_rider=5,
            detail_cost_terminal_rider=5,
            detail_cost_service_flight=6 if row["max_orders"] == 1 else 4,
            detail_cost_drone_wait=4 if row["max_orders"] == 1 else 10,
        )
    detailed = _cost_delta_rows(rows)
    assert len(detailed) == 11
    assert sum(
        r["delta_cny_per_hour"] for r in detailed if r["component"] != "net_total"
    ) == pytest.approx(-5)
    assert (
        next(r for r in detailed if r["component"] == "service_flight")[
            "delta_cny_per_hour"
        ]
        == -2
    )
    assert (
        next(r for r in detailed if r["component"] == "drone_wait")[
            "delta_cny_per_hour"
        ]
        == 6
    )


@pytest.mark.parametrize("zero_field", ["demand", "objective"])
def test_real_data_study_rejects_undefined_total_cost_or_demand_ratios(
    tmp_path, small_input, monkeypatch, zero_field
):
    case = deepcopy(small_input)
    case.update(max_orders=1, flow_policy="aggregate")
    result = dict(objective=55)
    if zero_field == "demand":
        case["demand"] = case["demand"] * 0
    else:
        result["objective"] = 0
    monkeypatch.setattr("src.reporting.study.load_seed", lambda path: (case, result))
    with pytest.raises(
        ValueError, match="positive total demand and positive incumbent cost"
    ):
        produce_study(tmp_path, runs={("test", 1): tmp_path / "case"})
    assert not (tmp_path / "report").exists()
