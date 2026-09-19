"""Physical reconstruction catches stale clocks, risk and capacity semantics."""

import math

import numpy as np
import pytest

from src.model.network import network
from src.operations.validation import build_operations
from src.operations.export import export_operations
from src.operations.routes import _trace


def example(*, swap=False, family="exponential", mean=0.05):
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.1]])
    flight = np.full((5, 5), 0.1)
    np.fill_diagonal(flight, 0.0)
    flight[0, 4] = flight[4, 2] = 0.04
    data = dict(
        name="手算核验",
        n_l=2,
        n_h=2,
        n_k=1,
        n_od=1,
        bases=xy[:2],
        hubs=xy[2:4],
        remote=xy[4:],
        flight_times=flight,
        demand=np.array([20.0]),
        od_origin=np.array([0]),
        od_destination=np.array([0]),
        ground=np.array([8.0]),
        first_cost=np.array([[1.0, 1.0]]),
        terminal_cost=np.array([[2.0, 2.0]]),
        pickup_share=np.array([[0.25, 0.25]]),
        ground_time=np.array([[mean, mean]]),
        terminal_time=np.array([[0.08, 0.08]]),
        direct_rider_hours=np.array([1.0]),
        delivery_deadline=np.array([1.0]),
        battery_time=1.0,
        epsilon=0.05,
        swap_time=0.025,
        pickup_handling_time=0.01,
        delivery_handling_time=0.01,
        fleet_cost=5.0,
        drone_time_cost=10.0,
        hover_cost=2.0,
        base_cost=2.0,
        swap_cost=4.0,
        capacity_levels=np.array([1.0, 2.0]),
        level_capacity=np.array([[38.0, 76.0], [38.0, 76.0]]),
        level_cost=np.array([[3.0, 6.0], [3.0, 6.0]]),
        rider_time_family=family,
    )
    legs = [[0, 4, 2] if swap else [0, 2], [2, 3], [3, 1]]
    prefix = 0.105 if swap else 0.1
    wait = (
        max(mean - prefix, 0.0)
        if family == "deterministic"
        else (mean * math.exp(-prefix / mean) if mean else 0.0)
    )
    duration = prefix + 0.12 + 0.1 + wait
    fleet = 1.0 + 10.0 * duration + 10.0 * 0.1
    net = network(data)
    airline = np.zeros(net["count"])
    for leg in legs + [[1, 0]]:
        for u, v in zip(leg[:-1], leg[1:]):
            airline[net["lookup"][u * 5 + v]] = 1
    assignment = np.zeros((1, 2, 2))
    assignment[0, 0, 1] = 1
    solution = dict(
        facility=np.array([1.0, 1.0, 1.0, 1.0, float(swap)]),
        capacity_level=np.array([[1.0, 0.0], [1.0, 0.0]]),
        hub_count=np.ones(2),
        assignment=assignment,
        airline=airline,
        rho=np.array([0.5, 0.0]),
        idle=np.array([1.0, 0.0]),
        fleet=fleet,
        empty=np.array([[0.0, 0.0], [10.0, 0.0]]),
    )
    flown = 0.28 if swap else 0.3
    op = 1 + 2 + 10 * flown + 2 * wait
    objective = (
        4
        + 6
        + 4 * swap
        + float(net["fees"] @ airline)
        + 5 * fleet
        + 10 * op
        + 10 * 1
        + 10 * 8
    )
    result = dict(
        objective=objective,
        served=10.0,
        fleet=fleet,
        bound=objective - 1,
        _solution=solution,
        _missions=[dict(od=0, l=0, m=1, i=0, j=1, flow=10.0, trace=dict(legs=legs))],
        _relocations=[dict(l=1, m=0, flow=10.0, trace=dict(legs=[[1, 0]]))],
    )
    return data, result


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize(
    "family,mean",
    [("exponential", 0.05), ("deterministic", 0.15), ("exponential", 0.0)],
)
def test_full_reconstruction_matches_hand_computed_clocks_costs_and_counts(
    swap, family, mean
):
    data, result = example(swap=swap, family=family, mean=mean)
    ops = build_operations(data, result)
    s = ops["summary"]
    assert s["checks"]["passed"], s["checks"]
    assert s["objective"] == pytest.approx(result["objective"])
    assert s["fleet_occupancy"] == pytest.approx(result["fleet"])
    assert s["served"] == 10
    assert s["cross_base_service"] == 10
    assert s["empty_flow"] == 10
    assert s["opened"]["swaps"] == int(swap)
    assert s["cabinet_total"] == 2
    assert s["min_joint_reliability"] >= 0.95
    assert s["time_averages_minutes"]["mean_delivery"] == pytest.approx(
        s["time_averages_minutes"]["time"] - 6 + 4.8
    )
    assert sum(r["drones"] for r in ops["tables"]["fleet"]) == pytest.approx(
        result["fleet"]
    )
    assert len(ops["segments"]) == 4 + int(swap)
    assert ops["tables"]["od_times"][0]["pickup_per_hour"] == 2.5
    assert ops["tables"]["od_times"][0]["direct_postfetch_minutes"] == 60


def test_zero_availability_empty_dispatch_has_no_unspecified_extra_constraint():
    data, result = example()
    assert result["_solution"]["rho"][1] == 0
    assert result["_relocations"][0]["flow"] > 0
    ops = build_operations(data, result)
    assert ops["summary"]["checks"]["passed"]
    assert not any(
        "request" in name or "rho_max" in name for name in ops["summary"]["checks"]
    )


@pytest.mark.parametrize("swap", [False, True])
def test_one_time_site_costs_follow_open_locations_not_cabinet_quantity(swap):
    from src.model.network import facility_costs

    data, result = example(swap=swap)
    data.update(
        base_site_cost=np.array([1.0, 2.0]),
        hub_site_cost=np.array([3.0, 4.0]),
        swap_site_cost=5.0,
    )
    extra = 1.0 + 2.0 + 3.0 + 4.0 + 5.0 * swap
    result["objective"] += extra
    ops = build_operations(data, result)
    assert ops["summary"]["checks"]["passed"]
    assert sum(row["cny_per_hour"] for row in ops["tables"]["site_costs"]) == extra
    assert facility_costs(data) == pytest.approx([3.0, 4.0, 3.0, 4.0, 9.0])
    # Add a cabinet at each already open location: only cabinet cost changes.
    result["_solution"]["capacity_level"][:] = [0.0, 1.0]
    result["_solution"]["hub_count"][:] = 2.0
    result["objective"] += 6.0
    larger = build_operations(data, result)
    assert larger["summary"]["checks"]["passed"]
    assert larger["tables"]["site_costs"] == ops["tables"]["site_costs"]


def test_battery_reconstruction_includes_handling_wait_and_excludes_swap_duration():
    data, result = example(swap=True)
    tr = _trace(data, result["_missions"][0], service=True)
    qw = max(-0.05 * math.log(0.05) - 0.105, 0.0)
    assert tr["deterministic_intervals"] == pytest.approx([0.04, 0.26])
    assert tr["battery_interval_consumptions"] == pytest.approx([0.04, 0.26 + qw])
    assert tr["time"] == pytest.approx(0.325 + 0.05 * math.exp(-0.105 / 0.05))
    assert tr["components"]["return_swap"] == 0
    assert tr["waiting_interval"] == 1


@pytest.mark.parametrize(
    "damage,check",
    [
        ("capacity", "locker_capacity_violation"),
        ("fractional", "binary_integrality_error"),
        ("assignment", "assignment_service_error"),
        ("fleet", "fleet_occupancy_violation"),
        ("closed_link", "link_activation_violation"),
        ("endpoint", "link_endpoint_violation"),
        ("energy", "battery_quantile_violation_hours"),
        ("deadline", "delivery_quantile_violation_hours"),
        ("objective", "objective_reconstruction_error"),
        ("empty", "base_flow_balance_error"),
        ("saved_count", "saved_cabinet_count_error"),
        ("old_duration", "coefficient_reconstruction_error"),
    ],
)
def test_rejects_stale_or_infeasible_operational_output(damage, check):
    data, result = example()
    sol = result["_solution"]
    if damage == "capacity":
        data["level_capacity"][:] = 5.0
    elif damage == "fractional":
        sol["capacity_level"][0, 0] = 0.8
    elif damage == "assignment":
        sol["assignment"][0, 0, 1] = 0
    elif damage == "fleet":
        sol["fleet"] -= 1
    elif damage == "closed_link":
        sol["airline"][network(data)["lookup"][2]] = 0
    elif damage == "endpoint":
        sol["facility"][0] = 0
    elif damage == "energy":
        data["battery_time"] = 0.34
    elif damage == "deadline":
        data["delivery_deadline"][:] = 0.32
    elif damage == "objective":
        result["objective"] += 1
    elif damage == "empty":
        result["_relocations"][0]["flow"] -= 1
    elif damage == "saved_count":
        sol["hub_count"][0] += 1
    elif damage == "old_duration":
        result["_missions"][0]["trace"]["time"] = 999.0
    ops = build_operations(data, result)
    assert ops["summary"]["checks"][check] > 0
    assert not ops["summary"]["checks"]["passed"]


@pytest.mark.parametrize("leg", [[0, 4, 4, 2], [0, 1, 2], [0, 3, 2], [0, 0, 2]])
def test_path_structure_is_validated_independently(leg):
    data, result = example()
    result["_missions"][0]["trace"]["legs"][0] = leg
    with pytest.raises(ValueError):
        build_operations(data, result)


def test_shared_wait_uncertainty_gives_single_joint_reliability():
    data, result = example()
    # Both conditions bind at the same exponential quantile, not 1-2epsilon.
    qt = -0.05 * math.log(0.05)
    data["battery_time"] = 0.32 + qt - 0.1
    data["delivery_deadline"][:] = qt + 0.12 + 0.08
    ops = build_operations(data, result)
    assert ops["summary"]["min_joint_reliability"] == pytest.approx(0.95)
    assert ops["summary"]["checks"]["passed"]


def test_oracle_trace_is_checked_against_independent_reconstruction():
    from src.pricing.oracle import price_design
    from src.pricing.routes import trace_mission, trace_empty

    data, result = example()
    priced = price_design(data)
    mission = trace_mission(priced, 0, 0, 1)
    empty = trace_empty(priced, 1, 0)
    for tr, service in ((mission, True), (empty, False)):
        independent = _trace(data, tr, service=service)
        assert max(independent["coefficient_errors"].values()) < 1e-10
        assert independent["battery_violation"] == 0
        assert independent["delivery_violation"] == 0


def test_markdown_and_machine_outputs_are_exported_and_bad_result_fails(tmp_path):
    data, result = example()
    payload = export_operations(data, result, tmp_path / "valid")
    assert payload["summary"]["checks"]["passed"]
    text = (tmp_path / "valid/OPERATIONS.md").read_text()
    assert "空中等餐" in text and "纯骑手取餐后配送" in text
    assert (tmp_path / "valid/operations.json").exists()
    assert (tmp_path / "valid/airlines.csv").exists()
    result["objective"] += 1
    with pytest.raises(ValueError, match="validation failed"):
        export_operations(data, result, tmp_path / "invalid")


def test_no_service_solution_has_well_defined_mobile_report(tmp_path):
    data, result = example()
    sol = result["_solution"]
    for key, val in sol.items():
        sol[key] = np.zeros_like(val) if isinstance(val, np.ndarray) else 0.0
    result.update(
        _missions=[], _relocations=[], objective=160.0, fleet=0.0, served=0.0, bound=0.0
    )
    ops = export_operations(data, result, tmp_path)
    assert ops["summary"]["checks"]["passed"]
    assert ops["summary"]["time_averages_minutes"] == {}
    assert ops["summary"]["direct_postfetch_average_minutes"] is None


def test_current_recourse_positive_solution_reconciles_complete_objective():
    from src.optimization.recourse import solve

    data, result = example()
    data["ground"][:] = 12.0
    sol = result["_solution"]
    sp = solve(
        data,
        sol["facility"],
        sol["capacity_level"],
        sol["assignment"],
        sol["airline"],
        time_limit=5.0,
    )
    fixed = 4 + 6 + float(network(data)["fees"] @ sol["airline"])
    sol.update({k: sp[k] for k in ("rho", "idle", "empty", "fleet")})
    result.update(
        objective=sp["objective"] + fixed,
        bound=sp["bound"] + fixed,
        served=sp["served"],
        fleet=sp["fleet"],
        _missions=sp["missions"],
        _relocations=sp["relocations"],
    )
    ops = build_operations(data, result)
    assert sp["cg_complete"]
    assert ops["summary"]["served"] > 10
    assert ops["summary"]["empty_flow"] > 10
    assert ops["summary"]["checks"]["passed"]


def test_terminal_constant_and_zero_pickup_cost_are_reconciled():
    data, result = example()
    data["last_rider_cost"] = data["terminal_cost"] / 0.75
    data["terminal_dispatch_time"] = 0.02
    data["last_ground_time"] = data["terminal_time"] - 0.02
    data["observed_postfetch_hours"] = np.array([0.8])
    data["observed_postfetch_count"] = np.array([5])
    ops = build_operations(data, result)
    assert ops["summary"]["checks"]["passed"]
    assert ops["summary"]["terminal_dispatch_minutes"] == pytest.approx(1.2)
    assert ops["summary"]["observed_postfetch_average_minutes"] == pytest.approx(48.0)
    data["terminal_cost"][0, 1] += 1
    assert build_operations(data, result)["summary"]["checks"][
        "terminal_cost_reconstruction_error"
    ] == pytest.approx(1.0)
