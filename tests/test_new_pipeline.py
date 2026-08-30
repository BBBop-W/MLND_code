from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from gurobipy import GRB

import main as cli_main
from main import (
    FORMULATION_ID,
    FORMULATION_SOURCE,
    _data_config,
    _model_config,
    _screening_is_exact,
    build_parser,
)
from src.config import (
    ChoiceConfig,
    DataConfig,
    ModelConfig,
    SolverConfig,
    _window_hours,
)
from src.data_processor.generator import DataGenerator
from src.data_processor.instance import DIRECT_ROUTE, Instance
from src.data_processor.loader import DataLoader
from src.model.gurobi_model import MLNDModel
from src.model_contract import RESULT_SCHEMA_VERSION
from src.utils.instance_plotter import _require_current_summary_contract
from src.utils.output_processor import ResultProcessor
from src.utils.parameter_report import (
    _collect_scenario,
    _require_current_contract as require_parameter_contract,
)
from src.utils.scale_report import (
    _require_current_contract as require_scale_contract,
)
from src.utils.scale_report import collect_scale_results


TEST_RUN_SIGNATURE_SHA256 = "f" * 64


def _write_canonical_instance(root: Path, *, include_swap: bool = True) -> Path:
    root.mkdir()
    sites = [
        ("H001", "H", 0.0, 0.0, 0.11),
        ("H002", "H", 2.0, 0.0, ""),
        ("H003", "H", 1.0, 1.0, 0.33),
        ("L001", "L", 0.0, 1.0, ""),
        ("E001", "E", 1.0, 0.0, ""),
    ]
    if not include_swap:
        sites = [row for row in sites if row[1] != "E"]
    with open(root / "sites.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "site_id",
                "type",
                "x_km",
                "y_km",
                "activity_rate",
                "locker_pickup_cost_per_order",
            ]
        )
        writer.writerows((*row[:4], 10.0, row[4]) for row in sites)
    nodes = [
        ("O0001", "origin", -0.5, 0.0),
        ("O0002", "origin", 0.2, 0.2),
        ("D0001", "destination", 2.4, 0.0),
        ("D0002", "destination", 1.1, 1.2),
    ]
    with open(root / "od_nodes.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "role", "x_km", "y_km", "raw_endpoint_count"])
        writer.writerows((*row, 20) for row in nodes)
    with open(root / "od_demand.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "od_id",
                "origin_node",
                "destination_node",
                "order_count",
                "exposure_hours",
                "lambda_per_hour",
                "direct_distance_km",
            ]
        )
        writer.writerow(["OD000001", "O0001", "D0001", 200, 10, 20.0, 2.9])
        writer.writerow(["OD000002", "O0002", "D0002", 150, 10, 15.0, 1.35])
    with open(root / "periods.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["period", "start_hour", "end_hour", "exposure_hours"])
        writer.writerow(["demand", 12, 13, 10])
    with open(root / "metadata.json", "w") as handle:
        json.dump(
            {
                "schema_version": 2,
                "source": "test",
                "coordinate_system": "local_km",
                "demand_exposure_hours": 10,
            },
            handle,
        )
    return root


def _exact_model_config() -> ModelConfig:
    return ModelConfig(
        choice=ChoiceConfig(max_walk_km=None),
        battery_range_km=10.0,
        display_origin_choices=None,
        general_corridor_choices=None,
    )


def test_confirmed_physical_defaults_and_piecewise_rider_tariff() -> None:
    config = ModelConfig()
    assert config.drone_speed_mps == pytest.approx(20.0)
    assert config.drone_speed_kmph == pytest.approx(72.0)
    assert config.battery_range_km == pytest.approx(10.0)
    assert config.usable_battery_time_hours == pytest.approx(10.0 / 72.0)
    assert config.locker_service_rates == (30.0, 40.0, 50.0)
    assert config.locker_capacity_costs == (0.5, 1.0, 2.0)
    assert config.drone_cost_per_hour == pytest.approx(5.0)
    assert config.flight_cost_per_km == pytest.approx(0.20)
    assert config.feeder_rider_cost_factor == pytest.approx(0.30)
    assert config.display_origin_choices is None
    assert config.general_corridor_choices is None

    instance = Instance(
        coords={
            "A": (0.0, 0.0),
            "B": (2.4, 0.0),  # 3 road km after the 1.25 conversion.
            "C": (3.2, 0.0),  # 4 road km after the 1.25 conversion.
        },
        config=config,
    )
    assert instance.rider_cost("A", "A") == pytest.approx(6.4)
    assert instance.rider_cost("A", "B") == pytest.approx(6.4)
    assert instance.rider_cost("A", "C") == pytest.approx(8.2)
    assert instance.direct_rider_cost("A", "C") == pytest.approx(8.2)
    assert instance.feeder_rider_cost("A", "C") == pytest.approx(0.30 * 8.2)

    discounted = Instance(
        coords=instance.coords,
        config=ModelConfig(feeder_rider_cost_factor=0.6),
    )
    assert discounted.direct_rider_cost("A", "C") == pytest.approx(8.2)
    assert discounted.feeder_rider_cost("A", "C") == pytest.approx(0.6 * 8.2)


def test_loader_requires_local_kilometre_coordinates(tmp_path: Path) -> None:
    instance_dir = _write_canonical_instance(tmp_path / "wrong_coordinate_system")
    metadata_path = instance_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["coordinate_system"] = "wgs84"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="coordinate_system='local_km'"):
        DataLoader().load_instance_dir(str(instance_dir))


@pytest.mark.parametrize("validator", [require_parameter_contract, require_scale_contract])
def test_reports_reject_incompatible_summary_and_run_config_contracts(
    validator: object,
) -> None:
    current = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
    }
    validator(current, "run_config")

    with pytest.raises(ValueError, match="legacy/incompatible summary"):
        validator({**current, "schema_version": RESULT_SCHEMA_VERSION - 1}, "summary")
    with pytest.raises(ValueError, match="legacy/incompatible summary"):
        validator({**current, "schema_version": float(RESULT_SCHEMA_VERSION)}, "summary")
    with pytest.raises(ValueError, match="legacy/incompatible summary"):
        validator({**current, "schema_version": True}, "summary")
    with pytest.raises(ValueError, match="legacy/incompatible run_config"):
        validator({**current, "formulation_id": "legacy"}, "run_config")


def test_instance_plotter_rejects_float_schema_version() -> None:
    with pytest.raises(ValueError, match="solution layout requires summary"):
        _require_current_summary_contract(
            {
                "schema_version": float(RESULT_SCHEMA_VERSION),
                "formulation_id": FORMULATION_ID,
            }
        )


@pytest.mark.parametrize("mode", ["MX", "HD", "LO"])
def test_mode_specific_corridors_and_variable_families(
    tmp_path: Path,
    mode: str,
) -> None:
    instance_dir = _write_canonical_instance(tmp_path / mode)
    config = ModelConfig(
        choice=ChoiceConfig(max_walk_km=None),
        battery_range_km=10.0,
        display_origin_choices=1,
        general_corridor_choices=2,
    )
    instance = DataLoader().load_instance_dir(str(instance_dir), model_config=config)
    assert set(instance.h_od) == {("O0001", "D0001"), ("O0002", "D0002")}
    assert set(instance.h_od).isdisjoint(instance.A_HH)
    assert instance.locker_pickup_cost("H001") == pytest.approx(0.11)
    assert instance.locker_pickup_cost("H002") == pytest.approx(
        config.locker_pickup_cost_per_order
    )
    assert instance.locker_pickup_cost("H003") == pytest.approx(0.33)
    for arc in instance.A_Y - instance.A_F:
        assert instance.arc_activation_cost[arc] == pytest.approx(
            config.connection_cost_per_km_hour * instance.distance(*arc)
        )

    model = MLNDModel(instance, mode).build()
    assert model.market_corridors == {
        market: instance.corridors_for_mode(mode, market) for market in instance.markets
    }
    assert len(model.chi_keys) == sum(
        len(instance.corridors_for_mode(mode, market)) * len(config.availability_levels)
        for market in instance.markets
    )
    assert len(model.vars["chi"]) == len(model.chi_keys) > 0
    assert len(model.vars["alpha"]) == (0 if mode == "HD" else len(model.alpha_keys))
    assert len(model.vars["epsilon"]) == (
        len(model.epsilon_keys) if mode == "MX" else 0
    )
    assert bool(model.alpha_keys) is (mode in {"MX", "LO"})
    assert bool(model.epsilon_keys) is (mode == "MX")
    assert {
        "x",
        "z",
        "y",
        "delta",
        "alpha",
        "gamma",
        "chi",
        "epsilon",
        "w",
        "psi",
        "varphi",
        "xi",
        "queue_v1",
        "queue_v2",
        "queue_t",
    } == set(model.vars)
    assert {"h", "lambda", "a_lk", "a_hm", "q_lk", "q_hm"}.isdisjoint(model.vars)
    assert model.model.ModelSense == GRB.MINIMIZE
    assert model.model.getConstrByName("energy_system") is not None
    assert model.model.getConstrByName("energy_base[L001]") is None
    assert all(
        model.model.getConstrByName(f"energy_locker[{locker}]") is None
        for locker in instance.H
    )


def test_system_energy_counts_base_departures_and_swaps_once_each(
    tmp_path: Path,
) -> None:
    instance = DataLoader().load_instance_dir(
        str(_write_canonical_instance(tmp_path / "system_energy_coefficients")),
        model_config=_exact_model_config(),
    )
    wrapper = MLNDModel(instance, "HD").build()
    constraint = wrapper.model.getConstrByName("energy_system")
    row = wrapper.model.getRow(constraint)
    coefficients = {
        row.getVar(position).VarName: row.getCoeff(position)
        for position in range(row.size())
    }
    endurance = instance.config.usable_battery_time_hours

    base_direct = next(
        key
        for key in wrapper.varphi_keys
        if key[0] in instance.L and key[2] == DIRECT_ROUTE
    )
    base_swap = next(
        key
        for key in wrapper.varphi_keys
        if key[0] in instance.L and key[2] != DIRECT_ROUTE
    )
    locker_direct = next(
        key
        for key in wrapper.varphi_keys
        if key[0] in instance.H and key[2] == DIRECT_ROUTE
    )
    locker_swap = next(
        key
        for key in wrapper.varphi_keys
        if key[0] in instance.H and key[2] != DIRECT_ROUTE
    )

    def route_coefficient(key: tuple[str, str, str]) -> float:
        return coefficients[wrapper.vars["varphi"][key].VarName]

    assert route_coefficient(base_direct) == pytest.approx(
        instance.route_time[base_direct] - endurance
    )
    assert route_coefficient(base_swap) == pytest.approx(
        instance.route_time[base_swap] - 2.0 * endurance
    )
    assert route_coefficient(locker_direct) == pytest.approx(
        instance.route_time[locker_direct]
    )
    assert route_coefficient(locker_swap) == pytest.approx(
        instance.route_time[locker_swap] - endurance
    )
    assert coefficients[wrapper.vars["xi"][instance.H[0]].VarName] == pytest.approx(1.0)


def test_short_direct_closed_network_is_feasible_without_swap_station(
    tmp_path: Path,
) -> None:
    instance = DataLoader().load_instance_dir(
        str(
            _write_canonical_instance(
                tmp_path / "direct_only_energy",
                include_swap=False,
            )
        ),
        model_config=_exact_model_config(),
    )
    assert instance.E == []
    wrapper = MLNDModel(instance, "HD").build()
    forced_key = wrapper.chi_keys[0]
    _origin, _destination, locker_origin, locker, rho_index = forced_key
    forced_base_source = ("L001", locker_origin, locker, rho_index)
    assert forced_base_source in wrapper.vars["w"]
    for key, variable in wrapper.vars["chi"].items():
        variable.UB = 1.0 if key == forced_key else 0.0
        if key == forced_key:
            variable.LB = 1.0
    wrapper.vars["w"][forced_base_source].LB = 1.0
    wrapper.model.Params.OutputFlag = 0
    wrapper.model.Params.Threads = 1
    wrapper.model.optimize()

    assert wrapper.model.Status == GRB.OPTIMAL
    assert sum(variable.X for variable in wrapper.vars["chi"].values()) > 1e-8
    processor = ResultProcessor(
        wrapper,
        instance,
        run_signature_sha256=TEST_RUN_SIGNATURE_SHA256,
    )
    energy = processor.summary["energy_balance"]
    assert energy["base_full_battery_departures_per_hour"] > 1e-8
    assert energy["swap_events_per_hour"] == pytest.approx(0.0, abs=1e-8)
    assert energy["maximum_constraint_violation"] <= 1e-7


def test_corridor_selection_rules_and_complete_screening(tmp_path: Path) -> None:
    screened_dir = _write_canonical_instance(tmp_path / "screened")
    screened = DataLoader().load_instance_dir(
        str(screened_dir),
        model_config=ModelConfig(
            choice=ChoiceConfig(max_walk_km=None),
            battery_range_km=10.0,
            display_origin_choices=1,
            general_corridor_choices=2,
        ),
    )
    for market in screened.markets:
        origin, destination = market
        display = tuple(
            dict.fromkeys(
                (locker_origin, locker)
                for locker in screened.display_lockers[destination]
                for locker_origin in screened.display_origins[
                    origin,
                    destination,
                    locker,
                ]
            )
        )
        general = screened.general_corridors[market]
        assert screened.corridors_for_mode("HD", market) == general
        assert screened.corridors_for_mode("LO", market) == display
        assert screened.corridors_for_mode("MX", market) == tuple(
            dict.fromkeys((*general, *display))
        )

    exact_dir = _write_canonical_instance(tmp_path / "exact")
    exact = DataLoader().load_instance_dir(
        str(exact_dir),
        model_config=_exact_model_config(),
    )
    for market in exact.markets:
        origin, destination = market
        assert set(exact.general_corridors[market]) == exact.A_HH
        assert set(exact.display_lockers[destination]) == set(exact.H)
        for locker in exact.H:
            assert set(exact.display_origins[origin, destination, locker]) == {
                i for i in exact.H if (i, locker) in exact.A_HH
            }
    assert _screening_is_exact(exact, ("MX", "HD", "LO")) == {
        "MX": True,
        "HD": True,
        "LO": True,
    }

    finite_walk = DataLoader().load_instance_dir(
        str(exact_dir),
        model_config=ModelConfig(
            choice=ChoiceConfig(max_walk_km=0.6),
            display_origin_choices=None,
            general_corridor_choices=None,
        ),
    )
    assert any(
        set(finite_walk.display_lockers[destination]) != set(finite_walk.H)
        for _origin, destination in finite_walk.markets
    )
    assert _screening_is_exact(finite_walk, ("MX", "HD", "LO")) == {
        "MX": True,
        "HD": True,
        "LO": True,
    }


@pytest.mark.parametrize("mode", ["MX", "HD", "LO"])
def test_solved_costs_and_conservation_reconcile(
    tmp_path: Path,
    mode: str,
) -> None:
    instance_dir = _write_canonical_instance(tmp_path / f"solved_{mode}")
    instance = DataLoader().load_instance_dir(
        str(instance_dir),
        model_config=_exact_model_config(),
    )
    model = MLNDModel(instance, mode).build()
    solved = model.solve(
        SolverConfig(
            time_limit=20,
            mip_gap=0.0,
            threads=1,
            output_flag=0,
            network_start_seconds=0.0,
            use_valid_inequalities=False,
            use_user_cuts=False,
            use_local_branching=False,
        )
    )
    assert solved
    assert model.model.Status == GRB.OPTIMAL
    assert model.model.ObjVal <= model.courier_only_baseline + 1e-5

    processor = ResultProcessor(
        model,
        instance,
        run_signature_sha256=TEST_RUN_SIGNATURE_SHA256,
    )
    summary = processor.summary
    # Gurobi's default feasibility tolerance is 1e-6.  Reconciliations that
    # aggregate independently rounded conic/OD values use a small margin.
    solver_tolerance = 2e-6
    expected_costs = {
        "facility",
        "capacity",
        "connection",
        "route_operation",
        "fleet",
        "direct_courier_fallback",
        "first_mile_courier",
        "last_mile_courier",
        "locker_handling",
    }
    assert set(model.cost_expressions) == expected_costs
    assert set(processor.costs) == expected_costs
    assert summary["schema_version"] == RESULT_SCHEMA_VERSION
    assert summary["formulation_id"] == FORMULATION_ID
    assert summary["run_signature_sha256"] == TEST_RUN_SIGNATURE_SHA256
    assert 0 <= summary["active_lockers"] <= summary["open_lockers"]
    assert summary["idle_open_lockers"] == (
        summary["open_lockers"] - summary["active_lockers"]
    )
    assert summary["facilities"]["active_lockers"] == summary["active_lockers"]
    assert summary["facilities"]["idle_open_lockers"] == summary["idle_open_lockers"]
    assert summary["network_opened"] is any(summary["facilities"]["open"].values())
    assert summary["screening_is_exact"] is True
    assert summary["costs_cny_per_hour"]["optimized_total"] == pytest.approx(
        sum(summary["costs_cny_per_hour"][name] for name in expected_costs),
        abs=solver_tolerance,
    )
    assert summary["costs_cny_per_hour"]["optimized_total"] == pytest.approx(
        model.model.ObjVal,
        abs=solver_tolerance,
    )
    assert summary["costs_cny_per_hour"][
        "objective_reconciliation_error"
    ] == pytest.approx(
        0.0,
        abs=solver_tolerance,
    )
    reconciliation = summary["reconciliation"]
    od_cost_errors = reconciliation["od_fulfillment_cost_errors"]
    assert (
        max(
            abs(value)
            for key, value in reconciliation.items()
            if key != "od_fulfillment_cost_errors"
        )
        <= solver_tolerance
    )
    assert (
        max((abs(value) for value in od_cost_errors.values()), default=0.0)
        <= solver_tolerance
    )
    assert summary["paper_objective_terms_cny_per_hour"][
        "objective_reconciliation_error"
    ] == pytest.approx(0.0, abs=solver_tolerance)
    assert summary["energy_balance"]["battery_range_km"] == pytest.approx(10.0)
    assert summary["energy_balance"]["drone_speed_mps"] == pytest.approx(20.0)
    assert summary["energy_balance"]["usable_battery_time_minutes"] == pytest.approx(
        10.0 / 72.0 * 60.0
    )
    assert summary["energy_balance"]["maximum_constraint_violation"] <= solver_tolerance
    assert all(
        row["energy_violation_equivalent_drones"] <= solver_tolerance
        for row in processor.energy_rows
    )
    assert (
        abs(summary["chaining"]["source_flow_reconciliation_error"]) <= solver_tolerance
    )
    assert (
        abs(summary["chaining"]["virtual_flow_reconciliation_error"])
        <= solver_tolerance
    )
    for node in instance.L + instance.H + instance.S:
        inflow = sum(processor.psi.get((u, v), 0.0) for u, v in instance.A if v == node)
        outflow = sum(
            processor.psi.get((u, v), 0.0) for u, v in instance.A if u == node
        )
        assert inflow == pytest.approx(outflow, abs=solver_tolerance)
    if mode == "HD":
        assert summary["demand_funnel_orders_per_hour"][
            "locker_pickup"
        ] == pytest.approx(0.0)
    if mode == "LO":
        assert summary["demand_funnel_orders_per_hour"][
            "assisted_home"
        ] == pytest.approx(0.0)

    if mode == "MX":
        output = tmp_path / "output"
        summary = processor.write_all(str(output))
        assert "active_virtual_state_count" in summary["discrete_availability"]
        assert isinstance(summary["network_is_nontrivial"], bool)
        assert summary["requested_mip_gap"] == pytest.approx(0.0)
        assert summary["open_bases"] == summary["facilities"]["open"]["bases"]
        assert summary["open_lockers"] == summary["facilities"]["open"]["lockers"]
        assert (
            summary["selected_rho_by_idle_state"]
            == summary["discrete_availability"]["selected_rho_by_idle_state"]
        )
        assert summary["solver"]["algorithm"]["name"] == (
            "chi_epsilon_perspective_branch_and_cut"
        )
        assert (
            json.loads((output / "summary.json").read_text())["schema_version"]
            == RESULT_SCHEMA_VERSION
        )
        for filename in (
            "od_corridor_reliability.csv",
            "displayed_locker_service.csv",
            "energy_balance.csv",
        ):
            assert (output / filename).stat().st_size > 0
        for stem in (
            "network_map",
            "od_service_map",
            "cost_breakdown",
            "demand_funnel",
            "locker_utilization",
        ):
            assert (output / f"{stem}.png").stat().st_size > 0
            assert (output / f"{stem}.svg").stat().st_size > 0

        report_run_config = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "formulation_id": FORMULATION_ID,
            "run_signature_sha256": TEST_RUN_SIGNATURE_SHA256,
            "modes": ["MX"],
            "implementation": {"sha256": "a" * 64},
            "formulation_source": {"sha256": "b" * 64},
            "demand_scale": 1.0,
            "model": cli_main.to_dict(instance.config),
            "model_derived": {
                "drone_speed_kmph": instance.config.drone_speed_kmph,
                "usable_battery_time_hours": (
                    instance.config.usable_battery_time_hours
                ),
                "usable_battery_time_minutes": (
                    60.0 * instance.config.usable_battery_time_hours
                ),
            },
            "solver": {
                "time_limit": 20.0,
                "mip_gap": 0.0,
                "threads": 1,
                "seed": 0,
            },
            "instance": {
                "directory": str(instance_dir),
                "metadata": instance.metadata,
                "canonical_file_sha256": {
                    filename: hashlib.sha256(
                        (instance_dir / filename).read_bytes()
                    ).hexdigest()
                    for filename in (
                        "od_nodes.csv",
                        "od_demand.csv",
                        "periods.csv",
                    )
                },
            },
        }
        parameter_row = _collect_scenario(
            tmp_path,
            tmp_path / "scenario",
            report_run_config,
            summary,
        )
        assert parameter_row["battery_range_km"] == pytest.approx(10.0)
        assert parameter_row["drone_speed_mps"] == pytest.approx(20.0)
        assert parameter_row["usable_battery_time_minutes"] == pytest.approx(
            10.0 / 72.0 * 60.0
        )
        assert parameter_row["maximum_energy_constraint_violation"] <= 1e-7

        scale_run = tmp_path / "scale" / f"H{len(instance.H):03d}"
        scale_run_config = json.loads(json.dumps(report_run_config))
        scale_run_config.update(
            {
                "modes": ["HD", "MX"],
                "cross_mode_start_policy": "HD_to_later_MX_by_variable_name",
                "warm_start_from": None,
            }
        )
        hd_summary = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "formulation_id": FORMULATION_ID,
            "run_signature_sha256": TEST_RUN_SIGNATURE_SHA256,
            "mode": "HD",
            "solver": {
                "runtime_seconds": 1.0,
                "external_start_installed": False,
                "external_start_source": None,
                "external_start_matched_variables": 0,
            },
        }
        scale_summary = json.loads(json.dumps(summary))
        scale_summary["solver"].update(
            {
                "external_start_installed": True,
                "external_start_source": "HD",
                "external_start_matched_variables": 1,
            }
        )
        (scale_run / "HD").mkdir(parents=True)
        (scale_run / "MX").mkdir(parents=True)
        (scale_run / "run_config.json").write_text(
            json.dumps(scale_run_config), encoding="utf-8"
        )
        (scale_run / "HD" / "summary.json").write_text(
            json.dumps(hd_summary), encoding="utf-8"
        )
        (scale_run / "MX" / "summary.json").write_text(
            json.dumps(scale_summary), encoding="utf-8"
        )
        scale_rows = collect_scale_results(tmp_path / "scale")
        assert len(scale_rows) == 1
        assert scale_rows[0]["maximum_energy_constraint_violation"] <= 1e-7
        assert scale_rows[0]["battery_range_km"] == pytest.approx(10.0)
        assert scale_rows[0]["drone_speed_mps"] == pytest.approx(20.0)

        incompatible_run = tmp_path / "scale" / "H999"
        (incompatible_run / "HD").mkdir(parents=True)
        (incompatible_run / "MX").mkdir(parents=True)
        incompatible_config = json.loads(json.dumps(scale_run_config))
        incompatible_config["model"]["battery_range_km"] = 9.0
        (incompatible_run / "run_config.json").write_text(
            json.dumps(incompatible_config), encoding="utf-8"
        )
        (incompatible_run / "HD" / "summary.json").write_text(
            json.dumps(hd_summary), encoding="utf-8"
        )
        (incompatible_run / "MX" / "summary.json").write_text(
            json.dumps(scale_summary), encoding="utf-8"
        )
        with pytest.raises(
            ValueError, match="inconsistent candidate-scale run contract"
        ):
            collect_scale_results(tmp_path / "scale")


def test_mx_display_does_not_force_chi(tmp_path: Path) -> None:
    instance = DataLoader().load_instance_dir(
        str(_write_canonical_instance(tmp_path / "mx_optional_offer")),
        model_config=_exact_model_config(),
    )
    wrapper = MLNDModel(instance, "MX").build()
    destination = "D0001"
    locker = min(
        instance.display_lockers[destination],
        key=lambda candidate: (instance.distance(destination, candidate), candidate),
    )
    wrapper.vars["alpha"][destination, locker].LB = 1.0
    for variable in wrapper.vars["chi"].values():
        variable.UB = 0.0
    wrapper.model.Params.OutputFlag = 0
    wrapper.model.Params.Threads = 1
    wrapper.model.optimize()

    assert wrapper.model.Status == GRB.OPTIMAL
    assert wrapper.vars["alpha"][destination, locker].X == pytest.approx(1.0)
    assert sum(
        variable.X for variable in wrapper.vars["chi"].values()
    ) == pytest.approx(0.0)
    assert sum(
        variable.X for variable in wrapper.vars["epsilon"].values()
    ) == pytest.approx(0.0)
    assert not any(
        constraint.ConstrName.startswith("lo_locker_allocation")
        for constraint in wrapper.model.getConstrs()
    )


def test_energy_constraints_preserve_all_courier_fallback_without_swap_sites() -> None:
    config = ModelConfig(
        choice=ChoiceConfig(max_walk_km=None),
        display_origin_choices=None,
        general_corridor_choices=None,
    )
    instance = Instance(
        L=["L001"],
        H=["H001", "H002"],
        N=["O001", "D001"],
        coords={
            "L001": (0.0, 1.0),
            "H001": (0.0, 0.0),
            "H002": (2.0, 0.0),
            "O001": (-0.2, 0.0),
            "D001": (2.2, 0.0),
        },
        h_od={("O001", "D001"): 2.0},
        config=config,
    ).build()
    wrapper = MLNDModel(instance, "MX").build()
    wrapper.model.Params.OutputFlag = 0
    wrapper.model.Params.Threads = 1
    wrapper.model.optimize()

    assert wrapper.model.Status == GRB.OPTIMAL
    assert sum(
        variable.X for variable in wrapper.vars["chi"].values()
    ) == pytest.approx(0.0)
    assert wrapper.model.ObjVal == pytest.approx(wrapper.courier_only_baseline)
    processor = ResultProcessor(
        wrapper,
        instance,
        run_signature_sha256=TEST_RUN_SIGNATURE_SHA256,
    )
    assert processor.summary["energy_balance"]["maximum_constraint_violation"] <= 1e-7


def test_mx_nonzero_epsilon_and_locker_pickup_reconcile(tmp_path: Path) -> None:
    instance = DataLoader().load_instance_dir(
        str(_write_canonical_instance(tmp_path / "mx_nonzero_epsilon")),
        model_config=_exact_model_config(),
    )
    wrapper = MLNDModel(instance, "MX").build()
    origin, destination = ("O0001", "D0001")
    locker = min(
        instance.display_lockers[destination],
        key=lambda candidate: (instance.distance(destination, candidate), candidate),
    )
    locker_origin = instance.display_origins[origin, destination, locker][0]
    rho_index = 0
    forced_key = (origin, destination, locker_origin, locker, rho_index)
    forced_base_source = ("L001", locker_origin, locker, rho_index)
    assert forced_base_source in wrapper.vars["w"]
    wrapper.vars["alpha"][destination, locker].LB = 1.0
    for key, variable in wrapper.vars["chi"].items():
        variable.UB = 1.0 if key == forced_key else 0.0
        if key == forced_key:
            variable.LB = 1.0
    wrapper.vars["w"][forced_base_source].LB = 1.0
    for (_u, _v, option), variable in wrapper.vars["varphi"].items():
        if option != DIRECT_ROUTE:
            variable.UB = 0.0
    wrapper.model.Params.OutputFlag = 0
    wrapper.model.Params.Threads = 1
    wrapper.model.optimize()

    assert wrapper.model.Status == GRB.OPTIMAL
    successful = instance.config.availability_levels[rho_index]
    epsilon = wrapper.vars["epsilon"][origin, destination, locker].X
    assert epsilon == pytest.approx(successful, abs=1e-7)

    processor = ResultProcessor(
        wrapper,
        instance,
        run_signature_sha256=TEST_RUN_SIGNATURE_SHA256,
    )
    swap_flow = sum(
        variable.X
        for (_u, _v, option), variable in wrapper.vars["varphi"].items()
        if option != DIRECT_ROUTE
    )
    assert swap_flow == pytest.approx(0.0, abs=1e-8)
    assert (
        sum(row["energy_use_equivalent_drones"] for row in processor.energy_rows) > 1e-8
    )
    assert len(processor.energy_rows) == 1
    system_energy = processor.energy_rows[0]
    assert system_energy["balance_scope"] == "system"
    assert system_energy["energy_use_equivalent_drones"] > 1e-8
    assert system_energy["base_full_battery_departures_per_hour"] > 1e-8
    assert system_energy["swap_events_per_hour"] == pytest.approx(0.0, abs=1e-8)
    assert system_energy["battery_supply_events_per_hour"] == pytest.approx(
        system_energy["base_full_battery_departures_per_hour"], abs=1e-8
    )
    assert (
        max(row["energy_violation_equivalent_drones"] for row in processor.energy_rows)
        <= 1e-7
    )
    display_row = next(
        row
        for row in processor.display_rows
        if (row["origin"], row["destination"], row["locker"])
        == (origin, destination, locker)
    )
    eta = instance.choice_share[origin, destination, locker]
    assert display_row["alpha_times_successful_per_hour"] == pytest.approx(
        successful,
        abs=1e-7,
    )
    assert display_row["epsilon_product_error"] == pytest.approx(0.0, abs=1e-7)
    assert display_row["locker_pickup_per_hour"] == pytest.approx(
        eta * epsilon,
        abs=1e-7,
    )
    od_row = next(
        row
        for row in processor.od_rows
        if (row["origin"], row["destination"]) == (origin, destination)
    )
    assert od_row["locker_pickup_per_hour"] == pytest.approx(
        eta * epsilon,
        abs=1e-7,
    )


def test_hd_nonzero_realized_flow_reconciles_energy(tmp_path: Path) -> None:
    instance = DataLoader().load_instance_dir(
        str(_write_canonical_instance(tmp_path / "hd_nonzero_flow")),
        model_config=_exact_model_config(),
    )
    wrapper = MLNDModel(instance, "HD").build()
    forced_key = wrapper.chi_keys[0]
    origin, destination, locker_origin, locker, rho_index = forced_key
    forced_base_source = ("L001", locker_origin, locker, rho_index)
    assert forced_base_source in wrapper.vars["w"]
    for key, variable in wrapper.vars["chi"].items():
        variable.UB = 1.0 if key == forced_key else 0.0
        if key == forced_key:
            variable.LB = 1.0
    wrapper.vars["w"][forced_base_source].LB = 1.0
    for (_u, _v, option), variable in wrapper.vars["varphi"].items():
        if option != DIRECT_ROUTE:
            variable.UB = 0.0
    wrapper.model.Params.OutputFlag = 0
    wrapper.model.Params.Threads = 1
    wrapper.model.optimize()

    assert wrapper.model.Status == GRB.OPTIMAL
    processor = ResultProcessor(
        wrapper,
        instance,
        run_signature_sha256=TEST_RUN_SIGNATURE_SHA256,
    )
    successful = instance.config.availability_levels[rho_index]
    funnel = processor.summary["demand_funnel_orders_per_hour"]
    assert funnel["total_drone_realized"] == pytest.approx(successful, abs=1e-7)
    assert funnel["locker_pickup"] == pytest.approx(0.0, abs=1e-7)
    assert funnel["assisted_home"] == pytest.approx(successful, abs=1e-7)
    assert funnel["direct_courier_fallback"] == pytest.approx(
        instance.total_demand - successful,
        abs=1e-7,
    )
    assert sum(
        variable.X
        for (_u, _v, option), variable in wrapper.vars["varphi"].items()
        if option != DIRECT_ROUTE
    ) == pytest.approx(0.0, abs=1e-8)
    energy = processor.summary["energy_balance"]
    assert energy["base_full_battery_departures_per_hour"] > 1e-8
    assert energy["swap_events_per_hour"] == pytest.approx(0.0, abs=1e-8)
    assert energy["battery_supply_events_per_hour"] == pytest.approx(
        energy["base_full_battery_departures_per_hour"], abs=1e-8
    )
    assert processor.summary["energy_balance"]["maximum_constraint_violation"] <= 1e-7
    assert (origin, destination) in instance.h_od


def test_lo_choice_allocation_is_an_equality(tmp_path: Path) -> None:
    instance = DataLoader().load_instance_dir(
        str(_write_canonical_instance(tmp_path / "lo_choice")),
        model_config=_exact_model_config(),
    )
    wrapper = MLNDModel(instance, "LO").build()
    destination = "D0001"
    locker = min(
        instance.display_lockers[destination],
        key=lambda candidate: (instance.distance(destination, candidate), candidate),
    )
    wrapper.vars["alpha"][destination, locker].LB = 1.0
    wrapper.model.Params.OutputFlag = 0
    wrapper.model.Params.Threads = 1
    wrapper.model.optimize()

    assert wrapper.model.Status == GRB.OPTIMAL
    assert wrapper.vars["alpha"][destination, locker].X == pytest.approx(1.0)
    for origin, destination in instance.markets:
        for candidate in instance.display_lockers[destination]:
            offered = sum(
                wrapper.vars["chi"][key].X
                for key in wrapper.chi_by_market_j[origin, destination, candidate]
            )
            expected = (
                instance.h_od[origin, destination]
                * instance.choice_share[origin, destination, candidate]
                * wrapper.vars["alpha"][destination, candidate].X
            )
            assert offered == pytest.approx(expected, abs=2e-6)

    processor = ResultProcessor(
        wrapper,
        instance,
        run_signature_sha256=TEST_RUN_SIGNATURE_SHA256,
    )
    funnel = processor.summary["demand_funnel_orders_per_hour"]
    assert funnel["total_drone_offered"] > 1e-8
    assert funnel["total_drone_realized"] > 1e-8
    assert funnel["locker_pickup"] == pytest.approx(
        funnel["total_drone_realized"], abs=1e-7
    )
    assert funnel["assisted_home"] == pytest.approx(0.0, abs=1e-7)
    energy = processor.summary["energy_balance"]
    assert energy["base_full_battery_departures_per_hour"] > 1e-8
    assert (
        energy["battery_supply_events_per_hour"]
        >= energy["base_full_battery_departures_per_hour"]
    )
    assert energy["maximum_constraint_violation"] <= 1e-7


def test_branch_and_cut_preserves_the_certified_small_instance_optimum(
    tmp_path: Path,
) -> None:
    instance_dir = _write_canonical_instance(tmp_path / "branch_cut_equivalence")
    config = ModelConfig(
        choice=ChoiceConfig(max_walk_km=None),
        battery_range_km=10.0,
        display_origin_choices=None,
        general_corridor_choices=None,
    )
    objectives = []
    strengthened_model = None
    for strengthened in (False, True):
        instance = DataLoader().load_instance_dir(
            str(instance_dir), model_config=config
        )
        model = MLNDModel(instance, "MX")
        solved = model.solve(
            SolverConfig(
                time_limit=20.0,
                mip_gap=0.0,
                threads=1,
                output_flag=0,
                network_start_seconds=0.0,
                use_valid_inequalities=strengthened,
                use_user_cuts=strengthened,
                use_local_branching=False,
                max_root_cut_rounds=3,
                max_cuts_per_round=100,
            )
        )
        assert solved and model.model.Status == GRB.OPTIMAL
        objectives.append(model.model.ObjVal)
        if strengthened:
            strengthened_model = model

    # Both solves are certified optimal; sub-micro objective drift comes from
    # different conic/root paths and remains within solver feasibility scale.
    assert objectives[1] == pytest.approx(objectives[0], abs=2e-6)
    assert strengthened_model is not None
    assert strengthened_model.static_cut_counts
    assert 0 <= strengthened_model.root_cut_rounds <= 3
    assert strengthened_model.model.Params.PreCrush == 1
    assert strengthened_model.model.Params.MIPFocus == 1
    assert strengthened_model.model.Params.Heuristics == pytest.approx(0.25)
    assert strengthened_model.model.Params.NoRelHeurTime == pytest.approx(1.0)
    assert all(count >= 0 for count in strengthened_model.user_cut_counts.values())


@pytest.mark.parametrize("mode", ["MX", "HD"])
def test_default_algorithm_path_runs_with_energy_constraints(
    tmp_path: Path,
    mode: str,
) -> None:
    instance = DataLoader().load_instance_dir(
        str(_write_canonical_instance(tmp_path / "default_algorithm")),
        model_config=_exact_model_config(),
    )
    wrapper = MLNDModel(instance, mode)
    original_x_bounds = {
        key: (float(variable.LB), float(variable.UB))
        for key, variable in wrapper.build().vars["x"].items()
    }
    solved = wrapper.solve(
        SolverConfig(
            time_limit=10.0,
            mip_gap=0.0,
            threads=1,
            output_flag=0,
            network_start_seconds=2.0,
            use_valid_inequalities=False,
            use_user_cuts=True,
            max_root_cut_rounds=2,
            max_cuts_per_round=50,
            use_local_branching=True,
            local_branching_radius=6,
            root_relaxation_method=2,
            miqcp_method=1,
        )
    )

    assert solved and wrapper.model.Status == GRB.OPTIMAL
    assert wrapper.network_start_found
    assert wrapper.network_start_used
    assert wrapper.network_partial_start_installed
    assert wrapper.network_partial_start_index == 1
    assert wrapper.network_partial_start_variable_count > 0
    assert set(wrapper.network_start_seed_open_counts) == {
        "bases",
        "lockers",
        "swap_stations",
    }
    assert sum(wrapper.network_start_seed_open_counts.values()) > 0
    assert {
        key: (float(variable.LB), float(variable.UB))
        for key, variable in wrapper.vars["x"].items()
    } == original_x_bounds
    assert wrapper.local_branching_status is not None
    assert wrapper.model.Params.PreCrush == 1
    assert wrapper.model.Params.MIQCPMethod == 1
    processor = ResultProcessor(
        wrapper,
        instance,
        run_signature_sha256=TEST_RUN_SIGNATURE_SHA256,
    )
    algorithm = processor.summary["solver"]["algorithm"]
    assert algorithm["miqcp_method"] == 1
    assert algorithm["installed_mip_start_count"] == 3
    assert algorithm["network_partial_start_installed"] is True
    assert algorithm["network_partial_start_index"] == 1
    assert algorithm["network_partial_start_variable_count"] > 0
    assert algorithm["network_partial_start_groups"] == [
        "x",
        "z",
        "delta",
        "alpha",
    ]
    assert (
        algorithm["network_start_seed_open_counts"]
        == wrapper.network_start_seed_open_counts
    )
    assert processor.summary["energy_balance"]["maximum_constraint_violation"] <= 1e-7


def test_hd_incumbent_is_a_feasible_equal_cost_mx_start(tmp_path: Path) -> None:
    instance = DataLoader().load_instance_dir(
        str(_write_canonical_instance(tmp_path / "hd_to_mx_start")),
        model_config=ModelConfig(
            choice=ChoiceConfig(max_walk_km=None),
            battery_range_km=10.0,
            display_origin_choices=None,
            general_corridor_choices=None,
            feeder_rider_cost_factor=0.1,
            base_cost_per_hour=0.1,
            locker_fixed_cost_per_hour=0.1,
            swap_cost_per_hour=0.1,
            locker_capacity_costs=(0.1, 0.2, 0.3),
            corridor_cost_per_km_hour=0.001,
            connection_cost_per_km_hour=0.001,
            flight_cost_per_km=0.01,
            drone_cost_per_hour=0.1,
        ),
    )
    solver = SolverConfig(
        time_limit=20.0,
        mip_gap=0.0,
        threads=1,
        output_flag=0,
        network_start_seconds=2.0,
        use_valid_inequalities=False,
        use_user_cuts=False,
        use_local_branching=False,
    )
    hd = MLNDModel(instance, "HD")
    assert hd.solve(solver) and hd.model.Status == GRB.OPTIMAL
    portable = hd.export_incumbent_start()
    assert sum(variable.X for variable in hd.vars["chi"].values()) > 1e-8

    mx = MLNDModel(instance, "MX").build()
    mx._append_external_start(portable, "HD")
    assert mx.external_start_installed
    assert mx.external_start_source == "HD"
    assert mx.external_start_matched_variables > 0
    for variable in mx.model.getVars():
        value = portable.get(variable.VarName, 0.0)
        if variable.VType in {GRB.BINARY, GRB.INTEGER}:
            value = round(value)
        variable.LB = value
        variable.UB = value
    mx.model.Params.OutputFlag = 0
    mx.model.optimize()

    assert mx.model.SolCount > 0
    assert mx.model.ObjVal == pytest.approx(hd.model.ObjVal, abs=1e-5)


def _ts(day: str, hour: int) -> int:
    local = datetime.strptime(f"{day} {hour}:00", "%Y%m%d %H:%M").replace(
        tzinfo=ZoneInfo("Asia/Shanghai")
    )
    return int(local.timestamp())


def _write_tsl_sample(path: Path) -> None:
    fields = [
        "order_id",
        "dt",
        "is_courier_grabbed",
        "platform_order_time",
        "sender_lng",
        "sender_lat",
        "recipient_lng",
        "recipient_lat",
        "poi_id",
    ]
    rows = [
        (
            "1",
            "20221017",
            "1",
            _ts("20221016", 12),
            114000000,
            22000000,
            114004000,
            22002000,
            "p1",
        ),
        (
            "2",
            "20221017",
            "1",
            _ts("20221017", 12),
            114002000,
            22001000,
            114006000,
            22003000,
            "p1",
        ),
        (
            "3",
            "20221018",
            "1",
            _ts("20221018", 12),
            113998000,
            22002000,
            114003000,
            22004000,
            "p2",
        ),
        (
            "4",
            "20221018",
            "1",
            _ts("20221018", 12),
            114001000,
            21999000,
            114005000,
            22001000,
            "p2",
        ),
        (
            "1",
            "20221017",
            "0",
            _ts("20221017", 12),
            114000000,
            22000000,
            114004000,
            22002000,
            "p1",
        ),
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for values in rows:
            writer.writerow(dict(zip(fields, values)))


def test_tsl_filtering_average_and_candidate_invariance(tmp_path: Path) -> None:
    raw = tmp_path / "tsl.csv"
    _write_tsl_sample(raw)
    totals = []
    market_counts = []
    od_payloads = []
    for candidate_count in (2, 3):
        output = tmp_path / f"h{candidate_count}"
        DataGenerator().generate(
            DataConfig(
                source="tsl",
                raw_path=str(raw),
                output_dir=str(output),
                candidate_count=candidate_count,
                num_bases=1,
                num_swaps=1,
                od_grid_km=0.2,
                site_grid_km=0.1,
            )
        )
        with open(output / "metadata.json") as handle:
            metadata = json.load(handle)
        totals.append(metadata["total_demand_per_hour"])
        market_counts.append(metadata["od_market_count"])
        od_payloads.append(
            (
                (output / "od_nodes.csv").read_bytes(),
                (output / "od_demand.csv").read_bytes(),
            )
        )
        assert metadata["source_orders_all_days"] == 4
        assert metadata["accepted_waybills_all_days"] == 4
        assert metadata["demand_exposure_hours"] == 2
        assert metadata["date_basis"] == "source_partition_dt"
        assert metadata["timestamp_date_mismatch_orders_all_days"] == 1
        assert len(metadata["raw_file_sha256"]) == 64
    assert totals == pytest.approx([2.0, 2.0])
    assert market_counts[0] == market_counts[1]
    assert od_payloads[0] == od_payloads[1]


def test_dates_are_unique_and_must_exist(tmp_path: Path) -> None:
    raw = tmp_path / "tsl.csv"
    _write_tsl_sample(raw)
    output = tmp_path / "unique_dates"
    DataGenerator().generate(
        DataConfig(
            source="tsl",
            raw_path=str(raw),
            output_dir=str(output),
            candidate_count=2,
            num_bases=1,
            num_swaps=1,
            dates=("20221017", "20221017"),
            od_grid_km=0.2,
            site_grid_km=0.1,
        )
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["selected_days"] == ["20221017"]
    assert metadata["demand_exposure_hours"] == 1
    assert metadata["total_demand_per_hour"] == pytest.approx(2.0)

    with pytest.raises(ValueError, match="requested dates are not present"):
        DataGenerator().generate(
            DataConfig(
                source="tsl",
                raw_path=str(raw),
                output_dir=str(tmp_path / "invalid_date"),
                candidate_count=2,
                num_bases=1,
                num_swaps=1,
                dates=("20990101",),
            )
        )


def _write_nanshan_sample(path: Path) -> None:
    fields = [
        "dt_seq_id",
        "order_seq_id",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "customer_pay_time",
        "fetch_time",
        "arrived_time",
        "package_price",
        "biz_line",
    ]
    rows = [
        ("1", "1", 22.540, 113.940, 22.545, 113.950, "12:05", "", "", 20, "waimai"),
        ("1", "2", 22.550, 113.950, 22.555, 113.960, "12:15", "", "", 25, "waimai"),
        ("2", "3", 22.560, 113.960, 22.565, 113.970, "12:25", "", "", 30, "waimai"),
        ("2", "4", 22.570, 113.970, 22.575, 113.980, "12:35", "", "", 35, "waimai"),
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for values in rows:
            writer.writerow(dict(zip(fields, values)))


def test_nanshan_uses_the_same_canonical_interface(tmp_path: Path) -> None:
    raw = tmp_path / "nanshan.csv"
    _write_nanshan_sample(raw)
    output = tmp_path / "nanshan"
    DataGenerator().generate(
        DataConfig(
            source="nanshan",
            raw_path=str(raw),
            output_dir=str(output),
            candidate_count=2,
            num_bases=1,
            num_swaps=1,
            od_grid_km=0.2,
            site_grid_km=0.1,
        )
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source"] == "nanshan"
    assert metadata["date_basis"] == "dt_seq_id"
    assert metadata["demand_exposure_hours"] == 2
    assert metadata["total_demand_per_hour"] == pytest.approx(2.0)
    loaded = DataLoader().load_instance_dir(str(output))
    assert loaded.total_demand == pytest.approx(2.0)


def test_invalid_windows_and_solver_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="start and end must differ"):
        _window_hours(12, 12)
    with pytest.raises(ValueError, match="mip_gap"):
        SolverConfig(mip_gap=float("nan")).validate()
    with pytest.raises(ValueError, match="ranges"):
        ModelConfig(battery_range_km=-1).validate()
    for invalid_range in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ranges"):
            ModelConfig(battery_range_km=invalid_range).validate()
    for invalid_speed in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ranges"):
            ModelConfig(drone_speed_mps=invalid_speed).validate()
    with pytest.raises(ValueError, match="data seed"):
        DataConfig(source="tsl", raw_path="x", output_dir="y", seed=-1).validate()
    with pytest.raises(ValueError, match="network_start_seconds"):
        SolverConfig(network_start_seconds=-1).validate()
    with pytest.raises(ValueError, match="no_rel_heur_seconds"):
        SolverConfig(no_rel_heur_seconds=-1).validate()
    with pytest.raises(ValueError, match="root_cut_passes"):
        SolverConfig(root_cut_passes=-2).validate()
    with pytest.raises(ValueError, match="miqcp_method"):
        SolverConfig(miqcp_method=2).validate()
    with pytest.raises(ValueError, match="strictly positive"):
        ModelConfig(drone_cost_per_hour=0.0).validate()
    with pytest.raises(ValueError, match="feeder_rider_cost_factor"):
        ModelConfig(feeder_rider_cost_factor=0.0).validate()
    with pytest.raises(ValueError, match="nonnegative"):
        ModelConfig(locker_pickup_cost_per_order=-0.01).validate()


def test_cli_imports_defaults_and_solve_seed() -> None:
    parser = build_parser()
    prepare = parser.parse_args(["prepare", "--source", "tsl"])
    data_config = _data_config(prepare)
    assert Path(data_config.raw_path).is_absolute()
    assert Path(data_config.output_dir).is_absolute()
    assert data_config.candidate_count == 34
    assert "tsl_od_h34_" in Path(data_config.output_dir).name

    solve = parser.parse_args(
        ["solve", "--instance-dir", "/tmp/example", "--modes", "MX,LO", "--seed", "17"]
    )
    assert solve.modes == ("MX", "LO")
    assert solve.solver_seed == 17
    assert solve.miqcp_method == -1
    assert solve.root_cut_passes == 2
    default_model_config = _model_config(solve)
    assert default_model_config.drone_speed_mps == pytest.approx(20.0)
    assert default_model_config.drone_speed_kmph == pytest.approx(72.0)
    assert default_model_config.battery_range_km == pytest.approx(10.0)
    assert default_model_config.usable_battery_time_hours == pytest.approx(10.0 / 72.0)
    assert default_model_config.locker_service_rates == (30.0, 40.0, 50.0)
    assert default_model_config.locker_capacity_costs == (0.5, 1.0, 2.0)
    assert default_model_config.drone_cost_per_hour == pytest.approx(5.0)
    assert default_model_config.flight_cost_per_km == pytest.approx(0.20)
    assert default_model_config.feeder_rider_cost_factor == pytest.approx(0.30)
    assert default_model_config.display_origin_choices == 8
    assert default_model_config.general_corridor_choices == 10

    tuned = parser.parse_args(
        [
            "solve",
            "--instance-dir",
            "/tmp/example",
            "--locker-service-rates",
            "100,200",
            "--locker-capacity-costs",
            "2,4",
            "--base-cost-per-hour",
            "20",
            "--flight-cost-per-km",
            "0.4",
            "--rider-extra-cost-per-km",
            "1.6",
            "--feeder-rider-cost-factor",
            "0.6",
            "--drone-speed-mps",
            "18",
            "--battery-range-km",
            "12",
            "--max-virtual-wait-minutes",
            "3",
            "--display-origin-choices",
            "all",
            "--general-corridor-choices",
            "7",
            "--locker-pickup-cost",
            "0.47",
        ]
    )
    tuned_config = _model_config(tuned)
    tuned_config.validate()
    assert tuned_config.locker_service_rates == (100.0, 200.0)
    assert tuned_config.locker_capacity_costs == (2.0, 4.0)
    assert tuned_config.base_cost_per_hour == pytest.approx(20.0)
    assert tuned_config.flight_cost_per_km == pytest.approx(0.4)
    assert tuned_config.rider_incremental_cost_per_km == pytest.approx(1.6)
    assert tuned_config.feeder_rider_cost_factor == pytest.approx(0.6)
    assert tuned_config.drone_speed_mps == pytest.approx(18.0)
    assert tuned_config.battery_range_km == pytest.approx(12.0)
    assert tuned_config.max_virtual_wait_hours == pytest.approx(3.0 / 60.0)
    assert tuned_config.display_origin_choices is None
    assert tuned_config.general_corridor_choices == 7
    assert tuned_config.locker_pickup_cost_per_order == pytest.approx(0.47)

    disabled_start = parser.parse_args(
        [
            "solve",
            "--instance-dir",
            "/tmp/example",
            "--network-start-seconds",
            "0",
        ]
    )
    assert disabled_start.network_start_seconds == pytest.approx(0.0)

    bounded_root = parser.parse_args(
        [
            "solve",
            "--instance-dir",
            "/tmp/example",
            "--no-rel-heur-seconds",
            "17",
            "--root-cut-passes",
            "2",
        ]
    )
    assert bounded_root.no_rel_heur_seconds == pytest.approx(17.0)
    assert bounded_root.root_cut_passes == 2

    for retired_flag in (
        "--origin-locker-choices",
        "--home-corridor-choices",
        "--locker-service-cost",
        "--direct-range-km",
        "--swap-leg-km",
        "--rider-base-cost",
        "--rider-cost-per-km",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["solve", "--instance-dir", "/tmp/example", retired_flag, "2"]
            )


def test_run_config_uses_current_schema_source_and_implementation_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.model.gurobi_model as gurobi_model_module
    import src.utils.output_processor as output_processor_module

    instance_dir = _write_canonical_instance(tmp_path / "run_config_instance")
    results_dir = tmp_path / "run_config_results"

    class FakeModel:
        def __init__(self, _instance: object, mode: str):
            self.mode = mode
            params = type("FakeParams", (), {"LogFile": ""})()
            self.model = type(
                "FakeGurobiModel",
                (),
                {
                    "Status": GRB.OPTIMAL,
                    "Params": params,
                    "write": lambda _model, path: Path(path).write_text(
                        "# mock incumbent\ngamma 0\n",
                        encoding="utf-8",
                    ),
                },
            )()

        def prepare_algorithm(self, _solver: SolverConfig) -> "FakeModel":
            return self

        def solve(self, _solver: SolverConfig, **_kwargs: object) -> bool:
            return True

        def export_incumbent_start(self) -> dict[str, float]:
            return {"gamma": 0.0}

    class FakeProcessor:
        def __init__(
            self,
            wrapper: FakeModel,
            _instance: object,
            *,
            run_signature_sha256: str,
        ):
            self.wrapper = wrapper
            self.run_signature_sha256 = run_signature_sha256

        def print_summary(self) -> None:
            return None

        def write_all(self, _output_dir: str) -> dict[str, object]:
            return {
                "schema_version": RESULT_SCHEMA_VERSION,
                "formulation_id": FORMULATION_ID,
                "run_signature_sha256": self.run_signature_sha256,
                "mode": self.wrapper.mode,
            }

    monkeypatch.setattr(gurobi_model_module, "MLNDModel", FakeModel)
    monkeypatch.setattr(output_processor_module, "ResultProcessor", FakeProcessor)
    monkeypatch.setattr(cli_main, "_write_mode_comparison", lambda *_args: None)

    args = build_parser().parse_args(
        [
            "solve",
            "--instance-dir",
            str(instance_dir),
            "--results-dir",
            str(results_dir),
            "--modes",
            "HD,MX,LO",
            "--display-origin-choices",
            "all",
            "--general-corridor-choices",
            "all",
            "--locker-pickup-cost",
            "0.47",
            "--quiet",
        ]
    )
    summaries = cli_main._solve(args, str(instance_dir))
    payload = json.loads((results_dir / "run_config.json").read_text())

    assert [summary["mode"] for summary in summaries] == ["HD", "MX", "LO"]
    assert payload["modes"] == ["HD", "MX", "LO"]
    assert payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert payload["formulation_id"] == FORMULATION_ID
    assert len(payload["run_signature_sha256"]) == 64
    assert {summary["run_signature_sha256"] for summary in summaries} == {
        payload["run_signature_sha256"]
    }
    assert payload["formulation_source"]["path"] == str(FORMULATION_SOURCE.resolve())
    assert (
        payload["formulation_source"]["sha256"]
        == hashlib.sha256(FORMULATION_SOURCE.read_bytes()).hexdigest()
    )
    assert len(payload["implementation"]["sha256"]) == 64
    assert "src/model/gurobi_model.py" in payload["implementation"]["files"]
    assert "src/data_processor/loader.py" in payload["implementation"]["files"]
    assert payload["screening_is_exact"] == {"MX": True, "HD": True, "LO": True}
    assert payload["model"]["display_origin_choices"] is None
    assert payload["model"]["general_corridor_choices"] is None
    assert payload["model"]["choice"]["max_walk_km"] == pytest.approx(1.5)
    assert payload["model"]["locker_pickup_cost_per_order"] == pytest.approx(0.47)
    assert payload["model_derived"]["drone_speed_kmph"] == pytest.approx(72.0)
    assert payload["solver"]["miqcp_method"] == -1
    assert payload["git"]["head"]
    assert payload["execution_environment"]["python_version"]
    completion = json.loads((results_dir / "run_complete.json").read_text())
    assert completion["completed"] is True
    assert completion["modes"] == ["HD", "MX", "LO"]
    assert completion["artifact_count"] >= 4
    assert payload["model_derived"]["usable_battery_time_minutes"] == pytest.approx(
        10.0 / 72.0 * 60.0
    )
    assert set(payload["instance"]["canonical_file_sha256"]) == {
        "metadata.json",
        "sites.csv",
        "od_nodes.csv",
        "od_demand.csv",
        "periods.csv",
    }
